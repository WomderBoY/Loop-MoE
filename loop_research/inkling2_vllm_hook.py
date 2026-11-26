"""Production vLLM Blackwell SM120 Triton Hook for LoopSched-MoE on Inkling-2-Mini.

Hooks `vllm.models.inkling2.nvidia.moe.Inkling2MoE.forward` in-place (zero weight conversion,
100% compatible with vLLM CUDA Graphs and native SM120 NVFP4/MXFP8 `bmm_sm120` kernels!) to support:
  1. Per-visit `(layer_id, pass_idx)` active-expert count `k_v in {0, 1, ..., 8}` over all 48 visits
     (6 prelude + 12*3 looped core + 6 coda).
  2. Exact generalized Top-8 -> k_v routing weight shrinkage:
       - `renorm`: renormalize kept top-k_v probabilities to sum to `route_scale` (standard baseline)
       - `noren`:  preserve original Top-8 denominator S_8 (`gamma = 1.0`, Orthogonal Shrinkage)
       - `opt`:    apply per-visit optimal shrinkage `gamma_v*` on top of Top-8 denominator S_8
  3. Zero-approximation per-visit geometry diagnostics (`cos(s_k, r_k)`, `gamma_k*`,
     `MSE_renorm(k)/MSE_noren(k)` for k=1..7) computed directly from exact NVFP4/MXFP8 `down` rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class Inkling2LoopSchedController:
    """Mutable runtime controller attached to a live vLLM Inkling2LoopModel."""

    num_visits: int = 48
    k_max: int = 8
    # Per-visit active expert budget k_v in [0..8]
    visit_k: list[int] = field(default_factory=lambda: [8] * 48)
    # Per-visit shrinkage mode: "renorm", "noren", or "opt"
    shrink_mode: str = "renorm"
    # Per-visit gamma multiplier (used when shrink_mode == "opt"; 1.0 for "noren")
    visit_gamma: list[float] = field(default_factory=lambda: [1.0] * 48)
    # Optional geometry diagnostics accumulator (enabled only during Stage 1 geometry pass)
    diag_enabled: bool = False
    diag_stats: dict[int, dict[int, dict[str, float]]] = field(default_factory=dict)

    def reset_baseline(self) -> None:
        self.visit_k = [self.k_max] * self.num_visits
        self.shrink_mode = "renorm"
        self.visit_gamma = [1.0] * self.num_visits
        self.diag_enabled = False

    def set_policy(
        self,
        visit_k: list[int],
        shrink_mode: str = "noren",
        visit_gamma: list[float] | None = None,
    ) -> None:
        assert len(visit_k) == self.num_visits
        assert shrink_mode in ("renorm", "noren", "opt")
        self.visit_k = [int(k) for k in visit_k]
        self.shrink_mode = shrink_mode
        if visit_gamma is not None:
            assert len(visit_gamma) == self.num_visits
            self.visit_gamma = [float(g) for g in visit_gamma]
        else:
            self.visit_gamma = [1.0] * self.num_visits

    def start_diagnostics(self) -> None:
        self.reset_baseline()
        self.diag_stats = {
            v: {
                k: {
                    "count": 0.0,
                    "dot_sr": 0.0,
                    "s_sq": 0.0,
                    "r_sq": 0.0,
                    "err_ren": 0.0,
                    "cos_sum": 0.0,
                    "alpha_sum": 0.0,
                }
                for k in range(1, self.k_max)
            }
            for v in range(self.num_visits)
        }
        self.diag_enabled = True

    def stop_diagnostics(self) -> dict[int, dict[int, dict[str, float]]]:
        self.diag_enabled = False
        summary: dict[int, dict[int, dict[str, float]]] = {}
        for v, by_k in self.diag_stats.items():
            summary[v] = {}
            for k, acc in by_k.items():
                n = max(acc["count"], 1.0)
                s_sq = max(acc["s_sq"], 1e-12)
                r_sq = max(acc["r_sq"], 1e-12)
                gamma_opt = 1.0 + acc["dot_sr"] / s_sq
                err_opt = max(r_sq - (acc["dot_sr"] ** 2) / s_sq, 1e-12)
                summary[v][k] = {
                    "count": int(acc["count"]),
                    "cos_kept_dropped": float(acc["cos_sum"] / n),
                    "gamma_opt": float(gamma_opt),
                    "mean_alpha_renorm": float(acc["alpha_sum"] / n),
                    "mse_ratio_ren_vs_noren": float(acc["err_ren"] / r_sq),
                    "mse_ratio_noren_vs_opt": float(r_sq / err_opt),
                }
        return summary

    def accumulate_visit_diag(
        self,
        vid: int,
        weights_desc: torch.Tensor,
        weighted_outs_desc: torch.Tensor,
    ) -> None:
        """Accumulate exact shrinkage stats for visit `vid`.

        Args:
            weights_desc: [T, K] positive routing weights sorted descending by selection key
            weighted_outs_desc: [T, K, H] exact weighted expert outputs (w_j * e_j) in same order
        """
        w = weights_desc.float()
        u = weighted_outs_desc.float()
        T = int(w.shape[0])
        if T == 0:
            return
        w_tot = w.sum(dim=1, keepdim=True)  # [T, 1]
        u_tot = u.sum(dim=1)                # [T, H]
        # Cumulative prefix sums over slots 0..K-1
        w_cum = torch.cumsum(w, dim=1)      # [T, K]
        u_cum = torch.cumsum(u, dim=1)      # [T, K, H]

        by_k = self.diag_stats[vid]
        for k in range(1, self.k_max):
            s_k = u_cum[:, k - 1, :]                            # [T, H]
            r_k = u_tot - s_k                                   # [T, H]
            alpha = w_tot / w_cum[:, k - 1 : k].clamp_min(1e-12)  # [T, 1]

            dot_sr = (s_k * r_k).sum(dim=-1)
            s_sq = (s_k * s_k).sum(dim=-1)
            r_sq = (r_k * r_k).sum(dim=-1)
            cos_sr = dot_sr / (s_sq.clamp_min(1e-12).sqrt() * r_sq.clamp_min(1e-12).sqrt())
            err_ren = ((alpha - 1.0) * s_k - r_k).pow(2).sum(dim=-1)

            acc = by_k[k]
            acc["count"] += float(T)
            acc["dot_sr"] += float(dot_sr.sum().item())
            acc["s_sq"] += float(s_sq.sum().item())
            acc["r_sq"] += float(r_sq.sum().item())
            acc["err_ren"] += float(err_ren.sum().item())
            acc["cos_sum"] += float(cos_sr.sum().item())
            acc["alpha_sum"] += float(alpha.sum().item())


# ---------------------------------------------------------------------------
# Lazy-Compiled General Top-K (k in 1..8) + Exact NoRen/Opt Triton Routing
# ---------------------------------------------------------------------------

_TRITON_KERNELS_BUILT = False
_ROUTE_GENERAL_FN = None


def _build_general_triton_route():
    """Build general Top-8 -> k_v (k_v in 1..8, including non-pow2 3/5/6/7) Triton routing kernels.

    Reuses vLLM's own Inkling2 activation quantization (`_quantize_x`, `_store_x`, `_row_scales`,
    `_place_experts`, `k_group_partials`, `operand`, `ExpertPlan`) so downstream SM120 NVFP4/MXFP8
    `bmm_sm120`, `swiglu_quantize`, and `combine` run unmodified with ONLY `k_v` active experts
    and full CUDA-Graph compatibility!
    """
    global _TRITON_KERNELS_BUILT, _ROUTE_GENERAL_FN
    if _TRITON_KERNELS_BUILT:
        return _ROUTE_GENERAL_FN

    from vllm.triton_utils import tl, tldevice, triton
    from vllm.models.inkling2.nvidia.ops.linear import k_group_partials
    from vllm.models.inkling2.nvidia.ops.moe import (
        LAUNCH_DEPENDENTS_MAX_TOKENS,
        _TOKENS_DONE,
        _group_sum,
        _load_bias,
        _local_experts,
        _place_experts,
        _quantize_x,
        _row_scales,
        _selection_keys,
        _store_x,
    )
    from vllm.models.inkling2.nvidia.ops.trtllm_bmm import ExpertPlan, operand

    @triton.jit
    def _select_top8_keep_k_kernel(
        Partials,
        Bias,
        ExpertBounds,
        Padding,
        Counts,
        Counters,
        Ids,
        Ranks,
        Weights,
        Starts,
        Tiles,
        T,
        route_scale,
        gamma_scale,
        E: tl.constexpr,
        E_POW2: tl.constexpr,
        GROUPS: tl.constexpr,
        K_BASE: tl.constexpr,
        TOPK: tl.constexpr,
        TOPK_POW2: tl.constexpr,
        SHRINK_NOREN: tl.constexpr,
        NUM_LOCAL: tl.constexpr,
        LOCAL_POW2: tl.constexpr,
        TILE: tl.constexpr,
        MAX_CTAS: tl.constexpr,
    ):
        """Select exact Top-8 by (sigmoid(logit)+bias), compute S_8 and S_k, keep only Top-k_v."""
        bias = _load_bias(Bias, E, E_POW2)
        lo = tl.load(ExpertBounds)
        hi = tl.load(ExpertBounds + 1)
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

        token = tl.full((1,), tl.program_id(0), tl.int64)
        rows = token[:, None]
        offs = tl.arange(0, E_POW2)[None, :]
        padding = None
        if Padding is not None:
            padding = tl.load(Padding + token)

        logits = _group_sum(Partials, rows, offs, offs < E, T, E, GROUPS)
        keys = _selection_keys(logits, bias, offs, E)

        # Extract exact Top-K_BASE (=8) in strictly descending selection-key order
        # (only 8 vector maxes over 128 elements -> fast and works for any TOPK in 1..8)
        slots_base = tl.arange(0, K_BASE)[None, :]
        ids_base = tl.zeros((1, K_BASE), tl.int32)
        for k_idx in tl.static_range(K_BASE):
            best = tl.max(keys, 1)[:, None]
            ids_base = tl.where(slots_base == k_idx, E - (best & 0xFFFF).to(tl.int32), ids_base)
            keys = tl.where(keys == best, 0, keys)

        selected_logits = _group_sum(Partials, rows, ids_base, ids_base >= 0, T, E, GROUPS)
        probs_base = tl.div_rn(1.0, 1.0 + tldevice.exp(-selected_logits))  # [1, K_BASE]
        sum_base = tl.sum(probs_base, 1)                                   # [1]
        sum_kept = tl.sum(tl.where(slots_base < TOPK, probs_base, 0.0), 1) # [1]

        if SHRINK_NOREN:
            denom = sum_base[:, None] + 1e-20
            weights_base = (probs_base / denom) * (route_scale * gamma_scale)
        else:
            denom = sum_kept[:, None] + 1e-20
            weights_base = (probs_base / denom) * route_scale

        # Slice to [1, TOPK_POW2] and mask slots >= TOPK to local=-1
        # Since K_BASE == 8 >= TOPK_POW2, extract slots [0..TOPK_POW2-1] cleanly
        slots_out = tl.arange(0, TOPK_POW2)[None, :]
        slot_valid = slots_out < TOPK

        # Map from [1, K_BASE] to [1, TOPK_POW2] via broadcast compare (both <= 8!)
        match = slots_base[:, None, :] == slots_out[:, :, None]  # [1, TOPK_POW2, K_BASE]
        ids_out = tl.sum(tl.where(match, ids_base[:, None, :], 0), 2)
        w_out = tl.sum(tl.where(match, weights_base[:, None, :], 0.0), 2)

        local = _local_experts(ids_out, lo, hi, padding)
        local = tl.where(slot_valid, local, -1)

        rank = tl.atomic_add(Counts + local, 1, mask=local >= 0, sem="relaxed")
        out_offsets = token[:, None] * TOPK + slots_out
        tl.store(Ranks + out_offsets, rank, mask=slot_valid)
        tl.store(Ids + out_offsets, local, mask=slot_valid)
        tl.store(Weights + out_offsets, w_out, mask=slot_valid)

        tl.debug_barrier()
        if tl.atomic_add(Counters + _TOKENS_DONE, 1, sem="acq_rel") == T - 1:
            _place_experts(Counts, Starts, Tiles, NUM_LOCAL, LOCAL_POW2, TILE, MAX_CTAS)
            tl.store(Counters + _TOKENS_DONE, 0)

    @triton.jit
    def _scatter_general_kernel(
        X,
        Starts,
        Ids,
        Ranks,
        Weights,
        Src2Dst,
        Gammas,
        Rows,
        QValues,
        QScales,
        QRowScales,
        WAmax,
        WDenominator,
        H: tl.constexpr,
        TOPK: tl.constexpr,
        TOPK_POW2: tl.constexpr,
        LOCAL_POW2: tl.constexpr,
        TILE: tl.constexpr,
        NVFP4: tl.constexpr,
        HEAD: tl.constexpr,
        TAIL: tl.constexpr,
        BROADCAST: tl.constexpr,
        LAUNCH_DEPENDENTS: tl.constexpr,
    ):
        """Scatter and quantize `X[token]` into `TOPK` layout rows (supports non-pow2 TOPK via TOPK_POW2 mask)."""
        token = tl.program_id(0).to(tl.int64)
        quantized = _quantize_x(X, token, H, NVFP4, HEAD, TAIL)
        tl.extra.cuda.gdc_wait()
        if LAUNCH_DEPENDENTS:
            tl.extra.cuda.gdc_launch_dependents()

        s = tl.arange(0, TOPK_POW2)
        valid_s = s < TOPK
        slots = token * TOPK + s
        local = tl.load(Ids + slots, mask=valid_s, other=-1)
        rank = tl.load(Ranks + slots, mask=valid_s, other=0)
        w = tl.load(Weights + slots, mask=valid_s, other=0.0)
        routed = valid_s & (local >= 0)
        dst = tl.where(routed, tl.load(Starts + local, routed, other=0) + rank, -1)

        tl.store(Src2Dst + slots, dst, mask=valid_s)
        tl.store(Gammas + dst, w, mask=routed)
        tl.store(Rows + tl.load(Starts + LOCAL_POW2 + local, routed, other=0) + rank, dst, mask=routed)

        _store_x(
            QValues,
            QScales,
            QRowScales,
            quantized,
            dst,
            _row_scales(quantized[4], WAmax, WDenominator, local, NVFP4),
            H,
            NVFP4,
            HEAD,
            TAIL,
            TILE,
            BROADCAST,
        )

    def route_general(
        x: torch.Tensor,
        gate_weight: torch.Tensor,
        bias: torch.Tensor,
        counters: torch.Tensor,
        counts: torch.Tensor,
        k_base: int,
        topk: int,
        route_scale: float,
        gamma_scale: float,
        shrink_noren: bool,
        expert_bounds: torch.Tensor,
        plan: ExpertPlan,
        weight_amax: torch.Tensor | None,
        weight_denominator: torch.Tensor | None,
        is_padding: torch.Tensor | None = None,
    ):
        """General Top-8 -> topk (1..8) Triton router with exact NoRen/Opt shrinkage & CUDA-Graph safety."""
        tokens, hidden = x.shape
        device = x.device
        experts = gate_weight.shape[0]
        num_local = counts.shape[0]
        nvfp4 = weight_amax is not None
        pack, group = (2, 16) if nvfp4 else (1, 32)
        q_dtype = torch.uint8 if nvfp4 else torch.float8_e4m3fn
        scale_dtype = torch.float8_e4m3fn if nvfp4 else torch.uint8
        head = 1 << (hidden.bit_length() - 1)
        tail = hidden - head

        src2dst = torch.empty(tokens * topk, dtype=torch.int32, device=device)
        gammas = operand(plan.rows, dtype=torch.float32, device=device)
        routed = torch.empty(tokens * topk, dtype=torch.int32, device=device)
        tiles = torch.empty(2 * plan.max_ctas + 3, dtype=torch.int32, device=device)
        q_values = operand(plan.rows, hidden // pack, dtype=q_dtype, device=device)
        q_scales = operand(plan.rows, hidden // group, dtype=scale_dtype, device=device)
        q_row_scales = operand(plan.rows, dtype=torch.float32, device=device) if nvfp4 else None

        outputs = (src2dst, gammas, routed, tiles)
        quantized = (q_values, q_scales, q_row_scales)

        partials = k_group_partials(x, gate_weight)
        local_pow2 = triton.next_power_of_2(num_local)
        topk_pow2 = triton.next_power_of_2(topk)
        ids = torch.empty(tokens * topk, dtype=torch.int32, device=device)
        ranks = torch.empty(tokens * topk, dtype=torch.int32, device=device)
        weights = torch.empty(tokens * topk, dtype=torch.float32, device=device)
        starts = torch.empty(3 * local_pow2, dtype=torch.int32, device=device)

        _select_top8_keep_k_kernel[(tokens,)](
            partials,
            bias,
            expert_bounds,
            is_padding,
            counts,
            counters,
            ids,
            ranks,
            weights,
            starts,
            tiles,
            tokens,
            route_scale,
            gamma_scale,
            E=experts,
            E_POW2=triton.next_power_of_2(experts),
            GROUPS=partials.shape[0],
            K_BASE=k_base,
            TOPK=topk,
            TOPK_POW2=topk_pow2,
            SHRINK_NOREN=shrink_noren,
            NUM_LOCAL=num_local,
            LOCAL_POW2=local_pow2,
            TILE=plan.tile,
            MAX_CTAS=plan.max_ctas,
            num_warps=4 if tokens <= 128 else 1,
            launch_pdl=True,
        )
        _scatter_general_kernel[(tokens,)](
            x,
            starts,
            ids,
            ranks,
            weights,
            src2dst,
            gammas,
            routed,
            *quantized,
            weight_amax,
            weight_denominator,
            H=hidden,
            TOPK=topk,
            TOPK_POW2=topk_pow2,
            LOCAL_POW2=local_pow2,
            TILE=plan.tile,
            NVFP4=nvfp4,
            HEAD=head,
            TAIL=tail,
            BROADCAST=tokens < (4096 if nvfp4 else 512),
            LAUNCH_DEPENDENTS=tokens <= LAUNCH_DEPENDENTS_MAX_TOKENS,
            num_warps=2,
            enable_fp_fusion=False,
            launch_pdl=True,
        )
        return outputs, quantized

    _TRITON_KERNELS_BUILT = True
    _ROUTE_GENERAL_FN = route_general
    return _ROUTE_GENERAL_FN


def _compute_desc_top8_weights_pytorch(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor,
    k_base: int,
    route_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference FP32 Top-8 selection matching Inkling2 exactly (used ONLY for Stage-1 geometry diag)."""
    logits = torch.nn.functional.linear(x.float(), gate_weight.float())
    scores = torch.sigmoid(logits)
    sel = scores + gate_bias.float().unsqueeze(0)
    _, ids_desc = torch.topk(sel, k=k_base, dim=-1, largest=True, sorted=True)
    p_desc = scores.gather(1, ids_desc)
    w_desc = (p_desc / (p_desc.sum(dim=-1, keepdim=True) + 1e-20)) * float(route_scale)
    return ids_desc, w_desc


def install_inkling2_loopsched_hook(
    vllm_model: Any,
    controller: Inkling2LoopSchedController,
) -> int:
    """Attach LoopSched-MoE hook to every `Inkling2MoE` module inside a loaded vLLM Inkling2LoopModel.

    Returns the number of hooked physical MoE layers (24 for Inkling-2-Mini).
    """
    from vllm.models.inkling2.nvidia.moe import Inkling2MoE
    from vllm.models.inkling2.nvidia.ops.moe import combine
    from vllm.models.inkling2.nvidia.ops.trtllm_bmm import expert_plan
    import vllm.envs as envs
    from vllm.forward_context import get_forward_context, is_forward_context_available

    route_general = _build_general_triton_route()
    hooked = 0

    for mod in vllm_model.modules():
        if not isinstance(mod, Inkling2MoE):
            continue
        if hasattr(mod, "_loopsched_orig_forward"):
            mod._loopsched_ctrl = controller
            hooked += 1
            continue

        orig_forward = mod.forward
        mod._loopsched_orig_forward = orig_forward
        mod._loopsched_ctrl = controller

        def _make_patched(moe_mod: Inkling2MoE, orig_fn):
            def patched_forward(
                x: torch.Tensor,
                reduce: bool = True,
                push=None,
                lora=None,
                pass_idx: int = 0,
            ):
                ctrl: Inkling2LoopSchedController = moe_mod._loopsched_ctrl
                vid = int(moe_mod.routing_captures[pass_idx].layer_id)
                k_v = int(ctrl.visit_k[vid])
                k_base = int(moe_mod.gate.topk)

                # Fast exact diagnostic capture path (Stage 1 only, baseline k=8)
                if ctrl.diag_enabled and reduce and push is None and lora is None:
                    x_c = x.contiguous()
                    T = int(x_c.shape[0])
                    gate, experts = moe_mod.gate, moe_mod.experts
                    plan = expert_plan(T, k_base, experts.num_experts, experts.nvfp4, experts.shapes)
                    routing, quantized = route_general(
                        x_c,
                        gate.weight,
                        gate.bias,
                        gate.counters,
                        experts.counts,
                        k_base=k_base,
                        topk=k_base,
                        route_scale=float(gate.route_scale),
                        gamma_scale=1.0,
                        shrink_noren=False,
                        expert_bounds=experts.expert_bounds,
                        plan=plan,
                        weight_amax=experts.scales("w13")[0],
                        weight_denominator=experts.scales("w13")[1],
                        is_padding=None,
                    )
                    down, src2dst = experts(x_c, routing, quantized, plan, reduce=False, push=None)
                    # Note: route_general preserves strictly descending selection-key order in slots 0..7!
                    _, w_desc = _compute_desc_top8_weights_pytorch(
                        x_c, gate.weight, gate.bias, k_base, float(gate.route_scale)
                    )
                    idx_2d = src2dst.view(T, k_base).long()
                    valid_mask = idx_2d >= 0
                    safe_idx = idx_2d.clamp_min(0)
                    u_desc = down.index_select(0, safe_idx.view(-1)).view(T, k_base, -1)
                    u_desc = u_desc * valid_mask.unsqueeze(-1).to(u_desc.dtype)
                    ctrl.accumulate_visit_diag(vid, w_desc, u_desc)
                    return combine(down, src2dst, k_base, x_c.dtype)

                # Native zero-overhead passthrough when baseline k=8 + renorm
                if k_v == k_base and ctrl.shrink_mode == "renorm":
                    return orig_fn(x, reduce=reduce, push=push, lora=lora, pass_idx=pass_idx)

                # Complete visit skip (k_v == 0): zero router + zero GEMM!
                if k_v == 0:
                    assert reduce and push is None and lora is None, "k_v=0 requires TP=1 reduce=True"
                    return torch.zeros_like(x)

                # General Top-8 -> k_v (1..8) with exact NoRen / Opt / Renorm inside native SM120 NVFP4/MXFP8
                assert lora is None
                x_c = x.contiguous()
                gate, experts = moe_mod.gate, moe_mod.experts
                is_padding = None
                if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                    is_padding = get_forward_context().is_padding

                plan = expert_plan(
                    int(x_c.shape[0]),
                    k_v,
                    experts.num_experts,
                    experts.nvfp4,
                    experts.shapes,
                )
                shrink_noren = ctrl.shrink_mode in ("noren", "opt")
                gamma_scale = float(ctrl.visit_gamma[vid]) if ctrl.shrink_mode == "opt" else 1.0

                routing, quantized = route_general(
                    x_c,
                    gate.weight,
                    gate.bias,
                    gate.counters,
                    experts.counts,
                    k_base=k_base,
                    topk=k_v,
                    route_scale=float(gate.route_scale),
                    gamma_scale=gamma_scale,
                    shrink_noren=shrink_noren,
                    expert_bounds=experts.expert_bounds,
                    plan=plan,
                    weight_amax=experts.scales("w13")[0],
                    weight_denominator=experts.scales("w13")[1],
                    is_padding=is_padding,
                )
                # Temporarily override experts.topk for combine(..., k_v, ...) (safe under CUDA graph capture!)
                saved_topk = experts.topk
                experts.topk = k_v
                try:
                    return experts(x_c, routing, quantized, plan, reduce=reduce, push=push)
                finally:
                    experts.topk = saved_topk

            return patched_forward

        mod.forward = _make_patched(mod, orig_forward)
        hooked += 1

    return hooked


class Inkling2VLLMClassPatch:
    """Context manager patching `Inkling2MoE.forward` at the class level BEFORE `vllm.LLM(...)` init.

    Because the patch is active during model construction and CUDA-Graph capture (with
    `VLLM_ENABLE_V1_MULTIPROCESSING=0`), every `Inkling2MoE` layer automatically bakes the
    active `controller.visit_k` schedule directly into production vLLM CUDA Graphs with zero
    internal engine poking.
    """

    def __init__(self, controller: Inkling2LoopSchedController) -> None:
        self.controller = controller
        self._orig_forward = None
        self._moe_cls = None

    def __enter__(self) -> "Inkling2VLLMClassPatch":
        from vllm.models.inkling2.nvidia.moe import Inkling2MoE
        from vllm.models.inkling2.nvidia.ops.moe import combine
        from vllm.models.inkling2.nvidia.ops.trtllm_bmm import expert_plan
        import vllm.envs as envs
        from vllm.forward_context import get_forward_context, is_forward_context_available

        self._moe_cls = Inkling2MoE
        self._orig_forward = Inkling2MoE.forward
        orig_forward = self._orig_forward
        ctrl = self.controller
        route_general = _build_general_triton_route()

        def patched_forward(
            moe_self: Inkling2MoE,
            x: torch.Tensor,
            reduce: bool = True,
            push=None,
            lora=None,
            pass_idx: int = 0,
        ):
            vid = int(moe_self.routing_captures[pass_idx].layer_id)
            k_v = int(ctrl.visit_k[vid])
            k_base = int(moe_self.gate.topk)

            # Stage-1 exact geometry diagnostics on baseline k=8
            if ctrl.diag_enabled and reduce and push is None and lora is None:
                x_c = x.contiguous()
                T = int(x_c.shape[0])
                gate, experts = moe_self.gate, moe_self.experts
                plan = expert_plan(T, k_base, experts.num_experts, experts.nvfp4, experts.shapes)
                routing, quantized = route_general(
                    x_c,
                    gate.weight,
                    gate.bias,
                    gate.counters,
                    experts.counts,
                    k_base=k_base,
                    topk=k_base,
                    route_scale=float(gate.route_scale),
                    gamma_scale=1.0,
                    shrink_noren=False,
                    expert_bounds=experts.expert_bounds,
                    plan=plan,
                    weight_amax=experts.scales("w13")[0],
                    weight_denominator=experts.scales("w13")[1],
                    is_padding=None,
                )
                down, src2dst = experts(x_c, routing, quantized, plan, reduce=False, push=None)
                _, w_desc = _compute_desc_top8_weights_pytorch(
                    x_c, gate.weight, gate.bias, k_base, float(gate.route_scale)
                )
                idx_2d = src2dst.view(T, k_base).long()
                valid_mask = idx_2d >= 0
                safe_idx = idx_2d.clamp_min(0)
                u_desc = down.index_select(0, safe_idx.view(-1)).view(T, k_base, -1)
                u_desc = u_desc * valid_mask.unsqueeze(-1).to(u_desc.dtype)
                ctrl.accumulate_visit_diag(vid, w_desc, u_desc)
                return combine(down, src2dst, k_base, x_c.dtype)

            # Native zero-overhead passthrough when baseline k=8 + renorm
            if k_v == k_base and ctrl.shrink_mode == "renorm":
                return orig_forward(moe_self, x, reduce=reduce, push=push, lora=lora, pass_idx=pass_idx)

            # Complete visit skip (k_v == 0) or empty batch
            if k_v == 0 or int(x.shape[0]) == 0:
                assert reduce and push is None and lora is None, "k_v=0 requires TP=1 reduce=True"
                return torch.zeros_like(x)

            # General Top-8 -> k_v (1..8) with exact NoRen / Opt / Renorm inside native SM120 NVFP4/MXFP8
            assert lora is None
            x_c = x.contiguous()
            gate, experts = moe_self.gate, moe_self.experts
            is_padding = None
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                is_padding = get_forward_context().is_padding

            from vllm.models.inkling2.nvidia.ops.trtllm_bmm import ExpertPlan

            plan = expert_plan(
                int(x_c.shape[0]),
                k_v,
                experts.num_experts,
                experts.nvfp4,
                experts.shapes,
            )
            if plan.max_ctas < 2:
                # Guarantee Triton associative_scan operates on length >= 2 even for T=1, k_v=1
                plan = ExpertPlan(plan.tile, 2, plan.kernels)
            shrink_noren = ctrl.shrink_mode in ("noren", "opt")
            gamma_scale = float(ctrl.visit_gamma[vid]) if ctrl.shrink_mode == "opt" else 1.0

            routing, quantized = route_general(
                x_c,
                gate.weight,
                gate.bias,
                gate.counters,
                experts.counts,
                k_base=k_base,
                topk=k_v,
                route_scale=float(gate.route_scale),
                gamma_scale=gamma_scale,
                shrink_noren=shrink_noren,
                expert_bounds=experts.expert_bounds,
                plan=plan,
                weight_amax=experts.scales("w13")[0],
                weight_denominator=experts.scales("w13")[1],
                is_padding=is_padding,
            )
            saved_topk = experts.topk
            experts.topk = k_v
            try:
                return experts(x_c, routing, quantized, plan, reduce=reduce, push=push)
            finally:
                experts.topk = saved_topk

        Inkling2MoE.forward = patched_forward
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._moe_cls is not None and self._orig_forward is not None:
            self._moe_cls.forward = self._orig_forward


def extract_vllm_model_from_llm(llm: Any) -> Any:
    """Locate the live nn.Module inside an in-process vLLM V1/V0 LLM instance."""
    # vLLM V1 (VLLM_ENABLE_V1_MULTIPROCESSING=0)
    candidates = [
        ("llm_engine", "engine_core", "engine_core", "model_executor", "driver_worker", "worker", "model_runner", "model"),
        ("llm_engine", "engine_core", "model_executor", "driver_worker", "worker", "model_runner", "model"),
        ("llm_engine", "model_executor", "driver_worker", "model_runner", "model"),
        ("llm_engine", "model_executor", "driver_worker", "worker", "model_runner", "model"),
    ]
    for path in candidates:
        obj = llm
        ok = True
        for attr in path:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok and obj is not None:
            return obj
    raise RuntimeError(
        "Could not locate live model inside vLLM LLM object. "
        "Ensure VLLM_ENABLE_V1_MULTIPROCESSING=0 is set before importing vllm."
    )
