"""Direction 2 Feasibility Pilot: Velocity-Modulated Shared-Subspace Looped MoE (V-SubMoE).

Builds directly on 5bce20b9 (where shared_plus_scalar beat per-loop unbinding by ~2x on
IFM Ouro-MoE and Huginn-MoE) to test the ICML 2027 Direction 2 hypotheses:
  0. Exactness verification: pure-PyTorch differentiable router vs native TopKRouting,
     per-visit W2 gradient decomposition, and arbitrary-depth execution schedule.
  1. Subspace Concentration & Routing Collapse Diagnostics:
     - Prove why unbinding loses to shared subspace: held-out Split-B gradient energy
       captured by a shared rank-(R*r0) basis (with/without loop-diagonal scaling) vs
       R separate rank-r0 per-loop bases fit on Split A.
     - Measure baseline cross-loop expert load cosine (LOOM Fig. 4) and top-k flip rate.
  2. Strictly Parameter-Matched Adaptation across 9 Conditions (Top-20% experts):
     - base_frozen
     - per_loop_lora (rank r0 per loop x R loops; clamps index on R_test > R_train)
     - pure_shared (shared rank R*r0, NO modulation)
     - loop_dropout (Loop Dropout [2609.34218] p=0.5 + inverse-survival 1/(1-p) on shared rank R*r0)
     - shared_plus_scalar (shared rank R*r0 + per-loop index scalar; clamps on R_test > R_train)
     - shared_plus_diag_idx (shared rank R*r0 + per-loop index diagonal core B diag(1+c_r) A)
     - velocity_submoe (INDEX-FREE shared subspace + per-token velocity-conditioned diagonal core)
     - velocity_submoe_ld (velocity_submoe + Loop Dropout p=0.35 regularization)
     - velocity_submoe_full (velocity_submoe_ld + index-free velocity-gated hysteresis router)
  3. Multi-Axis Evaluation:
     - In-depth held-out NLL (overall, text, code) with paired bootstrap 95% CIs.
     - Zero-shot depth elasticity: half-depth (0.5x), native (1x), extrapolated (1.5x, 2x R_train).
     - Single-loop activation diagnostic (Loop Dropout Fig. 1b).
     - Real downstream ARC-Challenge MC accuracy at R_train and 1.5x R_train.
     - Cross-loop load cosine & late-loop routing flip rate.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from loop_research import orx_run as orx
from loop_research.data_assets import build_disjoint_text_code_splits, load_arc_challenge_samples
from loop_research.gpu_utils import acquire_idle_gpu
from loop_research.ifm_runtime import bootstrap_xllm, load_ifm_model_for_autograd


# ---------------------------------------------------------------------------
# Arbitrary-Depth Execution Context for IFM Ouro-MoE & Huginn-MoE
# ---------------------------------------------------------------------------

@contextmanager
def set_ifm_loop_depth(model, n_loops: int):
    """Temporarily set the recurrent depth R for both LoopedTransformer (Ouro) and Huginn."""
    orig_loops = int(model.loop_times)
    orig_sched = getattr(model, "execution_schedule", None)
    try:
        model.loop_times = int(n_loops)
        if orig_sched is not None and getattr(model, "huginn_hierarchical_state", "none") == "none":
            from xllm.models.looped import build_loop_execution_schedule

            model.execution_schedule = build_loop_execution_schedule(
                num_layers=model.num_layers,
                loop_times=int(n_loops),
                loop_start_layer=model.loop_start_layer,
                loop_end_layers=model.loop_end_layers,
            )
        yield
    finally:
        model.loop_times = orig_loops
        if orig_sched is not None:
            model.execution_schedule = orig_sched


# ---------------------------------------------------------------------------
# Pure-PyTorch Index-Free Velocity Features & Differentiable Router (CPU Testable!)
# ---------------------------------------------------------------------------

VELOCITY_FEAT_DIM = 6


def compute_velocity_features(
    h_curr: torch.Tensor,
    h_prev: torch.Tensor | None,
    h_anchor: torch.Tensor | None,
) -> torch.Tensor:
    """Compute 6 strictly index-free, bounded per-token trajectory features [N, 6] in FP32.

    Args:
        h_curr: [N, D] hidden state entering MoE at current visit.
        h_prev: [N, D] hidden state at previous visit of same physical layer (None at visit 0).
        h_anchor: [N, D] hidden state at visit 0 of same physical layer (None at visit 0).
    """
    hc = h_curr.detach().float()
    n_tok, dim = hc.shape
    inv_sqrt_d = 1.0 / math.sqrt(max(dim, 1))
    norm_c = hc.norm(dim=-1)  # [N]

    if h_prev is None:
        dh = hc
        cos_prev = torch.zeros(n_tok, device=hc.device, dtype=torch.float32)
        cos_anc = torch.ones(n_tok, device=hc.device, dtype=torch.float32)
        cos_dh_anc = torch.ones(n_tok, device=hc.device, dtype=torch.float32)
    else:
        hp = h_prev.detach().float()
        ha = h_anchor.detach().float() if h_anchor is not None else hp
        dh = hc - hp
        cos_prev = F.cosine_similarity(hc, hp, dim=-1).clamp(-1.0, 1.0)
        cos_anc = F.cosine_similarity(hc, ha, dim=-1).clamp(-1.0, 1.0)
        cos_dh_anc = F.cosine_similarity(dh, ha, dim=-1).clamp(-1.0, 1.0)

    norm_dh = dh.norm(dim=-1)
    f0 = torch.log1p(norm_c * inv_sqrt_d).clamp(0.0, 6.0)
    f1 = torch.log1p(norm_dh * inv_sqrt_d).clamp(0.0, 6.0)
    f2 = (norm_dh / (norm_c + 1e-6)).clamp(0.0, 4.0)
    return torch.stack([f0, f1, f2, cos_prev, cos_anc, cos_dh_anc], dim=-1)


def pure_torch_topk_routing(
    x: torch.Tensor,
    weight: torch.Tensor,
    n_experts: int,
    topk: int,
    score_func: str,
    router_bias: torch.Tensor | None,
    scaling_factor: float | None,
    delta_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
    """Exact pure-PyTorch TopK routing matching xllm.modules.moe.router.TopKRouting with optional delta_logits."""
    logits = F.linear(x, weight).to(torch.float32)
    if delta_logits is not None:
        logits = logits + delta_logits.to(torch.float32)

    if score_func == "softmax":
        original_scores = F.softmax(logits, dim=-1, dtype=torch.float32)
    elif score_func == "sigmoid":
        original_scores = torch.sigmoid(logits)
    else:
        raise ValueError(f"Unknown score_func: {score_func}")

    sel_scores = original_scores + router_bias.to(original_scores) if router_bias is not None else original_scores
    indices = torch.topk(sel_scores, topk, dim=-1)[1]
    scores = torch.gather(original_scores, dim=-1, index=indices)
    if topk > 1:
        scores = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    if scaling_factor:
        scores = scores * scaling_factor
    scores = scores.to(x.dtype)
    tokens_per_expert = torch.bincount(indices.flatten(), minlength=n_experts)
    return scores, indices, tokens_per_expert, None


# ---------------------------------------------------------------------------
# Stage 1 Diagnostic: Shared-Subspace vs Per-Loop Split SVD Concentration
# ---------------------------------------------------------------------------

def analyze_subspace_concentration(
    grads_a: dict[int, torch.Tensor],
    grads_b: dict[int, torch.Tensor],
    selected_experts: list[tuple[int, int]],
    *,
    rank_per_loop: int,
    n_loops: int,
) -> dict[str, Any]:
    """Measure held-out Split-B gradient variance explained by matched-budget bases fit on Split A.

    Compares at STRICTLY equal parameter count (rank K = n_loops * rank_per_loop):
      1. per_loop_split: R independent rank-r0 SVD subspaces fit per loop on Split A.
      2. shared_subspace_single: 1 shared rank-K subspace fit on mean Split-A gradient (no loop modulation).
      3. shared_subspace_diag: 1 shared rank-K subspace fit on stacked Split-A gradients, with optimal
         per-loop diagonal coordinates on Split B (tests the shared-subspace + diagonal-core hypothesis!).
      4. shared_subspace_full_core: 1 shared rank-K subspace (U_K, V_K) projecting G_B[r] onto span(U_K, V_K).
    """
    k_shared = n_loops * rank_per_loop
    ev_per_loop: list[float] = []
    ev_shared_mean: list[float] = []
    ev_shared_diag: list[float] = []
    ev_shared_sub: list[float] = []

    for lid, e in selected_experts:
        ga = grads_a[lid][e].float()  # [R, D, I]
        gb = grads_b[lid][e].float()  # [R, D, I]
        tot_b = float(gb.pow(2).sum().item())
        if tot_b < 1e-16:
            continue

        # 1) R separate rank-r0 SVDs fit on Split A[r], evaluated by projecting Split B[r]
        cap_pl = 0.0
        for r in range(n_loops):
            u_r, _s_r, vh_r = torch.linalg.svd(ga[r], full_matrices=False)
            u_k = u_r[:, :rank_per_loop]  # [D, r0]
            v_k = vh_r[:rank_per_loop, :]  # [r0, I]
            # Orthogonal projection of gb[r] onto column/row rank-r0 subspace U_k (U_k^T gb[r] V_k^T) V_k
            proj_core = (u_k.T @ gb[r]) @ v_k.T  # [r0, r0]
            cap_pl += float(proj_core.pow(2).sum().item())
        ev_per_loop.append(cap_pl / tot_b)

        # 2) Shared rank-K subspace fit on Split-A mean gradient (single shared matrix direction)
        ga_mean = ga.mean(dim=0)  # [D, I]
        u_m, _s_m, vh_m = torch.linalg.svd(ga_mean, full_matrices=False)
        k_eff = min(k_shared, u_m.shape[1])
        um_k = u_m[:, :k_eff]
        vm_k = vh_m[:k_eff, :]
        # Same single rank-K matrix projected from mean B (tests no-modulation ceiling)
        gb_mean = gb.mean(dim=0)
        proj_m = (um_k.T @ gb_mean) @ vm_k.T
        # Energy captured by approximating every loop gb[r] with the shared projection of gb_mean
        approx_m = um_k @ proj_m @ vm_k
        resid_m = float((gb - approx_m.unsqueeze(0)).pow(2).sum().item())
        ev_shared_mean.append(max(0.0, 1.0 - resid_m / tot_b))

        # 3 & 4) Shared rank-K basis fit jointly across ALL loops of Split A:
        # Left singular vectors from [ga[0], ..., ga[R-1]] (D x R*I), right from stacked (R*D x I)
        ga_col = torch.cat([ga[r] for r in range(n_loops)], dim=1)  # [D, R*I]
        ga_row = torch.cat([ga[r] for r in range(n_loops)], dim=0)  # [R*D, I]
        u_all, _, _ = torch.linalg.svd(ga_col, full_matrices=False)
        _, _, vh_all = torch.linalg.svd(ga_row, full_matrices=False)
        k_j = min(k_shared, u_all.shape[1], vh_all.shape[0])
        u_j = u_all[:, :k_j]  # [D, K]
        v_j = vh_all[:k_j, :]  # [K, I]

        # Align paired singular directions via SVD of mean projected core so diagonal entries are aligned
        mean_core = (u_j.T @ ga_mean) @ v_j.T  # [K, K]
        uc, _, vhc = torch.linalg.svd(mean_core, full_matrices=False)
        u_aligned = u_j @ uc  # [D, K]
        v_aligned = vhc @ v_j  # [K, I]

        cap_sub = 0.0
        cap_diag = 0.0
        for r in range(n_loops):
            core_r = (u_aligned.T @ gb[r]) @ v_aligned.T  # [K, K]
            cap_sub += float(core_r.pow(2).sum().item())
            cap_diag += float(torch.diagonal(core_r).pow(2).sum().item())
        ev_shared_sub.append(cap_sub / tot_b)
        ev_shared_diag.append(cap_diag / tot_b)

    return {
        "n_experts_analyzed": len(ev_per_loop),
        "rank_per_loop": rank_per_loop,
        "rank_shared": k_shared,
        "ev_per_loop_split_mean": float(np.mean(ev_per_loop)) if ev_per_loop else 0.0,
        "ev_shared_mean_nomod": float(np.mean(ev_shared_mean)) if ev_shared_mean else 0.0,
        "ev_shared_subspace_diag": float(np.mean(ev_shared_diag)) if ev_shared_diag else 0.0,
        "ev_shared_subspace_full": float(np.mean(ev_shared_sub)) if ev_shared_sub else 0.0,
        "shared_over_split_ratio": (
            float(np.mean(ev_shared_sub)) / max(float(np.mean(ev_per_loop)), 1e-8)
            if ev_per_loop
            else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Unified Strictly Parameter-Matched Looped MoE Adapter Family
# ---------------------------------------------------------------------------

class SubspaceMoEAdapterHost(nn.Module):
    """Unified parameter-matched host covering all 8 adapted conditions of Direction 2.

    Modes:
      - per_loop:             R separate rank-r0 adapters per expert (clamps index on R_test > R_train)
      - pure_shared:          1 shared rank-K adapter per expert, no loop modulation
      - loop_dropout:         1 shared rank-K adapter with Loop Dropout (p=0.5, 1/(1-p) scaling during train)
      - shared_plus_scalar:   1 shared rank-K adapter + per-loop index scalar (clamps on R_test > R_train)
      - shared_plus_diag_idx: 1 shared rank-K adapter + per-loop index diagonal core (clamps on R_test > R_train)
      - velocity_submoe:      1 shared rank-K adapter + INDEX-FREE per-token velocity diagonal core
      - velocity_submoe_ld:   velocity_submoe + Loop Dropout (p=0.35)
      - velocity_submoe_full: velocity_submoe_ld + INDEX-FREE velocity-gated hysteresis router bias
    """

    def __init__(
        self,
        selected_experts: list[tuple[int, int]],
        tracked_layers: list[int],
        *,
        mode: str,
        rank_per_loop: int,
        train_loops: int,
        n_experts: int,
        model_dim: int,
        inter_dim: int,
        alpha: float = 6.0,
        loop_dropout_p: float = 0.5,
        use_router_adapter: bool = False,
        target_param_cap: int | None = None,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.train_loops = int(train_loops)
        self.n_experts = int(n_experts)
        self.tracked_layers = list(tracked_layers)
        self.loop_dropout_p = float(loop_dropout_p)
        self.use_router_adapter = bool(use_router_adapter)

        self.by_layer: dict[int, list[int]] = {}
        for lid, e in selected_experts:
            self.by_layer.setdefault(lid, []).append(e)

        if mode == "per_loop":
            self.rank = int(rank_per_loop)
        else:
            rank_shared = int(rank_per_loop * train_loops)
            # Trim shared rank if needed so total params NEVER exceed target_param_cap!
            if target_param_cap is not None:
                n_sel_total = sum(len(v) for v in self.by_layer.values())
                per_rank_cost = n_sel_total * (model_dim + inter_dim)
                extra = 0
                if mode == "shared_plus_scalar":
                    extra = n_sel_total * 2 * train_loops
                elif mode == "shared_plus_diag_idx":
                    extra = n_sel_total * (train_loops * rank_shared + train_loops)
                elif mode in ("velocity_submoe", "velocity_submoe_ld", "velocity_submoe_full"):
                    extra = n_sel_total * (VELOCITY_FEAT_DIM * rank_shared + VELOCITY_FEAT_DIM)
                    if use_router_adapter:
                        extra += len(tracked_layers) * (VELOCITY_FEAT_DIM * n_experts + VELOCITY_FEAT_DIM + 1)
                while rank_shared > 1 and (rank_shared * per_rank_cost + extra > target_param_cap):
                    rank_shared -= 1
                    if mode == "shared_plus_diag_idx":
                        extra = n_sel_total * (train_loops * rank_shared + train_loops)
                    elif mode in ("velocity_submoe", "velocity_submoe_ld", "velocity_submoe_full"):
                        extra = n_sel_total * (VELOCITY_FEAT_DIM * rank_shared + VELOCITY_FEAT_DIM)
                        if use_router_adapter:
                            extra += len(tracked_layers) * (VELOCITY_FEAT_DIM * n_experts + VELOCITY_FEAT_DIM + 1)
            self.rank = rank_shared

        # rsLoRA scaling: alpha / sqrt(rank)
        self.scale = float(alpha) / math.sqrt(max(self.rank, 1))
        self.params = nn.ParameterDict()

        for lid, e_list in sorted(self.by_layer.items()):
            n_sel = len(e_list)
            if mode == "per_loop":
                A = nn.Parameter(torch.empty(n_sel, train_loops, self.rank, inter_dim, dtype=torch.bfloat16))
                B = nn.Parameter(torch.zeros(n_sel, train_loops, model_dim, self.rank, dtype=torch.bfloat16))
                nn.init.kaiming_uniform_(A, a=math.sqrt(5))
                self.params[f"L{lid}_A"] = A
                self.params[f"L{lid}_B"] = B
            else:
                A = nn.Parameter(torch.empty(n_sel, self.rank, inter_dim, dtype=torch.bfloat16))
                B = nn.Parameter(torch.zeros(n_sel, model_dim, self.rank, dtype=torch.bfloat16))
                nn.init.kaiming_uniform_(A, a=math.sqrt(5))
                self.params[f"L{lid}_A"] = A
                self.params[f"L{lid}_B"] = B

                if mode == "shared_plus_scalar":
                    self.params[f"L{lid}_slora"] = nn.Parameter(torch.zeros(n_sel, train_loops, dtype=torch.bfloat16))
                    self.params[f"L{lid}_sbase"] = nn.Parameter(torch.zeros(n_sel, train_loops, dtype=torch.bfloat16))
                elif mode == "shared_plus_diag_idx":
                    self.params[f"L{lid}_cdiag"] = nn.Parameter(torch.zeros(n_sel, train_loops, self.rank, dtype=torch.bfloat16))
                    self.params[f"L{lid}_sbase"] = nn.Parameter(torch.zeros(n_sel, train_loops, dtype=torch.bfloat16))
                elif mode in ("velocity_submoe", "velocity_submoe_ld", "velocity_submoe_full"):
                    # Zero-init linear velocity maps so step 0 == pure_shared!
                    self.params[f"L{lid}_Wg"] = nn.Parameter(torch.zeros(n_sel, VELOCITY_FEAT_DIM, self.rank, dtype=torch.bfloat16))
                    self.params[f"L{lid}_Wb"] = nn.Parameter(torch.zeros(n_sel, VELOCITY_FEAT_DIM, dtype=torch.bfloat16))

        if self.use_router_adapter:
            for lid in self.tracked_layers:
                # Velocity-driven router bias [6, E] + hysteresis gate [6] + bias [1] (all zero-init!)
                self.params[f"R{lid}_Wvel"] = nn.Parameter(torch.zeros(VELOCITY_FEAT_DIM, n_experts, dtype=torch.bfloat16))
                self.params[f"R{lid}_Whyst"] = nn.Parameter(torch.zeros(VELOCITY_FEAT_DIM, dtype=torch.bfloat16))
                self.params[f"R{lid}_bhyst"] = nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def router_delta_logits(
        self,
        lid: int,
        psi: torch.Tensor,  # [B*L, 6] fp32
        prev_mask: torch.Tensor | None,  # [B*L, E] fp32 or None at visit 0
        bsz: int,
        slen: int,
    ) -> torch.Tensor | None:
        if not self.use_router_adapter or f"R{lid}_Wvel" not in self.params:
            return None
        w_vel = self.params[f"R{lid}_Wvel"].float()  # [6, E]
        delta = psi @ w_vel  # [B*L, E]
        if prev_mask is not None:
            w_h = self.params[f"R{lid}_Whyst"].float()
            b_h = self.params[f"R{lid}_bhyst"].float()
            beta_h = torch.tanh(psi @ w_h + b_h)  # [B*L], bounded hysteresis strength in [-1, 1]
            delta = delta + beta_h.unsqueeze(-1) * prev_mask
        return delta.view(bsz, slen, self.n_experts)

    def apply_expert_delta(
        self,
        lid: int,
        visit_r: int,
        hidden: torch.Tensor,
        y: torch.Tensor,
        group_sizes: list[int],
        *,
        permuted_psi: torch.Tensor | None = None,
        loop_gate: float = 1.0,
    ) -> torch.Tensor:
        if lid not in self.by_layer or loop_gate == 0.0:
            return y
        e_list = self.by_layer[lid]
        A = self.params[f"L{lid}_A"]
        B = self.params[f"L{lid}_B"]

        offsets = []
        acc = 0
        for g in group_sizes:
            offsets.append(acc)
            acc += g

        delta_y = torch.zeros_like(y)
        any_hit = False
        # Clamped index for index-bound baselines when extrapolating R_test > R_train
        r_idx = min(visit_r, self.train_loops - 1)

        if self.mode == "per_loop":
            for pos_i, e_idx in enumerate(e_list):
                cnt = group_sizes[e_idx]
                if cnt == 0:
                    continue
                any_hit = True
                s = offsets[e_idx]
                h_e = hidden[s : s + cnt]
                dy = F.linear(F.linear(h_e, A[pos_i, r_idx]), B[pos_i, r_idx]) * (self.scale * loop_gate)
                delta_y[s : s + cnt] = dy

        elif self.mode in ("pure_shared", "loop_dropout"):
            for pos_i, e_idx in enumerate(e_list):
                cnt = group_sizes[e_idx]
                if cnt == 0:
                    continue
                any_hit = True
                s = offsets[e_idx]
                h_e = hidden[s : s + cnt]
                dy = F.linear(F.linear(h_e, A[pos_i]), B[pos_i]) * (self.scale * loop_gate)
                delta_y[s : s + cnt] = dy

        elif self.mode == "shared_plus_scalar":
            s_lora = self.params[f"L{lid}_slora"]
            s_base = self.params[f"L{lid}_sbase"]
            for pos_i, e_idx in enumerate(e_list):
                cnt = group_sizes[e_idx]
                if cnt == 0:
                    continue
                any_hit = True
                s = offsets[e_idx]
                h_e = hidden[s : s + cnt]
                mid = F.linear(h_e, A[pos_i])
                dy = (1.0 + s_lora[pos_i, r_idx]) * F.linear(mid, B[pos_i]) * (self.scale * loop_gate)
                dy = dy + (s_base[pos_i, r_idx] * loop_gate) * y[s : s + cnt]
                delta_y[s : s + cnt] = dy

        elif self.mode == "shared_plus_diag_idx":
            c_diag = self.params[f"L{lid}_cdiag"]
            s_base = self.params[f"L{lid}_sbase"]
            for pos_i, e_idx in enumerate(e_list):
                cnt = group_sizes[e_idx]
                if cnt == 0:
                    continue
                any_hit = True
                s = offsets[e_idx]
                h_e = hidden[s : s + cnt]
                mid = F.linear(h_e, A[pos_i]) * (1.0 + c_diag[pos_i, r_idx]).unsqueeze(0)
                dy = F.linear(mid, B[pos_i]) * (self.scale * loop_gate)
                dy = dy + (s_base[pos_i, r_idx] * loop_gate) * y[s : s + cnt]
                delta_y[s : s + cnt] = dy

        elif self.mode in ("velocity_submoe", "velocity_submoe_ld", "velocity_submoe_full"):
            assert permuted_psi is not None
            Wg = self.params[f"L{lid}_Wg"]  # [n_sel, 6, K]
            Wb = self.params[f"L{lid}_Wb"]  # [n_sel, 6]
            psi_bf16 = permuted_psi.to(dtype=hidden.dtype)
            for pos_i, e_idx in enumerate(e_list):
                cnt = group_sizes[e_idx]
                if cnt == 0:
                    continue
                any_hit = True
                s = offsets[e_idx]
                h_e = hidden[s : s + cnt]  # [cnt, I]
                psi_e = psi_bf16[s : s + cnt]  # [cnt, 6]
                # Bounded velocity modulation in (0, 2) via 1 + tanh(...) for rock-solid R_test extrapolation!
                g_e = 1.0 + torch.tanh(psi_e @ Wg[pos_i])  # [cnt, K]
                sb_e = torch.tanh(psi_e @ Wb[pos_i])  # [cnt]
                mid = F.linear(h_e, A[pos_i]) * g_e  # [cnt, K]
                dy = F.linear(mid, B[pos_i]) * (self.scale * loop_gate)
                dy = dy + (sb_e * loop_gate).unsqueeze(-1) * y[s : s + cnt]
                delta_y[s : s + cnt] = dy
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return y + delta_y if any_hit else y


# ---------------------------------------------------------------------------
# Runtime Hook Context: Exact Velocity Capture, Router Stats, Loop Dropout & Single-Loop Probe
# ---------------------------------------------------------------------------

@contextmanager
def attached_vsubmoe_context(
    model,
    adapter_host: SubspaceMoEAdapterHost | None,
    tracked_layers: list[int],
    *,
    n_experts: int,
    collect_router_stats: bool = False,
):
    """Attach clean forward hooks on tracked NormalizedMoE modules for training or evaluation."""
    from einops import rearrange
    from xllm.modules.fused_ops import memory_efficient_dropout, mgmm

    orig_moe_fwds: dict[int, Any] = {}
    orig_rtr_fwds: dict[int, Any] = {}
    orig_exp_fwds: dict[int, Any] = {}

    state: dict[str, Any] = {
        "visit": {lid: 0 for lid in tracked_layers},
        "h_prev": {lid: None for lid in tracked_layers},
        "h_anchor": {lid: None for lid in tracked_layers},
        "psi_curr": {lid: None for lid in tracked_layers},
        "perm_psi": {lid: None for lid in tracked_layers},
        "prev_mask": {lid: None for lid in tracked_layers},
        "loop_gates": None,  # Optional list[float] indexed by visit_r (for Loop Dropout or Single-Loop probe!)
        "load_counts": {lid: [] for lid in tracked_layers},  # list over visits of [E] counts
        "prev_indices": {lid: None for lid in tracked_layers},
        "flip_numer": 0,
        "flip_denom": 0,
    }

    def reset_sequence(loop_gates: list[float] | None = None) -> None:
        state["loop_gates"] = loop_gates
        for lid in tracked_layers:
            state["visit"][lid] = 0
            state["h_prev"][lid] = None
            state["h_anchor"][lid] = None
            state["psi_curr"][lid] = None
            state["perm_psi"][lid] = None
            state["prev_mask"][lid] = None
            state["prev_indices"][lid] = None

    def reset_router_stats() -> None:
        for lid in tracked_layers:
            state["load_counts"][lid] = []
        state["flip_numer"] = 0
        state["flip_denom"] = 0

    def summarize_router_stats() -> dict[str, float]:
        cos_vals: list[float] = []
        for lid in tracked_layers:
            lc = state["load_counts"][lid]
            if len(lc) < 2:
                continue
            mat = np.stack(lc, axis=0).astype(np.float64)  # [R, E]
            norms = np.linalg.norm(mat, axis=1, keepdims=True).clip(min=1e-12)
            mat_n = mat / norms
            sim = mat_n @ mat_n.T
            off = ~np.eye(sim.shape[0], dtype=bool)
            cos_vals.append(float(sim[off].mean()))
        return {
            "cross_loop_load_cosine": float(np.mean(cos_vals)) if cos_vals else 0.0,
            "consecutive_topk_flip_rate": float(state["flip_numer"] / max(state["flip_denom"], 1)),
        }

    try:
        for lid in tracked_layers:
            moe_mod = model.layers[lid].moe
            rtr_mod = moe_mod.router
            exp_mod = moe_mod.experts
            orig_moe_fwds[lid] = moe_mod.forward
            orig_rtr_fwds[lid] = rtr_mod.forward
            orig_exp_fwds[lid] = exp_mod.forward

            def _make_moe_fwd(l_id: int, orig_fn):
                def _fwd(x: torch.Tensor, load_balancing_type=None):
                    bsz, slen, dim = x.shape
                    h_flat = x.detach().reshape(bsz * slen, dim)
                    psi = compute_velocity_features(h_flat, state["h_prev"][l_id], state["h_anchor"][l_id])
                    if state["h_anchor"][l_id] is None:
                        state["h_anchor"][l_id] = h_flat
                    state["h_prev"][l_id] = h_flat
                    state["psi_curr"][l_id] = psi
                    return orig_fn(x, load_balancing_type)

                return _fwd

            def _make_rtr_fwd(l_id: int, mod, orig_fn):
                def _fwd(x: torch.Tensor, load_balancing_type=None):
                    bsz, slen, _ = x.shape
                    r = state["visit"][l_id]
                    psi = state["psi_curr"][l_id]
                    delta_logits = None
                    if adapter_host is not None and adapter_host.use_router_adapter and psi is not None:
                        gates = state["loop_gates"]
                        lg = gates[r] if (gates is not None and r < len(gates)) else 1.0
                        if lg != 0.0:
                            dl = adapter_host.router_delta_logits(l_id, psi, state["prev_mask"][l_id], bsz, slen)
                            if dl is not None:
                                delta_logits = dl * lg

                    if delta_logits is not None:
                        scores, indices, tpe, aux = pure_torch_topk_routing(
                            x,
                            mod.weight,
                            mod.num_experts,
                            mod.topk,
                            mod.score_func,
                            mod.bias,
                            mod.scaling_factor,
                            delta_logits=delta_logits,
                        )
                    else:
                        scores, indices, tpe, aux = orig_fn(x, load_balancing_type)

                    # Compute exact 1-to-1 permutation of per-token velocity features into dispatched expert order!
                    if psi is not None:
                        flat_idx = indices.flatten()
                        sorted_idx = torch.argsort(flat_idx, stable=True)
                        perm_idx = sorted_idx // mod.topk
                        state["perm_psi"][l_id] = psi.index_select(0, perm_idx)

                    # Track previous-loop expert selection mask for hysteresis & flip stats
                    idx_2d = indices.reshape(bsz * slen, mod.topk)
                    if (adapter_host is not None and adapter_host.use_router_adapter) or collect_router_stats:
                        cur_mask = torch.zeros((bsz * slen, n_experts), device=x.device, dtype=torch.float32)
                        cur_mask.scatter_(1, idx_2d, 1.0)
                        if collect_router_stats:
                            lc_list = state["load_counts"][l_id]
                            tpe_np = tpe.detach().cpu().numpy().astype(np.float64)
                            if r < len(lc_list):
                                lc_list[r] += tpe_np
                            else:
                                lc_list.append(tpe_np.copy())
                            if state["prev_mask"][l_id] is not None:
                                overlap = (cur_mask * state["prev_mask"][l_id]).sum().item()
                                total_sel = bsz * slen * mod.topk
                                state["flip_numer"] += int(round(total_sel - overlap))
                                state["flip_denom"] += int(total_sel)
                        state["prev_mask"][l_id] = cur_mask.detach()

                    return scores, indices, tpe, aux

                return _fwd

            def _make_exp_fwd(l_id: int, mod, orig_fn):
                def _fwd(x: torch.Tensor, group_sizes: list[int]) -> torch.Tensor:
                    r = state["visit"][l_id]
                    state["visit"][l_id] = r + 1
                    if adapter_host is None or l_id not in adapter_host.by_layer:
                        return orig_fn(x, group_sizes)
                    gates = state["loop_gates"]
                    lg = gates[r] if (gates is not None and r < len(gates)) else 1.0
                    w1 = rearrange(mod.weight1, "(n s) d -> n s d", n=mod.n_local_experts)
                    w2 = rearrange(mod.weight2, "(n s) d -> n s d", n=mod.n_local_experts)
                    w3 = rearrange(mod.weight3, "(n s) d -> n s d", n=mod.n_local_experts)
                    h1 = F.silu(mgmm(x, w1, group_sizes, True, mod.backend))
                    h3 = mgmm(x, w3, group_sizes, True, mod.backend)
                    hidden = memory_efficient_dropout(h1 * h3, mod.hidden_dropout, mod.training)
                    y = mgmm(hidden, w2, group_sizes, True, mod.backend)
                    return adapter_host.apply_expert_delta(
                        l_id,
                        r,
                        hidden,
                        y,
                        group_sizes,
                        permuted_psi=state["perm_psi"][l_id],
                        loop_gate=lg,
                    )

                return _fwd

            moe_mod.forward = _make_moe_fwd(lid, orig_moe_fwds[lid])
            rtr_mod.forward = _make_rtr_fwd(lid, rtr_mod, orig_rtr_fwds[lid])
            exp_mod.forward = _make_exp_fwd(lid, exp_mod, orig_exp_fwds[lid])

        yield reset_sequence, reset_router_stats, summarize_router_stats
    finally:
        for lid, fn in orig_moe_fwds.items():
            model.layers[lid].moe.forward = fn
        for lid, fn in orig_rtr_fwds.items():
            model.layers[lid].moe.router.forward = fn
        for lid, fn in orig_exp_fwds.items():
            model.layers[lid].moe.experts.forward = fn


@torch.inference_mode()
def evaluate_nll_at_depth(
    model,
    seqs: list[dict[str, Any]],
    depth: int,
    reset_seq_fn,
    device: torch.device,
    *,
    loop_gates: list[float] | None = None,
) -> tuple[list[float], float, float, float]:
    from loop_research.idea2_expert_unbinding import compute_lm_loss

    nlls: list[float] = []
    text_nlls: list[float] = []
    code_nlls: list[float] = []
    with set_ifm_loop_depth(model, depth):
        for item in seqs:
            reset_seq_fn(loop_gates=loop_gates)
            val = float(compute_lm_loss(model, item["tokens"], device).item())
            nlls.append(val)
            if item["domain"] == "code":
                code_nlls.append(val)
            else:
                text_nlls.append(val)
    return (
        nlls,
        float(np.mean(nlls)),
        float(np.mean(text_nlls)) if text_nlls else 0.0,
        float(np.mean(code_nlls)) if code_nlls else 0.0,
    )


@torch.inference_mode()
def evaluate_arc_mc_ifm(
    model,
    tokenizer,
    arc_samples: list[dict[str, Any]],
    depth: int,
    reset_seq_fn,
    device: torch.device,
) -> tuple[float, list[int]]:
    """Evaluate ARC-Challenge exact length-normalized log-likelihood accuracy on IFM model."""
    flags: list[int] = []
    with set_ifm_loop_depth(model, depth):
        for item in arc_samples:
            prompt = f"Question: {item['question']}\nAnswer:"
            p_ids = tokenizer.encode(prompt, bos=True, eos=False)
            scores = []
            for c_txt in item["choice_texts"]:
                c_ids = tokenizer.encode(" " + c_txt, bos=False, eos=False)
                if not c_ids:
                    scores.append(-1e9)
                    continue
                full = p_ids + c_ids
                inp = torch.tensor([full[:-1]], device=device, dtype=torch.long)
                tgt = torch.tensor(full[len(p_ids) :], device=device, dtype=torch.long)
                reset_seq_fn()
                logits, _, _ = model(inp, multi_segments=False, cache=None)
                sl_logits = logits[0, len(p_ids) - 1 :, :].float()
                lp = F.log_softmax(sl_logits, dim=-1)
                tok_ll = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                scores.append(float(tok_ll.mean().item()))
            pred = int(np.argmax(scores))
            flags.append(1 if pred == item["gold_idx"] else 0)
    return float(np.mean(flags)), flags


def train_and_evaluate_vsubmoe_condition(
    model,
    tokenizer,
    cname: str,
    adapter_host: SubspaceMoEAdapterHost | None,
    *,
    tracked_layers: list[int],
    train_loops: int,
    eval_depths: list[int],
    arc_depths: list[int],
    n_experts: int,
    train_seqs: list[dict[str, Any]],
    eval_seqs: list[dict[str, Any]],
    arc_samples: list[dict[str, Any]],
    lr: float,
    grad_accum: int,
    device: torch.device,
    seed: int,
    run_single_loop_diag: bool = True,
) -> dict[str, Any]:
    """Train one matched-budget adapter and evaluate across depths, single-loop probe, ARC, and routing."""
    from loop_research.idea2_expert_unbinding import compute_lm_loss

    for p in model.parameters():
        p.requires_grad_(False)

    train_losses: list[float] = []
    n_params = 0
    if adapter_host is not None:
        adapter_host = adapter_host.to(device)
        for p in adapter_host.parameters():
            p.requires_grad_(True)
        n_params = adapter_host.num_trainable_params()

    rng = np.random.default_rng(seed)

    with attached_vsubmoe_context(
        model, adapter_host, tracked_layers, n_experts=n_experts, collect_router_stats=False
    ) as (reset_seq, _reset_rs, _sum_rs):
        if adapter_host is not None:
            use_ld = adapter_host.mode in ("loop_dropout", "velocity_submoe_ld", "velocity_submoe_full")
            p_drop = adapter_host.loop_dropout_p
            inv_surv = 1.0 / max(1.0 - p_drop, 1e-6)

            optimizer = torch.optim.AdamW(adapter_host.parameters(), lr=lr, weight_decay=0.0)
            n_opt_steps = max(1, (len(train_seqs) + grad_accum - 1) // grad_accum)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_opt_steps, eta_min=lr * 0.1)

            optimizer.zero_grad(set_to_none=True)
            for i, item in enumerate(train_seqs, 1):
                if use_ld:
                    # Exact Loop Dropout [2609.34218]: Bernoulli(1-p) per loop with 1/(1-p) inverse-survival scaling
                    mask = rng.random(train_loops) >= p_drop
                    if not mask.any():
                        mask[int(rng.integers(0, train_loops))] = True
                    gates = [inv_surv if m else 0.0 for m in mask]
                else:
                    gates = None

                reset_seq(loop_gates=gates)
                loss = compute_lm_loss(model, item["tokens"], device)
                train_losses.append(float(loss.item()))
                (loss / grad_accum).backward()
                if i % grad_accum == 0 or i == len(train_seqs):
                    torch.nn.utils.clip_grad_norm_(adapter_host.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

        # Multi-depth held-out NLL evaluation (0.5x, 1x, 1.5x, 2x R_train)
        depth_nlls: dict[str, dict[str, Any]] = {}
        for d in eval_depths:
            per_seq, mean_nll, txt_nll, cod_nll = evaluate_nll_at_depth(model, eval_seqs, d, reset_seq, device)
            depth_nlls[str(d)] = {
                "nll": mean_nll,
                "nll_text": txt_nll,
                "nll_code": cod_nll,
                "per_seq": per_seq,
            }

        # Single-loop activation diagnostic (Loop Dropout Fig. 1b) at R_train
        single_loop_nlls: list[float] = []
        if run_single_loop_diag and adapter_host is not None:
            for r_only in range(train_loops):
                gates = [1.0 if r == r_only else 0.0 for r in range(train_loops)]
                _, sl_mean, _, _ = evaluate_nll_at_depth(
                    model, eval_seqs[:16], train_loops, reset_seq, device, loop_gates=gates
                )
                single_loop_nlls.append(sl_mean)

        # ARC-Challenge MC accuracy at R_train and 1.5x R_train
        arc_res: dict[str, dict[str, Any]] = {}
        for d in arc_depths:
            acc, flags = evaluate_arc_mc_ifm(model, tokenizer, arc_samples, d, reset_seq, device)
            arc_res[str(d)] = {"accuracy": acc, "flags": flags}

    # Measure routing collapse & stability stats cleanly on eval_seqs[:16] at R_train and 1.5x R_train
    router_stats_by_depth: dict[str, dict[str, float]] = {}
    with attached_vsubmoe_context(
        model, adapter_host, tracked_layers, n_experts=n_experts, collect_router_stats=True
    ) as (reset_seq_rs, reset_rs, sum_rs):
        for d in arc_depths:
            reset_rs()
            evaluate_nll_at_depth(model, eval_seqs[:16], d, reset_seq_rs, device)
            router_stats_by_depth[str(d)] = sum_rs()

    if adapter_host is not None:
        del adapter_host
        torch.cuda.empty_cache()

    return {
        "condition": cname,
        "trainable_params": n_params,
        "train_loss_first8": float(np.mean(train_losses[:8])) if train_losses else 0.0,
        "train_loss_last8": float(np.mean(train_losses[-8:])) if train_losses else 0.0,
        "depth_nlls": depth_nlls,
        "single_loop_nlls": single_loop_nlls,
        "arc_results": arc_res,
        "router_stats": router_stats_by_depth,
    }


# ---------------------------------------------------------------------------
# Full Per-Model Pipeline (IFM Ouro-MoE & IFM Huginn-MoE)
# ---------------------------------------------------------------------------

def run_model_vsubmoe(
    model_key: str,
    mspec: dict[str, Any],
    *,
    model_dir: Path,
    dataset_dir: Path,
    arc_samples: list[dict[str, Any]],
    cfg: dict[str, Any],
    device: torch.device,
    run_obj: Any,
) -> dict[str, Any]:
    from loop_research.idea2_expert_unbinding import (
        paired_bootstrap_ci,
        per_visit_w2_grad_collector,
        score_expert_conflicts,
        select_group_experts,
        verify_backward_and_decomp,
    )

    artifact_dir = model_dir / mspec["local_dir"]
    tracked_layers: list[int] = list(mspec["tracked_layers"])
    train_loops = int(mspec["n_loops"])
    eval_depths: list[int] = [int(d) for d in mspec["eval_depths"]]
    arc_depths: list[int] = [int(d) for d in mspec["arc_depths"]]
    k_per_layer = int(mspec["k_per_layer"])
    rank_per_loop = int(mspec["rank_per_loop"])
    seed = int(cfg.get("seed", 20261124))

    print(f"\n[vsubmoe:{model_key}] Loading {artifact_dir} ...")
    model, tokenizer, _mcfg = load_ifm_model_for_autograd(artifact_dir, device=str(device), attention_backend="sdpa")
    n_experts = int(model.layers[tracked_layers[0]].moe.experts.n_experts)
    model_dim = int(model.layers[tracked_layers[0]].moe.experts.model_dim)
    inter_dim = int(model.layers[tracked_layers[0]].moe.experts.inter_dim)

    splits = build_disjoint_text_code_splits(
        dataset_dir,
        lambda t: tokenizer.encode(t, bos=True, eos=False),
        seq_len=int(cfg.get("seq_len", 320)),
        n_calib_a=int(cfg.get("n_calib_a", 16)),
        n_calib_b=int(cfg.get("n_calib_b", 16)),
        n_adapt_train=int(cfg.get("n_adapt_train", 64)),
        n_eval_heldout=int(cfg.get("n_eval_heldout", 32)),
        seed=seed,
    )

    # 0) Verify backward exactness + pure-PyTorch router exactness + depth switching
    with run_obj.phase(f"{model_key}_verify", total=1, unit="check") as pv:
        bwd_verify = verify_backward_and_decomp(
            model, tracked_layers[0], train_loops, n_experts, model_dim, inter_dim, splits["calib_a"][0]["tokens"], device
        )
        # Verify pure_torch_topk_routing matches native TopKRouter exactly
        rtr0 = model.layers[tracked_layers[0]].moe.router
        x_test = torch.randn(1, 16, model_dim, device=device, dtype=torch.bfloat16)
        with torch.inference_mode():
            s_nat, idx_nat, tpe_nat, _ = rtr0(x_test, None)
            s_py, idx_py, tpe_py, _ = pure_torch_topk_routing(
                x_test, rtr0.weight, rtr0.num_experts, rtr0.topk, rtr0.score_func, rtr0.bias, rtr0.scaling_factor
            )
            rtr_diff = float((s_nat.float() - s_py.float()).abs().max().item())
            idx_match = bool(torch.equal(idx_nat, idx_py))
            # Smoke-test depth extrapolation forward
            with set_ifm_loop_depth(model, eval_depths[-1]):
                short_inp = torch.tensor([splits["calib_a"][0]["tokens"][:32]], device=device, dtype=torch.long)
                ext_logits, _, _ = model(short_inp, multi_segments=False, cache=None)
                ext_ok = bool(torch.isfinite(ext_logits).all().item())
        verify_info = {
            "bwd_rel_l2_err": bwd_verify["relative_l2_error"],
            "router_score_max_diff": rtr_diff,
            "router_idx_exact": idx_match,
            "extrap_depth_ok": ext_ok,
            "extrap_depth_tested": eval_depths[-1],
        }
        print(f"[vsubmoe:{model_key}] Verification: {json.dumps(verify_info)}")
        if rtr_diff > 1e-3 or not idx_match or not ext_ok:
            raise RuntimeError(f"Router or depth verification failed: {verify_info}")
        pv.step(1)

    for p in model.parameters():
        p.requires_grad_(False)

    # 1) Calibration: capture Split A & Split B W2 gradients -> Top-20% experts + Subspace SVD Proof!
    w2_base_cpu = {
        lid: model.layers[lid].moe.experts.weight2.detach().float().cpu().reshape(n_experts, model_dim, inter_dim)
        for lid in tracked_layers
    }
    with run_obj.phase(f"{model_key}_calib", total=2, unit="splits") as pc:
        with per_visit_w2_grad_collector(
            model, tracked_layers, train_loops, n_experts, model_dim, inter_dim, device
        ) as (grads_cpu, counts, accum_fn, _reset):
            _ = accum_fn(splits["calib_a"], clear_first=True)
            grads_a = {lid: t.clone() for lid, t in grads_cpu.items()}
            counts_a = {lid: c.copy() for lid, c in counts.items()}
            pc.step(1)

            _ = accum_fn(splits["calib_b"], clear_first=True)
            grads_b = {lid: t.clone() for lid, t in grads_cpu.items()}
            counts_b = {lid: c.copy() for lid, c in counts.items()}
            pc.step(2)

    expert_records = score_expert_conflicts(grads_a, grads_b, counts_a, counts_b, w2_base_cpu)
    groups = select_group_experts(expert_records, k_per_layer=k_per_layer, score_key="s_orth_both", seed=seed)
    top_experts = groups["conflict_top"]

    subspace_diag = analyze_subspace_concentration(
        grads_a, grads_b, top_experts, rank_per_loop=rank_per_loop, n_loops=train_loops
    )
    print(f"[vsubmoe:{model_key}] Subspace SVD concentration proof: {json.dumps(subspace_diag)}")
    del grads_a, grads_b, w2_base_cpu
    gc.collect()

    # 2) Build strictly parameter-matched condition suite (cap = shared_plus_scalar params!)
    lr = float(cfg.get("adapt_lr", 3e-4))
    grad_accum = int(cfg.get("adapt_grad_accum", 2))
    alpha = float(cfg.get("lora_alpha", 6.0))
    ld_p = float(cfg.get("loop_dropout_p", 0.5))
    vld_p = float(cfg.get("vsubmoe_ld_p", 0.35))

    ref_host = SubspaceMoEAdapterHost(
        top_experts, tracked_layers, mode="shared_plus_scalar",
        rank_per_loop=rank_per_loop, train_loops=train_loops,
        n_experts=n_experts, model_dim=model_dim, inter_dim=inter_dim, alpha=alpha,
    )
    param_cap = ref_host.num_trainable_params()
    del ref_host

    cond_builders: list[tuple[str, SubspaceMoEAdapterHost | None]] = [
        ("base_frozen", None),
        ("per_loop_lora", SubspaceMoEAdapterHost(
            top_experts, tracked_layers, mode="per_loop",
            rank_per_loop=rank_per_loop, train_loops=train_loops,
            n_experts=n_experts, model_dim=model_dim, inter_dim=inter_dim, alpha=alpha,
        )),
        ("pure_shared", SubspaceMoEAdapterHost(
            top_experts, tracked_layers, mode="pure_shared",
            rank_per_loop=rank_per_loop, train_loops=train_loops,
            n_experts=n_experts, model_dim=model_dim, inter_dim=inter_dim, alpha=alpha,
        )),
        ("loop_dropout", SubspaceMoEAdapterHost(
            top_experts, tracked_layers, mode="loop_dropout",
            rank_per_loop=rank_per_loop, train_loops=train_loops,
            n_experts=n_experts, model_dim=model_dim, inter_dim=inter_dim, alpha=alpha,
            loop_dropout_p=ld_p,
        )),
        ("shared_plus_scalar", SubspaceMoEAdapterHost(
            top_experts, tracked_layers, mode="shared_plus_scalar",
            rank_per_loop=rank_per_loop, train_loops=train_loops,
            n_experts=n_experts, model_dim=model_dim, inter_dim=inter_dim, alpha=alpha,
        )),
        ("shared_plus_diag_idx", SubspaceMoEAdapterHost(
            top_experts, tracked_layers, mode="shared_plus_diag_idx",
            rank_per_loop=rank_per_loop, train_loops=train_loops,
            n_experts=n_experts, model_dim=model_dim, inter_dim=inter_dim, alpha=alpha,
            target_param_cap=param_cap,
        )),
        ("velocity_submoe", SubspaceMoEAdapterHost(
            top_experts, tracked_layers, mode="velocity_submoe",
            rank_per_loop=rank_per_loop, train_loops=train_loops,
            n_experts=n_experts, model_dim=model_dim, inter_dim=inter_dim, alpha=alpha,
            target_param_cap=param_cap,
        )),
        ("velocity_submoe_ld", SubspaceMoEAdapterHost(
            top_experts, tracked_layers, mode="velocity_submoe_ld",
            rank_per_loop=rank_per_loop, train_loops=train_loops,
            n_experts=n_experts, model_dim=model_dim, inter_dim=inter_dim, alpha=alpha,
            loop_dropout_p=vld_p, target_param_cap=param_cap,
        )),
        ("velocity_submoe_full", SubspaceMoEAdapterHost(
            top_experts, tracked_layers, mode="velocity_submoe_full",
            rank_per_loop=rank_per_loop, train_loops=train_loops,
            n_experts=n_experts, model_dim=model_dim, inter_dim=inter_dim, alpha=alpha,
            loop_dropout_p=vld_p, use_router_adapter=True, target_param_cap=param_cap,
        )),
    ]

    cond_results: dict[str, dict[str, Any]] = {}
    d_in = str(train_loops)
    d_ext1 = str(eval_depths[2])  # 1.5x R_train (6 for Ouro, 12 for Huginn)
    d_ext2 = str(eval_depths[3])  # 2x R_train (8 for Ouro, 16 for Huginn)

    with run_obj.phase(f"{model_key}_exp", total=len(cond_builders), unit="conds") as pexp:
        for c_idx, (cname, host) in enumerate(cond_builders, 1):
            torch.manual_seed(seed)
            res = train_and_evaluate_vsubmoe_condition(
                model,
                tokenizer,
                cname,
                host,
                tracked_layers=tracked_layers,
                train_loops=train_loops,
                eval_depths=eval_depths,
                arc_depths=arc_depths,
                n_experts=n_experts,
                train_seqs=splits["adapt_train"],
                eval_seqs=splits["eval_heldout"],
                arc_samples=arc_samples,
                lr=lr,
                grad_accum=grad_accum,
                device=device,
                seed=seed,
            )
            cond_results[cname] = res
            base_in = cond_results["base_frozen"]["depth_nlls"][d_in]["nll"]
            base_ext1 = cond_results["base_frozen"]["depth_nlls"][d_ext1]["nll"]
            dn_in = base_in - res["depth_nlls"][d_in]["nll"]
            dn_ext1 = base_ext1 - res["depth_nlls"][d_ext1]["nll"]
            arc_in = res["arc_results"][d_in]["accuracy"]
            arc_e1 = res["arc_results"][d_ext1]["accuracy"]
            rs_in = res["router_stats"][d_in]
            print(
                f"[vsubmoe:{model_key}] {cname:21s} params={res['trainable_params']:9d} "
                f"R={d_in} ΔNLL={dn_in:+.5f} | R={d_ext1} ΔNLL={dn_ext1:+.5f} | "
                f"ARC({d_in}/{d_ext1})={arc_in*100:.1f}%/{arc_e1*100:.1f}% | "
                f"load_cos={rs_in['cross_loop_load_cosine']:.3f} flip={rs_in['consecutive_topk_flip_rate']*100:.1f}%"
            )
            pexp.row(
                f"{model_key}_conds",
                condition=cname,
                delta_nll_in=dn_in,
                delta_nll_ext1=dn_ext1,
                arc_in=arc_in,
                params=res["trainable_params"],
            )
            pexp.step(c_idx)

    # Compute paired bootstrap contrasts at R_train, 1.5x R_train, and 2x R_train!
    def _nll_contrast(cand: str, ref: str, depth_key: str) -> dict[str, Any]:
        a = np.asarray(cond_results[ref]["depth_nlls"][depth_key]["per_seq"], dtype=np.float64)
        b = np.asarray(cond_results[cand]["depth_nlls"][depth_key]["per_seq"], dtype=np.float64)
        mean_d, lo, hi = paired_bootstrap_ci(a - b)  # positive = cand has LOWER NLL than ref
        return {"mean_adv": mean_d, "ci95": [lo, hi], "wins": int((a > b).sum()), "n": int(a.size)}

    contrasts: dict[str, dict[str, Any]] = {}
    for d_key in [str(d) for d in eval_depths]:
        contrasts[f"R{d_key}"] = {
            "vsub_vs_pure_shared": _nll_contrast("velocity_submoe", "pure_shared", d_key),
            "vsub_vs_loop_dropout": _nll_contrast("velocity_submoe", "loop_dropout", d_key),
            "vsub_vs_shared_scalar": _nll_contrast("velocity_submoe", "shared_plus_scalar", d_key),
            "vsub_vs_diag_idx": _nll_contrast("velocity_submoe", "shared_plus_diag_idx", d_key),
            "vsub_ld_vs_loop_dropout": _nll_contrast("velocity_submoe_ld", "loop_dropout", d_key),
            "vsub_full_vs_pure_shared": _nll_contrast("velocity_submoe_full", "pure_shared", d_key),
            "vsub_full_vs_loop_dropout": _nll_contrast("velocity_submoe_full", "loop_dropout", d_key),
            "vsub_full_vs_shared_scalar": _nll_contrast("velocity_submoe_full", "shared_plus_scalar", d_key),
            "vsub_full_vs_diag_idx": _nll_contrast("velocity_submoe_full", "shared_plus_diag_idx", d_key),
            "vsub_full_vs_per_loop": _nll_contrast("velocity_submoe_full", "per_loop_lora", d_key),
            "shared_scalar_vs_pure_shared": _nll_contrast("shared_plus_scalar", "pure_shared", d_key),
            "diag_idx_vs_shared_scalar": _nll_contrast("shared_plus_diag_idx", "shared_plus_scalar", d_key),
        }

    # Strip per-seq arrays before saving JSON
    clean_conds: dict[str, Any] = {}
    for cname, res in cond_results.items():
        clean_conds[cname] = {
            "condition": res["condition"],
            "trainable_params": res["trainable_params"],
            "train_loss_first8": res["train_loss_first8"],
            "train_loss_last8": res["train_loss_last8"],
            "depth_nlls": {
                dk: {k: v for k, v in dv.items() if k != "per_seq"} for dk, dv in res["depth_nlls"].items()
            },
            "single_loop_nlls": res["single_loop_nlls"],
            "arc_results": {dk: {"accuracy": dv["accuracy"]} for dk, dv in res["arc_results"].items()},
            "router_stats": res["router_stats"],
        }

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model_key": model_key,
        "artifact_dir": str(artifact_dir),
        "tracked_layers": tracked_layers,
        "train_loops": train_loops,
        "eval_depths": eval_depths,
        "arc_depths": arc_depths,
        "param_cap": param_cap,
        "verify": verify_info,
        "subspace_diag": subspace_diag,
        "conditions": clean_conds,
        "contrasts": contrasts,
    }


def render_report_md(summary: dict[str, Any]) -> str:
    lines = ["# Direction 2 Feasibility Report: Velocity-Modulated Shared-Subspace Looped MoE (V-SubMoE)", ""]
    lines.append(f"- **Elapsed**: `{summary['elapsed_sec']:.1f}s` on GPU `{summary['gpu_index']}` ({summary['gpu_name']})")
    lines.append("")

    for mkey, md in summary["models"].items():
        sd = md["subspace_diag"]
        d_list = [str(d) for d in md["eval_depths"]]
        d_in = str(md["train_loops"])
        d_ext1 = d_list[2]
        d_ext2 = d_list[3]
        lines.append(f"## Model: `{mkey}` (R_train={d_in}, depths={d_list}, param_cap={md['param_cap']:,})")
        lines.append(
            f"- **Exactness**: bwd_err=`{md['verify']['bwd_rel_l2_err']:.2e}`, "
            f"router_diff=`{md['verify']['router_score_max_diff']:.2e}`, "
            f"router_idx_exact=`{md['verify']['router_idx_exact']}`, extrap_ok=`{md['verify']['extrap_depth_ok']}`"
        )
        lines.append(
            f"- **Subspace Concentration Proof (Split A -> Split B, n={sd['n_experts_analyzed']} top experts)**: "
            f"Per-loop rank-{sd['rank_per_loop']} split EV = `{sd['ev_per_loop_split_mean']*100:.2f}%` vs "
            f"Shared rank-{sd['rank_shared']} full subspace EV = `{sd['ev_shared_subspace_full']*100:.2f}%` "
            f"(**{sd['shared_over_split_ratio']:.2f}x**), diagonal-core EV = `{sd['ev_shared_subspace_diag']*100:.2f}%`, "
            f"no-mod mean EV = `{sd['ev_shared_mean_nomod']*100:.2f}%`"
        )
        lines.append("")
        lines.append("### Parameter-Matched Held-Out NLL & Zero-Shot Depth Extrapolation")
        hdr = "| Condition | Params | " + " | ".join(f"R={d} NLL (Δ)" for d in d_list) + f" | ARC R={d_in} | ARC R={d_ext1} | LoadCos | Flip% |"
        sep = "|---|---:|" + "---:|" * len(d_list) + "---:|---:|---:|---:|"
        lines.extend([hdr, sep])
        base_d = {d: md["conditions"]["base_frozen"]["depth_nlls"][d]["nll"] for d in d_list}
        for cname, c in md["conditions"].items():
            d_cols = []
            for d in d_list:
                v = c["depth_nlls"][d]["nll"]
                dv = base_d[d] - v
                d_cols.append(f"{v:.4f} ({dv:+.4f})")
            a_in = c["arc_results"][d_in]["accuracy"] * 100.0
            a_e1 = c["arc_results"][d_ext1]["accuracy"] * 100.0
            rs = c["router_stats"][d_in]
            lines.append(
                f"| `{cname}` | {c['trainable_params']:,} | "
                + " | ".join(d_cols)
                + f" | {a_in:.1f}% | {a_e1:.1f}% | {rs['cross_loop_load_cosine']:.3f} | {rs['consecutive_topk_flip_rate']*100:.1f}% |"
            )
        lines.append("")

        lines.append("### Single-Loop Activation Diagnostic (NLL when adapter enabled at 1 loop only)")
        lines.append("| Condition | " + " | ".join(f"Only L{r+1}" for r in range(md["train_loops"])) + " | Early-Late Gap |")
        lines.append("|---|" + "---:|" * (md["train_loops"] + 1))
        for cname, c in md["conditions"].items():
            sl = c["single_loop_nlls"]
            if not sl:
                continue
            gap = sl[0] - sl[-1]  # positive means loop 1 worse than final loop (late-loop bias)
            lines.append(f"| `{cname}` | " + " | ".join(f"{x:.4f}" for x in sl) + f" | {gap:+.4f} |")
        lines.append("")

        lines.append("### Key Paired Contrasts (positive = first wins, lower NLL)")
        for d_key in [f"R{d_in}", f"R{d_ext1}", f"R{d_ext2}"]:
            lines.append(f"**Depth {d_key}**:")
            for ck, cv in md["contrasts"][d_key].items():
                lines.append(
                    f"- `{ck}`: `{cv['mean_adv']:+.5f}` CI `[{cv['ci95'][0]:+.5f}, {cv['ci95'][1]:+.5f}]` ({cv['wins']}/{cv['n']})"
                )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--xllm-loop-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    seed = int(cfg.get("seed", 20261124))

    gpu_idx, gpu_lock, gpu_info = acquire_idle_gpu(cfg.get("preferred_gpu"))
    try:
        device = torch.device(f"cuda:{gpu_idx}")
        torch.cuda.set_device(device)
        print(f"[vsubmoe] Acquired idle GPU {gpu_idx}: {gpu_info.name} (mem={gpu_info.mem_used_mib}MiB, util={gpu_info.util_pct}%)")

        boot_info = bootstrap_xllm(args.xllm_loop_dir, attention_backend="sdpa")
        print(f"[vsubmoe] xllm bootstrap: {json.dumps(boot_info)}")

        arc_samples = load_arc_challenge_samples(args.dataset_dir, n_samples=int(cfg.get("n_arc_eval", 100)), seed=seed)

        models_cfg: dict[str, dict[str, Any]] = cfg["models"]
        plan_stages: list[dict[str, Any]] = [
            {"phase": "setup", "of": 1, "unit": "env"},
            {"phase": "download", "of": 1, "unit": "assets"},
        ]
        for mkey in models_cfg:
            plan_stages.extend(
                [
                    {"phase": f"{mkey}_verify", "of": 1, "unit": "check"},
                    {"phase": f"{mkey}_calib", "of": 2, "unit": "splits"},
                    {"phase": f"{mkey}_exp", "of": 9, "unit": "conds"},
                ]
            )
        plan_stages.append({"phase": "report", "of": 1, "unit": "file"})
        orx.plan(plan_stages)

        run_obj = orx.progress()
        model_outputs: dict[str, Any] = {}
        for mkey, mspec in models_cfg.items():
            model_outputs[mkey] = run_model_vsubmoe(
                mkey,
                mspec,
                model_dir=Path(args.model_dir),
                dataset_dir=Path(args.dataset_dir),
                arc_samples=arc_samples,
                cfg=cfg,
                device=device,
                run_obj=run_obj,
            )

        with run_obj.phase("report", total=1, unit="file") as prep:
            summary = {
                "idea": "new_dir2_vsubmoe",
                "gpu_index": gpu_idx,
                "gpu_name": gpu_info.name,
                "elapsed_sec": time.time() - t0,
                "config": cfg,
                "models": model_outputs,
            }
            (out_dir / "summary_vsubmoe.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
            report_md = render_report_md(summary)
            (out_dir / "report_vsubmoe.md").write_text(report_md)

            metrics: dict[str, float] = {}
            for mk, md in model_outputs.items():
                d_in = f"R{md['train_loops']}"
                d_ext1 = f"R{md['eval_depths'][2]}"
                metrics[f"{mk}_subspace_ratio"] = md["subspace_diag"]["shared_over_split_ratio"]
                metrics[f"{mk}_vsub_vs_pure_in"] = md["contrasts"][d_in]["vsub_vs_pure_shared"]["mean_adv"]
                metrics[f"{mk}_vsub_vs_ld_in"] = md["contrasts"][d_in]["vsub_vs_loop_dropout"]["mean_adv"]
                metrics[f"{mk}_vsub_vs_scalar_in"] = md["contrasts"][d_in]["vsub_vs_shared_scalar"]["mean_adv"]
                metrics[f"{mk}_vfull_vs_pure_in"] = md["contrasts"][d_in]["vsub_full_vs_pure_shared"]["mean_adv"]
                metrics[f"{mk}_vfull_vs_ld_in"] = md["contrasts"][d_in]["vsub_full_vs_loop_dropout"]["mean_adv"]
                metrics[f"{mk}_vfull_vs_scalar_in"] = md["contrasts"][d_in]["vsub_full_vs_shared_scalar"]["mean_adv"]
                metrics[f"{mk}_vfull_vs_scalar_ext1"] = md["contrasts"][d_ext1]["vsub_full_vs_shared_scalar"]["mean_adv"]
                metrics[f"{mk}_vfull_vs_diag_ext1"] = md["contrasts"][d_ext1]["vsub_full_vs_diag_idx"]["mean_adv"]
            prep.step(1, **metrics)

            print("========== REPORT (report_vsubmoe.md) ==========")
            print(report_md)
            print("========== END REPORT ==========")
    finally:
        gpu_lock.release()


if __name__ == "__main__":
    main()
