"""Round 1 Probe 1: Looped MoE Low-Bit Quantization, Routing-Flip Cascade,
Cross-Loop Calibration Blindness & Feedback-Aware Mixed-Precision Allocation.

Evaluates both IFM Ouro-MoE (28 layers x 4 loops) and IFM Huginn-MoE (8+12x8+8):
  0. Exactness verification (no-op quant hook matches native BF16 bit-for-bit).
  1. Calibration pass (16 disjoint seqs): per-(layer, visit, expert) activation counts,
     gate energy, output L2 energy, and per-visit input RMS vectors to quantify
     cross-loop calibration blindness (cosine & channel-scale drift across loops).
  2. Bit-width & component sensitivity map:
     - BF16 reference (NLL + ARC-Challenge 200q + cached oracle routing traces)
     - Experts-only RTN: W4 g128, W3 g128, W2 g128, W2 g64
     - All-linear (experts + attention) W3 g128 & W4+router W4
     - Huginn-specific feedback-exposure contrast:
       protect recurrent (12L W4, 16L W2; 2.86-bit) vs protect non-recurrent (16L W4, 12L W2; 3.14-bit)
  3. 2x2 Counterfactual Routing-Flip Cascade Decomposition (at W3 g128, W2 g128, W2 g64):
     - (Quant Weights, Quant Routing)
     - (Quant Weights, BF16 Oracle Routing)
     - (BF16 Weights, Quant Routing)
     - Per-loop routing flip trajectory flip_top1(r), flip_any(r) across r=0..R-1.
  4. Iso-Bit Mixed-Precision Expert Allocation:
     - 2.40-bit budget (20% W4 + 80% W2, g128): random vs global_freq vs global_energy
       vs max_stage_freq vs early_feedback vs layer_balanced_feedback.
     - 3.00-bit budget (50% W4 + 50% W2, g128 vs uniform W3 g128): global_freq vs early_feedback.
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
import torch.nn.functional as F

from loop_research import orx_run as orx
from loop_research.data_assets import build_disjoint_text_code_splits, load_arc_challenge_samples
from loop_research.gpu_utils import acquire_idle_gpu
from loop_research.ifm_runtime import bootstrap_xllm, load_ifm_model_for_autograd


# ---------------------------------------------------------------------------
# Pure-PyTorch Asymmetric Group-Wise RTN Fake Quantization (CPU Testable!)
# ---------------------------------------------------------------------------

def fake_quantize_asym_group(
    w: torch.Tensor,
    bits: int,
    group_size: int = 128,
) -> torch.Tensor:
    """Asymmetric group-wise round-to-nearest fake quantization along the last dim.

    Args:
        w: Tensor of shape [..., in_features].
        bits: Bit-width in {2, 3, 4, 8} (bits >= 16 returns a clone unchanged).
        group_size: Number of input elements per quantization group (<=0 means per-row).
    """
    if bits >= 16:
        return w.clone()
    orig_dtype = w.dtype
    orig_shape = w.shape
    in_dim = orig_shape[-1]
    wf = w.detach().float().reshape(-1, in_dim)

    gs = in_dim if (group_size <= 0 or group_size >= in_dim) else int(group_size)
    pad = (gs - (in_dim % gs)) % gs
    if pad > 0:
        wf = F.pad(wf, (0, pad), value=0.0)

    g = wf.reshape(wf.shape[0], -1, gs)
    qmax = float((1 << bits) - 1)
    w_min = g.amin(dim=-1, keepdim=True)
    w_max = g.amax(dim=-1, keepdim=True)
    scale = (w_max - w_min) / qmax
    safe_scale = torch.where(scale > 1e-12, scale, torch.ones_like(scale))
    zp = torch.round(-w_min / safe_scale).clamp(0.0, qmax)
    q = torch.round(g / safe_scale + zp).clamp(0.0, qmax)
    deq = torch.where(scale > 1e-12, (q - zp) * safe_scale, w_min)
    out = deq.reshape(wf.shape[0], -1)
    if pad > 0:
        out = out[:, :in_dim]
    return out.reshape(orig_shape).to(orig_dtype)


def select_topk_experts_mask(
    scores: np.ndarray,
    total_k: int,
    *,
    per_layer_equal: bool = False,
) -> np.ndarray:
    """Select boolean mask [L, E] of exactly total_k experts (deterministic tie-break)."""
    n_layers, n_experts = scores.shape
    mask = np.zeros((n_layers, n_experts), dtype=bool)
    if total_k <= 0:
        return mask
    if per_layer_equal:
        base_k = total_k // n_layers
        rem = total_k % n_layers
        layer_totals = scores.sum(axis=1)
        extra_layers = set(
            sorted(range(n_layers), key=lambda l: (-float(layer_totals[l]), l))[:rem]
        )
        for l in range(n_layers):
            kl = min(n_experts, base_k + (1 if l in extra_layers else 0))
            if kl <= 0:
                continue
            order = sorted(range(n_experts), key=lambda e: (-float(scores[l, e]), e))
            for e in order[:kl]:
                mask[l, e] = True
        return mask

    flat = [(-float(scores[l, e]), l, e) for l in range(n_layers) for e in range(n_experts)]
    flat.sort()
    for _, l, e in flat[: min(total_k, len(flat))]:
        mask[l, e] = True
    return mask


def compute_loop_calibration_shift(
    rms_by_visit: np.ndarray,
    counts_by_visit: np.ndarray,
    min_tokens: int = 16,
) -> dict[str, float]:
    """Quantify cross-loop input activation distribution shift within each physical expert.

    Args:
        rms_by_visit: [L, R, E, D] per-channel RMS input activation for each (layer, visit, expert).
        counts_by_visit: [L, R, E] token counts.
    """
    n_layers, max_v, n_experts, _dim = rms_by_visit.shape
    adj_cos: list[float] = []
    first_last_cos: list[float] = []
    max_channel_ratio: list[float] = []
    stage_only_experts = 0
    active_recurrent_experts = 0
    loop0_blind_experts = 0

    for l in range(n_layers):
        for e in range(n_experts):
            valid_vs = [r for r in range(max_v) if counts_by_visit[l, r, e] >= min_tokens]
            if len(valid_vs) == 0:
                continue
            # Check if this layer is multi-visit (at least 2 visits have any tokens globally)
            layer_visits = [r for r in range(max_v) if counts_by_visit[l, r].sum() > 0]
            if len(layer_visits) < 2:
                continue
            active_recurrent_experts += 1
            if counts_by_visit[l, 0, e] < min_tokens:
                loop0_blind_experts += 1
            if len(valid_vs) == 1:
                stage_only_experts += 1
                continue
            vecs = rms_by_visit[l, valid_vs, e]  # [V, D]
            norms = np.linalg.norm(vecs, axis=-1, keepdims=True) + 1e-12
            unit = vecs / norms
            for i in range(len(valid_vs) - 1):
                adj_cos.append(float(np.dot(unit[i], unit[i + 1])))
            first_last_cos.append(float(np.dot(unit[0], unit[-1])))
            # Peak-to-trough ratio across visits for top-10% salient channels
            mean_vec = vecs.mean(axis=0)
            top_ch = np.argsort(mean_vec)[-(max(1, len(mean_vec) // 10)) :]
            sub = vecs[:, top_ch]
            ratios = sub.max(axis=0) / np.maximum(sub.min(axis=0), 1e-6)
            max_channel_ratio.append(float(np.median(ratios)))

    return {
        "active_recurrent_experts": float(active_recurrent_experts),
        "loop0_blind_frac": float(loop0_blind_experts / max(active_recurrent_experts, 1)),
        "single_stage_only_frac": float(stage_only_experts / max(active_recurrent_experts, 1)),
        "mean_adj_visit_rms_cos": float(np.mean(adj_cos)) if adj_cos else 1.0,
        "mean_first_last_rms_cos": float(np.mean(first_last_cos)) if first_last_cos else 1.0,
        "median_top10pct_ch_scale_ratio": float(np.median(max_channel_ratio)) if max_channel_ratio else 1.0,
    }


def paired_bootstrap_ci(
    diffs: np.ndarray,
    n_boot: int = 2000,
    seed: int = 20261126,
) -> tuple[float, float, float]:
    arr = np.asarray(diffs, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    return (
        float(arr.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


# ---------------------------------------------------------------------------
# IFM Model Weight Backup & In-Place Fake Quantization Manager
# ---------------------------------------------------------------------------

class IFMQuantManager:
    """Stores BF16 CPU backups of expert & attention/router weights for fast exact restore."""

    def __init__(self, model) -> None:
        self.model = model
        self.n_layers = len(model.layers)
        self.n_experts = int(model.layers[0].moe.experts.n_local_experts)
        self.model_dim = int(model.layers[0].moe.experts.model_dim)
        self.inter_dim = int(model.layers[0].moe.experts.inter_dim)
        self.recurrent_layers = set(range(int(model.loop_start_layer), int(model.loop_end_layers)))
        self.n_loops = int(model.loop_times)

        self.expert_backups: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.other_backups: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
        self.router_backups: list[tuple[torch.nn.Parameter, torch.Tensor]] = []

        for layer in model.layers:
            em = layer.moe.experts
            self.expert_backups.append(
                (
                    em.weight1.detach().cpu().clone(),
                    em.weight2.detach().cpu().clone(),
                    em.weight3.detach().cpu().clone(),
                )
            )
            rtr = layer.moe.router
            self.router_backups.append((rtr.weight, rtr.weight.detach().cpu().clone()))
            # Collect 2D linear weights inside attention (and shared experts if any)
            for name, p in layer.named_parameters():
                if "moe.experts" in name or "moe.router" in name:
                    continue
                if p.ndim == 2 and min(p.shape) >= 64:
                    self.other_backups.append((p, p.detach().cpu().clone()))

    @torch.no_grad()
    def restore_all(self) -> None:
        for lid, layer in enumerate(self.model.layers):
            em = layer.moe.experts
            w1, w2, w3 = self.expert_backups[lid]
            em.weight1.copy_(w1, non_blocking=False)
            em.weight2.copy_(w2, non_blocking=False)
            em.weight3.copy_(w3, non_blocking=False)
        for p, bak in self.other_backups:
            p.copy_(bak, non_blocking=False)
        for p, bak in self.router_backups:
            p.copy_(bak, non_blocking=False)

    @torch.no_grad()
    def apply_expert_bits(
        self,
        bits_matrix: np.ndarray,
        group_size: int = 128,
        *,
        quant_attn_bits: int = 16,
        quant_router_bits: int = 16,
    ) -> float:
        """Apply per-(layer, expert) bit-width matrix [L, E] in-place. Returns avg expert bits."""
        self.restore_all()
        E = self.n_experts
        D = self.model_dim
        I = self.inter_dim

        for lid, layer in enumerate(self.model.layers):
            em = layer.moe.experts
            layer_bits = bits_matrix[lid]
            uniq = sorted(set(int(b) for b in layer_bits))
            if uniq == [16]:
                continue
            w1_view = em.weight1.view(E, I, D)
            w2_view = em.weight2.view(E, D, I)
            w3_view = em.weight3.view(E, I, D)
            for b in uniq:
                if b >= 16:
                    continue
                e_idxs = [e for e in range(E) if int(layer_bits[e]) == b]
                idx_t = torch.tensor(e_idxs, device=em.weight1.device, dtype=torch.long)
                w1_view[idx_t] = fake_quantize_asym_group(w1_view[idx_t], b, group_size)
                w2_view[idx_t] = fake_quantize_asym_group(w2_view[idx_t], b, group_size)
                w3_view[idx_t] = fake_quantize_asym_group(w3_view[idx_t], b, group_size)

        if quant_attn_bits < 16:
            for p, _ in self.other_backups:
                p.copy_(fake_quantize_asym_group(p, quant_attn_bits, group_size))

        if quant_router_bits < 16:
            for p, _ in self.router_backups:
                p.copy_(fake_quantize_asym_group(p, quant_router_bits, group_size))

        return float(np.mean(bits_matrix))


# ---------------------------------------------------------------------------
# Routing Trace Recorder / Replayer + Calibration Activation Collector
# ---------------------------------------------------------------------------

@torch.inference_mode()
def collect_calibration_stats(
    model,
    calib_seqs: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    """Collect per-(layer, visit, expert) counts, gate sums, output energy, and input RMS."""
    n_layers = len(model.layers)
    n_experts = int(model.layers[0].moe.experts.n_local_experts)
    model_dim = int(model.layers[0].moe.experts.model_dim)
    max_visits = int(model.loop_times)
    recurrent_layers = set(range(int(model.loop_start_layer), int(model.loop_end_layers)))

    counts = np.zeros((n_layers, max_visits, n_experts), dtype=np.int64)
    gate_sum = np.zeros((n_layers, max_visits, n_experts), dtype=np.float64)
    out_energy = np.zeros((n_layers, max_visits, n_experts), dtype=np.float64)
    in_sq_sum = np.zeros((n_layers, max_visits, n_experts, model_dim), dtype=np.float64)

    visits = [0] * n_layers
    last_routed: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    orig_router_fwds = []
    orig_exp_fwds = []

    try:
        for lid, layer in enumerate(model.layers):
            router = layer.moe.router
            exp_mod = layer.moe.experts
            orig_router_fwds.append((router, router.forward))
            orig_exp_fwds.append((exp_mod, exp_mod.forward))

            def make_rtr(lid_c: int, orig_fn):
                def fwd(x, load_balancing_type=None):
                    weights, idx, tpe, aux = orig_fn(x, load_balancing_type)
                    v = visits[lid_c]
                    if v < max_visits:
                        last_routed[lid_c] = (x.detach(), idx.detach(), weights.detach())
                    return weights, idx, tpe, aux
                return fwd

            def make_exp(lid_c: int, orig_fn):
                def fwd(x, group_sizes):
                    y = orig_fn(x, group_sizes)
                    v = visits[lid_c]
                    if v < max_visits and lid_c in last_routed:
                        x_in, idx, weights = last_routed[lid_c]
                        # group_sizes may be a Python list[int] or Tensor; y is grouped output [sum(group_sizes), D]
                        gs_cpu = (
                            group_sizes.detach().cpu().tolist()
                            if torch.is_tensor(group_sizes)
                            else [int(g) for g in group_sizes]
                        )
                        # Accumulate counts & gate weights
                        flat_idx = idx.reshape(-1)
                        flat_w = weights.float().reshape(-1)
                        x_flat = x_in.reshape(-1, model_dim).float()
                        # Output energy per expert from grouped y
                        y_sq_row = y.float().pow(2).sum(dim=-1)  # [N_tot]
                        offset = 0
                        for e in range(n_experts):
                            cnt = int(gs_cpu[e])
                            if cnt > 0:
                                counts[lid_c, v, e] += cnt
                                out_energy[lid_c, v, e] += float(y_sq_row[offset : offset + cnt].sum().item())
                                offset += cnt
                        # Gate sum & input squared activation per expert
                        idx_cpu = flat_idx.cpu().numpy()
                        w_cpu = flat_w.cpu().numpy()
                        np.add.at(gate_sum[lid_c, v], idx_cpu, w_cpu)
                        # Accumulate per-channel input squared sum (indexed by token_idx = pos // topk)
                        topk = idx.shape[-1]
                        tok_ids = torch.arange(x_flat.shape[0], device=x_flat.device).repeat_interleave(topk)
                        for e in range(n_experts):
                            if gs_cpu[e] > 0:
                                sel_tok = tok_ids[flat_idx == e]
                                if sel_tok.numel() > 0:
                                    sq = x_flat.index_select(0, sel_tok).pow(2).sum(dim=0).cpu().numpy()
                                    in_sq_sum[lid_c, v, e] += sq
                    visits[lid_c] += 1
                    return y
                return fwd

            router.forward = make_rtr(lid, router.forward)
            exp_mod.forward = make_exp(lid, exp_mod.forward)

        for item in calib_seqs:
            for l in range(n_layers):
                visits[l] = 0
            last_routed.clear()
            inp = torch.tensor([item["tokens"][:-1]], device=device, dtype=torch.long)
            _ = model(inp, multi_segments=False, cache=None)
    finally:
        for router, fn in orig_router_fwds:
            router.forward = fn
        for exp_mod, fn in orig_exp_fwds:
            exp_mod.forward = fn

    denom = np.maximum(counts[..., None], 1).astype(np.float64)
    in_rms = np.sqrt(in_sq_sum / denom)
    shift_stats = compute_loop_calibration_shift(in_rms, counts, min_tokens=16)

    # Compute ranking matrices [L, E] for mixed-precision comparison
    global_freq = counts.sum(axis=1).astype(np.float64)
    global_energy = out_energy.sum(axis=1).astype(np.float64)

    # Stage-normalized max probability across visits: protects entry/exit specialists!
    visit_totals = np.maximum(counts.sum(axis=2, keepdims=True), 1).astype(np.float64)
    stage_prob = counts / visit_totals
    max_stage_freq = stage_prob.max(axis=1)

    # Feedback-exposure weighted energy: earlier visits cascade through MORE remaining loops!
    fb_weights = np.zeros((n_layers, max_visits), dtype=np.float64)
    for l in range(n_layers):
        if l in recurrent_layers:
            for r in range(max_visits):
                fb_weights[l, r] = float(max_visits - r)
        elif l < int(model.loop_start_layer):
            # Prelude feeds into ALL R loops
            fb_weights[l, 0] = float(max_visits + 1)
        else:
            # Coda has zero recurrent feedback
            fb_weights[l, 0] = 1.0
    early_feedback = (out_energy * fb_weights[:, :, None]).sum(axis=1)

    return {
        "counts": counts,
        "gate_sum": gate_sum,
        "out_energy": out_energy,
        "global_freq": global_freq,
        "global_energy": global_energy,
        "max_stage_freq": max_stage_freq,
        "early_feedback": early_feedback,
        "calib_shift": shift_stats,
    }


@contextmanager
def router_trace_context(
    model,
    *,
    mode: str = "record",  # "record" | "compare" | "replay"
    ref_traces: list[dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]]] | None = None,
):
    """Record, compare against reference, or force-replay routing decisions per (seq, layer, visit)."""
    n_layers = len(model.layers)
    max_visits = int(model.loop_times)
    recurrent_layers = set(range(int(model.loop_start_layer), int(model.loop_end_layers)))

    state: dict[str, Any] = {
        "seq_idx": 0,
        "visits": [0] * n_layers,
        "recorded": [],
        "current_rec": {},
        "flip_top1_num": np.zeros(max_visits, dtype=np.int64),
        "flip_any_num": np.zeros(max_visits, dtype=np.int64),
        "flip_den": np.zeros(max_visits, dtype=np.int64),
        "flip_nonrec_top1": 0,
        "flip_nonrec_any": 0,
        "flip_nonrec_den": 0,
    }

    def start_seq(seq_idx: int) -> None:
        state["seq_idx"] = seq_idx
        state["visits"] = [0] * n_layers
        if mode == "record":
            state["current_rec"] = {}
            state["recorded"].append(state["current_rec"])

    orig_fwds = []
    try:
        for lid, layer in enumerate(model.layers):
            router = layer.moe.router
            orig_fwds.append((router, router.forward))

            def make_rtr(lid_c: int, mod, orig_fn):
                def fwd(x, load_balancing_type=None):
                    v = state["visits"][lid_c]
                    state["visits"][lid_c] = v + 1

                    if mode == "replay":
                        assert ref_traces is not None
                        idx_cpu, w_cpu = ref_traces[state["seq_idx"]][(lid_c, v)]
                        idx = idx_cpu.to(device=x.device, dtype=torch.long)
                        weights = w_cpu.to(device=x.device, dtype=x.dtype)
                        tpe = torch.bincount(idx.flatten(), minlength=mod.num_experts)
                        return weights, idx, tpe, None

                    weights, idx, tpe, aux = orig_fn(x, load_balancing_type)

                    if mode == "record":
                        state["current_rec"][(lid_c, v)] = (
                            idx.detach().to(torch.int16).cpu(),
                            weights.detach().to(torch.bfloat16).cpu(),
                        )
                    elif mode == "compare" and ref_traces is not None:
                        ref_idx_cpu, _ = ref_traces[state["seq_idx"]][(lid_c, v)]
                        ref_idx = ref_idx_cpu.to(device=idx.device, dtype=idx.dtype)
                        # Top-1 flip and unordered top-2 set difference
                        top1_flip = int((idx[..., 0] != ref_idx[..., 0]).sum().item())
                        # Match as unordered sets of size 2
                        same_0 = (idx[..., 0] == ref_idx[..., 0]) | (idx[..., 0] == ref_idx[..., 1])
                        same_1 = (idx[..., 1] == ref_idx[..., 0]) | (idx[..., 1] == ref_idx[..., 1])
                        any_flip = int((~(same_0 & same_1)).sum().item())
                        n_tok = int(idx.numel() // idx.shape[-1])
                        if lid_c in recurrent_layers and v < max_visits:
                            state["flip_top1_num"][v] += top1_flip
                            state["flip_any_num"][v] += any_flip
                            state["flip_den"][v] += n_tok
                        else:
                            state["flip_nonrec_top1"] += top1_flip
                            state["flip_nonrec_any"] += any_flip
                            state["flip_nonrec_den"] += n_tok
                    return weights, idx, tpe, aux
                return fwd

            router.forward = make_rtr(lid, router, router.forward)
        yield state, start_seq
    finally:
        for router, fn in orig_fwds:
            router.forward = fn


@torch.inference_mode()
def eval_nll_with_trace(
    model,
    seqs: list[dict[str, Any]],
    device: torch.device,
    *,
    mode: str = "compare",
    ref_traces=None,
) -> dict[str, Any]:
    per_seq: list[float] = []
    txt: list[float] = []
    cod: list[float] = []
    with router_trace_context(model, mode=mode, ref_traces=ref_traces) as (st, start_seq):
        for i, item in enumerate(seqs):
            start_seq(i)
            toks = item["tokens"]
            inp = torch.tensor([toks[:-1]], device=device, dtype=torch.long)
            tgt = torch.tensor(toks[1:], device=device, dtype=torch.long)
            logits, _, _ = model(inp, multi_segments=False, cache=None)
            loss = float(F.cross_entropy(logits[0].float(), tgt).item())
            per_seq.append(loss)
            if item["domain"] == "code":
                cod.append(loss)
            else:
                txt.append(loss)

    flip_top1_by_loop = [
        float(st["flip_top1_num"][r] / max(st["flip_den"][r], 1))
        for r in range(len(st["flip_den"]))
    ]
    flip_any_by_loop = [
        float(st["flip_any_num"][r] / max(st["flip_den"][r], 1))
        for r in range(len(st["flip_den"]))
    ]
    overall_rec_top1 = float(st["flip_top1_num"].sum() / max(st["flip_den"].sum(), 1))
    overall_rec_any = float(st["flip_any_num"].sum() / max(st["flip_den"].sum(), 1))
    return {
        "nll": float(np.mean(per_seq)),
        "nll_text": float(np.mean(txt)) if txt else 0.0,
        "nll_code": float(np.mean(cod)) if cod else 0.0,
        "per_seq": per_seq,
        "recorded_traces": st["recorded"] if mode == "record" else None,
        "flip_top1_by_loop": flip_top1_by_loop,
        "flip_any_by_loop": flip_any_by_loop,
        "rec_flip_top1": overall_rec_top1,
        "rec_flip_any": overall_rec_any,
        "nonrec_flip_any": float(st["flip_nonrec_any"] / max(st["flip_nonrec_den"], 1)),
    }


@torch.inference_mode()
def eval_arc_ifm(
    model,
    tokenizer,
    arc_samples: list[dict[str, Any]],
    device: torch.device,
) -> tuple[float, list[int]]:
    flags: list[int] = []
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
            logits, _, _ = model(inp, multi_segments=False, cache=None)
            sl_logits = logits[0, len(p_ids) - 1 :, :].float()
            lp = F.log_softmax(sl_logits, dim=-1)
            tok_ll = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            scores.append(float(tok_ll.mean().item()))
        flags.append(1 if int(np.argmax(scores)) == item["gold_idx"] else 0)
    return float(np.mean(flags)), flags


# ---------------------------------------------------------------------------
# Full Per-Model Probe 1 Execution
# ---------------------------------------------------------------------------

def run_model_probe1(
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
    artifact_dir = model_dir / mspec["local_dir"]
    seed = int(cfg.get("seed", 20261126))
    print(f"\n[probe1:{model_key}] Loading {artifact_dir} ...", flush=True)
    model, tokenizer, _mcfg = load_ifm_model_for_autograd(
        artifact_dir, device=str(device), attention_backend="sdpa"
    )
    qm = IFMQuantManager(model)
    n_layers, n_experts = qm.n_layers, qm.n_experts
    total_experts = n_layers * n_experts

    splits = build_disjoint_text_code_splits(
        dataset_dir,
        lambda t: tokenizer.encode(t, bos=True, eos=False),
        seq_len=int(cfg.get("seq_len", 256)),
        n_calib_a=int(cfg.get("n_calib", 16)),
        n_calib_b=4,
        n_adapt_train=4,
        n_eval_heldout=int(cfg.get("n_eval", 32)),
        seed=seed,
    )
    calib_seqs = splits["calib_a"]
    eval_seqs = splits["eval_heldout"]

    # 1) Exactness verification
    with run_obj.phase(f"{model_key}_verify", total=1, unit="check") as pv:
        test_inp = torch.tensor([eval_seqs[0]["tokens"][:64]], device=device, dtype=torch.long)
        with torch.inference_mode():
            ref_logits = model(test_inp, multi_segments=False, cache=None)[0].float()
        qm.apply_expert_bits(np.full((n_layers, n_experts), 16, dtype=np.int32), group_size=128)
        with torch.inference_mode():
            noop_logits = model(test_inp, multi_segments=False, cache=None)[0].float()
        max_diff = float((ref_logits - noop_logits).abs().max().item())
        if max_diff > 0.0:
            raise RuntimeError(f"No-op restore exactness FAILED: max_diff={max_diff}")
        print(f"[probe1:{model_key}] Exactness OK: noop_max_diff={max_diff:.1e}", flush=True)
        pv.step(1)

    # 2) Calibration stats & cross-loop calibration blindness
    with run_obj.phase(f"{model_key}_calib", total=1, unit="pass") as pc:
        calib = collect_calibration_stats(model, calib_seqs, device)
        print(f"[probe1:{model_key}] Cross-loop calibration shift: {json.dumps(calib['calib_shift'])}", flush=True)
        pc.step(1)

    # 3) BF16 reference + record oracle routing traces
    with run_obj.phase(f"{model_key}_bf16", total=1, unit="eval") as pb:
        qm.restore_all()
        bf16_eval = eval_nll_with_trace(model, eval_seqs, device, mode="record")
        bf16_traces = bf16_eval["recorded_traces"]
        bf16_arc, bf16_arc_flags = eval_arc_ifm(model, tokenizer, arc_samples, device)
        bf16_per_seq = np.asarray(bf16_eval["per_seq"], dtype=np.float64)
        print(
            f"[probe1:{model_key}] BF16: NLL={bf16_eval['nll']:.4f} "
            f"(text={bf16_eval['nll_text']:.4f}, code={bf16_eval['nll_code']:.4f}) "
            f"ARC={bf16_arc*100:.1f}%",
            flush=True,
        )
        pb.step(1)

    # Build all quantization conditions
    rng = np.random.default_rng(seed)
    rand_scores = rng.random((n_layers, n_experts))
    k20 = int(round(0.20 * total_experts))
    k50 = int(round(0.50 * total_experts))

    def make_mix(scores: np.ndarray, k_hi: int, hi_bits: int, lo_bits: int, per_layer: bool = False) -> np.ndarray:
        mask = select_topk_experts_mask(scores, k_hi, per_layer_equal=per_layer)
        mat = np.full((n_layers, n_experts), lo_bits, dtype=np.int32)
        mat[mask] = hi_bits
        return mat

    conditions: list[dict[str, Any]] = [
        {"name": "exp_w4_g128", "bits": np.full((n_layers, n_experts), 4, dtype=np.int32), "gs": 128, "attn": 16, "rtr": 16, "arc": True, "cf": False},
        {"name": "exp_w3_g128", "bits": np.full((n_layers, n_experts), 3, dtype=np.int32), "gs": 128, "attn": 16, "rtr": 16, "arc": True, "cf": True},
        {"name": "exp_w2_g128", "bits": np.full((n_layers, n_experts), 2, dtype=np.int32), "gs": 128, "attn": 16, "rtr": 16, "arc": True, "cf": True},
        {"name": "exp_w2_g64",  "bits": np.full((n_layers, n_experts), 2, dtype=np.int32), "gs": 64,  "attn": 16, "rtr": 16, "arc": True, "cf": True},
        {"name": "all_w3_g128", "bits": np.full((n_layers, n_experts), 3, dtype=np.int32), "gs": 128, "attn": 3,  "rtr": 16, "arc": False, "cf": False},
        {"name": "all_w4_rtr4", "bits": np.full((n_layers, n_experts), 4, dtype=np.int32), "gs": 128, "attn": 4,  "rtr": 4,  "arc": False, "cf": False},
    ]

    # Huginn architectural feedback-exposure contrast (recurrent vs non-recurrent layers)
    if len(qm.recurrent_layers) < n_layers:
        rec_hi = np.full((n_layers, n_experts), 2, dtype=np.int32)
        nonrec_hi = np.full((n_layers, n_experts), 2, dtype=np.int32)
        for l in range(n_layers):
            if l in qm.recurrent_layers:
                rec_hi[l, :] = 4
            else:
                nonrec_hi[l, :] = 4
        conditions.append({"name": "protect_rec_w4_rest_w2", "bits": rec_hi, "gs": 128, "attn": 16, "rtr": 16, "arc": True, "cf": False})
        conditions.append({"name": "protect_nonrec_w4_rest_w2", "bits": nonrec_hi, "gs": 128, "attn": 16, "rtr": 16, "arc": True, "cf": False})

    # Iso-bit 2.40-bit mixed precision (20% W4 + 80% W2, g128)
    conditions.extend([
        {"name": "mix24_random",           "bits": make_mix(rand_scores, k20, 4, 2), "gs": 128, "attn": 16, "rtr": 16, "arc": False, "cf": False},
        {"name": "mix24_global_freq",      "bits": make_mix(calib["global_freq"], k20, 4, 2), "gs": 128, "attn": 16, "rtr": 16, "arc": True, "cf": False},
        {"name": "mix24_global_energy",    "bits": make_mix(calib["global_energy"], k20, 4, 2), "gs": 128, "attn": 16, "rtr": 16, "arc": False, "cf": False},
        {"name": "mix24_max_stage_freq",   "bits": make_mix(calib["max_stage_freq"], k20, 4, 2), "gs": 128, "attn": 16, "rtr": 16, "arc": True, "cf": False},
        {"name": "mix24_early_feedback",   "bits": make_mix(calib["early_feedback"], k20, 4, 2), "gs": 128, "attn": 16, "rtr": 16, "arc": True, "cf": False},
        {"name": "mix24_layer_bal_fb",     "bits": make_mix(calib["early_feedback"], k20, 4, 2, per_layer=True), "gs": 128, "attn": 16, "rtr": 16, "arc": False, "cf": False},
        # 3.00-bit matched vs uniform W3 (50% W4 + 50% W2, g128)
        {"name": "mix30_global_freq",      "bits": make_mix(calib["global_freq"], k50, 4, 2), "gs": 128, "attn": 16, "rtr": 16, "arc": False, "cf": False},
        {"name": "mix30_early_feedback",   "bits": make_mix(calib["early_feedback"], k50, 4, 2), "gs": 128, "attn": 16, "rtr": 16, "arc": True, "cf": False},
    ])

    results: dict[str, dict[str, Any]] = {
        "bf16": {
            "avg_expert_bits": 16.0,
            "nll": bf16_eval["nll"],
            "nll_text": bf16_eval["nll_text"],
            "nll_code": bf16_eval["nll_code"],
            "delta_nll": 0.0,
            "delta_nll_ci95": [0.0, 0.0],
            "arc_acc": bf16_arc,
            "rec_flip_top1": 0.0,
            "rec_flip_any": 0.0,
            "flip_top1_by_loop": [0.0] * qm.n_loops,
            "flip_any_by_loop": [0.0] * qm.n_loops,
        }
    }
    cf_results: dict[str, dict[str, Any]] = {}
    per_seq_store: dict[str, np.ndarray] = {"bf16": bf16_per_seq}

    with run_obj.phase(f"{model_key}_sweep", total=len(conditions), unit="conds") as ps:
        for step_i, cspec in enumerate(conditions, 1):
            cname = cspec["name"]
            avg_b = qm.apply_expert_bits(
                cspec["bits"],
                group_size=cspec["gs"],
                quant_attn_bits=cspec["attn"],
                quant_router_bits=cspec["rtr"],
            )
            ev = eval_nll_with_trace(model, eval_seqs, device, mode="compare", ref_traces=bf16_traces)
            arr = np.asarray(ev["per_seq"], dtype=np.float64)
            per_seq_store[cname] = arr
            dnll, lo, hi = paired_bootstrap_ci(arr - bf16_per_seq)

            arc_acc = None
            if cspec["arc"]:
                arc_acc, _ = eval_arc_ifm(model, tokenizer, arc_samples, device)

            rec = {
                "avg_expert_bits": avg_b,
                "group_size": cspec["gs"],
                "nll": ev["nll"],
                "nll_text": ev["nll_text"],
                "nll_code": ev["nll_code"],
                "delta_nll": dnll,
                "delta_nll_ci95": [lo, hi],
                "arc_acc": arc_acc,
                "rec_flip_top1": ev["rec_flip_top1"],
                "rec_flip_any": ev["rec_flip_any"],
                "nonrec_flip_any": ev["nonrec_flip_any"],
                "flip_top1_by_loop": ev["flip_top1_by_loop"],
                "flip_any_by_loop": ev["flip_any_by_loop"],
            }
            results[cname] = rec

            # 2x2 Counterfactual Routing-Flip Cascade Decomposition
            if cspec["cf"]:
                # Record quantized routing trajectory on eval_seqs
                q_rec_ev = eval_nll_with_trace(model, eval_seqs, device, mode="record")
                q_traces = q_rec_ev["recorded_traces"]
                # (Quant Weights, BF16 Oracle Routing)
                qw_bf16r = eval_nll_with_trace(model, eval_seqs, device, mode="replay", ref_traces=bf16_traces)
                # (BF16 Weights, Quant Routing)
                qm.restore_all()
                bf16w_qr = eval_nll_with_trace(model, eval_seqs, device, mode="replay", ref_traces=q_traces)

                tot_d = float(ev["nll"] - bf16_eval["nll"])
                wt_only_d = float(qw_bf16r["nll"] - bf16_eval["nll"])
                rt_only_d = float(bf16w_qr["nll"] - bf16_eval["nll"])
                route_saved_d = float(ev["nll"] - qw_bf16r["nll"])
                cf_results[cname] = {
                    "nll_bf16": bf16_eval["nll"],
                    "nll_qw_qr": ev["nll"],
                    "nll_qw_bf16r": qw_bf16r["nll"],
                    "nll_bf16w_qr": bf16w_qr["nll"],
                    "total_delta_nll": tot_d,
                    "weight_only_delta_nll": wt_only_d,
                    "route_only_delta_nll": rt_only_d,
                    "route_cascade_saved_nll": route_saved_d,
                    "route_cascade_frac": float(route_saved_d / tot_d) if abs(tot_d) > 1e-8 else 0.0,
                    "flip_top1_by_loop": ev["flip_top1_by_loop"],
                    "flip_any_by_loop": ev["flip_any_by_loop"],
                }
                print(
                    f"  [CF:{model_key}:{cname}] total ΔNLL={tot_d:+.4f} | "
                    f"Qw+BF16r={wt_only_d:+.4f} | BF16w+Qr={rt_only_d:+.4f} | "
                    f"route_cascade={route_saved_d:+.4f} ({cf_results[cname]['route_cascade_frac']*100:.1f}%) | "
                    f"flip_any_by_loop={[round(x*100,1) for x in ev['flip_any_by_loop']]}",
                    flush=True,
                )

            arc_str = f"{arc_acc*100:.1f}%" if arc_acc is not None else "n/a"
            print(
                f"[probe1:{model_key}] {cname:26s} bits={avg_b:4.2f} NLL={ev['nll']:.4f} "
                f"(Δ={dnll:+.4f}) ARC={arc_str:>5s} flip_any={ev['rec_flip_any']*100:5.1f}%",
                flush=True,
            )
            ps.row(
                f"{model_key}_conds",
                condition=cname,
                avg_bits=avg_b,
                delta_nll=dnll,
                arc_acc=arc_acc if arc_acc is not None else 0.0,
                rec_flip_any=ev["rec_flip_any"],
            )
            ps.step(step_i)

    # Key paired contrasts (positive = first method has LOWER NLL than second)
    contrast_pairs = [
        ("mix24_early_feedback", "mix24_global_freq"),
        ("mix24_early_feedback", "mix24_global_energy"),
        ("mix24_early_feedback", "mix24_random"),
        ("mix24_max_stage_freq", "mix24_global_freq"),
        ("mix24_layer_bal_fb", "mix24_global_freq"),
        ("mix30_early_feedback", "exp_w3_g128"),
        ("mix30_early_feedback", "mix30_global_freq"),
    ]
    if "protect_rec_w4_rest_w2" in per_seq_store:
        contrast_pairs.append(("protect_rec_w4_rest_w2", "protect_nonrec_w4_rest_w2"))

    contrasts: dict[str, dict[str, Any]] = {}
    for a, b in contrast_pairs:
        if a in per_seq_store and b in per_seq_store:
            diff = per_seq_store[b] - per_seq_store[a]  # positive => a is better (lower NLL)
            m, lo, hi = paired_bootstrap_ci(diff)
            contrasts[f"{a}_vs_{b}"] = {
                "advantage_nll": m,
                "ci95": [lo, hi],
                "wins": int((diff > 0).sum()),
                "n": int(len(diff)),
            }

    qm.restore_all()
    del qm, model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model_key": model_key,
        "n_layers": n_layers,
        "n_experts": n_experts,
        "calib_shift": calib["calib_shift"],
        "conditions": results,
        "counterfactuals": cf_results,
        "contrasts": contrasts,
    }


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Round 1 Probe 1: Looped MoE Quantization & Routing-Flip Cascade",
        "",
        f"- GPU: `{summary['gpu_index']}` ({summary['gpu_name']}), elapsed `{summary['elapsed_sec']:.1f}s`",
        "",
    ]
    for mk, mres in summary["models"].items():
        lines.append(f"## Model: `{mk}`")
        lines.append(f"- Cross-loop calibration shift: `{json.dumps(mres['calib_shift'])}`")
        lines.append("")
        lines.append("| Condition | Avg Bits | NLL | ΔNLL vs BF16 (95% CI) | ARC-200 | Rec Flip Any% | Flip-Any by Loop (%) |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for cname, rec in mres["conditions"].items():
            lo, hi = rec["delta_nll_ci95"]
            arc_s = f"{rec['arc_acc']*100:.1f}%" if rec["arc_acc"] is not None else "—"
            fl_s = ", ".join(f"{x*100:.1f}" for x in rec["flip_any_by_loop"])
            lines.append(
                f"| `{cname}` | {rec['avg_expert_bits']:.2f} | {rec['nll']:.4f} | "
                f"{rec['delta_nll']:+.4f} [{lo:+.4f}, {hi:+.4f}] | {arc_s} | "
                f"{rec['rec_flip_any']*100:.1f}% | [{fl_s}] |"
            )
        lines.append("")
        lines.append("### 2x2 Counterfactual Routing-Flip Cascade Decomposition")
        lines.append("| Quant Setting | Total ΔNLL | Weight-Only (Qw+BF16r) | Route-Only (BF16w+Qr) | Cascade Saved | Cascade Share |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for cname, cf in mres["counterfactuals"].items():
            lines.append(
                f"| `{cname}` | {cf['total_delta_nll']:+.4f} | {cf['weight_only_delta_nll']:+.4f} | "
                f"{cf['route_only_delta_nll']:+.4f} | {cf['route_cascade_saved_nll']:+.4f} | "
                f"{cf['route_cascade_frac']*100:.1f}% |"
            )
        lines.append("")
        lines.append("### Key Paired Contrasts (positive = first has LOWER NLL)")
        for ck, cv in mres["contrasts"].items():
            lo, hi = cv["ci95"]
            lines.append(f"- `{ck}`: `{cv['advantage_nll']:+.4f}` CI `[{lo:+.4f}, {hi:+.4f}]` ({cv['wins']}/{cv['n']} wins)")
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
    gpu_idx, gpu_lock, gpu_status = acquire_idle_gpu(cfg.get("preferred_gpu"))
    try:
        print(
            f"[probe1] Acquired idle GPU {gpu_idx}: {gpu_status.name} "
            f"(mem={gpu_status.mem_used_mib}MiB, util={gpu_status.util_pct}%)",
            flush=True,
        )
        device = torch.device(f"cuda:{gpu_idx}")
        torch.cuda.set_device(device)

        boot = bootstrap_xllm(args.xllm_loop_dir, attention_backend="sdpa")
        print(f"[probe1] xllm bootstrap: {json.dumps(boot)}", flush=True)

        arc_samples = load_arc_challenge_samples(
            Path(args.dataset_dir), n_samples=int(cfg.get("n_arc", 200)), seed=int(cfg.get("seed", 20261126))
        )

        model_keys = list(cfg["models"].keys())
        plan = [
            {"phase": "setup", "of": 1, "unit": "env"},
            {"phase": "download", "of": 1, "unit": "assets"},
        ]
        for mk in model_keys:
            n_conds = 16 if mk == "huginn_moe" else 14
            plan.extend([
                {"phase": f"{mk}_verify", "of": 1, "unit": "check"},
                {"phase": f"{mk}_calib", "of": 1, "unit": "pass"},
                {"phase": f"{mk}_bf16", "of": 1, "unit": "eval"},
                {"phase": f"{mk}_sweep", "of": n_conds, "unit": "conds"},
            ])
        plan.append({"phase": "report", "of": 1, "unit": "file"})
        orx.plan(plan)
        run_obj = orx.progress(total=1, unit="run")

        models_out: dict[str, Any] = {}
        for mk, mspec in cfg["models"].items():
            models_out[mk] = run_model_probe1(
                mk,
                mspec,
                model_dir=Path(args.model_dir),
                dataset_dir=Path(args.dataset_dir),
                arc_samples=arc_samples,
                cfg=cfg,
                device=device,
                run_obj=run_obj,
            )

        summary = {
            "idea": "probe1_loopmoe_quant",
            "gpu_index": gpu_idx,
            "gpu_name": gpu_status.name,
            "elapsed_sec": time.time() - t0,
            "config": cfg,
            "models": models_out,
        }
        with run_obj.phase("report", total=1, unit="file") as pr:
            (out_dir / "summary_probe1_quant.json").write_text(json.dumps(summary, indent=2))
            md = render_report(summary)
            (out_dir / "report_probe1_quant.md").write_text(md)
            print("========== REPORT (report_probe1_quant.md) ==========", flush=True)
            print(md, flush=True)
            print("========== END REPORT ==========", flush=True)

            headline: dict[str, float] = {}
            for mk, mres in models_out.items():
                if "exp_w2_g128" in mres["counterfactuals"]:
                    headline[f"{mk}_w2_cascade_frac"] = mres["counterfactuals"]["exp_w2_g128"]["route_cascade_frac"]
                    headline[f"{mk}_w2_delta_nll"] = mres["conditions"]["exp_w2_g128"]["delta_nll"]
                if "mix24_early_feedback_vs_mix24_global_freq" in mres["contrasts"]:
                    headline[f"{mk}_fb_vs_freq_24b"] = mres["contrasts"]["mix24_early_feedback_vs_mix24_global_freq"]["advantage_nll"]
                if "protect_rec_w4_rest_w2_vs_protect_nonrec_w4_rest_w2" in mres["contrasts"]:
                    headline[f"{mk}_rec_vs_nonrec"] = mres["contrasts"]["protect_rec_w4_rest_w2_vs_protect_nonrec_w4_rest_w2"]["advantage_nll"]
            pr.step(1, **headline)
    finally:
        gpu_lock.release()


if __name__ == "__main__":
    main()
