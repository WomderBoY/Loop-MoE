"""Direction 2 Feasibility Pilot: Which MoE experts suffer from cross-loop weight sharing?

Implements the exact preliminary study on IFM/LoopedLM-P1-moe-ouro-500b (4 loops)
and IFM/LoopedLM-P1-moe-huginn-500b (8 loops):
  0. Exactness verification: pure-PyTorch GroupRMSNorm backward shim + per-visit W2
     gradient decomposition vs native PyTorch experts.weight2.grad.
  1. Unbiased cross-split per-loop gradient conflict scoring (calib_a vs calib_b):
     - S_raw: unbiased centered inner product sum_r <g_tilde_A[r], g_tilde_B[r]>
     - S_orth_W: after projecting out base weight direction W2 (removes per-loop scaling)
     - S_orth_mean: after projecting out shared mean direction (removes shared-adapter+scalar)
     - S_orth_both: orthogonal to both W2 and shared mean
     - Cross-split direction cosine, concentration (top-20% share), token counts.
  2. Exp 2A (Single-Expert Counterfactual Scatter): for stratified experts, apply fixed-norm
     shared step vs per-loop step (computed from adapt_train) and measure exact held-out NLL
     unbinding gain through the native router; correlate with calib conflict predictions.
  3. Exp 2B (Strictly Parameter-Matched Held-Out Group Adaptation):
     Compare base_frozen, conflict_top, matched_high, matched_low, random, gradnorm_top,
     freq_top, shared_plus_scalar, uniform_all, recon_layer, and router_loop_mod on identical
     adapt_train batches and held-out eval_heldout (50% text + 50% code).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from loop_research import orx_run as orx
from loop_research.data_assets import build_disjoint_text_code_splits
from loop_research.gpu_utils import acquire_idle_gpu
from loop_research.ifm_runtime import bootstrap_xllm, load_ifm_model_for_autograd


def spearman_corr(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.size < 3 or np.std(xa) < 1e-12 or np.std(ya) < 1e-12:
        return 0.0
    rx = np.argsort(np.argsort(xa)).astype(np.float64)
    ry = np.argsort(np.argsort(ya)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / denom) if denom > 1e-12 else 0.0


def pearson_corr(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.size < 3 or np.std(xa) < 1e-12 or np.std(ya) < 1e-12:
        return 0.0
    xa = xa - xa.mean()
    ya = ya - ya.mean()
    denom = float(np.sqrt((xa * xa).sum() * (ya * ya).sum()))
    return float((xa * ya).sum() / denom) if denom > 1e-12 else 0.0


def paired_bootstrap_ci(
    diffs: list[float] | np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 12345,
) -> tuple[float, float, float]:
    arr = np.asarray(diffs, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(arr.mean()), float(lo), float(hi)


def compute_lm_loss(model, tokens_1d: list[int], device: torch.device) -> torch.Tensor:
    """Compute exact next-token cross-entropy loss through IFM xllm model."""
    inp = torch.tensor([tokens_1d[:-1]], device=device, dtype=torch.long)
    tgt = torch.tensor([tokens_1d[1:]], device=device, dtype=torch.long)
    logits, _aux, _cache = model(inp, multi_segments=False, cache=None)
    # logits: [1, L, V] fp32
    return F.cross_entropy(logits.view(-1, logits.shape[-1]), tgt.view(-1), reduction="mean")


@torch.inference_mode()
def evaluate_nll_per_seq(model, seqs: list[dict[str, Any]], device: torch.device) -> list[float]:
    nlls: list[float] = []
    for item in seqs:
        loss = compute_lm_loss(model, item["tokens"], device)
        nlls.append(float(loss.item()))
    return nlls


# ---------------------------------------------------------------------------
# Exact Per-Visit W2 Gradient Capture Wrapper
# ---------------------------------------------------------------------------

@contextmanager
def per_visit_w2_grad_collector(
    model,
    tracked_layers: list[int],
    n_loops: int,
    n_experts: int,
    model_dim: int,
    inter_dim: int,
    device: torch.device,
):
    """Wrap tracked Expert.forward modules to capture exact FP32 per-visit W2 gradients on CPU."""
    from xllm.modules.fused_ops import memory_efficient_dropout, mgmm

    # Allocate CPU FP32 accumulators: [n_tracked, n_experts, n_loops, model_dim, inter_dim]
    grads_cpu: dict[int, torch.Tensor] = {
        lid: torch.zeros((n_experts, n_loops, model_dim, inter_dim), dtype=torch.float32)
        for lid in tracked_layers
    }
    counts: dict[int, np.ndarray] = {
        lid: np.zeros((n_experts, n_loops), dtype=np.int64)
        for lid in tracked_layers
    }
    visit_cursor: dict[int, int] = {lid: 0 for lid in tracked_layers}
    orig_forwards: dict[int, Any] = {}
    anchor_holder: dict[str, torch.Tensor | None] = {"anchor": None}

    def _reset_visits() -> None:
        for lid in tracked_layers:
            visit_cursor[lid] = 0

    def _zero_accum() -> None:
        for lid in tracked_layers:
            grads_cpu[lid].zero_()
            counts[lid].fill(0)

    def _make_wrapped_forward(lid: int, expert_mod):
        def _wrapped(x: torch.Tensor, group_sizes: list[int]) -> torch.Tensor:
            r = visit_cursor[lid]
            visit_cursor[lid] = r + 1
            w1 = rearrange(expert_mod.weight1, "(n s) d -> n s d", n=expert_mod.n_local_experts)
            w2 = rearrange(expert_mod.weight2, "(n s) d -> n s d", n=expert_mod.n_local_experts)
            w3 = rearrange(expert_mod.weight3, "(n s) d -> n s d", n=expert_mod.n_local_experts)
            h1 = F.silu(mgmm(x, w1, group_sizes, True, expert_mod.backend))
            h3 = mgmm(x, w3, group_sizes, True, expert_mod.backend)
            hidden = memory_efficient_dropout(h1 * h3, expert_mod.hidden_dropout, expert_mod.training)
            y = mgmm(hidden, w2, group_sizes, True, expert_mod.backend)

            # Ensure y participates in autograd even when base weights are frozen
            anchor = anchor_holder["anchor"]
            if anchor is not None:
                y = y + 0.0 * anchor

            if y.requires_grad and r < n_loops:
                hidden_det = hidden.detach()
                gs_copy = list(group_sizes)
                for e_idx, cnt in enumerate(gs_copy):
                    counts[lid][e_idx, r] += cnt

                def _bwd_hook(grad_y: torch.Tensor) -> None:
                    off = 0
                    for e_idx, cnt in enumerate(gs_copy):
                        if cnt > 0:
                            gy_e = grad_y[off : off + cnt].float()
                            h_e = hidden_det[off : off + cnt].float()
                            # W2_e has shape [model_dim, inter_dim]; y_e = h_e @ W2_e^T
                            gw2_e = (gy_e.T @ h_e).cpu()
                            grads_cpu[lid][e_idx, r].add_(gw2_e)
                        off += cnt

                y.register_hook(_bwd_hook)
            return y

        return _wrapped

    try:
        for lid in tracked_layers:
            exp_mod = model.layers[lid].moe.experts
            orig_forwards[lid] = exp_mod.forward
            exp_mod.forward = _make_wrapped_forward(lid, exp_mod)

        def _accumulate_on_seqs(seqs: list[dict[str, Any]], *, clear_first: bool = True) -> float:
            if clear_first:
                _zero_accum()
            n_seq = len(seqs)
            total_loss = 0.0
            for item in seqs:
                _reset_visits()
                anchor = torch.zeros(1, device=device, dtype=torch.bfloat16, requires_grad=True)
                anchor_holder["anchor"] = anchor
                loss = compute_lm_loss(model, item["tokens"], device)
                total_loss += float(loss.item())
                (loss / max(n_seq, 1)).backward()
                anchor_holder["anchor"] = None
            return total_loss / max(n_seq, 1)

        yield grads_cpu, counts, _accumulate_on_seqs, _reset_visits
    finally:
        for lid, fn in orig_forwards.items():
            model.layers[lid].moe.experts.forward = fn


def verify_backward_and_decomp(
    model,
    tracked_layer: int,
    n_loops: int,
    n_experts: int,
    model_dim: int,
    inter_dim: int,
    sample_tokens: list[int],
    device: torch.device,
) -> dict[str, Any]:
    """Verify sum_r g_{e, r} from per-visit hook matches native PyTorch weight2.grad."""
    short_toks = sample_tokens[:64]
    for p in model.parameters():
        p.requires_grad_(False)
    w2_param = model.layers[tracked_layer].moe.experts.weight2
    w2_param.requires_grad_(True)
    if w2_param.grad is not None:
        w2_param.grad.zero_()

    with per_visit_w2_grad_collector(
        model, [tracked_layer], n_loops, n_experts, model_dim, inter_dim, device
    ) as (grads_cpu, _counts, accum_fn, _reset):
        loss_val = accum_fn([{"tokens": short_toks}], clear_first=True)

    native_grad = w2_param.grad.detach().float().cpu().reshape(n_experts, model_dim, inter_dim)
    summed_grad = grads_cpu[tracked_layer].sum(dim=1)  # sum over loops -> [E, D, inter]
    w2_param.grad = None
    w2_param.requires_grad_(False)

    diff_norm = float((native_grad - summed_grad).norm().item())
    ref_norm = float(native_grad.norm().item())
    rel_err = diff_norm / max(ref_norm, 1e-12)
    res = {
        "layer": tracked_layer,
        "loss": loss_val,
        "native_grad_norm": ref_norm,
        "summed_per_visit_norm": float(summed_grad.norm().item()),
        "relative_l2_error": rel_err,
    }
    print(f"[verify] Per-visit gradient decomposition check: {json.dumps(res)}")
    if not math.isfinite(rel_err) or rel_err > 5e-3:
        raise RuntimeError(f"Backward/decomposition check FAILED: {res}")
    return res


# ---------------------------------------------------------------------------
# Unbiased Cross-Split Conflict Scoring
# ---------------------------------------------------------------------------

def score_expert_conflicts(
    grads_a: dict[int, torch.Tensor],
    grads_b: dict[int, torch.Tensor],
    counts_a: dict[int, np.ndarray],
    counts_b: dict[int, np.ndarray],
    w2_base_cpu: dict[int, torch.Tensor],
) -> list[dict[str, Any]]:
    """Compute unbiased cross-split conflict scores (raw, orth-W, orth-mean, orth-both) per expert."""
    records: list[dict[str, Any]] = []
    for lid in sorted(grads_a.keys()):
        ga_all = grads_a[lid]  # [E, R, D, I]
        gb_all = grads_b[lid]
        w2_all = w2_base_cpu[lid]  # [E, D, I]
        n_exp, n_loops = ga_all.shape[0], ga_all.shape[1]

        for e in range(n_exp):
            ga = ga_all[e].reshape(n_loops, -1)  # [R, P]
            gb = gb_all[e].reshape(n_loops, -1)
            w_flat = w2_all[e].reshape(-1)
            w_unit = w_flat / w_flat.norm().clamp_min(1e-12)

            # Raw cross-split energies
            bar_a = ga.mean(dim=0)
            bar_b = gb.mean(dim=0)
            tilde_a = ga - bar_a.unsqueeze(0)
            tilde_b = gb - bar_b.unsqueeze(0)

            s_raw = float((tilde_a * tilde_b).sum().item())
            shared_energy = float((bar_a * bar_b).sum().item())
            total_energy = float((ga * gb).sum().item())
            grad_norm_mean = float(0.5 * (ga.norm().item() + gb.norm().item()))
            dir_cos = float(
                (tilde_a * tilde_b).sum().item()
                / max(tilde_a.norm().item() * tilde_b.norm().item(), 1e-12)
            )

            # 1) Project out base weight direction W2 (removes per-loop scalar scaling of W2)
            ga_ow = ga - (ga @ w_unit).unsqueeze(1) * w_unit.unsqueeze(0)
            gb_ow = gb - (gb @ w_unit).unsqueeze(1) * w_unit.unsqueeze(0)
            ta_ow = ga_ow - ga_ow.mean(dim=0, keepdim=True)
            tb_ow = gb_ow - gb_ow.mean(dim=0, keepdim=True)
            s_orth_w = float((ta_ow * tb_ow).sum().item())

            # 2) Project out shared mean update direction (cross-fitted u_b for A and u_a for B to stay unbiased!)
            ua = bar_a / bar_a.norm().clamp_min(1e-12)
            ub = bar_b / bar_b.norm().clamp_min(1e-12)
            ta_om = tilde_a - (tilde_a @ ub).unsqueeze(1) * ub.unsqueeze(0)
            tb_om = tilde_b - (tilde_b @ ua).unsqueeze(1) * ua.unsqueeze(0)
            s_orth_mean = float((ta_om * tb_om).sum().item())

            # 3) Project out BOTH W2 and shared mean direction (2D Gram-Schmidt per split)
            def _proj_out_2d(mat: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
                q1 = v1 / v1.norm().clamp_min(1e-12)
                v2p = v2 - (v2 @ q1) * q1
                q2 = v2p / v2p.norm().clamp_min(1e-12)
                return mat - (mat @ q1).unsqueeze(1) * q1.unsqueeze(0) - (mat @ q2).unsqueeze(1) * q2.unsqueeze(0)

            ta_ob = _proj_out_2d(tilde_a, w_unit, ub)
            tb_ob = _proj_out_2d(tilde_b, w_unit, ua)
            s_orth_both = float((ta_ob * tb_ob).sum().item())

            # Pairwise cross-loop cosine matrix (cross-split: ga[r1] . gb[r2])
            norms_a = ga.norm(dim=1).clamp_min(1e-12)
            norms_b = gb.norm(dim=1).clamp_min(1e-12)
            cos_mat = ((ga @ gb.T) / (norms_a.unsqueeze(1) * norms_b.unsqueeze(0))).numpy()
            diag_cos = float(np.mean(np.diag(cos_mat)))
            off_mask = ~np.eye(n_loops, dtype=bool)
            offdiag_cos = float(np.mean(cos_mat[off_mask])) if n_loops > 1 else diag_cos

            cnt_per_loop = (counts_a[lid][e] + counts_b[lid][e]).astype(np.int64)
            tot_cnt = int(cnt_per_loop.sum())
            probs = cnt_per_loop / max(tot_cnt, 1)
            loop_ent = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))) / math.log(max(n_loops, 2)))

            records.append(
                {
                    "layer": int(lid),
                    "expert": int(e),
                    "s_raw": s_raw,
                    "s_orth_w": s_orth_w,
                    "s_orth_mean": s_orth_mean,
                    "s_orth_both": s_orth_both,
                    "shared_energy": shared_energy,
                    "total_energy": total_energy,
                    "conflict_ratio": s_orth_both / max(abs(total_energy), 1e-8),
                    "grad_norm_mean": grad_norm_mean,
                    "cross_split_dir_cos": dir_cos,
                    "same_loop_cos": diag_cos,
                    "diff_loop_cos": offdiag_cos,
                    "token_count": tot_cnt,
                    "token_count_per_loop": cnt_per_loop.tolist(),
                    "loop_entropy": loop_ent,
                }
            )
    return records


# ---------------------------------------------------------------------------
# Expert Selection Rules (Strictly Matched Budgets!)
# ---------------------------------------------------------------------------

def build_matched_high_low_groups(
    expert_records: list[dict[str, Any]],
    k_per_layer: int,
    score_key: str = "s_orth_both",
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Within each layer, match experts by (log token_count, log grad_norm) then pick high vs low conflict."""
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for r in expert_records:
        by_layer.setdefault(r["layer"], []).append(r)

    high_grp: list[tuple[int, int]] = []
    low_grp: list[tuple[int, int]] = []
    for lid, recs in sorted(by_layer.items()):
        # Standardize nuisance covariates (token_count and grad_norm_mean)
        lc = np.array([math.log(r["token_count"] + 1.0) for r in recs])
        lg = np.array([math.log(r["grad_norm_mean"] + 1e-8) for r in recs])
        z_cov = (lc - lc.mean()) / (lc.std() + 1e-6) + (lg - lg.mean()) / (lg.std() + 1e-6)
        order = np.argsort(z_cov)
        # Form adjacent covariate-matched pairs, pick k_per_layer pairs with largest conflict contrast
        pairs: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        for idx in range(0, len(order) - 1, 2):
            r1 = recs[int(order[idx])]
            r2 = recs[int(order[idx + 1])]
            if r1[score_key] >= r2[score_key]:
                hi, lo = r1, r2
            else:
                hi, lo = r2, r1
            pairs.append((hi[score_key] - lo[score_key], hi, lo))
        pairs.sort(key=lambda x: x[0], reverse=True)
        for _gap, hi, lo in pairs[:k_per_layer]:
            high_grp.append((lid, hi["expert"]))
            low_grp.append((lid, lo["expert"]))
    return high_grp, low_grp


def select_group_experts(
    expert_records: list[dict[str, Any]],
    k_per_layer: int,
    *,
    score_key: str = "s_orth_both",
    seed: int = 20261124,
) -> dict[str, Any]:
    """Return exact layer-expert lists for every comparison condition."""
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for r in expert_records:
        by_layer.setdefault(r["layer"], []).append(r)

    rng = random.Random(seed)
    conflict_top: list[tuple[int, int]] = []
    gradnorm_top: list[tuple[int, int]] = []
    freq_top: list[tuple[int, int]] = []
    rand_grp: list[tuple[int, int]] = []
    all_experts: list[tuple[int, int]] = []

    layer_conflict_totals: list[tuple[float, int]] = []
    for lid, recs in sorted(by_layer.items()):
        all_experts.extend((lid, r["expert"]) for r in recs)
        s_sorted = sorted(recs, key=lambda r: r[score_key], reverse=True)
        g_sorted = sorted(recs, key=lambda r: r["grad_norm_mean"], reverse=True)
        f_sorted = sorted(recs, key=lambda r: r["token_count"], reverse=True)
        r_copy = list(recs)
        rng.shuffle(r_copy)

        conflict_top.extend((lid, r["expert"]) for r in s_sorted[:k_per_layer])
        gradnorm_top.extend((lid, r["expert"]) for r in g_sorted[:k_per_layer])
        freq_top.extend((lid, r["expert"]) for r in f_sorted[:k_per_layer])
        rand_grp.extend((lid, r["expert"]) for r in r_copy[:k_per_layer])
        layer_conflict_totals.append((sum(r[score_key] for r in recs), lid))

    matched_high, matched_low = build_matched_high_low_groups(expert_records, k_per_layer, score_key=score_key)
    layer_conflict_totals.sort(reverse=True)
    top_layer_id = layer_conflict_totals[0][1]
    recon_layer_experts = [(top_layer_id, r["expert"]) for r in by_layer[top_layer_id]]

    return {
        "conflict_top": conflict_top,
        "matched_high": matched_high,
        "matched_low": matched_low,
        "random": rand_grp,
        "gradnorm_top": gradnorm_top,
        "freq_top": freq_top,
        "all_experts": all_experts,
        "recon_layer": recon_layer_experts,
        "recon_layer_id": top_layer_id,
    }


# ---------------------------------------------------------------------------
# Exp 2A: Single-Expert Fixed-Norm Counterfactual Intervention (Scatter!)
# ---------------------------------------------------------------------------

def run_single_expert_counterfactuals(
    model,
    expert_records: list[dict[str, Any]],
    train_grads_cpu: dict[int, torch.Tensor],
    eval_seqs: list[dict[str, Any]],
    base_eval_nll: float,
    *,
    n_loops: int,
    model_dim: int,
    inter_dim: int,
    step_norm_eps: float,
    experts_per_layer: int,
    device: torch.device,
    run_phase: Any = None,
) -> dict[str, Any]:
    """For stratified individual experts, compare fixed-norm shared step vs per-loop step on held-out eval."""
    from xllm.modules.fused_ops import memory_efficient_dropout, mgmm

    by_layer: dict[int, list[dict[str, Any]]] = {}
    for r in expert_records:
        by_layer.setdefault(r["layer"], []).append(r)

    chosen: list[dict[str, Any]] = []
    for lid, recs in sorted(by_layer.items()):
        ordered = sorted(recs, key=lambda r: r["s_orth_both"])
        # Stratified evenly across quantiles from lowest to highest conflict
        idxs = np.linspace(0, len(ordered) - 1, experts_per_layer).round().astype(int).tolist()
        seen = set()
        for ii in idxs:
            if ii not in seen:
                seen.add(ii)
                chosen.append(ordered[ii])

    rows: list[dict[str, Any]] = []
    for step_i, rec in enumerate(chosen, 1):
        lid = rec["layer"]
        e = rec["expert"]
        g_tr = train_grads_cpu[lid][e]  # [R, D, I] from adapt_train (independent of calib & eval!)
        g_mean = g_tr.mean(dim=0)  # [D, I]
        mean_norm = float(g_mean.norm().item())
        rms_norm = float(math.sqrt(float(g_tr.pow(2).sum().item()) / n_loops))

        if mean_norm < 1e-10 or rms_norm < 1e-10:
            continue

        # Matched RMS perturbation magnitude per loop = step_norm_eps
        delta_shared = (-step_norm_eps * (g_mean / mean_norm)).to(device=device, dtype=torch.bfloat16)  # [D, I]
        delta_perloop = (-step_norm_eps * (g_tr / rms_norm)).to(device=device, dtype=torch.bfloat16)  # [R, D, I]

        exp_mod = model.layers[lid].moe.experts
        orig_fwd = exp_mod.forward

        def _eval_with_delta(mode: str) -> float:
            visit_r = {"r": 0}

            def _wrapped(x: torch.Tensor, group_sizes: list[int]) -> torch.Tensor:
                r = visit_r["r"]
                visit_r["r"] = r + 1
                w1 = rearrange(exp_mod.weight1, "(n s) d -> n s d", n=exp_mod.n_local_experts)
                w2 = rearrange(exp_mod.weight2, "(n s) d -> n s d", n=exp_mod.n_local_experts)
                w3 = rearrange(exp_mod.weight3, "(n s) d -> n s d", n=exp_mod.n_local_experts)
                h1 = F.silu(mgmm(x, w1, group_sizes, True, exp_mod.backend))
                h3 = mgmm(x, w3, group_sizes, True, exp_mod.backend)
                hidden = memory_efficient_dropout(h1 * h3, exp_mod.hidden_dropout, exp_mod.training)
                y = mgmm(hidden, w2, group_sizes, True, exp_mod.backend)
                cnt = group_sizes[e]
                if cnt > 0 and r < n_loops:
                    off = sum(group_sizes[:e])
                    dw = delta_shared if mode == "shared" else delta_perloop[r]
                    y = y.clone()
                    y[off : off + cnt] = y[off : off + cnt] + F.linear(hidden[off : off + cnt], dw)
                return y

            exp_mod.forward = _wrapped
            try:
                vals = []
                with torch.inference_mode():
                    for item in eval_seqs:
                        visit_r["r"] = 0
                        vals.append(float(compute_lm_loss(model, item["tokens"], device).item()))
                return float(np.mean(vals))
            finally:
                exp_mod.forward = orig_fwd

        nll_shared = _eval_with_delta("shared")
        nll_perloop = _eval_with_delta("perloop")
        unbinding_gain = nll_shared - nll_perloop  # positive = per-loop beats shared!
        rows.append(
            {
                "layer": lid,
                "expert": e,
                "s_raw": rec["s_raw"],
                "s_orth_w": rec["s_orth_w"],
                "s_orth_mean": rec["s_orth_mean"],
                "s_orth_both": rec["s_orth_both"],
                "token_count": rec["token_count"],
                "grad_norm_mean": rec["grad_norm_mean"],
                "gain_shared": base_eval_nll - nll_shared,
                "gain_perloop": base_eval_nll - nll_perloop,
                "unbinding_gain": unbinding_gain,
            }
        )
        if run_phase is not None:
            run_phase.step(step_i)

    ug = [r["unbinding_gain"] for r in rows]
    summary = {
        "n_experts_tested": len(rows),
        "step_norm_eps": step_norm_eps,
        "mean_unbinding_gain": float(np.mean(ug)) if ug else 0.0,
        "frac_positive_unbinding_gain": float(np.mean([x > 0 for x in ug])) if ug else 0.0,
        "spearman_s_raw_vs_gain": spearman_corr([r["s_raw"] for r in rows], ug),
        "spearman_s_orth_w_vs_gain": spearman_corr([r["s_orth_w"] for r in rows], ug),
        "spearman_s_orth_mean_vs_gain": spearman_corr([r["s_orth_mean"] for r in rows], ug),
        "spearman_s_orth_both_vs_gain": spearman_corr([r["s_orth_both"] for r in rows], ug),
        "pearson_s_orth_both_vs_gain": pearson_corr([r["s_orth_both"] for r in rows], ug),
        "spearman_gradnorm_vs_gain": spearman_corr([r["grad_norm_mean"] for r in rows], ug),
        "spearman_freq_vs_gain": spearman_corr([r["token_count"] for r in rows], ug),
    }
    return {"summary": summary, "rows": rows}


# ---------------------------------------------------------------------------
# Exp 2B: Strictly Parameter-Matched Group Adaptation Modules
# ---------------------------------------------------------------------------

class ExpertLoRAHost(nn.Module):
    """Parameter-matched per-loop LoRA or shared-LoRA+per-loop-scalar adapter for W2 experts."""

    def __init__(
        self,
        selected_experts: list[tuple[int, int]],
        *,
        mode: str,  # "per_loop" | "shared_plus_scalar"
        rank: int,
        n_loops: int,
        model_dim: int,
        inter_dim: int,
        alpha: float = 16.0,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.rank = rank
        self.n_loops = n_loops
        # Rank-stabilized rsLoRA scaling (alpha / sqrt(rank)) prevents artificial LR bias across ranks
        self.scale = alpha / math.sqrt(max(rank, 1))
        self.by_layer: dict[int, list[int]] = {}
        for lid, e in selected_experts:
            self.by_layer.setdefault(lid, []).append(e)

        self.params = nn.ParameterDict()
        for lid, e_list in sorted(self.by_layer.items()):
            n_sel = len(e_list)
            if mode == "per_loop":
                A = nn.Parameter(torch.empty(n_sel, n_loops, rank, inter_dim, dtype=torch.bfloat16))
                B = nn.Parameter(torch.zeros(n_sel, n_loops, model_dim, rank, dtype=torch.bfloat16))
                nn.init.kaiming_uniform_(A, a=math.sqrt(5))
                self.params[f"L{lid}_A"] = A
                self.params[f"L{lid}_B"] = B
            elif mode == "shared_plus_scalar":
                A = nn.Parameter(torch.empty(n_sel, rank, inter_dim, dtype=torch.bfloat16))
                B = nn.Parameter(torch.zeros(n_sel, model_dim, rank, dtype=torch.bfloat16))
                nn.init.kaiming_uniform_(A, a=math.sqrt(5))
                # Per-loop adapter scalar (1 + s) AND per-loop base output scalar (beta * y)
                s_lora = nn.Parameter(torch.zeros(n_sel, n_loops, dtype=torch.bfloat16))
                s_base = nn.Parameter(torch.zeros(n_sel, n_loops, dtype=torch.bfloat16))
                self.params[f"L{lid}_A"] = A
                self.params[f"L{lid}_B"] = B
                self.params[f"L{lid}_slora"] = s_lora
                self.params[f"L{lid}_sbase"] = s_base
            else:
                raise ValueError(f"Unknown mode: {mode}")

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def apply_layer_delta(
        self,
        lid: int,
        visit_r: int,
        hidden: torch.Tensor,
        y: torch.Tensor,
        group_sizes: list[int],
    ) -> torch.Tensor:
        if lid not in self.by_layer or visit_r >= self.n_loops:
            return y
        e_list = self.by_layer[lid]
        A = self.params[f"L{lid}_A"]
        B = self.params[f"L{lid}_B"]
        # Precompute offsets in group_sizes
        offsets = []
        acc = 0
        for g in group_sizes:
            offsets.append(acc)
            acc += g

        delta_y = torch.zeros_like(y)
        any_hit = False
        if self.mode == "per_loop":
            for pos_i, e_idx in enumerate(e_list):
                cnt = group_sizes[e_idx]
                if cnt == 0:
                    continue
                any_hit = True
                s = offsets[e_idx]
                h_e = hidden[s : s + cnt]  # [cnt, I]
                mid = F.linear(h_e, A[pos_i, visit_r])  # [cnt, rank]
                dy = F.linear(mid, B[pos_i, visit_r]) * self.scale  # [cnt, D]
                delta_y[s : s + cnt] = dy
        else:
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
                dy = (1.0 + s_lora[pos_i, visit_r]) * F.linear(mid, B[pos_i]) * self.scale
                dy = dy + s_base[pos_i, visit_r] * y[s : s + cnt]
                delta_y[s : s + cnt] = dy

        return y + delta_y if any_hit else y


def allocate_conflict_ranks(
    expert_records: list[dict[str, Any]],
    total_rank_budget: int,
    *,
    score_key: str = "s_orth_both",
    power: float = 0.5,
) -> dict[tuple[int, int], int]:
    """Allocate integer per-loop LoRA ranks summing exactly to total_rank_budget proportional to conflict^power."""
    weights = np.array([max(float(r[score_key]), 0.0) ** power for r in expert_records], dtype=np.float64)
    if weights.sum() <= 1e-12:
        weights = np.ones_like(weights)
    raw = weights / weights.sum() * total_rank_budget
    floors = np.floor(raw).astype(int)
    rem = int(total_rank_budget - floors.sum())
    if rem > 0:
        frac_idx = np.argsort(-(raw - floors))
        for i in frac_idx[:rem]:
            floors[i] += 1
    out: dict[tuple[int, int], int] = {}
    for r, rk in zip(expert_records, floors.tolist()):
        if rk > 0:
            out[(r["layer"], r["expert"])] = int(rk)
    return out


class VariableRankExpertLoRAHost(nn.Module):
    """Per-loop W2 LoRA with expert-specific integer ranks allocated by conflict score (rsLoRA scaled)."""

    def __init__(
        self,
        expert_ranks: dict[tuple[int, int], int],
        *,
        n_loops: int,
        model_dim: int,
        inter_dim: int,
        alpha: float = 6.0,
    ) -> None:
        super().__init__()
        self.n_loops = n_loops
        self.alpha = alpha
        self.by_layer: dict[int, list[tuple[int, int]]] = {}
        for (lid, e), rk in sorted(expert_ranks.items()):
            if rk > 0:
                self.by_layer.setdefault(lid, []).append((e, rk))

        self.params = nn.ParameterDict()
        for lid, pairs in sorted(self.by_layer.items()):
            for e_idx, rk in pairs:
                A = nn.Parameter(torch.empty(n_loops, rk, inter_dim, dtype=torch.bfloat16))
                B = nn.Parameter(torch.zeros(n_loops, model_dim, rk, dtype=torch.bfloat16))
                nn.init.kaiming_uniform_(A, a=math.sqrt(5))
                self.params[f"L{lid}_E{e_idx}_A"] = A
                self.params[f"L{lid}_E{e_idx}_B"] = B

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def apply_layer_delta(
        self,
        lid: int,
        visit_r: int,
        hidden: torch.Tensor,
        y: torch.Tensor,
        group_sizes: list[int],
    ) -> torch.Tensor:
        if lid not in self.by_layer or visit_r >= self.n_loops:
            return y
        offsets = []
        acc = 0
        for g in group_sizes:
            offsets.append(acc)
            acc += g
        delta_y = torch.zeros_like(y)
        any_hit = False
        for e_idx, rk in self.by_layer[lid]:
            cnt = group_sizes[e_idx]
            if cnt == 0:
                continue
            any_hit = True
            s = offsets[e_idx]
            h_e = hidden[s : s + cnt]
            A = self.params[f"L{lid}_E{e_idx}_A"][visit_r]
            B = self.params[f"L{lid}_E{e_idx}_B"][visit_r]
            scale = self.alpha / math.sqrt(max(rk, 1))
            delta_y[s : s + cnt] = F.linear(F.linear(h_e, A), B) * scale
        return y + delta_y if any_hit else y


class RouterLoopModHost(nn.Module):
    """Freeze all experts; train per-loop router weight deltas + per-loop input modulation on tracked layers."""

    def __init__(
        self,
        tracked_layers: list[int],
        *,
        n_loops: int,
        n_experts: int,
        model_dim: int,
    ) -> None:
        super().__init__()
        self.tracked_layers = tracked_layers
        self.n_loops = n_loops
        self.params = nn.ParameterDict()
        for lid in tracked_layers:
            self.params[f"L{lid}_dw_router"] = nn.Parameter(
                torch.zeros(n_loops, n_experts, model_dim, dtype=torch.bfloat16)
            )
            self.params[f"L{lid}_gamma"] = nn.Parameter(
                torch.zeros(n_loops, model_dim, dtype=torch.bfloat16)
            )
            self.params[f"L{lid}_beta"] = nn.Parameter(
                torch.zeros(n_loops, model_dim, dtype=torch.bfloat16)
            )

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_and_eval_condition(
    model,
    condition_name: str,
    adapter_host: nn.Module,
    *,
    tracked_layers: list[int],
    n_loops: int,
    train_seqs: list[dict[str, Any]],
    eval_seqs: list[dict[str, Any]],
    lr: float,
    grad_accum: int,
    device: torch.device,
) -> dict[str, Any]:
    """Train one matched-budget adapter on adapt_train and evaluate on held-out eval_heldout."""
    from xllm.modules.fused_ops import memory_efficient_dropout, mgmm
    from xllm.modules.moe.router import TopKRouting

    adapter_host = adapter_host.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in adapter_host.parameters():
        p.requires_grad_(True)

    visit_cursor: dict[int, int] = {lid: 0 for lid in tracked_layers}
    orig_expert_fwds: dict[int, Any] = {}
    orig_router_fwds: dict[int, Any] = {}

    def _reset_visits() -> None:
        for lid in tracked_layers:
            visit_cursor[lid] = 0

    try:
        if isinstance(adapter_host, (ExpertLoRAHost, VariableRankExpertLoRAHost)):
            for lid in adapter_host.by_layer.keys():
                exp_mod = model.layers[lid].moe.experts
                orig_expert_fwds[lid] = exp_mod.forward

                def _make_exp_fwd(l_id: int, mod):
                    def _fwd(x: torch.Tensor, group_sizes: list[int]) -> torch.Tensor:
                        r = visit_cursor[l_id]
                        visit_cursor[l_id] = r + 1
                        w1 = rearrange(mod.weight1, "(n s) d -> n s d", n=mod.n_local_experts)
                        w2 = rearrange(mod.weight2, "(n s) d -> n s d", n=mod.n_local_experts)
                        w3 = rearrange(mod.weight3, "(n s) d -> n s d", n=mod.n_local_experts)
                        h1 = F.silu(mgmm(x, w1, group_sizes, True, mod.backend))
                        h3 = mgmm(x, w3, group_sizes, True, mod.backend)
                        hidden = memory_efficient_dropout(h1 * h3, mod.hidden_dropout, mod.training)
                        y = mgmm(hidden, w2, group_sizes, True, mod.backend)
                        return adapter_host.apply_layer_delta(l_id, r, hidden, y, group_sizes)
                    return _fwd

                exp_mod.forward = _make_exp_fwd(lid, exp_mod)

        elif isinstance(adapter_host, RouterLoopModHost):
            for lid in adapter_host.tracked_layers:
                rtr = model.layers[lid].moe.router
                orig_router_fwds[lid] = rtr.forward

                def _make_rtr_fwd(l_id: int, mod):
                    def _fwd(x: torch.Tensor, load_balancing_type=None):
                        r = visit_cursor[l_id]
                        visit_cursor[l_id] = r + 1
                        if r < adapter_host.n_loops:
                            gamma = adapter_host.params[f"L{l_id}_gamma"][r]
                            beta = adapter_host.params[f"L{l_id}_beta"][r]
                            x_mod = x * (1.0 + gamma) + beta
                            w_eff = mod.weight + adapter_host.params[f"L{l_id}_dw_router"][r]
                        else:
                            x_mod = x
                            w_eff = mod.weight
                        return TopKRouting.apply(
                            x_mod,
                            w_eff,
                            mod.num_experts,
                            mod.topk,
                            mod.score_func,
                            mod.bias,
                            mod.bias_update_rate,
                            mod.scaling_factor,
                            load_balancing_type,
                        )
                    return _fwd

                rtr.forward = _make_rtr_fwd(lid, rtr)

        optimizer = torch.optim.AdamW(adapter_host.parameters(), lr=lr, weight_decay=0.0)
        n_opt_steps = max(1, (len(train_seqs) + grad_accum - 1) // grad_accum)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_opt_steps, eta_min=lr * 0.1)

        train_losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        for i, item in enumerate(train_seqs, 1):
            _reset_visits()
            loss = compute_lm_loss(model, item["tokens"], device)
            train_losses.append(float(loss.item()))
            (loss / grad_accum).backward()
            if i % grad_accum == 0 or i == len(train_seqs):
                torch.nn.utils.clip_grad_norm_(adapter_host.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        # Evaluate on held-out split (native router!)
        eval_nlls: list[float] = []
        eval_text: list[float] = []
        eval_code: list[float] = []
        with torch.inference_mode():
            for item in eval_seqs:
                _reset_visits()
                val = float(compute_lm_loss(model, item["tokens"], device).item())
                eval_nlls.append(val)
                if item["domain"] == "code":
                    eval_code.append(val)
                else:
                    eval_text.append(val)

        return {
            "condition": condition_name,
            "trainable_params": adapter_host.num_trainable_params(),
            "train_loss_first8": float(np.mean(train_losses[:8])),
            "train_loss_last8": float(np.mean(train_losses[-8:])),
            "eval_nll": float(np.mean(eval_nlls)),
            "eval_nll_text": float(np.mean(eval_text)) if eval_text else 0.0,
            "eval_nll_code": float(np.mean(eval_code)) if eval_code else 0.0,
            "eval_nll_per_seq": eval_nlls,
        }
    finally:
        for lid, fn in orig_expert_fwds.items():
            model.layers[lid].moe.experts.forward = fn
        for lid, fn in orig_router_fwds.items():
            model.layers[lid].moe.router.forward = fn
        del adapter_host
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Full Model Pipeline (runs cleanly for both Ouro MoE and Huginn MoE!)
# ---------------------------------------------------------------------------

def run_single_model_pilot(
    model_key: str,
    model_spec: dict[str, Any],
    *,
    model_dir: Path,
    dataset_dir: Path,
    xllm_loop_dir: Path,
    cfg: dict[str, Any],
    device: torch.device,
    run_obj: Any,
) -> dict[str, Any]:
    artifact_dir = model_dir / model_spec["local_dir"]
    tracked_layers: list[int] = list(model_spec["tracked_layers"])
    n_loops = int(model_spec["n_loops"])
    k_per_layer = int(model_spec["k_per_layer"])
    rank_selected = int(model_spec["rank_selected"])
    rank_recon = int(model_spec["rank_recon"])
    seed = int(cfg.get("seed", 20261124))

    print(f"\n[idea2:{model_key}] Loading {artifact_dir} ...")
    model, tokenizer, _mcfg = load_ifm_model_for_autograd(artifact_dir, device=str(device), attention_backend="sdpa")

    n_experts = int(model.layers[tracked_layers[0]].moe.experts.n_experts)
    model_dim = int(model.layers[tracked_layers[0]].moe.experts.model_dim)
    inter_dim = int(model.layers[tracked_layers[0]].moe.experts.inter_dim)

    # Disjoint text + code splits (zero document overlap!)
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
    print(
        f"[idea2:{model_key}] Disjoint splits: "
        + ", ".join(f"{k}={len(v)}" for k, v in splits.items())
    )

    # 0) Verify exact backward + per-visit decomposition
    with run_obj.phase(f"{model_key}_verify", total=1, unit="check") as pv:
        verify_res = verify_backward_and_decomp(
            model,
            tracked_layers[0],
            n_loops,
            n_experts,
            model_dim,
            inter_dim,
            splits["calib_a"][0]["tokens"],
            device,
        )
        pv.step(1)

    # Freeze all base params
    for p in model.parameters():
        p.requires_grad_(False)

    # Base held-out NLL (frozen)
    base_eval_per_seq = evaluate_nll_per_seq(model, splits["eval_heldout"], device)
    base_eval_nll = float(np.mean(base_eval_per_seq))
    base_text = float(np.mean([x for x, it in zip(base_eval_per_seq, splits["eval_heldout"]) if it["domain"] == "text"]))
    base_code = float(np.mean([x for x, it in zip(base_eval_per_seq, splits["eval_heldout"]) if it["domain"] == "code"]))
    print(f"[idea2:{model_key}] Frozen base held-out NLL = {base_eval_nll:.4f} (text={base_text:.4f}, code={base_code:.4f})")

    # Snapshot base W2 weights on CPU (FP32) for orthogonal projection
    w2_base_cpu = {
        lid: model.layers[lid].moe.experts.weight2.detach().float().cpu().reshape(n_experts, model_dim, inter_dim)
        for lid in tracked_layers
    }

    # 1) Calibration: capture Split A and Split B per-visit W2 gradients independently
    with run_obj.phase(f"{model_key}_calib", total=3, unit="splits") as pc:
        with per_visit_w2_grad_collector(
            model, tracked_layers, n_loops, n_experts, model_dim, inter_dim, device
        ) as (grads_cpu, counts, accum_fn, _reset):
            loss_ca = accum_fn(splits["calib_a"], clear_first=True)
            grads_a = {lid: t.clone() for lid, t in grads_cpu.items()}
            counts_a = {lid: c.copy() for lid, c in counts.items()}
            pc.step(1)

            loss_cb = accum_fn(splits["calib_b"], clear_first=True)
            grads_b = {lid: t.clone() for lid, t in grads_cpu.items()}
            counts_b = {lid: c.copy() for lid, c in counts.items()}
            pc.step(2)

            # Also capture adapt_train gradients (first 16 seqs) for Exp 2A single-expert counterfactuals
            n_cf_tr = min(int(cfg.get("n_cf_train_seqs", 16)), len(splits["adapt_train"]))
            _ = accum_fn(splits["adapt_train"][:n_cf_tr], clear_first=True)
            grads_tr_cf = {lid: t.clone() for lid, t in grads_cpu.items()}
            pc.step(3)

    expert_records = score_expert_conflicts(grads_a, grads_b, counts_a, counts_b, w2_base_cpu)
    del grads_a, grads_b, w2_base_cpu
    gc.collect()

    # Concentration & reproducibility summary
    s_ob = np.array([r["s_orth_both"] for r in expert_records])
    s_pos = np.clip(s_ob, 0.0, None)
    s_pos_sorted = np.sort(s_pos)[::-1]
    n_tot_exp = len(expert_records)
    top20_share = float(s_pos_sorted[: max(1, n_tot_exp // 5)].sum() / max(s_pos_sorted.sum(), 1e-12))
    top10_share = float(s_pos_sorted[: max(1, n_tot_exp // 10)].sum() / max(s_pos_sorted.sum(), 1e-12))
    calib_summary = {
        "calib_a_loss": loss_ca,
        "calib_b_loss": loss_cb,
        "n_experts_scored": n_tot_exp,
        "frac_positive_s_orth_both": float(np.mean(s_ob > 0)),
        "mean_cross_split_dir_cos": float(np.mean([r["cross_split_dir_cos"] for r in expert_records])),
        "median_cross_split_dir_cos": float(np.median([r["cross_split_dir_cos"] for r in expert_records])),
        "mean_same_loop_cos": float(np.mean([r["same_loop_cos"] for r in expert_records])),
        "mean_diff_loop_cos": float(np.mean([r["diff_loop_cos"] for r in expert_records])),
        "top10pct_conflict_share": top10_share,
        "top20pct_conflict_share": top20_share,
    }
    print(f"[idea2:{model_key}] Calib summary: {json.dumps(calib_summary)}")

    # 2) Exp 2A: Single-expert fixed-norm counterfactual scatter
    n_cf_eval = min(int(cfg.get("n_cf_eval_seqs", 16)), len(splits["eval_heldout"]))
    cf_eval_seqs = splits["eval_heldout"][:n_cf_eval]
    cf_base_nll = float(np.mean(base_eval_per_seq[:n_cf_eval]))
    cf_per_layer = int(model_spec.get("cf_experts_per_layer", 5))
    with run_obj.phase(f"{model_key}_exp2a", total=len(tracked_layers) * cf_per_layer, unit="experts") as p2a:
        exp2a_out = run_single_expert_counterfactuals(
            model,
            expert_records,
            grads_tr_cf,
            cf_eval_seqs,
            cf_base_nll,
            n_loops=n_loops,
            model_dim=model_dim,
            inter_dim=inter_dim,
            step_norm_eps=float(cfg.get("cf_step_norm_eps", 0.08)),
            experts_per_layer=cf_per_layer,
            device=device,
            run_phase=p2a,
        )
    del grads_tr_cf
    gc.collect()
    print(f"[idea2:{model_key}] Exp 2A scatter summary: {json.dumps(exp2a_out['summary'])}")

    # 3) Exp 2B: Strictly Parameter-Matched Group Adaptation
    groups = select_group_experts(expert_records, k_per_layer=k_per_layer, score_key="s_orth_both", seed=seed)
    lr = float(cfg.get("adapt_lr", 3e-4))
    grad_accum = int(cfg.get("adapt_grad_accum", 2))
    alpha = float(cfg.get("lora_alpha", 16.0))

    # Build ordered list of matched-budget conditions
    # Parameter budget check:
    #   selected: (len(tracked_layers)*k_per_layer) * n_loops * rank_selected
    #   uniform_all: (len(tracked_layers)*n_experts) * n_loops * 1
    #   shared_plus_scalar: (len(tracked_layers)*k_per_layer) * (n_loops * rank_selected) + scalars
    #   recon_layer: n_experts * n_loops * rank_recon
    total_rank_per_loop = len(tracked_layers) * k_per_layer * rank_selected
    adaptive_ranks = allocate_conflict_ranks(expert_records, total_rank_per_loop, score_key="s_orth_both", power=0.5)
    print(
        f"[idea2:{model_key}] conflict_adaptive_rank: {len(adaptive_ranks)}/{len(expert_records)} active experts, "
        f"sum_rank={sum(adaptive_ranks.values())}, max_rank={max(adaptive_ranks.values())}"
    )

    cond_specs: list[tuple[str, nn.Module]] = [
        ("conflict_top", ExpertLoRAHost(groups["conflict_top"], mode="per_loop", rank=rank_selected, n_loops=n_loops, model_dim=model_dim, inter_dim=inter_dim, alpha=alpha)),
        ("conflict_adaptive_rank", VariableRankExpertLoRAHost(adaptive_ranks, n_loops=n_loops, model_dim=model_dim, inter_dim=inter_dim, alpha=alpha)),
        ("matched_high", ExpertLoRAHost(groups["matched_high"], mode="per_loop", rank=rank_selected, n_loops=n_loops, model_dim=model_dim, inter_dim=inter_dim, alpha=alpha)),
        ("matched_low", ExpertLoRAHost(groups["matched_low"], mode="per_loop", rank=rank_selected, n_loops=n_loops, model_dim=model_dim, inter_dim=inter_dim, alpha=alpha)),
        ("random", ExpertLoRAHost(groups["random"], mode="per_loop", rank=rank_selected, n_loops=n_loops, model_dim=model_dim, inter_dim=inter_dim, alpha=alpha)),
        ("gradnorm_top", ExpertLoRAHost(groups["gradnorm_top"], mode="per_loop", rank=rank_selected, n_loops=n_loops, model_dim=model_dim, inter_dim=inter_dim, alpha=alpha)),
        ("freq_top", ExpertLoRAHost(groups["freq_top"], mode="per_loop", rank=rank_selected, n_loops=n_loops, model_dim=model_dim, inter_dim=inter_dim, alpha=alpha)),
        ("shared_plus_scalar", ExpertLoRAHost(groups["conflict_top"], mode="shared_plus_scalar", rank=rank_selected * n_loops, n_loops=n_loops, model_dim=model_dim, inter_dim=inter_dim, alpha=alpha)),
        ("uniform_all", ExpertLoRAHost(groups["all_experts"], mode="per_loop", rank=1, n_loops=n_loops, model_dim=model_dim, inter_dim=inter_dim, alpha=alpha)),
        ("recon_layer", ExpertLoRAHost(groups["recon_layer"], mode="per_loop", rank=rank_recon, n_loops=n_loops, model_dim=model_dim, inter_dim=inter_dim, alpha=alpha)),
        ("router_loop_mod", RouterLoopModHost(tracked_layers, n_loops=n_loops, n_experts=n_experts, model_dim=model_dim)),
    ]

    cond_results: dict[str, dict[str, Any]] = {
        "base_frozen": {
            "condition": "base_frozen",
            "trainable_params": 0,
            "eval_nll": base_eval_nll,
            "eval_nll_text": base_text,
            "eval_nll_code": base_code,
            "delta_nll_vs_base": 0.0,
            "eval_nll_per_seq": base_eval_per_seq,
        }
    }

    with run_obj.phase(f"{model_key}_exp2b", total=len(cond_specs), unit="conds") as p2b:
        for c_idx, (cname, host) in enumerate(cond_specs, 1):
            torch.manual_seed(seed)
            res = train_and_eval_condition(
                model,
                cname,
                host,
                tracked_layers=tracked_layers,
                n_loops=n_loops,
                train_seqs=splits["adapt_train"],
                eval_seqs=splits["eval_heldout"],
                lr=lr,
                grad_accum=grad_accum,
                device=device,
            )
            gain_vs_base, ci_lo, ci_hi = paired_bootstrap_ci(
                np.array(base_eval_per_seq) - np.array(res["eval_nll_per_seq"])
            )
            res["delta_nll_vs_base"] = gain_vs_base
            res["delta_nll_ci95"] = [ci_lo, ci_hi]
            cond_results[cname] = res
            print(
                f"[idea2:{model_key}] {cname:20s} params={res['trainable_params']:9d} "
                f"eval_nll={res['eval_nll']:.4f} Δvs_base={gain_vs_base:+.5f} [{ci_lo:+.5f}, {ci_hi:+.5f}]"
            )
            p2b.row(
                f"{model_key}_conds",
                condition=cname,
                eval_nll=res["eval_nll"],
                delta_nll=gain_vs_base,
                params=res["trainable_params"],
            )
            p2b.step(c_idx)

    # Paired bootstrap contrasts for the decisive scientific questions!
    def _contrast(better_name: str, ref_name: str) -> dict[str, Any]:
        a = np.array(cond_results[ref_name]["eval_nll_per_seq"])
        b = np.array(cond_results[better_name]["eval_nll_per_seq"])
        mean_d, lo, hi = paired_bootstrap_ci(a - b)  # positive = better_name has LOWER NLL than ref_name
        return {"mean_nll_advantage": mean_d, "ci95": [lo, hi], "wins_seqs": int(np.sum(a > b)), "n_seqs": int(a.size)}

    contrasts = {
        "conflict_top_vs_random": _contrast("conflict_top", "random"),
        "matched_high_vs_matched_low": _contrast("matched_high", "matched_low"),
        "conflict_top_vs_shared_plus_scalar": _contrast("conflict_top", "shared_plus_scalar"),
        "conflict_top_vs_uniform_all": _contrast("conflict_top", "uniform_all"),
        "adaptive_vs_uniform_all": _contrast("conflict_adaptive_rank", "uniform_all"),
        "adaptive_vs_conflict_top": _contrast("conflict_adaptive_rank", "conflict_top"),
        "conflict_top_vs_gradnorm_top": _contrast("conflict_top", "gradnorm_top"),
        "conflict_top_vs_freq_top": _contrast("conflict_top", "freq_top"),
        "conflict_top_vs_recon_layer": _contrast("conflict_top", "recon_layer"),
        "conflict_top_vs_router_loop_mod": _contrast("conflict_top", "router_loop_mod"),
    }

    # Covariate balance audit for matched_high vs matched_low
    rec_map = {(r["layer"], r["expert"]): r for r in expert_records}
    mh_recs = [rec_map[k] for k in groups["matched_high"]]
    ml_recs = [rec_map[k] for k in groups["matched_low"]]
    match_audit = {
        "matched_high_mean_tokens": float(np.mean([r["token_count"] for r in mh_recs])),
        "matched_low_mean_tokens": float(np.mean([r["token_count"] for r in ml_recs])),
        "matched_high_mean_gradnorm": float(np.mean([r["grad_norm_mean"] for r in mh_recs])),
        "matched_low_mean_gradnorm": float(np.mean([r["grad_norm_mean"] for r in ml_recs])),
        "matched_high_mean_s_orth_both": float(np.mean([r["s_orth_both"] for r in mh_recs])),
        "matched_low_mean_s_orth_both": float(np.mean([r["s_orth_both"] for r in ml_recs])),
    }

    # Strip verbose per-seq arrays before returning summary
    cond_clean = {k: {ck: cv for ck, cv in v.items() if ck != "eval_nll_per_seq"} for k, v in cond_results.items()}

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model_key": model_key,
        "artifact_dir": str(artifact_dir),
        "tracked_layers": tracked_layers,
        "n_loops": n_loops,
        "verify": verify_res,
        "calib_summary": calib_summary,
        "match_audit": match_audit,
        "exp2a_summary": exp2a_out["summary"],
        "exp2a_rows": exp2a_out["rows"],
        "exp2b_conditions": cond_clean,
        "exp2b_contrasts": contrasts,
        "expert_records": expert_records,
    }


def render_report_md(full_summary: dict[str, Any]) -> str:
    lines = ["# Direction 2 Feasibility Report: Expert Cross-Loop Unbinding (IFM Looped MoE)", ""]
    lines.append(f"- **Elapsed**: `{full_summary['elapsed_sec']:.1f}s` on GPU `{full_summary['gpu_index']}` ({full_summary['gpu_name']})")
    lines.append("")

    for mkey, mdata in full_summary["models"].items():
        cs = mdata["calib_summary"]
        e2a = mdata["exp2a_summary"]
        ma = mdata["match_audit"]
        lines.append(f"## Model: `{mkey}` (layers={mdata['tracked_layers']}, loops={mdata['n_loops']})")
        lines.append(
            f"- **Backward Exactness**: rel L2 error = `{mdata['verify']['relative_l2_error']:.3e}`"
        )
        lines.append(
            f"- **Cross-Split Conflict Calibration** (n={cs['n_experts_scored']} experts): "
            f"positive S_orth_both = `{cs['frac_positive_s_orth_both']*100:.1f}%`, "
            f"mean cross-split dir cos = `{cs['mean_cross_split_dir_cos']:+.3f}` (median `{cs['median_cross_split_dir_cos']:+.3f}`), "
            f"same-loop cos = `{cs['mean_same_loop_cos']:+.3f}` vs diff-loop cos = `{cs['mean_diff_loop_cos']:+.3f}`, "
            f"top-10% share = `{cs['top10pct_conflict_share']*100:.1f}%`, top-20% share = `{cs['top20pct_conflict_share']*100:.1f}%`"
        )
        lines.append(
            f"- **Exp 2A Single-Expert Scatter** (n={e2a['n_experts_tested']}): "
            f"mean unbinding gain = `{e2a['mean_unbinding_gain']:+.6f}` ({e2a['frac_positive_unbinding_gain']*100:.1f}% positive), "
            f"Spearman(S_orth_both, gain) = `{e2a['spearman_s_orth_both_vs_gain']:+.3f}`, "
            f"Pearson = `{e2a['pearson_s_orth_both_vs_gain']:+.3f}`, "
            f"Spearman(S_raw) = `{e2a['spearman_s_raw_vs_gain']:+.3f}`, "
            f"Spearman(gradnorm) = `{e2a['spearman_gradnorm_vs_gain']:+.3f}`, "
            f"Spearman(freq) = `{e2a['spearman_freq_vs_gain']:+.3f}`"
        )
        lines.append(
            f"- **Matched-Group Audit**: high vs low tokens=`{ma['matched_high_mean_tokens']:.0f}` vs `{ma['matched_low_mean_tokens']:.0f}`, "
            f"gradnorm=`{ma['matched_high_mean_gradnorm']:.4f}` vs `{ma['matched_low_mean_gradnorm']:.4f}`, "
            f"S_orth_both=`{ma['matched_high_mean_s_orth_both']:.3e}` vs `{ma['matched_low_mean_s_orth_both']:.3e}`"
        )
        lines.append("")
        lines.append("### Exp 2B: Parameter-Matched Held-Out Group Adaptation")
        lines.append("| Condition | Trainable Params | Eval NLL (↓) | Δ NLL vs Base (↑) | 95% CI | Text NLL | Code NLL |")
        lines.append("|---|---:|---:|---:|---|---:|---:|")
        order = [
            "base_frozen", "conflict_top", "conflict_adaptive_rank", "matched_high", "matched_low", "random",
            "gradnorm_top", "freq_top", "shared_plus_scalar", "uniform_all",
            "recon_layer", "router_loop_mod",
        ]
        for cname in order:
            if cname not in mdata["exp2b_conditions"]:
                continue
            c = mdata["exp2b_conditions"][cname]
            ci = c.get("delta_nll_ci95")
            ci_str = f"[{ci[0]:+.5f}, {ci[1]:+.5f}]" if ci else "—"
            lines.append(
                f"| `{cname}` | {c['trainable_params']:,} | {c['eval_nll']:.4f} | {c['delta_nll_vs_base']:+.5f} | {ci_str} | {c['eval_nll_text']:.4f} | {c['eval_nll_code']:.4f} |"
            )
        lines.append("")
        lines.append("### Key Paired Contrasts (positive = first wins)")
        for ck, cv in mdata["exp2b_contrasts"].items():
            lines.append(
                f"- `{ck}`: `{cv['mean_nll_advantage']:+.5f}` 95% CI `[{cv['ci95'][0]:+.5f}, {cv['ci95'][1]:+.5f}]` ({cv['wins_seqs']}/{cv['n_seqs']} seqs)"
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

    gpu_idx, gpu_lock, gpu_info = acquire_idle_gpu(cfg.get("preferred_gpu"))
    try:
        device = torch.device(f"cuda:{gpu_idx}")
        torch.cuda.set_device(device)
        print(f"[idea2] Acquired idle GPU {gpu_idx}: {gpu_info.name} (mem={gpu_info.mem_used_mib}MiB, util={gpu_info.util_pct}%)")

        boot_info = bootstrap_xllm(args.xllm_loop_dir, attention_backend="sdpa")
        print(f"[idea2] xllm bootstrap: {json.dumps(boot_info)}")

        models_cfg: dict[str, dict[str, Any]] = cfg["models"]
        plan_stages: list[dict[str, Any]] = [
            {"phase": "setup", "of": 1, "unit": "env"},
            {"phase": "download", "of": 1, "unit": "assets"},
        ]
        for mkey, mspec in models_cfg.items():
            plan_stages.extend(
                [
                    {"phase": f"{mkey}_verify", "of": 1, "unit": "check"},
                    {"phase": f"{mkey}_calib", "of": 3, "unit": "splits"},
                    {"phase": f"{mkey}_exp2a", "of": len(mspec["tracked_layers"]) * int(mspec.get("cf_experts_per_layer", 5)), "unit": "experts"},
                    {"phase": f"{mkey}_exp2b", "of": 11, "unit": "conds"},
                ]
            )
        plan_stages.append({"phase": "report", "of": 1, "unit": "file"})
        orx.plan(plan_stages)

        run_obj = orx.progress()
        model_outputs: dict[str, Any] = {}
        for mkey, mspec in models_cfg.items():
            m_res = run_single_model_pilot(
                mkey,
                mspec,
                model_dir=Path(args.model_dir),
                dataset_dir=Path(args.dataset_dir),
                xllm_loop_dir=Path(args.xllm_loop_dir),
                cfg=cfg,
                device=device,
                run_obj=run_obj,
            )
            (out_dir / f"experts_{mkey}.json").write_text(json.dumps(m_res["expert_records"], indent=2))
            (out_dir / f"exp2a_{mkey}.json").write_text(json.dumps(m_res["exp2a_rows"], indent=2))
            model_outputs[mkey] = {k: v for k, v in m_res.items() if k not in ("expert_records", "exp2a_rows")}

        with run_obj.phase("report", total=1, unit="file") as prep:
            full_summary = {
                "idea": "idea2_expert_unbinding",
                "gpu_index": gpu_idx,
                "gpu_name": gpu_info.name,
                "elapsed_sec": time.time() - t0,
                "config": cfg,
                "models": model_outputs,
            }
            (out_dir / "summary_idea2.json").write_text(json.dumps(full_summary, indent=2, ensure_ascii=False))
            report_md = render_report_md(full_summary)
            (out_dir / "report_idea2.md").write_text(report_md)

            metrics: dict[str, float] = {}
            for mkey, md in model_outputs.items():
                metrics[f"{mkey}_dir_cos"] = md["calib_summary"]["mean_cross_split_dir_cos"]
                metrics[f"{mkey}_top20_share"] = md["calib_summary"]["top20pct_conflict_share"]
                metrics[f"{mkey}_exp2a_spearman"] = md["exp2a_summary"]["spearman_s_orth_both_vs_gain"]
                metrics[f"{mkey}_vs_rand"] = md["exp2b_contrasts"]["conflict_top_vs_random"]["mean_nll_advantage"]
                metrics[f"{mkey}_hi_vs_lo"] = md["exp2b_contrasts"]["matched_high_vs_matched_low"]["mean_nll_advantage"]
                metrics[f"{mkey}_vs_shared"] = md["exp2b_contrasts"]["conflict_top_vs_shared_plus_scalar"]["mean_nll_advantage"]
                metrics[f"{mkey}_vs_unif"] = md["exp2b_contrasts"]["conflict_top_vs_uniform_all"]["mean_nll_advantage"]
                metrics[f"{mkey}_vs_router"] = md["exp2b_contrasts"]["conflict_top_vs_router_loop_mod"]["mean_nll_advantage"]
            prep.step(1, **metrics)

            print("========== REPORT (report_idea2.md) ==========")
            print(report_md)
            print("========== END REPORT ==========")
    finally:
        gpu_lock.release()


if __name__ == "__main__":
    main()
