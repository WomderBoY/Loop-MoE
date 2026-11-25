"""Round 3: LoopSched-MoE v2 — Orthogonal Shrinkage, Stage×Layer Scheduling & Non-Additive IterGreedy.

Direct child of R2-A (d65f0a7c-9ad5-4518-b275-d4d3a4103fcd).
Validates the complete ICML 2027 LoopSched-MoE framework on IFM Ouro-MoE & IFM Huginn-MoE:
  1. Pillar 1 — Orthogonal Expert Shrinkage (NoRen & Stage-Pooled gamma_s*):
     Measures empirical cos(e1, e2), theoretical MSE_renorm / MSE_noren = 2(1 - rho), and
     low-variance stage-pooled optimal top-1 scale gamma_s*.
  2. Pillar 2 — Pure k in {1, 2} Zero-Overhead Boundary-Protected Static Schedules (allow_k0=False):
     Fixes R2-A's additive k=0 DP cascade completely! Evaluates matched -25%, -37.5%, and
     natural boundary-protected cuts (-43.8% Ouro, -46.4% Huginn: keeping ONLY coda at k=2)
     against Layer-Only-Static (proving loop-stage info drives real gains) and dynamic baselines.
  3. Pillar 3 — Low-Variance Factorized Stage × Layer Dynamic Scheduling (k in {1, 2}, NO k=0):
     Estimates noise-free stage-joint sensitivity S_stage(r) and physical layer-joint sensitivity
     L_layer(l), combining I_alpha(l, r) = S_stage(r)^alpha * L_layer(l)^(1-alpha) with alpha
     selected on strictly disjoint calib_b (zero eval leakage!).
  4. Pillar 4 — Non-Additive Iterative Greedy k=0 Swap/Drop (LoopSched-IterGreedy):
     Breaks the -50% all-top-1 barrier (swapping high-value boundary k=1->2 for sparse middle
     k=1->0 ONLY when BOTH calib_a and disjoint calib_b cumulative NLL improve) and cleanly
     extends to -62.5% ultra-sparse budgets without nonlinear cascade.
  5. Real Hardware Wall-Clock Speedup Benchmark (RTX PRO 5000):
     Measures BOTH unmodified xllm sparse dispatch (Prefill B=8 L=256, Decode B=1/8/32 L=1)
     AND a verified zero-sync small-batch BMM decode kernel (apples-to-apples fast_bf16_base vs
     fast_LoopSched at B=1 and B=8, isolating true HBM bandwidth speedup for k=1 and k=0!).
  6. Bonus Orthogonality Check:
     Composes 5.1x W3-g128 Trajectory-Pooled AWQ with LoopSched-Static-k12 simultaneously.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from loop_research import orx_run as orx
from loop_research.data_assets import (
    build_disjoint_text_code_splits,
    load_arc_challenge_samples,
    load_arc_easy_samples,
    load_openbookqa_samples,
)
from loop_research.gpu_utils import acquire_idle_gpu
from loop_research.ifm_runtime import bootstrap_xllm, load_ifm_model_for_autograd


# ---------------------------------------------------------------------------
# 1. Exact Visit Topology & Pure-k12 / Factorized Importance Helpers
# ---------------------------------------------------------------------------

def build_visit_topology(
    n_layers: int,
    loop_start: int,
    loop_end: int,
    n_loops: int,
) -> list[dict[str, Any]]:
    """Ordered logical visits matching xLLM convention: recurrent layers = [loop_start, loop_end)."""
    visits: list[dict[str, Any]] = []
    for lid in range(0, loop_start):
        visits.append({"vid": len(visits), "layer_id": lid, "loop_idx": 0, "stage": "prelude", "is_rec": False})
    rec_layers = list(range(loop_start, loop_end))
    for r in range(n_loops):
        for lid in rec_layers:
            visits.append({"vid": len(visits), "layer_id": lid, "loop_idx": r, "stage": f"loop_{r}", "is_rec": True})
    for lid in range(loop_end, n_layers):
        visits.append({"vid": len(visits), "layer_id": lid, "loop_idx": 0, "stage": "coda", "is_rec": False})
    return visits


def compute_factorized_importance(
    topo: list[dict[str, Any]],
    stage_per_visit_sens: dict[str, float],
    layer_per_visit_sens: np.ndarray,
    alpha: float,
    eps: float = 1e-4,
) -> np.ndarray:
    """Factorized Stage × Layer importance I_alpha(v) = S_stage(r)^alpha * L_layer(l)^(1-alpha)."""
    V = len(topo)
    imp = np.zeros(V, dtype=np.float64)
    for t in topo:
        v = t["vid"]
        s_val = max(float(stage_per_visit_sens[t["stage"]]), eps)
        l_val = max(float(layer_per_visit_sens[t["layer_id"]]), eps)
        imp[v] = (s_val ** alpha) * (l_val ** (1.0 - alpha))
    return imp


def solve_pure_k12_schedule(
    importance: np.ndarray,
    target_cut_frac: float,
) -> list[int]:
    """Exact zero-k0 pure k in {1, 2} schedule cutting target_cut_frac <= 0.50 of active slots."""
    V = len(importance)
    n_k1 = max(0, min(V, int(round(2.0 * V * target_cut_frac))))
    # Deterministic ascending sort by importance (tie-break by visit id)
    order = sorted(range(V), key=lambda v: (float(importance[v]), v))
    vk = [2] * V
    for v in order[:n_k1]:
        vk[v] = 1
    return vk


def build_natural_boundary_k12_schedule(
    topo: list[dict[str, Any]],
    layer_per_visit_sens: np.ndarray,
) -> tuple[list[int], str]:
    """Natural architecture-driven boundary-protected pure k in {1, 2} schedule:
    - Huginn-MoE (has explicit coda): prelude + loop_0..7 -> k=1 (104 visits), ONLY coda -> k=2 (8 visits) = -46.4%
    - Ouro-MoE (all recurrent): loop_0..2 -> k=1 (84 visits) + lowest 14 layers of loop_3 -> k=1 = -43.8%
    """
    V = len(topo)
    stages = {t["stage"] for t in topo}
    if "coda" in stages:
        vk = [2 if t["stage"] == "coda" else 1 for t in topo]
        return vk, "coda_only_k2_46.4pct"
    # Ouro-like all-recurrent: identify final loop stage
    max_r = max(t["loop_idx"] for t in topo)
    last_stage = f"loop_{max_r}"
    last_vids = [t["vid"] for t in topo if t["stage"] == last_stage]
    last_sorted = sorted(last_vids, key=lambda v: (float(layer_per_visit_sens[topo[v]["layer_id"]]), v))
    n_drop_last = len(last_vids) // 2  # drop bottom half of final loop -> 84 + 14 = 98 -> -43.75%
    vk = [1] * V
    for v in last_sorted[n_drop_last:]:
        vk[v] = 2
    return vk, f"{last_stage}_tophalf_k2_43.8pct"


def build_stage_only_cut38_schedule(
    topo: list[dict[str, Any]],
    stage_per_visit_sens: dict[str, float],
) -> list[int]:
    """Pure stage-level -37.5% schedule (84 visits at k=1, 28 at k=2):
    - Ouro-MoE: exact loop_0 + loop_1 + loop_2 = 84 visits at k=1, loop_3 = 28 at k=2!
    - Huginn-MoE: lowest-sensitivity recurrent stages summing to 84 visits (7 of 8 loops at k=1)!
    """
    V = len(topo)
    n_k1 = int(round(2.0 * V * 0.375))  # 84 for V=112
    stages = sorted(
        list(stage_per_visit_sens.keys()),
        key=lambda st: (float(stage_per_visit_sens[st]), st),
    )
    vk = [2] * V
    rem = n_k1
    for st in stages:
        if rem <= 0:
            break
        vids = [t["vid"] for t in topo if t["stage"] == st]
        if len(vids) <= rem:
            for v in vids:
                vk[v] = 1
            rem -= len(vids)
        else:
            for v in vids[:rem]:
                vk[v] = 1
            rem = 0
    return vk


def paired_bootstrap_ci(
    diffs: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    arr = np.asarray(diffs, dtype=np.float64)
    mean = float(np.mean(arr))
    if len(arr) <= 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return mean, float(lo), float(hi)


def calibrate_dynamic_threshold(
    visit_ratios: list[np.ndarray],
    visit_k: list[int],
    visit_weights: np.ndarray,
    target_cut_frac: float,
) -> float:
    """Exact quantile threshold tau on calibration tokens hitting target_cut_frac."""
    V = len(visit_k)
    fixed_k0 = sum(1 for k in visit_k if k == 0)
    fixed_k1 = sum(1 for k in visit_k if k == 1)
    n_k2 = sum(1 for k in visit_k if k == 2)
    fixed_cut = (2.0 * fixed_k0 + 1.0 * fixed_k1) / (2.0 * V)
    rem_cut = target_cut_frac - fixed_cut
    if rem_cut <= 1e-6 or n_k2 == 0:
        return -1.0
    p_drop = min(1.0, max(0.0, rem_cut * (2.0 * V) / float(n_k2)))
    if p_drop >= 0.9999:
        return 1e9
    weighted_scores: list[np.ndarray] = []
    for v in range(V):
        if visit_k[v] == 2 and visit_ratios[v].size > 0:
            weighted_scores.append(visit_ratios[v] * float(visit_weights[v]))
    if not weighted_scores:
        return -1.0
    all_scores = np.concatenate(weighted_scores, axis=0)
    return float(np.quantile(all_scores, p_drop))


# ---------------------------------------------------------------------------
# 2. Real Sparse + Zero-Sync Fast Small-Batch Decode MoE Executor
# ---------------------------------------------------------------------------

class LoopSchedV2Executor:
    """Real sparse MoE executor supporting native xllm dispatch and zero-sync small-batch BMM."""

    def __init__(
        self,
        model: Any,
        topo: list[dict[str, Any]],
        *,
        mode: str = "native",  # native | verify_k2 | static | dynamic
        visit_k: list[int] | None = None,
        visit_gamma: list[float] | None = None,
        renorm_on_drop2: bool = False,
        visit_weights: np.ndarray | None = None,
        dyn_threshold: float = -1.0,
        fast_small_batch: bool = False,
        small_batch_max_pairs: int = 16,
        track_stats: bool = True,
    ) -> None:
        self.model = model
        self.topo = topo
        self.V = len(topo)
        self.n_layers = len(model.layers)
        self.n_experts = int(model.layers[0].moe.experts.n_local_experts)
        self.model_dim = int(model.layers[0].moe.experts.model_dim)
        self.inter_dim = int(model.layers[0].moe.experts.inter_dim)
        self.topo_map: dict[tuple[int, int], int] = {}
        seen = [0] * self.n_layers
        for t in topo:
            lid = t["layer_id"]
            self.topo_map[(lid, seen[lid])] = t["vid"]
            seen[lid] += 1

        self.mode = mode
        self.visit_k = list(visit_k) if visit_k is not None else [2] * self.V
        self.visit_gamma = list(visit_gamma) if visit_gamma is not None else [1.0] * self.V
        self.renorm_on_drop2 = renorm_on_drop2
        self.visit_weights = visit_weights
        self.dyn_threshold = float(dyn_threshold)
        self.fast_small_batch = fast_small_batch
        self.small_batch_max_pairs = small_batch_max_pairs
        self.track_stats = track_stats

        # Pre-cache reshaped 3D views [E, I, D] / [E, D, I] per layer (zero-copy views!)
        self._w_views: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for layer in model.layers:
            em = layer.moe.experts
            self._w_views.append(
                (
                    em.weight1.view(self.n_experts, self.inter_dim, self.model_dim),
                    em.weight2.view(self.n_experts, self.model_dim, self.inter_dim),
                    em.weight3.view(self.n_experts, self.inter_dim, self.model_dim),
                )
            )

        self._orig_moe_fwd: list[tuple[Any, Any]] = []
        self._visits = [0] * self.n_layers
        self.total_slots = 0
        self.active_slots = 0
        self.k0_visits_run = 0
        self.k1_visits_run = 0
        self.k2_visits_run = 0

    def reset_pass(self) -> None:
        for i in range(self.n_layers):
            self._visits[i] = 0

    def reset_counters(self) -> None:
        self.total_slots = 0
        self.active_slots = 0
        self.k0_visits_run = 0
        self.k1_visits_run = 0
        self.k2_visits_run = 0

    @property
    def active_expert_frac(self) -> float:
        return float(self.active_slots / max(self.total_slots, 1))

    @property
    def cut_pct(self) -> float:
        return 100.0 * (1.0 - self.active_expert_frac)

    def _dispatch_fast_small_batch(
        self,
        lid: int,
        x_2d: torch.Tensor,
        tok_ids: torch.Tensor,
        exp_ids: torch.Tensor,
        w_vals: torch.Tensor,
    ) -> torch.Tensor:
        """Zero-sync GPU BMM dispatch reading ONLY active experts (zero .tolist() sync!)."""
        n_tok = x_2d.shape[0]
        n_pairs = exp_ids.numel()
        w1_all, w2_all, w3_all = self._w_views[lid]
        if n_tok == 1:
            x_p = x_2d.unsqueeze(1).expand(n_pairs, 1, -1)
        else:
            x_p = x_2d.index_select(0, tok_ids).unsqueeze(1)
        w1_p = w1_all.index_select(0, exp_ids)
        w3_p = w3_all.index_select(0, exp_ids)
        w2_p = w2_all.index_select(0, exp_ids)
        h = F.silu(torch.bmm(x_p, w1_p.transpose(1, 2))) * torch.bmm(x_p, w3_p.transpose(1, 2))
        y_p = torch.bmm(h, w2_p.transpose(1, 2)).squeeze(1)
        y_w = y_p * w_vals.to(x_2d.dtype).unsqueeze(-1)
        if n_tok == 1:
            if n_pairs == 1:
                return y_w
            return y_w.sum(dim=0, keepdim=True)
        y_out = torch.zeros_like(x_2d)
        y_out.index_add_(0, tok_ids, y_w)
        return y_out

    def _dispatch_routed_pairs(
        self,
        lid: int,
        moe: Any,
        x_2d: torch.Tensor,
        tok_ids: torch.Tensor,
        exp_ids: torch.Tensor,
        w_vals: torch.Tensor,
    ) -> torch.Tensor:
        n_pairs = tok_ids.numel()
        if n_pairs == 0:
            return torch.zeros_like(x_2d)
        if self.fast_small_batch and n_pairs <= self.small_batch_max_pairs:
            return self._dispatch_fast_small_batch(lid, x_2d, tok_ids, exp_ids, w_vals)
        order = torch.argsort(exp_ids, stable=False)
        tok_s = tok_ids.index_select(0, order)
        exp_s = exp_ids.index_select(0, order)
        w_s = w_vals.index_select(0, order).to(x_2d.dtype)
        group_sizes = torch.bincount(exp_s, minlength=self.n_experts).tolist()
        x_perm = x_2d.index_select(0, tok_s)
        y_perm = moe.experts(x_perm, group_sizes)
        y_out = torch.zeros_like(x_2d)
        y_out.index_add_(0, tok_s, y_perm * w_s.unsqueeze(-1))
        return y_out

    def _run_moe_visit(
        self,
        lid: int,
        moe: Any,
        orig_fn: Any,
        x_in: torch.Tensor,
        load_balancing_type: Any,
    ) -> tuple[torch.Tensor, Any]:
        v_num = self._visits[lid]
        self._visits[lid] = v_num + 1
        vid = self.topo_map[(lid, v_num)]
        kv = int(self.visit_k[vid])
        bsz, slen, dim = x_in.shape
        n_tok = bsz * slen

        if self.mode == "native":
            if self.track_stats:
                self.total_slots += 2 * n_tok
                self.active_slots += 2 * n_tok
                self.k2_visits_run += 1
            return orig_fn(x_in, load_balancing_type)

        # k_v == 0: TRUE FULL ROUTED-MoE BYPASS (skips router AND experts!)
        if kv == 0:
            if self.track_stats:
                self.total_slots += 2 * n_tok
                self.k0_visits_run += 1
            residual = x_in
            x = moe.norm(x_in)
            if moe.fc1 is not None:
                y1 = moe.fc2(F.silu(moe.fc1(x)) * moe.fc3(x))
            else:
                y1 = torch.zeros_like(x)
            return moe.residual(y1, residual, c=y1), None

        # Untouched k=2 in static mode (use native UNLESS fast_small_batch is enabled for small batch!)
        if self.mode == "static" and kv == 2 and not (self.fast_small_batch and 2 * n_tok <= self.small_batch_max_pairs):
            if self.track_stats:
                self.total_slots += 2 * n_tok
                self.active_slots += 2 * n_tok
                self.k2_visits_run += 1
            return orig_fn(x_in, load_balancing_type)

        residual = x_in
        x = moe.norm(x_in)
        y1 = moe.fc2(F.silu(moe.fc1(x)) * moe.fc3(x)) if moe.fc1 is not None else None
        routing_scores, routing_indices, _tpe, aux_loss = moe.router(x, load_balancing_type)

        x_2d = x.reshape(n_tok, dim)
        w_f = routing_scores.reshape(n_tok, -1).float()
        idx_2d = routing_indices.reshape(n_tok, -1)
        topk = idx_2d.shape[-1]
        scale = float(moe.router.scaling_factor) if getattr(moe.router, "scaling_factor", None) else 1.0
        gamma = float(self.visit_gamma[vid])

        if self.mode == "verify_k2" or (self.mode == "static" and kv == 2):
            if self.track_stats:
                self.total_slots += n_tok * topk
                self.active_slots += n_tok * topk
                self.k2_visits_run += 1
            tok_ids = torch.arange(n_tok, device=x_in.device).unsqueeze(1).expand(n_tok, topk).reshape(-1)
            exp_ids = idx_2d.reshape(-1)
            w_vals = w_f.reshape(-1)
            y2 = self._dispatch_routed_pairs(lid, moe, x_2d, tok_ids, exp_ids, w_vals).reshape(bsz, slen, dim)
            y = y1 + y2 if y1 is not None else y2
            return moe.residual(y, residual, c=y), aux_loss

        w_min, min_pos = w_f.min(dim=-1)
        max_pos = 1 - min_pos
        w_max = w_f.gather(-1, max_pos.unsqueeze(-1)).squeeze(-1)
        e_max = idx_2d.gather(-1, max_pos.unsqueeze(-1)).squeeze(-1)
        e_min = idx_2d.gather(-1, min_pos.unsqueeze(-1)).squeeze(-1)

        if kv == 1:
            if self.track_stats:
                self.total_slots += 2 * n_tok
                self.active_slots += n_tok
                self.k1_visits_run += 1
            tok_ids = torch.arange(n_tok, device=x_in.device)
            w_use = torch.full_like(w_max, scale) if self.renorm_on_drop2 else (w_max * gamma)
            y2 = self._dispatch_routed_pairs(lid, moe, x_2d, tok_ids, e_max, w_use).reshape(bsz, slen, dim)
            y = y1 + y2 if y1 is not None else y2
            return moe.residual(y, residual, c=y), aux_loss

        # Dynamic mode on k_v == 2 visit
        sw = 1.0 if self.visit_weights is None else float(self.visit_weights[vid])
        ratio = w_min / w_max.clamp_min(1e-12)
        score = ratio * sw
        drop_mask = score < self.dyn_threshold
        keep2_mask = ~drop_mask

        n_keep2 = int(keep2_mask.sum().item())
        if self.track_stats:
            self.total_slots += 2 * n_tok
            self.active_slots += n_tok + n_keep2
            self.k2_visits_run += 1

        tok_all = torch.arange(n_tok, device=x_in.device)
        if self.renorm_on_drop2:
            w1_use = torch.where(drop_mask, torch.full_like(w_max, scale), w_max)
        else:
            w1_use = torch.where(drop_mask, w_max * gamma, w_max)

        if n_keep2 == 0:
            y2 = self._dispatch_routed_pairs(lid, moe, x_2d, tok_all, e_max, w1_use).reshape(bsz, slen, dim)
        else:
            tok_cat = torch.cat([tok_all, tok_all[keep2_mask]], dim=0)
            exp_cat = torch.cat([e_max, e_min[keep2_mask]], dim=0)
            w_cat = torch.cat([w1_use, w_min[keep2_mask]], dim=0)
            y2 = self._dispatch_routed_pairs(lid, moe, x_2d, tok_cat, exp_cat, w_cat).reshape(bsz, slen, dim)

        y = y1 + y2 if y1 is not None else y2
        return moe.residual(y, residual, c=y), aux_loss

    def __enter__(self) -> "LoopSchedV2Executor":
        self._orig_moe_fwd = []
        for lid, layer in enumerate(self.model.layers):
            moe = layer.moe
            orig = moe.forward
            self._orig_moe_fwd.append((moe, orig))

            def make_fn(lid_c: int, moe_c: Any, fn_c: Any):
                def wrapped(x_in, load_balancing_type=None):
                    return self._run_moe_visit(lid_c, moe_c, fn_c, x_in, load_balancing_type)
                return wrapped

            moe.forward = make_fn(lid, moe, orig)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for moe, orig in self._orig_moe_fwd:
            moe.forward = orig
        self._orig_moe_fwd = []


# ---------------------------------------------------------------------------
# 3. Fast Batched NLL & MCQ Evaluation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def eval_nll_batched(
    model: Any,
    seqs: list[dict[str, Any]],
    executor: LoopSchedV2Executor,
    device: torch.device,
    *,
    batch_size: int = 8,
) -> dict[str, Any]:
    executor.reset_counters()
    per_seq: list[float] = []
    text_nlls: list[float] = []
    code_nlls: list[float] = []
    with executor:
        for s0 in range(0, len(seqs), batch_size):
            batch = seqs[s0 : s0 + batch_size]
            toks = torch.tensor([b["tokens"] for b in batch], device=device, dtype=torch.long)
            inp, tgt = toks[:, :-1], toks[:, 1:]
            executor.reset_pass()
            logits, _, _ = model(inp, multi_segments=False, cache=None)
            loss_tok = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                tgt.reshape(-1),
                reduction="none",
            ).reshape(len(batch), -1)
            for item, val in zip(batch, loss_tok.mean(dim=-1).cpu().tolist()):
                per_seq.append(float(val))
                (text_nlls if item.get("domain") == "text" else code_nlls).append(float(val))
    return {
        "nll": float(np.mean(per_seq)),
        "text_nll": float(np.mean(text_nlls)) if text_nlls else 0.0,
        "code_nll": float(np.mean(code_nlls)) if code_nlls else 0.0,
        "per_seq": per_seq,
        "active_expert_frac": executor.active_expert_frac,
        "cut_pct": executor.cut_pct,
    }


def prepare_mcq_encoded(samples: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    encoded: list[dict[str, Any]] = []
    for q_idx, item in enumerate(samples):
        prompt = f"Question: {item['question']}\nAnswer:"
        p_ids = tokenizer.encode(prompt, bos=True, eos=False)
        p_len = len(p_ids)
        choices_enc: list[tuple[list[int], int]] = []
        for c_txt in item["choice_texts"]:
            c_ids = tokenizer.encode(" " + c_txt, bos=False, eos=False)
            choices_enc.append((p_ids + c_ids, p_len) if c_ids else ([], p_len))
        encoded.append({"q_idx": q_idx, "gold_idx": int(item["gold_idx"]), "choices": choices_enc})
    return encoded


@torch.inference_mode()
def eval_mcq_batched(
    model: Any,
    encoded_items: list[dict[str, Any]],
    executor: LoopSchedV2Executor,
    device: torch.device,
    *,
    batch_size: int = 32,
    pad_token_id: int = 0,
) -> tuple[float, list[int]]:
    flat: list[tuple[int, int, list[int], int]] = []
    scores_by_q: list[list[float]] = [[-1e9] * len(it["choices"]) for it in encoded_items]
    for it in encoded_items:
        qi = it["q_idx"]
        for ci, (full_ids, p_len) in enumerate(it["choices"]):
            if len(full_ids) > p_len:
                flat.append((qi, ci, full_ids, p_len))
    flat.sort(key=lambda x: len(x[2]))

    with executor:
        for s0 in range(0, len(flat), batch_size):
            chunk = flat[s0 : s0 + batch_size]
            bsz = len(chunk)
            max_inp_len = max(len(x[2]) - 1 for x in chunk)
            inp = torch.full((bsz, max_inp_len), pad_token_id, device=device, dtype=torch.long)
            for bi, (_qi, _ci, full_ids, _plen) in enumerate(chunk):
                L = len(full_ids) - 1
                inp[bi, :L] = torch.tensor(full_ids[:-1], device=device, dtype=torch.long)
            executor.reset_pass()
            logits, _, _ = model(inp, multi_segments=False, cache=None)
            for bi, (qi, ci, full_ids, p_len) in enumerate(chunk):
                c_len = len(full_ids) - p_len
                sl_logits = logits[bi, p_len - 1 : p_len - 1 + c_len, :].float()
                lp = F.log_softmax(sl_logits, dim=-1)
                tgt = torch.tensor(full_ids[p_len:], device=device, dtype=torch.long)
                scores_by_q[qi][ci] = float(lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean().item())

    flags = [1 if int(np.argmax(scores_by_q[qi])) == encoded_items[qi]["gold_idx"] else 0 for qi in range(len(encoded_items))]
    return float(np.mean(flags)), flags


# ---------------------------------------------------------------------------
# 4. High-SNR Calibration + Non-Additive IterGreedy Solver
# ---------------------------------------------------------------------------

def solve_iter_greedy_k012(
    eval_a_fn,
    eval_b_fn,
    topo: list[dict[str, Any]],
    importance: np.ndarray,
    screen_k0_cost: np.ndarray,
    eligible_k0_vids: list[int],
    *,
    max_swaps_50: int = 16,
    cand_pool_size: int = 8,
    patience_b: int = 2,
) -> dict[str, Any]:
    """Non-additive iterative greedy solver for -50% (swap) and -62.5% (drop) budgets.

    Every single step evaluates TRUE cumulative NLL on the modified network (zero additivity assumption!).
    Separated into a pure solver taking callables eval_a_fn(vk) and eval_b_fn(vk) for exact CPU unit testing.
    """
    V = len(topo)
    # --- Part A: -50.0% (exact V active slots = All-Top1 + verified boundary upgrades paid by sparse k=0) ---
    vk_cur = [1] * V
    a_cur = float(eval_a_fn(vk_cur))
    b_cur = float(eval_b_fn(vk_cur))
    best_vk_50 = list(vk_cur)
    best_b_50 = b_cur
    best_a_50 = a_cur
    swap_trace: list[dict[str, Any]] = []
    no_improve_b = 0

    # Candidate upgrade ordering: highest importance visits first
    up_order = sorted(range(V), key=lambda v: (-float(importance[v]), v))
    # Candidate downgrade ordering: lowest screened k=0 cost among eligible non-boundary visits
    dn_order_base = sorted(eligible_k0_vids, key=lambda v: (float(screen_k0_cost[v]), v))

    for step_idx in range(max_swaps_50):
        # Pick highest-importance remaining k=1 visit
        v_up = next((v for v in up_order if vk_cur[v] == 1), None)
        if v_up is None:
            break
        # Candidate pool for v_dn: top M remaining k=1 eligible visits != v_up
        cands_dn = [v for v in dn_order_base if vk_cur[v] == 1 and v != v_up][:cand_pool_size]
        if not cands_dn:
            break

        best_dn = -1
        best_cand_a = 1e18
        for v_dn in cands_dn:
            vk_try = list(vk_cur)
            vk_try[v_up] = 2
            vk_try[v_dn] = 0
            val_a = float(eval_a_fn(vk_try))
            if val_a < best_cand_a:
                best_cand_a = val_a
                best_dn = v_dn

        if best_dn < 0 or best_cand_a >= a_cur - 1e-4:
            # No swap improves even calib_a -> stop cleanly!
            break

        vk_next = list(vk_cur)
        vk_next[v_up] = 2
        vk_next[best_dn] = 0
        b_next = float(eval_b_fn(vk_next))
        swap_trace.append(
            {
                "swap": step_idx + 1,
                "v_up": int(v_up),
                "stage_up": topo[v_up]["stage"],
                "v_dn": int(best_dn),
                "stage_dn": topo[best_dn]["stage"],
                "calib_a_nll": best_cand_a,
                "calib_b_nll": b_next,
            }
        )
        vk_cur = vk_next
        a_cur = best_cand_a
        if b_next < best_b_50 - 1e-4:
            best_b_50 = b_next
            best_a_50 = best_cand_a
            best_vk_50 = list(vk_cur)
            no_improve_b = 0
        else:
            no_improve_b += 1
            if no_improve_b >= patience_b:
                break

    # --- Part B: -62.5% (exact round(0.75 * V) active slots = 84 for V=112) ---
    target_slots_62 = int(round(2.0 * V * (1.0 - 0.625)))
    vk_62 = list(best_vk_50)
    drop_trace: list[dict[str, Any]] = []
    while sum(vk_62) > target_slots_62:
        # Candidate pool: top cand_pool_size remaining k=1 visits from eligible_k0_vids (fallback to any k=1/2)
        cands = [v for v in dn_order_base if vk_62[v] == 1][:cand_pool_size]
        if not cands:
            cands = sorted([v for v in range(V) if vk_62[v] > 0], key=lambda v: (float(importance[v]), v))[:cand_pool_size]
        # Also include the lowest-importance k=2 visit (if any) so solver can choose 2->1 vs 1->0!
        k2_vids = sorted([v for v in range(V) if vk_62[v] == 2], key=lambda v: (float(importance[v]), v))
        if k2_vids:
            for vk2 in k2_vids[:2]:
                if vk2 not in cands:
                    cands.append(vk2)

        best_v = -1
        best_a = 1e18
        for v in cands:
            vk_try = list(vk_62)
            vk_try[v] -= 1
            val_a = float(eval_a_fn(vk_try))
            if val_a < best_a:
                best_a = val_a
                best_v = v

        vk_62[best_v] -= 1
        drop_trace.append(
            {
                "slots_left": int(sum(vk_62)),
                "vid": int(best_v),
                "stage": topo[best_v]["stage"],
                "new_k": int(vk_62[best_v]),
                "calib_a_nll": best_a,
            }
        )

    return {
        "vk_50": best_vk_50,
        "vk_62": vk_62,
        "calib_a_50": best_a_50,
        "calib_b_50": best_b_50,
        "swap_trace": swap_trace,
        "drop_trace": drop_trace,
    }


@torch.inference_mode()
def calibrate_r3(
    model: Any,
    topo: list[dict[str, Any]],
    calib_a: list[dict[str, Any]],
    calib_b: list[dict[str, Any]],
    device: torch.device,
    *,
    batch_size: int = 8,
    phase_ctx: Any = None,
) -> dict[str, Any]:
    """Single-pass geometry + AWQ RMS + stage/layer joint sensitivities + calib_b alpha + IterGreedy."""
    V = len(topo)
    n_layers = len(model.layers)
    n_experts = int(model.layers[0].moe.experts.n_local_experts)
    D = int(model.layers[0].moe.experts.model_dim)
    topo_map: dict[tuple[int, int], int] = {}
    seen = [0] * n_layers
    for t in topo:
        lid = t["layer_id"]
        topo_map[(lid, seen[lid])] = t["vid"]
        seen[lid] += 1

    # 1) Pass 1 over calib_a: geometry (cos(e1,e2), MSE ratio, stage gamma_s*) + ratios + AWQ act_rms
    visit_dot = np.zeros(V, dtype=np.float64)
    visit_e1_sq = np.zeros(V, dtype=np.float64)
    visit_w2e2_sq = np.zeros(V, dtype=np.float64)
    visit_renorm_err_sq = np.zeros(V, dtype=np.float64)
    visit_cos_sum = np.zeros(V, dtype=np.float64)
    visit_cnt = np.zeros(V, dtype=np.int64)
    visit_ratios_lists: list[list[np.ndarray]] = [[] for _ in range(V)]
    # Pooled per-(layer, expert) input sq-sum for Bonus W3-Pool-AWQ
    act_sq_sum = torch.zeros((n_layers, n_experts, D), dtype=torch.float64, device=device)
    act_tok_cnt = torch.zeros((n_layers, n_experts), dtype=torch.float64, device=device)

    visits_cursor = [0] * n_layers
    orig_fwds: list[tuple[Any, Any]] = []
    for lid, layer in enumerate(model.layers):
        moe = layer.moe
        orig = moe.forward
        orig_fwds.append((moe, orig))

        def make_probe(lid_c: int, moe_c: Any, fn_c: Any):
            def wrapped(x_in, load_balancing_type=None):
                vn = visits_cursor[lid_c]
                visits_cursor[lid_c] = vn + 1
                vid = topo_map[(lid_c, vn)]
                bsz, slen, dim = x_in.shape
                n_tok = bsz * slen
                x = moe_c.norm(x_in)
                r_scores, r_idx, _tpe, _aux = moe_c.router(x, load_balancing_type)
                x_2d = x.reshape(n_tok, dim)
                w_f = r_scores.reshape(n_tok, -1).float()
                idx_2d = r_idx.reshape(n_tok, -1)

                # Accumulate trajectory-pooled input activation energy for AWQ bonus
                x_sq = x_2d.float().pow(2)
                for slot in range(idx_2d.shape[-1]):
                    e_s = idx_2d[:, slot]
                    act_sq_sum[lid_c].index_add_(0, e_s, x_sq.to(torch.float64))
                    act_tok_cnt[lid_c].index_add_(0, e_s, torch.ones(n_tok, dtype=torch.float64, device=device))

                # Geometry of top-1 vs top-2
                w_min, min_pos = w_f.min(dim=-1)
                max_pos = 1 - min_pos
                w_max = w_f.gather(-1, max_pos.unsqueeze(-1)).squeeze(-1)
                e_max = idx_2d.gather(-1, max_pos.unsqueeze(-1)).squeeze(-1)
                e_min = idx_2d.gather(-1, min_pos.unsqueeze(-1)).squeeze(-1)
                visit_ratios_lists[vid].append((w_min / w_max.clamp_min(1e-12)).cpu().numpy().astype(np.float32))

                sub = min(n_tok, 512)
                tok_sub = torch.arange(sub, device=x_in.device)
                x_sub = x_2d[:sub]
                o1 = torch.argsort(e_max[:sub], stable=False)
                gs1 = torch.bincount(e_max[:sub].index_select(0, o1), minlength=n_experts).tolist()
                y1_p = moe_c.experts(x_sub.index_select(0, o1), gs1)
                e1_out = torch.zeros_like(x_sub).index_copy_(0, o1, y1_p).float()

                o2 = torch.argsort(e_min[:sub], stable=False)
                gs2 = torch.bincount(e_min[:sub].index_select(0, o2), minlength=n_experts).tolist()
                y2_p = moe_c.experts(x_sub.index_select(0, o2), gs2)
                e2_out = torch.zeros_like(x_sub).index_copy_(0, o2, y2_p).float()

                w1_s = w_max[:sub]
                w2_s = w_min[:sub]
                dot_e = (e1_out * e2_out).sum(dim=-1)
                n1_sq = (e1_out * e1_out).sum(dim=-1)
                n2_sq = (e2_out * e2_out).sum(dim=-1)
                cos_v = dot_e / (n1_sq.clamp_min(1e-12).sqrt() * n2_sq.clamp_min(1e-12).sqrt())
                # Renorm error vector: (w1+w2)*e1 - (w1*e1 + w2*e2) = w2*(e1 - e2)
                # NoRen error vector: w1*e1 - (w1*e1 + w2*e2) = -w2*e2
                ren_err_sq = (w2_s.pow(2) * (n1_sq + n2_sq - 2.0 * dot_e)).sum().item()
                noren_err_sq = (w2_s.pow(2) * n2_sq).sum().item()

                visit_dot[vid] += float((w1_s * w2_s * dot_e).sum().item())
                visit_e1_sq[vid] += float((w1_s.pow(2) * n1_sq).sum().item())
                visit_w2e2_sq[vid] += float(noren_err_sq)
                visit_renorm_err_sq[vid] += float(ren_err_sq)
                visit_cos_sum[vid] += float(cos_v.sum().item())
                visit_cnt[vid] += sub

                return fn_c(x_in, load_balancing_type)
            return wrapped

        moe.forward = make_probe(lid, moe, orig)

    try:
        for s0 in range(0, len(calib_a), batch_size):
            batch = calib_a[s0 : s0 + batch_size]
            toks = torch.tensor([b["tokens"] for b in batch], device=device, dtype=torch.long)
            for i in range(n_layers):
                visits_cursor[i] = 0
            _ = model(toks[:, :-1], multi_segments=False, cache=None)
    finally:
        for moe, orig in orig_fwds:
            moe.forward = orig

    # Aggregate stage-pooled gamma_s*, cos, and empirical MSE_renorm/MSE_noren
    stage_order: list[str] = []
    for t in topo:
        if t["stage"] not in stage_order:
            stage_order.append(t["stage"])

    stage_gamma: dict[str, float] = {}
    stage_cos: dict[str, float] = {}
    stage_mse_ratio: dict[str, float] = {}
    for st in stage_order:
        vids_s = [t["vid"] for t in topo if t["stage"] == st]
        num = sum(visit_dot[v] for v in vids_s)
        den = sum(visit_e1_sq[v] for v in vids_s)
        g_s = 1.0 + num / max(den, 1e-12)
        stage_gamma[st] = float(min(max(g_s, 0.90), 1.35))
        stage_cos[st] = float(sum(visit_cos_sum[v] for v in vids_s) / max(sum(visit_cnt[v] for v in vids_s), 1))
        stage_mse_ratio[st] = float(
            sum(visit_renorm_err_sq[v] for v in vids_s) / max(sum(visit_w2e2_sq[v] for v in vids_s), 1e-12)
        )

    visit_gamma_stage = [stage_gamma[t["stage"]] for t in topo]
    visit_ratios = [
        np.concatenate(visit_ratios_lists[v], axis=0) if visit_ratios_lists[v] else np.zeros(0, dtype=np.float32)
        for v in range(V)
    ]

    # Compute pooled activation RMS [L, E, D] on CPU for Bonus W3-Pool-AWQ
    act_rms_cpu = torch.sqrt(
        (act_sq_sum / act_tok_cnt.clamp_min(1.0).unsqueeze(-1)).clamp_min(1e-12)
    ).float().cpu()
    del act_sq_sum, act_tok_cnt
    torch.cuda.empty_cache()
    if phase_ctx is not None:
        phase_ctx.step(1)

    # Fast calibration helper on calib_a[:8] (2,048 tokens) and calib_b (2,048 tokens)
    calib_fast = calib_a[: min(8, len(calib_a))]
    ex_base = LoopSchedV2Executor(model, topo, mode="native", track_stats=False)
    base_a_nll = eval_nll_batched(model, calib_fast, ex_base, device, batch_size=batch_size)["nll"]
    base_b_nll = eval_nll_batched(model, calib_b, ex_base, device, batch_size=batch_size)["nll"]

    def eval_static_a(vk: list[int], vg: list[float] | None = None, renorm: bool = False) -> float:
        ex = LoopSchedV2Executor(
            model, topo, mode="static", visit_k=vk, visit_gamma=vg or [1.0] * V,
            renorm_on_drop2=renorm, track_stats=False,
        )
        return eval_nll_batched(model, calib_fast, ex, device, batch_size=batch_size)["nll"]

    def eval_static_b(vk: list[int], vg: list[float] | None = None, renorm: bool = False) -> float:
        ex = LoopSchedV2Executor(
            model, topo, mode="static", visit_k=vk, visit_gamma=vg or [1.0] * V,
            renorm_on_drop2=renorm, track_stats=False,
        )
        return eval_nll_batched(model, calib_b, ex, device, batch_size=batch_size)["nll"]

    # 2) Noise-Free Stage-Joint Sensitivities
    stage_summary: list[dict[str, Any]] = []
    stage_per_visit_sens: dict[str, float] = {}
    stage_k0_map: dict[str, float] = {}
    for st in stage_order:
        vids_s = [t["vid"] for t in topo if t["stage"] == st]
        vk1 = [1 if v in vids_s else 2 for v in range(V)]
        vk0 = [0 if v in vids_s else 2 for v in range(V)]
        d_ren = eval_static_a(vk1, renorm=True) - base_a_nll
        d_nor = eval_static_a(vk1, renorm=False) - base_a_nll
        d_opt = eval_static_a(vk1, vg=visit_gamma_stage, renorm=False) - base_a_nll
        d_k0 = eval_static_a(vk0, renorm=False) - base_a_nll
        stage_k0_map[st] = float(d_k0)
        # Per-visit stage sensitivity (clamped positive)
        stage_per_visit_sens[st] = max(float(d_nor), 1e-4) / float(len(vids_s))
        stage_summary.append(
            {
                "stage": st,
                "n_visits": len(vids_s),
                "cos_e1_e2": stage_cos[st],
                "mse_ratio_ren_vs_noren": stage_mse_ratio[st],
                "gamma_stage": stage_gamma[st],
                "k1_renorm": float(d_ren),
                "k1_noren": float(d_nor),
                "k1_opt": float(d_opt),
                "k0_sens": float(d_k0),
            }
        )

    # 3) Low-Variance Physical Layer-Joint Sensitivities (28 forwards!)
    layer_per_visit_sens = np.zeros(n_layers, dtype=np.float64)
    for lid in range(n_layers):
        vids_l = [t["vid"] for t in topo if t["layer_id"] == lid]
        vk_l = [1 if v in vids_l else 2 for v in range(V)]
        d_l = eval_static_a(vk_l, renorm=False) - base_a_nll
        layer_per_visit_sens[lid] = max(float(d_l), 1e-4) / float(len(vids_l))
    if phase_ctx is not None:
        phase_ctx.step(2)

    # 4) Disjoint calib_b Selection of Stage × Layer Exponent alpha in {0.25, 0.50, 0.75}
    alpha_calib_b_nlls: dict[str, float] = {}
    best_alpha = 0.5
    best_alpha_nll = 1e18
    for a_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        imp_a = compute_factorized_importance(topo, stage_per_visit_sens, layer_per_visit_sens, alpha=a_val)
        w_a = np.sqrt(np.maximum(imp_a, 1e-6))
        w_a = w_a / max(float(np.median(w_a)), 1e-6)
        tau_a = calibrate_dynamic_threshold(visit_ratios, [2] * V, w_a, 0.375)
        ex_a = LoopSchedV2Executor(
            model, topo, mode="dynamic", visit_k=[2] * V, visit_gamma=[1.0] * V,
            renorm_on_drop2=False, visit_weights=w_a, dyn_threshold=tau_a, track_stats=False,
        )
        nll_b = eval_nll_batched(model, calib_b, ex_a, device, batch_size=batch_size)["nll"]
        alpha_calib_b_nlls[f"{a_val:.2f}"] = float(nll_b - base_b_nll)
        if a_val in (0.25, 0.5, 0.75) and nll_b < best_alpha_nll:
            best_alpha_nll = nll_b
            best_alpha = a_val

    imp_factorized = compute_factorized_importance(topo, stage_per_visit_sens, layer_per_visit_sens, alpha=best_alpha)
    imp_layer_only = compute_factorized_importance(topo, stage_per_visit_sens, layer_per_visit_sens, alpha=0.0)
    imp_stage_only = compute_factorized_importance(topo, stage_per_visit_sens, layer_per_visit_sens, alpha=1.0)

    # 5) Screen Single-Visit k=0 on Eligible Non-Boundary Stages (for IterGreedy candidate pool)
    # Exclude high-damage boundary stages (prelude / coda / stage_k0 > 2.0) from k=0 candidate pool
    eligible_k0_vids = [
        t["vid"] for t in topo
        if t["is_rec"] and stage_k0_map[t["stage"]] < 2.0
    ]
    if len(eligible_k0_vids) < 32:
        # Fallback: take all stages except the top-2 most sensitive k=0 stages
        sorted_st = sorted(stage_order, key=lambda s: stage_k0_map[s])
        keep_st = set(sorted_st[:-2]) if len(sorted_st) > 2 else set(sorted_st)
        eligible_k0_vids = [t["vid"] for t in topo if t["stage"] in keep_st]

    screen_k0_cost = np.full(V, 1e6, dtype=np.float64)
    for vid in eligible_k0_vids:
        vk_s = [2] * V
        vk_s[vid] = 0
        screen_k0_cost[vid] = max(0.0, eval_static_a(vk_s, renorm=False) - base_a_nll)
    if phase_ctx is not None:
        phase_ctx.step(3)

    # 6) Run Non-Additive Iterative Greedy Solver (TRUE cumulative NLL on calib_a + calib_b verification!)
    iter_res = solve_iter_greedy_k012(
        eval_a_fn=lambda vk: eval_static_a(vk, renorm=False),
        eval_b_fn=lambda vk: eval_static_b(vk, renorm=False),
        topo=topo,
        importance=imp_factorized,
        screen_k0_cost=screen_k0_cost,
        eligible_k0_vids=eligible_k0_vids,
        max_swaps_50=16,
        cand_pool_size=8,
        patience_b=2,
    )

    gc.collect()
    torch.cuda.empty_cache()

    return {
        "base_a_nll": base_a_nll,
        "base_b_nll": base_b_nll,
        "stage_summary": stage_summary,
        "stage_per_visit_sens": stage_per_visit_sens,
        "layer_per_visit_sens": layer_per_visit_sens,
        "stage_gamma": stage_gamma,
        "visit_gamma_stage": visit_gamma_stage,
        "visit_ratios": visit_ratios,
        "best_alpha": best_alpha,
        "alpha_calib_b_dnlls": alpha_calib_b_nlls,
        "imp_factorized": imp_factorized,
        "imp_layer_only": imp_layer_only,
        "imp_stage_only": imp_stage_only,
        "iter_greedy": iter_res,
        "act_rms_cpu": act_rms_cpu,
    }


# ---------------------------------------------------------------------------
# 5. Bonus Orthogonality: Vectorized W3-g128 Trajectory-Pooled AWQ
# ---------------------------------------------------------------------------

def fake_quantize_asym_group(w: torch.Tensor, bits: int, group_size: int = 128) -> torch.Tensor:
    orig_shape = w.shape
    orig_dtype = w.dtype
    in_dim = orig_shape[-1]
    pad = (group_size - (in_dim % group_size)) % group_size
    wf = w.float()
    if pad > 0:
        wf = F.pad(wf, (0, pad))
    w_g = wf.reshape(*orig_shape[:-1], -1, group_size)
    qmax = float((1 << bits) - 1)
    w_min = w_g.amin(dim=-1, keepdim=True)
    w_max = w_g.amax(dim=-1, keepdim=True)
    scale = ((w_max - w_min) / qmax).clamp_min(1e-8)
    zp = torch.round(-w_min / scale).clamp(0.0, qmax)
    q = torch.round(w_g / scale + zp).clamp(0.0, qmax)
    dq = (q - zp) * scale
    dq = dq.reshape(*orig_shape[:-1], in_dim + pad)[..., :in_dim]
    return dq.to(orig_dtype)


@torch.no_grad()
def apply_w3_pool_awq_inplace(
    model: Any,
    act_rms_cpu: torch.Tensor,
    *,
    bits: int = 3,
    group_size: int = 128,
    alpha_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> dict[str, float]:
    """Apply W3-g128 Trajectory-Pooled AWQ in-place to all routed MoE experts."""
    n_layers = len(model.layers)
    E = int(model.layers[0].moe.experts.n_local_experts)
    D = int(model.layers[0].moe.experts.model_dim)
    I = int(model.layers[0].moe.experts.inter_dim)
    dev = model.layers[0].moe.experts.weight1.device
    chosen_alphas: list[float] = []

    for lid, layer in enumerate(model.layers):
        em = layer.moe.experts
        s_in = act_rms_cpu[lid].to(dev, dtype=torch.float32).clamp_min(1e-5)  # [E, D]
        w1_v = em.weight1.view(E, I, D)
        w2_v = em.weight2.view(E, D, I)
        w3_v = em.weight3.view(E, I, D)
        w1_f = w1_v.float()
        w3_f = w3_v.float()

        best_err = torch.full((E,), float("inf"), device=dev)
        best_alpha = torch.zeros((E,), device=dev)
        s_sq = s_in.pow(2).unsqueeze(1)  # [E, 1, D]

        for a in alpha_grid:
            sc = s_in.pow(a)
            sc = sc / sc.mean(dim=-1, keepdim=True).clamp_min(1e-6)  # [E, D]
            sc_u = sc.unsqueeze(1)
            q1 = fake_quantize_asym_group(w1_f * sc_u, bits, group_size) / sc_u
            q3 = fake_quantize_asym_group(w3_f * sc_u, bits, group_size) / sc_u
            err = ((w1_f - q1).pow(2) * s_sq).mean(dim=(1, 2)) + ((w3_f - q3).pow(2) * s_sq).mean(dim=(1, 2))
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_alpha = torch.where(better, torch.full_like(best_alpha, a), best_alpha)

        chosen_alphas.extend(best_alpha.cpu().tolist())
        # Apply per-expert optimal AWQ scaling to W1 & W3, and standard group quant to W2
        sc_best = torch.pow(s_in, best_alpha.unsqueeze(-1))
        sc_best = sc_best / sc_best.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        sc_u = sc_best.unsqueeze(1)
        w1_v.copy_((fake_quantize_asym_group(w1_f * sc_u, bits, group_size) / sc_u).to(w1_v.dtype))
        w3_v.copy_((fake_quantize_asym_group(w3_f * sc_u, bits, group_size) / sc_u).to(w3_v.dtype))
        w2_v.copy_(fake_quantize_asym_group(w2_v, bits, group_size))

    eff_bits = bits + 32.0 / float(group_size)
    total_params = n_layers * E * 3 * D * I
    vram_gb = (total_params * eff_bits) / (8.0 * (1024.0 ** 3))
    bf16_gb = (total_params * 16.0) / (8.0 * (1024.0 ** 3))
    return {
        "bits": float(bits),
        "eff_bits": eff_bits,
        "expert_vram_gb": vram_gb,
        "bf16_expert_vram_gb": bf16_gb,
        "vram_compression": bf16_gb / max(vram_gb, 1e-9),
        "mean_awq_alpha": float(np.mean(chosen_alphas)),
    }


# ---------------------------------------------------------------------------
# 6. Real GPU Wall-Clock Speedup Benchmark (Native xLLM + Fast Decode BMM)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def benchmark_r3_gpu_speed(
    model: Any,
    topo: list[dict[str, Any]],
    schedules: dict[str, tuple[list[int], list[float]]],
    device: torch.device,
    *,
    run_fast_decode: bool = True,
    vocab_size: int = 32000,
    n_warmup: int = 8,
    n_iters: int = 35,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def _bench_suite(suite_tag: str, shapes: list[tuple[str, int, int]], fast_bmm: bool) -> None:
        base_ms_by_shape: dict[str, float] = {}
        for shape_name, bsz, slen in shapes:
            inp = torch.randint(10, min(vocab_size, 10000), (bsz, slen), device=device, dtype=torch.long)
            for sched_name, (vk, vg) in schedules.items():
                mode = "native" if (sched_name == "bf16_base" and not fast_bmm) else "static"
                ex = LoopSchedV2Executor(
                    model, topo, mode=mode, visit_k=vk, visit_gamma=vg,
                    renorm_on_drop2=False, fast_small_batch=fast_bmm, track_stats=False,
                )
                with ex:
                    for _ in range(n_warmup):
                        ex.reset_pass()
                        _ = model(inp, multi_segments=False, cache=None)
                    torch.cuda.synchronize(device)
                    times_ms: list[float] = []
                    for _ in range(n_iters):
                        ev0 = torch.cuda.Event(enable_timing=True)
                        ev1 = torch.cuda.Event(enable_timing=True)
                        ev0.record()
                        ex.reset_pass()
                        _ = model(inp, multi_segments=False, cache=None)
                        ev1.record()
                        torch.cuda.synchronize(device)
                        times_ms.append(float(ev0.elapsed_time(ev1)))
                med_ms = float(np.median(times_ms))
                iqr_ms = float(np.percentile(times_ms, 75) - np.percentile(times_ms, 25))
                if sched_name == "bf16_base":
                    base_ms_by_shape[shape_name] = med_ms
                speedup = float(base_ms_by_shape[shape_name] / max(med_ms, 1e-6))
                cut_pct = 100.0 * (1.0 - float(sum(vk)) / (2.0 * len(vk)))
                tok_s = float((bsz * slen) / (med_ms / 1000.0))
                rows.append(
                    {
                        "backend": suite_tag,
                        "shape": shape_name,
                        "schedule": sched_name,
                        "cut_pct": cut_pct,
                        "median_ms": med_ms,
                        "iqr_ms": iqr_ms,
                        "speedup": speedup,
                        "tok_per_sec": tok_s,
                        "k0_count": int(sum(1 for k in vk if k == 0)),
                        "k1_count": int(sum(1 for k in vk if k == 1)),
                        "k2_count": int(sum(1 for k in vk if k == 2)),
                    }
                )

    _bench_suite(
        "native_xllm",
        [
            ("decode_B1_L1", 1, 1),
            ("decode_B8_L1", 8, 1),
            ("decode_B32_L1", 32, 1),
            ("prefill_B8_L256", 8, 256),
        ],
        fast_bmm=False,
    )
    if run_fast_decode:
        _bench_suite(
            "fast_bmm",
            [
                ("fast_decode_B1_L1", 1, 1),
                ("fast_decode_B8_L1", 8, 1),
            ],
            fast_bmm=True,
        )
    return rows


# ---------------------------------------------------------------------------
# 7. Full Per-Model R3 Execution
# ---------------------------------------------------------------------------

def run_model_r3(
    model_key: str,
    mspec: dict[str, Any],
    *,
    model_dir: Path,
    dataset_dir: Path,
    mcq_raw: dict[str, list[dict[str, Any]]],
    cfg: dict[str, Any],
    device: torch.device,
    run_obj: Any,
) -> dict[str, Any]:
    artifact_dir = model_dir / mspec["local_dir"]
    seed = int(cfg.get("seed", 20261126))
    print(f"\n[r3:{model_key}] Loading {artifact_dir} ...", flush=True)
    model, tokenizer, _mcfg = load_ifm_model_for_autograd(
        artifact_dir, device=str(device), attention_backend="sdpa"
    )
    n_layers = len(model.layers)
    loop_start = int(model.loop_start_layer)
    loop_end = int(model.loop_end_layers)
    n_loops = int(model.loop_times)
    topo = build_visit_topology(n_layers, loop_start, loop_end, n_loops)
    V = len(topo)

    splits = build_disjoint_text_code_splits(
        dataset_dir,
        lambda t: tokenizer.encode(t, bos=True, eos=False),
        seq_len=int(cfg.get("seq_len", 256)),
        n_calib_a=int(cfg.get("n_calib_a", 16)),
        n_calib_b=int(cfg.get("n_calib_b", 8)),
        n_adapt_train=4,
        n_eval_heldout=int(cfg.get("n_eval", 64)),
        seed=seed,
    )
    calib_a = splits["calib_a"]
    calib_b = splits["calib_b"]
    eval_seqs = splits["eval_heldout"]
    mcq_enc = {k: prepare_mcq_encoded(v, tokenizer) for k, v in mcq_raw.items()}
    raw_pad = getattr(tokenizer, "pad_id", None)
    if raw_pad is None or int(raw_pad) < 0:
        raw_pad = getattr(tokenizer, "eos_id", None)
    if raw_pad is None or int(raw_pad) < 0:
        raw_pad = getattr(tokenizer, "bos_id", 0)
    pad_id = max(0, int(raw_pad))

    # 1) Startup Exactness & Parity Verification (including fast_decode_bmm!)
    nll_bsz = 8
    mcq_bsz = 32
    fast_bmm_ok = False
    with run_obj.phase(f"{model_key}_verify", total=1, unit="check") as pv:
        with torch.inference_mode():
            t_in = torch.tensor([eval_seqs[0]["tokens"][:64], eval_seqs[1]["tokens"][:64]], device=device, dtype=torch.long)
            ref_logits = model(t_in, multi_segments=False, cache=None)[0].float()
            ex_v = LoopSchedV2Executor(model, topo, mode="verify_k2", track_stats=False)
            with ex_v:
                ex_v.reset_pass()
                our_logits = model(t_in, multi_segments=False, cache=None)[0].float()
            k2_diff = float((ref_logits - our_logits).abs().max().item())

            # Fast small-batch BMM exactness at B=1, L=1 and B=8, L=1
            d1_in = torch.tensor([[eval_seqs[0]["tokens"][0]]], device=device, dtype=torch.long)
            d1_ref = model(d1_in, multi_segments=False, cache=None)[0].float()
            ex_f = LoopSchedV2Executor(model, topo, mode="verify_k2", fast_small_batch=True, track_stats=False)
            with ex_f:
                ex_f.reset_pass()
                d1_fast = model(d1_in, multi_segments=False, cache=None)[0].float()
            d1_fast_diff = float((d1_ref - d1_fast).abs().max().item())

            d8_in = torch.tensor([[eval_seqs[i]["tokens"][0]] for i in range(8)], device=device, dtype=torch.long)
            d8_ref = model(d8_in, multi_segments=False, cache=None)[0].float()
            with ex_f:
                ex_f.reset_pass()
                d8_fast = model(d8_in, multi_segments=False, cache=None)[0].float()
            d8_fast_diff = float((d8_ref - d8_fast).abs().max().item())
            fast_bmm_ok = (d1_fast_diff < 0.25 and d8_fast_diff < 0.25)

            # Batched causal parity check
            ex_nat = LoopSchedV2Executor(model, topo, mode="native", track_stats=False)
            nll_b1 = eval_nll_batched(model, eval_seqs[:8], ex_nat, device, batch_size=1)["nll"]
            nll_b8 = eval_nll_batched(model, eval_seqs[:8], ex_nat, device, batch_size=8)["nll"]
            nll_batch_diff = abs(nll_b1 - nll_b8)
            acc_b1_25, fl_b1_25 = eval_mcq_batched(model, mcq_enc["arc_c"][:25], ex_nat, device, batch_size=1, pad_token_id=pad_id)
            acc_b32_25, fl_b32_25 = eval_mcq_batched(model, mcq_enc["arc_c"][:25], ex_nat, device, batch_size=32, pad_token_id=pad_id)
            agree_25 = float(np.mean([a == b for a, b in zip(fl_b1_25, fl_b32_25)]))

        print(
            f"[r3:{model_key}] Exactness OK: k2_diff={k2_diff:.2e}, fast_bmm(B1)={d1_fast_diff:.3e}, "
            f"fast_bmm(B8)={d8_fast_diff:.3e} (fast_bmm_ok={fast_bmm_ok}) | "
            f"Batch parity: nll_diff={nll_batch_diff:.5f}, arc25_agree={agree_25*100:.1f}% "
            f"(B1={acc_b1_25*100:.1f}% vs B32={acc_b32_25*100:.1f}%)",
            flush=True,
        )
        if k2_diff > 1e-2:
            raise RuntimeError(f"k2 dispatcher exactness failed: {k2_diff}")
        if not (nll_batch_diff < 0.01 and agree_25 >= 0.92):
            raise RuntimeError(f"Batch parity failed: nll_diff={nll_batch_diff}, agree_25={agree_25}")
        pv.step(1)

    gc.collect()
    torch.cuda.empty_cache()

    # 2) Calibration Phase
    with run_obj.phase(f"{model_key}_calib", total=4, unit="steps") as pc:
        t_cal0 = time.time()
        calib = calibrate_r3(model, topo, calib_a, calib_b, device, batch_size=nll_bsz, phase_ctx=pc)
        print(
            f"[r3:{model_key}] Calibration done in {time.time() - t_cal0:.1f}s | "
            f"calib_b alpha ΔNLLs={calib['alpha_calib_b_dnlls']} -> best_alpha={calib['best_alpha']:.2f}",
            flush=True,
        )
        print(f"[r3:{model_key}] Per-stage geometry & sensitivity:", flush=True)
        for st in calib["stage_summary"]:
            print(
                f"  stage={st['stage']:8s} (V={st['n_visits']:2d}) | cos={st['cos_e1_e2']:+.3f} "
                f"| MSE_ren/noren={st['mse_ratio_ren_vs_noren']:.2f}x | gamma_s*={st['gamma_stage']:.3f} "
                f"| k1_ren={st['k1_renorm']:+.4f} | k1_noren={st['k1_noren']:+.4f} "
                f"| k1_opt={st['k1_opt']:+.4f} | k0={st['k0_sens']:+.4f}",
                flush=True,
            )
            pc.row(
                f"{model_key}_geom",
                stage=st["stage"],
                cos_e1_e2=st["cos_e1_e2"],
                mse_ratio=st["mse_ratio_ren_vs_noren"],
                gamma_opt=st["gamma_stage"],
                k1_renorm=st["k1_renorm"],
                k1_noren=st["k1_noren"],
                k0_sens=st["k0_sens"],
            )
        ig = calib["iter_greedy"]
        print(
            f"[r3:{model_key}] IterGreedy -50% accepted {len(ig['swap_trace'])} swaps -> "
            f"(k0/k1/k2)=({sum(k==0 for k in ig['vk_50'])}/{sum(k==1 for k in ig['vk_50'])}/{sum(k==2 for k in ig['vk_50'])}) "
            f"| calib_b NLL={ig['calib_b_50']:.4f}",
            flush=True,
        )
        for sw in ig["swap_trace"]:
            print(
                f"    swap {sw['swap']:2d}: +{sw['v_up']:3d}({sw['stage_up']}) / -{sw['v_dn']:3d}({sw['stage_dn']}) "
                f"-> calib_a={sw['calib_a_nll']:.4f}, calib_b={sw['calib_b_nll']:.4f}",
                flush=True,
            )
        print(
            f"[r3:{model_key}] IterGreedy -62.5% -> "
            f"(k0/k1/k2)=({sum(k==0 for k in ig['vk_62'])}/{sum(k==1 for k in ig['vk_62'])}/{sum(k==2 for k in ig['vk_62'])})",
            flush=True,
        )
        pc.step(4)

    # Build schedules & dynamic sensitivity weights
    vg_one = [1.0] * V
    vg_stage = calib["visit_gamma_stage"]
    imp_fact = calib["imp_factorized"]
    imp_layer = calib["imp_layer_only"]
    imp_stage = calib["imp_stage_only"]

    def norm_w(arr: np.ndarray) -> np.ndarray:
        w = np.sqrt(np.maximum(arr, 1e-6))
        return w / max(float(np.median(w)), 1e-6)

    w_unif = np.ones(V, dtype=np.float64)
    w_layer = norm_w(imp_layer)
    w_stage = norm_w(imp_stage)
    w_fact = norm_w(imp_fact)

    vk25_layer_st = solve_pure_k12_schedule(imp_layer, 0.25)
    vk25_loop_st = solve_pure_k12_schedule(imp_fact, 0.25)
    vk38_layer_st = solve_pure_k12_schedule(imp_layer, 0.375)
    vk38_stage_st = build_stage_only_cut38_schedule(topo, calib["stage_per_visit_sens"])
    vk38_loop_st = solve_pure_k12_schedule(imp_fact, 0.375)
    vk_nat_bound, nat_bound_label = build_natural_boundary_k12_schedule(topo, calib["layer_per_visit_sens"])
    vk50_all_k1 = [1] * V
    vk50_iter = calib["iter_greedy"]["vk_50"]
    vk62_unif = [0 if (v % 4 == 2) else 1 for v in range(V)]
    vk62_iter = calib["iter_greedy"]["vk_62"]

    def make_dyn(name: str, cut_frac: float, w: np.ndarray, *, renorm: bool, vg: list[float]) -> dict[str, Any]:
        tau = calibrate_dynamic_threshold(calib["visit_ratios"], [2] * V, w, cut_frac)
        return {"name": name, "mode": "dynamic", "vk": [2] * V, "vg": vg, "renorm": renorm, "w": w, "tau": tau}

    def make_stat(name: str, vk: list[int], *, renorm: bool, vg: list[float]) -> dict[str, Any]:
        return {"name": name, "mode": "static", "vk": vk, "vg": vg, "renorm": renorm, "w": None, "tau": -1.0}

    bf16_conditions: list[dict[str, Any]] = [
        {"name": "bf16_base", "mode": "native", "vk": [2] * V, "vg": vg_one, "renorm": False, "w": None, "tau": -1.0},
        # --- Regime A: -25.0% ---
        make_dyn("cut25_unif_renorm", 0.25, w_unif, renorm=True, vg=vg_one),
        make_dyn("cut25_unif_noren", 0.25, w_unif, renorm=False, vg=vg_one),
        make_stat("cut25_layer_only_static", vk25_layer_st, renorm=False, vg=vg_one),
        make_stat("cut25_loopsched_static_k12", vk25_loop_st, renorm=False, vg=vg_one),
        make_dyn("cut25_layer_only_dyn", 0.25, w_layer, renorm=False, vg=vg_one),
        make_dyn("cut25_stage_only_dyn", 0.25, w_stage, renorm=False, vg=vg_one),
        make_dyn("cut25_loopsched_dyn_k12", 0.25, w_fact, renorm=False, vg=vg_one),
        # --- Regime B: -37.5% ---
        make_dyn("cut38_unif_renorm", 0.375, w_unif, renorm=True, vg=vg_one),
        make_dyn("cut38_unif_noren", 0.375, w_unif, renorm=False, vg=vg_one),
        make_stat("cut38_layer_only_static", vk38_layer_st, renorm=False, vg=vg_one),
        make_stat("cut38_stage_only_static", vk38_stage_st, renorm=False, vg=vg_one),
        make_stat("cut38_loopsched_static_k12", vk38_loop_st, renorm=False, vg=vg_stage),
        make_dyn("cut38_layer_only_dyn", 0.375, w_layer, renorm=False, vg=vg_one),
        make_dyn("cut38_loopsched_dyn_k12", 0.375, w_fact, renorm=False, vg=vg_one),
        # --- Regime C: Natural Boundary-Protected Max k12 (-43.8% Ouro / -46.4% Huginn) ---
        make_stat("cut_nat_boundary_k12_noren", vk_nat_bound, renorm=False, vg=vg_one),
        make_stat("cut_nat_boundary_k12_opt", vk_nat_bound, renorm=False, vg=vg_stage),
        # --- Regime D: -50.0% ---
        make_stat("cut50_all_top1_renorm", vk50_all_k1, renorm=True, vg=vg_one),
        make_stat("cut50_all_top1_noren", vk50_all_k1, renorm=False, vg=vg_one),
        make_stat("cut50_loopsched_iter_greedy", vk50_iter, renorm=False, vg=vg_one),
        # --- Regime E: -62.5% ---
        make_stat("cut62_unif_k01", vk62_unif, renorm=False, vg=vg_one),
        make_stat("cut62_loopsched_iter_greedy", vk62_iter, renorm=False, vg=vg_one),
    ]

    results: dict[str, dict[str, Any]] = {}
    base_per_seq: np.ndarray | None = None
    total_conds = len(bf16_conditions) + 2  # +2 Bonus W3-Pool-AWQ orthogonality conditions

    def _eval_and_record(
        idx_c: int,
        c: dict[str, Any],
        pe: Any,
    ) -> dict[str, Any]:
        nonlocal base_per_seq
        t_c0 = time.time()
        ex = LoopSchedV2Executor(
            model, topo, mode=c["mode"], visit_k=c["vk"], visit_gamma=c["vg"],
            renorm_on_drop2=c["renorm"], visit_weights=c["w"], dyn_threshold=c["tau"],
        )
        nll_res = eval_nll_batched(model, eval_seqs, ex, device, batch_size=nll_bsz)
        cut_pct = nll_res["cut_pct"]

        mcq_accs: dict[str, float] = {}
        mcq_flags: dict[str, list[int]] = {}
        for ds_name, enc_items in mcq_enc.items():
            acc, fl = eval_mcq_batched(model, enc_items, ex, device, batch_size=mcq_bsz, pad_token_id=pad_id)
            mcq_accs[ds_name] = acc
            mcq_flags[ds_name] = fl
        avg_mcq = float(np.mean(list(mcq_accs.values())))

        if c["name"] == "bf16_base":
            base_per_seq = np.asarray(nll_res["per_seq"], dtype=np.float64)
        assert base_per_seq is not None
        diffs = np.asarray(nll_res["per_seq"], dtype=np.float64) - base_per_seq
        dnll, dnll_lo, dnll_hi = paired_bootstrap_ci(diffs)
        elapsed_c = time.time() - t_c0

        rec = {
            "name": c["name"],
            "mode": c["mode"],
            "cut_pct": cut_pct,
            "nll": nll_res["nll"],
            "text_nll": nll_res["text_nll"],
            "code_nll": nll_res["code_nll"],
            "delta_nll": dnll,
            "delta_nll_ci": [dnll_lo, dnll_hi],
            "per_seq": nll_res["per_seq"],
            "arc_c": mcq_accs["arc_c"],
            "arc_e": mcq_accs["arc_e"],
            "obqa": mcq_accs["obqa"],
            "avg_mcq": avg_mcq,
            "mcq_flags": mcq_flags,
            "k0_count": int(sum(1 for k in c["vk"] if k == 0)),
            "k1_count": int(sum(1 for k in c["vk"] if k == 1)),
            "k2_count": int(sum(1 for k in c["vk"] if k == 2)),
            "elapsed_s": elapsed_c,
        }
        results[c["name"]] = rec
        print(
            f"[r3:{model_key}] {c['name']:29s} cut={cut_pct:5.1f}% | NLL={nll_res['nll']:.4f} (Δ={dnll:+.4f}) "
            f"| ARC-C={mcq_accs['arc_c']*100:5.1f}% ARC-E={mcq_accs['arc_e']*100:5.1f}% OBQA={mcq_accs['obqa']*100:5.1f}% "
            f"Avg={avg_mcq*100:5.1f}% | ({rec['k0_count']}/{rec['k1_count']}/{rec['k2_count']}) | t={elapsed_c:.1f}s",
            flush=True,
        )
        pe.row(
            f"{model_key}_results",
            policy=c["name"],
            cut_pct=cut_pct,
            delta_nll=dnll,
            arc_c=mcq_accs["arc_c"],
            arc_e=mcq_accs["arc_e"],
            obqa=mcq_accs["obqa"],
            avg_mcq=avg_mcq,
        )
        pe.step(idx_c, latest_avg_mcq=avg_mcq, latest_dnll=dnll)
        return rec

    # 3) Evaluate all 22 BF16 conditions
    with run_obj.phase(f"{model_key}_eval", total=len(bf16_conditions), unit="conds") as pe:
        for idx_c, c in enumerate(bf16_conditions, start=1):
            _eval_and_record(idx_c, c, pe)

        # 4) Run BF16 Real GPU Wall-Clock Benchmark BEFORE modifying weights for W3-AWQ!
        gc.collect()
        torch.cuda.empty_cache()

    with run_obj.phase(f"{model_key}_speed", total=1, unit="bench") as ps:
        bench_scheds = {
            "bf16_base": ([2] * V, vg_one),
            "cut38_loopsched_static_k12": (vk38_loop_st, vg_stage),
            "cut_nat_boundary_k12": (vk_nat_bound, vg_one),
            "cut50_all_top1": (vk50_all_k1, vg_one),
            "cut50_loopsched_iter_greedy": (vk50_iter, vg_one),
            "cut62_loopsched_iter_greedy": (vk62_iter, vg_one),
        }
        speed_rows = benchmark_r3_gpu_speed(
            model, topo, bench_scheds, device, run_fast_decode=fast_bmm_ok
        )
        print(f"\n[r3:{model_key}] Real GPU Wall-Clock Latency Benchmark (Native xLLM + Fast Decode BMM):", flush=True)
        for r in speed_rows:
            print(
                f"  [{r['backend']:11s}] {r['shape']:17s} | {r['schedule']:27s} (cut={r['cut_pct']:4.1f}%) | "
                f"{r['median_ms']:7.2f} ms (IQR {r['iqr_ms']:.2f}) | {r['speedup']:.3f}x | {r['tok_per_sec']:,.0f} tok/s",
                flush=True,
            )
            ps.row(
                f"{model_key}_speed",
                shape=r["shape"],
                schedule=r["schedule"],
                cut_pct=r["cut_pct"],
                median_ms=r["median_ms"],
                speedup=r["speedup"],
            )
        ps.step(1)

    # 5) Bonus Regime F: Apply W3-g128 Trajectory-Pooled AWQ in-place & check orthogonality with LoopSched!
    with run_obj.phase(f"{model_key}_quant_ortho", total=2, unit="conds") as pq:
        t_q0 = time.time()
        awq_meta = apply_w3_pool_awq_inplace(model, calib["act_rms_cpu"], bits=3, group_size=128)
        print(
            f"\n[r3:{model_key}] Applied W3-g128 Pool-AWQ in {time.time() - t_q0:.1f}s: "
            f"{awq_meta['bf16_expert_vram_gb']:.2f}GB -> {awq_meta['expert_vram_gb']:.2f}GB "
            f"({awq_meta['vram_compression']:.2f}x VRAM compression, mean alpha={awq_meta['mean_awq_alpha']:.2f})",
            flush=True,
        )
        quant_conds = [
            {"name": "w3_pool_awq_only", "mode": "native", "vk": [2] * V, "vg": vg_one, "renorm": False, "w": None, "tau": -1.0},
            make_stat(
                "w3_pool_awq_plus_loopsched",
                vk_nat_bound if "coda" in {t["stage"] for t in topo} else vk38_loop_st,
                renorm=False,
                vg=vg_one,
            ),
        ]
        for qi, qc in enumerate(quant_conds, start=1):
            _eval_and_record(qi, qc, pq)

    # 6) Compute Paired Bootstrap Contrasts (positive = first beats second)
    def contrast(a: str, b: str) -> dict[str, Any]:
        ra, rb = results[a], results[b]
        nll_gain = np.asarray(rb["per_seq"], dtype=np.float64) - np.asarray(ra["per_seq"], dtype=np.float64)
        ng_m, ng_lo, ng_hi = paired_bootstrap_ci(nll_gain)
        wins = int(np.sum(nll_gain > 0))
        fa = np.concatenate([np.asarray(ra["mcq_flags"][k], dtype=np.float64) for k in ("arc_c", "arc_e", "obqa")])
        fb = np.concatenate([np.asarray(rb["mcq_flags"][k], dtype=np.float64) for k in ("arc_c", "arc_e", "obqa")])
        mg_m, mg_lo, mg_hi = paired_bootstrap_ci((fa - fb) * 100.0)
        return {
            "nll_gain": ng_m,
            "nll_ci": [ng_lo, ng_hi],
            "nll_wins": f"{wins}/{len(nll_gain)}",
            "mcq_pooled_gain_pp": mg_m,
            "mcq_pooled_ci_pp": [mg_lo, mg_hi],
        }

    contrast_pairs = [
        # Pillar 1: Orthogonal Shrinkage (NoRen / Stage gamma_s* vs Renorm)
        ("cut25_unif_noren", "cut25_unif_renorm"),
        ("cut38_unif_noren", "cut38_unif_renorm"),
        ("cut50_all_top1_noren", "cut50_all_top1_renorm"),
        ("cut_nat_boundary_k12_opt", "cut_nat_boundary_k12_noren"),
        # Pillar 2: Pure k12 Loop-Stage Static vs Layer-Only Static & Dynamic Renorm!
        ("cut25_loopsched_static_k12", "cut25_layer_only_static"),
        ("cut25_loopsched_static_k12", "cut25_unif_renorm"),
        ("cut38_loopsched_static_k12", "cut38_layer_only_static"),
        ("cut38_stage_only_static", "cut38_layer_only_static"),
        ("cut38_loopsched_static_k12", "cut38_unif_renorm"),
        # Pillar 3: Factorized Stage x Layer Dynamic vs Layer-Only / Stage-Only / Uniform!
        ("cut25_loopsched_dyn_k12", "cut25_layer_only_dyn"),
        ("cut25_loopsched_dyn_k12", "cut25_stage_only_dyn"),
        ("cut25_loopsched_dyn_k12", "cut25_unif_noren"),
        ("cut38_loopsched_dyn_k12", "cut38_layer_only_dyn"),
        ("cut38_loopsched_dyn_k12", "cut38_unif_noren"),
        ("cut38_loopsched_dyn_k12", "cut38_unif_renorm"),
        # Pillar 4: Non-Additive IterGreedy vs All-Top1 (-50%) & Uniform-k01 (-62.5%)!
        ("cut50_loopsched_iter_greedy", "cut50_all_top1_noren"),
        ("cut50_loopsched_iter_greedy", "cut50_all_top1_renorm"),
        ("cut62_loopsched_iter_greedy", "cut62_unif_k01"),
        # Bonus Pillar 6: Quantization + Scheduling orthogonality
        ("w3_pool_awq_plus_loopsched", "w3_pool_awq_only"),
    ]
    contrasts = {f"{a}_vs_{b}": contrast(a, b) for a, b in contrast_pairs}

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Strip large raw flag/per_seq arrays for clean JSON summary while keeping all stats
    compact_results = {
        k: {kk: vv for kk, vv in v.items() if kk not in ("mcq_flags", "per_seq")}
        for k, v in results.items()
    }
    return {
        "model_key": model_key,
        "V": V,
        "nat_bound_label": nat_bound_label,
        "best_alpha": calib["best_alpha"],
        "alpha_calib_b_dnlls": calib["alpha_calib_b_dnlls"],
        "stage_summary": calib["stage_summary"],
        "iter_greedy_50_swaps": calib["iter_greedy"]["swap_trace"],
        "awq_meta": awq_meta,
        "results": compact_results,
        "speed_rows": speed_rows,
        "contrasts": contrasts,
    }


def write_r3_report(all_out: dict[str, Any], out_path: Path) -> str:
    lines: list[str] = []
    lines.append("# Round 3: LoopSched-MoE v2 — Complete ICML 2027 Validation Report\n")
    lines.append(
        f"- GPU: `{all_out['gpu_index']}` ({all_out['gpu_name']}), total elapsed `{all_out['elapsed_s']:.1f}s`\n"
        f"- Evaluation: Held-out NLL (`n=64`: 32 WikiText + 32 HumanEval) + 2,272 MCQ "
        f"(Full ARC-C `1,172` + ARC-E `600` + Full OBQA `500`)\n"
    )
    for mk, mo in all_out["models"].items():
        lines.append(
            f"\n## Model: `{mk}` (V={mo['V']}, best_alpha={mo['best_alpha']:.2f}, "
            f"nat_bound={mo['nat_bound_label']})\n"
        )
        lines.append(f"- Disjoint `calib_b` α sweep ΔNLL (`-37.5%`): `{mo['alpha_calib_b_dnlls']}`\n")
        lines.append("### 1. Per-Stage Geometry & Sensitivity")
        lines.append("| Stage | V | cos(e1,e2) | MSE_ren/noren | γ_s* | k=1 Renorm | k=1 NoRen | k=1 Opt γ_s* | k=0 Skip |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for st in mo["stage_summary"]:
            lines.append(
                f"| `{st['stage']}` | {st['n_visits']} | {st['cos_e1_e2']:+.3f} | "
                f"{st['mse_ratio_ren_vs_noren']:.2f}x | {st['gamma_stage']:.3f} | "
                f"{st['k1_renorm']:+.4f} | {st['k1_noren']:+.4f} | {st['k1_opt']:+.4f} | {st['k0_sens']:+.4f} |"
            )

        lines.append("\n### 2. Main Quality-Compute Pareto Table (BF16 + W3-AWQ Orthogonality)")
        lines.append("| Policy | Cut% | NLL | ΔNLL (95% CI) | ARC-C | ARC-E | OBQA | Avg MCQ | (k0/k1/k2) |")
        lines.append("|---|---:|---:|---|---:|---:|---:|---:|---|")
        for r in mo["results"].values():
            lo, hi = r["delta_nll_ci"]
            lines.append(
                f"| `{r['name']}` | {r['cut_pct']:.1f}% | {r['nll']:.4f} | "
                f"{r['delta_nll']:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
                f"{r['arc_c']*100:.1f}% | {r['arc_e']*100:.1f}% | {r['obqa']*100:.1f}% | "
                f"**{r['avg_mcq']*100:.1f}%** | {r['k0_count']}/{r['k1_count']}/{r['k2_count']} |"
            )

        lines.append("\n### 3. Real GPU Wall-Clock Speedup (Native xLLM + Zero-Sync Fast Decode BMM)")
        lines.append("| Backend | Shape | Schedule | Cut% | Median (ms) | IQR (ms) | Speedup | Tok/s |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|")
        for sr in mo["speed_rows"]:
            lines.append(
                f"| `{sr['backend']}` | `{sr['shape']}` | `{sr['schedule']}` | {sr['cut_pct']:.1f}% | "
                f"{sr['median_ms']:.2f} | {sr['iqr_ms']:.2f} | **{sr['speedup']:.3f}x** | {sr['tok_per_sec']:,.0f} |"
            )

        lines.append("\n### 4. Paired Bootstrap Contrasts (positive = first beats second)")
        for cname, cval in mo["contrasts"].items():
            nlo, nhi = cval["nll_ci"]
            mlo, mhi = cval["mcq_pooled_ci_pp"]
            lines.append(
                f"- `{cname}`: NLL `{cval['nll_gain']:+.4f}` CI `[{nlo:+.4f}, {nhi:+.4f}]` ({cval['nll_wins']} wins) | "
                f"Pooled MCQ `{cval['mcq_pooled_gain_pp']:+.2f} pp` CI `[{mlo:+.2f}, {mhi:+.2f}]`"
            )

    md = "\n".join(lines) + "\n"
    out_path.write_text(md, encoding="utf-8")
    return md


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--xllm-loop-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    t0 = time.time()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu_idx, gpu_lock, gpu_status = acquire_idle_gpu(cfg.get("preferred_gpu"))
    try:
        device = torch.device(f"cuda:{gpu_idx}")
        torch.cuda.set_device(device)
        print(
            f"[r3] Acquired idle GPU {gpu_idx}: {gpu_status.name} "
            f"(mem={gpu_status.mem_used_mib}MiB, util={gpu_status.util_pct}%)",
            flush=True,
        )

        boot = bootstrap_xllm(args.xllm_loop_dir, attention_backend="sdpa")
        print(f"[r3] xllm bootstrap: {json.dumps(boot)}", flush=True)

        seed = int(cfg.get("seed", 20261126))
        mcq_raw = {
            "arc_c": load_arc_challenge_samples(args.dataset_dir, n_samples=int(cfg.get("n_arc_c", 0)), seed=seed),
            "arc_e": load_arc_easy_samples(args.dataset_dir, n_samples=int(cfg.get("n_arc_e", 600)), seed=seed),
            "obqa": load_openbookqa_samples(args.dataset_dir, n_samples=int(cfg.get("n_obqa", 500)), seed=seed),
        }
        print(f"[r3] Loaded MCQ suites: ARC-C={len(mcq_raw['arc_c'])}, ARC-E={len(mcq_raw['arc_e'])}, OBQA={len(mcq_raw['obqa'])}", flush=True)

        model_specs = cfg["models"]
        plan = [
            {"phase": "setup", "of": 1, "unit": "env"},
            {"phase": "download", "of": 1, "unit": "assets"},
        ]
        for mk in model_specs:
            plan.extend(
                [
                    {"phase": f"{mk}_verify", "of": 1, "unit": "check"},
                    {"phase": f"{mk}_calib", "of": 4, "unit": "steps"},
                    {"phase": f"{mk}_eval", "of": 22, "unit": "conds"},
                    {"phase": f"{mk}_speed", "of": 1, "unit": "bench"},
                    {"phase": f"{mk}_quant_ortho", "of": 2, "unit": "conds"},
                ]
            )
        plan.append({"phase": "report", "of": 1, "unit": "file"})
        orx.plan(plan)
        run_obj = orx.progress(total=1, unit="run")

        all_out: dict[str, Any] = {
            "gpu_index": gpu_idx,
            "gpu_name": gpu_status.name,
            "models": {},
        }
        for mk, mspec in model_specs.items():
            all_out["models"][mk] = run_model_r3(
                mk,
                mspec,
                model_dir=Path(args.model_dir),
                dataset_dir=Path(args.dataset_dir),
                mcq_raw=mcq_raw,
                cfg=cfg,
                device=device,
                run_obj=run_obj,
            )

        all_out["elapsed_s"] = time.time() - t0
        with run_obj.phase("report", total=1, unit="file") as pr:
            json_path = out_dir / "results_r3_loopsched_v2.json"
            json_path.write_text(json.dumps(all_out, indent=2), encoding="utf-8")
            md = write_r3_report(all_out, out_dir / "report_r3_loopsched_v2.md")
            print("\n========== REPORT (report_r3_loopsched_v2.md) ==========")
            print(md, flush=True)
            print("========== END REPORT ==========", flush=True)

            summary_m: dict[str, float] = {}
            for mk, mo in all_out["models"].items():
                summary_m[f"{mk}_cut38_static_vs_layer_nll"] = mo["contrasts"]["cut38_loopsched_static_k12_vs_cut38_layer_only_static"]["nll_gain"]
                summary_m[f"{mk}_cut38_dyn_vs_layer_nll"] = mo["contrasts"]["cut38_loopsched_dyn_k12_vs_cut38_layer_only_dyn"]["nll_gain"]
                summary_m[f"{mk}_nat_bound_dnll"] = mo["results"]["cut_nat_boundary_k12_noren"]["delta_nll"]
                summary_m[f"{mk}_nat_bound_avg_mcq"] = mo["results"]["cut_nat_boundary_k12_noren"]["avg_mcq"]
                summary_m[f"{mk}_cut50_iter_vs_top1_nll"] = mo["contrasts"]["cut50_loopsched_iter_greedy_vs_cut50_all_top1_noren"]["nll_gain"]
                summary_m[f"{mk}_cut50_iter_vs_top1_mcq"] = mo["contrasts"]["cut50_loopsched_iter_greedy_vs_cut50_all_top1_noren"]["mcq_pooled_gain_pp"]
                summary_m[f"{mk}_cut62_iter_vs_unif_nll"] = mo["contrasts"]["cut62_loopsched_iter_greedy_vs_cut62_unif_k01"]["nll_gain"]
                summary_m[f"{mk}_w3_plus_sched_avg_mcq"] = mo["results"]["w3_pool_awq_plus_loopsched"]["avg_mcq"]
                for sr in mo["speed_rows"]:
                    if sr["shape"] == "prefill_B8_L256" and sr["schedule"] == "cut_nat_boundary_k12":
                        summary_m[f"{mk}_nat_prefill_speedup"] = sr["speedup"]
                    if sr["shape"] == "fast_decode_B1_L1" and sr["schedule"] == "cut_nat_boundary_k12":
                        summary_m[f"{mk}_nat_fast_decode_b1_speedup"] = sr["speedup"]
                    if sr["shape"] == "fast_decode_B1_L1" and sr["schedule"] == "cut50_loopsched_iter_greedy":
                        summary_m[f"{mk}_cut50_fast_decode_b1_speedup"] = sr["speedup"]
            pr.step(1, **summary_m)
    finally:
        gpu_lock.release()


if __name__ == "__main__":
    main()
