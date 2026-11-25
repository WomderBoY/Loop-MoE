"""Round 2-A: LoopSched-MoE — Boundary-Protected & Trajectory-Aware Expert Scheduling.

Child of R1 Probe 2 (1285003c-e163-45af-8d24-c5633339ab6e).
Tests on IFM Ouro-MoE (4 recurrent loops) & IFM Huginn-MoE (prelude + 8 loops + coda):
  1. Theory & Mechanism: Per-visit (layer, loop) empirical cos(e1, e2), closed-form MSE-optimal
     top-1 shrinkage scale gamma_v* = 1 + E[w1*w2*<e1,e2>] / E[w1^2*||e1||^2], and per-visit
     marginal NLL sensitivities Delta(2->1) and Delta(1->0).
  2. Tri-Level Static DP Schedule (LoopSched-Static, k_v in {0, 1, 2}): zero dynamic overhead,
     hardware-native real skipping (k=0 skips router+experts; k=1 dispatches 1xN tokens).
  3. Visit-Sensitivity-Normalized Dynamic Skipping (LoopSched-Dynamic, k in {0, 1, 2} + gamma_v*).
  4. Strong Controlled Baselines at matched active-expert cuts (-25%, -37.5%, -50%, -62.5%):
     - Uniform-Renorm (standard MoE dynamic top-k / ACE style)
     - Uniform-NoRen (isolates geometry of NoRen vs loop awareness)
     - Layer-Only-Opt (ablation: physical layer sensitivity without loop index!)
     - Static-All-Top1 (Renorm & NoRen)
     - Drop-Last-Loops (early-exit / loop-truncation baseline)
  5. Comprehensive Batched Evaluation (exact B=1 vs B>1 causal parity verified at startup):
     - Held-out NLL (n=64: 32 WikiText + 32 HumanEval, seq_len=256) with paired bootstrap 95% CIs
     - Full ARC-Challenge test (n=1,172) + ARC-Easy (n=600) + OpenBookQA (n=500) = 2,272 MCQ items
  6. Real GPU Wall-Clock Prefill (B=8, L=256) & Decode (B=1, 8, 32, L=1) Speedup Benchmark
     using true reduced-expert dispatch (zero fake masking).
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
# Visit Topology Helpers (Exact (layer_id, loop_visit) <-> visit_idx Mapping)
# ---------------------------------------------------------------------------

def build_visit_topology(
    n_layers: int,
    loop_start: int,
    loop_end: int,
    n_loops: int,
) -> list[dict[str, Any]]:
    """Return ordered list of logical layer visits in one forward pass.

    Follows xLLM's exact convention where recurrent layers are range(loop_start, loop_end):
      - Ouro-MoE:   n_layers=28, loop_start=0, loop_end=28, n_loops=4 -> 112 visits
      - Huginn-MoE: n_layers=28, loop_start=8, loop_end=20, n_loops=8 -> 8 + 12*8 + 8 = 112 visits
    """
    visits: list[dict[str, Any]] = []
    # Prelude layers: [0, loop_start)
    for lid in range(0, loop_start):
        visits.append({"vid": len(visits), "layer_id": lid, "loop_idx": 0, "stage": "prelude", "is_rec": False})
    # Recurrent body: [loop_start, loop_end) repeated n_loops times
    rec_layers = list(range(loop_start, loop_end))
    for r in range(n_loops):
        for lid in rec_layers:
            visits.append({"vid": len(visits), "layer_id": lid, "loop_idx": r, "stage": f"loop_{r}", "is_rec": True})
    # Coda layers: [loop_end, n_layers)
    for lid in range(loop_end, n_layers):
        visits.append({"vid": len(visits), "layer_id": lid, "loop_idx": 0, "stage": "coda", "is_rec": False})
    return visits


def lookup_visit_id(
    topo_map: dict[tuple[int, int], int],
    layer_id: int,
    visit_num: int,
) -> int:
    return topo_map[(layer_id, visit_num)]


# ---------------------------------------------------------------------------
# Exact Tri-Level DP Knapsack Solver: Assign k_v in {0, 1, 2} per visit
# ---------------------------------------------------------------------------

def solve_trilevel_dp_schedule(
    cost_k1: np.ndarray,
    cost_k0: np.ndarray,
    *,
    target_total_k: int,
    allow_k0: bool = True,
    k0_forbidden_mask: np.ndarray | None = None,
) -> list[int]:
    """Exactly minimize sum_v cost(v, k_v) subject to sum_v k_v == target_total_k, k_v in {0, 1, 2}.

    cost(v, 2) = 0.0, cost(v, 1) = cost_k1[v], cost(v, 0) = cost_k0[v] (or +inf if forbidden).
    Uses exact O(V^2) dynamic programming.
    """
    V = len(cost_k1)
    min_k = 0 if allow_k0 else V
    if k0_forbidden_mask is not None and allow_k0:
        min_k = int(np.sum(k0_forbidden_mask))
    tgt = int(max(min_k, min(2 * V, target_total_k)))

    INF = 1e30
    dp = np.full((V + 1, 2 * V + 1), INF, dtype=np.float64)
    choice = np.full((V + 1, 2 * V + 1), -1, dtype=np.int32)
    dp[0, 0] = 0.0

    for v in range(V):
        c1 = float(max(cost_k1[v], 0.0))
        can_k0 = allow_k0 and (k0_forbidden_mask is None or not bool(k0_forbidden_mask[v]))
        c0 = float(max(cost_k0[v], 0.0)) if can_k0 else INF
        opts = [(2, 0.0), (1, c1)]
        if can_k0:
            opts.append((0, c0))
        prev = dp[v]
        valid_s = np.nonzero(prev < INF * 0.5)[0]
        for s in valid_s:
            base_cost = prev[s]
            for kv, ck in opts:
                ns = s + kv
                if ns <= 2 * V:
                    cand = base_cost + ck
                    if cand < dp[v + 1, ns]:
                        dp[v + 1, ns] = cand
                        choice[v + 1, ns] = kv

    # Backtrack from exact target (or nearest feasible if target was infeasible)
    if dp[V, tgt] >= INF * 0.5:
        feas = np.nonzero(dp[V] < INF * 0.5)[0]
        tgt = int(feas[np.argmin(np.abs(feas - tgt))])

    sched = [2] * V
    cur = tgt
    for v in range(V - 1, -1, -1):
        kv = int(choice[v + 1, cur])
        sched[v] = kv
        cur -= kv
    return sched


def compute_optimal_gamma(
    sum_w1_sq_e1_sq: float,
    sum_w1_w2_dot: float,
    *,
    clamp_lo: float = 0.85,
    clamp_hi: float = 1.60,
) -> float:
    """Closed-form MSE-optimal top-1 scale gamma* = 1 + E[w1 w2 <e1,e2>] / E[w1^2 ||e1||^2]."""
    if sum_w1_sq_e1_sq <= 1e-12:
        return 1.0
    g = 1.0 + (sum_w1_w2_dot / sum_w1_sq_e1_sq)
    return float(max(clamp_lo, min(clamp_hi, g)))


def calibrate_dynamic_threshold(
    visit_ratios: list[np.ndarray],
    visit_weights: np.ndarray,
    *,
    target_drop_frac: float,
    fixed_k_visits: list[int] | None = None,
) -> float:
    """Calibrate threshold tau on weighted score = ratio * visit_weights[v] to hit target_drop_frac.

    target_drop_frac is fraction of ALL routed expert slots (out of 2*V*N) to drop!
    If fixed_k_visits has some visits already set to k=0 (drops 2 slots) or k=1 (drops 1 slot),
    remaining dynamic drops only occur on visits with fixed_k == 2.
    """
    V = len(visit_ratios)
    total_slots = 0
    already_dropped = 0
    cand_scores: list[np.ndarray] = []

    for v in range(V):
        arr = visit_ratios[v]
        n = int(arr.size)
        if n == 0:
            continue
        total_slots += 2 * n
        fk = 2 if fixed_k_visits is None else int(fixed_k_visits[v])
        if fk == 0:
            already_dropped += 2 * n
        elif fk == 1:
            already_dropped += n
        else:
            cand_scores.append(arr * float(visit_weights[v]))

    target_drops = int(round(target_drop_frac * total_slots))
    need_dyn = target_drops - already_dropped
    if need_dyn <= 0 or not cand_scores:
        return -1.0
    all_s = np.concatenate(cand_scores)
    if need_dyn >= all_s.size:
        return float(all_s.max() + 1.0)
    q = float(need_dyn / all_s.size)
    return float(np.quantile(all_s, q))


# ---------------------------------------------------------------------------
# Real-Dispatch LoopSched-MoE Executor (Zero Fake Masking for k=0 and k=1!)
# ---------------------------------------------------------------------------

class LoopSchedExecutor:
    """Context manager executing real reduced-expert MoE schedules on xLLM Looped MoE models.

    Supports:
      - mode="native": unmodified native model.layers[l].moe
      - mode="verify_k2": executes our dispatcher at k=2 everywhere (for exactness check vs native)
      - mode="static": per-visit schedule k_v in {0, 1, 2} with per-visit top-1 scale gamma_v
        * k_v == 0: skips moe.router AND moe.experts completely (keeps shared expert y1 + residual)
        * k_v == 1: dispatches ONLY 1xN top-1 token-expert pairs through moe.experts (real 2x cut!)
        * k_v == 2: dispatches native top-2
      - mode="dynamic": combines optional fixed k_v == 0/1 visits with token-adaptive dropping on
        k_v == 2 visits via score = (w2/w1) * visit_weights[v] < tau, dispatching ONLY surviving
        (2N - n_drop) token-expert pairs through moe.experts!
    """

    def __init__(
        self,
        model: Any,
        topo: list[dict[str, Any]],
        *,
        mode: str = "native",
        visit_k: list[int] | None = None,
        visit_gamma: list[float] | None = None,
        renorm_on_drop2: bool = False,
        visit_weights: np.ndarray | None = None,
        dyn_threshold: float = -1.0,
        track_stats: bool = True,
    ):
        self.model = model
        self.topo = topo
        self.topo_map = {(d["layer_id"], d["loop_idx"]): d["vid"] for d in topo}
        self.n_layers = len(model.layers)
        self.n_experts = int(model.layers[0].moe.experts.n_local_experts)
        self.mode = mode
        self.visit_k = visit_k if visit_k is not None else [2] * len(topo)
        self.visit_gamma = visit_gamma if visit_gamma is not None else [1.0] * len(topo)
        self.renorm_on_drop2 = renorm_on_drop2
        self.visit_weights = visit_weights
        self.dyn_threshold = float(dyn_threshold)
        self.track_stats = track_stats

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

    def _dispatch_routed_pairs(
        self,
        moe: Any,
        x_2d: torch.Tensor,
        tok_ids: torch.Tensor,
        exp_ids: torch.Tensor,
        w_vals: torch.Tensor,
    ) -> torch.Tensor:
        """Real sparse dispatch through xllm's native moe.experts(x_perm, group_sizes)."""
        if tok_ids.numel() == 0:
            return torch.zeros_like(x_2d)
        order = torch.argsort(exp_ids, stable=False)
        tok_s = tok_ids.index_select(0, order)
        exp_s = exp_ids.index_select(0, order)
        w_s = w_vals.index_select(0, order).to(x_2d.dtype)
        # bincount -> Python list[int] expected by xllm experts.forward
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

        # k_v == 0: TRUE FULL ROUTED-MoE SKIP (skips router AND experts!)
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

        # Fast path: untouched k=2 visit in static mode -> run native moe directly!
        if self.mode == "static" and kv == 2:
            if self.track_stats:
                self.total_slots += 2 * n_tok
                self.active_slots += 2 * n_tok
                self.k2_visits_run += 1
            return orig_fn(x_in, load_balancing_type)

        # Otherwise run norm + shared expert + router + real sparse dispatch
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

        if self.mode == "verify_k2":
            if self.track_stats:
                self.total_slots += n_tok * topk
                self.active_slots += n_tok * topk
                self.k2_visits_run += 1
            tok_ids = torch.arange(n_tok, device=x_in.device).unsqueeze(1).expand(n_tok, topk).reshape(-1)
            exp_ids = idx_2d.reshape(-1)
            w_vals = w_f.reshape(-1)
            y2 = self._dispatch_routed_pairs(moe, x_2d, tok_ids, exp_ids, w_vals).reshape(bsz, slen, dim)
            y = y1 + y2 if y1 is not None else y2
            return moe.residual(y, residual, c=y), aux_loss

        # Identify larger-weight (top1) and smaller-weight (top2) slots per token
        w_min, min_pos = w_f.min(dim=-1)
        max_pos = 1 - min_pos
        w_max = w_f.gather(-1, max_pos.unsqueeze(-1)).squeeze(-1)
        e_max = idx_2d.gather(-1, max_pos.unsqueeze(-1)).squeeze(-1)
        e_min = idx_2d.gather(-1, min_pos.unsqueeze(-1)).squeeze(-1)

        if kv == 1:
            # TRUE TOP-1 DISPATCH: 1xN tokens only!
            if self.track_stats:
                self.total_slots += 2 * n_tok
                self.active_slots += n_tok
                self.k1_visits_run += 1
            tok_ids = torch.arange(n_tok, device=x_in.device)
            w_use = torch.full_like(w_max, scale) if self.renorm_on_drop2 else (w_max * gamma)
            y2 = self._dispatch_routed_pairs(moe, x_2d, tok_ids, e_max, w_use).reshape(bsz, slen, dim)
            y = y1 + y2 if y1 is not None else y2
            return moe.residual(y, residual, c=y), aux_loss

        # Dynamic mode on a k_v == 2 visit
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
            y2 = self._dispatch_routed_pairs(moe, x_2d, tok_all, e_max, w1_use).reshape(bsz, slen, dim)
        else:
            tok_k2 = tok_all[keep2_mask]
            e_k2 = e_min[keep2_mask]
            w_k2 = w_min[keep2_mask]
            tok_cat = torch.cat([tok_all, tok_k2], dim=0)
            exp_cat = torch.cat([e_max, e_k2], dim=0)
            w_cat = torch.cat([w1_use, w_k2], dim=0)
            y2 = self._dispatch_routed_pairs(moe, x_2d, tok_cat, exp_cat, w_cat).reshape(bsz, slen, dim)

        y = y1 + y2 if y1 is not None else y2
        return moe.residual(y, residual, c=y), aux_loss

    def __enter__(self) -> "LoopSchedExecutor":
        self._orig_moe_fwd = []
        if self.mode == "native":
            for lid, layer in enumerate(self.model.layers):
                moe = layer.moe
                orig = moe.forward
                self._orig_moe_fwd.append((moe, orig))

                def make_native(lid_c: int, moe_c: Any, fn_c: Any):
                    def wrapped(x_in, load_balancing_type=None):
                        return self._run_moe_visit(lid_c, moe_c, fn_c, x_in, load_balancing_type)
                    return wrapped

                moe.forward = make_native(lid, moe, orig)
            return self

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
# Fast Batched Evaluation (NLL + Right-Padded MCQ Benchmarks)
# ---------------------------------------------------------------------------

def paired_bootstrap_ci(
    diffs: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 20261126,
) -> tuple[float, float, float]:
    arr = np.asarray(diffs, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


@torch.inference_mode()
def eval_nll_batched(
    model: Any,
    seqs: list[dict[str, Any]],
    executor: LoopSchedExecutor,
    device: torch.device,
    *,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Batched held-out NLL evaluation (all seqs have identical length seq_len)."""
    executor.reset_counters()
    per_seq: list[float] = []
    text_nlls: list[float] = []
    code_nlls: list[float] = []

    with executor:
        for s0 in range(0, len(seqs), batch_size):
            batch = seqs[s0 : s0 + batch_size]
            toks = torch.tensor([b["tokens"] for b in batch], device=device, dtype=torch.long)
            inp = toks[:, :-1]
            tgt = toks[:, 1:]
            executor.reset_pass()
            logits, _, _ = model(inp, multi_segments=False, cache=None)
            # Per-sequence mean NLL
            loss_tok = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                tgt.reshape(-1),
                reduction="none",
            ).reshape(len(batch), -1)
            seq_means = loss_tok.mean(dim=-1).cpu().tolist()
            for item, val in zip(batch, seq_means):
                per_seq.append(float(val))
                if item.get("domain") == "text":
                    text_nlls.append(float(val))
                else:
                    code_nlls.append(float(val))

    return {
        "nll": float(np.mean(per_seq)),
        "text_nll": float(np.mean(text_nlls)) if text_nlls else 0.0,
        "code_nll": float(np.mean(code_nlls)) if code_nlls else 0.0,
        "per_seq": per_seq,
        "active_expert_frac": executor.active_expert_frac,
        "cut_pct": executor.cut_pct,
    }


def prepare_mcq_encoded(
    samples: list[dict[str, Any]],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    """Pre-tokenize MCQ items once so repeated evaluations across conditions have zero tokenizer overhead."""
    encoded: list[dict[str, Any]] = []
    for q_idx, item in enumerate(samples):
        prompt = f"Question: {item['question']}\nAnswer:"
        p_ids = tokenizer.encode(prompt, bos=True, eos=False)
        p_len = len(p_ids)
        choices_enc: list[tuple[list[int], int]] = []
        for c_txt in item["choice_texts"]:
            c_ids = tokenizer.encode(" " + c_txt, bos=False, eos=False)
            if not c_ids:
                choices_enc.append(([], p_len))
            else:
                choices_enc.append((p_ids + c_ids, p_len))
        encoded.append(
            {
                "q_idx": q_idx,
                "gold_idx": int(item["gold_idx"]),
                "choices": choices_enc,
            }
        )
    return encoded


@torch.inference_mode()
def eval_mcq_batched(
    model: Any,
    encoded_items: list[dict[str, Any]],
    executor: LoopSchedExecutor,
    device: torch.device,
    *,
    batch_size: int = 32,
    pad_token_id: int = 0,
) -> tuple[float, list[int]]:
    """Length-sorted right-padded batched MCQ log-likelihood evaluation.

    Causal attention guarantees right-padding tokens never influence prefix/choice logits.
    Verified against B=1 at startup!
    """
    # Flatten all valid (q_idx, c_idx, full_ids, p_len)
    flat: list[tuple[int, int, list[int], int]] = []
    n_q = len(encoded_items)
    scores_by_q: list[list[float]] = [[-1e9] * len(it["choices"]) for it in encoded_items]
    for it in encoded_items:
        qi = it["q_idx"]
        for ci, (full_ids, p_len) in enumerate(it["choices"]):
            if len(full_ids) > p_len:
                flat.append((qi, ci, full_ids, p_len))

    # Sort by sequence length to minimize right-padding waste (<10% overhead)
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
            # Compute exact choice mean log-prob for each item in batch
            for bi, (qi, ci, full_ids, p_len) in enumerate(chunk):
                c_len = len(full_ids) - p_len
                sl_logits = logits[bi, p_len - 1 : p_len - 1 + c_len, :].float()
                lp = F.log_softmax(sl_logits, dim=-1)
                tgt = torch.tensor(full_ids[p_len:], device=device, dtype=torch.long)
                tok_ll = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                scores_by_q[qi][ci] = float(tok_ll.mean().item())

    flags = [
        1 if int(np.argmax(scores_by_q[qi])) == encoded_items[qi]["gold_idx"] else 0
        for qi in range(n_q)
    ]
    return float(np.mean(flags)), flags


# ---------------------------------------------------------------------------
# Calibration: Per-Visit cos(e1, e2), Closed-Form gamma_v*, & Sensitivities
# ---------------------------------------------------------------------------

@torch.inference_mode()
def calibrate_loopsched(
    model: Any,
    topo: list[dict[str, Any]],
    calib_seqs: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    """Single-pass geometric calibration (cos(e1, e2), optimal gamma_v*, ratios) +
    per-visit marginal NLL sensitivity calibration for k=2->1 and k=1->0 on calib_seqs ONLY.
    """
    V = len(topo)
    n_layers = len(model.layers)
    n_experts = int(model.layers[0].moe.experts.n_local_experts)
    topo_map = {(d["layer_id"], d["loop_idx"]): d["vid"] for d in topo}

    sum_w1_sq_e1_sq = np.zeros(V, dtype=np.float64)
    sum_w1_w2_dot = np.zeros(V, dtype=np.float64)
    sum_cos_num = np.zeros(V, dtype=np.float64)
    sum_cos_cnt = np.zeros(V, dtype=np.int64)
    ratios_by_visit: list[list[np.ndarray]] = [[] for _ in range(V)]

    visits = [0] * n_layers
    orig_moe = []
    try:
        for lid, layer in enumerate(model.layers):
            moe = layer.moe
            orig_moe.append((moe, moe.forward))

            def make_calib_moe(lid_c: int, moe_c: Any, orig_fn: Any):
                def wrapped(x_in, load_balancing_type=None):
                    v_num = visits[lid_c]
                    visits[lid_c] = v_num + 1
                    vid = topo_map[(lid_c, v_num)]

                    residual = x_in
                    x = moe_c.norm(x_in)
                    y1 = moe_c.fc2(F.silu(moe_c.fc1(x)) * moe_c.fc3(x)) if moe_c.fc1 is not None else None
                    routing_scores, routing_indices, _tpe, aux_loss = moe_c.router(x, load_balancing_type)

                    bsz, slen, dim = x_in.shape
                    n_tok = bsz * slen
                    x_2d = x.reshape(n_tok, dim)
                    w_f = routing_scores.reshape(n_tok, -1).float()
                    idx_2d = routing_indices.reshape(n_tok, -1)

                    w_min, min_pos = w_f.min(dim=-1)
                    max_pos = 1 - min_pos
                    w_max = w_f.gather(-1, max_pos.unsqueeze(-1)).squeeze(-1)
                    e_max = idx_2d.gather(-1, max_pos.unsqueeze(-1)).squeeze(-1)
                    e_min = idx_2d.gather(-1, min_pos.unsqueeze(-1)).squeeze(-1)
                    ratios_by_visit[vid].append((w_min / w_max.clamp_min(1e-12)).cpu().numpy())

                    # Dispatch top-1 and top-2 separately to measure exact <e1, e2> & ||e1||^2
                    def run_single(e_ids: torch.Tensor) -> torch.Tensor:
                        order = torch.argsort(e_ids, stable=False)
                        gs = torch.bincount(e_ids.index_select(0, order), minlength=n_experts).tolist()
                        yp = moe_c.experts(x_2d.index_select(0, order), gs)
                        out = torch.empty_like(x_2d)
                        out.index_copy_(0, order, yp)
                        return out

                    out1 = run_single(e_max)
                    out2 = run_single(e_min)

                    # Accumulate geometry in fp32
                    o1f = out1.float()
                    o2f = out2.float()
                    n1_sq = o1f.pow(2).sum(dim=-1)
                    n2_sq = o2f.pow(2).sum(dim=-1)
                    dot12 = (o1f * o2f).sum(dim=-1)
                    cos12 = dot12 / (n1_sq.sqrt() * n2_sq.sqrt()).clamp_min(1e-12)

                    sum_w1_sq_e1_sq[vid] += float((w_max.pow(2) * n1_sq).sum().item())
                    sum_w1_w2_dot[vid] += float((w_max * w_min * dot12).sum().item())
                    sum_cos_num[vid] += float(cos12.sum().item())
                    sum_cos_cnt[vid] += n_tok

                    y2 = (out1 * w_max.to(out1.dtype).unsqueeze(-1) + out2 * w_min.to(out2.dtype).unsqueeze(-1)).reshape(bsz, slen, dim)
                    y = y1 + y2 if y1 is not None else y2
                    return moe_c.residual(y, residual, c=y), aux_loss
                return wrapped

            moe.forward = make_calib_moe(lid, moe, moe.forward)

        # Batched pass over calib_seqs
        bsz = 8
        for s0 in range(0, len(calib_seqs), bsz):
            batch = calib_seqs[s0 : s0 + bsz]
            toks = torch.tensor([b["tokens"] for b in batch], device=device, dtype=torch.long)
            for l in range(n_layers):
                visits[l] = 0
            _ = model(toks[:, :-1], multi_segments=False, cache=None)
    finally:
        for moe, fn in orig_moe:
            moe.forward = fn

    visit_gamma = [
        compute_optimal_gamma(sum_w1_sq_e1_sq[v], sum_w1_w2_dot[v]) for v in range(V)
    ]
    visit_cos = [
        float(sum_cos_num[v] / max(sum_cos_cnt[v], 1)) for v in range(V)
    ]
    visit_ratios = [np.concatenate(r) if r else np.zeros(0, dtype=np.float32) for r in ratios_by_visit]

    # Measure per-visit marginal NLL sensitivity on calib_seqs[:8] ONLY
    calib_sub = calib_seqs[:8]
    base_exec = LoopSchedExecutor(model, topo, mode="native")
    base_nll = eval_nll_batched(model, calib_sub, base_exec, device, batch_size=8)["nll"]

    sens_k1_opt = np.zeros(V, dtype=np.float64)
    sens_k0 = np.zeros(V, dtype=np.float64)

    for v in range(V):
        # k=1 with optimal gamma*
        vk = [2] * V
        vk[v] = 1
        ex_opt = LoopSchedExecutor(model, topo, mode="static", visit_k=vk, visit_gamma=visit_gamma, renorm_on_drop2=False)
        sens_k1_opt[v] = max(1e-6, eval_nll_batched(model, calib_sub, ex_opt, device, batch_size=8)["nll"] - base_nll)

        # k=0 (skip routed MoE)
        vk0 = [2] * V
        vk0[v] = 0
        ex_k0 = LoopSchedExecutor(model, topo, mode="static", visit_k=vk0, visit_gamma=visit_gamma, renorm_on_drop2=False)
        sens_k0[v] = max(1e-6, eval_nll_batched(model, calib_sub, ex_k0, device, batch_size=8)["nll"] - base_nll)

    # Layer-Only pooled sensitivity (ablation that strips loop index!)
    layer_sens_k1 = np.zeros(V, dtype=np.float64)
    layer_sens_k0 = np.zeros(V, dtype=np.float64)
    for lid in range(n_layers):
        vids_l = [d["vid"] for d in topo if d["layer_id"] == lid]
        m1 = float(np.mean(sens_k1_opt[vids_l]))
        m0 = float(np.mean(sens_k0[vids_l]))
        for v in vids_l:
            layer_sens_k1[v] = m1
            layer_sens_k0[v] = m0

    # Joint per-stage ablation summary (exact Renorm vs NoRen vs Opt gamma* vs k=0 per stage)
    stages_order: list[str] = []
    for d in topo:
        if d["stage"] not in stages_order:
            stages_order.append(d["stage"])
    stage_summary: list[dict[str, Any]] = []
    for st in stages_order:
        vids_s = [d["vid"] for d in topo if d["stage"] == st]
        vk_s1 = [1 if v in vids_s else 2 for v in range(V)]
        vk_s0 = [0 if v in vids_s else 2 for v in range(V)]
        ex_sr = LoopSchedExecutor(model, topo, mode="static", visit_k=vk_s1, visit_gamma=[1.0] * V, renorm_on_drop2=True)
        ex_sn = LoopSchedExecutor(model, topo, mode="static", visit_k=vk_s1, visit_gamma=[1.0] * V, renorm_on_drop2=False)
        ex_so = LoopSchedExecutor(model, topo, mode="static", visit_k=vk_s1, visit_gamma=visit_gamma, renorm_on_drop2=False)
        ex_s0 = LoopSchedExecutor(model, topo, mode="static", visit_k=vk_s0, visit_gamma=visit_gamma, renorm_on_drop2=False)
        d_sr = eval_nll_batched(model, calib_sub, ex_sr, device, batch_size=8)["nll"] - base_nll
        d_sn = eval_nll_batched(model, calib_sub, ex_sn, device, batch_size=8)["nll"] - base_nll
        d_so = eval_nll_batched(model, calib_sub, ex_so, device, batch_size=8)["nll"] - base_nll
        d_s0 = eval_nll_batched(model, calib_sub, ex_s0, device, batch_size=8)["nll"] - base_nll
        stage_summary.append(
            {
                "stage": st,
                "n_visits": len(vids_s),
                "mean_cos_e1_e2": float(np.mean([visit_cos[v] for v in vids_s])),
                "mean_gamma_opt": float(np.mean([visit_gamma[v] for v in vids_s])),
                "sum_sens_k1_renorm": float(d_sr),
                "sum_sens_k1_noren": float(d_sn),
                "sum_sens_k1_opt": float(d_so),
                "sum_sens_k0": float(d_s0),
            }
        )

    return {
        "base_calib_nll": base_nll,
        "visit_gamma": visit_gamma,
        "visit_cos": visit_cos,
        "visit_ratios": visit_ratios,
        "sens_k1_opt": sens_k1_opt,
        "sens_k1_noren": sens_k1_opt,
        "sens_k0": sens_k0,
        "layer_sens_k1": layer_sens_k1,
        "layer_sens_k0": layer_sens_k0,
        "stage_summary": stage_summary,
    }


# ---------------------------------------------------------------------------
# Real GPU Wall-Clock Prefill & Decode Latency Benchmark
# ---------------------------------------------------------------------------

@torch.inference_mode()
def benchmark_real_gpu_speed(
    model: Any,
    topo: list[dict[str, Any]],
    schedules: dict[str, tuple[list[int], list[float]]],
    device: torch.device,
    *,
    vocab_size: int = 32000,
    n_warmup: int = 10,
    n_iters: int = 40,
) -> list[dict[str, Any]]:
    """Measure real GPU wall-clock latency (torch.cuda.Event) for Prefill & Decode shapes."""
    shapes = [
        ("decode_B1_L1", 1, 1),
        ("decode_B8_L1", 8, 1),
        ("decode_B32_L1", 32, 1),
        ("prefill_B8_L256", 8, 256),
    ]
    rows: list[dict[str, Any]] = []
    base_ms_by_shape: dict[str, float] = {}

    for shape_name, bsz, slen in shapes:
        inp = torch.randint(10, min(vocab_size, 10000), (bsz, slen), device=device, dtype=torch.long)
        for sched_name, (vk, vg) in schedules.items():
            mode = "native" if sched_name == "bf16_base" else "static"
            ex = LoopSchedExecutor(
                model, topo, mode=mode, visit_k=vk, visit_gamma=vg, renorm_on_drop2=False, track_stats=False
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
    return rows


# ---------------------------------------------------------------------------
# Full Per-Model R2-A Execution
# ---------------------------------------------------------------------------

def run_model_r2a(
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
    print(f"\n[r2a:{model_key}] Loading {artifact_dir} ...", flush=True)
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
        n_calib_a=int(cfg.get("n_calib", 16)),
        n_calib_b=4,
        n_adapt_train=4,
        n_eval_heldout=int(cfg.get("n_eval", 64)),
        seed=seed,
    )
    calib_seqs = splits["calib_a"]
    eval_seqs = splits["eval_heldout"]
    mcq_enc = {k: prepare_mcq_encoded(v, tokenizer) for k, v in mcq_raw.items()}
    pad_id = int(getattr(tokenizer, "pad_id", 0) or 0)

    # 1) Verify exactness of real dispatcher AND right-padded batching
    with run_obj.phase(f"{model_key}_verify", total=1, unit="check") as pv:
        with torch.inference_mode():
            # 1a. verify_k2 vs native at B=2, L=64
            t_in = torch.tensor([eval_seqs[0]["tokens"][:64], eval_seqs[1]["tokens"][:64]], device=device, dtype=torch.long)
            ref_logits = model(t_in, multi_segments=False, cache=None)[0].float()
            ex_v = LoopSchedExecutor(model, topo, mode="verify_k2")
            with ex_v:
                ex_v.reset_pass()
                our_logits = model(t_in, multi_segments=False, cache=None)[0].float()
            k2_diff = float((ref_logits - our_logits).abs().max().item())

            # 1b. verify B=1 vs B=4 right-padded causal invariance on first 4 ARC-C choices
            sample_seqs = [mcq_enc["arc_c"][i]["choices"][0][0][:-1] for i in range(4)]
            max_l = max(len(s) for s in sample_seqs)
            pad_in = torch.full((4, max_l), pad_id, device=device, dtype=torch.long)
            for bi, s in enumerate(sample_seqs):
                pad_in[bi, : len(s)] = torch.tensor(s, device=device, dtype=torch.long)
            batched_logits = model(pad_in, multi_segments=False, cache=None)[0].float()
            pad_max_diff = 0.0
            for bi, s in enumerate(sample_seqs):
                s_in = torch.tensor([s], device=device, dtype=torch.long)
                one_logits = model(s_in, multi_segments=False, cache=None)[0].float()
                d = float((one_logits[0] - batched_logits[bi, : len(s)]).abs().max().item())
                pad_max_diff = max(pad_max_diff, d)

            # 1c. verify B=1, L=1 single-token decode parity
            d1_in = torch.tensor([[eval_seqs[0]["tokens"][0]]], device=device, dtype=torch.long)
            d1_ref = model(d1_in, multi_segments=False, cache=None)[0].float()
            with ex_v:
                ex_v.reset_pass()
                d1_our = model(d1_in, multi_segments=False, cache=None)[0].float()
            d1_diff = float((d1_ref - d1_our).abs().max().item())

        print(
            f"[r2a:{model_key}] Exactness OK: k2_diff={k2_diff:.2e}, pad_batch_diff={pad_max_diff:.2e}, decode1_diff={d1_diff:.2e}",
            flush=True,
        )
        if k2_diff > 1e-2 or pad_max_diff > 5e-2 or d1_diff > 1e-2:
            raise RuntimeError(f"Exactness verification failed: k2={k2_diff}, pad={pad_max_diff}, d1={d1_diff}")
        pv.step(1)

    # 2) Calibration pass (geometry + optimal gamma_v* + per-visit sensitivities)
    with run_obj.phase(f"{model_key}_calib", total=1, unit="pass") as pc:
        calib = calibrate_loopsched(model, topo, calib_seqs, device)
        print(f"[r2a:{model_key}] Per-stage geometry & sensitivity summary:", flush=True)
        for st in calib["stage_summary"]:
            print(
                f"  stage={st['stage']:8s} (V={st['n_visits']:2d}) | cos(e1,e2)={st['mean_cos_e1_e2']:+.3f} "
                f"| gamma*={st['mean_gamma_opt']:.3f} | k1_ren={st['sum_sens_k1_renorm']:+.4f} "
                f"| k1_noren={st['sum_sens_k1_noren']:+.4f} | k1_opt={st['sum_sens_k1_opt']:+.4f} | k0={st['sum_sens_k0']:+.4f}",
                flush=True,
            )
            pc.row(
                f"{model_key}_geom",
                stage=st["stage"],
                cos_e1_e2=st["mean_cos_e1_e2"],
                gamma_opt=st["mean_gamma_opt"],
                k1_renorm=st["sum_sens_k1_renorm"],
                k1_opt=st["sum_sens_k1_opt"],
                k0_sens=st["sum_sens_k0"],
            )
        pc.step(1)

    vg_opt = calib["visit_gamma"]
    vg_one = [1.0] * V
    # Visit sensitivity weights for dynamic dropping: sqrt(sens_k1_opt) normalized to median=1
    w_loop = np.sqrt(np.maximum(calib["sens_k1_opt"], 1e-6))
    w_loop = w_loop / max(float(np.median(w_loop)), 1e-6)
    w_layer = np.sqrt(np.maximum(calib["layer_sens_k1"], 1e-6))
    w_layer = w_layer / max(float(np.median(w_layer)), 1e-6)
    w_unif = np.ones(V, dtype=np.float64)

    # Helper: build tri-level DP schedules for target active-expert cut fractions
    def dp_sched(cut_frac: float, use_opt: bool = True, allow_k0: bool = True, layer_only: bool = False) -> list[int]:
        tgt_k = int(round((1.0 - cut_frac) * 2 * V))
        if layer_only:
            c1 = calib["layer_sens_k1"]
            c0 = calib["layer_sens_k0"]
        else:
            c1 = calib["sens_k1_opt"] if use_opt else calib["sens_k1_noren"]
            c0 = calib["sens_k0"]
        return solve_trilevel_dp_schedule(c1, c0, target_total_k=tgt_k, allow_k0=allow_k0)

    # Drop-last-loops baseline at -50% cut: set k=0 on the last half of recurrent loops!
    rec_vids = [d["vid"] for d in topo if d["is_rec"]]
    n_drop_k0_50 = V // 2  # dropping V/2 visits to k=0 cuts 50% of active experts
    vk_drop_last50 = [2] * V
    # Drop from the end of recurrent visits (leaving prelude & coda intact or dropping last visits)
    for v in reversed(rec_vids):
        if sum(1 for k in vk_drop_last50 if k == 0) >= n_drop_k0_50:
            break
        vk_drop_last50[v] = 0
    # If not enough rec visits, drop last remaining visits
    for v in range(V - 1, -1, -1):
        if sum(1 for k in vk_drop_last50 if k == 0) >= n_drop_k0_50:
            break
        vk_drop_last50[v] = 0

    # Pre-solve DP schedules
    sched_25_noren = dp_sched(0.25, use_opt=False, allow_k0=True)
    sched_25_opt = dp_sched(0.25, use_opt=True, allow_k0=True)
    sched_38_opt = dp_sched(0.375, use_opt=True, allow_k0=True)
    sched_38_layer = dp_sched(0.375, use_opt=True, allow_k0=True, layer_only=True)
    sched_50_opt = dp_sched(0.50, use_opt=True, allow_k0=True)
    sched_62_opt = dp_sched(0.625, use_opt=True, allow_k0=True)

    # For dynamic tri-level policies at 50% & 62.5%: fix only the k=0 visits from DP (or none at 25/37.5%),
    # then dynamically allocate remaining top-2->top-1 cuts via visit-sensitivity-weighted ratio!
    k0_only_38 = [0 if k == 0 else 2 for k in sched_38_opt]
    k0_only_50 = [0 if k == 0 else 2 for k in sched_50_opt]
    k0_only_62 = [0 if k == 0 else 2 for k in sched_62_opt]
    # Uniform k012 baseline at 62.5%: all visits k=1 (50% cut), plus random/even 25% of visits at k=0 -> 62.5%
    vk_unif_62 = [1] * V
    step_k0 = max(1, V // max(1, int(round(0.25 * V))))
    n_k0_need = max(1, int(round((0.625 - 0.50) * 2 * V)))
    for i in range(n_k0_need):
        vk_unif_62[(i * step_k0) % V] = 0

    # Construct all conditions cleanly
    conds: list[dict[str, Any]] = [
        {"name": "bf16_base", "mode": "native", "vk": [2] * V, "vg": vg_one, "renorm": False, "w": None, "tau": -1.0},
        # --- Budget: -25% active experts ---
        {
            "name": "cut25_unif_renorm", "mode": "dynamic", "vk": [2] * V, "vg": vg_one, "renorm": True,
            "w": w_unif, "tau": calibrate_dynamic_threshold(calib["visit_ratios"], w_unif, target_drop_frac=0.25),
        },
        {
            "name": "cut25_unif_noren", "mode": "dynamic", "vk": [2] * V, "vg": vg_one, "renorm": False,
            "w": w_unif, "tau": calibrate_dynamic_threshold(calib["visit_ratios"], w_unif, target_drop_frac=0.25),
        },
        {
            "name": "cut25_layer_only_opt", "mode": "dynamic", "vk": [2] * V, "vg": vg_opt, "renorm": False,
            "w": w_layer, "tau": calibrate_dynamic_threshold(calib["visit_ratios"], w_layer, target_drop_frac=0.25),
        },
        {
            "name": "cut25_loopsched_static_noren", "mode": "static", "vk": sched_25_noren, "vg": vg_one, "renorm": False,
            "w": None, "tau": -1.0,
        },
        {
            "name": "cut25_loopsched_static_opt", "mode": "static", "vk": sched_25_opt, "vg": vg_opt, "renorm": False,
            "w": None, "tau": -1.0,
        },
        {
            "name": "cut25_loopsched_dyn_opt", "mode": "dynamic", "vk": [2] * V, "vg": vg_opt, "renorm": False,
            "w": w_loop, "tau": calibrate_dynamic_threshold(calib["visit_ratios"], w_loop, target_drop_frac=0.25),
        },
        # --- Budget: -37.5% active experts ---
        {
            "name": "cut38_unif_renorm", "mode": "dynamic", "vk": [2] * V, "vg": vg_one, "renorm": True,
            "w": w_unif, "tau": calibrate_dynamic_threshold(calib["visit_ratios"], w_unif, target_drop_frac=0.375),
        },
        {
            "name": "cut38_unif_noren", "mode": "dynamic", "vk": [2] * V, "vg": vg_one, "renorm": False,
            "w": w_unif, "tau": calibrate_dynamic_threshold(calib["visit_ratios"], w_unif, target_drop_frac=0.375),
        },
        {
            "name": "cut38_layer_only_opt", "mode": "dynamic", "vk": [2] * V, "vg": vg_opt, "renorm": False,
            "w": w_layer, "tau": calibrate_dynamic_threshold(calib["visit_ratios"], w_layer, target_drop_frac=0.375),
        },
        {
            "name": "cut38_loopsched_static_opt", "mode": "static", "vk": sched_38_opt, "vg": vg_opt, "renorm": False,
            "w": None, "tau": -1.0,
        },
        {
            "name": "cut38_loopsched_dyn_opt", "mode": "dynamic", "vk": k0_only_38, "vg": vg_opt, "renorm": False,
            "w": w_loop, "tau": calibrate_dynamic_threshold(calib["visit_ratios"], w_loop, target_drop_frac=0.375, fixed_k_visits=k0_only_38),
        },
        # --- Budget: -50% active experts ---
        {
            "name": "cut50_all_top1_renorm", "mode": "static", "vk": [1] * V, "vg": vg_one, "renorm": True,
            "w": None, "tau": -1.0,
        },
        {
            "name": "cut50_all_top1_noren", "mode": "static", "vk": [1] * V, "vg": vg_one, "renorm": False,
            "w": None, "tau": -1.0,
        },
        {
            "name": "cut50_all_top1_opt", "mode": "static", "vk": [1] * V, "vg": vg_opt, "renorm": False,
            "w": None, "tau": -1.0,
        },
        {
            "name": "cut50_drop_last_loops", "mode": "static", "vk": vk_drop_last50, "vg": vg_opt, "renorm": False,
            "w": None, "tau": -1.0,
        },
        {
            "name": "cut50_loopsched_static_opt", "mode": "static", "vk": sched_50_opt, "vg": vg_opt, "renorm": False,
            "w": None, "tau": -1.0,
        },
        {
            "name": "cut50_loopsched_dyn_opt", "mode": "dynamic", "vk": k0_only_50, "vg": vg_opt, "renorm": False,
            "w": w_loop, "tau": calibrate_dynamic_threshold(calib["visit_ratios"], w_loop, target_drop_frac=0.50, fixed_k_visits=k0_only_50),
        },
        # --- Budget: -62.5% active experts (beyond top-1 ceiling!) ---
        {
            "name": "cut62_unif_k01", "mode": "static", "vk": vk_unif_62, "vg": vg_one, "renorm": False,
            "w": None, "tau": -1.0,
        },
        {
            "name": "cut62_loopsched_static_opt", "mode": "static", "vk": sched_62_opt, "vg": vg_opt, "renorm": False,
            "w": None, "tau": -1.0,
        },
        {
            "name": "cut62_loopsched_dyn_opt", "mode": "dynamic", "vk": k0_only_62, "vg": vg_opt, "renorm": False,
            "w": w_loop, "tau": calibrate_dynamic_threshold(calib["visit_ratios"], w_loop, target_drop_frac=0.625, fixed_k_visits=k0_only_62),
        },
    ]

    results: dict[str, Any] = {}
    base_per_seq: np.ndarray | None = None
    base_nll = 0.0
    base_mcq_flags: dict[str, np.ndarray] = {}

    with run_obj.phase(f"{model_key}_eval", total=len(conds), unit="conds") as pe:
        for idx_c, c in enumerate(conds, 1):
            t_c0 = time.time()
            ex = LoopSchedExecutor(
                model,
                topo,
                mode=c["mode"],
                visit_k=c["vk"],
                visit_gamma=c["vg"],
                renorm_on_drop2=c["renorm"],
                visit_weights=c["w"],
                dyn_threshold=c["tau"],
            )
            nll_res = eval_nll_batched(model, eval_seqs, ex, device, batch_size=8)
            cut_pct = nll_res["cut_pct"]

            mcq_accs: dict[str, float] = {}
            mcq_flags: dict[str, list[int]] = {}
            for ds_name, enc_items in mcq_enc.items():
                acc, fl = eval_mcq_batched(model, enc_items, ex, device, batch_size=32, pad_token_id=pad_id)
                mcq_accs[ds_name] = acc
                mcq_flags[ds_name] = fl
            avg_mcq = float(np.mean(list(mcq_accs.values())))

            if c["name"] == "bf16_base":
                base_per_seq = np.asarray(nll_res["per_seq"], dtype=np.float64)
                base_nll = nll_res["nll"]
                for ds_name, fl in mcq_flags.items():
                    base_mcq_flags[ds_name] = np.asarray(fl, dtype=np.float64)

            assert base_per_seq is not None
            diffs = np.asarray(nll_res["per_seq"], dtype=np.float64) - base_per_seq
            dnll, dnll_lo, dnll_hi = paired_bootstrap_ci(diffs)
            elapsed_c = time.time() - t_c0

            results[c["name"]] = {
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
                "visit_k": c["vk"],
                "elapsed_s": elapsed_c,
            }
            print(
                f"[r2a:{model_key}] {c['name']:28s} cut={cut_pct:5.1f}% | NLL={nll_res['nll']:.4f} (Δ={dnll:+.4f}) "
                f"| ARC-C={mcq_accs['arc_c']*100:5.1f}% ARC-E={mcq_accs['arc_e']*100:5.1f}% OBQA={mcq_accs['obqa']*100:5.1f}% "
                f"Avg={avg_mcq*100:5.1f}% | t={elapsed_c:.1f}s",
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

    # 3) Real GPU wall-clock speed benchmark
    with run_obj.phase(f"{model_key}_speed", total=1, unit="bench") as ps:
        bench_scheds = {
            "bf16_base": ([2] * V, vg_one),
            "cut25_loopsched_static": (sched_25_opt, vg_opt),
            "cut38_loopsched_static": (sched_38_opt, vg_opt),
            "cut50_loopsched_static": (sched_50_opt, vg_opt),
            "cut62_loopsched_static": (sched_62_opt, vg_opt),
        }
        speed_rows = benchmark_real_gpu_speed(model, topo, bench_scheds, device)
        print(f"\n[r2a:{model_key}] Real GPU Wall-Clock Latency Benchmark:", flush=True)
        for r in speed_rows:
            print(
                f"  {r['shape']:16s} | {r['schedule']:24s} (cut={r['cut_pct']:4.1f}%) | "
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

    # 4) Key paired contrasts (positive = first method better: lower NLL, higher Avg MCQ)
    contrast_pairs = [
        ("cut25_loopsched_dyn_opt", "cut25_unif_renorm"),
        ("cut25_loopsched_dyn_opt", "cut25_unif_noren"),
        ("cut25_loopsched_dyn_opt", "cut25_layer_only_opt"),
        ("cut25_loopsched_static_opt", "cut25_unif_renorm"),
        ("cut25_loopsched_static_opt", "cut25_loopsched_static_noren"),
        ("cut38_loopsched_dyn_opt", "cut38_unif_renorm"),
        ("cut38_loopsched_dyn_opt", "cut38_unif_noren"),
        ("cut38_loopsched_dyn_opt", "cut38_layer_only_opt"),
        ("cut38_loopsched_static_opt", "cut38_unif_renorm"),
        ("cut50_all_top1_noren", "cut50_all_top1_renorm"),
        ("cut50_all_top1_opt", "cut50_all_top1_noren"),
        ("cut50_loopsched_static_opt", "cut50_all_top1_renorm"),
        ("cut50_loopsched_static_opt", "cut50_all_top1_noren"),
        ("cut50_loopsched_static_opt", "cut50_drop_last_loops"),
        ("cut50_loopsched_dyn_opt", "cut50_all_top1_noren"),
        ("cut62_loopsched_static_opt", "cut62_unif_k01"),
        ("cut62_loopsched_dyn_opt", "cut62_unif_k01"),
    ]
    contrasts: dict[str, Any] = {}
    for a, b in contrast_pairs:
        if a in results and b in results:
            da = np.asarray(results[b]["per_seq"], dtype=np.float64) - np.asarray(results[a]["per_seq"], dtype=np.float64)
            m, lo, hi = paired_bootstrap_ci(da)
            # Paired MCQ accuracy gain pooled across all 2,272 questions
            fl_a = np.concatenate([results[a]["mcq_flags"][k] for k in ("arc_c", "arc_e", "obqa")])
            fl_b = np.concatenate([results[b]["mcq_flags"][k] for k in ("arc_c", "arc_e", "obqa")])
            mcq_m, mcq_lo, mcq_hi = paired_bootstrap_ci(fl_a - fl_b)
            contrasts[f"{a}_vs_{b}"] = {
                "nll_gain": m,
                "nll_ci": [lo, hi],
                "nll_wins": int((da > 0).sum()),
                "n_seq": int(da.size),
                "mcq_gain_pp": mcq_m * 100.0,
                "mcq_ci_pp": [mcq_lo * 100.0, mcq_hi * 100.0],
            }

    # Strip heavy per-item arrays from serialized JSON
    compact_results = {
        k: {kk: vv for kk, vv in v.items() if kk not in ("per_seq", "mcq_flags")}
        for k, v in results.items()
    }

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model_key": model_key,
        "n_visits": V,
        "stage_summary": calib["stage_summary"],
        "mean_cos_e1_e2": float(np.mean(calib["visit_cos"])),
        "mean_gamma_opt": float(np.mean(calib["visit_gamma"])),
        "results": compact_results,
        "speed_rows": speed_rows,
        "contrasts": contrasts,
    }


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Round 2-A: LoopSched-MoE — Stage-Heterogeneous & Dynamic Expert Scheduling",
        "",
        f"- GPU: `{summary['gpu_index']}` ({summary['gpu_name']}), elapsed `{summary['elapsed_sec']:.1f}s`",
        f"- Benchmarks: Held-out NLL (`n={summary['n_eval']}`), ARC-Challenge (`n={summary['n_arc_c']}`), ARC-Easy (`n={summary['n_arc_e']}`), OpenBookQA (`n={summary['n_obqa']}`)",
        "",
    ]
    for mk, mres in summary["models"].items():
        lines.append(f"## Model: `{mk}` (Visits={mres['n_visits']}, mean cos(e1,e2)={mres['mean_cos_e1_e2']:+.3f}, mean γ*={mres['mean_gamma_opt']:.3f})")
        lines.append("")
        lines.append("### 1. Per-Stage Geometry & Sensitivity")
        lines.append("| Stage | Visits | cos(e1,e2) | γ* | k=1 Renorm | k=1 NoRen | k=1 Opt γ* | k=0 Skip |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for st in mres["stage_summary"]:
            lines.append(
                f"| `{st['stage']}` | {st['n_visits']} | {st['mean_cos_e1_e2']:+.3f} | {st['mean_gamma_opt']:.3f} | "
                f"{st['sum_sens_k1_renorm']:+.4f} | {st['sum_sens_k1_noren']:+.4f} | {st['sum_sens_k1_opt']:+.4f} | {st['sum_sens_k0']:+.4f} |"
            )
        lines.append("")
        lines.append("### 2. Main Quality-Compute Pareto Table")
        lines.append("| Policy | Cut% | NLL | ΔNLL (95% CI) | ARC-C | ARC-E | OBQA | Avg MCQ | (k0/k1/k2) |")
        lines.append("|---|---:|---:|---|---:|---:|---:|---:|---|")
        for pname, r in mres["results"].items():
            lo, hi = r["delta_nll_ci"]
            lines.append(
                f"| `{pname}` | {r['cut_pct']:.1f}% | {r['nll']:.4f} | {r['delta_nll']:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
                f"{r['arc_c']*100:.1f}% | {r['arc_e']*100:.1f}% | {r['obqa']*100:.1f}% | **{r['avg_mcq']*100:.1f}%** | "
                f"{r['k0_count']}/{r['k1_count']}/{r['k2_count']} |"
            )
        lines.append("")
        lines.append("### 3. Real GPU Wall-Clock Speedup (RTX PRO 5000)")
        lines.append("| Shape | Schedule | Cut% | Median (ms) | IQR (ms) | Speedup | Tok/s |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for sr in mres["speed_rows"]:
            lines.append(
                f"| `{sr['shape']}` | `{sr['schedule']}` | {sr['cut_pct']:.1f}% | {sr['median_ms']:.2f} | "
                f"{sr['iqr_ms']:.2f} | **{sr['speedup']:.3f}x** | {sr['tok_per_sec']:,.0f} |"
            )
        lines.append("")
        lines.append("### 4. Paired Bootstrap Contrasts (positive = first beats second)")
        for cname, c in mres["contrasts"].items():
            nlo, nhi = c["nll_ci"]
            mlo, mhi = c["mcq_ci_pp"]
            lines.append(
                f"- `{cname}`: NLL `{c['nll_gain']:+.4f}` CI `[{nlo:+.4f}, {nhi:+.4f}]` ({c['nll_wins']}/{c['n_seq']} wins) | "
                f"Pooled MCQ `{c['mcq_gain_pp']:+.2f} pp` CI `[{mlo:+.2f}, {mhi:+.2f}]`"
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

    gpu_idx, gpu_lock, gpu_status = acquire_idle_gpu(cfg.get("preferred_gpu"))
    try:
        print(
            f"[r2a] Acquired idle GPU {gpu_idx}: {gpu_status.name} "
            f"(mem={gpu_status.mem_used_mib}MiB, util={gpu_status.util_pct}%)",
            flush=True,
        )
        boot = bootstrap_xllm(args.xllm_loop_dir, attention_backend="sdpa")
        print(f"[r2a] xllm bootstrap: {json.dumps(boot)}", flush=True)
        device = torch.device(f"cuda:{gpu_idx}")
        torch.cuda.set_device(device)

        seed = int(cfg.get("seed", 20261126))
        mcq_raw = {
            "arc_c": load_arc_challenge_samples(args.dataset_dir, n_samples=int(cfg.get("n_arc_c", 0)), seed=seed),
            "arc_e": load_arc_easy_samples(args.dataset_dir, n_samples=int(cfg.get("n_arc_e", 600)), seed=seed),
            "obqa": load_openbookqa_samples(args.dataset_dir, n_samples=int(cfg.get("n_obqa", 500)), seed=seed),
        }
        print(
            f"[r2a] Loaded MCQ suites: ARC-C={len(mcq_raw['arc_c'])}, ARC-E={len(mcq_raw['arc_e'])}, OBQA={len(mcq_raw['obqa'])}",
            flush=True,
        )

        plan = [
            {"phase": "setup", "of": 1, "unit": "env"},
            {"phase": "download", "of": 1, "unit": "assets"},
        ]
        for mk in cfg["models"]:
            plan.append({"phase": f"{mk}_verify", "of": 1, "unit": "check"})
            plan.append({"phase": f"{mk}_calib", "of": 1, "unit": "pass"})
            plan.append({"phase": f"{mk}_eval", "of": 21, "unit": "conds"})
            plan.append({"phase": f"{mk}_speed", "of": 1, "unit": "bench"})
        plan.append({"phase": "report", "of": 1, "unit": "file"})
        orx.plan(plan)
        run_obj = orx.progress(total=1, unit="run")

        models_out: dict[str, Any] = {}
        for mk, mspec in cfg["models"].items():
            models_out[mk] = run_model_r2a(
                mk,
                mspec,
                model_dir=Path(args.model_dir),
                dataset_dir=Path(args.dataset_dir),
                mcq_raw=mcq_raw,
                cfg=cfg,
                device=device,
                run_obj=run_obj,
            )

        summary = {
            "gpu_index": gpu_idx,
            "gpu_name": gpu_status.name,
            "elapsed_sec": time.time() - t0,
            "n_eval": int(cfg.get("n_eval", 64)),
            "n_arc_c": len(mcq_raw["arc_c"]),
            "n_arc_e": len(mcq_raw["arc_e"]),
            "n_obqa": len(mcq_raw["obqa"]),
            "models": models_out,
        }

        with run_obj.phase("report", total=1, unit="file") as pr:
            (out_dir / "summary_r2a_loopsched.json").write_text(json.dumps(summary, indent=2))
            rep = render_report(summary)
            (out_dir / "report_r2a_loopsched.md").write_text(rep)
            print("========== REPORT (report_r2a_loopsched.md) ==========", flush=True)
            print(rep, flush=True)
            print("========== END REPORT ==========", flush=True)

            # Headline summary metrics
            m_out: dict[str, float] = {}
            for mk, mres in models_out.items():
                res = mres["results"]
                m_out[f"{mk}_cut50_dp_vs_top1_nll"] = mres["contrasts"]["cut50_loopsched_static_opt_vs_cut50_all_top1_renorm"]["nll_gain"]
                m_out[f"{mk}_cut50_dp_vs_top1_mcq"] = mres["contrasts"]["cut50_loopsched_static_opt_vs_cut50_all_top1_renorm"]["mcq_gain_pp"]
                m_out[f"{mk}_cut38_dyn_vs_unif_nll"] = mres["contrasts"]["cut38_loopsched_dyn_opt_vs_cut38_unif_renorm"]["nll_gain"]
                m_out[f"{mk}_cut38_dyn_vs_layer_nll"] = mres["contrasts"]["cut38_loopsched_dyn_opt_vs_cut38_layer_only_opt"]["nll_gain"]
                m_out[f"{mk}_cut50_dp_avg_mcq"] = res["cut50_loopsched_static_opt"]["avg_mcq"]
                # Decode B=1 and Prefill B=8 speedup at -50% cut
                for sr in mres["speed_rows"]:
                    if sr["schedule"] == "cut50_loopsched_static" and sr["shape"] == "decode_B1_L1":
                        m_out[f"{mk}_cut50_decode_b1_speedup"] = sr["speedup"]
                    if sr["schedule"] == "cut50_loopsched_static" and sr["shape"] == "prefill_B8_L256":
                        m_out[f"{mk}_cut50_prefill_speedup"] = sr["speedup"]
            pr.step(1, **m_out)
    finally:
        gpu_lock.release()


if __name__ == "__main__":
    main()
