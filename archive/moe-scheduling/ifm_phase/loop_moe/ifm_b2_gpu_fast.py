"""B2-v2: Vectorized (<1 ms/tok CPU overhead) GPU Offload & CUDA Copy-Stream Async Prefetch Benchmark.

Eliminates the 16-17 ms/tok pure-Python dict/math.log/heapify overhead measured in B2-v1 (run 30388e48)
by compiling all A2 conditional transition tables into L2-resident NumPy float32 arrays and replacing
28-pool Python heap scans with a contiguous flat-array slot manager (`np.argmin` over C slots).

Also adds `full_resident_fast` (all 700 experts on GPU using the exact same active-2-expert `F.linear`
executor as all offloaded policies) for a 100% apples-to-apples speed ceiling alongside:
  `static_hot`, `general_nopf`, `loop_nopf`, `general_pf`, `loop_pf`.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import orx_run as orx
from .ifm_a1_collect import download_ifm_artifact, select_idle_gpu
from .ifm_a2_predict import A2Models, _build_schedule_info, read_ifm_trace
from .ifm_b1_kv_verify import LoopedIncrementalEngine
from .ifm_b2_offline_replay import load_trace_splits
from .ifm_prompts import format_ifm_prompt
from .ifm_runtime import bootstrap_xllm, load_ifm_model
from .trace import read_json, read_jsonl, write_csv, write_json


FAST_GPU_POLICIES = (
    "static_hot",
    "general_nopf",
    "loop_nopf",
    "general_pf",
    "loop_pf",
)


def _logit_np(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(pc / (1.0 - pc)).astype(np.float32)


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    xc = np.clip(x, -20.0, 20.0)
    return (1.0 / (1.0 + np.exp(-xc))).astype(np.float32)


class CompiledA2Tables:
    """Dense L2-resident float32 tables compiled from fitted A2Models for <1 ms/tok online scoring."""

    def __init__(
        self,
        meta: dict[str, Any],
        sched: dict[str, Any],
        a2: A2Models,
        *,
        gamma: float = 0.75,
        freq_weight: float = 0.08,
    ):
        self.order = sched["order"]
        self.n_visits = len(self.order)
        self.pools = sorted(meta["pools"].keys())
        self.pool_to_idx = {p: i for i, p in enumerate(self.pools)}
        self.n_pools = len(self.pools)
        self.n_experts = int(a2.num_experts)
        self.alpha = float(a2.alpha)
        self.gamma = float(gamma)
        self.freq_weight = float(freq_weight)

        pool_positions = sched["pool_positions"]
        visits_per_tok = sched["pool_visits_per_token"]

        self.v_pool = np.array([self.pool_to_idx[p] for p, _ in self.order], dtype=np.int32)
        self.pool_h_span = np.array([int(visits_per_tok[p]) for p in self.pools], dtype=np.int32)
        self.pool_density = np.array(
            [float(visits_per_tok[p]) / float(self.n_visits) for p in self.pools], dtype=np.float32
        )

        # Future cycle offsets and next-visit mapping per visit v
        self.v_next_v = np.zeros(self.n_visits, dtype=np.int32)
        self.v_w1 = np.zeros(self.n_visits, dtype=np.float32)
        self.v_is_boundary = np.zeros(self.n_visits, dtype=np.bool_)
        self.v_has_prev_loop = np.zeros(self.n_visits, dtype=np.bool_)

        V, P, E = self.n_visits, self.n_pools, self.n_experts
        self.stage_prior_p = np.zeros((V, E), dtype=np.float32)
        self.stage_prior_lo = np.zeros((V, E), dtype=np.float32)
        self.pool_prior_p = np.zeros((P, E), dtype=np.float32)
        self.pool_prior_lo = np.zeros((P, E), dtype=np.float32)

        for p_name, pi in self.pool_to_idx.items():
            pp = np.array(a2._pool_prior(p_name), dtype=np.float32)
            self.pool_prior_p[pi] = pp
            self.pool_prior_lo[pi] = _logit_np(pp)

        for v, (p_name, lp) in enumerate(self.order):
            sp = np.array(a2._stage_prior(p_name, lp), dtype=np.float32)
            self.stage_prior_p[v] = sp
            self.stage_prior_lo[v] = _logit_np(sp)
            if lp is not None and lp > 0:
                self.v_has_prev_loop[v] = True

        # Precompute hazard future cycle base (sum over m >= 2 in 1 full token cycle)
        self.haz_future_loop = np.zeros((V, E), dtype=np.float32)
        self.haz_future_gen = np.zeros((V, E), dtype=np.float32)
        for v, (p_name, _lp) in enumerate(self.order):
            pi = self.pool_to_idx[p_name]
            pos_list = pool_positions[p_name]
            upcoming = [(p - v, p) for p in pos_list if p > v]
            upcoming.extend((p + V - v, p) for p in pos_list if p <= v)
            d1, nxt_v = upcoming[0]
            self.v_next_v[v] = nxt_v
            self.v_w1[v] = np.float32(1.0 / ((1.0 + d1) ** self.gamma))
            self.v_is_boundary[v] = (v + d1) >= V
            for dm, vm in upcoming[1:]:
                wm = np.float32(1.0 / ((1.0 + dm) ** self.gamma))
                self.haz_future_loop[v] += wm * self.stage_prior_p[vm]
                self.haz_future_gen[v] += wm * self.pool_prior_p[pi]

        # Precompute single-source conditional probability matrices (all float32, L2-cache resident!)
        self.gen_lag1_p = np.zeros((P, E, E), dtype=np.float32)
        self.gen_lagH_p = np.zeros((P, E, E), dtype=np.float32)
        for p_name, pi in self.pool_to_idx.items():
            prior_l = self.pool_prior_p[pi].tolist()
            for se in range(E):
                self.gen_lag1_p[pi, se] = a2._trans_prob(a2.gen_lag1_trans, a2.gen_lag1_totals, [(p_name, se)], prior_l)
                self.gen_lagH_p[pi, se] = a2._trans_prob(a2.gen_lagH_trans, a2.gen_lagH_totals, [(p_name, se)], prior_l)

        self.prev_layer_p = np.zeros((V, E, E), dtype=np.float32)
        self.gen_prev_pool_p = np.zeros((V, E, E), dtype=np.float32)
        self.prev_loop_p = np.zeros((V, E, E), dtype=np.float32)
        self.prev_loop_hier_p = np.zeros((V, E, E), dtype=np.float32)
        self.prev_tok_p = np.zeros((V, E, E), dtype=np.float32)
        self.prev_tok_hier_p = np.zeros((V, E, E), dtype=np.float32)

        for v, (p_name, lp) in enumerate(self.order):
            pi = self.pool_to_idx[p_name]
            prev_p_name = self.order[(v - 1) % V][0]
            sp_l = self.stage_prior_p[v].tolist()
            pp_l = self.pool_prior_p[pi].tolist()
            for se in range(E):
                self.prev_layer_p[v, se] = a2._trans_prob(
                    a2.prev_layer_trans, a2.prev_layer_totals, [(prev_p_name, p_name, lp, se)], sp_l
                )
                self.gen_prev_pool_p[v, se] = a2._trans_prob(
                    a2.gen_prev_pool_trans, a2.gen_prev_pool_totals, [(prev_p_name, p_name, se)], pp_l
                )
                self.prev_tok_p[v, se] = a2._trans_prob(
                    a2.prev_tok_trans, a2.prev_tok_totals, [(p_name, lp, se)], sp_l
                )
                self.prev_tok_hier_p[v, se] = a2._trans_prob(
                    a2.prev_tok_trans, a2.prev_tok_totals, [(p_name, lp, se)], self.gen_lagH_p[pi, se].tolist()
                )
                if lp is not None and lp > 0:
                    self.prev_loop_p[v, se] = a2._trans_prob(
                        a2.prev_loop_trans, a2.prev_loop_totals, [(p_name, lp - 1, lp, se)], sp_l
                    )
                    self.prev_loop_hier_p[v, se] = a2._trans_prob(
                        a2.prev_loop_trans, a2.prev_loop_totals, [(p_name, lp - 1, lp, se)], self.gen_lag1_p[pi, se].tolist()
                    )
                else:
                    self.prev_loop_p[v, se] = self.stage_prior_p[v]
                    self.prev_loop_hier_p[v, se] = self.gen_lag1_p[pi, se]


class FastPinnedExpertStore:
    """Packs all 700 routed experts into pinned host RAM (indexed as [p][e]) and frees GPU copies."""

    def __init__(self, model, device: torch.device):
        self.device = device
        self.num_layers = len(model.layers)
        first_exp = model.layers[0].moe.experts
        self.model_dim = int(first_exp.model_dim)
        self.inter_dim = int(first_exp.inter_dim)
        self.n_experts = int(first_exp.n_experts)
        self.w1_numel = self.inter_dim * self.model_dim
        self.w2_numel = self.model_dim * self.inter_dim
        self.w3_numel = self.inter_dim * self.model_dim
        self.expert_numel = self.w1_numel + self.w2_numel + self.w3_numel
        self.bytes_per_expert = self.expert_numel * 2  # bf16

        self.host_experts: list[list[torch.Tensor]] = []
        with torch.no_grad():
            for layer_id, layer in enumerate(model.layers):
                exp = layer.moe.experts
                w1 = exp.weight1.detach().view(self.n_experts, self.inter_dim, self.model_dim).cpu()
                w2 = exp.weight2.detach().view(self.n_experts, self.model_dim, self.inter_dim).cpu()
                w3 = exp.weight3.detach().view(self.n_experts, self.inter_dim, self.model_dim).cpu()
                layer_list: list[torch.Tensor] = []
                o1 = self.w1_numel
                o2 = o1 + self.w2_numel
                for e in range(self.n_experts):
                    flat = torch.empty(self.expert_numel, dtype=torch.bfloat16, pin_memory=True)
                    flat[:o1].copy_(w1[e].reshape(-1))
                    flat[o1:o2].copy_(w2[e].reshape(-1))
                    flat[o2:].copy_(w3[e].reshape(-1))
                    layer_list.append(flat)
                self.host_experts.append(layer_list)
                exp.weight1 = nn.Parameter(torch.empty(0, device=device, dtype=torch.bfloat16), requires_grad=False)
                exp.weight2 = nn.Parameter(torch.empty(0, device=device, dtype=torch.bfloat16), requires_grad=False)
                exp.weight3 = nn.Parameter(torch.empty(0, device=device, dtype=torch.bfloat16), requires_grad=False)

        gc.collect()
        torch.cuda.empty_cache()

    def expert_views(self, slot_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        o1 = self.w1_numel
        o2 = o1 + self.w2_numel
        return (
            slot_tensor[:o1].view(self.inter_dim, self.model_dim),
            slot_tensor[o1:o2].view(self.model_dim, self.inter_dim),
            slot_tensor[o2:].view(self.inter_dim, self.model_dim),
        )


class FullResidentFastExecutor:
    """Apples-to-apples full-resident GPU executor using the exact same active-2-expert F.linear path."""

    def __init__(self, model):
        self.w1: list[torch.Tensor] = []
        self.w2: list[torch.Tensor] = []
        self.w3: list[torch.Tensor] = []
        for layer in model.layers:
            exp = layer.moe.experts
            n_e = int(exp.n_experts)
            d = int(exp.model_dim)
            inter = int(exp.inter_dim)
            self.w1.append(exp.weight1.detach().view(n_e, inter, d))
            self.w2.append(exp.weight2.detach().view(n_e, d, inter))
            self.w3.append(exp.weight3.detach().view(n_e, inter, d))

    def execute_moe(
        self,
        layer_id: int,
        v_idx: int,
        layer: Any,
        x_in: torch.Tensor,
        load_balancing_type: str | None,
    ) -> tuple[torch.Tensor, Any]:
        moe = layer.moe
        bsz, slen, dim = x_in.shape
        residual = x_in
        x = moe.norm(x_in)
        if moe.fc1 is not None:
            y1 = moe.fc2(F.silu(moe.fc1(x)) * moe.fc3(x))
        else:
            y1 = None
        routing_scores, routing_indices, _tpe, aux_loss = moe.router(x, load_balancing_type)
        x_2d = x.view(bsz * slen, dim)
        w1_l = self.w1[layer_id]
        w2_l = self.w2[layer_id]
        w3_l = self.w3[layer_id]

        if slen == 1:
            e0, e1 = int(routing_indices[0, 0, 0].item()), int(routing_indices[0, 0, 1].item())
            w_scores = routing_scores[0, 0]
            y0 = F.linear(F.silu(F.linear(x_2d, w1_l[e0])) * F.linear(x_2d, w3_l[e0]), w2_l[e0])
            y_b = F.linear(F.silu(F.linear(x_2d, w1_l[e1])) * F.linear(x_2d, w3_l[e1]), w2_l[e1])
            y2 = (y0 * w_scores[0].to(y0.dtype) + y_b * w_scores[1].to(y_b.dtype)).view(bsz, slen, dim)
        else:
            idx_flat = routing_indices[0]
            w_flat = routing_scores[0]
            y2 = torch.zeros_like(x_2d)
            for e in torch.unique(idx_flat).tolist():
                mask = idx_flat == e
                tok_pos, k_pos = torch.nonzero(mask, as_tuple=True)
                xe = x_2d.index_select(0, tok_pos)
                ye = F.linear(F.silu(F.linear(xe, w1_l[e])) * F.linear(xe, w3_l[e]), w2_l[e])
                y2.index_add_(0, tok_pos, ye * w_flat[tok_pos, k_pos].unsqueeze(-1).to(ye.dtype))
            y2 = y2.view(bsz, slen, dim)

        y = y1 + y2 if y1 is not None else y2
        return moe.residual(y, residual, c=y), aux_loss


class FastGPUOffloadManager:
    """Vectorized (<1 ms/tok CPU overhead) strict peak-budget GPU expert offload + async prefetch manager."""

    def __init__(
        self,
        store: FastPinnedExpertStore,
        tables: CompiledA2Tables,
        hot_rank_idx: list[tuple[int, int]],
        *,
        policy: str,
        capacity: int,
        prefetch_k: int = 2,
        prefetch_thresh: float = 0.25,
        profile_events: bool = False,
    ):
        self.store = store
        self.device = store.device
        self.tab = tables
        self.policy = policy
        self.capacity = capacity
        self.prefetch_k = prefetch_k
        self.prefetch_thresh = float(prefetch_thresh)
        self.profile_events = profile_events

        self.use_pf = policy.endswith("_pf")
        self.is_loop = policy.startswith("loop")
        self.is_static = policy == "static_hot"

        P, E = tables.n_pools, tables.n_experts
        self.static_mask = np.zeros((P, E), dtype=np.bool_)
        if self.is_static:
            max_pin = capacity if capacity >= P * E else max(0, capacity - 1)
            for pi, ei in hot_rank_idx[:max_pin]:
                self.static_mask[pi, ei] = True

        # Fixed physical GPU slot pool (strictly `capacity` slots: resident + in-flight + staging <= capacity)
        self.gpu_pool = torch.empty((capacity, store.expert_numel), dtype=torch.bfloat16, device=self.device)
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.slot_events = [torch.cuda.Event(enable_timing=False) for _ in range(capacity)]
        self.use_events = [torch.cuda.Event(enable_timing=False) for _ in range(capacity)]
        self.reset_request()

    def reset_request(self) -> None:
        P, E, V = self.tab.n_pools, self.tab.n_experts, self.tab.n_visits
        C = self.capacity
        self.free_slots = deque(range(C))
        self.key_to_slot = np.full((P, E), -1, dtype=np.int32)
        self.slot_pool = np.full(C, -1, dtype=np.int32)
        self.slot_exp = np.full(C, -1, dtype=np.int32)
        self.slot_score = np.full(C, -1e9, dtype=np.float32)

        self.pool_priority = np.zeros((P, E), dtype=np.float32)
        self.pool_ema = np.zeros((P, E), dtype=np.float32)
        self.pool_freq = np.zeros((P, E), dtype=np.float32)
        self.pool_seen = np.zeros(P, dtype=np.float32)

        # Ring buffer of recent top-2 experts per pool (for lag-1 & lag-H) + last top-2 per visit v
        self.max_h = int(self.tab.pool_h_span.max())
        self.pool_hist = np.full((P, self.max_h, 2), -1, dtype=np.int32)
        self.pool_hist_len = np.zeros(P, dtype=np.int32)
        self.pool_hist_ptr = np.zeros(P, dtype=np.int32)
        self.visit_last = np.full((V, 2), -1, dtype=np.int32)

        # Pending prefetches for immediate next visit: list of (pi, ei, slot)
        self.in_flight: list[tuple[int, int, int]] = []
        self.protected_slots: list[int] = []

        self.in_decode = False
        self.reset_decode_stats()

    def reset_decode_stats(self) -> None:
        self.accesses = 0
        self.demand_misses = 0
        self.prefetch_issued = 0
        self.prefetch_timely = 0
        self.prefetch_late = 0
        self.prefetch_wasted = 0
        self.exposed_wait_ms = 0.0
        self.predictor_cpu_ms = 0.0
        self.demand_h2d_bytes = 0
        self.prefetch_h2d_bytes = 0

    def _acquire_slot(self) -> int:
        if self.free_slots:
            return self.free_slots.popleft()
        # Mask protected slots (<=4 ints) to +inf, run single SIMD np.argmin over C float32 slots (~0.5 us!)
        saved: list[tuple[int, float]] = []
        for s in self.protected_slots:
            if s >= 0:
                saved.append((s, float(self.slot_score[s])))
                self.slot_score[s] = 1e9
        victim = int(np.argmin(self.slot_score))
        for s, old in saved:
            self.slot_score[s] = old
        vp = int(self.slot_pool[victim])
        ve = int(self.slot_exp[victim])
        if vp >= 0:
            self.key_to_slot[vp, ve] = -1
        return victim

    def _start_async_copy(self, pi: int, ei: int, slot: int) -> None:
        host_t = self.store.host_experts[pi][ei]
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_event(self.use_events[slot])
            self.gpu_pool[slot].copy_(host_t, non_blocking=True)
            self.slot_events[slot].record(self.copy_stream)

    def _update_hazard_and_prefetch(self, pi: int, v_idx: int, e0: int, e1: int, w0: float, w1: float) -> None:
        t0 = time.perf_counter()
        tab = self.tab

        # Update ring history & EMA (pure C NumPy ops on 25-element slice)
        self.visit_last[v_idx, 0] = e0
        self.visit_last[v_idx, 1] = e1

        h_span = int(tab.pool_h_span[pi])
        ptr = int(self.pool_hist_ptr[pi])
        self.pool_hist[pi, ptr, 0] = e0
        self.pool_hist[pi, ptr, 1] = e1
        self.pool_hist_ptr[pi] = (ptr + 1) % h_span
        if self.pool_hist_len[pi] < h_span:
            self.pool_hist_len[pi] += 1

        ema = self.pool_ema[pi]
        ema *= (1.0 - tab.alpha)
        ema[e0] += tab.alpha * w0
        ema[e1] += tab.alpha * w1
        self.pool_seen[pi] += 1.0

        # 1. Vectorized hazard score update for pool `pi`
        next_v = int(tab.v_next_v[v_idx])
        w_d1 = float(tab.v_w1[v_idx])
        has_lagH = int(self.pool_hist_len[pi]) >= h_span
        # lagH is the oldest slot in ring buffer (= current pool_hist_ptr after increment!)
        lagH_ptr = int(self.pool_hist_ptr[pi])

        if self.is_loop:
            lo = tab.stage_prior_lo[next_v].copy()
            base_lo = tab.stage_prior_lo[next_v]
            # Prev-token same-stage (hierarchical backoff)
            pt0 = int(self.visit_last[next_v, 0])
            if pt0 >= 0:
                pt1 = int(self.visit_last[next_v, 1])
                p_pt = 0.5 * (tab.prev_tok_hier_p[next_v, pt0] + tab.prev_tok_hier_p[next_v, pt1])
                lo += _logit_np(p_pt) - base_lo
            elif has_lagH:
                h0, h1 = int(self.pool_hist[pi, lagH_ptr, 0]), int(self.pool_hist[pi, lagH_ptr, 1])
                p_lh = 0.5 * (tab.gen_lagH_p[pi, h0] + tab.gen_lagH_p[pi, h1])
                lo += _logit_np(p_lh) - base_lo

            # Prev-loop same-pool (when next_v is in same token)
            if not tab.v_is_boundary[v_idx] and tab.v_has_prev_loop[next_v]:
                p_pl = 0.5 * (tab.prev_loop_hier_p[next_v, e0] + tab.prev_loop_hier_p[next_v, e1])
            else:
                p_pl = 0.5 * (tab.gen_lag1_p[pi, e0] + tab.gen_lag1_p[pi, e1])
            lo += _logit_np(p_pl) - base_lo

            p_next = _sigmoid_np(lo)
            haz = w_d1 * (0.85 * p_next + 0.15 * np.minimum(1.0, ema)) + tab.haz_future_loop[v_idx]
        else:
            base_lo = tab.pool_prior_lo[pi]
            p_l1 = 0.5 * (tab.gen_lag1_p[pi, e0] + tab.gen_lag1_p[pi, e1])
            lo = _logit_np(p_l1)
            if has_lagH:
                h0, h1 = int(self.pool_hist[pi, lagH_ptr, 0]), int(self.pool_hist[pi, lagH_ptr, 1])
                p_lh = 0.5 * (tab.gen_lagH_p[pi, h0] + tab.gen_lagH_p[pi, h1])
                lo = lo + _logit_np(p_lh) - base_lo
            p_next = _sigmoid_np(lo)
            haz = w_d1 * (0.85 * p_next + 0.15 * np.minimum(1.0, ema)) + tab.haz_future_gen[v_idx]

        # Blend online frequency density (fixes long-horizon residency at medium budgets!)
        if tab.freq_weight > 0.0:
            haz = haz + (tab.freq_weight * tab.pool_density[pi] / (self.pool_seen[pi] + 8.0)) * self.pool_freq[pi]

        self.pool_priority[pi] = haz
        # Refresh scores of resident slots belonging to pool `pi` in ONE vectorized op
        res_slots = self.key_to_slot[pi]
        valid = res_slots >= 0
        if np.any(valid):
            self.slot_score[res_slots[valid]] = haz[valid]

        # 2. Vectorized async prefetch for immediate next visit nv_idx = (v_idx + 1) % V
        if self.use_pf and self.in_decode:
            nv = (v_idx + 1) % tab.n_visits
            npi = int(tab.v_pool[nv])
            nh_span = int(tab.pool_h_span[npi])
            nh_len = int(self.pool_hist_len[npi])
            # Most recent entry in ring buffer for npi
            n_last_ptr = (int(self.pool_hist_ptr[npi]) - 1) % nh_span

            if self.is_loop:
                base_lo = tab.stage_prior_lo[nv]
                # prev_phys_layer (e0, e1 from current visit v_idx!)
                p_pl = 0.5 * (tab.prev_layer_p[nv, e0] + tab.prev_layer_p[nv, e1])
                lo = _logit_np(p_pl)
                # prev_tok_same_stage
                npt0 = int(self.visit_last[nv, 0])
                if npt0 >= 0:
                    npt1 = int(self.visit_last[nv, 1])
                    p_pt = 0.5 * (tab.prev_tok_p[nv, npt0] + tab.prev_tok_p[nv, npt1])
                    lo = lo + _logit_np(p_pt) - base_lo
                # prev_loop_trans
                if tab.v_has_prev_loop[nv] and nh_len >= 1:
                    nl0, nl1 = int(self.pool_hist[npi, n_last_ptr, 0]), int(self.pool_hist[npi, n_last_ptr, 1])
                    p_lp = 0.5 * (tab.prev_loop_p[nv, nl0] + tab.prev_loop_p[nv, nl1])
                    lo = lo + _logit_np(p_lp) - base_lo
                probs = _sigmoid_np(lo)
            else:
                base_lo = tab.pool_prior_lo[npi]
                p_pp = 0.5 * (tab.gen_prev_pool_p[nv, e0] + tab.gen_prev_pool_p[nv, e1])
                lo = _logit_np(p_pp)
                if nh_len >= 1:
                    nl0, nl1 = int(self.pool_hist[npi, n_last_ptr, 0]), int(self.pool_hist[npi, n_last_ptr, 1])
                    p_l1 = 0.5 * (tab.gen_lag1_p[npi, nl0] + tab.gen_lag1_p[npi, nl1])
                    lo = lo + _logit_np(p_l1) - base_lo
                if nh_len >= nh_span:
                    nh_ptr = int(self.pool_hist_ptr[npi])
                    nh0, nh1 = int(self.pool_hist[npi, nh_ptr, 0]), int(self.pool_hist[npi, nh_ptr, 1])
                    p_lh = 0.5 * (tab.gen_lagH_p[npi, nh0] + tab.gen_lagH_p[npi, nh1])
                    lo = lo + _logit_np(p_lh) - base_lo
                probs = _sigmoid_np(lo)

            # Mask out already-resident experts in O(E)
            not_res = self.key_to_slot[npi] < 0
            cand_probs = np.where(not_res, probs, -1.0)
            # Pick top-prefetch_k missing candidates above threshold
            for _ in range(self.prefetch_k):
                best_e = int(np.argmax(cand_probs))
                best_p = float(cand_probs[best_e])
                if best_p < self.prefetch_thresh:
                    break
                cand_probs[best_e] = -1.0
                slot = self._acquire_slot()
                self.key_to_slot[npi, best_e] = slot
                self.slot_pool[slot] = npi
                self.slot_exp[slot] = best_e
                self.slot_score[slot] = float(self.pool_priority[npi, best_e])
                self.protected_slots.append(slot)
                self.in_flight.append((npi, best_e, slot))
                self._start_async_copy(npi, best_e, slot)
                self.prefetch_issued += 1
                self.prefetch_h2d_bytes += self.store.bytes_per_expert

        if self.in_decode:
            self.predictor_cpu_ms += (time.perf_counter() - t0) * 1000.0

    def execute_moe(
        self,
        layer_id: int,
        v_idx: int,
        layer: Any,
        x_in: torch.Tensor,
        load_balancing_type: str | None,
    ) -> tuple[torch.Tensor, Any]:
        moe = layer.moe
        bsz, slen, dim = x_in.shape
        residual = x_in
        x = moe.norm(x_in)

        if moe.fc1 is not None:
            y1 = moe.fc2(F.silu(moe.fc1(x)) * moe.fc3(x))
        else:
            y1 = None

        routing_scores, routing_indices, _tpe, aux_loss = moe.router(x, load_balancing_type)
        pi = layer_id
        x_2d = x.view(bsz * slen, dim)
        default_stream = torch.cuda.current_stream(self.device)

        if slen == 1:
            e0 = int(routing_indices[0, 0, 0].item())
            e1 = int(routing_indices[0, 0, 1].item())
            tgt_w = routing_scores[0, 0]

            # 1. Settle in-flight prefetches from previous visit
            if self.in_flight:
                for pf_pi, pf_ei, pf_slot in self.in_flight:
                    useful = (pf_pi == pi) and (pf_ei == e0 or pf_ei == e1)
                    if self.in_decode:
                        if useful:
                            if self.slot_events[pf_slot].query():
                                self.prefetch_timely += 1
                            else:
                                self.prefetch_late += 1
                        else:
                            self.prefetch_wasted += 1
                self.in_flight.clear()

            # Protect slots already holding e0 or e1
            self.protected_slots = []
            s0 = int(self.key_to_slot[pi, e0])
            s1 = int(self.key_to_slot[pi, e1])
            if s0 >= 0:
                self.protected_slots.append(s0)
            if s1 >= 0:
                self.protected_slots.append(s1)

            # 2. Fetch any missing demand experts (pipelined on copy_stream)
            slots_info: list[tuple[int, int, bool]] = []
            for e in (e0, e1):
                if self.in_decode:
                    self.accesses += 1
                self.pool_freq[pi, e] += 1.0
                slot = int(self.key_to_slot[pi, e])
                if slot >= 0:
                    slots_info.append((e, slot, False))
                else:
                    if self.in_decode:
                        self.demand_misses += 1
                        self.demand_h2d_bytes += self.store.bytes_per_expert
                    slot = self._acquire_slot()
                    self.protected_slots.append(slot)
                    self._start_async_copy(pi, e, slot)
                    admit = (not self.is_static) or bool(self.static_mask[pi, e])
                    if admit:
                        self.key_to_slot[pi, e] = slot
                        self.slot_pool[slot] = pi
                        self.slot_exp[slot] = e
                        self.slot_score[slot] = 1.0 if self.is_static else float(self.pool_priority[pi, e])
                        slots_info.append((e, slot, False))
                    else:
                        slots_info.append((e, slot, True))

            # 3. Compute both routed experts
            y2 = None
            for k_i, (e, slot, release_after) in enumerate(slots_info):
                ev = self.slot_events[slot]
                if self.profile_events and self.in_decode and not ev.query():
                    tw0 = time.perf_counter()
                    ev.synchronize()
                    self.exposed_wait_ms += (time.perf_counter() - tw0) * 1000.0
                else:
                    default_stream.wait_event(ev)
                w1, w2, w3 = self.store.expert_views(self.gpu_pool[slot])
                ye = F.linear(F.silu(F.linear(x_2d, w1)) * F.linear(x_2d, w3), w2)
                self.use_events[slot].record(default_stream)
                term = ye * tgt_w[k_i].to(ye.dtype)
                y2 = term if y2 is None else (y2 + term)
                if release_after:
                    self.slot_pool[slot] = -1
                    self.slot_exp[slot] = -1
                    self.slot_score[slot] = -1e9
                    self.free_slots.append(slot)

            # 4. Update vectorized causal hazard & launch async prefetch for NEXT visit!
            if not self.is_static:
                self._update_hazard_and_prefetch(
                    pi, v_idx, e0, e1, float(tgt_w[0].item()), float(tgt_w[1].item())
                )
            y2 = y2.view(bsz, slen, dim)
        else:
            # Prefill path
            idx_flat = routing_indices[0]
            w_flat = routing_scores[0]
            y2 = torch.zeros_like(x_2d)
            for e in torch.unique(idx_flat).tolist():
                self.pool_freq[pi, e] += 1.0
                slot = int(self.key_to_slot[pi, e])
                release_after = False
                if slot < 0:
                    self.protected_slots = []
                    slot = self._acquire_slot()
                    self.protected_slots = [slot]
                    self._start_async_copy(pi, e, slot)
                    admit = (not self.is_static) or bool(self.static_mask[pi, e])
                    if admit:
                        self.key_to_slot[pi, e] = slot
                        self.slot_pool[slot] = pi
                        self.slot_exp[slot] = e
                        self.slot_score[slot] = 1.0 if self.is_static else float(self.pool_priority[pi, e])
                    else:
                        release_after = True
                default_stream.wait_event(self.slot_events[slot])
                w1, w2, w3 = self.store.expert_views(self.gpu_pool[slot])
                mask = idx_flat == e
                tok_pos, k_pos = torch.nonzero(mask, as_tuple=True)
                xe = x_2d.index_select(0, tok_pos)
                ye = F.linear(F.silu(F.linear(xe, w1)) * F.linear(xe, w3), w2)
                self.use_events[slot].record(default_stream)
                y2.index_add_(0, tok_pos, ye * w_flat[tok_pos, k_pos].unsqueeze(-1).to(ye.dtype))
                if release_after:
                    self.slot_pool[slot] = -1
                    self.slot_exp[slot] = -1
                    self.slot_score[slot] = -1e9
                    self.free_slots.append(slot)
            if not self.is_static:
                le0 = int(idx_flat[-1, 0].item())
                le1 = int(idx_flat[-1, 1].item())
                self._update_hazard_and_prefetch(
                    pi, v_idx, le0, le1, float(w_flat[-1, 0].item()), float(w_flat[-1, 1].item())
                )
            y2 = y2.view(bsz, slen, dim)

        y = y1 + y2 if y1 is not None else y2
        return moe.residual(y, residual, c=y), aux_loss


def _bootstrap_ci_pct(a_vals: list[float], b_vals: list[float], n_boot: int = 1000, seed: int = 20261122) -> tuple[float, float, float]:
    """Paired percent speedup of `a` over `b` (positive = `a` is faster / lower than `b`)."""
    import random

    diffs = [100.0 * (b - a) / b for a, b in zip(a_vals, b_vals) if b > 1e-9]
    if not diffs:
        return 0.0, 0.0, 0.0
    mean_v = statistics.mean(diffs)
    if len(diffs) == 1:
        return mean_v, mean_v, mean_v
    rng = random.Random(seed)
    n = len(diffs)
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return mean_v, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]


@torch.inference_mode()
def run_fast_gpu_benchmark_for_model(
    model_name: str,
    model_spec: dict[str, Any],
    model_dir: Path,
    trace_path: Path,
    prompts_all: list[dict[str, Any]],
    cfg: dict[str, Any],
    *,
    device: torch.device,
    budgets: list[int],
    n_eval_prompts: int = 12,
    decode_tokens: int = 48,
    gamma: float = 0.75,
    freq_weight: float = 0.15,
    prefetch_thresh: float = 0.25,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    meta, trace_reqs = read_ifm_trace(trace_path)
    sched = _build_schedule_info(meta)
    fit_set, _val_set, eval_set = load_trace_splits(trace_path, meta)
    fit_reqs = [r for r in trace_reqs if r["request_id"] in fit_set]

    a2 = A2Models(meta, sched, cfg["a2"])
    a2.fit(fit_reqs, shuffle_seed=int(cfg["a2"]["shuffle_seed"]))
    tables = CompiledA2Tables(meta, sched, a2, gamma=gamma, freq_weight=freq_weight)

    order = sched["order"]
    hot_counts: Counter = Counter()
    for req in fit_reqs:
        for step in req["steps"]:
            for v_idx, visit in enumerate(step["visits"]):
                pi = tables.pool_to_idx[order[v_idx][0]]
                for e in visit[0]:
                    hot_counts[pi, e] += 1
    all_keys = [(pi, e) for pi in range(tables.n_pools) for e in range(tables.n_experts)]
    hot_rank_idx = sorted(all_keys, key=lambda k: (-hot_counts[k], k[0], k[1]))

    eval_prompts_pool = [p for p in prompts_all if p["id"] in eval_set]
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in eval_prompts_pool:
        by_cat[p["category"]].append(p)
    per_cat = max(1, n_eval_prompts // 3)
    eval_prompts: list[dict[str, Any]] = []
    for i in range(per_cat):
        for cat in ("text", "math", "code"):
            if i < len(by_cat[cat]):
                eval_prompts.append(by_cat[cat][i])
    eval_prompts = eval_prompts[:n_eval_prompts]

    torch.cuda.reset_peak_memory_stats(device)
    model, tokenizer, _ = load_ifm_model(model_dir, device=str(device), attention_backend="sdpa")
    engine = LoopedIncrementalEngine(model, model_spec)
    full_fast = FullResidentFastExecutor(model)

    encoded_prompts = [
        (p, tokenizer.encode(format_ifm_prompt(p)[0], bos=True, eos=False)[-512:])
        for p in eval_prompts
    ]

    # Warmup GPU kernels
    engine.reset()
    _ = engine.forward_step(
        torch.tensor([encoded_prompts[0][1][:16]], device=device, dtype=torch.long),
        moe_executor=full_fast.execute_moe,
    )
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    ref_tokens: dict[str, list[int]] = {}
    req_rows: list[dict[str, Any]] = []

    # 1. Apples-to-apples `full_resident_fast` ceiling
    for rec, pids in encoded_prompts:
        engine.reset()
        torch.cuda.synchronize(device)
        t_start = time.perf_counter()
        logits = engine.forward_step(
            torch.tensor([pids], device=device, dtype=torch.long),
            moe_executor=full_fast.execute_moe,
        )
        nxt = int(logits[0, -1].argmax().item())
        torch.cuda.synchronize(device)
        t_first = time.perf_counter()
        gen = [nxt]
        for _ in range(decode_tokens - 1):
            logits = engine.forward_step(
                torch.tensor([[nxt]], device=device, dtype=torch.long),
                moe_executor=full_fast.execute_moe,
            )
            nxt = int(logits[0, -1].argmax().item())
            gen.append(nxt)
        torch.cuda.synchronize(device)
        t_last = time.perf_counter()
        ref_tokens[rec["id"]] = gen
        req_rows.append(
            {
                "model": model_name,
                "budget_experts": 700,
                "budget_gib": 14.6118,
                "policy": "full_resident_fast",
                "request_id": rec["id"],
                "category": rec["category"],
                "prompt_tokens": len(pids),
                "decode_tokens": len(gen),
                "exact_token_match": 1,
                "ttft_ms": round((t_first - t_start) * 1000.0, 3),
                "tpot_ms": round(((t_last - t_first) * 1000.0) / max(1, len(gen) - 1), 3),
                "demand_hit_rate": 1.0,
                "demand_h2d_mib_per_tok": 0.0,
                "prefetch_h2d_mib_per_tok": 0.0,
                "total_h2d_mib_per_tok": 0.0,
                "timely_pf": 0,
                "late_pf": 0,
                "wasted_pf": 0,
                "exposed_wait_ms_per_tok": 0.0,
                "predictor_cpu_ms_per_tok": 0.0,
                "peak_allocated_gib": round(torch.cuda.max_memory_allocated(device) / (1024**3), 3),
                "peak_reserved_gib": round(torch.cuda.max_memory_reserved(device) / (1024**3), 3),
            }
        )

    del full_fast
    store = FastPinnedExpertStore(model, device)
    bpe = store.bytes_per_expert

    # 2. All 5 offloaded policies across tight VRAM budgets
    for cap in budgets:
        budget_gib = round(cap * bpe / (1024**3), 4)
        for pol in FAST_GPU_POLICIES:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            mgr = FastGPUOffloadManager(
                store,
                tables,
                hot_rank_idx,
                policy=pol,
                capacity=cap,
                prefetch_k=2,
                prefetch_thresh=prefetch_thresh,
                profile_events=False,
            )

            # Warmup 1 short decode step on slot pool
            engine.reset()
            mgr.reset_request()
            _ = engine.forward_step(
                torch.tensor([encoded_prompts[0][1][:16]], device=device, dtype=torch.long),
                moe_executor=mgr.execute_moe,
            )
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

            # Clean timing pass (zero tracing, zero per-event sync!)
            for rec, pids in encoded_prompts:
                engine.reset()
                mgr.reset_request()
                mgr.in_decode = False
                torch.cuda.synchronize(device)
                t_start = time.perf_counter()
                logits = engine.forward_step(
                    torch.tensor([pids], device=device, dtype=torch.long),
                    moe_executor=mgr.execute_moe,
                )
                nxt = int(logits[0, -1].argmax().item())
                torch.cuda.synchronize(device)
                t_first = time.perf_counter()

                mgr.in_decode = True
                mgr.reset_decode_stats()
                gen = [nxt]
                for _ in range(decode_tokens - 1):
                    logits = engine.forward_step(
                        torch.tensor([[nxt]], device=device, dtype=torch.long),
                        moe_executor=mgr.execute_moe,
                    )
                    nxt = int(logits[0, -1].argmax().item())
                    gen.append(nxt)
                torch.cuda.synchronize(device)
                t_last = time.perf_counter()

                n_dec = max(1, len(gen) - 1)
                req_rows.append(
                    {
                        "model": model_name,
                        "budget_experts": cap,
                        "budget_gib": budget_gib,
                        "policy": pol,
                        "request_id": rec["id"],
                        "category": rec["category"],
                        "prompt_tokens": len(pids),
                        "decode_tokens": len(gen),
                        "exact_token_match": int(gen == ref_tokens[rec["id"]]),
                        "ttft_ms": round((t_first - t_start) * 1000.0, 3),
                        "tpot_ms": round(((t_last - t_first) * 1000.0) / n_dec, 3),
                        "demand_hit_rate": round(1.0 - mgr.demand_misses / max(1, mgr.accesses), 6),
                        "demand_h2d_mib_per_tok": round((mgr.demand_h2d_bytes / (1024**2)) / n_dec, 3),
                        "prefetch_h2d_mib_per_tok": round((mgr.prefetch_h2d_bytes / (1024**2)) / n_dec, 3),
                        "total_h2d_mib_per_tok": round(
                            ((mgr.demand_h2d_bytes + mgr.prefetch_h2d_bytes) / (1024**2)) / n_dec, 3
                        ),
                        "timely_pf": mgr.prefetch_timely,
                        "late_pf": mgr.prefetch_late,
                        "wasted_pf": mgr.prefetch_wasted,
                        "exposed_wait_ms_per_tok": 0.0,
                        "predictor_cpu_ms_per_tok": round(mgr.predictor_cpu_ms / n_dec, 3),
                        "peak_allocated_gib": round(torch.cuda.max_memory_allocated(device) / (1024**3), 3),
                        "peak_reserved_gib": round(torch.cuda.max_memory_reserved(device) / (1024**3), 3),
                    }
                )

            # Separate 3-prompt profiling pass for exposed CUDA wait
            mgr.profile_events = True
            prof_waits: list[float] = []
            for rec, pids in encoded_prompts[:3]:
                engine.reset()
                mgr.reset_request()
                mgr.in_decode = False
                logits = engine.forward_step(
                    torch.tensor([pids], device=device, dtype=torch.long),
                    moe_executor=mgr.execute_moe,
                )
                nxt = int(logits[0, -1].argmax().item())
                torch.cuda.synchronize(device)
                mgr.in_decode = True
                mgr.reset_decode_stats()
                for _ in range(decode_tokens - 1):
                    logits = engine.forward_step(
                        torch.tensor([[nxt]], device=device, dtype=torch.long),
                        moe_executor=mgr.execute_moe,
                    )
                    nxt = int(logits[0, -1].argmax().item())
                torch.cuda.synchronize(device)
                prof_waits.append(mgr.exposed_wait_ms / max(1, decode_tokens - 1))
            mean_wait = round(statistics.mean(prof_waits), 3) if prof_waits else 0.0
            for r in req_rows:
                if r["model"] == model_name and r["budget_experts"] == cap and r["policy"] == pol:
                    r["exposed_wait_ms_per_tok"] = mean_wait

            del mgr
            gc.collect()
            torch.cuda.empty_cache()

    # Summary & paired TPOT comparisons
    by_bp: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for r in req_rows:
        by_bp[int(r["budget_experts"]), r["policy"]].append(r)

    summary_rows: list[dict[str, Any]] = []
    for (cap, pol), items in sorted(by_bp.items()):
        tpots = [x["tpot_ms"] for x in items]
        tpots_s = sorted(tpots)
        p95 = tpots_s[min(len(tpots_s) - 1, int(round(0.95 * (len(tpots_s) - 1))))]
        tot_pf = sum(x["timely_pf"] + x["late_pf"] + x["wasted_pf"] for x in items)
        summary_rows.append(
            {
                "model": model_name,
                "budget_experts": cap,
                "budget_gib": items[0]["budget_gib"],
                "policy": pol,
                "requests": len(items),
                "exact_token_match_rate": round(statistics.mean(x["exact_token_match"] for x in items), 4),
                "tpot_mean_ms": round(statistics.mean(tpots), 3),
                "tpot_p50_ms": round(statistics.median(tpots), 3),
                "tpot_p95_ms": round(p95, 3),
                "ttft_mean_ms": round(statistics.mean(x["ttft_ms"] for x in items), 2),
                "demand_hit_rate": round(statistics.mean(x["demand_hit_rate"] for x in items), 5),
                "demand_h2d_mib_per_tok": round(statistics.mean(x["demand_h2d_mib_per_tok"] for x in items), 2),
                "prefetch_h2d_mib_per_tok": round(statistics.mean(x["prefetch_h2d_mib_per_tok"] for x in items), 2),
                "total_h2d_mib_per_tok": round(statistics.mean(x["total_h2d_mib_per_tok"] for x in items), 2),
                "timely_pf_rate": round(sum(x["timely_pf"] for x in items) / max(1, tot_pf), 4),
                "useful_pf_rate": round(sum(x["timely_pf"] + x["late_pf"] for x in items) / max(1, tot_pf), 4),
                "wasted_pf_rate": round(sum(x["wasted_pf"] for x in items) / max(1, tot_pf), 4),
                "exposed_wait_ms_per_tok": round(statistics.mean(x["exposed_wait_ms_per_tok"] for x in items), 3),
                "predictor_cpu_ms_per_tok": round(statistics.mean(x["predictor_cpu_ms_per_tok"] for x in items), 3),
                "peak_allocated_gib": max(x["peak_allocated_gib"] for x in items),
                "peak_reserved_gib": max(x["peak_reserved_gib"] for x in items),
            }
        )

    paired_rows: list[dict[str, Any]] = []
    pair_specs = [
        ("loop_nopf", "static_hot"),
        ("loop_nopf", "general_nopf"),
        ("loop_pf", "static_hot"),
        ("loop_pf", "general_nopf"),
        ("loop_pf", "loop_nopf"),
        ("loop_pf", "general_pf"),
        ("general_pf", "static_hot"),
        ("general_pf", "general_nopf"),
    ]
    for cap in budgets:
        for pol, base in pair_specs:
            p_map = {x["request_id"]: x for x in by_bp[cap, pol]}
            b_map = {x["request_id"]: x for x in by_bp[cap, base]}
            rids = sorted(p_map.keys())
            p_tpot = [p_map[r]["tpot_ms"] for r in rids]
            b_tpot = [b_map[r]["tpot_ms"] for r in rids]
            m_pct, lo_pct, hi_pct = _bootstrap_ci_pct(p_tpot, b_tpot)
            wins = sum(1 for a, b in zip(p_tpot, b_tpot) if a < b)
            losses = sum(1 for a, b in zip(p_tpot, b_tpot) if a > b)
            paired_rows.append(
                {
                    "model": model_name,
                    "budget_experts": cap,
                    "budget_gib": by_bp[cap, pol][0]["budget_gib"],
                    "policy": pol,
                    "baseline": base,
                    "requests": len(rids),
                    "policy_tpot_ms": round(statistics.mean(p_tpot), 3),
                    "baseline_tpot_ms": round(statistics.mean(b_tpot), 3),
                    "tpot_saved_ms": round(statistics.mean(b_tpot) - statistics.mean(p_tpot), 3),
                    "tpot_saved_pct": round(m_pct, 3),
                    "ci95_lo_pct": round(lo_pct, 3),
                    "ci95_hi_pct": round(hi_pct, 3),
                    "wins": wins,
                    "losses": losses,
                }
            )

    del store, engine, model
    gc.collect()
    torch.cuda.empty_cache()
    return summary_rows, req_rows, paired_rows


def _build_fast_report(summary_rows: list[dict[str, Any]], paired_rows: list[dict[str, Any]], gpu: dict[str, Any]) -> str:
    lines = [
        "# IFM Part I MoE B2-v2 真机向量化 GPU Offload 与异步预取实测报告\n",
        f"- 测试 GPU：`{gpu['name']}`（独立空闲卡 GPU `{gpu['index']}`）；采用预编译 NumPy float32 log-odds 查表 + 连续数组槽池（消除纯 Python 预测器/堆开销），并加入同执行路径 `full_resident_fast` 公平参照。\n",
        "## 一、端到端实测总表（12 配对 Eval 请求 × 48 Decode Tokens）\n",
        "| 模型 | 预算 (GiB / 专家) | 策略 | Token 一致 | TPOT Mean (ms) | TPOT p50 (ms) | TTFT (ms) | Demand Hit | Total H2D (MiB/tok) | 有效/及时预取 | 暴露等待 (ms/tok) | 预测器 CPU (ms/tok) | 峰值 Alloc/Res (GiB) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['model']} | {r['budget_gib']:.2f} ({r['budget_experts']}) | `{r['policy']}` | "
            f"{100*r['exact_token_match_rate']:.2f}% | **{r['tpot_mean_ms']:.2f}** | {r['tpot_p50_ms']:.2f} | "
            f"{r['ttft_mean_ms']:.1f} | {100*r['demand_hit_rate']:.2f}% | {r['total_h2d_mib_per_tok']:.1f} | "
            f"{100*r['useful_pf_rate']:.2f}% / {100*r['timely_pf_rate']:.2f}% | {r['exposed_wait_ms_per_tok']:.2f} | "
            f"{r['predictor_cpu_ms_per_tok']:.2f} | {r['peak_allocated_gib']:.2f} / {r['peak_reserved_gib']:.2f} |"
        )
    lines.append("\n## 二、配对端到端 TPOT 收益（95% Bootstrap CI）\n")
    lines.append("| 模型 | 预算 (GiB) | 对比 | 策略 TPOT | 基线 TPOT | 净省 (ms) | TPOT 降低 (95% CI) | 胜/负 |")
    lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for p in paired_rows:
        lines.append(
            f"| {p['model']} | {p['budget_gib']:.2f} | `{p['policy']}` vs `{p['baseline']}` | "
            f"{p['policy_tpot_ms']:.2f} | {p['baseline_tpot_ms']:.2f} | {p['tpot_saved_ms']:+.2f} | "
            f"{p['tpot_saved_pct']:+.2f}% [{p['ci95_lo_pct']:+.2f}, {p['ci95_hi_pct']:+.2f}] | {p['wins']}/{p['losses']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="B2-v2 vectorized real GPU offload benchmark")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--xllm-loop-dir", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--gpu", default=None)
    args = parser.parse_args()

    cfg = read_json(Path(args.config))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bcfg = cfg["stage_b2_fast"]

    bootstrap_xllm(args.xllm_loop_dir, attention_backend=cfg.get("attention_backend", "sdpa"))
    gpu = select_idle_gpu(args.gpu)
    device = torch.device(f"cuda:{gpu['index']}")
    torch.cuda.set_device(device)
    print(f"[B2-v2] Selected strictly idle GPU {gpu['index']} ({gpu['name']}, {gpu['memory_used_mib']} MiB used)")

    def _assert_no_foreign_gpu_procs(gpu_uuid: str, gpu_index: str) -> None:
        my_pid = str(os.getpid())
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
            check=True,
            text=True,
            capture_output=True,
            timeout=15,
        ).stdout.splitlines()
        for line in out:
            if not line.strip():
                continue
            u, pid = [p.strip() for p in line.split(",")]
            if u == gpu_uuid and pid != my_pid:
                raise RuntimeError(f"Foreign process PID={pid} appeared on GPU {gpu_index}; aborting safely.")

    orx.plan(
        [
            {"phase": "b2_gpu_fast", "of": len(cfg["models"]), "unit": "models"},
            {"phase": "report", "of": 1, "unit": "report"},
        ]
    )
    prog = orx.progress(total=len(cfg["models"]), unit="models", phase="b2_gpu_fast")

    all_prompts = list(read_jsonl(Path(cfg["prompts_file"])))
    trace_root = Path(args.trace_dir)
    budgets = [int(b) for b in bcfg["gpu_budgets_experts"]]

    all_summary: list[dict[str, Any]] = []
    all_reqs: list[dict[str, Any]] = []
    all_paired: list[dict[str, Any]] = []

    for idx, model_name in enumerate(cfg["models"], start=1):
        _assert_no_foreign_gpu_procs(gpu["uuid"], gpu["index"])
        spec = cfg["checkpoints"][model_name]
        mdir = download_ifm_artifact(spec["repo"], Path(args.model_root), local_files_only=True)
        trace_path = trace_root / model_name / "trace_freegen.jsonl.gz"
        srows, rrows, prows = run_fast_gpu_benchmark_for_model(
            model_name,
            spec,
            mdir,
            trace_path,
            all_prompts,
            cfg,
            device=device,
            budgets=budgets,
            n_eval_prompts=int(bcfg.get("gpu_eval_prompts", 12)),
            decode_tokens=int(bcfg.get("gpu_decode_tokens", 48)),
            gamma=float(bcfg.get("gamma", 0.75)),
            freq_weight=float(bcfg.get("freq_weight", 0.15)),
            prefetch_thresh=float(bcfg.get("prefetch_thresh", 0.25)),
        )
        all_summary.extend(srows)
        all_reqs.extend(rrows)
        all_paired.extend(prows)
        prog.step(idx)
        prog.flush()
        print(f"[B2-v2 {model_name}] done ({len(srows)} rows)")

    write_csv(out_dir / "b2_fast_gpu_summary.csv", all_summary)
    write_csv(out_dir / "b2_fast_gpu_requests.csv", all_reqs)
    write_csv(out_dir / "b2_fast_gpu_paired.csv", all_paired)

    with prog.phase("report", total=1, unit="report"):
        report_md = _build_fast_report(all_summary, all_paired, gpu)
        (out_dir / "report_b2_fast.md").write_text(report_md, encoding="utf-8")

        headline: dict[str, Any] = {}
        for m in cfg["models"]:
            tag = "ouro" if "Ouro" in m else "huginn"
            gmap = {(int(r["budget_experts"]), r["policy"]): r for r in all_summary if r["model"] == m}
            if (700, "full_resident_fast") in gmap:
                headline[f"{tag}_tpot_full_fast_ms"] = gmap[700, "full_resident_fast"]["tpot_mean_ms"]
            for cap, btag in ((95, "2g"), (191, "4g"), (302, "6p3g")):
                if (cap, "loop_pf") in gmap:
                    headline[f"{tag}_tpot_{btag}_loop_pf_ms"] = gmap[cap, "loop_pf"]["tpot_mean_ms"]
                    headline[f"{tag}_tpot_{btag}_gen_pf_ms"] = gmap[cap, "general_pf"]["tpot_mean_ms"]
                    headline[f"{tag}_tpot_{btag}_static_ms"] = gmap[cap, "static_hot"]["tpot_mean_ms"]
                    headline[f"{tag}_pred_cpu_{btag}_ms"] = gmap[cap, "loop_pf"]["predictor_cpu_ms_per_tok"]
        write_json(out_dir / "headline_metrics_b2_fast.json", headline)
        prog.step(1, **headline)
        prog.flush()


if __name__ == "__main__":
    main()
