"""B2 Part 3: Real GPU Pinned-Host Expert Offload + CUDA Copy-Stream Async Prefetch Benchmark.

Runs directly on the verified B1 `LoopedIncrementalEngine` on 1 strictly idle GPU.
Every offloaded policy packs all 700 routed experts into pinned CPU RAM (21.375 MiB contiguous
flat BF16 buffer per expert) and frees original GPU routed-expert weights so ONLY `capacity`
expert slots physically exist on GPU (resident + in-flight prefetch buffers <= capacity).

Compares 6 systems on the exact same held-out prompts (tracing OFF during main timing):
  1. full_resident  (all 700 experts on GPU, speed & greedy-token upper bound)
  2. static_hot     (fit-only static hot pinning, Prefetch OFF)
  3. general_nopf   (matched stage-blind general_hazard eviction, Prefetch OFF)
  4. loop_nopf      (multi-loop hazard eviction, Prefetch OFF)
  5. general_pf     (general_hazard eviction + stage-blind async prefetch ON)
  6. loop_pf        (loop_hazard eviction + stage-conditioned combined_loop async prefetch ON)
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import gc
from heapq import heapify, heappop, heappush
import json
from pathlib import Path
import statistics
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ifm_a1_collect import download_ifm_artifact, select_idle_gpu
from .ifm_a2_predict import A2Models, _build_schedule_info, read_ifm_trace
from .ifm_b1_kv_verify import LoopedIncrementalEngine
from .ifm_prompts import format_ifm_prompt
from .ifm_runtime import bootstrap_xllm, load_ifm_model
from .trace import read_json, read_jsonl, write_csv, write_json


GPU_POLICIES = (
    "static_hot",
    "general_nopf",
    "loop_nopf",
    "general_pf",
    "loop_pf",
)


class PinnedExpertStore:
    """Packs all 700 routed experts into contiguous pinned host buffers and frees GPU copies."""

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

        self.host_experts: dict[tuple[str, int], torch.Tensor] = {}
        with torch.no_grad():
            for layer_id, layer in enumerate(model.layers):
                pool = f"layer:{layer_id:02d}"
                exp = layer.moe.experts
                w1 = exp.weight1.detach().view(self.n_experts, self.inter_dim, self.model_dim).cpu()
                w2 = exp.weight2.detach().view(self.n_experts, self.model_dim, self.inter_dim).cpu()
                w3 = exp.weight3.detach().view(self.n_experts, self.inter_dim, self.model_dim).cpu()
                for e in range(self.n_experts):
                    flat = torch.empty(self.expert_numel, dtype=torch.bfloat16, pin_memory=True)
                    o1 = self.w1_numel
                    o2 = o1 + self.w2_numel
                    flat[:o1].copy_(w1[e].reshape(-1))
                    flat[o1:o2].copy_(w2[e].reshape(-1))
                    flat[o2:].copy_(w3[e].reshape(-1))
                    self.host_experts[pool, e] = flat
                # Release GPU routed expert weights so GPU truly holds only cached slots!
                exp.weight1 = nn.Parameter(torch.empty(0, device=device, dtype=torch.bfloat16), requires_grad=False)
                exp.weight2 = nn.Parameter(torch.empty(0, device=device, dtype=torch.bfloat16), requires_grad=False)
                exp.weight3 = nn.Parameter(torch.empty(0, device=device, dtype=torch.bfloat16), requires_grad=False)

        gc.collect()
        torch.cuda.empty_cache()

    def expert_views(self, slot_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        o1 = self.w1_numel
        o2 = o1 + self.w2_numel
        w1 = slot_tensor[:o1].view(self.inter_dim, self.model_dim)
        w2 = slot_tensor[o1:o2].view(self.model_dim, self.inter_dim)
        w3 = slot_tensor[o2:].view(self.inter_dim, self.model_dim)
        return w1, w2, w3


class RealGPUOffloadManager:
    """Strict peak-budget GPU expert cache with dedicated CUDA copy stream & async prefetch."""

    def __init__(
        self,
        store: PinnedExpertStore,
        meta: dict[str, Any],
        sched: dict[str, Any],
        a2: A2Models,
        hot_rank: list[tuple[str, int]],
        *,
        policy: str,
        capacity: int,
        gamma: float = 0.75,
        freq_weight: float = 0.08,
        prefetch_k: int = 2,
        prefetch_thresh: float = 0.25,
        profile_events: bool = False,
    ):
        self.store = store
        self.device = store.device
        self.meta = meta
        self.sched = sched
        self.a2 = a2
        self.policy = policy
        self.capacity = capacity
        self.gamma = gamma
        self.freq_weight = freq_weight
        self.prefetch_k = prefetch_k
        self.prefetch_thresh = prefetch_thresh
        self.profile_events = profile_events

        self.use_pf = policy.endswith("_pf")
        self.base_pol = policy[:-3] if policy.endswith("_pf") else (policy[:-5] if policy.endswith("_nopf") else policy)
        self.order = sched["order"]
        self.n_order = len(self.order)
        self.pool_positions = sched["pool_positions"]
        self.visits_per_tok = sched["pool_visits_per_token"]
        self.pools = list(meta["pools"].keys())
        self.num_experts = store.n_experts

        # Precompute future cycle offsets for multi-loop hazard
        self.future_cycle: list[list[tuple[int, int | None]]] = []
        for idx, (pool, _) in enumerate(self.order):
            pos_list = self.pool_positions[pool]
            upcoming = [(p - idx, self.order[p][1]) for p in pos_list if p > idx]
            upcoming.extend((p + self.n_order - idx, self.order[p][1]) for p in pos_list if p <= idx)
            self.future_cycle.append(upcoming)
        self.next_is_boundary = [(idx + self.future_cycle[idx][0][0]) >= self.n_order for idx in range(self.n_order)]
        self.stage_priors = {
            (p, lp): a2._stage_prior(p, lp)
            for p in self.pools
            for lp in {l for pp, l in self.order if pp == p}
        }
        self.pool_priors = {p: a2._pool_prior(p) for p in self.pools}

        max_pin = capacity if capacity >= len(self.pools) * self.num_experts else max(0, capacity - 1)
        self.static_keys = set(hot_rank[:max_pin]) if self.base_pol == "static_hot" else set()

        # Fixed physical GPU slot pool: strictly `capacity` slots (resident + in-flight + demand staging!)
        self.gpu_pool = torch.empty(
            (capacity, store.expert_numel),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.slot_events: list[torch.cuda.Event] = [
            torch.cuda.Event(enable_timing=False) for _ in range(capacity)
        ]
        self.use_events: list[torch.cuda.Event] = [
            torch.cuda.Event(enable_timing=False) for _ in range(capacity)
        ]

        # Pre-populate static_hot keys if static_hot (otherwise cold start per request)
        self.reset_request()

    def reset_request(self) -> None:
        self.free_slots: deque[int] = deque(range(self.capacity))
        # key (pool, e) -> slot_idx
        self.key_to_slot: dict[tuple[str, int], int] = {}
        self.residents: dict[str, dict[int, int]] = {p: {} for p in self.pools}
        self.heaps: dict[str, list[tuple[float, int, int]]] = {p: [] for p in self.pools}
        self.priority: dict[str, list[float]] = {p: [0.0] * self.num_experts for p in self.pools}
        self.pool_recent: dict[str, deque[list[int]]] = defaultdict(deque)
        self.pool_stage_last: dict[tuple[str, int | None], list[int]] = {}
        self.pool_ema: dict[str, list[float]] = defaultdict(lambda: [0.0] * self.num_experts)
        self.pool_seen_visits: Counter = Counter()
        self.frequency: Counter = Counter()

        # Pending prefetches: key -> slot_idx
        self.in_flight: dict[tuple[str, int], int] = {}
        self.protected: set[tuple[str, int]] = set()

        # Stats (measured separately for prefill vs decode)
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

    def _val(self, pool: str, expert: int) -> float:
        if self.base_pol == "static_hot":
            return 1.0 if (pool, expert) in self.static_keys else 0.0
        h_score = self.priority[pool][expert]
        if self.freq_weight > 0.0:
            v_seen = self.pool_seen_visits[pool]
            rate = self.frequency[pool, expert] / (v_seen + 8.0)
            density = self.visits_per_tok[pool] / float(self.n_order)
            h_score += self.freq_weight * rate * density
        return h_score

    def _refresh(self, pool: str) -> None:
        self.heaps[pool] = [(self._val(pool, e), last, e) for e, last in self.residents[pool].items()]
        heapify(self.heaps[pool])

    def _acquire_slot(self) -> int:
        if self.free_slots:
            return self.free_slots.popleft()
        best: tuple[float, int, str, int] | None = None
        for p in self.pools:
            heap = self.heaps[p]
            skipped: list[tuple[float, int, int]] = []
            cand = None
            while heap:
                sc, last, ex = heap[0]
                if self.residents[p].get(ex) != last:
                    heappop(heap)
                    continue
                if (p, ex) in self.protected:
                    skipped.append(heappop(heap))
                    continue
                cand = (sc, last, p, ex)
                break
            for item in skipped:
                heappush(heap, item)
            if cand is not None and (best is None or cand < best):
                best = cand
        if best is None:
            raise RuntimeError("No evictable slot available in GPU cache")
        _, _, vp, ve = best
        del self.residents[vp][ve]
        slot = self.key_to_slot.pop((vp, ve))
        return slot

    def _start_async_copy(self, key: tuple[str, int], slot: int) -> None:
        host_t = self.store.host_experts[key]
        with torch.cuda.stream(self.copy_stream):
            # Ensure default stream has finished any prior FFN read on this slot before overwriting
            self.copy_stream.wait_event(self.use_events[slot])
            self.gpu_pool[slot].copy_(host_t, non_blocking=True)
            self.slot_events[slot].record(self.copy_stream)

    def _update_causal_hazard_and_prefetch(
        self,
        pool: str,
        v_idx: int,
        tgt_ids: list[int],
        cand_ids: list[int],
        cand_scores: list[float],
    ) -> None:
        t0 = time.perf_counter()
        a2 = self.a2
        loop = self.order[v_idx][1]

        self.pool_stage_last[pool, loop] = tgt_ids
        hist = self.pool_recent[pool]
        hist.appendleft(tgt_ids)
        h_span = self.visits_per_tok[pool]
        if len(hist) > h_span:
            hist.pop()

        pema = self.pool_ema[pool]
        alpha = a2.alpha
        decay = 1.0 - alpha
        for e in range(self.num_experts):
            pema[e] *= decay
        for cid, cs in zip(cand_ids, cand_scores):
            pema[cid] += alpha * cs

        self.pool_seen_visits[pool] += 1

        # 1. Update eviction hazard priority for current pool
        if self.base_pol in ("loop_hazard", "loop", "general_hazard", "general"):
            is_loop = self.base_pol.startswith("loop")
            pool_prior = self.pool_priors[pool]
            p_gen_lag1 = a2._trans_prob(
                a2.gen_lag1_trans, a2.gen_lag1_totals, [(pool, se) for se in hist[0]], pool_prior
            )
            if len(hist) >= h_span:
                p_gen_lagH = a2._trans_prob(
                    a2.gen_lagH_trans, a2.gen_lagH_totals, [(pool, se) for se in hist[h_span - 1]], pool_prior
                )
            else:
                p_gen_lagH = None

            cycle = self.future_cycle[v_idx]
            d1, next_loop = cycle[0]
            w1 = 1.0 / ((1.0 + d1) ** self.gamma)

            if is_loop:
                next_sp = self.stage_priors[pool, next_loop]
                boundary = self.next_is_boundary[v_idx]
                h_comps = []
                prev_st = self.pool_stage_last.get((pool, next_loop))
                if prev_st is not None:
                    tok_pr = p_gen_lagH if p_gen_lagH is not None else next_sp
                    h_comps.append(
                        a2._trans_prob(a2.prev_tok_trans, a2.prev_tok_totals, [(pool, next_loop, se) for se in prev_st], tok_pr)
                    )
                elif p_gen_lagH is not None:
                    h_comps.append(p_gen_lagH)
                if not boundary and loop is not None and next_loop is not None:
                    h_comps.append(
                        a2._trans_prob(a2.prev_loop_trans, a2.prev_loop_totals, [(pool, loop, next_loop, se) for se in tgt_ids], p_gen_lag1)
                    )
                else:
                    h_comps.append(p_gen_lag1)
                p_next = a2._combine_probs(next_sp, h_comps)
                haz = [w1 * (0.85 * p_next[e] + 0.15 * min(1.0, pema[e])) for e in range(self.num_experts)]
                for dm, lm in cycle[1:]:
                    wm = 1.0 / ((1.0 + dm) ** self.gamma)
                    sp_m = self.stage_priors[pool, lm]
                    pm_prev = self.pool_stage_last.get((pool, lm))
                    pm = (
                        a2._trans_prob(a2.prev_tok_trans, a2.prev_tok_totals, [(pool, lm, se) for se in pm_prev], sp_m)
                        if pm_prev is not None
                        else sp_m
                    )
                    for e in range(self.num_experts):
                        haz[e] += wm * pm[e]
            else:
                p_next = a2._combine_probs(pool_prior, [p_gen_lag1, p_gen_lagH]) if p_gen_lagH is not None else p_gen_lag1
                haz = [w1 * (0.85 * p_next[e] + 0.15 * min(1.0, pema[e])) for e in range(self.num_experts)]
                for dm, _ in cycle[1:]:
                    wm = 1.0 / ((1.0 + dm) ** self.gamma)
                    for e in range(self.num_experts):
                        haz[e] += wm * pool_prior[e]

            self.priority[pool] = haz
            self._refresh(pool)

        # 2. Issue async prefetch for NEXT visit (v_idx + 1) during decode!
        if self.use_pf and self.in_decode:
            nv_idx = (v_idx + 1) % self.n_order
            npool, nloop = self.order[nv_idx]
            is_loop = self.base_pol.startswith("loop")
            nhist = self.pool_recent[npool]
            nh_span = self.visits_per_tok[npool]

            if is_loop:
                nstage_prior = self.stage_priors[npool, nloop]
                comps = [
                    a2._trans_prob(a2.prev_layer_trans, a2.prev_layer_totals, [(pool, npool, nloop, se) for se in tgt_ids], nstage_prior)
                ]
                prev_nst = self.pool_stage_last.get((npool, nloop))
                if prev_nst is not None:
                    comps.append(
                        a2._trans_prob(a2.prev_tok_trans, a2.prev_tok_totals, [(npool, nloop, se) for se in prev_nst], nstage_prior)
                    )
                if nloop is not None and nloop > 0 and len(nhist) >= 1:
                    comps.append(
                        a2._trans_prob(a2.prev_loop_trans, a2.prev_loop_totals, [(npool, nloop - 1, nloop, se) for se in nhist[0]], nstage_prior)
                    )
                probs = a2._combine_probs(nstage_prior, comps)
            else:
                npool_prior = self.pool_priors[npool]
                comps = [
                    a2._trans_prob(a2.gen_prev_pool_trans, a2.gen_prev_pool_totals, [(pool, npool, se) for se in tgt_ids], npool_prior)
                ]
                if len(nhist) >= 1:
                    comps.append(a2._trans_prob(a2.gen_lag1_trans, a2.gen_lag1_totals, [(npool, se) for se in nhist[0]], npool_prior))
                if len(nhist) >= nh_span:
                    comps.append(a2._trans_prob(a2.gen_lagH_trans, a2.gen_lagH_totals, [(npool, se) for se in nhist[nh_span - 1]], npool_prior))
                probs = a2._combine_probs(npool_prior, comps)

            missing = [
                (probs[e], e)
                for e in range(self.num_experts)
                if e not in self.residents[npool] and (npool, e) not in self.in_flight
            ]
            missing.sort(reverse=True)
            for prob, e in missing[: self.prefetch_k]:
                if prob < self.prefetch_thresh:
                    break
                slot = self._acquire_slot()
                self.key_to_slot[(npool, e)] = slot
                self.in_flight[(npool, e)] = slot
                self.protected.add((npool, e))
                self._start_async_copy((npool, e), slot)
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
        """Exact MoE layer execution using pinned host offload + GPU cache + copy_stream overlap."""
        moe = layer.moe
        bsz, slen, dim = x_in.shape
        residual = x_in
        x = moe.norm(x_in)

        # Shared experts (always GPU-resident)
        if moe.fc1 is not None:
            sh = F.silu(moe.fc1(x)) * moe.fc3(x)
            y1 = moe.fc2(sh)
        else:
            y1 = None

        # Router (always GPU-resident)
        routing_scores, routing_indices, _tpe, aux_loss = moe.router(x, load_balancing_type)
        pool = f"layer:{layer_id:02d}"
        x_2d = x.view(bsz * slen, dim)
        default_stream = torch.cuda.current_stream(self.device)

        if slen == 1:
            # Fast decode path (B=1, L=1)
            tgt_ids = routing_indices[0, 0].tolist()
            tgt_w = routing_scores[0, 0]  # [K]
            demand_set = set(tgt_ids)
            self.protected = {(pool, e) for e in demand_set}

            # 1. Settle any in-flight prefetches from previous visit
            if self.in_flight:
                items = list(self.in_flight.items())
                self.in_flight.clear()
                for key, slot in items:
                    pf_pool, pf_e = key
                    is_useful = (pf_pool == pool) and (pf_e in demand_set)
                    if self.in_decode:
                        if is_useful:
                            if self.slot_events[slot].query():
                                self.prefetch_timely += 1
                            else:
                                self.prefetch_late += 1
                        else:
                            self.prefetch_wasted += 1
                    self.residents[pf_pool][pf_e] = self.accesses
                    heappush(self.heaps[pf_pool], (self._val(pf_pool, pf_e), self.accesses, pf_e))

            # 2. Issue synchronous H2D for any missing demand experts first (pipelined on copy_stream)
            slots_for_k: list[tuple[int, bool]] = []
            for e in tgt_ids:
                key = (pool, e)
                if self.in_decode:
                    self.accesses += 1
                self.frequency[key] += 1
                if e in self.residents[pool]:
                    slot = self.key_to_slot[key]
                    self.residents[pool][e] = self.accesses
                    heappush(self.heaps[pool], (self._val(pool, e), self.accesses, e))
                    slots_for_k.append((slot, False))
                else:
                    if self.in_decode:
                        self.demand_misses += 1
                        self.demand_h2d_bytes += self.store.bytes_per_expert
                    slot = self._acquire_slot()
                    self.key_to_slot[key] = slot
                    self._start_async_copy(key, slot)
                    admit = (self.base_pol != "static_hot") or (key in self.static_keys)
                    if admit:
                        self.residents[pool][e] = self.accesses
                        heappush(self.heaps[pool], (self._val(pool, e), self.accesses, e))
                    else:
                        # Staging only: release slot back right after compute
                        slots_for_k.append((slot, True))
                        continue
                    slots_for_k.append((slot, False))

            # 3. Compute routed experts (waiting on slot copy event if still in flight)
            y2 = torch.zeros_like(x_2d)
            for k_i, (slot, release_after) in enumerate(slots_for_k):
                ev = self.slot_events[slot]
                if self.profile_events and self.in_decode and not ev.query():
                    t_w0 = time.perf_counter()
                    ev.synchronize()
                    self.exposed_wait_ms += (time.perf_counter() - t_w0) * 1000.0
                else:
                    default_stream.wait_event(ev)
                w1, w2, w3 = self.store.expert_views(self.gpu_pool[slot])
                ye = F.linear(F.silu(F.linear(x_2d, w1)) * F.linear(x_2d, w3), w2)
                self.use_events[slot].record(default_stream)
                y2 = y2 + ye * tgt_w[k_i].to(ye.dtype)
                if release_after:
                    self.key_to_slot.pop((pool, tgt_ids[k_i]), None)
                    self.free_slots.append(slot)

            # 4. Update causal predictors & kick off async prefetch for NEXT visit (overlaps with residual + next attn!)
            if self.base_pol != "static_hot":
                # Use top-k routing scores as lightweight EMA update (zero extra GEMM!)
                self._update_causal_hazard_and_prefetch(
                    pool, v_idx, tgt_ids, tgt_ids, [float(v) for v in tgt_w.tolist()]
                )
            y2 = y2.view(bsz, slen, dim)
        else:
            # Prefill path (L > 1): group prompt tokens by expert and load each active expert through the same GPU pool
            idx_flat = routing_indices[0]  # [L, K]
            w_flat = routing_scores[0]  # [L, K]
            y2 = torch.zeros_like(x_2d)
            active_exps = torch.unique(idx_flat).tolist()
            for e in active_exps:
                key = (pool, e)
                self.protected = {key}
                self.frequency[key] += 1
                release_after = False
                if e in self.residents[pool]:
                    slot = self.key_to_slot[key]
                else:
                    slot = self._acquire_slot()
                    self.key_to_slot[key] = slot
                    self._start_async_copy(key, slot)
                    admit = (self.base_pol != "static_hot") or (key in self.static_keys)
                    if admit:
                        self.residents[pool][e] = self.accesses
                        heappush(self.heaps[pool], (self._val(pool, e), self.accesses, e))
                    else:
                        release_after = True
                default_stream.wait_event(self.slot_events[slot])
                w1, w2, w3 = self.store.expert_views(self.gpu_pool[slot])
                mask = idx_flat == e  # [L, K]
                tok_pos, k_pos = torch.nonzero(mask, as_tuple=True)
                xe = x_2d.index_select(0, tok_pos)
                ye = F.linear(F.silu(F.linear(xe, w1)) * F.linear(xe, w3), w2)
                self.use_events[slot].record(default_stream)
                we = w_flat[tok_pos, k_pos].unsqueeze(-1).to(ye.dtype)
                y2.index_add_(0, tok_pos, ye * we)
                if release_after:
                    self.key_to_slot.pop(key, None)
                    self.free_slots.append(slot)
            # Seed causal history from last prefill token so decode step 1 has immediate history
            if self.base_pol != "static_hot":
                last_ids = idx_flat[-1].tolist()
                last_w = [float(v) for v in w_flat[-1].tolist()]
                self._update_causal_hazard_and_prefetch(pool, v_idx, last_ids, last_ids, last_w)
            y2 = y2.view(bsz, slen, dim)

        y = y1 + y2 if y1 is not None else y2
        out = moe.residual(y, residual, c=y)
        return out, aux_loss


@torch.inference_mode()
def run_real_gpu_benchmark_for_model(
    model_name: str,
    model_spec: dict[str, Any],
    model_dir: Path,
    trace_path: Path,
    prompts_all: list[dict[str, Any]],
    cfg: dict[str, Any],
    *,
    device: torch.device,
    budgets: list[int],
    n_eval_prompts: int = 18,
    decode_tokens: int = 48,
    tuned_params: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run real GPU full_resident + 5 offloaded systems across budgets on 1 idle GPU."""
    # Fit A2 tables strictly on fit_request_ids from formal trace
    meta, trace_reqs = read_ifm_trace(trace_path)
    sched = _build_schedule_info(meta)
    fit_set = set(meta["splits"]["fit"])
    eval_set = set(meta["splits"]["eval"])
    fit_reqs = [r for r in trace_reqs if r["request_id"] in fit_set]

    a2 = A2Models(meta, sched, cfg["a2"])
    a2.fit(fit_reqs, shuffle_seed=int(cfg["a2"]["shuffle_seed"]))

    # Fit-only static hot ranking
    order = sched["order"]
    hot_counts: Counter = Counter()
    for req in fit_reqs:
        for step in req["steps"]:
            for v_idx, visit in enumerate(step["visits"]):
                pool, _ = order[v_idx]
                for e in visit[0]:
                    hot_counts[pool, e] += 1
    all_keys = [(p, e) for p, d in meta["pools"].items() for e in range(int(d["experts"]))]
    hot_rank = sorted(all_keys, key=lambda k: (-hot_counts[k], k[0], k[1]))

    # Select balanced held-out eval prompts (text / math / code)
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

    gamma = float((tuned_params or {}).get("gamma", 0.75))
    freq_weight = float((tuned_params or {}).get("freq_weight", 0.08))
    prefetch_thresh = float((tuned_params or {}).get("prefetch_thresh", 0.25))

    # 1. Load model and measure full_resident upper bound + reference greedy tokens
    torch.cuda.reset_peak_memory_stats(device)
    model, tokenizer, _ = load_ifm_model(model_dir, device=str(device), attention_backend="sdpa")
    engine = LoopedIncrementalEngine(model, model_spec)

    encoded_prompts = [
        (p, tokenizer.encode(format_ifm_prompt(p), bos=True, eos=False)[-512:])
        for p in eval_prompts
    ]

    # Warmup 1 short run
    engine.reset()
    _ = engine.forward_step(torch.tensor([encoded_prompts[0][1][:16]], device=device, dtype=torch.long))
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    ref_tokens: dict[str, list[int]] = {}
    req_rows: list[dict[str, Any]] = []

    for rec, pids in encoded_prompts:
        engine.reset()
        torch.cuda.synchronize(device)
        t_start = time.perf_counter()
        logits = engine.forward_step(torch.tensor([pids], device=device, dtype=torch.long))
        nxt = int(logits[0, -1].argmax().item())
        torch.cuda.synchronize(device)
        t_first = time.perf_counter()
        gen = [nxt]
        for _ in range(decode_tokens - 1):
            logits = engine.forward_step(torch.tensor([[nxt]], device=device, dtype=torch.long))
            nxt = int(logits[0, -1].argmax().item())
            gen.append(nxt)
        torch.cuda.synchronize(device)
        t_last = time.perf_counter()
        ref_tokens[rec["id"]] = gen
        ttft_ms = (t_first - t_start) * 1000.0
        tpot_ms = ((t_last - t_first) * 1000.0) / max(1, len(gen) - 1)
        req_rows.append(
            {
                "model": model_name,
                "budget_experts": 700,
                "budget_gib": 14.6118,
                "policy": "full_resident",
                "request_id": rec["id"],
                "category": rec["category"],
                "prompt_tokens": len(pids),
                "decode_tokens": len(gen),
                "exact_token_match": 1,
                "ttft_ms": round(ttft_ms, 3),
                "tpot_ms": round(tpot_ms, 3),
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

    # 2. Pack all 700 routed experts into pinned host RAM & free original GPU routed weights!
    store = PinnedExpertStore(model, device)
    bpe = store.bytes_per_expert

    for cap in budgets:
        budget_gib = round(cap * bpe / (1024**3), 4)
        for pol in GPU_POLICIES:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            mgr = RealGPUOffloadManager(
                store,
                meta,
                sched,
                a2,
                hot_rank,
                policy=pol,
                capacity=cap,
                gamma=gamma,
                freq_weight=freq_weight,
                prefetch_k=2,
                prefetch_thresh=prefetch_thresh,
                profile_events=False,
            )

            # Main clean timing pass (tracing & event-sync profiling OFF!)
            for p_idx, (rec, pids) in enumerate(encoded_prompts):
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
                ttft_ms = (t_first - t_start) * 1000.0
                tpot_ms = ((t_last - t_first) * 1000.0) / n_dec
                exact_match = int(gen == ref_tokens[rec["id"]])
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
                        "exact_token_match": exact_match,
                        "ttft_ms": round(ttft_ms, 3),
                        "tpot_ms": round(tpot_ms, 3),
                        "demand_hit_rate": round(1.0 - mgr.demand_misses / max(1, mgr.accesses), 6),
                        "demand_h2d_mib_per_tok": round((mgr.demand_h2d_bytes / (1024**2)) / n_dec, 3),
                        "prefetch_h2d_mib_per_tok": round((mgr.prefetch_h2d_bytes / (1024**2)) / n_dec, 3),
                        "total_h2d_mib_per_tok": round(((mgr.demand_h2d_bytes + mgr.prefetch_h2d_bytes) / (1024**2)) / n_dec, 3),
                        "timely_pf": mgr.prefetch_timely,
                        "late_pf": mgr.prefetch_late,
                        "wasted_pf": mgr.prefetch_wasted,
                        "exposed_wait_ms_per_tok": 0.0,  # filled by short profiling pass below
                        "predictor_cpu_ms_per_tok": round(mgr.predictor_cpu_ms / n_dec, 3),
                        "peak_allocated_gib": round(torch.cuda.max_memory_allocated(device) / (1024**3), 3),
                        "peak_reserved_gib": round(torch.cuda.max_memory_reserved(device) / (1024**3), 3),
                    }
                )

            # Short profiling pass (first 3 prompts) to measure exposed CUDA event wait separately without polluting main TPOT
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
            mean_prof_wait = round(statistics.mean(prof_waits), 3) if prof_waits else 0.0
            for r in req_rows:
                if r["model"] == model_name and r["budget_experts"] == cap and r["policy"] == pol:
                    r["exposed_wait_ms_per_tok"] = mean_prof_wait

            del mgr
            gc.collect()
            torch.cuda.empty_cache()

    # Aggregate summary rows
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

    del store, engine, model
    gc.collect()
    torch.cuda.empty_cache()
    return summary_rows, req_rows
