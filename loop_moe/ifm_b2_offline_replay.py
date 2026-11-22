"""B2 Part 2: 96-request CPU co-simulation of Multi-Loop Hazard Eviction & 2x2 Async Prefetch.

Runs strictly causally on the already-collected formal 192-request IFM traces (64 fit / 32 val / 96 eval).
Key design goals:
1. Resolve the A3 eviction horizon mismatch with `loop_hazard` (multi-loop distance-discounted
   expected hits across the full token cycle + online LFU rate) vs a strictly matched stage-blind
   `general_hazard` baseline.
2. Run the B2 2x2 factorial under STRICT SAME PEAK BYTE BUDGET (resident + in-flight prefetch
   buffers + demand staging <= C):
     (general_hazard, loop_hazard) x (prefetch OFF, prefetch ON)
   plus `static_hot` and offline `belady`.
3. Separate H2D bytes, exposed GPU stall time (calibrated by real measured hardware H2D &
   per-visit compute timings), timely/late/wasted prefetch bytes, and paired 95% bootstrap CIs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from heapq import heapify, heappop, heappush
import json
import math
import os
from pathlib import Path
import random
import statistics
from typing import Any

from .ifm_a2_predict import A2Models, _build_schedule_info, read_ifm_trace
from .trace import read_json, write_csv, write_json


B2_POLICIES = (
    "static_hot_nopf",
    "lru_nopf",
    "lfu_nopf",
    "general_hazard_nopf",
    "loop_hazard_nopf",
    "general_hazard_pf",
    "loop_hazard_pf",
    "belady_nopf",
)


@dataclass
class B2SimOutcome:
    accesses: int
    demand_misses: int
    prefetch_issued: int
    prefetch_timely: int
    prefetch_late: int
    prefetch_wasted: int
    demand_h2d_bytes: int
    prefetch_h2d_bytes: int
    exposed_stall_ms: float
    total_step_ms: float
    hit_bits: bytearray
    peak_bytes: int


def _prepare_b2_events(
    req: dict[str, Any],
    meta: dict[str, Any],
    sched: dict[str, Any],
    a2: A2Models,
    *,
    gamma: float = 0.75,
) -> list[dict[str, Any]]:
    """Build strictly causal per-visit stream containing:
    - Post-visit eviction scores for current pool (multi-loop hazard vs matched stage-blind general)
    - Pre-visit prefetch candidate probabilities for the NEXT visit (v_idx + 1), computed right
      after current visit v_idx executes (so prev_phys_layer IS causally available for prefetch!).
    """
    order = sched["order"]
    n_order = len(order)
    pool_positions = sched["pool_positions"]
    visits_per_tok = sched["pool_visits_per_token"]
    num_experts = a2.num_experts
    alpha = a2.alpha

    # Precompute for each visit_idx: list of upcoming (offset, loop_idx) for the SAME pool over 1 full token cycle
    future_cycle: list[list[tuple[int, int | None]]] = []
    for idx, (pool, _) in enumerate(order):
        pos_list = pool_positions[pool]
        # Visits in current token after idx, then visits in next token up to idx
        upcoming: list[tuple[int, int | None]] = []
        for p in pos_list:
            if p > idx:
                upcoming.append((p - idx, order[p][1]))
        for p in pos_list:
            if p <= idx:
                upcoming.append((p + n_order - idx, order[p][1]))
        future_cycle.append(upcoming)

    next_delta = [future_cycle[idx][0][0] for idx in range(n_order)]
    next_is_boundary = [(idx + next_delta[idx]) >= n_order for idx in range(n_order)]

    # Precompute stage & pool priors
    stage_priors = {
        (p, lp): a2._stage_prior(p, lp)
        for p in meta["pools"]
        for lp in {l for pp, l in order if pp == p}
    }
    pool_priors = {p: a2._pool_prior(p) for p in meta["pools"]}

    pool_recent: dict[str, deque[list[int]]] = defaultdict(deque)
    pool_stage_last: dict[tuple[str, int | None], list[int]] = {}
    pool_ema: dict[str, list[float]] = defaultdict(lambda: [0.0] * num_experts)

    events: list[dict[str, Any]] = []
    global_idx = 0
    steps = req["steps"]
    n_steps = len(steps)

    for s_idx, step in enumerate(steps):
        visits = step["visits"]
        for v_idx, visit in enumerate(visits):
            pool, loop = order[v_idx]
            tgt_ids = visit[0]
            tgt_weights = visit[1]
            cand_ids = visit[3]
            cand_scores = visit[4]
            logit_margin = float(visit[6]) if len(visit) > 6 and visit[6] is not None else 0.0

            # Update causal state upon observing current visit
            pool_stage_last[pool, loop] = tgt_ids
            hist = pool_recent[pool]
            hist.appendleft(tgt_ids)
            h_span = visits_per_tok[pool]
            if len(hist) > h_span:
                hist.pop()
            pema = pool_ema[pool]
            decay = 1.0 - alpha
            for e in range(num_experts):
                pema[e] *= decay
            for cid, cscore in zip(cand_ids, cand_scores):
                pema[cid] += alpha * float(cscore)

            # ---- 1. Post-visit eviction hazard scores for current pool ----
            pool_prior = pool_priors[pool]
            p_gen_lag1 = a2._trans_prob(
                a2.gen_lag1_trans,
                a2.gen_lag1_totals,
                [(pool, se) for se in hist[0]],
                pool_prior,
            )
            if len(hist) >= h_span:
                p_gen_lagH = a2._trans_prob(
                    a2.gen_lagH_trans,
                    a2.gen_lagH_totals,
                    [(pool, se) for se in hist[h_span - 1]],
                    pool_prior,
                )
                p_gen_next = a2._combine_probs(pool_prior, [p_gen_lag1, p_gen_lagH])
            else:
                p_gen_lagH = None
                p_gen_next = p_gen_lag1

            cycle = future_cycle[v_idx]
            d1, next_loop = cycle[0]
            boundary = next_is_boundary[v_idx]
            next_stage_prior = stage_priors[pool, next_loop]

            # Immediate next visit (m=1) hierarchical stage-conditioned probability
            hier_comps = []
            prev_stage_ids = pool_stage_last.get((pool, next_loop))
            if prev_stage_ids is not None:
                tok_prior = p_gen_lagH if p_gen_lagH is not None else next_stage_prior
                hier_comps.append(
                    a2._trans_prob(
                        a2.prev_tok_trans,
                        a2.prev_tok_totals,
                        [(pool, next_loop, se) for se in prev_stage_ids],
                        tok_prior,
                    )
                )
            elif p_gen_lagH is not None:
                hier_comps.append(p_gen_lagH)

            if not boundary and loop is not None and next_loop is not None:
                hier_comps.append(
                    a2._trans_prob(
                        a2.prev_loop_trans,
                        a2.prev_loop_totals,
                        [(pool, loop, next_loop, se) for se in tgt_ids],
                        p_gen_lag1,
                    )
                )
            else:
                hier_comps.append(p_gen_lag1)

            p_loop_next = a2._combine_probs(next_stage_prior, hier_comps)

            # Multi-loop cycle hazard: sum_{m=1..H_p} P(e in S_{v_m}) / (1 + d_m)^gamma
            # For loop_hazard: m=1 uses p_loop_next, m>=2 uses stage-conditioned prev-token / stage_prior.
            # For general_hazard (matched stage-blind): m=1 uses p_gen_next, m>=2 uses pool_prior.
            w1 = 1.0 / ((1.0 + d1) ** gamma)
            haz_loop = [w1 * (0.85 * p_loop_next[e] + 0.15 * min(1.0, pema[e])) for e in range(num_experts)]
            haz_gen = [w1 * (0.85 * p_gen_next[e] + 0.15 * min(1.0, pema[e])) for e in range(num_experts)]

            for dm, lm in cycle[1:]:
                wm = 1.0 / ((1.0 + dm) ** gamma)
                sp_m = stage_priors[pool, lm]
                prev_m_ids = pool_stage_last.get((pool, lm))
                if prev_m_ids is not None:
                    pm_loop = a2._trans_prob(
                        a2.prev_tok_trans,
                        a2.prev_tok_totals,
                        [(pool, lm, se) for se in prev_m_ids],
                        sp_m,
                    )
                else:
                    pm_loop = sp_m
                for e in range(num_experts):
                    haz_loop[e] += wm * pm_loop[e]
                    haz_gen[e] += wm * pool_prior[e]

            # ---- 2. Causal Prefetch Scores for the IMMEDIATELY FOLLOWING visit (next_global = global_idx + 1) ----
            # Computed right here after current visit (pool, loop, tgt_ids) completes -> prev_phys_layer IS known!
            pf_target_pool: str | None = None
            pf_loop_probs: list[float] | None = None
            pf_gen_probs: list[float] | None = None

            has_next = (v_idx + 1 < n_order) or (s_idx + 1 < n_steps)
            if has_next:
                nv_idx = (v_idx + 1) % n_order
                npool, nloop = order[nv_idx]
                pf_target_pool = npool
                npool_prior = pool_priors[npool]
                nstage_prior = stage_priors[npool, nloop]
                nhist = pool_recent[npool]
                nh_span = visits_per_tok[npool]

                # (a) Stage-conditioned full combined_loop predictor for (npool, nloop)
                # Components: prev_phys_layer (pool -> npool at nloop), prev_loop_trans (if nloop > 0),
                # prev_tok_same_stage (if seen before).
                pf_l_comps = [
                    a2._trans_prob(
                        a2.prev_layer_trans,
                        a2.prev_layer_totals,
                        [(pool, npool, nloop, se) for se in tgt_ids],
                        nstage_prior,
                    )
                ]
                prev_nstage_ids = pool_stage_last.get((npool, nloop))
                if prev_nstage_ids is not None:
                    pf_l_comps.append(
                        a2._trans_prob(
                            a2.prev_tok_trans,
                            a2.prev_tok_totals,
                            [(npool, nloop, se) for se in prev_nstage_ids],
                            nstage_prior,
                        )
                    )
                if nloop is not None and nloop > 0 and len(nhist) >= 1:
                    pf_l_comps.append(
                        a2._trans_prob(
                            a2.prev_loop_trans,
                            a2.prev_loop_totals,
                            [(npool, nloop - 1, nloop, se) for se in nhist[0]],
                            nstage_prior,
                        )
                    )
                pf_loop_probs = a2._combine_probs(nstage_prior, pf_l_comps)

                # (b) Matched stage-blind general_history predictor for npool
                pf_g_comps = [
                    a2._trans_prob(
                        a2.gen_prev_pool_trans,
                        a2.gen_prev_pool_totals,
                        [(pool, npool, se) for se in tgt_ids],
                        npool_prior,
                    )
                ]
                if len(nhist) >= 1:
                    pf_g_comps.append(
                        a2._trans_prob(
                            a2.gen_lag1_trans,
                            a2.gen_lag1_totals,
                            [(npool, se) for se in nhist[0]],
                            npool_prior,
                        )
                    )
                if len(nhist) >= nh_span:
                    pf_g_comps.append(
                        a2._trans_prob(
                            a2.gen_lagH_trans,
                            a2.gen_lagH_totals,
                            [(npool, se) for se in nhist[nh_span - 1]],
                            npool_prior,
                        )
                    )
                pf_gen_probs = a2._combine_probs(npool_prior, pf_g_comps)

            demands = tuple(sorted(zip(tgt_ids, tgt_weights), key=lambda kv: kv[0]))
            events.append(
                {
                    "index": global_idx,
                    "pool": pool,
                    "loop": loop,
                    "next_index": global_idx + next_delta[v_idx],
                    "demands": demands,
                    "logit_margin": logit_margin,
                    "haz_loop": haz_loop,
                    "haz_gen": haz_gen,
                    "pf_target_pool": pf_target_pool,
                    "pf_loop_probs": pf_loop_probs,
                    "pf_gen_probs": pf_gen_probs,
                }
            )
            global_idx += 1
    return events


def simulate_b2_policy(
    meta: dict[str, Any],
    sched: dict[str, Any],
    events: list[dict[str, Any]],
    policy: str,
    capacity: int,
    *,
    hot_rank: list[tuple[str, int]],
    h2d_ms: float,
    visit_comp_ms: float,
    freq_weight: float = 0.08,
    prefetch_k: int = 2,
    prefetch_thresh: float = 0.25,
) -> B2SimOutcome:
    """Simulate residency + optional 1-visit-ahead async prefetch under strict peak budget C.

    Peak invariant: at every point in time,
      len(residents) + len(in_flight_buffers) + active_demand_staging <= capacity
    (when capacity < total_experts; when capacity >= total_experts all experts fit).
    """
    pools = meta["pools"]
    size = int(next(iter(pools.values()))["bytes_per_expert"])
    limits = {p: int(d["experts"]) for p, d in pools.items()}
    total_experts = sum(limits.values())
    visits_per_tok = sched["pool_visits_per_token"]
    n_order = len(sched["order"])

    use_pf = policy.endswith("_pf") and capacity < total_experts and capacity > 4
    base_pol = policy[:-3] if policy.endswith("_pf") else policy[:-5]

    residents: dict[str, dict[int, int]] = {p: {} for p in pools}
    heaps: dict[str, list[tuple[float, int, int]]] = {p: [] for p in pools}
    priority: dict[str, list[float]] = {p: [0.0] * limits[p] for p in pools}
    pool_seen_visits: Counter = Counter()
    frequency: Counter = Counter()

    static_keys: set[tuple[str, int]] = set()
    if base_pol == "static_hot":
        max_pin = capacity if capacity >= total_experts else max(0, capacity - 1)
        static_keys = set(hot_rank[:max_pin])

    future: dict[tuple[str, int], deque[int]] = defaultdict(deque)
    if base_pol == "belady":
        serial = 0
        for ev in events:
            p = ev["pool"]
            for e, _ in ev["demands"]:
                future[p, e].append(serial)
                serial += 1

    # In-flight prefetches for the immediate next visit: {(pool, expert): ready_time_ms}
    in_flight: dict[tuple[str, int], float] = {}
    # Protect currently-needed / just-prefetched keys from immediate eviction
    protected: set[tuple[str, int]] = set()

    accesses = 0
    demand_misses = 0
    prefetch_issued = 0
    prefetch_timely = 0
    prefetch_late = 0
    prefetch_wasted = 0
    occupancy = 0
    peak_slots = 0
    hit_bits = bytearray()

    gpu_time = 0.0
    pcie_free = 0.0
    exposed_stall = 0.0

    def _val(pool: str, expert: int) -> float:
        if base_pol == "belady":
            q = future[pool, expert]
            return -float(q[0]) if q else -math.inf
        if base_pol == "lfu":
            return float(frequency[pool, expert])
        if base_pol == "static_hot":
            return 1.0 if (pool, expert) in static_keys else 0.0
        if base_pol == "lru":
            return 0.0
        # loop_hazard / general_hazard: multi-visit hazard + online frequency density
        h_score = priority[pool][expert]
        if freq_weight > 0.0:
            v_seen = pool_seen_visits[pool]
            rate = frequency[pool, expert] / (v_seen + 8.0)
            density = visits_per_tok[pool] / float(n_order)
            h_score = h_score + freq_weight * rate * density
        return h_score

    def _refresh(pool: str) -> None:
        heaps[pool] = [(_val(pool, e), last, e) for e, last in residents[pool].items()]
        heapify(heaps[pool])

    def _evict_one() -> None:
        nonlocal occupancy
        best: tuple[float, int, str, int] | None = None
        for p in pools:
            heap = heaps[p]
            skipped: list[tuple[float, int, int]] = []
            cand = None
            while heap:
                sc, last, ex = heap[0]
                if residents[p].get(ex) != last:
                    heappop(heap)
                    continue
                if (p, ex) in protected:
                    skipped.append(heappop(heap))
                    continue
                cand = (sc, last, p, ex)
                break
            for item in skipped:
                heappush(heap, item)
            if cand is not None and (best is None or cand < best):
                best = cand
        if best is None:
            raise RuntimeError("No evictable expert found under peak budget")
        _, _, vp, ve = best
        del residents[vp][ve]
        occupancy -= 1

    for idx, ev in enumerate(events):
        pool = ev["pool"]
        demands = ev["demands"]
        demand_set = {e for e, _ in demands}
        protected = {(pool, e) for e in demand_set}

        # Non-MoE + router compute runs before routed experts are needed
        t_need = gpu_time + visit_comp_ms

        # Settle in-flight prefetches targeting THIS visit
        if in_flight:
            settled_items = list(in_flight.items())
            in_flight.clear()
            for (pf_pool, pf_e), t_ready in settled_items:
                if pf_pool == pool and pf_e in demand_set:
                    if t_ready <= t_need:
                        prefetch_timely += 1
                    else:
                        prefetch_late += 1
                        wait = t_ready - t_need
                        exposed_stall += wait
                        t_need = t_ready
                else:
                    prefetch_wasted += 1
                # Prefetched expert has already been allocated a slot inside capacity; admit into residents
                if pf_e not in residents[pf_pool]:
                    residents[pf_pool][pf_e] = accesses
                    heappush(heaps[pf_pool], (_val(pf_pool, pf_e), accesses, pf_e))
                else:
                    # Duplicate safeguard
                    occupancy -= 1

        # Update pool hazard score with causally observed post-router signals
        pool_seen_visits[pool] += 1
        if base_pol == "loop_hazard":
            priority[pool] = ev["haz_loop"]
        elif base_pol == "general_hazard":
            priority[pool] = ev["haz_gen"]

        # Serve demands
        for expert, _ in demands:
            key = (pool, expert)
            hit = expert in residents[pool]
            hit_bits.append(1 if hit else 0)
            accesses += 1
            frequency[key] += 1
            if base_pol == "belady":
                future[key].popleft()

            if not hit:
                demand_misses += 1
                # Synchronous demand H2D on PCIe queue
                t_start = max(t_need, pcie_free)
                t_done = t_start + h2d_ms
                pcie_free = t_done
                exposed_stall += t_done - t_need
                t_need = t_done

                if occupancy == capacity:
                    _evict_one()
                peak_slots = max(peak_slots, occupancy + 1)

                admit = True
                if base_pol == "static_hot" and key not in static_keys:
                    admit = False
                if admit:
                    occupancy += 1
                    residents[pool][expert] = accesses
                    heappush(heaps[pool], (_val(pool, expert), accesses, expert))
            else:
                peak_slots = max(peak_slots, occupancy)
                residents[pool][expert] = accesses
                heappush(heaps[pool], (_val(pool, expert), accesses, expert))

        _refresh(pool)
        gpu_time = t_need

        # ---- Async Prefetch for NEXT visit (strictly inside peak budget C) ----
        if use_pf and ev["pf_target_pool"] is not None:
            npool = ev["pf_target_pool"]
            probs = ev["pf_loop_probs"] if base_pol == "loop_hazard" else ev["pf_gen_probs"]
            assert probs is not None
            # Rank missing experts in target pool by predicted probability
            missing_cands = [
                (probs[e], e)
                for e in range(limits[npool])
                if e not in residents[npool] and (npool, e) not in in_flight
            ]
            missing_cands.sort(reverse=True)
            issued_here = 0
            for prob, e in missing_cands[:prefetch_k]:
                if prob < prefetch_thresh:
                    break
                # Reserve a slot INSIDE capacity before launching DMA (strict peak budget!)
                if occupancy >= capacity:
                    _evict_one()
                occupancy += 1
                peak_slots = max(peak_slots, occupancy)
                t_start = max(gpu_time, pcie_free)
                t_ready = t_start + h2d_ms
                pcie_free = t_ready
                in_flight[(npool, e)] = t_ready
                protected.add((npool, e))
                prefetch_issued += 1
                issued_here += 1

    # Any lingering in-flight at end of request counts as wasted
    if in_flight:
        prefetch_wasted += len(in_flight)
        in_flight.clear()

    return B2SimOutcome(
        accesses=accesses,
        demand_misses=demand_misses,
        prefetch_issued=prefetch_issued,
        prefetch_timely=prefetch_timely,
        prefetch_late=prefetch_late,
        prefetch_wasted=prefetch_wasted,
        demand_h2d_bytes=demand_misses * size,
        prefetch_h2d_bytes=prefetch_issued * size,
        exposed_stall_ms=exposed_stall,
        total_step_ms=gpu_time,
        hit_bits=hit_bits,
        peak_bytes=peak_slots * size,
    )


def _bootstrap_ci(values: list[float], n_boot: int = 1000, seed: int = 20261122) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    n = len(values)
    means = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)]
    means.sort()
    return means[max(0, int(0.025 * n_boot))], means[min(n_boot - 1, int(0.975 * n_boot))]


def _replay_worker(args: tuple) -> list[dict[str, Any]]:
    (
        req,
        meta,
        sched,
        a2,
        hot_rank,
        budgets,
        policies,
        h2d_ms,
        visit_comp_ms,
        freq_weight,
        gamma,
        prefetch_k,
        prefetch_thresh,
    ) = args
    events = _prepare_b2_events(req, meta, sched, a2, gamma=gamma)
    n_tok = max(1, len(req["steps"]))
    rows = []
    for cap in budgets:
        for pol in policies:
            out = simulate_b2_policy(
                meta,
                sched,
                events,
                pol,
                cap,
                hot_rank=hot_rank,
                h2d_ms=h2d_ms,
                visit_comp_ms=visit_comp_ms,
                freq_weight=freq_weight,
                prefetch_k=prefetch_k,
                prefetch_thresh=prefetch_thresh,
            )
            bpe = int(next(iter(meta["pools"].values()))["bytes_per_expert"])
            total_h2d = out.demand_h2d_bytes + out.prefetch_h2d_bytes
            rows.append(
                {
                    "request_id": req["request_id"],
                    "category": req["category"],
                    "output_tokens": n_tok,
                    "budget_experts": cap,
                    "budget_gib": round(cap * bpe / (1024**3), 4),
                    "policy": pol,
                    "accesses": out.accesses,
                    "demand_misses": out.demand_misses,
                    "prefetch_issued": out.prefetch_issued,
                    "prefetch_timely": out.prefetch_timely,
                    "prefetch_late": out.prefetch_late,
                    "prefetch_wasted": out.prefetch_wasted,
                    "demand_hit_rate": 1.0 - (out.demand_misses / max(1, out.accesses)),
                    "effective_hit_rate": 1.0 - (out.demand_misses / max(1, out.accesses)),
                    "demand_h2d_mib_per_tok": (out.demand_h2d_bytes / (1024**2)) / n_tok,
                    "prefetch_h2d_mib_per_tok": (out.prefetch_h2d_bytes / (1024**2)) / n_tok,
                    "total_h2d_mib_per_tok": (total_h2d / (1024**2)) / n_tok,
                    "timely_prefetch_mib": (out.prefetch_timely * bpe) / (1024**2),
                    "late_prefetch_mib": (out.prefetch_late * bpe) / (1024**2),
                    "wasted_prefetch_mib": (out.prefetch_wasted * bpe) / (1024**2),
                    "exposed_stall_ms_per_tok": out.exposed_stall_ms / n_tok,
                    "total_ms_per_tok": out.total_step_ms / n_tok,
                    "peak_bytes": out.peak_bytes,
                }
            )
    return rows


def _tune_on_val(
    val_reqs: list[dict[str, Any]],
    meta: dict[str, Any],
    sched: dict[str, Any],
    a2: A2Models,
    hot_rank: list[tuple[str, int]],
    *,
    h2d_ms: float,
    visit_comp_ms: float,
    tune_budget: int = 302,
) -> dict[str, Any]:
    """Select (gamma, freq_weight, prefetch_thresh) strictly on val_reqs (zero eval leakage)."""
    sample_val = val_reqs[:12]
    best_evict = (float("inf"), 0.75, 0.08)
    for gamma in (0.5, 0.75, 1.0):
        ev_cache = [_prepare_b2_events(r, meta, sched, a2, gamma=gamma) for r in sample_val]
        for fw in (0.0, 0.05, 0.15, 0.35):
            misses = sum(
                simulate_b2_policy(
                    meta,
                    sched,
                    evs,
                    "loop_hazard_nopf",
                    tune_budget,
                    hot_rank=hot_rank,
                    h2d_ms=h2d_ms,
                    visit_comp_ms=visit_comp_ms,
                    freq_weight=fw,
                ).demand_misses
                for evs in ev_cache
            )
            if misses < best_evict[0]:
                best_evict = (misses, gamma, fw)

    _, best_gamma, best_fw = best_evict
    ev_cache = [_prepare_b2_events(r, meta, sched, a2, gamma=best_gamma) for r in sample_val]

    # Tune prefetch confidence threshold separately for loop_pf and gen_pf (or shared best) to minimize exposed stall
    best_pf = (float("inf"), 0.25)
    for thresh in (0.15, 0.25, 0.35, 0.50):
        stall = sum(
            simulate_b2_policy(
                meta,
                sched,
                evs,
                "loop_hazard_pf",
                tune_budget,
                hot_rank=hot_rank,
                h2d_ms=h2d_ms,
                visit_comp_ms=visit_comp_ms,
                freq_weight=best_fw,
                prefetch_k=2,
                prefetch_thresh=thresh,
            ).exposed_stall_ms
            for evs in ev_cache
        )
        if stall < best_pf[0]:
            best_pf = (stall, thresh)

    return {
        "gamma": best_gamma,
        "freq_weight": best_fw,
        "prefetch_thresh": best_pf[1],
        "val_sample_n": len(sample_val),
        "tune_budget": tune_budget,
    }


def load_trace_splits(trace_path: Path, meta: dict[str, Any]) -> tuple[set[str], set[str], set[str]]:
    """Load (fit, val, eval) request ID sets from adjacent manifest JSON or meta['splits']."""
    if "splits" in meta:
        sp = meta["splits"]
        return set(sp["fit"]), set(sp.get("val", [])), set(sp["eval"])
    manifest_name = "requests_manifest_fixed.json" if "fixed" in trace_path.name else "requests_manifest_freegen.json"
    manifest_path = trace_path.parent / manifest_name
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Expected split manifest at {manifest_path} (and no meta['splits'] found)")
    manifest = read_json(manifest_path)
    return (
        set(manifest["fit_request_ids"]),
        set(manifest.get("val_request_ids", [])),
        set(manifest["eval_request_ids"]),
    )


def run_b2_offline_replay(
    trace_path: Path,
    out_dir: Path,
    cfg: dict[str, Any],
    *,
    h2d_ms: float,
    visit_comp_ms: float,
    regime_name: str = "measured_b1",
    workers: int | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta, requests = read_ifm_trace(trace_path)
    sched = _build_schedule_info(meta)

    fit_set, val_set, eval_set = load_trace_splits(trace_path, meta)
    fit_reqs = [r for r in requests if r["request_id"] in fit_set]
    val_reqs = [r for r in requests if r["request_id"] in val_set]
    eval_reqs = [r for r in requests if r["request_id"] in eval_set]

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

    tuned = _tune_on_val(
        val_reqs if val_reqs else fit_reqs[:6],
        meta,
        sched,
        a2,
        hot_rank,
        h2d_ms=h2d_ms,
        visit_comp_ms=visit_comp_ms,
    )

    budgets = [int(b) for b in cfg.get("b2_offline", {}).get("budgets_experts", [95, 191, 287, 302, 383, 479])]
    n_workers = workers or min(16, max(1, (os.cpu_count() or 4) // 2), len(eval_reqs))

    tasks = [
        (
            req,
            meta,
            sched,
            a2,
            hot_rank,
            budgets,
            B2_POLICIES,
            h2d_ms,
            visit_comp_ms,
            tuned["freq_weight"],
            tuned["gamma"],
            2,
            tuned["prefetch_thresh"],
        )
        for req in eval_reqs
    ]

    all_rows: list[dict[str, Any]] = []
    if n_workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for chunk in ex.map(_replay_worker, tasks):
                all_rows.extend(chunk)
    else:
        for t in tasks:
            all_rows.extend(_replay_worker(t))

    write_csv(out_dir / f"b2_offline_requests_{regime_name}.csv", all_rows)

    # Aggregate summary per (budget, policy)
    by_bp: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for r in all_rows:
        by_bp[int(r["budget_experts"]), r["policy"]].append(r)

    summary_rows: list[dict[str, Any]] = []
    for cap in budgets:
        for pol in B2_POLICIES:
            items = by_bp[cap, pol]
            tot_acc = sum(x["accesses"] for x in items)
            tot_miss = sum(x["demand_misses"] for x in items)
            tot_tok = sum(x["output_tokens"] for x in items)
            tot_pf = sum(x["prefetch_issued"] for x in items)
            tot_timely = sum(x["prefetch_timely"] for x in items)
            tot_late = sum(x["prefetch_late"] for x in items)
            tot_wasted = sum(x["prefetch_wasted"] for x in items)
            bpe = int(next(iter(meta["pools"].values()))["bytes_per_expert"])
            stall_vals = [x["exposed_stall_ms_per_tok"] for x in items]
            summary_rows.append(
                {
                    "model": meta["model"],
                    "regime": regime_name,
                    "budget_experts": cap,
                    "budget_gib": items[0]["budget_gib"],
                    "policy": pol,
                    "requests": len(items),
                    "demand_hit_rate": 1.0 - tot_miss / max(1, tot_acc),
                    "demand_h2d_mib_per_tok": (tot_miss * bpe / (1024**2)) / max(1, tot_tok),
                    "prefetch_h2d_mib_per_tok": (tot_pf * bpe / (1024**2)) / max(1, tot_tok),
                    "total_h2d_mib_per_tok": ((tot_miss + tot_pf) * bpe / (1024**2)) / max(1, tot_tok),
                    "timely_pf_rate": tot_timely / max(1, tot_pf),
                    "useful_pf_rate": (tot_timely + tot_late) / max(1, tot_pf),
                    "wasted_pf_rate": tot_wasted / max(1, tot_pf),
                    "exposed_stall_ms_per_tok": statistics.mean(stall_vals),
                    "exposed_stall_p50_ms_per_tok": statistics.median(stall_vals),
                    "total_ms_per_tok": statistics.mean(x["total_ms_per_tok"] for x in items),
                }
            )
    write_csv(out_dir / f"b2_offline_summary_{regime_name}.csv", summary_rows)

    # Paired comparisons (95% bootstrap CI on stall reduction & H2D reduction)
    pairs = [
        ("loop_hazard_nopf", "general_hazard_nopf"),
        ("loop_hazard_nopf", "lfu_nopf"),
        ("loop_hazard_nopf", "lru_nopf"),
        ("loop_hazard_pf", "general_hazard_pf"),
        ("loop_hazard_pf", "loop_hazard_nopf"),
        ("loop_hazard_pf", "general_hazard_nopf"),
        ("loop_hazard_pf", "static_hot_nopf"),
        ("general_hazard_pf", "general_hazard_nopf"),
    ]
    paired_rows: list[dict[str, Any]] = []
    for cap in budgets:
        for pol, base in pairs:
            p_map = {x["request_id"]: x for x in by_bp[cap, pol]}
            b_map = {x["request_id"]: x for x in by_bp[cap, base]}
            req_ids = sorted(p_map.keys())
            stall_red_pct = []
            h2d_red_pct = []
            wins_stall = ties_stall = losses_stall = 0
            for rid in req_ids:
                ps = p_map[rid]["exposed_stall_ms_per_tok"]
                bs = b_map[rid]["exposed_stall_ms_per_tok"]
                if bs > 1e-9:
                    stall_red_pct.append(100.0 * (bs - ps) / bs)
                ph = p_map[rid]["demand_h2d_mib_per_tok"]
                bh = b_map[rid]["demand_h2d_mib_per_tok"]
                if bh > 1e-9:
                    h2d_red_pct.append(100.0 * (bh - ph) / bh)
                if ps + 1e-9 < bs:
                    wins_stall += 1
                elif bs + 1e-9 < ps:
                    losses_stall += 1
                else:
                    ties_stall += 1
            s_lo, s_hi = _bootstrap_ci(stall_red_pct)
            h_lo, h_hi = _bootstrap_ci(h2d_red_pct)
            paired_rows.append(
                {
                    "model": meta["model"],
                    "regime": regime_name,
                    "budget_experts": cap,
                    "budget_gib": by_bp[cap, pol][0]["budget_gib"],
                    "policy": pol,
                    "baseline": base,
                    "requests": len(req_ids),
                    "mean_stall_saved_pct": statistics.mean(stall_red_pct) if stall_red_pct else 0.0,
                    "stall_ci95_lo_pct": s_lo,
                    "stall_ci95_hi_pct": s_hi,
                    "mean_demand_h2d_saved_pct": statistics.mean(h2d_red_pct) if h2d_red_pct else 0.0,
                    "h2d_ci95_lo_pct": h_lo,
                    "h2d_ci95_hi_pct": h_hi,
                    "wins_stall": wins_stall,
                    "ties_stall": ties_stall,
                    "losses_stall": losses_stall,
                }
            )
    write_csv(out_dir / f"b2_offline_paired_{regime_name}.csv", paired_rows)

    headline = {
        "model": meta["model"],
        "regime": regime_name,
        "h2d_ms": h2d_ms,
        "visit_comp_ms": visit_comp_ms,
        "tuned": tuned,
        "eval_requests": len(eval_reqs),
    }
    write_json(out_dir / f"b2_offline_headline_{regime_name}.json", headline)
    return headline


def main() -> None:
    parser = argparse.ArgumentParser(description="B2 offline 96-request CPU co-simulation")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--h2d-ms", type=float, required=True)
    parser.add_argument("--visit-comp-ms", type=float, required=True)
    parser.add_argument("--regime", default="measured_b1")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    cfg = read_json(Path(args.config))
    res = run_b2_offline_replay(
        Path(args.trace),
        Path(args.out_dir),
        cfg,
        h2d_ms=args.h2d_ms,
        visit_comp_ms=args.visit_comp_ms,
        regime_name=args.regime,
        workers=args.workers,
    )
    print(f"[B2 offline {res['model']}/{args.regime}] {json.dumps(res, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
