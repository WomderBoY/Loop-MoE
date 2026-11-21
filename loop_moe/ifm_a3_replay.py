"""A3: Same-budget expert residency replay from static pinning to Belady oracle.

Strict peak-budget accounting: currently computing/loading expert counts inside the
byte budget. Identical cold start, physical access order, and service order for all
policies, including exact offline Belady and Huginn recurrent-block pinning.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from heapq import heapify, heappop, heappush
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any

from .ifm_a2_predict import A2Models, _build_schedule_info, read_ifm_trace
from .trace import read_json, write_csv


POLICIES = (
    "lru",
    "lfu",
    "static_hot",
    "score_ema",
    "oled_gate_only",
    "oled_adaptive",
    "general_history",
    "general_history_distance",
    "loop_a2",
    "loop_a2_distance",
    "loop_shuffled_distance",
    "pin_recurrent_lru",
    "pin_recurrent_static_hot",
    "pin_recurrent_loop",
    "belady",
)

ONLINE_POLICIES = tuple(p for p in POLICIES if p != "belady")


def _allocate_quotas(capacity: int, weights: dict[str, float], limits: dict[str, int]) -> dict[str, int]:
    quotas = {p: 0 for p in limits}
    remaining = min(capacity, sum(limits.values()))
    while remaining > 0:
        active = [p for p in limits if quotas[p] < limits[p]]
        total = sum(weights[p] for p in active)
        shares = {p: remaining * (weights[p] / total if total else 1.0 / len(active)) for p in active}
        assigned = {p: min(limits[p] - quotas[p], int(shares[p])) for p in active}
        if not sum(assigned.values()):
            for p in sorted(active, key=lambda x: (-shares[x], x))[:remaining]:
                assigned[p] = 1
        for p, v in assigned.items():
            quotas[p] += v
            remaining -= v
    return quotas


@dataclass
class SimOutcome:
    accesses: int
    misses: int
    reloads: int
    hit_bits: bytearray
    peak_bytes: int


def _prepare_request_events(
    req: dict[str, Any],
    meta: dict[str, Any],
    sched: dict[str, Any],
    a2: A2Models,
) -> list[dict[str, Any]]:
    """Build per-visit replay stream with strictly causal next-pool-visit priorities."""
    order = sched["order"]
    n_order = len(order)
    pool_positions = sched["pool_positions"]
    delta = []
    for idx, (pool, _) in enumerate(order):
        pos = pool_positions[pool]
        nxt = next((p for p in pos if p > idx), pos[0] + n_order)
        delta.append(nxt - idx)

    # First run A2 causal predictions at every visit (strictly before target fires)
    # Note: for cache retention AFTER visit i executes at pool P, we want the predicted
    # probability for the NEXT visit to pool P (at index i + delta[i % n_order]), using
    # only information available right after visit i completes!
    # A2Models.predict_request scores visit i BEFORE visit i fires; here we compute
    # post-visit retention scores for pool P's next visit using the exact same fitted tables.
    alpha = a2.alpha
    num_experts = a2.num_experts
    visits_per_tok = sched["pool_visits_per_token"]

    # Map each visit_idx to next visit's loop index for the same pool
    next_loop_for_visit = [order[(idx + delta[idx]) % n_order][1] for idx in range(n_order)]
    is_boundary_for_visit = [(idx + delta[idx]) >= n_order for idx in range(n_order)]

    pool_recent: dict[str, deque[list[int]]] = defaultdict(deque)
    pool_stage_last: dict[tuple[str, int | None], list[int]] = {}
    pool_ema: dict[str, list[float]] = defaultdict(lambda: [0.0] * num_experts)

    events = []
    global_idx = 0
    for step in req["steps"]:
        for v_idx, visit in enumerate(step["visits"]):
            pool, loop = order[v_idx]
            tgt_ids = visit[0]
            tgt_weights = visit[1]
            cand_ids = visit[3]
            cand_scores = visit[4]

            # Observe current visit (available right when current visit executes!)
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

            next_loop = next_loop_for_visit[v_idx]
            boundary = is_boundary_for_visit[v_idx]
            stage_prior = a2._stage_prior(pool, next_loop)
            pool_prior = a2._pool_prior(pool)

            # Loop-aware retention score for pool's NEXT visit (at next_loop):
            comps_loop = []
            # Prev-token same-stage (if we have seen next_loop before in this request)
            prev_stage_ids = pool_stage_last.get((pool, next_loop))
            if prev_stage_ids is not None:
                comps_loop.append(
                    a2._trans_prob(
                        a2.prev_tok_trans,
                        a2.prev_tok_totals,
                        [(pool, next_loop, se) for se in prev_stage_ids],
                        stage_prior,
                    )
                )
            # Same-token prev-loop transition (when next visit is in the SAME token!)
            if not boundary and loop is not None and next_loop is not None:
                comps_loop.append(
                    a2._trans_prob(
                        a2.prev_loop_trans,
                        a2.prev_loop_totals,
                        [(pool, loop, next_loop, se) for se in tgt_ids],
                        stage_prior,
                    )
                )
                p_shuf = a2._trans_prob(
                    a2.shuf_loop_trans,
                    a2.shuf_loop_totals,
                    [(pool, loop, next_loop, se) for se in tgt_ids],
                    stage_prior,
                )
            else:
                p_shuf = list(stage_prior)
            p_loop = a2._combine_probs(stage_prior, comps_loop) if comps_loop else list(stage_prior)
            p_loop = [0.85 * a + 0.15 * min(1.0, b) for a, b in zip(p_loop, pema)]

            # Matched general-history retention score (stage-blind, same lag-1 + lag-H + EMA)
            gen_comps = [
                a2._trans_prob(
                    a2.gen_lag1_trans,
                    a2.gen_lag1_totals,
                    [(pool, se) for se in hist[0]],
                    pool_prior,
                )
            ]
            if len(hist) >= h_span:
                gen_comps.append(
                    a2._trans_prob(
                        a2.gen_lagH_trans,
                        a2.gen_lagH_totals,
                        [(pool, se) for se in hist[h_span - 1]],
                        pool_prior,
                    )
                )
            p_gen_raw = a2._combine_probs(pool_prior, gen_comps)
            p_gen = [0.85 * a + 0.15 * min(1.0, b) for a, b in zip(p_gen_raw, pema)]

            demands = tuple(sorted(zip(tgt_ids, tgt_weights), key=lambda kv: kv[0]))
            events.append(
                {
                    "index": global_idx,
                    "pool": pool,
                    "loop": loop,
                    "next_index": global_idx + delta[v_idx],
                    "demands": demands,
                    "ema": list(pema),
                    "p_loop": p_loop,
                    "p_gen": p_gen,
                    "p_shuf": p_shuf,
                }
            )
            global_idx += 1
    return events


def simulate_policy(
    meta: dict[str, Any],
    sched: dict[str, Any],
    events: list[dict[str, Any]],
    policy: str,
    capacity: int,
    *,
    hot_rank: list[tuple[str, int]],
    recurrent_hot_rank: list[tuple[str, int]],
    nonrec_hot_rank: list[tuple[str, int]],
    cfg: dict[str, Any],
) -> SimOutcome:
    pools = meta["pools"]
    size = int(next(iter(pools.values()))["bytes_per_expert"])
    limits = {p: int(d["experts"]) for p, d in pools.items()}
    residents: dict[str, dict[int, int]] = {p: {} for p in pools}
    heaps: dict[str, list[tuple[float, int, int]]] = {p: [] for p in pools}
    pending = {p: pos[0] for p, pos in sched["pool_positions"].items()}
    priority: dict[str, list[float]] = {p: [0.0] * limits[p] for p in pools}

    distance = policy.endswith("_distance") or policy == "pin_recurrent_loop"
    adaptive = policy == "oled_adaptive"
    is_pin_rec = policy.startswith("pin_recurrent_")

    # Determine pinned expert sets (strictly <= capacity - 1 if unpinned experts exist,
    # so peak mode ALWAYS has >= 1 staging slot for computing an unpinned miss expert!)
    total_experts = sum(limits.values())
    static_keys: set[tuple[str, int]] = set()
    pinned_keys: set[tuple[str, int]] = set()
    nonrec_static_keys: set[tuple[str, int]] = set()

    if policy == "static_hot":
        max_pin = capacity if capacity >= total_experts else max(0, capacity - 1)
        static_keys = set(hot_rank[:max_pin])
    elif is_pin_rec:
        n_rec = len(recurrent_hot_rank)
        if capacity >= total_experts:
            pinned_keys = set(recurrent_hot_rank)
            rem = capacity - len(pinned_keys)
        else:
            # Keep 1 slot free for computing misses unless every expert fits
            max_usable = max(0, capacity - 1)
            n_pin_rec = min(n_rec, max_usable)
            pinned_keys = set(recurrent_hot_rank[:n_pin_rec])
            rem = max(0, max_usable - len(pinned_keys))
        if policy == "pin_recurrent_static_hot":
            nonrec_static_keys = set(nonrec_hot_rank[:rem])

    frequency: Counter = Counter()
    seen: set[tuple[str, int]] = set()
    accesses = misses = reloads = occupancy = peak = 0
    hit_bits = bytearray()

    weights = {p: 1.0 for p in pools}
    activation = {p: 1.0 for p in pools}
    utilization = {p: 1.0 for p in pools}
    quotas = _allocate_quotas(capacity, weights, limits)
    alpha = float(cfg["score_alpha"])
    oled_period = len(sched["order"]) * int(cfg["oled_update_tokens"])

    future: dict[tuple[str, int], deque[int]] = defaultdict(deque)
    if policy == "belady":
        serial = 0
        for ev in events:
            p = ev["pool"]
            for e, _ in ev["demands"]:
                future[p, e].append(serial)
                serial += 1

    def _val(pool: str, expert: int) -> float:
        if policy == "belady":
            q = future[pool, expert]
            return -float(q[0]) if q else -math.inf
        if policy == "lfu":
            return float(frequency[pool, expert])
        if policy == "static_hot":
            return 1.0 if (pool, expert) in static_keys else 0.0
        if policy == "lru":
            return 0.0
        if is_pin_rec:
            if (pool, expert) in pinned_keys:
                return 1e9  # never evict pinned recurrent experts
            if policy == "pin_recurrent_static_hot":
                return 1.0 if (pool, expert) in nonrec_static_keys else 0.0
            if policy == "pin_recurrent_lru":
                return 0.0
            return priority[pool][expert]
        return priority[pool][expert]

    def _refresh(pool: str) -> None:
        heaps[pool] = [(_val(pool, e), last, e) for e, last in residents[pool].items()]
        heapify(heaps[pool])

    def _lowest(pool: str, cur_idx: int) -> tuple[float, int, str, int] | None:
        heap = heaps[pool]
        while heap and residents[pool].get(heap[0][2]) != heap[0][1]:
            heappop(heap)
        if not heap:
            return None
        score, last, expert = heap[0]
        if distance and score < 1e8:
            score = score / (1.0 + max(0, pending[pool] - cur_idx - 1))
        return score, last, pool, expert

    def _evict(cur_idx: int, pool: str | None = None) -> None:
        nonlocal occupancy
        if pool is not None:
            cand = _lowest(pool, cur_idx)
            if cand is None:
                raise RuntimeError(f"Cannot evict from empty pool {pool}")
            victim = cand
        else:
            cands = [_lowest(p, cur_idx) for p in pools]
            victim = min(c for c in cands if c is not None)
        _, _, vp, ve = victim
        del residents[vp][ve]
        occupancy -= 1

    for idx, ev in enumerate(events):
        pool = ev["pool"]
        demands = ev["demands"]

        # Update pool priority with causally observed post-router signals
        if policy in ("loop_a2", "loop_a2_distance", "pin_recurrent_loop"):
            priority[pool] = ev["p_loop"]
        elif policy in ("general_history", "general_history_distance"):
            priority[pool] = ev["p_gen"]
        elif policy == "loop_shuffled_distance":
            priority[pool] = ev["p_shuf"]
        elif policy == "score_ema":
            priority[pool] = ev["ema"]
        elif policy in ("oled_gate_only", "oled_adaptive"):
            tot_w = sum(w for _, w in demands) or 1.0
            pr = [0.0] * limits[pool]
            for e, w in demands:
                pr[e] = w / tot_w
            priority[pool] = pr

        pending[pool] = ev["next_index"]
        _refresh(pool)

        if adaptive:
            start_res = set(residents[pool])
            start_hits = sum(1 for e, _ in demands if e in start_res)
            activation[pool] = (1.0 - alpha) * activation[pool] + alpha * len(demands) / limits[pool]
            useful = start_hits / len(start_res) if start_res else 1.0
            utilization[pool] = (1.0 - alpha) * utilization[pool] + alpha * useful

        for expert, _ in demands:
            key = (pool, expert)
            hit = expert in residents[pool]
            hit_bits.append(1 if hit else 0)
            accesses += 1
            frequency[key] += 1
            if policy == "belady":
                future[key].popleft()

            if not hit:
                misses += 1
                if key in seen:
                    reloads += 1
                seen.add(key)
                if occupancy == capacity:
                    if adaptive and len(residents[pool]) >= quotas[pool] and residents[pool]:
                        _evict(idx, pool)
                    else:
                        _evict(idx)
                peak = max(peak, (occupancy + 1) * size)
            else:
                peak = max(peak, occupancy * size)

            # Admission check
            admit = True
            if not hit:
                if policy == "static_hot" and key not in static_keys:
                    admit = False
                elif policy == "pin_recurrent_static_hot" and (
                    key not in pinned_keys and key not in nonrec_static_keys
                ):
                    admit = False
                elif is_pin_rec and len(pinned_keys) >= capacity - 1 and key not in pinned_keys:
                    # Budget completely full of pinned recurrent experts (except 1 staging slot)
                    admit = False
                elif adaptive and quotas[pool] == 0:
                    admit = False

            if not admit:
                continue

            if not hit:
                occupancy += 1
            residents[pool][expert] = accesses
            heappush(heaps[pool], (_val(pool, expert), accesses, expert))
            if adaptive and len(residents[pool]) > quotas[pool]:
                _evict(idx, pool)
            if occupancy > capacity:
                _evict(idx)
            if len(heaps[pool]) > 4 * max(1, len(residents[pool])):
                _refresh(pool)

        if adaptive and (idx + 1) % oled_period == 0:
            weights = {p: activation[p] * utilization[p] for p in pools}
            quotas = _allocate_quotas(capacity, weights, limits)
            for p in pools:
                while len(residents[p]) > quotas[p]:
                    _evict(idx, p)

    return SimOutcome(accesses, misses, reloads, hit_bits, peak)


def _bootstrap_ci(values: list[float], n_boot: int = 1000, seed: int = 20261121) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    n = len(values)
    means = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)]
    means.sort()
    return means[max(0, int(0.025 * n_boot))], means[min(n_boot - 1, int(0.975 * n_boot))]


def run_a3(
    trace_path: Path,
    manifest: dict[str, Any],
    a2_cfg: dict[str, Any],
    a3_cfg: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    meta, requests = read_ifm_trace(trace_path)
    sched = _build_schedule_info(meta)
    req_by_id = {r["request_id"]: r for r in requests}
    fit_reqs = [req_by_id[rid] for rid in manifest["fit_request_ids"] if rid in req_by_id]
    eval_reqs = [req_by_id[rid] for rid in manifest["eval_request_ids"] if rid in req_by_id]
    if not fit_reqs or not eval_reqs:
        half = max(1, len(requests) // 2)
        fit_reqs = requests[:half]
        eval_reqs = requests[half:]

    a2 = A2Models(meta, sched, a2_cfg)
    a2.fit(fit_reqs, int(a2_cfg["shuffle_seed"]))

    # Fit-only expert frequency rankings (global, recurrent-only, non-recurrent-only)
    fit_freq: Counter = Counter()
    order = sched["order"]
    for req in fit_reqs:
        for step in req["steps"]:
            for v_idx, visit in enumerate(step["visits"]):
                pool = order[v_idx][0]
                for e in visit[0]:
                    fit_freq[pool, e] += 1

    pools = meta["pools"]
    all_keys = [(p, e) for p, d in pools.items() for e in range(int(d["experts"]))]
    hot_rank = sorted(all_keys, key=lambda k: (-fit_freq[k], k))
    rec_keys = [(p, e) for p, e in all_keys if pools[p]["region"] == "recurrent"]
    nonrec_keys = [(p, e) for p, e in all_keys if pools[p]["region"] != "recurrent"]
    recurrent_hot_rank = sorted(rec_keys, key=lambda k: (-fit_freq[k], k))
    nonrec_hot_rank = sorted(nonrec_keys, key=lambda k: (-fit_freq[k], k))

    size = int(next(iter(pools.values()))["bytes_per_expert"])
    total_experts = len(all_keys)
    budgets_experts: list[int] = list(a3_cfg["budgets_experts"])
    non_expert_gib = float(a3_cfg.get("non_expert_gib", 1.75))
    kv_kib_per_tok = float(a3_cfg.get("kv_kib_per_token", 168.0))

    req_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []

    for req in eval_reqs:
        events = _prepare_request_events(req, meta, sched, a2)
        n_decode = len(req["steps"])
        seq_len = int(req.get("prompt_len", 0)) + n_decode
        est_kv_gib = seq_len * kv_kib_per_tok / (1024.0 * 1024.0)

        for cap in budgets_experts:
            budget_bytes = cap * size
            budget_gib = budget_bytes / (1024**3)
            est_total_vram_gib = budget_gib + non_expert_gib + est_kv_gib

            outcomes: dict[str, SimOutcome] = {}
            for policy in POLICIES:
                out = simulate_policy(
                    meta,
                    sched,
                    events,
                    policy,
                    cap,
                    hot_rank=hot_rank,
                    recurrent_hot_rank=recurrent_hot_rank,
                    nonrec_hot_rank=nonrec_hot_rank,
                    cfg=a3_cfg,
                )
                # Strict invariants
                if out.peak_bytes > budget_bytes:
                    raise AssertionError(
                        f"Peak budget violation: {policy} cap={cap} peak={out.peak_bytes} > {budget_bytes}"
                    )
                outcomes[policy] = out

            belady_misses = outcomes["belady"].misses
            for policy, out in outcomes.items():
                if out.misses < belady_misses:
                    raise AssertionError(
                        f"Online policy {policy} beat exact Belady ({out.misses} < {belady_misses})"
                    )
                h2d = out.misses * size
                req_rows.append(
                    {
                        "model": meta["model"],
                        "budget_experts": cap,
                        "budget_gib": round(budget_gib, 4),
                        "budget_percent": round(100.0 * cap / total_experts, 2),
                        "est_total_vram_gib": round(est_total_vram_gib, 3),
                        "policy": policy,
                        "request_id": req["request_id"],
                        "category": req["category"],
                        "decode_tokens": n_decode,
                        "accesses": out.accesses,
                        "misses": out.misses,
                        "reloads": out.reloads,
                        "h2d_bytes": h2d,
                        "h2d_mib_per_token": (h2d / (1024**2)) / n_decode if n_decode else 0.0,
                        "hit_rate_by_bytes": 1.0 - out.misses / out.accesses if out.accesses else 0.0,
                        "peak_weight_bytes": out.peak_bytes,
                    }
                )

            # Paired comparisons for key candidate policies against baselines
            compare_baselines = (
                "lru",
                "static_hot",
                "oled_adaptive",
                "general_history_distance",
                "pin_recurrent_lru",
                "pin_recurrent_static_hot",
            )
            for cand_pol in ("loop_a2", "loop_a2_distance", "pin_recurrent_loop", "pin_recurrent_lru", "general_history_distance"):
                cand_out = outcomes[cand_pol]
                for base_pol in compare_baselines:
                    if cand_pol == base_pol:
                        continue
                    base_out = outcomes[base_pol]
                    gain = sum(a > b for a, b in zip(cand_out.hit_bits, base_out.hit_bits)) * size
                    loss = sum(a < b for a, b in zip(cand_out.hit_bits, base_out.hit_bits)) * size
                    net = gain - loss
                    expected_net = (base_out.misses - cand_out.misses) * size
                    if net != expected_net:
                        raise AssertionError(f"Gained-lost mismatch: {net} != {expected_net}")
                    paired_rows.append(
                        {
                            "budget_experts": cap,
                            "budget_gib": round(budget_gib, 4),
                            "policy": cand_pol,
                            "baseline": base_pol,
                            "request_id": req["request_id"],
                            "decode_tokens": n_decode,
                            "gained_hit_bytes": gain,
                            "lost_hit_bytes": loss,
                            "net_saved_h2d_bytes": net,
                            "saved_h2d_fraction": net / (base_out.misses * size) if base_out.misses else 0.0,
                            "delta_h2d_mib_per_token": -(net / (1024**2)) / n_decode if n_decode else 0.0,
                        }
                    )

    # Aggregate summary per (budget, policy)
    agg: dict[tuple[int, str], Counter] = defaultdict(Counter)
    by_bp: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for r in req_rows:
        key = (r["budget_experts"], r["policy"])
        agg[key].update(
            decode_tokens=r["decode_tokens"],
            accesses=r["accesses"],
            misses=r["misses"],
            h2d_bytes=r["h2d_bytes"],
        )
        by_bp[key].append(r)

    summary_rows = []
    for cap in budgets_experts:
        budget_gib = cap * size / (1024**3)
        belady_h2d = agg[cap, "belady"]["h2d_bytes"]
        strongest_online = min(ONLINE_POLICIES, key=lambda p: (agg[cap, p]["h2d_bytes"], p))
        strongest_h2d = agg[cap, strongest_online]["h2d_bytes"]
        # Strongest non-loop online baseline (excluding loop_a2* and pin_recurrent_loop)
        non_loop_baselines = [
            p for p in ONLINE_POLICIES if p not in ("loop_a2", "loop_a2_distance", "pin_recurrent_loop")
        ]
        strongest_non_loop = min(non_loop_baselines, key=lambda p: (agg[cap, p]["h2d_bytes"], p))
        strongest_non_loop_h2d = agg[cap, strongest_non_loop]["h2d_bytes"]

        for policy in POLICIES:
            c = agg[cap, policy]
            h2d = c["h2d_bytes"]
            dtok = c["decode_tokens"]
            acc = c["accesses"]
            summary_rows.append(
                {
                    "model": meta["model"],
                    "budget_experts": cap,
                    "budget_gib": round(budget_gib, 4),
                    "budget_percent": round(100.0 * cap / total_experts, 2),
                    "policy": policy,
                    "accesses": acc,
                    "misses": c["misses"],
                    "h2d_bytes": h2d,
                    "hit_rate_by_bytes": 1.0 - c["misses"] / acc if acc else 0.0,
                    "h2d_mib_per_token": (h2d / (1024**2)) / dtok if dtok else 0.0,
                    "belady_hit_rate": 1.0 - agg[cap, "belady"]["misses"] / acc if acc else 0.0,
                    "gap_to_belady_pp": (
                        100.0 * (c["misses"] - agg[cap, "belady"]["misses"]) / acc if acc else 0.0
                    ),
                    "strongest_non_loop_baseline": strongest_non_loop,
                    "saved_h2d_pct_vs_strongest_non_loop": (
                        100.0 * (1.0 - h2d / strongest_non_loop_h2d) if strongest_non_loop_h2d else 0.0
                    ),
                    "strongest_online_overall": strongest_online,
                    "saved_h2d_pct_vs_strongest_online": (
                        100.0 * (1.0 - h2d / strongest_h2d) if strongest_h2d else 0.0
                    ),
                }
            )

    # Paired summary with bootstrap 95% CI
    pair_groups: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in paired_rows:
        pair_groups[r["budget_experts"], r["policy"], r["baseline"]].append(r)

    paired_summary = []
    n_boot = int(a3_cfg.get("bootstrap_samples", 1000))
    for (cap, pol, base), grp in sorted(pair_groups.items()):
        fracs = [100.0 * r["saved_h2d_fraction"] for r in grp]
        lo, hi = _bootstrap_ci(fracs, n_boot)
        gained = sum(r["gained_hit_bytes"] for r in grp)
        lost = sum(r["lost_hit_bytes"] for r in grp)
        net = gained - lost
        paired_summary.append(
            {
                "budget_experts": cap,
                "budget_gib": round(cap * size / (1024**3), 4),
                "policy": pol,
                "baseline": base,
                "requests": len(grp),
                "gained_hit_mib": round(gained / (1024**2), 3),
                "lost_hit_mib": round(lost / (1024**2), 3),
                "net_saved_h2d_mib": round(net / (1024**2), 3),
                "mean_saved_h2d_pct": statistics.mean(fracs) if fracs else 0.0,
                "ci95_lo_pct": lo,
                "ci95_hi_pct": hi,
                "wins": sum(r["net_saved_h2d_bytes"] > 0 for r in grp),
                "ties": sum(r["net_saved_h2d_bytes"] == 0 for r in grp),
                "losses": sum(r["net_saved_h2d_bytes"] < 0 for r in grp),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "a3_requests.csv", list(req_rows[0]), req_rows)
    write_csv(out_dir / "a3_summary.csv", list(summary_rows[0]), summary_rows)
    write_csv(out_dir / "a3_paired_requests.csv", list(paired_rows[0]), paired_rows)
    write_csv(out_dir / "a3_paired_summary.csv", list(paired_summary[0]), paired_summary)

    # Headline focus around 4, 6, 6.30, 8 GiB
    sidx = {(r["budget_experts"], r["policy"]): r for r in summary_rows}
    focus_caps = [191, 287, 302, 383]
    headline = {
        "model": meta["model"],
        "trace": str(trace_path),
        "fit_requests": len(fit_reqs),
        "eval_requests": len(eval_reqs),
        "focus": {
            str(cap): {
                "budget_gib": round(cap * size / (1024**3), 3),
                "lru_hit_rate": sidx[cap, "lru"]["hit_rate_by_bytes"],
                "static_hot_hit_rate": sidx[cap, "static_hot"]["hit_rate_by_bytes"],
                "oled_adaptive_hit_rate": sidx[cap, "oled_adaptive"]["hit_rate_by_bytes"],
                "pin_recurrent_lru_hit_rate": sidx[cap, "pin_recurrent_lru"]["hit_rate_by_bytes"],
                "general_history_distance_hit_rate": sidx[cap, "general_history_distance"]["hit_rate_by_bytes"],
                "loop_a2_distance_hit_rate": sidx[cap, "loop_a2_distance"]["hit_rate_by_bytes"],
                "pin_recurrent_loop_hit_rate": sidx[cap, "pin_recurrent_loop"]["hit_rate_by_bytes"],
                "belady_hit_rate": sidx[cap, "belady"]["hit_rate_by_bytes"],
                "strongest_non_loop_baseline": sidx[cap, "loop_a2_distance"]["strongest_non_loop_baseline"],
                "loop_saved_h2d_pct_vs_strongest_non_loop": sidx[cap, "loop_a2_distance"][
                    "saved_h2d_pct_vs_strongest_non_loop"
                ],
                "pin_loop_saved_h2d_pct_vs_strongest_non_loop": sidx[cap, "pin_recurrent_loop"][
                    "saved_h2d_pct_vs_strongest_non_loop"
                ],
            }
            for cap in focus_caps
            if (cap, "belady") in sidx
        },
    }
    (out_dir / "a3_headline.json").write_text(json.dumps(headline, indent=2) + "\n", encoding="utf-8")
    return headline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/ifm_a1_a3.json"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    full_cfg = read_json(args.config)
    manifest = read_json(args.manifest)
    headline = run_a3(args.trace, manifest, full_cfg["a2"], full_cfg["a3"], args.out_dir)
    print(json.dumps(headline, indent=2), flush=True)


if __name__ == "__main__":
    main()
