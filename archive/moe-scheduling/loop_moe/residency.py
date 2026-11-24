"""Causal next-pool residency experiment (standard library, no model execution)."""

from __future__ import annotations

import argparse
from array import array
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from heapq import heapify, heappop, heappush
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import time

from .replay import Visit, split_requests
from .trace import read_json, read_trace, write_csv

PREDICTORS = ("gate_ema", "history_a", "history_b", "phase_ab", "pooled_ab",
              "boundary_ab", "generic_lag", "phase_ab_shuffled")
BASELINES = ("lru", "lfu", "static_hot", "lru_loop", "oled_gate_only", "score_ema",
             "score_ema_loop", "oled_adaptive", "belady")
POLICIES = BASELINES + PREDICTORS + ("phase_ab_distance", "generic_lag_distance")
COMPARATORS = tuple(p for p in BASELINES if p != "belady") + (
    "gate_ema", "pooled_ab", "boundary_ab", "generic_lag", "generic_lag_distance")


class Schedule:
    """Execution topology only. Never infer distances from routed experts."""

    def __init__(self, order, pools):
        self.order = tuple((pool, phase) for pool, phase in order)
        if not self.order or len(set(self.order)) != len(self.order):
            raise ValueError("Expected a nonempty fixed schedule with unique pool/phase visits")
        if set(pool for pool, _ in self.order) != set(pools):
            raise ValueError("Schedule and physical pool metadata disagree")
        self.positions = defaultdict(list)
        for index, (pool, _) in enumerate(self.order):
            self.positions[pool].append(index)
        self.delta = []
        self.target = []
        self.boundary = []
        for index, (pool, _) in enumerate(self.order):
            positions = self.positions[pool]
            following = next((pos for pos in positions if pos > index), positions[0] + len(self.order))
            self.delta.append(following - index)
            self.target.append(self.order[following % len(self.order)][1])
            self.boundary.append(following >= len(self.order))

    @classmethod
    def from_trace(cls, meta, requests):
        order = meta.get("execution_order")
        if order is None:
            # Legacy traces: read ONLY structural pool/phase fields, including
            # prefill. The metadata topology is validated against every step.
            first_step = requests[0]["steps"][0]
            order = [(visit["pool"], visit["loop"]) for visit in first_step["visits"]]
        return cls(order, meta["pools"])

    def visits(self, request):
        result = []
        for step in request["steps"]:
            if [(v["pool"], v["loop"]) for v in step["visits"]] != list(self.order):
                raise ValueError(f"Variable or incomplete schedule: {request['request_id']}")
            if step["phase"] != "decode":
                continue
            for visit in step["visits"]:
                if len(visit["routes"]) != 1:
                    raise ValueError("Residency requires batch=1, one projected decode position")
                route = visit["routes"][0]
                if len(route[0]) != len(set(route[0])) or not route[0]:
                    raise ValueError("Expected unique, nonempty routed experts")
                result.append(Visit(visit["pool"], visit["loop"],
                                    tuple(sorted(zip(route[0], route[1], strict=True))), tuple(route[6])))
        return result


class History:
    def __init__(self, schedule, meta, alpha):
        self.schedule, self.meta, self.alpha = schedule, meta, alpha
        self.stage = {}
        self.recent = defaultdict(deque)
        self.ema = defaultdict(dict)

    def observe(self, index, visit):
        """Features are formed after the source gate arrives, before its target."""
        position = index % len(self.schedule.order)
        pool = visit.pool
        count = self.meta["pools"][pool]["experts"]
        k = len(visit.demands)
        selected = {expert for expert, _ in visit.demands}
        if any(not 0 <= e < count for e in selected):
            raise ValueError("Routed expert outside physical pool")
        self.stage[pool, visit.loop] = selected
        boundary = self.schedule.boundary[position]
        previous_stage = self.stage.get((pool, self.schedule.target[position]))
        history = self.recent[pool]
        history.appendleft(selected)
        horizon = len(self.schedule.positions[pool])
        while len(history) > horizon:
            history.pop()
        scores = self.ema[pool]
        for expert in scores:
            scores[expert] *= 1 - self.alpha
        ranked = sorted(visit.top_p, key=lambda pair: (-pair[1], pair[0]))
        for expert, score in ranked:
            scores[expert] = scores.get(expert, 0.0) + self.alpha * score
        ranks = {expert: rank for rank, (expert, _) in enumerate(ranked)}
        base, dual, generic, fallback = [], [], [], []
        for expert in range(count):
            rank = ranks.get(expert, 2 * k)
            rank_bin = 0 if rank == 0 else 1 if rank < k else 2 if rank < 2 * k else 3
            ema_bin = 0 if scores.get(expert, 0) == 0 else 1 if scores[expert] < 1 / count else 2
            common = (rank_bin, ema_bin)
            # A is missing across tokens. Last-loop membership is an explicit
            # boundary feature, not silently relabeled as same-token history.
            a = int(expert in selected) if not boundary else -1
            b = int(expert in previous_stage) if previous_stage is not None else -1
            boundary_member = int(expert in selected) if boundary else -1
            base.append(common)
            dual.append((a, b, boundary_member))
            bits = sum(int(expert in ids) << lag for lag, ids in enumerate(history))
            generic.append((bits, len(history), horizon))
            fallback.append((int(expert in selected), int(any(expert in ids for ids in list(history)[1:]))))
        return Features(pool, visit.loop, self.schedule.target[position], boundary,
                        base, dual, generic, fallback, dict(scores), k)


@dataclass
class Features:
    pool: str
    source: int | None
    target: int | None
    boundary: bool
    base: list
    dual: list
    generic: list
    generic_fallback: list
    ema: dict
    k: int

    def keys(self, mode, expert):
        common = self.base[expert]
        a, b, edge = self.dual[expert]
        if mode == "gate_ema":
            feature = ()
        elif mode == "history_a":
            feature = (a, edge)
        elif mode == "history_b":
            feature = (b,)
        elif mode == "generic_lag":
            feature = self.generic_fallback[expert]
        else:
            feature = (a, b, edge)
        pooled = common + feature
        if mode == "generic_lag":
            detail = (self.pool, self.generic[expert], common)
        elif mode == "pooled_ab" or mode == "gate_ema":
            detail = (self.pool, None, pooled)
        elif mode == "boundary_ab":
            detail = (self.pool, self.boundary, pooled)
        else:
            detail = (self.pool, (self.source, self.target), pooled)
        keys = (pooled, (self.pool, pooled), detail)
        return keys[:2] if mode in ("gate_ema", "pooled_ab") else keys


class Predictor:
    """Empirical Bernoulli probabilities with shared -> pool -> stage shrinkage."""

    def __init__(self, mode, config, hot):
        self.mode, self.config, self.hot = mode, config, hot
        self.tables = [defaultdict(lambda: [0, 0]) for _ in range(3)]

    def add(self, features, target):
        for expert in range(len(features.base)):
            label = int(expert in target)
            for table, key in zip(self.tables, features.keys(self.mode, expert)):
                cell = table[key]
                cell[0] += label
                cell[1] += 1  # ALL experts, including outside top-2k and nonresidents.

    def freeze(self):
        cap = self.config["max_table_cells"]
        self.tables = [dict(sorted(table.items(), key=lambda pair: (-pair[1][1], repr(pair[0])))[:cap])
                       for table in self.tables]

    def predict(self, features):
        strength = self.config["shrinkage"]
        result = array("f")
        for expert in range(len(features.base)):
            # Unseen combinations back off to fit-only expert hotness and
            # observed (never future) gate EMA. No evaluation-label updates.
            probability = (self.hot[features.pool, expert] +
                           min(1.0, features.k * features.ema.get(expert, 0.0))) / 2
            for table, key in zip(self.tables, features.keys(self.mode, expert)):
                positive, opportunities = table.get(key, (0, 0))
                if opportunities:
                    probability = (positive + strength * probability) / (opportunities + strength)
            result.append(probability)
        return result


def fit_predictors(meta, fit, schedule, config):
    frequency, opportunities = Counter(), Counter()
    targets = defaultdict(list)
    for request in fit:
        visits = schedule.visits(request)
        for index, visit in enumerate(visits):
            opportunities[visit.pool] += 1
            frequency.update((visit.pool, e) for e, _ in visit.demands)
            target_index = index + schedule.delta[index % len(schedule.order)]
            if target_index < len(visits):
                target = visits[target_index]
                targets[visit.pool, schedule.target[index % len(schedule.order)]].append(
                    (request["request_id"], index, {e for e, _ in target.demands}))
    hot = {(pool, e): frequency[pool, e] / opportunities[pool] if opportunities[pool] else 0
           for pool, data in meta["pools"].items() for e in range(data["experts"])}
    shuffled = {}
    rng = random.Random(config["shuffle_seed"])
    for key in sorted(targets, key=repr):
        examples = targets[key]
        labels = [entry[2] for entry in examples]
        rng.shuffle(labels)
        for (request_id, index, _), label in zip(examples, labels):
            shuffled[request_id, index] = label
    predictors = {mode: Predictor(mode, config, hot) for mode in PREDICTORS}
    for request in fit:
        history = History(schedule, meta, config["score_alpha"])
        visits = schedule.visits(request)
        for index, visit in enumerate(visits):
            features = history.observe(index, visit)
            target_index = index + schedule.delta[index % len(schedule.order)]
            if target_index >= len(visits):
                continue
            target = {e for e, _ in visits[target_index].demands}
            for mode, predictor in predictors.items():
                predictor.add(features, shuffled[request["request_id"], index]
                              if mode == "phase_ab_shuffled" else target)
    for predictor in predictors.values():
        predictor.freeze()
    return predictors, frequency


@dataclass
class Prepared:
    visit: Visit
    target_index: int
    scores: dict
    ema: dict
    protected: frozenset


def prepare(meta, request, schedule, config, predictors):
    history = History(schedule, meta, config["score_alpha"])
    rows = []
    for index, visit in enumerate(schedule.visits(request)):
        features = history.observe(index, visit)
        position = index % len(schedule.order)
        rows.append(Prepared(visit, index + schedule.delta[position],
                             {mode: predictor.predict(features) for mode, predictor in predictors.items()},
                             features.ema,
                             frozenset(e for e, _ in visit.demands) if not schedule.boundary[position]
                             else frozenset()))
    return rows


def allocate(capacity, weights, limits):
    """Deterministic capped proportional allocation, conserving integer slots."""
    quotas = {pool: 0 for pool in limits}
    remaining = min(capacity, sum(limits.values()))
    while remaining:
        active = [pool for pool in limits if quotas[pool] < limits[pool]]
        total = sum(weights[pool] for pool in active)
        shares = {pool: remaining * (weights[pool] / total if total else 1 / len(active)) for pool in active}
        assigned = {pool: min(limits[pool] - quotas[pool], int(shares[pool])) for pool in active}
        if not sum(assigned.values()):
            for pool in sorted(active, key=lambda p: (-shares[p], p))[:remaining]:
                assigned[pool] = 1
        for pool, value in assigned.items():
            quotas[pool] += value
            remaining -= value
    return quotas


@dataclass
class Outcome:
    accesses: int
    misses: int
    reloads: int
    hit_bits: bytearray
    peak_bytes: int
    resident_recall: list
    quota_updates: int


def simulate(meta, rows, schedule, config, policy, capacity, static_keys, audit=False):
    """Same sequential service and admission model for every policy, incl. oracle.

    peak: evict BEFORE transfer; currently computing weights count in the budget.
    resident: legacy bypass semantics; explicitly report the extra staging peak.
    """
    size = next(iter(meta["pools"].values()))["bytes_per_expert"]
    residents = {pool: {} for pool in meta["pools"]}
    heaps = {pool: [] for pool in residents}
    limits = {pool: data["experts"] for pool, data in meta["pools"].items()}
    pending = {pool: positions[0] for pool, positions in schedule.positions.items()}
    priority = {pool: {} for pool in residents}
    frequency = Counter()
    seen = set()
    accesses = misses = reloads = occupancy = peak = quota_updates = 0
    hit_bits, recalls = bytearray(), []
    distance = policy.endswith("_distance")
    predictor = policy.removesuffix("_distance")
    adaptive = policy == "oled_adaptive"
    weights = {pool: 1.0 for pool in residents}
    activation = {pool: 1.0 for pool in residents}
    utilization = {pool: 1.0 for pool in residents}
    quotas = allocate(capacity, weights, limits)
    future = defaultdict(deque)
    if policy == "belady":
        serial = 0
        for row in rows:
            for expert, _ in row.visit.demands:
                future[row.visit.pool, expert].append(serial)
                serial += 1

    def value(pool, expert):
        if policy == "belady":
            return -future[pool, expert][0] if future[pool, expert] else -math.inf
        if policy == "lfu":
            return frequency[pool, expert]
        if policy == "static_hot":
            return int((pool, expert) in static_keys)
        if policy == "lru":
            return 0.0
        return priority[pool].get(expert, 0.0)

    def refresh(pool):
        heaps[pool] = [(value(pool, expert), last, expert) for expert, last in residents[pool].items()]
        heapify(heaps[pool])

    def lowest(pool):
        heap = heaps[pool]
        while heap and residents[pool].get(heap[0][2]) != heap[0][1]:
            heappop(heap)
        if not heap:
            return None
        score, last, expert = heap[0]
        if distance:
            score /= 1 + max(0, pending[pool] - index - 1)
        return score, last, pool, expert

    def evict(pool=None):
        nonlocal occupancy
        candidates = (lowest(pool),) if pool is not None else (lowest(p) for p in residents)
        victim = min(candidate for candidate in candidates if candidate is not None)
        _, _, victim_pool, expert = victim
        del residents[victim_pool][expert]
        occupancy -= 1
        return victim_pool, expert

    for index, row in enumerate(rows):
        visit, pool = row.visit, row.visit.pool
        if predictor in PREDICTORS:
            priority[pool] = dict(enumerate(row.scores[predictor]))
        elif policy in ("score_ema", "score_ema_loop"):
            priority[pool] = {e: score + (config["loop_bonus"] if e in row.protected
                                        and policy == "score_ema_loop" else 0)
                              for e, score in row.ema.items()}
            if policy == "score_ema_loop":
                for e in row.protected:
                    priority[pool].setdefault(e, config["loop_bonus"])
        elif policy == "lru_loop":
            priority[pool] = {e: 1.0 for e in row.protected}
        elif policy in ("oled_gate_only", "oled_adaptive"):
            total = sum(score for _, score in visit.demands)
            priority[pool] = {e: score / total for e, score in visit.demands}
        pending[pool] = row.target_index
        refresh(pool)
        start_residents = set(residents[pool])
        start_hits = len(start_residents & {e for e, _ in visit.demands})
        if adaptive:
            alpha = config["score_alpha"]
            activation[pool] = (1 - alpha) * activation[pool] + alpha * len(visit.demands) / limits[pool]
            # Empty pools have neutral utilization so zero quotas can recover.
            useful = start_hits / len(start_residents) if start_residents else 1.0
            utilization[pool] = (1 - alpha) * utilization[pool] + alpha * useful
        for expert, _ in visit.demands:
            key = pool, expert
            hit = expert in residents[pool]
            hit_bits.append(int(hit))
            accesses += 1
            frequency[key] += 1
            if policy == "belady":
                future[key].popleft()
            if not hit:
                misses += 1
                reloads += key in seen
                seen.add(key)
                if config["memory_mode"] == "peak":
                    if capacity == 0:
                        raise ValueError("Peak budget cannot hold even one computing expert")
                    if occupancy == capacity:
                        # No free staging slot outside the common peak budget.
                        if adaptive and len(residents[pool]) >= quotas[pool] and residents[pool]:
                            evict(pool)
                        else:
                            evict()
                peak = max(peak, (occupancy + 1) * size)
            else:
                peak = max(peak, occupancy * size)
            if not hit and (capacity == 0 or (policy == "static_hot" and key not in static_keys)
                            or (adaptive and quotas[pool] == 0)):
                continue
            if not hit:
                occupancy += 1
            residents[pool][expert] = accesses
            heappush(heaps[pool], (value(pool, expert), accesses, expert))
            if adaptive and len(residents[pool]) > quotas[pool]:
                evict(pool)
            if occupancy > capacity:
                evict()
            if len(heaps[pool]) > 4 * max(1, len(residents[pool])):
                refresh(pool)
        if adaptive and (index + 1) % (len(schedule.order) * config["oled_update_tokens"]) == 0:
            weights = {p: activation[p] * utilization[p] for p in residents}
            quotas = allocate(capacity, weights, limits)
            quota_updates += 1
            for p in residents:
                while len(residents[p]) > quotas[p]:
                    evict(p)
        if audit:
            # One common anchor cache supplies exactly the same resident
            # candidates to every predictor in the diagnostic below.
            recalls.append((index, tuple(residents[pool])))
    return Outcome(accesses, misses, reloads, hit_bits, peak, recalls, quota_updates)


def validate_config(config):
    if not 0 < config["score_alpha"] <= 1 or config["loop_bonus"] < 0:
        raise ValueError("Invalid EMA alpha or loop bonus")
    if config["shrinkage"] <= 0 or config["max_table_cells"] < 1:
        raise ValueError("Positive shrinkage and table capacity required")
    if config["memory_mode"] not in ("peak", "resident"):
        raise ValueError("memory_mode must be peak or resident")
    if not config["budgets_percent"] or any(not 0 <= x <= 100 for x in config["budgets_percent"]):
        raise ValueError("budgets_percent must be nonempty and within [0, 100]")
    if config["oled_update_tokens"] < 1 or config["retention_slots"] < 1:
        raise ValueError("Positive OLED interval and retention_slots required")
    if len(set(config["policies"])) != len(config["policies"]) or not config["policies"]:
        raise ValueError("Unique, nonempty policy list required")
    if set(config["policies"]) - set(POLICIES):
        raise ValueError("Unknown residency policy")


def partition(requests, manifest=None):
    ids = [request["request_id"] for request in requests]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate request IDs")
    if manifest is None:
        fit, evaluation = split_requests(requests)
        validation = []
    else:
        fit_ids = manifest["fit_request_ids"]
        eval_ids = manifest["evaluation_request_ids"]
        validation_ids = manifest.get("validation_request_ids", [])
        groups = [set(fit_ids), set(validation_ids), set(eval_ids)]
        if any(len(group) != len(raw) for group, raw in zip(groups, (fit_ids, validation_ids, eval_ids))):
            raise ValueError("Duplicate IDs in split manifest")
        if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
            raise ValueError("Request leakage between fit/validation/evaluation")
        if set.union(*groups) != set(ids):
            raise ValueError("Split manifest must cover the trace request IDs exactly")
        fit, validation, evaluation = [[r for r in requests if r["request_id"] in group] for group in groups]
    if not fit or not evaluation:
        raise ValueError("Need nonempty, disjoint fitting and evaluation requests")
    return fit, validation, evaluation


def prediction_diagnostics(rows, schedule, request_id, retention_slots, candidates=None, budget=None):
    """Offline metrics only. Source scores and source-time candidates are frozen."""
    totals = defaultdict(Counter)
    source_candidates = candidates if candidates is not None else [(i, None) for i in range(len(rows))]
    for index, eligible in source_candidates:
        row = rows[index]
        if row.target_index >= len(rows):
            continue
        target = {e for e, _ in rows[row.target_index].visit.demands}
        boundary = int(schedule.boundary[index % len(schedule.order)])
        for mode in PREDICTORS:
            scores = row.scores[mode]
            universe = range(len(scores)) if eligible is None else eligible
            count = len(target) if eligible is None else retention_slots
            chosen = sorted(universe, key=lambda e: (-scores[e], e))[:count]
            key = mode, row.visit.pool, row.visit.loop, rows[row.target_index].visit.loop, boundary
            totals[key].update(targets=len(target), hits=len(set(chosen) & target), selected=len(chosen),
                               opportunities=1, eligible_targets=len(target) if eligible is None
                               else len(set(eligible) & target))
    return [{"request_id": request_id, "budget_percent": budget,
             "candidate_set": "all_physical_experts" if candidates is None else "score_ema_residents",
             "predictor": key[0], "pool": key[1], "source_phase": key[2], "target_phase": key[3],
             "boundary": key[4], **counts,
             "topk_recall": counts["hits"] / counts["targets"] if counts["targets"] else None,
             "conditional_recall": counts["hits"] / counts["eligible_targets"]
             if counts["eligible_targets"] else None}
            for key, counts in totals.items()]


def emit_csv(path, rows):
    if rows:
        write_csv(path, list(rows[0]), rows)


def run(meta, requests, config, out_dir, manifest=None):
    validate_config(config)
    sizes = {data["bytes_per_expert"] for data in meta["pools"].values()}
    if len(sizes) != 1 or min(sizes) <= 0:
        raise ValueError("This first experiment requires equal positive expert byte sizes (exact Belady)")
    if any(data["experts"] < 1 for data in meta["pools"].values()):
        raise ValueError("Empty expert pool")
    size = sizes.pop()
    expert_count = sum(data["experts"] for data in meta["pools"].values())
    if config["memory_mode"] == "peak" and any(int(expert_count * p / 100) < 1
                                                 for p in config["budgets_percent"]):
        raise ValueError("Peak budget must hold at least one computing expert; use resident for zero-budget analysis")
    schedule = Schedule.from_trace(meta, requests)
    # Validate structure across ALL requests, before fitting or producing results.
    for request in requests:
        schedule.visits(request)
    fit, validation, evaluation = partition(requests, manifest)
    started = time.monotonic()
    predictors, frequency = fit_predictors(meta, fit, schedule, config)
    hot_rank = sorted(((p, e) for p, data in meta["pools"].items() for e in range(data["experts"])),
                      key=lambda key: (-frequency[key], key))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    policy_rows, position_rows, paired_rows, prediction_rows, phase_rows = [], [], [], [], []
    policies = config["policies"]
    baseline_names = [p for p in policies if p in COMPARATORS]
    print(f"[{meta.get('model', 'trace')}] fit={len(fit)}, validation={len(validation)}, "
          f"evaluation={len(evaluation)}, fitting_seconds={time.monotonic()-started:.1f}", flush=True)
    for number, request in enumerate(evaluation, 1):
        rows = prepare(meta, request, schedule, config, predictors)
        prediction_rows.extend(prediction_diagnostics(rows, schedule, request["request_id"],
                                                       config["retention_slots"]))
        decode_positions = sum(step["phase"] == "decode" for step in request["steps"])
        # Exactly one immutable outcome per request/policy/budget. No tuning.
        for percent in config["budgets_percent"]:
            budget = int(expert_count * size * percent / 100)
            capacity = budget // size
            outcomes = {policy: simulate(meta, rows, schedule, config, policy, capacity,
                                         set(hot_rank[:capacity]), audit=policy == "score_ema")
                        for policy in policies}
            if "score_ema" in outcomes:
                prediction_rows.extend(prediction_diagnostics(
                    rows, schedule, request["request_id"], config["retention_slots"],
                    candidates=outcomes["score_ema"].resident_recall, budget=percent))
            for policy, result in outcomes.items():
                policy_rows.append({"budget_percent": percent, "budget_bytes": budget,
                                    "capacity_experts": capacity, "policy": policy,
                                    "request_id": request["request_id"], "category": request["category"],
                                    "decode_positions": decode_positions, "accesses": result.accesses,
                                    "misses": result.misses, "reloads": result.reloads,
                                    "h2d_bytes": result.misses * size,
                                    "h2d_bytes_per_decode_position": result.misses * size / decode_positions
                                    if decode_positions else None,
                                    "hit_rate_by_bytes": 1 - result.misses / result.accesses if result.accesses else None,
                                    "peak_weight_bytes": result.peak_bytes, "quota_updates": result.quota_updates})
                offset = 0
                positions, phases = defaultdict(Counter), defaultdict(Counter)
                for index, row in enumerate(rows):
                    count = len(row.visit.demands)
                    hits = sum(result.hit_bits[offset:offset + count])
                    offset += count
                    positions[index // len(schedule.order)].update(accesses=count, misses=count - hits)
                    phases[row.visit.pool, row.visit.loop].update(accesses=count, misses=count-hits)
                position_rows.extend({"budget_percent": percent, "policy": policy,
                                      "request_id": request["request_id"], "decode_position": pos, **counts,
                                      "h2d_bytes": counts["misses"] * size} for pos, counts in positions.items())
                phase_rows.extend({"budget_percent": percent, "policy": policy,
                                   "request_id": request["request_id"], "pool": key[0], "target_phase": key[1],
                                   "boundary": int(key[1] == schedule.order[schedule.positions[key[0]][0]][1]),
                                   **counts, "h2d_bytes": counts["misses"] * size}
                                  for key, counts in phases.items())
                if policy in BASELINES:
                    continue
                for baseline in baseline_names:
                    if baseline == policy:
                        continue
                    reference = outcomes[baseline]
                    gain = sum(a > b for a, b in zip(result.hit_bits, reference.hit_bits)) * size
                    loss = sum(a < b for a, b in zip(result.hit_bits, reference.hit_bits)) * size
                    paired_rows.append({"budget_percent": percent, "policy": policy, "baseline": baseline,
                                        "request_id": request["request_id"], "decode_positions": decode_positions,
                                        "gained_hit_bytes": gain, "lost_hit_bytes": loss,
                                        "net_saved_h2d_bytes": gain-loss,
                                        "delta_h2d_bytes_per_decode_position": (loss-gain) / decode_positions
                                        if decode_positions else None,
                                        "delta_hit_rate": (gain-loss) / (result.accesses*size)
                                        if result.accesses else None})
        print(f"[{meta.get('model', 'trace')}] evaluation {number}/{len(evaluation)}: "
              f"{request['request_id']} ({time.monotonic()-started:.1f}s)", flush=True)
    summaries = defaultdict(Counter)
    for row in policy_rows:
        summaries[row["budget_percent"], row["policy"]].update(
            {field: row[field] for field in ("decode_positions", "accesses", "misses", "h2d_bytes")})
    summary_rows = []
    for (percent, policy), counts in summaries.items():
        strongest = min(baseline_names, key=lambda p: (summaries[percent, p]["h2d_bytes"], p)) if baseline_names else None
        reference = summaries[percent, strongest]["h2d_bytes"] if strongest else None
        summary_rows.append({"budget_percent": percent, "policy": policy, **counts,
                             "hit_rate_by_bytes": 1-counts["misses"]/counts["accesses"] if counts["accesses"] else None,
                             "h2d_bytes_per_decode_position": counts["h2d_bytes"]/counts["decode_positions"]
                             if counts["decode_positions"] else None,
                             "strongest_online_baseline": strongest,
                             "saved_h2d_fraction_vs_strongest": 1-counts["h2d_bytes"]/reference if reference else None})
    paired_summary = []
    groups = defaultdict(list)
    for row in paired_rows:
        groups[row["budget_percent"], row["policy"], row["baseline"]].append(row)
    for key, group in groups.items():
        differences = [row["delta_h2d_bytes_per_decode_position"] for row in group
                       if row["delta_h2d_bytes_per_decode_position"] is not None]
        paired_summary.append({"budget_percent": key[0], "policy": key[1], "baseline": key[2],
                               "requests": len(group),
                               "gained_hit_bytes": sum(row["gained_hit_bytes"] for row in group),
                               "lost_hit_bytes": sum(row["lost_hit_bytes"] for row in group),
                               "net_saved_h2d_bytes": sum(row["net_saved_h2d_bytes"] for row in group),
                               "mean_delta_h2d_bytes_per_position": statistics.mean(differences) if differences else None,
                               "std_delta_h2d_bytes_per_position": statistics.stdev(differences) if len(differences)>1 else 0,
                               "requests_improved": sum(row["net_saved_h2d_bytes"] > 0 for row in group),
                               "requests_worsened": sum(row["net_saved_h2d_bytes"] < 0 for row in group)})
    for name, records in (("residency_requests", policy_rows), ("residency_summary", summary_rows),
                          ("residency_positions", position_rows), ("residency_phases", phase_rows),
                          ("paired_requests", paired_rows), ("paired_summary", paired_summary),
                          ("prediction_requests", prediction_rows)):
        emit_csv(out_dir / f"{name}.csv", records)
    details = {**config, "model": meta.get("model"), "python": platform.python_version(),
               "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                 for name in ("residency.py", "replay.py", "trace.py")},
               "online_comparators": baseline_names, "trace_revision": meta.get("revision"),
               "fit_request_ids": [r["request_id"] for r in fit],
               "validation_request_ids": [r["request_id"] for r in validation],
               "evaluation_request_ids": [r["request_id"] for r in evaluation],
               "expert_bytes": size, "expert_count": expert_count,
               "execution_order": schedule.order,
               "fitted_table_cells": {mode: [len(table) for table in predictor.tables]
                                      for mode, predictor in predictors.items()},
               "wall_seconds": time.monotonic()-started,
               "evaluation_role": "development: previously inspected pilot requests" if manifest is None
               else "explicit request split; validation held out, no automatic parameter search",
               "service_order": "expert ID ascending; current router observed before service",
               "resident_audit_anchor": "score_ema; identical source-time candidates for every predictor",
               "strongest_baseline_selection": "retrospective aggregate evaluation comparator, not parameter tuning",
               "oled_adaptive": "retention adaptation only; EMA activation * utilization; capped largest-remainder quotas; "
                                "neutral empty-pool utilization; no CPU compensation",
               "scope": "projected decode trace, conditional continuation, cold start per request; "
                        "no end-to-end speed or scheduling overhead measurement"}
    (out_dir / "residency_config.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    # Store the fit artifacts for inspection; evaluation never mutates these tables.
    (out_dir / "predictors.json").write_text(json.dumps({
        mode: {"expert_hotness": [[pool, e, probability] for (pool, e), probability in predictor.hot.items()],
               "tables": [[{"key": key, "selected": value[0], "opportunities": value[1]}
                            for key, value in table.items()] for table in predictor.tables]}
        for mode, predictor in predictors.items()}, separators=(",", ":")) + "\n", encoding="utf-8")
    return summary_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/residency.json"))
    parser.add_argument("--split", type=Path, help="Exact fit/validation/evaluation request-ID manifest")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    config.setdefault("policies", list(POLICIES))
    summary = run(*read_trace(args.trace), config, args.out_dir, read_json(args.split) if args.split else None)
    digest = hashlib.sha256(args.trace.read_bytes()).hexdigest()
    metadata_path = args.out_dir / "residency_config.json"
    metadata = read_json(metadata_path)
    metadata["trace_sha256"] = digest
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(summary)} policy/budget rows to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
