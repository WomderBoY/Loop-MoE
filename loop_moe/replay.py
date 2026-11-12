"""Compare equal-byte expert caches on the projected single-position decode trace."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

from .trace import read_json, read_trace, write_csv


POLICIES = ("lru", "lfu", "static_hot", "lru_loop", "oled_gate_only", "score_ema", "score_ema_loop", "belady")


@dataclass(frozen=True)
class Visit:
    pool: str
    loop: int | None
    demands: tuple
    top_p: tuple


def decode_visits(request):
    for step in request["steps"]:
        if step["phase"] != "decode":
            continue
        for visit in step["visits"]:
            (route,) = visit["routes"]
            yield Visit(visit["pool"], visit["loop"], tuple(sorted(zip(route[0], route[1]))), tuple(route[6]))


def split_requests(requests):
    """Use disjoint request IDs, stratified by prompt category when possible."""
    categories = defaultdict(list)
    for request in requests:
        categories[request["category"]].append(request)
    fit, evaluation = [], []
    for group in categories.values():
        group.sort(key=lambda request: hashlib.sha256(request["request_id"].encode()).digest())
        fit_count = max(1, len(group) // 3) if len(group) > 1 else 0
        fit.extend(group[:fit_count])
        evaluation.extend(group[fit_count:])
    return fit, evaluation


def future_accesses(visits):
    future = defaultdict(deque)
    index = 0
    for visit in visits:
        for expert, _ in visit.demands:
            future[visit.pool, expert].append(index)
            index += 1
    return future


def simulate(visits, policy, capacity, static_keys, alpha, bonus, loop_count):
    cache = {}  # Physical (pool, expert) -> last use index.
    seen = set()
    frequency = Counter()
    gate_priority = {}
    ema = defaultdict(dict)
    protected = {}
    future = future_accesses(visits) if policy == "belady" else None
    accesses = misses = reloads = 0

    def priority(key):
        pool, expert = key
        if policy == "belady":
            upcoming = future[key]
            return (-upcoming[0] if upcoming else float("-inf"), cache[key])
        if policy == "lfu":
            return frequency[key], cache[key]
        if policy == "oled_gate_only":
            return gate_priority.get(pool, {}).get(expert, 0.0), cache[key]
        if policy == "lru_loop":
            return expert in protected.get(pool, ()), cache[key]
        if policy in ("score_ema", "score_ema_loop"):
            score = ema[pool].get(expert, 0.0)
            if policy == "score_ema_loop" and expert in protected.get(pool, ()):
                score += bonus
            return score, cache[key]
        return (cache[key],)

    for visit in visits:
        for expert, _ in visit.demands:
            key = visit.pool, expert
            accesses += 1
            if policy == "lfu":
                frequency[key] += 1
            if future is not None:
                future[key].popleft()
            if key not in cache:
                misses += 1
                reloads += key in seen
                seen.add(key)
                if capacity == 0 or (policy == "static_hot" and key not in static_keys):
                    continue
            cache[key] = accesses
            if len(cache) > capacity:
                del cache[min(cache, key=priority)]

        if policy == "oled_gate_only":
            score_total = sum(score for _, score in visit.demands)
            gate_priority[visit.pool] = {expert: score / score_total for expert, score in visit.demands}
        if policy in ("score_ema", "score_ema_loop"):
            scores = ema[visit.pool]
            for expert in scores:
                scores[expert] *= 1 - alpha
            for expert, score in visit.top_p:
                scores[expert] = scores.get(expert, 0.0) + alpha * score
        if policy in ("lru_loop", "score_ema_loop") and visit.loop is not None:
            protected[visit.pool] = {expert for expert, _ in visit.demands} if visit.loop + 1 < loop_count else set()
    return accesses, misses, reloads


def replay(meta, requests, config, out_dir):
    expert_sizes = {pool["bytes_per_expert"] for pool in meta["pools"].values()}
    if len(expert_sizes) != 1:
        raise ValueError("This replay requires uniform expert size for a shared byte budget and exact Belady")
    (expert_size,) = expert_sizes
    total_experts = sum(pool["experts"] for pool in meta["pools"].values())
    fit, evaluation = split_requests(requests)
    frequency = Counter((visit.pool, expert) for request in fit for visit in decode_visits(request)
                        for expert, _ in visit.demands)
    expert_keys = ((pool, expert) for pool, data in meta["pools"].items() for expert in range(data["experts"]))
    hot_rank = sorted(expert_keys, key=lambda key: (-frequency[key], key))
    policies = POLICIES if frequency else tuple(policy for policy in POLICIES if policy != "static_hot")
    eval_visits = [(request, list(decode_visits(request))) for request in evaluation]
    alpha, bonus = config["score_alpha"], config["loop_bonus"]
    if not 0 < alpha <= 1 or not bonus >= 0:
        raise ValueError("Expected 0 < score_alpha <= 1 and loop_bonus >= 0")
    if not config["budgets_percent"]:
        raise ValueError("At least one cache budget is required")

    rows = []
    for percent in config["budgets_percent"]:
        if not 0 <= percent <= 100:
            raise ValueError("Cache budget percentage must be between 0 and 100")
        budget_bytes = int(total_experts * expert_size * percent / 100)
        capacity = budget_bytes // expert_size
        static_keys = set(hot_rank[:capacity])
        for policy in policies:
            for request, visits in eval_visits:
                accesses, misses, reloads = simulate(visits, policy, capacity, static_keys, alpha, bonus, meta["loop_count"])
                rows.append({
                    "budget_percent": percent,
                    "budget_bytes": budget_bytes,
                    "capacity_experts": capacity,
                    "policy": policy,
                    "request_id": request["request_id"],
                    "category": request["category"],
                    "decode_positions": len(request["steps"]) - 1,
                    "accesses": accesses,
                    "misses": misses,
                    "reloads": reloads,
                    "h2d_bytes": misses * expert_size,
                    "hit_rate_by_bytes": (accesses - misses) / accesses if accesses else None,
                })
    write_csv(out_dir / "replay_requests.csv", list(rows[0]), rows)
    sums = defaultdict(Counter)
    fields = ("decode_positions", "accesses", "misses", "reloads", "h2d_bytes")
    for row in rows:
        key = row["budget_percent"], row["budget_bytes"], row["capacity_experts"], row["policy"]
        sums[key].update({field: row[field] for field in fields})
    columns = ["budget_percent", "budget_bytes", "capacity_experts", "policy"]
    summary = (dict(zip(columns, key), **totals,
                    hit_rate_by_bytes=(1 - totals["misses"] / totals["accesses"]) if totals["accesses"] else None,
                    h2d_bytes_per_decode_position=totals["h2d_bytes"] / totals["decode_positions"]
                    if totals["decode_positions"] else None)
               for key, totals in sums.items())
    write_csv(out_dir / "replay_summary.csv", columns + list(fields) +
              ["hit_rate_by_bytes", "h2d_bytes_per_decode_position"], summary)
    (out_dir / "replay_config.json").write_text(json.dumps({
        "fit_request_ids": [request["request_id"] for request in fit],
        "evaluation_request_ids": [request["request_id"] for request in evaluation],
        "budgets_percent": config["budgets_percent"], "score_alpha": alpha, "loop_bonus": bonus,
        "expert_bytes": expert_size, "expert_count": total_experts, "policies": policies,
    }, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/foil_pilot.json"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    meta, requests = read_trace(args.trace)
    replay(meta, requests, read_json(args.config), args.out_dir)
    print(args.out_dir)


if __name__ == "__main__":
    main()
