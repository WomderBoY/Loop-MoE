"""Summarize Foil route overlap, router statistics and the decode working set."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from html import escape
from math import log
from pathlib import Path

from .trace import read_trace, write_csv


@dataclass
class Overlap:
    count: int = 0
    covered: float = 0.0
    jaccard: float = 0.0
    cosine: float = 0.0
    distribution: Counter = field(default_factory=Counter)
    source_counts: Counter = field(default_factory=Counter)
    target_counts: Counter = field(default_factory=Counter)

    def add(self, source, target, cosine=None):
        source_ids, target_ids = set(source[0]), set(target[0])
        intersection = len(source_ids & target_ids)
        self.count += 1
        self.covered += intersection / len(target_ids)
        self.jaccard += intersection / len(source_ids | target_ids)
        self.distribution[intersection] += 1
        self.source_counts.update(source_ids)
        self.target_counts.update(target_ids)
        if cosine is not None:
            self.cosine += cosine

    def summary(self):
        mean_k = sum(self.target_counts.values()) / self.count
        independent = sum(count * self.target_counts[expert] for expert, count in self.source_counts.items())
        return {
            "positions": self.count,
            "overlap": self.covered / self.count,
            "jaccard": self.jaccard / self.count,
            "marginal_null_overlap": independent / (self.count ** 2 * mean_k),
            "intersection_counts": ";".join(f"{k}={v}" for k, v in sorted(self.distribution.items())),
        }


CONFIDENCE_BINS = {
    "margin": (0.01, 0.025, 0.05, 0.1, 0.2),
    "normalized_entropy": (0.2, 0.4, 0.6, 0.8, 0.9),
}


def accumulate_routes(meta, request, overlaps, confidence, streaks):
    previous_decode = {}
    category = request["category"]
    for step in request["steps"]:
        phase = step["phase"]
        previous_loop = {}
        current_decode = {}
        active_streaks = defaultdict(dict)
        for visit in step["visits"]:
            pool, loop = visit["pool"], visit["loop"]
            if loop is None:
                continue
            for position, route in enumerate(visit["routes"], start=step["first_position"]):
                if phase == "decode":
                    current_decode[pool, loop] = route
                    prior = previous_decode.get((pool, loop))
                    if prior is not None:
                        overlaps[phase, category, pool, "previous_decode_position", loop, loop].add(prior, route)
                key = pool, position
                prior = previous_loop.get(key)
                if prior is not None:
                    overlaps[phase, category, pool, "adjacent_loop", loop - 1, loop].add(prior, route, route[5])
                    values = (prior[3], prior[4] / log(meta["pools"][pool]["experts"]))
                    for signal, value in zip(CONFIDENCE_BINS, values):
                        bucket = bisect_right(CONFIDENCE_BINS[signal], value)
                        confidence[phase, category, pool, loop, signal, bucket].add(prior, route)
                previous_loop[key] = route

                active = active_streaks[key]
                current = {expert: active.get(expert, 0) + 1 for expert in route[0]}
                streaks[phase, category, pool].update(active[expert] for expert in active.keys() - current.keys())
                active_streaks[key] = current
        for (pool, _), active in active_streaks.items():
            streaks[phase, category, pool].update(active.values())
        previous_decode = current_decode


def working_set(meta, request, distances):
    stack = []  # Least recent at index 0; at most one entry per physical expert.
    for step in request["steps"]:
        if step["phase"] != "decode":
            continue
        for visit in step["visits"]:
            for expert in sorted(visit["routes"][0][0]):
                key = visit["pool"], expert
                if key in stack:
                    index = stack.index(key)
                    distance = len(stack) - index - 1
                    stack.pop(index)
                else:
                    distance = "cold"
                distances[request["category"], distance] += 1
                stack.append(key)
    return {
        "request_id": request["request_id"],
        "category": request["category"],
        "prompt_tokens": len(request["prompt_ids"]),
        "output_tokens": len(request["generated_ids"]),
        "decode_positions": len(request["steps"]) - 1,
        "unique_decode_experts": len(stack),
        "decode_working_set_bytes": sum(meta["pools"][pool]["bytes_per_expert"] for pool, _ in stack),
    }


def write_heatmap(out_dir, rows):
    groups = defaultdict(dict)
    for row in rows:
        if row["comparison"] == "adjacent_loop":
            groups[row["phase"], row["category"]][row["pool"], row["target_loop"]] = row
    with (out_dir / "routing_heatmap.html").open("w", encoding="utf-8") as stream:
        stream.write('''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>相邻循环路由重合</title><style>
body{font:16px system-ui,sans-serif;margin:2rem;color:#182433} table{border-collapse:collapse;margin-bottom:2rem}
td,th{padding:.35rem .6rem;text-align:center;border:1px solid #cdd6df;white-space:nowrap}
.table{overflow-x:auto} td{background:hsl(208 67% calc(100% - 50% * var(--v)));color:#101920}
</style><h1>相邻循环路由重合</h1>''')
        for (phase, category), cells in sorted(groups.items()):
            stream.write(f"<h2>{escape(phase)} / {escape(category)}</h2>")
            pools = sorted({pool for pool, _ in cells}, key=lambda pool: int(pool.split(":")[1]))
            loops = sorted({loop for _, loop in cells})
            for metric, label in (("overlap", "Overlap"), ("jaccard", "Jaccard")):
                stream.write(f'<div class="table"><table><caption>{label}</caption><thead><tr><th>物理层</th>')
                stream.write("".join(f"<th>{loop - 1}→{loop}</th>" for loop in loops))
                stream.write("</tr></thead><tbody>")
                for pool in pools:
                    stream.write(f'<tr><th scope="row">{escape(pool)}</th>')
                    for loop in loops:
                        row = cells.get((pool, loop))
                        if row is None:
                            stream.write('<td style="--v:0">—</td>')
                        else:
                            value = row[metric]
                            stream.write(f'<td style="--v:{value:.4f}" title="n={row["positions"]}">{value:.2f}</td>')
                    stream.write("</tr>")
                stream.write("</tbody></table></div>")
        stream.write("</html>\n")


def analyze(meta, requests, out_dir):
    overlaps, confidence = defaultdict(Overlap), defaultdict(Overlap)
    streaks = defaultdict(Counter)
    distances = Counter()
    request_rows = []
    for request in requests:
        accumulate_routes(meta, request, overlaps, confidence, streaks)
        request_rows.append(working_set(meta, request, distances))

    summary_columns = ["positions", "overlap", "jaccard", "marginal_null_overlap", "intersection_counts"]
    route_columns = ["phase", "category", "pool", "comparison", "source_loop", "target_loop"]
    route_rows = [
        dict(zip(route_columns, key), **stats.summary(),
             mean_residual_cosine=stats.cosine / stats.count if key[3] == "adjacent_loop" else None)
        for key, stats in sorted(overlaps.items())
    ]
    write_csv(out_dir / "routing.csv", route_columns + summary_columns + ["mean_residual_cosine"], route_rows)
    write_heatmap(out_dir, route_rows)
    confidence_columns = ["phase", "category", "pool", "target_loop", "signal", "bin"]
    confidence_rows = (
        dict(zip(confidence_columns, key), **stats.summary()) for key, stats in sorted(confidence.items())
    )
    write_csv(out_dir / "confidence.csv", confidence_columns + summary_columns, confidence_rows)
    write_csv(out_dir / "requests.csv", list(request_rows[0]), request_rows)
    write_csv(out_dir / "reuse_distance.csv", ["category", "distinct_experts_since_last_use", "accesses"], (
        {"category": category, "distinct_experts_since_last_use": distance, "accesses": count}
        for (category, distance), count in sorted(distances.items(), key=lambda pair: (pair[0][0], str(pair[0][1])))
    ))
    write_csv(out_dir / "reuse_lengths.csv", ["phase", "category", "pool", "consecutive_loops", "runs"], (
        {"phase": phase, "category": category, "pool": pool, "consecutive_loops": length, "runs": count}
        for (phase, category, pool), lengths in sorted(streaks.items()) for length, count in sorted(lengths.items())
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    meta, requests = read_trace(args.trace)
    analyze(meta, requests, args.out_dir)
    print(args.out_dir)


if __name__ == "__main__":
    main()
