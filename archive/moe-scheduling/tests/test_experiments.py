import csv
import gzip
import itertools
import tempfile
import unittest
from functools import lru_cache
from pathlib import Path

from loop_moe.analyze import analyze
from loop_moe.replay import POLICIES, Visit, replay, simulate, split_requests
from loop_moe.trace import SCHEMA_VERSION, TRACE_SCOPE, read_trace, write_line


def make_request(request_id, first, second):
    def step(phase, index, selections):
        return {
            "phase": phase, "first_position": index,
            "visits": [
                {"pool": "loop:0", "loop": loop, "routes": [[
                    list(ids), [0.45, 0.3], [0.6, 0.4], 0.15, 1.0, 0.95 if loop else None,
                    list(zip(ids, [0.45, 0.3])),
                ]]}
                for loop, ids in enumerate(selections)
            ],
        }

    return {
        "request_id": request_id, "category": "text", "prompt_ids": [10], "generated_ids": [11, 12, 13],
        "steps": [step("prefill", 0, first), step("decode", 1, first), step("decode", 2, second)],
    }


class ExperimentTests(unittest.TestCase):
    def test_belady_matches_exhaustive_cache_with_bypass(self):
        for sequence in itertools.product(range(3), repeat=5):
            visits = [Visit("loop:0", 0, ((expert, 1.0),), ((expert, 1.0),)) for expert in sequence]
            for capacity in (0, 1, 2, 3):
                @lru_cache(None)
                def optimal(index, resident):
                    if index == len(sequence):
                        return 0
                    expert = sequence[index]
                    if expert in resident:
                        return optimal(index + 1, resident)
                    candidates = resident | {expert}
                    keep = min(capacity, len(candidates))
                    return 1 + min(optimal(index + 1, frozenset(retained))
                                   for retained in itertools.combinations(candidates, keep))

                _, misses, _ = simulate(visits, "belady", capacity, set(), 0.2, 0.1, 2)
                self.assertEqual(misses, optimal(0, frozenset()), (sequence, capacity))

    def test_zero_loop_bonus_matches_score_ema(self):
        visits = [Visit("loop:0", step % 2, ((expert, 1.0),), ((expert, 1.0),))
                  for step, expert in enumerate((1, 2, 2, 0, 1, 1, 2, 2, 0))]
        baseline = simulate(visits, "score_ema", 2, set(), 0.2, 0, 2)
        self.assertEqual(simulate(visits, "score_ema_loop", 2, set(), 0.2, 0, 2), baseline)

    def test_trace_to_analysis_and_same_budget_replay(self):
        meta = {
            "schema_version": SCHEMA_VERSION, "trace_scope": TRACE_SCOPE, "request_count": 2,
            "loop_count": 2, "pools": {"loop:0": {"experts": 4, "bytes_per_expert": 100}},
        }
        first, second = [(0, 1), (1, 2)], [(1, 2), (2, 3)]
        requests = [make_request(request_id, first, second) for request_id in ("a", "b")]
        fit, evaluation = split_requests(requests)
        self.assertEqual(len(fit), 1)
        self.assertNotEqual(fit[0]["request_id"], evaluation[0]["request_id"])

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            trace_path = root / "trace.jsonl.gz"
            with gzip.open(trace_path, "wt", encoding="utf-8") as stream:
                for record in [meta, *requests]:
                    write_line(stream, record)
            loaded_meta, loaded_requests = read_trace(trace_path)
            analyze(loaded_meta, loaded_requests, root)
            replay(loaded_meta, loaded_requests, {
                "budgets_percent": [0, 50, 100], "score_alpha": 0.2, "loop_bonus": 0.1,
            }, root)
            with (root / "routing.csv").open() as stream:
                row = next(row for row in csv.DictReader(stream)
                           if row["phase"] == "decode" and row["comparison"] == "adjacent_loop")
            self.assertEqual(int(row["positions"]), 4)
            self.assertAlmostEqual(float(row["overlap"]), 0.5)
            self.assertAlmostEqual(float(row["jaccard"]), 1 / 3)
            self.assertAlmostEqual(float(row["marginal_null_overlap"]), 0.5)
            self.assertAlmostEqual(float(row["mean_residual_cosine"]), 0.95)
            heatmap = (root / "routing_heatmap.html").read_text(encoding="utf-8")
            self.assertIn('title="n=4">0.50</td>', heatmap)
            with (root / "replay_summary.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), len(POLICIES) * 3)
            for row in rows:
                self.assertEqual(int(row["accesses"]), 8)
                if row["budget_percent"] == "0":
                    self.assertEqual(int(row["h2d_bytes"]), 800)
                if row["budget_percent"] == "100":
                    self.assertEqual(int(row["misses"]), 4)


if __name__ == "__main__":
    unittest.main()
