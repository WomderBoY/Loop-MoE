"""Unit tests for Stage B offline co-simulation (strict peak budget, causality, prefetch accounting)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import random
import tempfile
import unittest

from loop_moe.ifm_b2_offline_replay import run_b2_offline_replay


def _make_synthetic_trace(path: Path, n_reqs: int = 12, n_steps: int = 8) -> None:
    rng = random.Random(7)
    pools = {
        "layer:00": {"layer_id": 0, "region": "prelude", "experts": 8, "bytes_per_expert": 1024},
        "layer:01": {"layer_id": 1, "region": "recurrent", "experts": 8, "bytes_per_expert": 1024},
        "layer:02": {"layer_id": 2, "region": "recurrent", "experts": 8, "bytes_per_expert": 1024},
        "layer:03": {"layer_id": 3, "region": "coda", "experts": 8, "bytes_per_expert": 1024},
    }
    order = [
        ("layer:00", None),
        ("layer:01", 0),
        ("layer:02", 0),
        ("layer:01", 1),
        ("layer:02", 1),
        ("layer:01", 2),
        ("layer:02", 2),
        ("layer:03", None),
    ]
    req_ids = [f"req-{i:03d}" for i in range(n_reqs)]
    meta = {
        "schema_version": 2,
        "model": "Synthetic-Huginn",
        "request_count": n_reqs,
        "top_k": 2,
        "pools": pools,
        "execution_order": order,
        "splits": {
            "fit": req_ids[:4],
            "val": req_ids[4:6],
            "eval": req_ids[6:],
        },
    }
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")
        for i, rid in enumerate(req_ids):
            steps = []
            for _s in range(n_steps):
                visits = []
                prev_e = rng.randrange(4)
                for pool, loop in order:
                    # Structured stage-conditioned transition so combined_loop has clear signal
                    if loop is None:
                        e0 = (prev_e + 1) % 8
                    else:
                        e0 = (prev_e + loop + 1) % 8
                    e1 = (e0 + 1) % 8
                    prev_e = e0
                    cands = [(e0 + k) % 8 for k in range(4)]
                    visits.append(
                        [
                            [e0, e1],
                            [0.6, 0.4],
                            [0.6, 0.4],
                            cands,
                            [0.6, 0.4, 0.1, 0.05],
                            [2.0, 1.5, 0.2, 0.1],
                            1.3,
                            0.3,
                            0.4,
                            0.8 if loop and loop > 0 else None,
                            0.5 if loop and loop > 0 else None,
                        ]
                    )
                steps.append({"visits": visits})
            f.write(json.dumps({"request_id": rid, "category": ["text", "math", "code"][i % 3], "steps": steps}) + "\n")


class TestStageBOfflineReplay(unittest.TestCase):
    def test_peak_budget_and_2x2_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tpath = Path(tmp) / "trace.jsonl.gz"
            out_dir = Path(tmp) / "out"
            _make_synthetic_trace(tpath)
            cfg = {
                "a2": {"score_alpha": 0.2, "shrinkage": 4.0, "max_table_cells": 2048, "shuffle_seed": 42},
                "b2_offline": {"budgets_experts": [8, 16, 32]},
            }
            hl = run_b2_offline_replay(
                tpath,
                out_dir,
                cfg,
                h2d_ms=1.0,
                visit_comp_ms=0.6,
                regime_name="unit_test",
                workers=2,
            )
            self.assertEqual(hl["eval_requests"], 6)
            self.assertTrue((out_dir / "b2_offline_summary_unit_test.csv").is_file())
            self.assertTrue((out_dir / "b2_offline_paired_unit_test.csv").is_file())

            # Verify strict peak byte budget invariant across every single request×budget×policy row!
            import csv

            rows = list(csv.DictReader((out_dir / "b2_offline_requests_unit_test.csv").open()))
            self.assertGreater(len(rows), 0)
            for r in rows:
                cap = int(r["budget_experts"])
                peak = int(r["peak_bytes"])
                self.assertLessEqual(peak, cap * 1024, f"Peak budget exceeded: {r}")
                # Timely + late + wasted == prefetch_issued
                self.assertEqual(
                    int(r["prefetch_issued"]),
                    int(r["prefetch_timely"]) + int(r["prefetch_late"]) + int(r["prefetch_wasted"]),
                )


if __name__ == "__main__":
    unittest.main()
