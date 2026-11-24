"""Pure-CPU unit tests for IFM A1-A3 prompts, A2 predictor decomposition, and A3 residency replay."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import random
import tempfile
import unittest

from loop_moe.ifm_a2_predict import run_a2
from loop_moe.ifm_a3_replay import run_a3
from loop_moe.ifm_prompts import (
    evaluate_mbpp_code,
    format_ifm_prompt,
    grade_gsm8k,
    ngram_repetition_rate,
    split_requests_three_way,
)
from loop_moe.ifm_report import generate_report
from loop_moe.trace import write_line


def _make_synthetic_ifm_trace(path: Path, *, arch: str, n_requests: int = 6, n_steps: int = 8) -> dict:
    rng = random.Random(42 if arch == "transformer" else 99)
    n_experts = 25
    bpe = 22413312
    pools = {}
    if arch == "transformer":
        for i in range(28):
            pools[f"layer:{i:02d}"] = {"layer_id": i, "region": "recurrent", "experts": n_experts, "bytes_per_expert": bpe}
        order = [(f"layer:{i:02d}", lp) for lp in range(4) for i in range(28)]
        model_name = "IFM-Ouro-500B"
    else:
        for i in range(28):
            reg = "prelude" if i < 8 else "recurrent" if i < 20 else "coda"
            pools[f"layer:{i:02d}"] = {"layer_id": i, "region": reg, "experts": n_experts, "bytes_per_expert": bpe}
        order = (
            [(f"layer:{i:02d}", None) for i in range(8)]
            + [(f"layer:{i:02d}", lp) for lp in range(8) for i in range(8, 20)]
            + [(f"layer:{i:02d}", None) for i in range(20, 28)]
        )
        model_name = "IFM-Huginn-500B"

    assert len(order) == 112
    cats = ["text", "math", "code"]
    req_ids = []
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        write_line(
            stream,
            {
                "schema_version": 2,
                "trace_scope": "ifm_a1_new_positions",
                "model": model_name,
                "repo": f"IFM/{model_name}",
                "arch": arch,
                "pools": pools,
                "execution_order": [list(x) for x in order],
                "top_k": 2,
                "top_candidates": 6,
                "request_count": n_requests,
            },
        )
        for r_idx in range(n_requests):
            rid = f"req-{r_idx:02d}"
            req_ids.append(rid)
            cat = cats[r_idx % 3]
            steps = []
            for s_idx in range(n_steps):
                visits = []
                prev_by_pool = {}
                for pool, loop in order:
                    layer_id = int(pool.split(":")[1])
                    if loop is not None and pool in prev_by_pool:
                        prev_ids, prev_margin = prev_by_pool[pool]
                        # 60% retain top-1, 40% deterministic A->(A+1)%25 replacement
                        if rng.random() < 0.6:
                            e0 = prev_ids[0]
                            e1 = prev_ids[1]
                            linf = 0.25 * prev_margin  # strictly < margin/2 -> no flip!
                            cos = 0.96
                        else:
                            e0 = (prev_ids[0] + 1) % n_experts
                            e1 = (prev_ids[1] + 2) % n_experts
                            if e1 == e0:
                                e1 = (e0 + 1) % n_experts
                            linf = 2.0 * prev_margin
                            cos = 0.72
                    else:
                        e0 = (layer_id + (loop or 0)) % n_experts
                        e1 = (e0 + 3 + (r_idx % 2)) % n_experts
                        linf = None
                        cos = None
                    margin = 0.4 + 0.1 * (layer_id % 3)
                    cands = [e0, e1] + [e for e in range(n_experts) if e not in (e0, e1)][:4]
                    cand_logits = [2.0, 1.5, 1.5 - margin, 0.8, 0.5, 0.2]
                    cand_scores = [0.88, 0.82, 0.70, 0.65, 0.60, 0.55]
                    visits.append(
                        [
                            [e0, e1],
                            [0.55, 0.45],
                            [0.88, 0.82],
                            cands,
                            cand_scores,
                            cand_logits,
                            margin,
                            0.12,
                            0.75,
                            cos,
                            linf,
                        ]
                    )
                    if loop is not None:
                        prev_by_pool[pool] = ([e0, e1], margin)
                steps.append({"step": s_idx, "token_pos": 20 + s_idx, "visits": visits})
            write_line(
                stream,
                {
                    "request_id": rid,
                    "category": cat,
                    "mode": "freegen",
                    "prompt_len": 20,
                    "generated_ids": list(range(10, 10 + n_steps)),
                    "generated_text": "synthetic output",
                    "stop_reason": "max_new_tokens",
                    "quality": {"eos": False, "truncated": True, "rep4_rate": 0.0, "output_tokens": n_steps},
                    "steps": steps,
                },
            )
    return {
        "fit_request_ids": req_ids[:3],
        "val_request_ids": [],
        "eval_request_ids": req_ids[3:],
    }


class TestIFMA1A3(unittest.TestCase):
    def test_shim_inspect_compat(self) -> None:
        import inspect
        import sys
        import types

        # Verify dunder-safe __getattr__ on dynamic shim modules (matches ifm_runtime.py)
        mod = types.ModuleType("_dummy_shim_ops")
        mod.__file__ = __file__

        def _ga(name: str):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return lambda *a, **kw: None

        mod.__getattr__ = _ga  # type: ignore[attr-defined]
        sys.modules["_dummy_shim_ops"] = mod
        try:
            self.assertEqual(inspect.getsourcefile(mod), __file__)
            self.assertIsNotNone(inspect.getmodule(TestIFMA1A3))
        finally:
            sys.modules.pop("_dummy_shim_ops", None)

    def test_prompts_and_graders(self) -> None:
        math_prompt, stops = format_ifm_prompt({"id": "m1", "category": "math", "question": "What is 2+2?"})
        self.assertIn("Q: What is 2+2?\nA:", math_prompt)
        self.assertIn("\n\nQ:", stops)
        ok, pred = grade_gsm8k("Let's think step by step. 2 + 2 = 4. The answer is 4.", "4")
        self.assertTrue(ok)
        self.assertEqual(pred, "4")

        code_prompt, cstops = format_ifm_prompt(
            {"id": "c1", "category": "code", "text": "Return x+1", "test_list": ["assert f(1) == 2"]}
        )
        self.assertIn("[BEGIN]", code_prompt)
        self.assertEqual(cstops, ["[DONE]"])
        passed, msg = evaluate_mbpp_code("def f(x):\n    return x + 1\n[DONE]\nextra", ["assert f(1) == 2"])
        self.assertTrue(passed, msg)

        self.assertAlmostEqual(ngram_repetition_rate([1, 2, 3, 4, 1, 2, 3, 4], n=4), 1.0 / 5.0)
        split = split_requests_three_way(
            [{"id": f"r{i}", "category": ["text", "math", "code"][i % 3]} for i in range(18)]
        )
        self.assertEqual(len(split["fit_request_ids"]), 6)
        self.assertEqual(len(split["val_request_ids"]), 3)
        self.assertEqual(len(split["eval_request_ids"]), 9)
        self.assertEqual(
            len(set(split["fit_request_ids"]) | set(split["val_request_ids"]) | set(split["eval_request_ids"])), 18
        )

    def test_a2_a3_and_report_invariants(self) -> None:
        a2_cfg = {"score_alpha": 0.2, "shrinkage": 8.0, "max_table_cells": 4096, "shuffle_seed": 7, "candidate_ks": [2, 4, 6], "bootstrap_samples": 50}
        a3_cfg = {
            "budgets_experts": [95, 191, 287, 300, 302, 383, 700],
            "memory_mode": "peak",
            "score_alpha": 0.2,
            "oled_update_tokens": 5,
            "non_expert_gib": 1.75,
            "kv_kib_per_token": 168.0,
            "bootstrap_samples": 50,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for model_name, arch in (("IFM-Ouro-500B", "transformer"), ("IFM-Huginn-500B", "huginn")):
                m_dir = root / model_name
                m_dir.mkdir(parents=True)
                for mode in ("freegen", "fixed"):
                    trace = m_dir / f"trace_{mode}.jsonl.gz"
                    n_req = 12 if mode == "freegen" else 6  # >=8 eval reqs exercises ProcessPoolExecutor pickling!
                    manifest = _make_synthetic_ifm_trace(trace, arch=arch, n_requests=n_req, n_steps=5)
                    h2 = run_a2(trace, manifest, a2_cfg, m_dir / f"a2_{mode}")
                    self.assertEqual(h2["suff_cond_violations"], 0)
                    self.assertGreater(h2["suff_cond_total"], 0)
                    h3 = run_a3(trace, manifest, a2_cfg, a3_cfg, m_dir / f"a3_{mode}")
                    self.assertIn("302", h3["focus"])
                    self.assertIn("loop_hier_distance_hit_rate", h3["focus"]["302"])
                    self.assertIn("loop_rem_distance_hit_rate", h3["focus"]["302"])
                    self.assertGreaterEqual(h3["focus"]["302"]["belady_hit_rate"], h3["focus"]["302"]["loop_hier_distance_hit_rate"])
                (m_dir / "a1_quality_summary.json").write_text(
                    json.dumps(
                        {
                            "model": model_name,
                            "expected_visits_per_pos": 112,
                            "mean_output_tokens": 6.0,
                            "mean_ms_per_step": 50.0,
                            "eos_rate": 0.0,
                            "stop_string_rate": 0.5,
                            "truncation_rate": 0.5,
                            "mean_rep4_rate": 0.0,
                            "math_acc": 0.5,
                            "math_n": 2,
                            "code_pass1": 0.5,
                            "code_n": 2,
                        }
                    ),
                    encoding="utf-8",
                )
            cfg = {"mode": "unit-test", "models": ["IFM-Ouro-500B", "IFM-Huginn-500B"], "a3": a3_cfg}
            metrics = generate_report(cfg, root)
            self.assertIn("ouro_hit_6p3g_loop", metrics)
            self.assertIn("huginn_hit_6p3g_pin", metrics)
            self.assertTrue((root / "report_a1_a3.md").is_file())


if __name__ == "__main__":
    unittest.main()
