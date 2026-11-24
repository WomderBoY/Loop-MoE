"""Lightweight unit tests for Idea 1 pure-Python analysis logic."""

from __future__ import annotations

import numpy as np

from loop_research.idea1_kv_depth import (
    COMBOS_2_OF_8,
    analyze_qa_probe,
    spearman_corr,
    split_prefix_into_blocks,
)


def test_split_prefix_and_combos():
    spans = split_prefix_into_blocks(515, 8)
    assert len(spans) == 8
    assert spans[0][0] == 0 and spans[-1][1] == 515
    for i in range(7):
        assert spans[i][1] == spans[i + 1][0]
    assert len(COMBOS_2_OF_8) == 28
    assert abs(spearman_corr([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-6


def test_analyze_qa_probe_synthetic_sanity():
    records = []
    for doc_i in range(8):
        for chain, ev in [("A", [0, 1]), ("B", [6, 7])]:
            ev_idx = COMBOS_2_OF_8.index(tuple(ev))
            combo_nlls = [1.8] * 28
            combo_nlls[ev_idx] = 0.6
            combo_ems = [0.2] * 28
            combo_ems[ev_idx] = 1.0
            feats = [
                {
                    "block": float(b),
                    "pos_norm": b / 7.0,
                    "entropy_mean": 1.0 + 0.1 * b,
                    "entropy_max": 2.0,
                    "kl_1_0_mean": 0.2,
                    "state_delta_mean": 0.3,
                    "exit_continue_score": 0.5,
                    "h1_norm_mean": 10.0,
                    "nll1_mean": 2.0,
                    "curr_nll_gain_1_2": 0.1,
                }
                for b in range(8)
            ]
            records.append(
                {
                    "qa_id": f"d{doc_i}_{chain}",
                    "doc_id": f"d{doc_i}",
                    "base_id": doc_i // 2,
                    "variant": "orig" if doc_i % 2 == 0 else "swapped",
                    "which_chain": chain,
                    "evidence_blocks": ev,
                    "nll_shallow": 2.0,
                    "nll_deep": 0.5,
                    "combo_nlls": combo_nlls,
                    "em_shallow": 0.0,
                    "em_deep": 1.0,
                    "combo_ems": combo_ems,
                    "f1_shallow": 0.0,
                    "f1_deep": 1.0,
                    "combo_f1s": combo_ems,
                    "oracle_combo_idx": ev_idx,
                    "oracle_blocks": ev,
                    "causal_block_features": feats,
                }
            )
    out = analyze_qa_probe(records, probe_name="unit_test")
    assert abs(out["policies"]["oracle_2"]["nll_recovery"] - (1.4 / 1.5)) < 1e-6
    assert abs(out["paired_control"]["oracle_overlap_own_evidence"] - 1.0) < 1e-6
    assert abs(out["paired_control"]["oracle_overlap_other_evidence"] - 0.0) < 1e-6
