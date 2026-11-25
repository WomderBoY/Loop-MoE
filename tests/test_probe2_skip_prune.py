"""CPU unit tests for Probe 2 (Looped MoE Dynamic Skipping & Stage Pruning)."""

from __future__ import annotations

import unittest

import numpy as np

from loop_research.probe2_loopmoe_skip_prune import (
    build_alive_expert_mask,
    calibrate_drop_threshold,
    stage_id_for,
    stage_names_for,
)


class TestProbe2SkipPruneLogic(unittest.TestCase):
    def test_stage_mapping_ouro_and_huginn(self) -> None:
        self.assertEqual(stage_names_for(28, 0, 28, 4), ["loop_0", "loop_1", "loop_2", "loop_3"])
        self.assertEqual(stage_id_for(5, 2, 0, 28, 4), 2)
        h_names = stage_names_for(28, 8, 20, 8)
        self.assertEqual(len(h_names), 10)
        self.assertEqual(h_names[0], "prelude")
        self.assertEqual(h_names[-1], "coda")
        self.assertEqual(stage_id_for(3, 0, 8, 20, 8), 0)
        self.assertEqual(stage_id_for(12, 0, 8, 20, 8), 1)
        self.assertEqual(stage_id_for(12, 7, 8, 20, 8), 8)
        self.assertEqual(stage_id_for(25, 0, 8, 20, 8), 9)

    def test_build_alive_expert_mask_guarantees_min_alive(self) -> None:
        scores = np.zeros((4, 5), dtype=np.float64)
        scores[0] = [100, 90, 80, 70, 60]
        scores[1] = [1, 2, 3, 4, 5]
        scores[2] = [0.1, 0.2, 0.3, 0.4, 0.5]
        scores[3] = [10, 20, 30, 40, 50]
        alive = build_alive_expert_mask(scores, prune_frac=0.50, min_alive_per_layer=2)
        self.assertEqual(int((~alive).sum()), 10)
        self.assertTrue((alive.sum(axis=1) >= 2).all())
        # Layer 2's top-2 (indices 3, 4) must survive despite tiny global scores
        self.assertTrue(alive[2, 3] and alive[2, 4])

    def test_calibrate_drop_threshold(self) -> None:
        arr = np.linspace(0.0, 1.0, 1001)
        tau = calibrate_drop_threshold([arr], 0.5)
        self.assertAlmostEqual(tau, 0.5, places=2)


if __name__ == "__main__":
    unittest.main()
