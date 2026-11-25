"""CPU unit tests for Probe 3 (Ouro-Thinking Depth & Floor-Protected Exit)."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from loop_research.probe3_thinking_depth import (
    calibrate_floor_threshold,
    compute_kl_div,
)


class TestProbe3DepthLogic(unittest.TestCase):
    def test_kl_div_properties(self) -> None:
        p = torch.tensor([[0.7, 0.2, 0.1], [0.5, 0.5, 0.0]])
        q = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        kl = compute_kl_div(p, q)
        self.assertAlmostEqual(float(kl[0].item()), 0.0, places=5)
        self.assertGreater(float(kl[1].item()), 0.1)

    def test_calibrate_floor_threshold_rmin2(self) -> None:
        rng = np.random.default_rng(42)
        vals_kl = {2: rng.uniform(0.0, 2.0, size=1000), 3: rng.uniform(0.0, 1.0, size=1000)}
        tau, mr, probs = calibrate_floor_threshold(vals_kl, mode="kl", r_min=2, n_loops=4, target_mean_r=2.6)
        self.assertGreater(tau, 0.0)
        self.assertAlmostEqual(probs[0], 0.0, places=6)  # R=1 exit strictly forbidden!
        self.assertAlmostEqual(mr, 2.6, delta=0.05)

        vals_gate = {1: rng.uniform(0.0, 0.6, size=1000), 2: rng.uniform(0.2, 0.9, size=1000), 3: rng.uniform(0.5, 1.0, size=1000)}
        tau_g2, mr_g2, p_g2 = calibrate_floor_threshold(vals_gate, mode="gate", r_min=2, n_loops=4, target_mean_r=2.7)
        self.assertAlmostEqual(p_g2[0], 0.0, places=6)
        self.assertAlmostEqual(mr_g2, 2.7, delta=0.05)


if __name__ == "__main__":
    unittest.main()
