"""CPU unit tests for Probe 1 (Looped MoE Quantization & Routing-Flip Cascade)."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from loop_research.probe1_loopmoe_quant import (
    compute_loop_calibration_shift,
    fake_quantize_asym_group,
    select_topk_experts_mask,
)


class TestProbe1QuantLogic(unittest.TestCase):
    def test_fake_quant_bits_monotonic_error(self) -> None:
        torch.manual_seed(0)
        w = torch.randn(4, 256, dtype=torch.bfloat16)
        w16 = fake_quantize_asym_group(w, bits=16, group_size=128)
        self.assertTrue(torch.equal(w, w16))
        e4 = (w.float() - fake_quantize_asym_group(w, 4, 128).float()).abs().mean().item()
        e3 = (w.float() - fake_quantize_asym_group(w, 3, 128).float()).abs().mean().item()
        e2 = (w.float() - fake_quantize_asym_group(w, 2, 128).float()).abs().mean().item()
        e2_g64 = (w.float() - fake_quantize_asym_group(w, 2, 64).float()).abs().mean().item()
        self.assertLess(e4, e3)
        self.assertLess(e3, e2)
        self.assertLessEqual(e2_g64, e2)

    def test_fake_quant_non_multiple_dim(self) -> None:
        w = torch.randn(3, 150, dtype=torch.float32)
        q = fake_quantize_asym_group(w, bits=3, group_size=64)
        self.assertEqual(q.shape, w.shape)
        self.assertTrue(torch.isfinite(q).all())

    def test_select_topk_mask_exact_counts(self) -> None:
        scores = np.arange(12, dtype=np.float64).reshape(3, 4)
        m_glob = select_topk_experts_mask(scores, 5, per_layer_equal=False)
        self.assertEqual(int(m_glob.sum()), 5)
        self.assertTrue(m_glob[2, 3])
        m_bal = select_topk_experts_mask(scores, 5, per_layer_equal=True)
        self.assertEqual(int(m_bal.sum()), 5)
        self.assertEqual(sorted(m_bal.sum(axis=1).tolist()), [1, 2, 2])

    def test_loop_calibration_shift_metrics(self) -> None:
        rms = np.ones((2, 4, 3, 8), dtype=np.float64)
        rms[0, 3, 0, :] = np.array([10.0, 1, 1, 1, 1, 1, 1, 1], dtype=np.float64)
        counts = np.full((2, 4, 3), 20, dtype=np.int64)
        counts[0, 0, 1] = 0  # expert (0,1) blind at loop 0
        out = compute_loop_calibration_shift(rms, counts, min_tokens=16)
        self.assertGreater(out["loop0_blind_frac"], 0.0)
        self.assertLess(out["mean_first_last_rms_cos"], 1.0)


if __name__ == "__main__":
    unittest.main()
