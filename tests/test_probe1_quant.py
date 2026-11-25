"""CPU unit tests for Probe 1 (Looped MoE Quantization & Routing-Flip Cascade)."""

from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from loop_research.probe1_loopmoe_quant import (
    IFMQuantManager,
    collect_calibration_stats,
    compute_loop_calibration_shift,
    eval_nll_with_trace,
    fake_quantize_asym_group,
    select_topk_experts_mask,
)


class _DummyRouter(nn.Module):
    def __init__(self, dim: int, n_experts: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_experts, dim) * 0.1)
        self.bias = None
        self.num_experts = n_experts
        self.topk = 2
        self.score_func = "sigmoid"
        self.scaling_factor = 1.0

    def forward(self, x: torch.Tensor, load_balancing_type=None):
        scores = F.linear(x, self.weight).float().sigmoid()
        idx = scores.topk(self.topk, dim=-1).indices
        w = scores.gather(-1, idx)
        w = w / w.sum(-1, keepdim=True)
        tpe = torch.bincount(idx.flatten(), minlength=self.num_experts)
        return w.to(x.dtype), idx, tpe, None


class _DummyExperts(nn.Module):
    def __init__(self, n_experts: int, dim: int, inter: int) -> None:
        super().__init__()
        self.n_local_experts = n_experts
        self.model_dim = dim
        self.inter_dim = inter
        self.weight1 = nn.Parameter(torch.randn(n_experts * inter, dim) * 0.05)
        self.weight2 = nn.Parameter(torch.randn(n_experts * dim, inter) * 0.05)
        self.weight3 = nn.Parameter(torch.randn(n_experts * inter, dim) * 0.05)

    def forward(self, x: torch.Tensor, group_sizes: list[int]) -> torch.Tensor:
        w1 = self.weight1.view(self.n_local_experts, self.inter_dim, self.model_dim)
        w2 = self.weight2.view(self.n_local_experts, self.model_dim, self.inter_dim)
        w3 = self.weight3.view(self.n_local_experts, self.inter_dim, self.model_dim)
        outs = []
        off = 0
        for e, cnt in enumerate(group_sizes):
            if cnt == 0:
                continue
            xe = x[off : off + cnt]
            h = F.silu(xe @ w1[e].T) * (xe @ w3[e].T)
            outs.append(h @ w2[e].T)
            off += cnt
        return torch.cat(outs, dim=0) if outs else x.new_zeros((0, self.model_dim))


class _DummyIFMModel(nn.Module):
    def __init__(self, vocab: int = 32, dim: int = 16, inter: int = 32, n_layers: int = 3, n_experts: int = 5, n_loops: int = 2) -> None:
        super().__init__()
        self.loop_start_layer = 1
        self.loop_end_layers = 2
        self.loop_times = n_loops
        self.emb = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = nn.Module()
            layer.moe = nn.Module()
            layer.moe.router = _DummyRouter(dim, n_experts)
            layer.moe.experts = _DummyExperts(n_experts, dim, inter)
            self.layers.append(layer)

    def forward(self, inp: torch.Tensor, multi_segments: bool = False, cache=None):
        h = self.emb(inp)
        order = [0] + [1] * self.loop_times + [2]
        for lid in order:
            w, idx, tpe, _ = self.layers[lid].moe.router(h, None)
            gs_list = [int(v) for v in tpe.tolist()]
            x_rep = h.reshape(-1, h.shape[-1]).repeat_interleave(2, dim=0)
            y_exp = self.layers[lid].moe.experts(x_rep, gs_list)
            y_comb = (y_exp.reshape(*idx.shape, -1) * w.unsqueeze(-1)).sum(dim=-2)
            h = h + y_comb
        return self.head(h), None, None


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
        counts[0, 0, 1] = 0
        out = compute_loop_calibration_shift(rms, counts, min_tokens=16)
        self.assertGreater(out["loop0_blind_frac"], 0.0)
        self.assertLess(out["mean_first_last_rms_cos"], 1.0)

    def test_end_to_end_mock_ifm_quant_and_counterfactual(self) -> None:
        torch.manual_seed(0)
        m = _DummyIFMModel()
        qm = IFMQuantManager(m)
        seqs = [{"tokens": [1, 2, 3, 4, 5, 6], "domain": "text"}, {"tokens": [6, 5, 4, 3, 2, 1], "domain": "code"}]
        cal = collect_calibration_stats(m, seqs, torch.device("cpu"))
        self.assertEqual(cal["counts"].shape, (3, 2, 5))
        bf16 = eval_nll_with_trace(m, seqs, torch.device("cpu"), mode="record")
        qm.apply_expert_bits(np.full((3, 5), 2, dtype=np.int32), group_size=8)
        q_ev = eval_nll_with_trace(m, seqs, torch.device("cpu"), mode="compare", ref_traces=bf16["recorded_traces"])
        cf_ev = eval_nll_with_trace(m, seqs, torch.device("cpu"), mode="replay", ref_traces=bf16["recorded_traces"])
        self.assertTrue(np.isfinite(q_ev["nll"]) and np.isfinite(cf_ev["nll"]))
        qm.restore_all()


if __name__ == "__main__":
    unittest.main()
