"""Unit & mock end-to-end CPU tests for Round 2-B (LoopQuant-MoE)."""

from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from loop_research.r2b_loopquant_moe import (
    LoopQuantManager,
    awq_quantize_expert_bank,
    build_visit_topology,
    calibrate_trajectory_quant,
    compute_expert_vram_gb,
    eval_nll_batched,
    rank_to_trilevel_bits,
    solve_trilevel_bit_allocation,
)


class _MockIFMExperts(nn.Module):
    def __init__(self, n_experts: int, dim: int, inter: int):
        super().__init__()
        self.n_local_experts = n_experts
        self.n_experts = n_experts
        self.model_dim = dim
        self.inter_dim = inter
        self.weight1 = nn.Parameter(torch.randn(n_experts * inter, dim) * 0.2)
        self.weight2 = nn.Parameter(torch.randn(n_experts * dim, inter) * 0.2)
        self.weight3 = nn.Parameter(torch.randn(n_experts * inter, dim) * 0.2)

    def forward(self, x_perm: torch.Tensor, group_sizes: list[int]) -> torch.Tensor:
        w1 = self.weight1.view(self.n_local_experts, self.inter_dim, self.model_dim)
        w2 = self.weight2.view(self.n_local_experts, self.model_dim, self.inter_dim)
        w3 = self.weight3.view(self.n_local_experts, self.inter_dim, self.model_dim)
        outs = []
        off = 0
        for e, cnt in enumerate(group_sizes):
            if cnt > 0:
                xe = x_perm[off : off + cnt]
                h = F.silu(F.linear(xe, w1[e])) * F.linear(xe, w3[e])
                outs.append(F.linear(h, w2[e]))
                off += cnt
        return torch.cat(outs, dim=0) if outs else torch.empty_like(x_perm)


class _MockRouter(nn.Module):
    def __init__(self, dim: int, n_experts: int):
        super().__init__()
        self.topk = 2
        self.num_experts = n_experts
        self.scaling_factor = 1.0
        self.weight = nn.Parameter(torch.randn(n_experts, dim) * 0.2)

    def forward(self, x: torch.Tensor, load_balancing_type=None):
        probs = F.linear(x, self.weight).softmax(-1)
        vals, idx = probs.topk(2, dim=-1)
        weights = vals / vals.sum(-1, keepdim=True).clamp_min(1e-12)
        tpe = torch.bincount(idx.reshape(-1), minlength=self.num_experts)
        return weights, idx, tpe, None


class _MockMoE(nn.Module):
    def __init__(self, dim: int, inter: int, n_experts: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, inter, bias=False)
        self.fc2 = nn.Linear(inter, dim, bias=False)
        self.fc3 = nn.Linear(dim, inter, bias=False)
        self.router = _MockRouter(dim, n_experts)
        self.experts = _MockIFMExperts(n_experts, dim, inter)

    def residual(self, y: torch.Tensor, residual: torch.Tensor, c=None):
        return residual + y

    def forward(self, x_in: torch.Tensor, load_balancing_type=None):
        residual = x_in
        x = self.norm(x_in)
        y1 = self.fc2(F.silu(self.fc1(x)) * self.fc3(x))
        w, idx, _tpe, aux = self.router(x, load_balancing_type)
        bsz, slen, dim = x.shape
        x_2d = x.reshape(bsz * slen, dim)
        tok = torch.arange(bsz * slen, device=x.device).unsqueeze(1).expand(bsz * slen, 2).reshape(-1)
        exp = idx.reshape(-1)
        wf = w.reshape(-1)
        order = torch.argsort(exp)
        gs = torch.bincount(exp.index_select(0, order), minlength=self.experts.n_local_experts).tolist()
        yp = self.experts(x_2d.index_select(0, tok.index_select(0, order)), gs)
        y2 = torch.zeros_like(x_2d)
        y2.index_add_(0, tok.index_select(0, order), yp * wf.index_select(0, order).unsqueeze(-1))
        return self.residual(y1 + y2.reshape(bsz, slen, dim), residual, c=y1), aux


class _MockLayer(nn.Module):
    def __init__(self, dim: int, inter: int, n_experts: int):
        super().__init__()
        self.moe = _MockMoE(dim, inter, n_experts)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out, _ = self.moe(h, None)
        return out


class _MockLoopedModel(nn.Module):
    def __init__(self, vocab: int = 32, dim: int = 16, inter: int = 32, n_layers: int = 4, loop_start: int = 1, loop_end: int = 3, n_loops: int = 3, n_experts: int = 6):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList([_MockLayer(dim, inter, n_experts) for _ in range(n_layers)])
        self.head = nn.Linear(dim, vocab, bias=False)
        self.loop_start_layer = loop_start
        self.loop_end_layers = loop_end
        self.loop_times = n_loops

    def forward(self, inp: torch.Tensor, multi_segments: bool = False, cache=None):
        h = self.emb(inp)
        for lid in range(0, self.loop_start_layer):
            h = self.layers[lid](h)
        for _r in range(self.loop_times):
            for lid in range(self.loop_start_layer, self.loop_end_layers):
                h = self.layers[lid](h)
        for lid in range(self.loop_end_layers, len(self.layers)):
            h = self.layers[lid](h)
        return self.head(h), None, None


class TestR2BLoopQuant(unittest.TestCase):
    def test_awq_reduces_weighted_mse_under_salient_channels(self):
        torch.manual_seed(0)
        E, Out, In = 4, 32, 64
        w = torch.randn(E, Out, In) * 0.2
        # Make 4 channels 50x more salient in activations
        rms = torch.ones(E, In)
        rms[:, :4] = 50.0
        _, err_rtn = awq_quantize_expert_bank(w, rms, rms, bits=2, group_size=64, alphas=(0.0,))
        _, err_awq = awq_quantize_expert_bank(w, rms, rms, bits=2, group_size=64, alphas=(0.0, 0.25, 0.5, 0.75, 1.0))
        self.assertTrue(bool((err_awq < err_rtn * 0.5).all().item()))

    def test_trilevel_bit_allocator_exact_budget_and_optimality(self):
        # 2 layers x 4 experts = 8 experts
        rng = np.random.default_rng(42)
        d2 = rng.uniform(1.0, 10.0, size=(2, 4))
        d3 = d2 * 0.2
        d4 = d2 * 0.05
        dist = np.stack([d2, d3, d4], axis=-1)
        for tgt in (2.25, 2.50, 3.00):
            bm = solve_trilevel_bit_allocation(dist, tgt)
            self.assertAlmostEqual(float(bm.mean()), tgt, places=6)
            self.assertTrue(set(np.unique(bm).tolist()).issubset({2, 3, 4}))
            # Verify rank allocator hits exact budget too
            rm = rank_to_trilevel_bits(d2, tgt)
            self.assertAlmostEqual(float(rm.mean()), tgt, places=6)

    def test_end_to_end_mock_calib_and_quant(self):
        torch.manual_seed(11)
        model = _MockLoopedModel()
        model.eval()
        topo = build_visit_topology(len(model.layers), model.loop_start_layer, model.loop_end_layers, model.loop_times)
        seqs = [{"tokens": torch.randint(1, 30, (16,)).tolist(), "domain": "text" if i % 2 == 0 else "code"} for i in range(10)]
        dev = torch.device("cpu")
        calib = calibrate_trajectory_quant(model, topo, seqs[:8], dev)
        qm = LoopQuantManager(model, topo, calib)
        dists = qm.precompute_distortion_tables(awq_mode="stagesens", group_size=16, device=dev)
        self.assertEqual(dists["stagesens_norm"].shape, (4, 6, 3))
        bm = solve_trilevel_bit_allocation(dists["stagesens_norm"], 2.50)
        avg_b = qm.apply_quantization(bm, awq_mode="stagesens", group_size=16)
        self.assertAlmostEqual(avg_b, 2.50, places=6)
        res = eval_nll_batched(model, seqs, dev, batch_size=4)
        self.assertTrue(np.isfinite(res["nll"]))
        qm.restore_all()
        vram = compute_expert_vram_gb(bm, 16, 32, group_size=16)
        self.assertGreater(vram["compression_ratio"], 3.0)


if __name__ == "__main__":
    unittest.main()
