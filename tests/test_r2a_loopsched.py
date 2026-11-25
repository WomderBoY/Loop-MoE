"""Unit & mock end-to-end CPU tests for Round 2-A (LoopSched-MoE)."""

from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from loop_research.r2a_loopsched_moe import (
    LoopSchedExecutor,
    build_visit_topology,
    calibrate_dynamic_threshold,
    calibrate_loopsched,
    compute_optimal_gamma,
    eval_mcq_batched,
    eval_nll_batched,
    prepare_mcq_encoded,
    solve_trilevel_dp_schedule,
)


class _MockExperts(nn.Module):
    def __init__(self, n_experts: int, dim: int):
        super().__init__()
        self.n_local_experts = n_experts
        self.n_experts = n_experts
        self.weights = nn.Parameter(torch.randn(n_experts, dim, dim) * 0.2)

    def forward(self, x_perm: torch.Tensor, group_sizes: list[int]) -> torch.Tensor:
        outs = []
        off = 0
        for e, cnt in enumerate(group_sizes):
            if cnt > 0:
                outs.append(F.linear(x_perm[off : off + cnt], self.weights[e]))
                off += cnt
        return torch.cat(outs, dim=0) if outs else torch.empty_like(x_perm)


class _MockRouter(nn.Module):
    def __init__(self, dim: int, n_experts: int, topk: int = 2):
        super().__init__()
        self.topk = topk
        self.num_experts = n_experts
        self.scaling_factor = 1.0
        self.proj = nn.Linear(dim, n_experts, bias=False)

    def forward(self, x: torch.Tensor, load_balancing_type=None):
        logits = self.proj(x)
        probs = logits.softmax(dim=-1)
        vals, idx = probs.topk(self.topk, dim=-1)
        weights = vals / vals.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        tpe = torch.bincount(idx.reshape(-1), minlength=self.num_experts)
        return weights, idx, tpe, None


class _MockMoE(nn.Module):
    def __init__(self, dim: int, n_experts: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.fc2 = nn.Linear(dim, dim, bias=False)
        self.fc3 = nn.Linear(dim, dim, bias=False)
        self.router = _MockRouter(dim, n_experts, topk=2)
        self.experts = _MockExperts(n_experts, dim)

    def residual(self, y: torch.Tensor, residual: torch.Tensor, c=None) -> torch.Tensor:
        return residual + y

    def forward(self, x_in: torch.Tensor, load_balancing_type=None):
        residual = x_in
        x = self.norm(x_in)
        y1 = self.fc2(F.silu(self.fc1(x)) * self.fc3(x))
        w, idx, _tpe, aux = self.router(x, load_balancing_type)
        bsz, slen, dim = x.shape
        x_2d = x.reshape(bsz * slen, dim)
        w_2d = w.reshape(bsz * slen, -1)
        idx_2d = idx.reshape(bsz * slen, -1)
        tok_ids = torch.arange(bsz * slen, device=x.device).unsqueeze(1).expand_as(idx_2d).reshape(-1)
        exp_ids = idx_2d.reshape(-1)
        order = torch.argsort(exp_ids)
        gs = torch.bincount(exp_ids.index_select(0, order), minlength=self.experts.n_local_experts).tolist()
        yp = self.experts(x_2d.index_select(0, tok_ids.index_select(0, order)), gs)
        y2 = torch.zeros_like(x_2d)
        y2.index_add_(0, tok_ids.index_select(0, order), yp * w_2d.reshape(-1).index_select(0, order).unsqueeze(-1))
        return self.residual(y1 + y2.reshape(bsz, slen, dim), residual, c=y1), aux


class _MockLayer(nn.Module):
    def __init__(self, dim: int, n_experts: int):
        super().__init__()
        self.moe = _MockMoE(dim, n_experts)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        attn = F.scaled_dot_product_attention(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), is_causal=True).squeeze(1)
        h = h + self.proj(attn) * 0.1
        h, _ = self.moe(h, None)
        return h


class _MockLoopedModel(nn.Module):
    def __init__(self, vocab: int = 32, dim: int = 16, n_layers: int = 4, loop_start: int = 1, loop_end: int = 1, n_loops: int = 3, n_experts: int = 6):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList([_MockLayer(dim, n_experts) for _ in range(n_layers)])
        self.head = nn.Linear(dim, vocab, bias=False)
        self.loop_start_layer = loop_start
        self.loop_end_layers = loop_end
        self.loop_times = n_loops

    def forward(self, inp: torch.Tensor, multi_segments: bool = False, cache=None):
        h = self.emb(inp)
        for lid in range(0, self.loop_start_layer):
            h = self.layers[lid](h)
        for _r in range(self.loop_times):
            for lid in range(self.loop_start_layer, len(self.layers) - self.loop_end_layers):
                h = self.layers[lid](h)
        for lid in range(len(self.layers) - self.loop_end_layers, len(self.layers)):
            h = self.layers[lid](h)
        return self.head(h), None, None


class _MockTok:
    def encode(self, text: str, bos: bool = True, eos: bool = False) -> list[int]:
        ids = [(ord(ch) % 28) + 2 for ch in text[:18]]
        return ([1] + ids) if bos else ids


class TestR2ALoopSched(unittest.TestCase):
    def test_dp_knapsack_exact_optimality(self):
        # 4 visits: middle visits 1 & 2 cheap for k=0; boundary visits 0 & 3 expensive
        c1 = np.array([0.05, 0.01, 0.01, 0.30])
        c0 = np.array([5.00, 0.03, 0.02, 4.00])
        # Target sum(k) = 4 (50% cut out of 8 slots): best should be [2, 0, 0, 2] cost 0.05 (vs [1,1,1,1] cost 0.37!)
        sched = solve_trilevel_dp_schedule(c1, c0, target_total_k=4, allow_k0=True)
        self.assertEqual(sched, [2, 0, 0, 2])
        self.assertEqual(sum(sched), 4)

    def test_closed_form_optimal_gamma_orthogonal_vs_identical(self):
        # Orthogonal experts -> dot = 0 -> gamma* = 1.0 (NoRen!)
        self.assertAlmostEqual(compute_optimal_gamma(10.0, 0.0), 1.0, places=6)
        # Identical unit experts with w1=0.6, w2=0.4 -> gamma* = 1 + (0.24)/(0.36) = 1.667 -> clamped 1.60
        self.assertAlmostEqual(compute_optimal_gamma(0.36, 0.24), 1.60, places=6)
        # Mild correlation (rho=0.2, w1=0.7, w2=0.3) -> gamma* = 1 + 0.2*0.3/0.7 = 1.0857
        self.assertAlmostEqual(compute_optimal_gamma(0.49, 0.2 * 0.21), 1.0 + 0.042 / 0.49, places=5)

    def test_executor_parity_and_end_to_end_pipeline(self):
        torch.manual_seed(7)
        model = _MockLoopedModel()
        model.eval()
        topo = build_visit_topology(len(model.layers), model.loop_start_layer, model.loop_end_layers, model.loop_times)
        V = len(topo)
        self.assertEqual(V, 1 + 2 * 3 + 1)  # prelude + 2 rec * 3 loops + coda = 8

        # Verify verify_k2 exact parity vs native
        inp = torch.randint(1, 30, (3, 12))
        ref = model(inp)[0]
        ex_v = LoopSchedExecutor(model, topo, mode="verify_k2")
        with ex_v:
            ex_v.reset_pass()
            got = model(inp)[0]
        self.assertLess(float((ref - got).abs().max().item()), 1e-5)

        # Verify calib + static + dynamic + batched MCQ parity vs B=1
        seqs = [{"tokens": torch.randint(1, 30, (16,)).tolist(), "domain": "text" if i % 2 == 0 else "code"} for i in range(10)]
        calib = calibrate_loopsched(model, topo, seqs[:8], torch.device("cpu"))
        self.assertEqual(len(calib["visit_gamma"]), V)
        self.assertEqual(len(calib["sens_k0"]), V)

        sched50 = solve_trilevel_dp_schedule(calib["sens_k1_opt"], calib["sens_k0"], target_total_k=V, allow_k0=True)
        ex_s = LoopSchedExecutor(model, topo, mode="static", visit_k=sched50, visit_gamma=calib["visit_gamma"])
        res_s = eval_nll_batched(model, seqs, ex_s, torch.device("cpu"), batch_size=4)
        self.assertAlmostEqual(res_s["cut_pct"], 50.0, delta=0.1)

        w_loop = np.ones(V, dtype=np.float64)
        tau = calibrate_dynamic_threshold(calib["visit_ratios"], w_loop, target_drop_frac=0.30)
        ex_d = LoopSchedExecutor(model, topo, mode="dynamic", visit_k=[2] * V, visit_gamma=calib["visit_gamma"], visit_weights=w_loop, dyn_threshold=tau)
        res_d = eval_nll_batched(model, seqs, ex_d, torch.device("cpu"), batch_size=4)
        self.assertGreater(res_d["cut_pct"], 10.0)
        self.assertLess(res_d["cut_pct"], 50.0)

        # Batched MCQ exact parity vs B=1
        tok = _MockTok()
        items = [
            {"id": f"q{i}", "question": f"What is item {i}?" + " extra" * (i % 3), "choice_labels": ["A", "B", "C"], "choice_texts": ["cat", "doggy", "elephant"], "gold_idx": i % 3}
            for i in range(6)
        ]
        enc = prepare_mcq_encoded(items, tok)
        ex_n = LoopSchedExecutor(model, topo, mode="native")
        acc_b1, fl_b1 = eval_mcq_batched(model, enc, ex_n, torch.device("cpu"), batch_size=1)
        acc_b4, fl_b4 = eval_mcq_batched(model, enc, ex_n, torch.device("cpu"), batch_size=4)
        self.assertEqual(fl_b1, fl_b4)
        self.assertAlmostEqual(acc_b1, acc_b4, places=6)


if __name__ == "__main__":
    unittest.main()
