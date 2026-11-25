"""CPU unit tests for Round 3: LoopSched-MoE v2 (Zero GPU Required)."""

from __future__ import annotations

import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from loop_research.r3_loopsched_v2 import (
    LoopSchedV2Executor,
    apply_w3_pool_awq_inplace,
    build_natural_boundary_k12_schedule,
    build_stage_only_cut38_schedule,
    build_visit_topology,
    compute_factorized_importance,
    solve_iter_greedy_k012,
    solve_pure_k12_schedule,
)


class _DummyExperts(nn.Module):
    def __init__(self, n_experts: int, dim: int, inter: int):
        super().__init__()
        self.n_local_experts = n_experts
        self.model_dim = dim
        self.inter_dim = inter
        self.weight1 = nn.Parameter(torch.randn(n_experts * inter, dim) * 0.1)
        self.weight2 = nn.Parameter(torch.randn(n_experts * dim, inter) * 0.1)
        self.weight3 = nn.Parameter(torch.randn(n_experts * inter, dim) * 0.1)

    def forward(self, x_perm: torch.Tensor, group_sizes: list[int]) -> torch.Tensor:
        w1 = self.weight1.view(self.n_local_experts, self.inter_dim, self.model_dim)
        w2 = self.weight2.view(self.n_local_experts, self.model_dim, self.inter_dim)
        w3 = self.weight3.view(self.n_local_experts, self.inter_dim, self.model_dim)
        outs = []
        off = 0
        for e, cnt in enumerate(group_sizes):
            if cnt == 0:
                continue
            xe = x_perm[off : off + cnt]
            off += cnt
            h = F.silu(F.linear(xe, w1[e])) * F.linear(xe, w3[e])
            outs.append(F.linear(h, w2[e]))
        if not outs:
            return torch.zeros_like(x_perm)
        return torch.cat(outs, dim=0)


class _DummyRouter(nn.Module):
    def __init__(self, dim: int, n_experts: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_experts, dim) * 0.1)
        self.scaling_factor = 1.0

    def forward(self, x: torch.Tensor, load_balancing_type=None):
        logits = F.linear(x, self.weight)
        probs = F.softmax(logits, dim=-1)
        vals, idx = torch.topk(probs, k=2, dim=-1)
        vals = vals / vals.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return vals, idx, None, None


class _DummyMoE(nn.Module):
    def __init__(self, dim: int, inter: int, n_experts: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.router = _DummyRouter(dim, n_experts)
        self.experts = _DummyExperts(n_experts, dim, inter)
        self.fc1 = nn.Linear(dim, inter, bias=False)
        self.fc2 = nn.Linear(inter, dim, bias=False)
        self.fc3 = nn.Linear(dim, inter, bias=False)

    def residual(self, y, residual, c=None):
        return residual + y

    def forward(self, x_in: torch.Tensor, load_balancing_type=None):
        bsz, slen, dim = x_in.shape
        n_tok = bsz * slen
        x = self.norm(x_in)
        y1 = self.fc2(F.silu(self.fc1(x)) * self.fc3(x))
        w, idx, _, aux = self.router(x, load_balancing_type)
        x_2d = x.reshape(n_tok, dim)
        tok_ids = torch.arange(n_tok).unsqueeze(1).expand(n_tok, 2).reshape(-1)
        exp_ids = idx.reshape(-1)
        w_vals = w.reshape(-1)
        order = torch.argsort(exp_ids, stable=False)
        tok_s = tok_ids.index_select(0, order)
        exp_s = exp_ids.index_select(0, order)
        w_s = w_vals.index_select(0, order)
        gs = torch.bincount(exp_s, minlength=self.experts.n_local_experts).tolist()
        y_p = self.experts(x_2d.index_select(0, tok_s), gs)
        y2 = torch.zeros_like(x_2d).index_add_(0, tok_s, y_p * w_s.unsqueeze(-1)).reshape(bsz, slen, dim)
        return self.residual(y1 + y2, x_in), aux


class _DummyLayer(nn.Module):
    def __init__(self, dim: int, inter: int, n_experts: int):
        super().__init__()
        self.moe = _DummyMoE(dim, inter, n_experts)


class _DummyLoopedModel(nn.Module):
    def __init__(self, n_layers: int = 4, loop_start: int = 1, loop_end: int = 3, loop_times: int = 3):
        super().__init__()
        self.loop_start_layer = loop_start
        self.loop_end_layers = loop_end
        self.loop_times = loop_times
        self.layers = nn.ModuleList([_DummyLayer(16, 12, 6) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor):
        for lid in range(0, self.loop_start_layer):
            x, _ = self.layers[lid].moe(x)
        for _ in range(self.loop_times):
            for lid in range(self.loop_start_layer, self.loop_end_layers):
                x, _ = self.layers[lid].moe(x)
        for lid in range(self.loop_end_layers, len(self.layers)):
            x, _ = self.layers[lid].moe(x)
        return x


class TestR3LoopSchedV2(unittest.TestCase):
    def test_topology_and_pure_k12_and_natural_boundary(self):
        topo_ouro = build_visit_topology(28, 0, 28, 4)
        self.assertEqual(len(topo_ouro), 112)
        topo_huginn = build_visit_topology(28, 8, 20, 8)
        self.assertEqual(len(topo_huginn), 112)

        # Pure k12 must NEVER inject k=0!
        imp = np.arange(1, 113, dtype=np.float64)
        vk38 = solve_pure_k12_schedule(imp, 0.375)
        self.assertEqual(sum(k == 0 for k in vk38), 0)
        self.assertEqual(sum(k == 1 for k in vk38), 84)
        self.assertEqual(sum(k == 2 for k in vk38), 28)

        # Natural boundary schedule on Huginn protects ONLY coda (8 visits) -> 104 at k=1 (-46.4%)
        vk_hug, lbl_hug = build_natural_boundary_k12_schedule(topo_huginn, np.ones(28))
        self.assertEqual(lbl_hug, "coda_only_k2_46.4pct")
        self.assertEqual(sum(k == 2 for k in vk_hug), 8)
        self.assertEqual(sum(k == 1 for k in vk_hug), 104)
        self.assertTrue(all(vk_hug[t["vid"]] == 2 for t in topo_huginn if t["stage"] == "coda"))

        # Pure stage -37.5% on Ouro sets loop_0..2 to k=1 and loop_3 to k=2!
        st_sens = {"loop_0": 0.01, "loop_1": 0.02, "loop_2": 0.015, "loop_3": 0.10}
        vk_st_ouro = build_stage_only_cut38_schedule(topo_ouro, st_sens)
        self.assertEqual(sum(k == 1 for k in vk_st_ouro), 84)
        self.assertTrue(all(vk_st_ouro[t["vid"]] == 2 for t in topo_ouro if t["stage"] == "loop_3"))

        # Factorized importance interpolation checks
        imp_f = compute_factorized_importance(topo_ouro, st_sens, np.linspace(1.0, 2.0, 28), alpha=0.5)
        self.assertEqual(imp_f.shape, (112,))
        self.assertGreater(imp_f[84:].mean(), imp_f[:84].mean())

    def test_executor_parity_including_fast_small_batch_bmm(self):
        torch.manual_seed(0)
        model = _DummyLoopedModel(n_layers=4, loop_start=1, loop_end=3, loop_times=3)
        topo = build_visit_topology(4, 1, 3, 3)  # 1 + 6 + 1 = 8 visits

        for shape in [(1, 1, 16), (4, 1, 16), (2, 5, 16)]:
            x = torch.randn(*shape)
            with torch.no_grad():
                y_ref = model(x)

                ex_v = LoopSchedV2Executor(model, topo, mode="verify_k2", fast_small_batch=False)
                with ex_v:
                    ex_v.reset_pass()
                    y_v = model(x)
                self.assertLess((y_ref - y_v).abs().max().item(), 1e-5)

                ex_fast = LoopSchedV2Executor(model, topo, mode="verify_k2", fast_small_batch=True, small_batch_max_pairs=16)
                with ex_fast:
                    ex_fast.reset_pass()
                    y_fast = model(x)
                self.assertLess((y_ref - y_fast).abs().max().item(), 1e-5)

        # Verify static k=0, k=1, k=2 slot accounting on fast_small_batch=True
        vk = [2, 1, 0, 1, 0, 1, 1, 2]  # sum = 8 / 16 = 50% cut
        ex_st = LoopSchedV2Executor(model, topo, mode="static", visit_k=vk, fast_small_batch=True)
        with torch.no_grad(), ex_st:
            ex_st.reset_pass()
            _ = model(torch.randn(1, 1, 16))
        self.assertAlmostEqual(ex_st.cut_pct, 50.0, places=5)
        self.assertEqual(ex_st.k0_visits_run, 2)
        self.assertEqual(ex_st.k1_visits_run, 4)
        self.assertEqual(ex_st.k2_visits_run, 2)

    def test_iter_greedy_avoids_nonlinear_k0_cascade(self):
        # Synthetic model with 8 visits: visits 6,7 are high-cost boundary (coda),
        # visits 1..5 are recurrent middle with cheap single k=0, EXCEPT adjacent k=0 pairs explode (+5.0)!
        topo = build_visit_topology(4, 1, 3, 3)  # V=8: 0=prelude, 1..6=loop_0..2, 7=coda
        # Override stages so 6,7 are high-importance coda-like visits
        imp = np.array([1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 10.0, 10.0])
        screen_k0 = np.array([5.0, 0.05, 0.05, 0.05, 0.05, 0.05, 5.0, 5.0])
        eligible = [1, 2, 3, 4, 5]

        def nonlinear_loss(vk: list[int]) -> float:
            # k=1 cost: huge on 6,7 (0.5 each), tiny elsewhere (0.01)
            loss = 0.0
            for v, k in enumerate(vk):
                if k == 1:
                    loss += 0.50 if v in (6, 7) else 0.01
                elif k == 0:
                    loss += 5.0 if v in (0, 6, 7) else 0.05
            # Severe nonlinear cascade penalty for adjacent k=0 visits!
            for v in range(len(vk) - 1):
                if vk[v] == 0 and vk[v + 1] == 0:
                    loss += 5.0
            return loss

        res = solve_iter_greedy_k012(
            eval_a_fn=nonlinear_loss,
            eval_b_fn=nonlinear_loss,
            topo=topo,
            importance=imp,
            screen_k0_cost=screen_k0,
            eligible_k0_vids=eligible,
            max_swaps_50=5,
            cand_pool_size=5,
            patience_b=1,
        )
        vk50 = res["vk_50"]
        # Must upgrade BOTH 6 and 7 to k=2, paid by 2 NON-ADJACENT k=0 visits in 1..5!
        self.assertEqual(sum(vk50), 8)
        self.assertEqual(vk50[6], 2)
        self.assertEqual(vk50[7], 2)
        self.assertEqual(sum(k == 0 for k in vk50), 2)
        # Verify zero adjacent k=0 collisions!
        self.assertFalse(any(vk50[i] == 0 and vk50[i + 1] == 0 for i in range(7)))
        self.assertLess(nonlinear_loss(vk50), nonlinear_loss([1] * 8) - 0.8)

        # -62.5% -> 6 active slots
        vk62 = res["vk_62"]
        self.assertEqual(sum(vk62), 6)

    def test_w3_pool_awq_inplace(self):
        torch.manual_seed(1)
        model = _DummyLoopedModel(n_layers=2, loop_start=0, loop_end=2, loop_times=2)
        act_rms = torch.rand(2, 6, 16) + 0.1
        meta = apply_w3_pool_awq_inplace(model, act_rms, bits=3, group_size=8)
        self.assertGreater(meta["vram_compression"], 2.0)
        for layer in model.layers:
            self.assertTrue(torch.isfinite(layer.moe.experts.weight1).all())
            self.assertTrue(torch.isfinite(layer.moe.experts.weight2).all())
            self.assertTrue(torch.isfinite(layer.moe.experts.weight3).all())


if __name__ == "__main__":
    unittest.main()
