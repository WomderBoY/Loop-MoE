"""Lightweight CPU unit tests for Round 4: IFM Fast Engine, Stratified Swap Pareto & Inkling-2-Mini."""

from __future__ import annotations

import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from loop_research.ifm_fast_engine import (
    IFMFastMoEExecutor,
    fast_gpu_route_k1_or_k2,
    pytorch_indexed_moe_ref,
)
from loop_research.r3_loopsched_v2 import LoopSchedV2Executor, build_visit_topology
from loop_research.r4a_ifm_fast_and_swaps import (
    build_stratified_k0_order,
    solve_stratified_swap_pareto_and_full_spectrum,
)
from loop_research.inkling2_vllm_hook import Inkling2LoopSchedController
from loop_research.r4b_inkling2_loopsched import (
    build_inkling2_topology,
    build_stage_schedule_inkling2,
    generalized_shrinkage_stats,
    paired_mcnemar_test,
    solve_convex_bounded_swap_topk,
    solve_general_topk_iter_greedy,
    solve_layer_only_topk_schedule,
    solve_stage_static_topk,
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
        return torch.cat(outs, dim=0) if outs else torch.zeros_like(x_perm)


class _DummyRouter(nn.Module):
    def __init__(self, dim: int, n_experts: int, with_bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_experts, dim) * 0.2)
        self.score_func = "softmax"
        self.scaling_factor = 1.0
        self.topk = 2
        self.router_bias = nn.Parameter(torch.randn(n_experts) * 0.05) if with_bias else None

    def forward(self, x: torch.Tensor, load_balancing_type=None):
        logits = F.linear(x, self.weight).float()
        orig = F.softmax(logits, dim=-1)
        sel = orig + self.router_bias if self.router_bias is not None else orig
        idx = torch.topk(sel, 2, dim=-1).indices
        vals = torch.gather(orig, -1, idx)
        vals = vals / vals.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return vals.to(x.dtype), idx, None, None


class _DummyMoE(nn.Module):
    def __init__(self, dim: int, inter: int, n_experts: int, with_bias: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.router = _DummyRouter(dim, n_experts, with_bias=with_bias)
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
        w1 = self.experts.weight1.view(self.experts.n_local_experts, self.experts.inter_dim, dim)
        w2 = self.experts.weight2.view(self.experts.n_local_experts, dim, self.experts.inter_dim)
        w3 = self.experts.weight3.view(self.experts.n_local_experts, self.experts.inter_dim, dim)
        y2 = pytorch_indexed_moe_ref(x_2d, idx.reshape(n_tok, 2), w.reshape(n_tok, 2), w1, w2, w3).reshape(bsz, slen, dim)
        return self.residual(y1 + y2, x_in), aux


class _DummyLayer(nn.Module):
    def __init__(self, dim: int, inter: int, n_experts: int, with_bias: bool = False):
        super().__init__()
        self.moe = _DummyMoE(dim, inter, n_experts, with_bias=with_bias)


class _DummyLoopedModel(nn.Module):
    def __init__(self, with_bias: bool = False):
        super().__init__()
        self.loop_start_layer = 1
        self.loop_end_layers = 3
        self.loop_times = 3
        self.layers = nn.ModuleList([_DummyLayer(16, 12, 6, with_bias=with_bias) for _ in range(4)])

    def forward(self, x: torch.Tensor):
        for lid in range(0, self.loop_start_layer):
            x, _ = self.layers[lid].moe(x)
        for _ in range(self.loop_times):
            for lid in range(self.loop_start_layer, self.loop_end_layers):
                x, _ = self.layers[lid].moe(x)
        for lid in range(self.loop_end_layers, len(self.layers)):
            x, _ = self.layers[lid].moe(x)
        return x


class TestR4FastAndInkling(unittest.TestCase):
    def test_ifm_fast_executor_bit_exact_against_r3_and_native(self):
        for with_bias in (False, True):
            torch.manual_seed(123 + int(with_bias))
            model = _DummyLoopedModel(with_bias=with_bias)
            topo = build_visit_topology(4, 1, 3, 3)  # V = 8

            for shape in [(1, 1, 16), (4, 1, 16), (2, 5, 16)]:
                x = torch.randn(*shape)
                with torch.no_grad():
                    y_nat = model(x)
                    ex_f2 = IFMFastMoEExecutor(model, topo, visit_k=[2] * 8, use_fast_router=True)
                    with ex_f2:
                        ex_f2.reset_pass()
                        y_f2 = model(x)
                    self.assertLess((y_nat - y_f2).abs().max().item(), 1e-5)

                    # Compare hybrid k={0,1,2} schedule between IFMFastMoEExecutor and LoopSchedV2Executor
                    vk = [2, 1, 0, 1, 1, 0, 1, 2]
                    ex_ref = LoopSchedV2Executor(model, topo, mode="static", visit_k=vk, renorm_on_drop2=False)
                    with ex_ref:
                        ex_ref.reset_pass()
                        y_ref = model(x)
                    ex_fast = IFMFastMoEExecutor(model, topo, visit_k=vk, renorm_on_drop2=False, use_fast_router=True)
                    with ex_fast:
                        ex_fast.reset_pass()
                        y_fast = model(x)
                    self.assertLess((y_ref - y_fast).abs().max().item(), 1e-5)

    def test_stratified_order_and_swap_pareto_solver(self):
        # 16-visit Huginn-like synthetic topology: 2 prelude + 4 loops x 3 layers + 2 coda = 16
        topo = build_visit_topology(7, 2, 5, 4)  # 2 + 12 + 2 = 16
        self.assertEqual(len(topo), 16)
        stage_k0_map = {
            "prelude": 6.0,
            "loop_0": 0.40,
            "loop_1": 0.04,  # ultra-redundant middle!
            "loop_2": 0.05,  # ultra-redundant middle!
            "loop_3": 0.18,
            "coda": 2.5,
        }
        # Deliberately put noisy tiny screen_k0 in sensitive loop_0 to test stratification!
        screen_k0 = np.full(16, 0.1, dtype=np.float64)
        screen_k0[2] = 0.001  # loop_0 noise trap!
        eligible = [t["vid"] for t in topo if t["is_rec"]]

        order = build_stratified_k0_order(topo, stage_k0_map, screen_k0, eligible)
        # First 6 items MUST come from primary low-sens stages loop_1 and loop_2 (NOT loop_0 trap vid=2!)
        self.assertTrue(all(topo[v]["stage"] in ("loop_1", "loop_2") for v in order[:6]))

        imp = np.array([1.0 if t["stage"] != "coda" else 10.0 for t in topo], dtype=np.float64)

        def nonlinear_loss(vk: list[int]) -> float:
            loss = 0.0
            for v, k in enumerate(vk):
                st = topo[v]["stage"]
                if k == 1:
                    loss += 0.6 if st == "coda" else 0.02
                elif k == 0:
                    loss += 0.06 if st in ("loop_1", "loop_2") else 0.5
            for i in range(len(vk) - 1):
                if vk[i] == 0 and vk[i + 1] == 0:
                    loss += 3.0  # adjacent k=0 cascade
            return loss

        sol = solve_stratified_swap_pareto_and_full_spectrum(
            eval_a_fn=nonlinear_loss,
            eval_b_fn=nonlinear_loss,
            topo=topo,
            stage_k0_map=stage_k0_map,
            importance=imp,
            screen_k0_cost=screen_k0,
            eligible_k0_vids=eligible,
            max_swaps=4,
            swap_checkpoints=(0, 2, 4),
            cand_pool_size=6,
            up_pool_size=2,
        )
        # Check exact active slot counts for -25%, -37.5%, -50%, -62.5%
        self.assertEqual(sum(sol["vk_25_iter"]), 24)
        self.assertEqual(sum(sol["vk_38_iter"]), 20)
        self.assertEqual(sum(sol["vk_50_best"]), 16)
        self.assertEqual(sum(sol["vk_62_strat"]), 12)
        # Best swap must upgrade BOTH coda visits (14, 15) to k=2 using non-adjacent loop_1/2 k=0s!
        self.assertEqual(sol["best_m"], 2)
        self.assertEqual(sol["vk_50_best"][14], 2)
        self.assertEqual(sol["vk_50_best"][15], 2)
        self.assertFalse(any(sol["vk_50_best"][i] == 0 and sol["vk_50_best"][i + 1] == 0 for i in range(15)))

    def test_inkling2_topology_generalized_shrinkage_and_topk_greedy(self):
        topo = build_inkling2_topology(24, 6, 18, 3)
        self.assertEqual(len(topo), 48)
        self.assertEqual(sum(t["stage"] == "prelude" for t in topo), 6)
        self.assertEqual(sum(t["stage"] == "coda" for t in topo), 6)
        self.assertEqual(sum(t["is_rec"] for t in topo), 36)

        # Synthetic orthogonal Top-8 experts with decaying norms -> MSE_renorm/MSE_noren must be >> 2.0!
        torch.manual_seed(7)
        N, K, D = 256, 8, 64
        w = torch.linspace(0.25, 0.04, K).unsqueeze(0).expand(N, K)
        # Near-orthogonal unit-ish experts with norm decay
        e = torch.randn(N, K, D) / (D ** 0.5)
        e = e * torch.linspace(1.5, 0.7, K).view(1, K, 1)
        st4 = generalized_shrinkage_stats(w, e, k_keep=4)
        self.assertGreater(st4["mse_ratio_ren_vs_noren"], 2.0)
        self.assertLess(abs(st4["gamma_opt"] - 1.0), 0.15)

        # Generalized Top-8 -> k_v greedy cutting 50% (384 -> 192 slots) protecting prelude/coda >= 6
        def mock_loss(vk: list[int]) -> float:
            tot = 0.0
            for t, k in zip(topo, vk):
                w_st = 5.0 if not t["is_rec"] else (0.2 if t["stage"] == "pass_1" else 0.8)
                tot += w_st * ((8 - k) ** 1.5)
            return tot

        vk50 = solve_general_topk_iter_greedy(
            mock_loss, topo, K_max=8, target_cut_frac=0.50,
            stage_priority_order=["pass_1", "pass_0", "pass_2", "prelude", "coda"],
            min_k_boundary=6, min_k_rec=1, step_delta_k=1, cand_pool_size=12,
        )
        self.assertEqual(sum(vk50), 192)
        # Prelude & coda must stay high while pass_1 is heavily thinned
        coda_mean = np.mean([vk50[t["vid"]] for t in topo if t["stage"] == "coda"])
        pass1_mean = np.mean([vk50[t["vid"]] for t in topo if t["stage"] == "pass_1"])
        self.assertGreater(coda_mean, pass1_mean + 2.0)

        # Deterministic stage schedule check
        vk_st = build_stage_schedule_inkling2(topo, {"prelude": 8, "pass_0": 4, "pass_1": 2, "pass_2": 4, "coda": 8})
        self.assertEqual(sum(vk_st), 6 * 8 + 12 * 4 + 12 * 2 + 12 * 4 + 6 * 8)

        # Layer-only ablation MUST assign identical k across all 3 passes of every looped physical layer [6..17]!
        visit_sens = np.array([5.0 if not t["is_rec"] else (0.1 if t["stage"] == "pass_1" else 1.0) for t in topo])
        vk_lo = solve_layer_only_topk_schedule(topo, visit_sens, K_max=8, target_cut_frac=0.50, min_k=1)
        self.assertEqual(sum(vk_lo), 192)
        for lid in range(6, 18):
            vids = [t["vid"] for t in topo if t["layer_id"] == lid]
            self.assertEqual(len(set(vk_lo[v] for v in vids)), 1)

        # Stage-static solver check
        vk_ss = solve_stage_static_topk(
            topo, {"prelude": 3.0, "pass_0": 0.4, "pass_1": 0.1, "pass_2": 0.6, "coda": 2.5}, K_max=8, target_cut_frac=0.50
        )
        self.assertEqual(sum(vk_ss), 192)
        self.assertGreater(vk_ss[0], vk_ss[18])  # prelude > pass_1

        # Inkling2LoopSchedController exact diagnostic accumulation parity vs generalized_shrinkage_stats!
        ctrl = Inkling2LoopSchedController(num_visits=48, k_max=8)
        ctrl.start_diagnostics()
        u_weighted = w.unsqueeze(-1) * e  # [N, 8, D]
        ctrl.accumulate_visit_diag(12, w[:100], u_weighted[:100])
        ctrl.accumulate_visit_diag(12, w[100:], u_weighted[100:])
        summary = ctrl.stop_diagnostics()
        self.assertAlmostEqual(summary[12][4]["gamma_opt"], st4["gamma_opt"], places=5)
        self.assertAlmostEqual(summary[12][4]["mse_ratio_ren_vs_noren"], st4["mse_ratio_ren_vs_noren"], places=4)
        self.assertAlmostEqual(summary[12][4]["cos_kept_dropped"], st4["cos_kept_dropped"], places=5)

    def test_r4c_zero_sens_bugfix_convex_swap_and_mcnemar(self):
        topo = build_inkling2_topology(24, 6, 18, 3)

        # 1. Regression test for R4-B zero-sensitivity pool lockout bug:
        # 10 visits have exact 0.0 sensitivity, convex per-visit loss strongly penalizes k=1.
        visit_sens = np.full(48, 0.004, dtype=np.float64)
        for z in [0, 2, 3, 4, 6, 13, 14, 28, 34, 37]:
            visit_sens[z] = 0.0

        def convex_loss(vk: list[int]) -> float:
            # Strongly convex in dropped experts (1/k), with gentle pass gradient (pass_0 < pass_2 < coda)
            stage_w = {"prelude": 0.8, "pass_0": 0.6, "pass_1": 1.0, "pass_2": 1.4, "coda": 2.2}
            return sum(stage_w[topo[v]["stage"]] * ((8.0 / max(k, 1)) - 1.0) ** 1.6 for v, k in enumerate(vk))

        vk_fixed = solve_general_topk_iter_greedy(
            convex_loss, topo, K_max=8, target_cut_frac=0.50,
            visit_sens=visit_sens, min_k_boundary=3, min_k_rec=2,
            step_delta_k=1, cand_pool_size=10, batch_drop_per_round=4, max_spread=2,
        )
        self.assertEqual(sum(vk_fixed), 192)
        self.assertLessEqual(max(vk_fixed) - min(vk_fixed), 2)
        self.assertGreaterEqual(min(vk_fixed), 2)
        # Repaired IterGreedy must beat uniform k=4 on the heterogeneous convex objective!
        self.assertLess(convex_loss(vk_fixed), convex_loss([4] * 48))

        # 2. Convex-bounded +-1 swap solver from uniform k=4:
        sw = solve_convex_bounded_swap_topk(
            convex_loss, convex_loss, topo, K_max=8, k_bar=4,
            max_delta=1, max_swaps=12, drop_pool=5, boost_pool=5, patience=3,
        )
        self.assertEqual(sum(sw["best_vk"]), 192)
        self.assertTrue(all(3 <= k <= 5 for k in sw["best_vk"]))
        self.assertGreater(sw["best_step"], 0)
        self.assertLess(convex_loss(sw["best_vk"]), convex_loss([4] * 48))
        # Coda visits must be boosted above pass_0 visits
        coda_k = np.mean([sw["best_vk"][t["vid"]] for t in topo if t["stage"] == "coda"])
        pass0_k = np.mean([sw["best_vk"][t["vid"]] for t in topo if t["stage"] == "pass_0"])
        self.assertGreater(coda_k, pass0_k)

        # 3. calib_b overfitting guard: if calib_a wants swaps that HURT calib_b, must revert to uniform (best_step=0)!
        def trap_a(vk: list[int]) -> float:
            return sum((0.1 if v < 6 else 5.0) * (4 - k) for v, k in enumerate(vk)) + 0.01 * sum((k - 4) ** 2 for k in vk)

        def clean_b(vk: list[int]) -> float:
            # Uniform k=4 is strictly optimal on calib_b!
            return 1.0 + sum((k - 4) ** 2 for k in vk)

        sw_guard = solve_convex_bounded_swap_topk(
            trap_a, clean_b, topo, K_max=8, k_bar=4,
            max_delta=1, max_swaps=8, drop_pool=4, boost_pool=4, patience=2,
        )
        self.assertEqual(sw_guard["best_step"], 0)
        self.assertEqual(sw_guard["best_vk"], [4] * 48)

        # 4. Exact paired McNemar test checks
        m_eq = paired_mcnemar_test([1, 0, 1, 0], [0, 1, 1, 0])
        self.assertEqual((m_eq["n10"], m_eq["n01"]), (1, 1))
        self.assertAlmostEqual(m_eq["p_value"], 1.0, places=6)
        m_sig = paired_mcnemar_test([1] * 20, [0] * 20)
        self.assertEqual((m_sig["n10"], m_sig["n01"]), (20, 0))
        self.assertLess(m_sig["p_value"], 1e-5)

    def test_fast_range_downloader_bit_exact_and_shard_check(self):
        import http.server
        import tempfile
        import threading
        from pathlib import Path
        from loop_research.data_assets import _model_shards_complete, fast_range_download_file

        payload = bytes((i * 131 + 7) % 256 for i in range(250_000))

        class _RangeHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                rng = self.headers.get("Range", "")
                if rng.startswith("bytes="):
                    a_s, b_s = rng[6:].split("-")
                    a, b = int(a_s), int(b_s)
                    chunk = payload[a : b + 1]
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {a}-{b}/{len(payload)}")
                    self.send_header("Content-Length", str(len(chunk)))
                    self.end_headers()
                    self.wfile.write(chunk)
                else:
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)

            def log_message(self, *_args):
                pass

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            with tempfile.TemporaryDirectory() as td:
                dest = Path(td) / "shard.safetensors"
                fast_range_download_file(
                    [f"http://127.0.0.1:{port}/shard.safetensors"],
                    dest,
                    num_workers=4,
                    chunk_size_bytes=32_768,
                    expected_size=len(payload),
                    log_interval_sec=999.0,
                )
                self.assertEqual(dest.read_bytes(), payload)

                # Verify _model_shards_complete rejects partial directory with only mtp.safetensors!
                (Path(td) / "config.json").write_text("{}")
                (Path(td) / "model.safetensors.index.json").write_text(
                    '{"weight_map": {"w": "model-00001-of-00001.safetensors"}}'
                )
                (Path(td) / "mtp.safetensors").write_bytes(b"abc")
                self.assertFalse(_model_shards_complete(Path(td)))
                (Path(td) / "model-00001-of-00001.safetensors").write_bytes(b"xyz")
                self.assertTrue(_model_shards_complete(Path(td)))
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
