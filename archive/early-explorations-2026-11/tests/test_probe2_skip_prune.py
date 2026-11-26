"""CPU unit tests for Probe 2 (Looped MoE Dynamic Skipping & Stage Pruning)."""

from __future__ import annotations

import types
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from loop_research.probe2_loopmoe_skip_prune import (
    LoopMoEIntervention,
    build_alive_expert_mask,
    calibrate_drop_threshold,
    collect_calib_skip_and_prune_stats,
    evaluate_nll,
    stage_id_for,
    stage_names_for,
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
    def __init__(self, n_experts: int, dim: int) -> None:
        super().__init__()
        self.n_local_experts = n_experts
        self.model_dim = dim
        self.inter_dim = dim * 2

    def forward(self, x: torch.Tensor, group_sizes: list[int]) -> torch.Tensor:
        return x * 0.5


class _DummyIFMModel(nn.Module):
    def __init__(self, vocab: int = 32, dim: int = 8, n_layers: int = 3, n_experts: int = 5, n_loops: int = 2) -> None:
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
            layer.moe.experts = _DummyExperts(n_experts, dim)
            self.layers.append(layer)

    def forward(self, inp: torch.Tensor, multi_segments: bool = False, cache=None):
        h = self.emb(inp)
        order = [0] + [1] * self.loop_times + [2]
        for lid in order:
            w, idx, tpe, _ = self.layers[lid].moe.router(h, None)
            gs_list = [int(v) for v in tpe.tolist()]  # matches xllm passing Python list[int]!
            x_rep = h.reshape(-1, h.shape[-1]).repeat_interleave(2, dim=0)
            y_exp = self.layers[lid].moe.experts(x_rep, gs_list)
            y_comb = (y_exp.reshape(*idx.shape, -1) * w.unsqueeze(-1)).sum(dim=-2)
            h = h + y_comb
        return self.head(h), None, None


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
        self.assertTrue(alive[2, 3] and alive[2, 4])

    def test_calibrate_drop_threshold(self) -> None:
        arr = np.linspace(0.0, 1.0, 1001)
        tau = calibrate_drop_threshold([arr], 0.5)
        self.assertAlmostEqual(tau, 0.5, places=2)

    def test_end_to_end_mock_ifm_hooks(self) -> None:
        torch.manual_seed(0)
        m = _DummyIFMModel()
        seqs = [{"tokens": [1, 2, 3, 4, 5, 6], "domain": "text"}, {"tokens": [6, 5, 4, 3, 2, 1], "domain": "code"}]
        cal = collect_calib_skip_and_prune_stats(m, seqs, torch.device("cpu"))
        self.assertEqual(cal["counts"].shape, (3, 2, 5))
        self.assertEqual(len(cal["calib_stage_sens"]), 4)
        for ctrl in [
            LoopMoEIntervention(m, mode="native"),
            LoopMoEIntervention(m, mode="stage_ablation", target_stage=1, stage_action="drop2_renorm"),
            LoopMoEIntervention(m, mode="stage_ablation", target_stage=1, stage_action="skip_moe"),
            LoopMoEIntervention(m, mode="dynamic_skip", dyn_threshold=0.95, stage_weights=np.ones(4), stay_factor=0.7),
            LoopMoEIntervention(m, mode="prune", alive_mask=torch.ones((3, 5), dtype=torch.bool)),
        ]:
            res = evaluate_nll(m, seqs, ctrl, torch.device("cpu"))
            self.assertTrue(np.isfinite(res["nll"]))


if __name__ == "__main__":
    unittest.main()
