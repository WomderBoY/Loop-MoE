"""CPU unit tests for Probe 3 (Ouro-Thinking Depth & Floor-Protected Exit)."""

from __future__ import annotations

import types
import unittest

import numpy as np
import torch
import torch.nn as nn

from loop_research.probe3_thinking_depth import (
    DepthPolicy,
    OuroDepthEngine,
    calibrate_floor_threshold,
    compute_kl_div,
)


class _DummyRoPE(nn.Module):
    def __init__(self, head_dim: int) -> None:
        super().__init__()
        self.head_dim = head_dim

    def forward(self, h: torch.Tensor, pos_ids: torch.Tensor):
        shape = (*pos_ids.shape, self.head_dim)
        return torch.ones(shape, dtype=h.dtype), torch.zeros(shape, dtype=h.dtype)


class _DummyLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv: int) -> None:
        super().__init__()
        hd = dim // n_heads
        self.input_layernorm = nn.Identity()
        self.input_layernorm_2 = nn.Identity()
        self.post_attention_layernorm = nn.Identity()
        self.post_attention_layernorm_2 = nn.Identity()
        self.self_attn = types.SimpleNamespace(
            q_proj=nn.Linear(dim, n_heads * hd, bias=False),
            k_proj=nn.Linear(dim, n_kv * hd, bias=False),
            v_proj=nn.Linear(dim, n_kv * hd, bias=False),
            o_proj=nn.Linear(n_heads * hd, dim, bias=False),
            scaling=hd ** -0.5,
        )
        self.mlp = nn.Linear(dim, dim, bias=False)


class _DummyOuroModel(nn.Module):
    def __init__(self, vocab: int = 32, dim: int = 16, n_heads: int = 4, n_kv: int = 2) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(
            total_ut_steps=4,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv,
            head_dim=dim // n_heads,
            hidden_size=dim,
        )
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, dim)
        self.model.rotary_emb = _DummyRoPE(dim // n_heads)
        self.model.layers = nn.ModuleList([_DummyLayer(dim, n_heads, n_kv) for _ in range(2)])
        self.model.norm = nn.Identity()
        self.model.early_exit_gate = nn.Linear(dim, 1)
        self.lm_head = nn.Linear(dim, vocab, bias=False)


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
        self.assertAlmostEqual(probs[0], 0.0, places=6)
        self.assertAlmostEqual(mr, 2.6, delta=0.05)

        vals_gate = {1: rng.uniform(0.0, 0.6, size=1000), 2: rng.uniform(0.2, 0.9, size=1000), 3: rng.uniform(0.5, 1.0, size=1000)}
        tau_g2, mr_g2, p_g2 = calibrate_floor_threshold(vals_gate, mode="gate", r_min=2, n_loops=4, target_mean_r=2.7)
        self.assertAlmostEqual(p_g2[0], 0.0, places=6)
        self.assertAlmostEqual(mr_g2, 2.7, delta=0.05)

    def test_engine_batched_partial_finished_and_all_modes(self) -> None:
        torch.manual_seed(0)
        model = _DummyOuroModel()
        eng = OuroDepthEngine(model, torch.device("cpu"))
        prompts = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        # Make one token act as EOS so partial-batch finishing is exercised!
        first_gen, _ = eng.generate_batched(
            prompts, DepthPolicy(name="f", n_loops=4, exit_mode="fixed"),
            max_new_tokens=2, pad_token_id=0, eos_token_ids=set(), think_end_ids=set(),
        )
        eos_id = first_gen[0][0]
        # Calib collection with partially finished batch (this reproduces the exact crash!)
        _, st_cal = eng.generate_batched(
            prompts, DepthPolicy(name="cal", n_loops=4, exit_mode="fixed"),
            max_new_tokens=5, pad_token_id=0, eos_token_ids={eos_id}, think_end_ids=set(),
            collect_calib=True,
        )
        self.assertIn(2, st_cal["calib_kl"])
        self.assertGreater(len(st_cal["calib_kl"][2]), 0)
        # Exercise every exit policy mode including ragged_kv=True & False
        for pol in [
            DepthPolicy(name="g1", n_loops=4, exit_mode="ouro_gate", r_min=1, thresh=0.3, ragged_kv=True),
            DepthPolicy(name="g2", n_loops=4, exit_mode="ouro_gate", r_min=2, thresh=0.5, ragged_kv=True),
            DepthPolicy(name="kl", n_loops=4, exit_mode="kl", r_min=2, thresh=0.5, ragged_kv=True),
            DepthPolicy(name="kl_f", n_loops=4, exit_mode="kl", r_min=2, thresh=0.5, ragged_kv=False),
            DepthPolicy(name="gap", n_loops=4, exit_mode="gap_agree", r_min=2, thresh=0.05, ragged_kv=True),
            DepthPolicy(name="ps", n_loops=4, exit_mode="phase_split", think_loops=2, answer_loops=4, ragged_kv=True),
            DepthPolicy(name="pkl", n_loops=4, exit_mode="phase_kl", r_min=2, thresh=0.5, ragged_kv=True),
        ]:
            gens, st = eng.generate_batched(
                prompts, pol, max_new_tokens=5, pad_token_id=0,
                eos_token_ids={eos_id}, think_end_ids={first_gen[1][0]},
            )
            self.assertEqual(len(gens), 3)
            self.assertGreaterEqual(st["mean_loops"], 1.0)


if __name__ == "__main__":
    unittest.main()
