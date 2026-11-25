"""CPU unit tests for Route 1: TCRE open-loop geometry, Aitken cancellation,
Ragged-KV engine batched vs B=1 equivalence, and Thinking </think> answer parsing."""

from __future__ import annotations

from types import SimpleNamespace
import torch
import torch.nn as nn

from loop_research.data_assets import extract_predicted_math_answer
from loop_research.route1_tcre_ragged import (
    OuroRoute1Engine,
    Route1Policy,
    _simulate_entropy_agree_mean_r,
    compute_openloop_readout,
    compute_trajectory_consistency,
)


def test_fisher_cosine_shift_and_scale_invariance() -> None:
    torch.manual_seed(0)
    z0 = torch.randn(4, 64)
    dz1 = torch.randn(4, 64) * 0.3
    dz2 = 0.6 * dz1 + torch.randn(4, 64) * 0.05
    z1 = z0 + dz1
    z2 = z1 + dz2

    tc_base = compute_trajectory_consistency(z0, z1, z2)
    # Shift logits by arbitrary per-row constants and scale dz2 positively
    shift = torch.tensor([[10.0], [-5.0], [3.3], [7.7]])
    tc_shift = compute_trajectory_consistency(z0 + shift, z1 - 2.0 * shift, z2 + 4.0 * shift)
    assert torch.allclose(tc_base["cos_fisher"], tc_shift["cos_fisher"], atol=1e-5)
    assert torch.allclose(tc_base["lambda_hat"], tc_shift["lambda_hat"], atol=1e-5)
    assert torch.allclose(tc_base["delta_h"], tc_shift["delta_h"], atol=1e-5)

    # Positive scaling of dz2 leaves cos_fisher unchanged!
    z2_scaled = z1 + 3.5 * dz2
    tc_scaled = compute_trajectory_consistency(z0, z1, z2_scaled)
    assert torch.allclose(tc_base["cos_fisher"], tc_scaled["cos_fisher"], atol=1e-5)


def test_tcre_period2_oscillation_exact_cancellation() -> None:
    """When trajectory oscillates z* + (-1)^r v, both tcre_cos and tcre_aitken cancel v exactly!"""
    torch.manual_seed(1)
    z_star = torch.randn(3, 32)
    v = torch.randn(3, 32) * 0.8
    # z_0 = z* - v, z_1 = z* + v, z_2 = z* - v  => dz_1 = +2v, dz_2 = -2v => cos_F = -1, lambda = -1!
    z0 = z_star - v
    z1 = z_star + v
    z2 = z_star - v

    tc = compute_trajectory_consistency(z0, z1, z2)
    assert torch.allclose(tc["cos_fisher"], torch.full((3,), -1.0), atol=1e-5)
    assert torch.allclose(tc["lambda_hat"], torch.full((3,), -1.0), atol=1e-5)

    out_cos, _ = compute_openloop_readout([z0, z1, z2], mode="tcre_cos", omega_max=1.0)
    out_ait, _ = compute_openloop_readout([z0, z1, z2], mode="tcre_aitken", omega_max=1.0)
    # Both must recover z_star = 0.5*(z1 + z2) exactly!
    assert torch.allclose(out_cos, z_star, atol=1e-5)
    assert torch.allclose(out_ait, z_star, atol=1e-5)


def test_tcre_monotone_contraction_extrapolates_toward_limit() -> None:
    """When z_r = z* - rho^r u (0 < rho < 1) and entropy sharpens, TCRE extrapolates forward."""
    z_star = torch.zeros(2, 16)
    z_star[:, 0] = 3.0  # clear sharpening target
    u = torch.zeros(2, 16)
    u[:, 0] = 2.5
    rho = 0.5
    z1 = z_star - (rho ** 1) * u
    z2 = z_star - (rho ** 2) * u
    z3 = z_star - (rho ** 3) * u
    z4 = z_star - (rho ** 4) * u

    out_cos, diag_c = compute_openloop_readout([z1, z2, z3, z4], mode="tcre_cos", omega_max=1.0)
    out_ait, diag_a = compute_openloop_readout([z1, z2, z3, z4], mode="tcre_aitken", omega_max=1.0)
    assert float(diag_c["cos_fisher"].min()) > 0.999
    assert float(diag_a["lambda_hat"].mean()) > 0.49
    # Both must push target logit 0 ABOVE z4[:, 0] (towards z_star)
    assert bool((out_cos[:, 0] > z4[:, 0]).all())
    assert bool((out_ait[:, 0] > z4[:, 0]).all())


def test_tcre_cos_continuity_at_zero() -> None:
    """Verify no discontinuous jump across cos_fisher = 0."""
    torch.manual_seed(2)
    z0 = torch.randn(1, 32)
    z1 = z0 + torch.randn(1, 32) * 0.5
    # Construct orthogonal increment under p1
    p1 = torch.softmax(z1, dim=-1)
    dz1 = z1 - z0
    c1 = dz1 - (p1 * dz1).sum(dim=-1, keepdim=True)
    rand = torch.randn(1, 32)
    c_orth = rand - (p1 * rand).sum(dim=-1, keepdim=True)
    proj = (p1 * c_orth * c1).sum(dim=-1, keepdim=True) / ((p1 * c1 * c1).sum(dim=-1, keepdim=True) + 1e-8)
    c_orth = c_orth - proj * c1

    z2_neg = z1 + c_orth - 1e-4 * c1
    z2_pos = z1 + c_orth + 1e-4 * c1
    o_neg, _ = compute_openloop_readout([z0, z1, z2_neg], mode="tcre_cos", delta_h_tol=10.0)
    o_pos, _ = compute_openloop_readout([z0, z1, z2_pos], mode="tcre_cos", delta_h_tol=10.0)
    assert float((o_neg - o_pos).abs().max().item()) < 0.05


def test_thinking_special_token_answer_parsing() -> None:
    raw = "<think>First try 99, wait no 17+25=42.</think>\nTherefore the final answer is \\boxed{42}.<|im_end|>"
    assert extract_predicted_math_answer(raw, mode="gsm8k") == "42"
    raw_no_box = "<think>Intermediate 100 then 42</think>\nThe answer is 42.<|im_end|>"
    assert extract_predicted_math_answer(raw_no_box, mode="gsm8k") == "42"
    raw_math = "<think>Try 1/2 then 3/4</think>\nSo we get \\boxed{\\frac{3}{4}}.<|im_end|>"
    assert extract_predicted_math_answer(raw_math, mode="math500") == "\\frac{3}{4}"


# ---------------------------------------------------------------------------
# Tiny Synthetic Ouro Model to Unit-Test Pre-Allocated KV & Ragged-KV Batched == B=1
# ---------------------------------------------------------------------------

class _TinyRotary(nn.Module):
    def __init__(self, head_dim: int) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, pos_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = pos_ids.float().unsqueeze(-1) * self.inv_freq[None, None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


class _TinyLayer(nn.Module):
    def __init__(self, hidden: int, n_heads: int) -> None:
        super().__init__()
        self.input_layernorm = nn.RMSNorm(hidden)
        self.input_layernorm_2 = nn.RMSNorm(hidden)
        self.post_attention_layernorm = nn.RMSNorm(hidden)
        self.post_attention_layernorm_2 = nn.RMSNorm(hidden)
        self.self_attn = SimpleNamespace(
            q_proj=nn.Linear(hidden, hidden, bias=False),
            k_proj=nn.Linear(hidden, hidden, bias=False),
            v_proj=nn.Linear(hidden, hidden, bias=False),
            o_proj=nn.Linear(hidden, hidden, bias=False),
            scaling=(hidden // n_heads) ** -0.5,
        )
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden * 2, bias=False), nn.SiLU(), nn.Linear(hidden * 2, hidden, bias=False))


class _TinyOuro(nn.Module):
    def __init__(self, vocab: int = 48, hidden: int = 32, n_heads: int = 4, n_layers: int = 2) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            num_hidden_layers=n_layers,
            total_ut_steps=4,
            num_attention_heads=n_heads,
            num_key_value_heads=n_heads,
            head_dim=hidden // n_heads,
            hidden_size=hidden,
            vocab_size=vocab,
        )
        self.model = SimpleNamespace(
            embed_tokens=nn.Embedding(vocab, hidden),
            rotary_emb=_TinyRotary(hidden // n_heads),
            layers=nn.ModuleList([_TinyLayer(hidden, n_heads) for _ in range(n_layers)]),
            norm=nn.RMSNorm(hidden),
            early_exit_gate=nn.Linear(hidden, 1),
        )
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        # Register submodules so parameters() works
        self._embed = self.model.embed_tokens
        self._layers = self.model.layers
        self._norm = self.model.norm
        self._gate = self.model.early_exit_gate


def test_ragged_kv_batched_equals_b1_and_skips_loops() -> None:
    torch.manual_seed(42)
    model = _TinyOuro().eval()
    eng = OuroRoute1Engine(model, torch.device("cpu"))

    p1 = [1, 5, 9, 12, 3, 7]
    p2 = [1, 4, 8]
    p3 = [1, 2, 6, 11, 14]

    for exit_mode in ("none", "gap_agree", "ouro_gate", "random"):
        pol = Route1Policy(
            name=f"test_{exit_mode}",
            n_loops=4,
            readout_mode="tcre_cos",
            exit_mode=exit_mode,
            gate_cdf_thresh=0.65,
            exit_gap1=0.35,
            exit_gap2=0.15,
            exit_gap3=0.25,
            rand_exit_probs=(0.3, 0.3, 0.2, 0.2),
        )
        gen_b, st_b = eng.generate_batched([p1, p2, p3], pol, max_new_tokens=10, pad_token_id=0)
        for idx_i, p_i in enumerate([p1, p2, p3]):
            gen_1, _ = eng.generate_batched([p_i], pol, max_new_tokens=10, pad_token_id=0, seq_id_offset=idx_i)
            assert gen_b[idx_i] == gen_1[0], f"Mismatch for {exit_mode} row {idx_i}: {gen_b[idx_i]} vs {gen_1[0]}"
        if exit_mode != "none":
            assert st_b["mean_loops"] < 4.0, f"Expected early exits for {exit_mode}, got mean_loops={st_b['mean_loops']}"
        else:
            assert abs(st_b["mean_loops"] - 4.0) < 1e-6


def test_quantile_calibration_sharp_distribution() -> None:
    """Verify underflow-free logit-gap quantile calibration hits 1.8, 2.2, 2.8 cleanly."""
    import numpy as np
    rng = np.random.default_rng(7)
    n = 2000
    # High-confidence logit gaps (many > 18 where softmax entropy underflows to 0.0!)
    gap1 = np.abs(rng.normal(14.0, 8.0, size=n))
    gap2 = gap1 + np.abs(rng.normal(2.5, 2.0, size=n))
    gap3 = gap2 + np.abs(rng.normal(1.0, 1.5, size=n))
    trace = {
        "gap1": gap1,
        "gap2": gap2, "agree12": (rng.random(n) < 0.92).astype(np.float32),
        "gap3": gap3, "agree23": (rng.random(n) < 0.95).astype(np.float32),
    }
    for r_tgt, min_p4 in [(1.8, 0.08), (2.2, 0.18), (2.8, 0.35)]:
        lo, hi = 0.0, 1.0
        for _ in range(32):
            mid = 0.5 * (lo + hi)
            mr, _, _ = _simulate_entropy_agree_mean_r(trace, mid)
            if mr > r_tgt:
                lo = mid
            else:
                hi = mid
        mr, probs, _ = _simulate_entropy_agree_mean_r(trace, 0.5 * (lo + hi))
        assert abs(mr - r_tgt) < 0.05, f"target={r_tgt}, got mr={mr:.3f}, probs={probs}"
        assert probs[3] >= min_p4, f"target={r_tgt}, too few R=4 tokens: probs={probs}"


if __name__ == "__main__":
    test_fisher_cosine_shift_and_scale_invariance()
    test_tcre_period2_oscillation_exact_cancellation()
    test_tcre_monotone_contraction_extrapolates_toward_limit()
    test_tcre_cos_continuity_at_zero()
    test_thinking_special_token_answer_parsing()
    test_ragged_kv_batched_equals_b1_and_skips_loops()
    test_quantile_calibration_sharp_distribution()
    print("ALL ROUTE 1 CPU UNIT TESTS PASSED!")
