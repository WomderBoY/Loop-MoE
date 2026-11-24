"""Unit tests for Direction 1 (FTRS) geometry, trust-region math, and answer parsing."""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from loop_research.data_assets import extract_boxed, extract_predicted_math_answer, normalize_math_answer
from loop_research.ftrs_stepping import (
    FTRSGate,
    apply_calibrated_step,
    binary_auroc,
    compute_fisher_geometry,
    ftrs_zero_feedback_alpha,
    stack_index_free_features,
    unified_readout_logits,
)


def test_prop34_and_fisher_curvature_exactness() -> None:
    torch.manual_seed(42)
    z0 = torch.randn(16, 100)
    dz = torch.randn(16, 100) * 0.3
    z1 = z0 + dz
    h0 = torch.randn(16, 32)
    d1 = torch.randn(16, 32)
    geom = compute_fisher_geometry(z0, z1, d1, d1, h0)

    # Check exact Prop 3.4: Delta_U(1) == A_r - KL(p0 || p1) for every target token
    y = torch.randint(0, 100, (16,))
    ll0 = F.log_softmax(z0, dim=-1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
    ll1 = F.log_softmax(z1, dim=-1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
    du1 = ll1 - ll0
    slope_a = dz.gather(-1, y.unsqueeze(-1)).squeeze(-1) - geom["mean_dz"]
    err = (du1 - (slope_a - geom["kl_fwd"])).abs().max().item()
    assert err < 1e-5, f"Proposition 3.4 identity failed: max err={err}"

    # Check 2nd-order Taylor match in FP64: KL(p0 || p(z0 + eps*dz)) ~ 0.5 * eps^2 * kappa
    eps = 1e-3
    z0_d, dz_d = z0.double(), dz.double()
    z_eps = z0_d + eps * dz_d
    p0 = F.softmax(z0_d, dim=-1)
    kl_eps = (p0 * (F.log_softmax(z0_d, dim=-1) - F.log_softmax(z_eps, dim=-1))).sum(dim=-1)
    quad = 0.5 * (eps ** 2) * geom["kappa"].double()
    rel = ((kl_eps - quad).abs() / quad.clamp_min(1e-12)).max().item()
    assert rel < 1e-3, f"Fisher curvature 2nd-order Taylor mismatch: rel={rel}"


def test_trust_region_and_manifold_rms_preservation() -> None:
    kappa = torch.tensor([0.01, 0.30, 10.0])
    cos_m = torch.tensor([0.8, 0.5, -0.8])
    alpha = ftrs_zero_feedback_alpha(kappa, cos_m, tau_fb=0.15, use_momentum=True)
    assert math.isclose(alpha[0].item(), 1.0, abs_tol=1e-5)
    assert math.isclose(alpha[1].item(), 1.0, abs_tol=1e-5)
    assert alpha[2].item() < 0.05

    h0 = torch.randn(4, 64)
    h1 = torch.randn(4, 64)
    a = torch.tensor([0.2, 0.5, 0.8, 1.3])
    h_cal = apply_calibrated_step(h0, h1, a, manifold="rms_match")
    rms_target = torch.sqrt(h1.pow(2).mean(dim=-1))
    rms_cal = torch.sqrt(h_cal.pow(2).mean(dim=-1))
    assert torch.allclose(rms_target, rms_cal, atol=1e-5)


def test_unified_readout_and_gate() -> None:
    torch.manual_seed(7)
    zf = torch.randn(4, 50)
    zp = torch.randn(4, 50)
    zc = zp + 0.1 * torch.randn(4, 50)
    h0 = torch.randn(4, 16)
    d1 = torch.randn(4, 16)
    geom = compute_fisher_geometry(zp, zc, d1, d1, h0)
    feats = stack_index_free_features(geom)
    assert feats.shape == (4, 5)
    gate = FTRSGate(in_dim=5, hidden_dim=16, max_alpha=1.8)
    gate.set_norm_stats(feats)
    pred = gate(feats)
    assert pred.shape == (4,)
    assert torch.all((pred > 0.0) & (pred < 1.8))

    # High curvature case should damp and NOT extrapolate
    z_out = unified_readout_logits(
        zf, zp, zc,
        alpha_fb=torch.tensor([1.0, 0.3, 1.0, 0.1]),
        kappa_step=geom["kappa"],
        cos_mom=torch.tensor([0.9, 0.9, -0.5, -0.9]),
    )
    assert z_out.shape == (4, 50)
    assert torch.isfinite(z_out).all()


def test_math_answer_extractors_and_auroc() -> None:
    assert extract_boxed(r"Thus the answer is \boxed{\frac{3}{4}}.") == r"\frac{3}{4}"
    assert extract_predicted_math_answer("We get 12 + 30 = 42.\n#### 42", mode="gsm8k") == "42"
    assert extract_predicted_math_answer("<think>check 10</think>\nFinal: \\boxed{1,250.0}", mode="gsm8k") == "1250"
    assert normalize_math_answer("$42.00") == "42"
    assert math.isclose(binary_auroc([0.1, 0.4, 0.8, 0.9], [0, 0, 1, 1]), 1.0)
