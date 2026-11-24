"""Lightweight CPU unit tests for scaffolding, backward shims, and data builders."""

from __future__ import annotations

import torch

from loop_research.data_assets import (
    generate_synthetic_paired_twohop,
    map_char_spans_to_blocks,
    normalize_answer_text,
)
from loop_research.ifm_runtime import (
    _group_layer_norm_bwd_affine,
    _group_layer_norm_fwd_affine,
    _group_rms_norm_bwd,
    _group_rms_norm_bwd_affine,
    _group_rms_norm_fwd,
    _group_rms_norm_fwd_affine,
)


def test_group_rms_norm_backward_matches_pytorch_autograd():
    torch.manual_seed(0)
    x = torch.randn(3, 7, 32, dtype=torch.float32, requires_grad=True)
    gamma = torch.randn(32, dtype=torch.float32, requires_grad=True)
    num_groups = 4
    eps = 1e-6

    # Reference autograd
    xr = x.reshape(3, 7, num_groups, 8)
    rstd_ref = torch.rsqrt(xr.pow(2).mean(dim=-1, keepdim=True) + eps)
    y_ref = (xr * rstd_ref * gamma.reshape(1, 1, num_groups, 8)).reshape_as(x)
    dy = torch.randn_like(y_ref)
    gx_ref, gg_ref = torch.autograd.grad(y_ref, (x, gamma), dy)

    # Shim forward + backward
    y_sh, rstd_sh = _group_rms_norm_fwd_affine(x.detach(), 32, num_groups, gamma.detach(), eps)
    gx_sh, gg_sh = _group_rms_norm_bwd_affine(dy, x.detach(), 32, num_groups, rstd_sh, gamma.detach(), False)

    assert torch.allclose(y_sh, y_ref.detach(), atol=1e-5, rtol=1e-5)
    assert torch.allclose(gx_sh, gx_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(gg_sh, gg_ref, atol=1e-5, rtol=1e-5)

    # Non-affine
    x2 = torch.randn(2, 5, 16, dtype=torch.float32, requires_grad=True)
    x2r = x2.reshape(2, 5, 2, 8)
    y2_ref = (x2r * torch.rsqrt(x2r.pow(2).mean(dim=-1, keepdim=True) + eps)).reshape_as(x2)
    dy2 = torch.randn_like(y2_ref)
    (gx2_ref,) = torch.autograd.grad(y2_ref, (x2,), dy2)
    y2_sh, rstd2_sh = _group_rms_norm_fwd(x2.detach(), 16, 2, eps)
    gx2_sh = _group_rms_norm_bwd(dy2, x2.detach(), 16, 2, rstd2_sh, False)
    assert torch.allclose(y2_sh, y2_ref.detach(), atol=1e-5, rtol=1e-5)
    assert torch.allclose(gx2_sh, gx2_ref, atol=1e-5, rtol=1e-5)


def test_group_layer_norm_backward_matches_pytorch_autograd():
    torch.manual_seed(1)
    x = torch.randn(2, 6, 24, dtype=torch.float32, requires_grad=True)
    gamma = torch.randn(24, dtype=torch.float32, requires_grad=True)
    beta = torch.randn(24, dtype=torch.float32, requires_grad=True)
    num_groups = 3
    eps = 1e-5

    xr = x.reshape(2, 6, num_groups, 8)
    mean = xr.mean(dim=-1, keepdim=True)
    var = (xr - mean).pow(2).mean(dim=-1, keepdim=True)
    y_ref = ((xr - mean) * torch.rsqrt(var + eps) * gamma.reshape(1, 1, num_groups, 8) + beta.reshape(1, 1, num_groups, 8)).reshape_as(x)
    dy = torch.randn_like(y_ref)
    gx_ref, gg_ref, gb_ref = torch.autograd.grad(y_ref, (x, gamma, beta), dy)

    y_sh, m_sh, r_sh = _group_layer_norm_fwd_affine(x.detach(), 24, num_groups, gamma.detach(), beta.detach(), eps)
    gx_sh, gg_sh, gb_sh = _group_layer_norm_bwd_affine(
        dy, x.detach(), 24, num_groups, m_sh, r_sh, gamma.detach(), beta.detach(), False
    )
    assert torch.allclose(y_sh, y_ref.detach(), atol=1e-5, rtol=1e-5)
    assert torch.allclose(gx_sh, gx_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(gg_sh, gg_ref, atol=1e-5, rtol=1e-5)
    assert gb_sh is not None and torch.allclose(gb_sh, gb_ref, atol=1e-5, rtol=1e-5)


def test_synthetic_paired_twohop_structure():
    items = generate_synthetic_paired_twohop(n_base_docs=8, seed=42)
    assert len(items) == 32  # 8 base * 2 variants * 2 questions
    # Check Q_A and Q_B share identical prefix_text and disjoint evidence slots
    for i in range(0, len(items), 2):
        qa, qb = items[i], items[i + 1]
        assert qa["prefix_text"] == qb["prefix_text"]
        assert qa["doc_id"] == qb["doc_id"]
        assert set(qa["evidence_slots"]).isdisjoint(set(qb["evidence_slots"]))
        assert qa["gold_answer_norm"] != qb["gold_answer_norm"]
        mapped = map_char_spans_to_blocks(
            qa["prefix_text"], 400, [qa["para_spans"][s] for s in qa["evidence_slots"]], n_blocks=8
        )
        assert len(mapped) >= 2
    assert normalize_answer_text("  The Bergen, Norway! ") == "bergen norway"
