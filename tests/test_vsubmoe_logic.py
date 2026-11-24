"""Unit tests for Direction 2 (V-SubMoE) velocity features, parameter cap, router math, and SVD concentration."""

from __future__ import annotations

import torch

from loop_research.vsubmoe_pilot import (
    VELOCITY_FEAT_DIM,
    SubspaceMoEAdapterHost,
    analyze_subspace_concentration,
    compute_velocity_features,
    pure_torch_topk_routing,
)


def test_velocity_features_index_free_and_bounded() -> None:
    torch.manual_seed(0)
    h0 = torch.randn(12, 64)
    h1 = h0 + 0.5 * torch.randn(12, 64)
    h2 = h1 + 0.05 * torch.randn(12, 64)
    psi0 = compute_velocity_features(h0, None, None)
    psi1 = compute_velocity_features(h1, h0, h0)
    psi2 = compute_velocity_features(h2, h1, h0)
    assert psi0.shape == (12, VELOCITY_FEAT_DIM)
    assert torch.isfinite(psi0).all() and torch.isfinite(psi1).all() and torch.isfinite(psi2).all()
    # Relative velocity f2 should shrink as updates converge
    assert psi2[:, 2].mean().item() < psi1[:, 2].mean().item()


def test_strict_parameter_matching_and_extrapolation_forward() -> None:
    torch.manual_seed(1)
    selected = [(2, 0), (2, 1), (7, 0), (7, 2)]
    tracked = [2, 7]
    ref = SubspaceMoEAdapterHost(
        selected, tracked, mode="shared_plus_scalar",
        rank_per_loop=4, train_loops=4, n_experts=8, model_dim=64, inter_dim=32,
    )
    cap = ref.num_trainable_params()

    for mode, use_rtr in [
        ("per_loop", False),
        ("pure_shared", False),
        ("loop_dropout", False),
        ("shared_plus_diag_idx", False),
        ("velocity_submoe", False),
        ("velocity_submoe_ld", False),
        ("velocity_submoe_full", True),
    ]:
        host = SubspaceMoEAdapterHost(
            selected, tracked, mode=mode,
            rank_per_loop=4, train_loops=4, n_experts=8, model_dim=64, inter_dim=32,
            use_router_adapter=use_rtr, target_param_cap=cap,
        )
        assert host.num_trainable_params() <= cap, f"{mode} exceeded cap: {host.num_trainable_params()} > {cap}"

        # Forward + backward at BOTH in-depth (visit_r=2) and extrapolated (visit_r=7 > train_loops=4)!
        hidden = torch.randn(10, 32, dtype=torch.bfloat16)
        y = torch.randn(10, 64, dtype=torch.bfloat16)
        psi = torch.randn(10, VELOCITY_FEAT_DIM, dtype=torch.float32)
        group_sizes = [3, 2, 2, 1, 1, 1, 0, 0]
        for vr in [0, 2, 7]:
            out = host.apply_expert_delta(2, vr, hidden, y, group_sizes, permuted_psi=psi)
            assert out.shape == y.shape and torch.isfinite(out).all()
            if use_rtr:
                prev_m = torch.zeros(10, 8)
                prev_m[:, 0] = 1.0
                dl = host.router_delta_logits(2, psi, prev_m, bsz=1, slen=10)
                assert dl is not None and dl.shape == (1, 10, 8)
                (out.float().sum() + dl.sum()).backward()
            else:
                out.float().sum().backward()


def test_pure_torch_topk_routing_and_subspace_concentration() -> None:
    torch.manual_seed(2)
    x = torch.randn(1, 8, 32, dtype=torch.bfloat16)
    w = torch.randn(16, 32, dtype=torch.bfloat16)
    bias = torch.randn(16, dtype=torch.float32)
    dl = torch.zeros(1, 8, 16, dtype=torch.float32, requires_grad=True)
    scores, idx, tpe, aux = pure_torch_topk_routing(x, w, 16, 2, "sigmoid", bias, 2.5, delta_logits=dl)
    assert scores.shape == (1, 8, 2) and idx.shape == (1, 8, 2) and int(tpe.sum().item()) == 16
    scores.float().sum().backward()
    assert dl.grad is not None and dl.grad.abs().sum().item() > 0

    # Synthetic shared-subspace gradients with different per-loop signs/weights
    u = torch.linalg.qr(torch.randn(32, 6))[0]
    v = torch.linalg.qr(torch.randn(24, 6))[0]
    ga = {0: torch.zeros(2, 4, 32, 24)}
    gb = {0: torch.zeros(2, 4, 32, 24)}
    for r in range(4):
        c = torch.tensor([1.0, -1.0, 0.5, -0.5, 0.2, 0.8]) * ((-1) ** r)
        signal = u @ torch.diag(c) @ v.T
        ga[0][0, r] = signal + 0.05 * torch.randn(32, 24)
        gb[0][0, r] = signal + 0.05 * torch.randn(32, 24)
    diag = analyze_subspace_concentration(ga, gb, [(0, 0)], rank_per_loop=2, n_loops=4)
    assert diag["ev_shared_subspace_full"] > diag["ev_per_loop_split_mean"]
