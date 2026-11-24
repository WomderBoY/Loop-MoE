"""Lightweight CPU unit tests for Idea 2 conflict scoring and matched-group selection."""

from __future__ import annotations

import numpy as np
import torch

from loop_research.idea2_expert_unbinding import (
    ExpertLoRAHost,
    score_expert_conflicts,
    select_group_experts,
)


def test_unbiased_conflict_scoring_separates_true_conflict_from_scalar_scaling():
    torch.manual_seed(7)
    n_exp, n_loops, d, inter = 6, 4, 8, 12
    w2 = torch.randn(n_exp, d, inter)
    ga = torch.zeros(n_exp, n_loops, d, inter)
    gb = torch.zeros(n_exp, n_loops, d, inter)

    # Expert 0: identical gradient across loops -> zero conflict
    v0 = torch.randn(d, inter)
    for r in range(n_loops):
        ga[0, r] = v0
        gb[0, r] = v0

    # Expert 1: pure scalar scaling along shared direction -> S_raw > 0, S_orth_mean ~ 0
    v1 = torch.randn(d, inter)
    scales = [0.2, 0.8, 1.5, 2.5]
    for r, sc in enumerate(scales):
        ga[1, r] = sc * v1
        gb[1, r] = sc * v1

    # Expert 2: true orthogonal directional conflict across loops -> S_orth_both >> 0
    dirs = [torch.randn(d, inter) for _ in range(n_loops)]
    for r in range(n_loops):
        ga[2, r] = dirs[r]
        gb[2, r] = dirs[r]

    counts = {0: np.full((n_exp, n_loops), 50, dtype=np.int64)}
    recs = score_expert_conflicts({0: ga}, {0: gb}, counts, counts, {0: w2})
    assert abs(recs[0]["s_raw"]) < 1e-5
    assert recs[1]["s_raw"] > 1.0 and abs(recs[1]["s_orth_mean"]) < 1e-4
    assert recs[2]["s_orth_both"] > 10.0
    assert recs[2]["cross_split_dir_cos"] > 0.99


def test_parameter_matching_across_conditions():
    # 6 layers x 25 experts x 4 loops: 30 * 4 * 5 == 150 * 4 * 1 == 25 * 4 * 6 == 600 rank slots
    recs = []
    for lid in [2, 7, 12, 17, 22, 26]:
        for e in range(25):
            recs.append(
                {
                    "layer": lid,
                    "expert": e,
                    "s_orth_both": float(e) + 0.1 * lid,
                    "grad_norm_mean": float(1 + (e % 5)),
                    "token_count": 100 + 10 * (e % 5),
                }
            )
    grps = select_group_experts(recs, k_per_layer=5)
    h_sel = ExpertLoRAHost(grps["conflict_top"], mode="per_loop", rank=5, n_loops=4, model_dim=64, inter_dim=96)
    h_unif = ExpertLoRAHost(grps["all_experts"], mode="per_loop", rank=1, n_loops=4, model_dim=64, inter_dim=96)
    h_recon = ExpertLoRAHost(grps["recon_layer"], mode="per_loop", rank=6, n_loops=4, model_dim=64, inter_dim=96)
    h_sh = ExpertLoRAHost(grps["conflict_top"], mode="shared_plus_scalar", rank=20, n_loops=4, model_dim=64, inter_dim=96)
    assert h_sel.num_trainable_params() == h_unif.num_trainable_params()
    assert h_sel.num_trainable_params() == h_recon.num_trainable_params()
    assert h_sh.num_trainable_params() == h_sel.num_trainable_params() + 30 * 4 * 2
