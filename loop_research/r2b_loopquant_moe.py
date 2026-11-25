"""Round 2-B: LoopQuant-MoE — Trajectory-Calibrated AWQ & Stage-Sensitivity Mixed-Bit PTQ.

Child of R1 Probe 1 (7c0b4300-73ae-49ad-9d1a-db9958883b43).
Evaluates on IFM Ouro-MoE (28L x 4 loops) & IFM Huginn-MoE (8 + 12L x 8 loops + 8):
  1. Single-pass Trajectory Activation & Sensitivity Calibration (calib_a n=16 only):
     - Per-(layer, visit, expert) input RMS (for W1, W3), intermediate SwiGLU RMS (for W2),
       weighted output energy, and routing counts.
     - Per-visit routed-MoE sensitivity alpha_{l,v} = DeltaNLL(skip_moe at visit (l,v)) in nats.
  2. Trajectory-Calibrated AWQ (addressing 1.7x cross-loop salient channel scale drift):
     Compares RTN vs Loop0-AWQ vs Standard-Pool-AWQ vs StageSens-Traj-AWQ vs PeakCh-Traj-AWQ
     at uniform W3-g128, W2-g128, and W2-g64.
  3. Stage-Sensitivity-Weighted Tri-Level {2, 3, 4}-Bit Knapsack Mixed-Precision Allocation:
     Uses dimensionless fractional stage perturbation D(l,e,b) = sum_v alpha_{l,v} * MSE_{l,v,e}(b) / Energy_{l,v},
     naturally capturing U-shaped boundary importance (Ouro loop_0/loop_3 AND Huginn prelude/coda)
     without layer-norm blowup or monotonic decay starvation!
     Compares at exact iso-bit 2.25b, 2.50b, and 3.00b against Global-Freq (ProQuant-style),
     Max-Stage-Freq, and Unweighted-Recon-MSE (ablation removing loop stage sensitivity).
  4. Fast Batched Evaluation (exact B=1 vs B>1 parity verified at startup):
     Held-out NLL (n=64: 32 WikiText + 32 HumanEval) + Full ARC-Challenge (1,172)
     + ARC-Easy (600) + OpenBookQA (500) = 2,272 MCQ items, plus expert VRAM footprint.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from loop_research import orx_run as orx
from loop_research.data_assets import (
    build_disjoint_text_code_splits,
    load_arc_challenge_samples,
    load_arc_easy_samples,
    load_openbookqa_samples,
)
from loop_research.gpu_utils import acquire_idle_gpu
from loop_research.ifm_runtime import bootstrap_xllm, load_ifm_model_for_autograd
from loop_research.probe1_loopmoe_quant import fake_quantize_asym_group, paired_bootstrap_ci


# ---------------------------------------------------------------------------
# Visit Topology (Exact xLLM Convention: rec = range(loop_start, loop_end))
# ---------------------------------------------------------------------------

def build_visit_topology(
    n_layers: int,
    loop_start: int,
    loop_end: int,
    n_loops: int,
) -> list[dict[str, Any]]:
    visits: list[dict[str, Any]] = []
    for lid in range(0, loop_start):
        visits.append({"vid": len(visits), "layer_id": lid, "loop_idx": 0, "stage": "prelude", "is_rec": False})
    for r in range(n_loops):
        for lid in range(loop_start, loop_end):
            visits.append({"vid": len(visits), "layer_id": lid, "loop_idx": r, "stage": f"loop_{r}", "is_rec": True})
    for lid in range(loop_end, n_layers):
        visits.append({"vid": len(visits), "layer_id": lid, "loop_idx": 0, "stage": "coda", "is_rec": False})
    return visits


# ---------------------------------------------------------------------------
# Vectorized Per-Expert AWQ Search & Exact Weighted MSE Computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def awq_quantize_expert_bank(
    w_bank: torch.Tensor,
    scale_rms: torch.Tensor,
    obj_rms: torch.Tensor,
    bits: int,
    group_size: int = 128,
    *,
    alphas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized AWQ per-channel scale search over all E experts of a layer simultaneously.

    Args:
        w_bank: [E, Out, In] BF16/FP32 weight tensor on GPU.
        scale_rms: [E, In] activation RMS used to construct candidate smoothing scales s = a^alpha.
        obj_rms: [E, In] activation RMS defining the reconstruction objective ||(W - W_q) diag(obj_rms)||_F^2.
        bits: Target bit-width in {2, 3, 4, 16}.
        group_size: Asymmetric quantization group size along In.
        alphas: Grid of AWQ scaling exponents.

    Returns:
        w_best: [E, Out, In] dequantized weight tensor in w_bank.dtype.
        best_err: [E] FP32 weighted squared error per expert.
    """
    E = w_bank.shape[0]
    if bits >= 16:
        return w_bank.clone(), torch.zeros(E, device=w_bank.device, dtype=torch.float32)

    wf = w_bank.float()
    a_sc = scale_rms.float().clamp_min(1e-6)
    a_ob = obj_rms.float().clamp_min(0.0)

    best_err = torch.full((E,), float("inf"), device=w_bank.device, dtype=torch.float32)
    best_w = wf.clone()

    for alpha in alphas:
        if alpha == 0.0:
            w_deq = fake_quantize_asym_group(wf, bits, group_size)
        else:
            s = a_sc.pow(alpha)
            s_max = s.amax(dim=-1, keepdim=True)
            s_min = s.amin(dim=-1, keepdim=True)
            norm = (s_max * s_min).sqrt().clamp_min(1e-6)
            s = (s / norm).clamp(1e-3, 1e3)
            w_scaled = wf * s.unsqueeze(1)
            w_deq = fake_quantize_asym_group(w_scaled, bits, group_size) / s.unsqueeze(1)

        diff = (wf - w_deq) * a_ob.unsqueeze(1)
        err = diff.pow(2).sum(dim=(1, 2))  # [E]
        improve = err < best_err
        if improve.any():
            best_err = torch.where(improve, err, best_err)
            best_w = torch.where(improve[:, None, None], w_deq, best_w)

    return best_w.to(w_bank.dtype), best_err


# ---------------------------------------------------------------------------
# Tri-Level {2, 3, 4}-Bit Mixed-Precision Knapsack Solver (CPU Unit-Testable!)
# ---------------------------------------------------------------------------

def solve_trilevel_bit_allocation(
    distortions: np.ndarray,
    target_avg_bits: float,
    *,
    bit_choices: tuple[int, int, int] = (2, 3, 4),
) -> np.ndarray:
    """Allocate bits in {2, 3, 4} per (layer, expert) to minimize sum_{l,e} D[l, e, b_idx]
    subject to mean(bits) == target_avg_bits.

    Args:
        distortions: Array of shape [L, E, 3] giving estimated loss at bits (2, 3, 4).
        target_avg_bits: Desired average bit-width in [2.0, 4.0].

    Returns:
        bits_matrix: Int array of shape [L, E] with values in {2, 3, 4}.
    """
    L, E, n_b = distortions.shape
    assert n_b == 3 and bit_choices == (2, 3, 4)
    N = L * E
    d2 = distortions[:, :, 0].reshape(N).astype(np.float64)
    d3 = np.minimum(d2, distortions[:, :, 1].reshape(N).astype(np.float64))
    d4 = np.minimum(d3, distortions[:, :, 2].reshape(N).astype(np.float64))

    total_budget = int(round(float(target_avg_bits) * N))
    extra_bits = max(0, min(2 * N, total_budget - 2 * N))
    bits = np.full(N, 2, dtype=np.int32)
    if extra_bits == 0:
        return bits.reshape(L, E)
    if extra_bits >= 2 * N:
        return np.full((L, E), 4, dtype=np.int32)

    # Exact greedy over convex-hull marginal gains:
    # For each item i, step1 = d2 - d3 (1 bit), step2 = d3 - d4 (1 bit).
    # If step1 >= step2 (concave benefit), two independent 1-bit upgrades.
    # If step2 > step1 (non-concave jump), bundle as a 2-bit upgrade with avg gain (d2 - d4)/2.
    events: list[tuple[float, int, int]] = []  # (-gain_per_bit, cost_bits, item_idx)
    for i in range(N):
        g1 = max(0.0, float(d2[i] - d3[i]))
        g2 = max(0.0, float(d3[i] - d4[i]))
        if g1 >= g2:
            events.append((-g1, 1, i))
            events.append((-g2, 1, i))
        else:
            events.append((-(g1 + g2) / 2.0, 2, i))

    events.sort(key=lambda t: (t[0], t[2]))
    rem = extra_bits
    for neg_g, cost, idx in events:
        if rem <= 0:
            break
        if cost == 1:
            if bits[idx] < 4:
                bits[idx] += 1
                rem -= 1
        else:
            if rem >= 2 and bits[idx] == 2:
                bits[idx] = 4
                rem -= 2
            elif rem == 1 and bits[idx] == 2:
                # Best 1-bit fallback among remaining un-upgraded items handled below
                continue

    # If rem == 1 remained because last selected event was cost==2, greedily assign best 1-bit step
    if rem == 1:
        best_i = -1
        best_g = -1.0
        for i in range(N):
            if bits[i] == 2:
                g = float(d2[i] - d3[i])
            elif bits[i] == 3:
                g = float(d3[i] - d4[i])
            else:
                continue
            if g > best_g:
                best_g = g
                best_i = i
        if best_i >= 0:
            bits[best_i] += 1

    return bits.reshape(L, E)


def rank_to_trilevel_bits(
    scores: np.ndarray,
    target_avg_bits: float,
) -> np.ndarray:
    """Deterministic score-rank tri-level {2, 3, 4}-bit allocation for frequency baselines.

    Protects top fraction to 4-bit, next tier to 3-bit, rest 2-bit, matching target_avg_bits exactly.
    - For 2.25b: 12.5% W4 + 0% W3? Better: split 1/3 of extra bits into W4 upgrades and rest W3
      -> actually standard ProQuant-style continuous top-k:
         * 2.25b: top 5% W4 + next 15% W3 + 80% W2 -> avg = 2 + 0.10 + 0.15 = 2.25b
         * 2.50b: top 12.5% W4 + next 25% W3 + 62.5% W2 -> avg = 2 + 0.25 + 0.25 = 2.50b
         * 3.00b: top 25% W4 + next 50% W3 + 25% W2 -> avg = 2 + 0.50 + 0.50 = 3.00b
    """
    L, E = scores.shape
    N = L * E
    extra = max(0, min(2 * N, int(round((target_avg_bits - 2.0) * N))))
    # Allocate half of extra budget to promoting top experts all the way to 4-bit, half to 3-bit
    n4 = min(N, extra // 4)
    n3 = min(N - n4, extra - 2 * n4)
    order = sorted(
        [(float(scores[l, e]), l, e) for l in range(L) for e in range(E)],
        key=lambda t: (-t[0], t[1], t[2]),
    )
    bits = np.full((L, E), 2, dtype=np.int32)
    for _, l, e in order[:n4]:
        bits[l, e] = 4
    for _, l, e in order[n4 : n4 + n3]:
        bits[l, e] = 3
    return bits


def compute_expert_vram_gb(
    bits_matrix: np.ndarray,
    model_dim: int,
    inter_dim: int,
    group_size: int = 128,
) -> dict[str, float]:
    """Compute exact expert weight VRAM footprint (including FP16 group scale + zero-point metadata)."""
    n_params_per_expert = 3 * model_dim * inter_dim  # W1, W2, W3
    meta_bits_per_param = 32.0 / float(group_size)  # fp16 scale + fp16 zp per group
    avg_bits = float(np.mean(bits_matrix))
    eff_bits = 16.0 if avg_bits >= 15.9 else (avg_bits + meta_bits_per_param)
    total_params = int(bits_matrix.size) * n_params_per_expert
    vram_gb = (total_params * eff_bits) / (8.0 * (1024.0 ** 3))
    bf16_gb = (total_params * 16.0) / (8.0 * (1024.0 ** 3))
    return {
        "avg_bits": avg_bits,
        "eff_bits_with_meta": eff_bits,
        "expert_vram_gb": vram_gb,
        "bf16_expert_vram_gb": bf16_gb,
        "compression_ratio": bf16_gb / max(vram_gb, 1e-9),
    }


# ---------------------------------------------------------------------------
# Fast Batched Evaluation (Shared Exact Right-Padded MCQ & NLL)
# ---------------------------------------------------------------------------

def prepare_mcq_encoded(samples: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    encoded: list[dict[str, Any]] = []
    for q_idx, item in enumerate(samples):
        prompt = f"Question: {item['question']}\nAnswer:"
        p_ids = tokenizer.encode(prompt, bos=True, eos=False)
        p_len = len(p_ids)
        choices_enc: list[tuple[list[int], int]] = []
        for c_txt in item["choice_texts"]:
            c_ids = tokenizer.encode(" " + c_txt, bos=False, eos=False)
            choices_enc.append((p_ids + c_ids, p_len) if c_ids else ([], p_len))
        encoded.append({"q_idx": q_idx, "gold_idx": int(item["gold_idx"]), "choices": choices_enc})
    return encoded


@torch.inference_mode()
def eval_nll_batched(
    model: Any,
    seqs: list[dict[str, Any]],
    device: torch.device,
    *,
    batch_size: int = 8,
) -> dict[str, Any]:
    per_seq: list[float] = []
    text_nlls: list[float] = []
    code_nlls: list[float] = []
    for s0 in range(0, len(seqs), batch_size):
        batch = seqs[s0 : s0 + batch_size]
        toks = torch.tensor([b["tokens"] for b in batch], device=device, dtype=torch.long)
        inp, tgt = toks[:, :-1], toks[:, 1:]
        logits, _, _ = model(inp, multi_segments=False, cache=None)
        loss_tok = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            tgt.reshape(-1),
            reduction="none",
        ).reshape(len(batch), -1)
        for item, val in zip(batch, loss_tok.mean(dim=-1).cpu().tolist()):
            per_seq.append(float(val))
            (text_nlls if item.get("domain") == "text" else code_nlls).append(float(val))
    return {
        "nll": float(np.mean(per_seq)),
        "text_nll": float(np.mean(text_nlls)) if text_nlls else 0.0,
        "code_nll": float(np.mean(code_nlls)) if code_nlls else 0.0,
        "per_seq": per_seq,
    }


@torch.inference_mode()
def eval_mcq_batched(
    model: Any,
    encoded_items: list[dict[str, Any]],
    device: torch.device,
    *,
    batch_size: int = 32,
    pad_token_id: int = 0,
) -> tuple[float, list[int]]:
    flat: list[tuple[int, int, list[int], int]] = []
    n_q = len(encoded_items)
    scores_by_q: list[list[float]] = [[-1e9] * len(it["choices"]) for it in encoded_items]
    for it in encoded_items:
        qi = it["q_idx"]
        for ci, (full_ids, p_len) in enumerate(it["choices"]):
            if len(full_ids) > p_len:
                flat.append((qi, ci, full_ids, p_len))
    flat.sort(key=lambda x: len(x[2]))

    for s0 in range(0, len(flat), batch_size):
        chunk = flat[s0 : s0 + batch_size]
        bsz = len(chunk)
        max_inp_len = max(len(x[2]) - 1 for x in chunk)
        inp = torch.full((bsz, max_inp_len), pad_token_id, device=device, dtype=torch.long)
        for bi, (_qi, _ci, full_ids, _plen) in enumerate(chunk):
            inp[bi, : len(full_ids) - 1] = torch.tensor(full_ids[:-1], device=device, dtype=torch.long)
        logits, _, _ = model(inp, multi_segments=False, cache=None)
        for bi, (qi, ci, full_ids, p_len) in enumerate(chunk):
            c_len = len(full_ids) - p_len
            sl_logits = logits[bi, p_len - 1 : p_len - 1 + c_len, :].float()
            lp = F.log_softmax(sl_logits, dim=-1)
            tgt = torch.tensor(full_ids[p_len:], device=device, dtype=torch.long)
            scores_by_q[qi][ci] = float(lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean().item())

    flags = [1 if int(np.argmax(scores_by_q[qi])) == encoded_items[qi]["gold_idx"] else 0 for qi in range(n_q)]
    return float(np.mean(flags)), flags


# ---------------------------------------------------------------------------
# Trajectory Activation & Per-Visit Sensitivity Calibration
# ---------------------------------------------------------------------------

@torch.inference_mode()
def calibrate_trajectory_quant(
    model: Any,
    topo: list[dict[str, Any]],
    calib_seqs: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    """Collect per-(layer, visit, expert) input & SwiGLU-mid squared activation sums,
    weighted output energy, counts, AND per-visit skip-MoE NLL sensitivity alpha_{l,v}.
    """
    n_layers = len(model.layers)
    n_experts = int(model.layers[0].moe.experts.n_local_experts)
    D = int(model.layers[0].moe.experts.model_dim)
    I = int(model.layers[0].moe.experts.inter_dim)
    max_v = int(model.loop_times)
    topo_map = {(d["layer_id"], d["loop_idx"]): d["vid"] for d in topo}
    V = len(topo)

    counts = np.zeros((n_layers, max_v, n_experts), dtype=np.int64)
    in_sq_sum = np.zeros((n_layers, max_v, n_experts, D), dtype=np.float64)
    mid_sq_sum = np.zeros((n_layers, max_v, n_experts, I), dtype=np.float64)
    out_energy = np.zeros((n_layers, max_v, n_experts), dtype=np.float64)
    visit_total_energy = np.zeros((n_layers, max_v), dtype=np.float64)

    visits = [0] * n_layers
    orig_moe = []
    try:
        for lid, layer in enumerate(model.layers):
            moe = layer.moe
            orig_moe.append((moe, moe.forward))

            def make_calib(lid_c: int, moe_c: Any):
                em = moe_c.experts
                w1_l = em.weight1.detach().view(n_experts, I, D)
                w2_l = em.weight2.detach().view(n_experts, D, I)
                w3_l = em.weight3.detach().view(n_experts, I, D)

                def wrapped(x_in, load_balancing_type=None):
                    v = visits[lid_c]
                    visits[lid_c] = v + 1
                    residual = x_in
                    x = moe_c.norm(x_in)
                    y1 = moe_c.fc2(F.silu(moe_c.fc1(x)) * moe_c.fc3(x)) if moe_c.fc1 is not None else None
                    routing_scores, routing_indices, _tpe, aux_loss = moe_c.router(x, load_balancing_type)

                    bsz, slen, dim = x_in.shape
                    n_tok = bsz * slen
                    x_2d = x.reshape(n_tok, dim)
                    idx_2d = routing_indices.reshape(n_tok, -1)
                    w_2d = routing_scores.reshape(n_tok, -1).float()
                    topk = idx_2d.shape[-1]

                    tok_ids = torch.arange(n_tok, device=x_in.device).unsqueeze(1).expand(n_tok, topk).reshape(-1)
                    exp_ids = idx_2d.reshape(-1)
                    w_flat = w_2d.reshape(-1)

                    order = torch.argsort(exp_ids, stable=False)
                    tok_s = tok_ids.index_select(0, order)
                    exp_s = exp_ids.index_select(0, order)
                    w_s = w_flat.index_select(0, order)
                    gs = torch.bincount(exp_s, minlength=n_experts).tolist()

                    x_perm = x_2d.index_select(0, tok_s)
                    y_perm_parts = []
                    off = 0
                    for e in range(n_experts):
                        cnt = int(gs[e])
                        if cnt == 0:
                            continue
                        xe = x_perm[off : off + cnt]
                        we = w_s[off : off + cnt]
                        h_mid = F.silu(F.linear(xe, w1_l[e])) * F.linear(xe, w3_l[e])
                        ye = F.linear(h_mid, w2_l[e])
                        y_perm_parts.append(ye)

                        if v < max_v:
                            counts[lid_c, v, e] += cnt
                            in_sq_sum[lid_c, v, e] += xe.float().pow(2).sum(dim=0).cpu().numpy()
                            mid_sq_sum[lid_c, v, e] += h_mid.float().pow(2).sum(dim=0).cpu().numpy()
                            en = float((we.pow(2) * ye.float().pow(2).sum(dim=-1)).sum().item())
                            out_energy[lid_c, v, e] += en
                            visit_total_energy[lid_c, v] += en
                        off += cnt

                    y_perm = torch.cat(y_perm_parts, dim=0) if y_perm_parts else torch.empty_like(x_perm)
                    y2 = torch.zeros_like(x_2d)
                    y2.index_add_(0, tok_s, y_perm * w_s.to(y_perm.dtype).unsqueeze(-1))
                    y = (y1 + y2.reshape(bsz, slen, dim)) if y1 is not None else y2.reshape(bsz, slen, dim)
                    return moe_c.residual(y, residual, c=y), aux_loss
                return wrapped

            moe.forward = make_calib(lid, moe)

        for s0 in range(0, len(calib_seqs), 8):
            batch = calib_seqs[s0 : s0 + 8]
            toks = torch.tensor([b["tokens"] for b in batch], device=device, dtype=torch.long)
            for l in range(n_layers):
                visits[l] = 0
            _ = model(toks[:, :-1], multi_segments=False, cache=None)
    finally:
        for moe, fn in orig_moe:
            moe.forward = fn

    # Measure per-visit routed-MoE sensitivity alpha_{l,v} = DeltaNLL(skip routed MoE at (l,v)) on calib[:8]
    calib_sub = calib_seqs[:8]
    base_cal = eval_nll_batched(model, calib_sub, device, batch_size=8)["nll"]
    visit_sens = np.zeros((n_layers, max_v), dtype=np.float64)

    for d in topo:
        lid_t = d["layer_id"]
        v_t = d["loop_idx"]
        moe_t = model.layers[lid_t].moe
        orig_t = moe_t.forward
        ctr = [0]

        def skip_once(x_in, load_balancing_type=None, _orig=orig_t, _moe=moe_t, _vt=v_t, _ctr=ctr):
            cur_v = _ctr[0]
            _ctr[0] = cur_v + 1
            if cur_v == _vt:
                residual = x_in
                x = _moe.norm(x_in)
                y1 = _moe.fc2(F.silu(_moe.fc1(x)) * _moe.fc3(x)) if _moe.fc1 is not None else torch.zeros_like(x)
                return _moe.residual(y1, residual, c=y1), None
            return _orig(x_in, load_balancing_type)

        try:
            moe_t.forward = skip_once
            # Note: batch_size=8 runs all 8 calib sequences in a SINGLE forward pass so ctr resets cleanly!
            ctr[0] = 0
            toks = torch.tensor([b["tokens"] for b in calib_sub], device=device, dtype=torch.long)
            logits, _, _ = model(toks[:, :-1], multi_segments=False, cache=None)
            nll_s = float(F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), toks[:, 1:].reshape(-1)).item())
            visit_sens[lid_t, v_t] = max(1e-5, nll_s - base_cal)
        finally:
            moe_t.forward = orig_t

    return {
        "counts": counts,
        "in_sq_sum": in_sq_sum,
        "mid_sq_sum": mid_sq_sum,
        "out_energy": out_energy,
        "visit_total_energy": visit_total_energy,
        "visit_sens": visit_sens,
        "base_calib_nll": base_cal,
    }


# ---------------------------------------------------------------------------
# Trajectory AWQ Bank & Distortion Table Builder (GPU Vectorized!)
# ---------------------------------------------------------------------------

class LoopQuantManager:
    """Stores BF16 CPU backups and applies RTN / Loop0-AWQ / Pool-AWQ / StageSens-Traj-AWQ / PeakCh-Traj-AWQ."""

    def __init__(self, model: Any, topo: list[dict[str, Any]], calib: dict[str, Any]) -> None:
        self.model = model
        self.topo = topo
        self.calib = calib
        self.n_layers = len(model.layers)
        self.n_experts = int(model.layers[0].moe.experts.n_local_experts)
        self.D = int(model.layers[0].moe.experts.model_dim)
        self.I = int(model.layers[0].moe.experts.inter_dim)
        self.max_v = int(model.loop_times)

        self.backups: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for layer in model.layers:
            em = layer.moe.experts
            self.backups.append(
                (
                    em.weight1.detach().cpu().clone(),
                    em.weight2.detach().cpu().clone(),
                    em.weight3.detach().cpu().clone(),
                )
            )

        # Precompute per-layer [E, D] and [E, I] activation RMS profiles for each calibration mode!
        self.in_rms: dict[str, np.ndarray] = {}
        self.mid_rms: dict[str, np.ndarray] = {}
        self._build_rms_profiles()

    def _build_rms_profiles(self) -> None:
        counts = self.calib["counts"]          # [L, V, E]
        in_sq = self.calib["in_sq_sum"]        # [L, V, E, D]
        mid_sq = self.calib["mid_sq_sum"]      # [L, V, E, I]
        vsens = self.calib["visit_sens"]       # [L, V]
        v_en = self.calib["visit_total_energy"]  # [L, V]
        L, max_v, E, D = in_sq.shape
        I = mid_sq.shape[-1]

        # Valid visits per layer
        layer_visits = [
            sorted(set(d["loop_idx"] for d in self.topo if d["layer_id"] == l))
            for l in range(L)
        ]

        for mode in ("loop0", "pool", "stagesens", "peakch"):
            arr_in = np.ones((L, E, D), dtype=np.float32)
            arr_mid = np.ones((L, E, I), dtype=np.float32)
            for l in range(L):
                vlist = layer_visits[l]
                for e in range(E):
                    tot_c = int(counts[l, vlist, e].sum())
                    if tot_c <= 0:
                        continue
                    if mode == "loop0":
                        v0 = vlist[0]
                        c0 = int(counts[l, v0, e])
                        if c0 > 0:
                            arr_in[l, e] = np.sqrt(np.maximum(in_sq[l, v0, e] / c0, 1e-12))
                            arr_mid[l, e] = np.sqrt(np.maximum(mid_sq[l, v0, e] / c0, 1e-12))
                        else:
                            # Blind in loop 0 -> fallback to uniform ones (true loop0 blindness!)
                            arr_in[l, e] = 1.0
                            arr_mid[l, e] = 1.0
                    elif mode == "pool":
                        arr_in[l, e] = np.sqrt(np.maximum(in_sq[l, vlist, e].sum(axis=0) / tot_c, 1e-12))
                        arr_mid[l, e] = np.sqrt(np.maximum(mid_sq[l, vlist, e].sum(axis=0) / tot_c, 1e-12))
                    elif mode == "stagesens":
                        # Weight each visit v by alpha_{l,v} / (Energy_{l,v} + eps) * in_sq[l,v,e]
                        # (minimizes exact stage-sensitivity-weighted fractional output perturbation!)
                        num_in = np.zeros(D, dtype=np.float64)
                        num_mid = np.zeros(I, dtype=np.float64)
                        den = 0.0
                        for v in vlist:
                            cv = int(counts[l, v, e])
                            if cv <= 0:
                                continue
                            w_v = float(vsens[l, v]) / max(float(v_en[l, v]), 1e-6)
                            num_in += w_v * in_sq[l, v, e]
                            num_mid += w_v * mid_sq[l, v, e]
                            den += w_v * cv
                        if den > 0:
                            arr_in[l, e] = np.sqrt(np.maximum(num_in / den, 1e-12))
                            arr_mid[l, e] = np.sqrt(np.maximum(num_mid / den, 1e-12))
                    elif mode == "peakch":
                        # Channel-wise peak RMS across active visits (prevents clipping any stage's salient channels)
                        per_v_in = []
                        per_v_mid = []
                        for v in vlist:
                            cv = int(counts[l, v, e])
                            if cv >= 2:
                                per_v_in.append(np.sqrt(np.maximum(in_sq[l, v, e] / cv, 1e-12)))
                                per_v_mid.append(np.sqrt(np.maximum(mid_sq[l, v, e] / cv, 1e-12)))
                        if per_v_in:
                            arr_in[l, e] = np.max(np.stack(per_v_in, axis=0), axis=0)
                            arr_mid[l, e] = np.max(np.stack(per_v_mid, axis=0), axis=0)
                        else:
                            arr_in[l, e] = np.sqrt(np.maximum(in_sq[l, vlist, e].sum(axis=0) / tot_c, 1e-12))
                            arr_mid[l, e] = np.sqrt(np.maximum(mid_sq[l, vlist, e].sum(axis=0) / tot_c, 1e-12))

            self.in_rms[mode] = arr_in
            self.mid_rms[mode] = arr_mid

    @torch.no_grad()
    def restore_all(self) -> None:
        for lid, layer in enumerate(self.model.layers):
            em = layer.moe.experts
            w1, w2, w3 = self.backups[lid]
            em.weight1.copy_(w1, non_blocking=False)
            em.weight2.copy_(w2, non_blocking=False)
            em.weight3.copy_(w3, non_blocking=False)

    @torch.no_grad()
    def precompute_distortion_tables(
        self,
        *,
        awq_mode: str = "stagesens",
        group_size: int = 128,
        device: torch.device,
    ) -> dict[str, np.ndarray]:
        """Compute exact per-(layer, expert, bit in {2,3,4}) distortion tables:
          - 'stagesens_norm': sum_v alpha_{l,v} * (MSE_{l,v,e}(b) / Energy_{l,v})   [OURS!]
          - 'unweighted_norm': sum_v (MSE_{l,v,e}(b) / Energy_{l,v})                [ABLATION: no alpha_{l,v}!]
        Uses the exact quantized weights W_q(b) produced by awq_mode!
        """
        L, E, D, I = self.n_layers, self.n_experts, self.D, self.I
        in_sq = self.calib["in_sq_sum"]
        mid_sq = self.calib["mid_sq_sum"]
        vsens = self.calib["visit_sens"]
        v_en = self.calib["visit_total_energy"]
        layer_visits = [sorted(set(d["loop_idx"] for d in self.topo if d["layer_id"] == l)) for l in range(L)]

        dist_loop = np.zeros((L, E, 3), dtype=np.float64)
        dist_unw = np.zeros((L, E, 3), dtype=np.float64)

        alphas = (0.0,) if awq_mode == "rtn" else (0.0, 0.25, 0.5, 0.75, 1.0)
        sc_key = "pool" if awq_mode == "rtn" else awq_mode

        for l in range(L):
            w1_cpu, w2_cpu, w3_cpu = self.backups[l]
            w1 = w1_cpu.to(device).view(E, I, D)
            w2 = w2_cpu.to(device).view(E, D, I)
            w3 = w3_cpu.to(device).view(E, I, D)

            a_in = torch.from_numpy(self.in_rms[sc_key][l]).to(device)
            a_mid = torch.from_numpy(self.mid_rms[sc_key][l]).to(device)
            vlist = layer_visits[l]

            # Build per-visit weighted channel squared-activation tensors on GPU for exact MSE evaluation
            # For W1 & W3: input x; for W2: mid h_mid (scaled to match output space directly!)
            # Note: W2 maps mid -> out directly, so ||(W2 - W2q) h_mid||^2 is directly in output energy units!
            # For W1/W3, normalize their relative weight-error by ||W1||_F^2 / ||W2||_F^2 or evaluate directly
            # as fractional energy perturbation: sum_v w_v * [ ||dW2 * h_mid||^2 + (||dW1 * x||^2 + ||dW3 * x||^2) * scale_mid_to_out ]
            w2_rms_ratio = (w2.float().pow(2).sum(dim=(1, 2)) / (w1.float().pow(2).sum(dim=(1, 2)) + 1e-8)).clamp(0.1, 10.0)

            for bi, b in enumerate((2, 3, 4)):
                w1q, _ = awq_quantize_expert_bank(w1, a_in, a_in, b, group_size, alphas=alphas)
                w3q, _ = awq_quantize_expert_bank(w3, a_in, a_in, b, group_size, alphas=alphas)
                w2q, _ = awq_quantize_expert_bank(w2, a_mid, a_mid, b, group_size, alphas=alphas)

                dw1_sq = (w1.float() - w1q.float()).pow(2).sum(dim=1)  # [E, D]
                dw3_sq = (w3.float() - w3q.float()).pow(2).sum(dim=1)  # [E, D]
                dw2_sq = (w2.float() - w2q.float()).pow(2).sum(dim=1)  # [E, I]

                for v in vlist:
                    en_v = max(float(v_en[l, v]), 1e-6)
                    alpha_v = float(vsens[l, v])
                    xsq_v = torch.from_numpy(in_sq[l, v].astype(np.float32)).to(device)    # [E, D]
                    msq_v = torch.from_numpy(mid_sq[l, v].astype(np.float32)).to(device)   # [E, I]
                    # Output-space MSE for W2 + scaled input-space MSE for W1/W3
                    mse_e = (
                        (dw2_sq * msq_v).sum(dim=-1)
                        + 0.5 * w2_rms_ratio * ((dw1_sq + dw3_sq) * xsq_v).sum(dim=-1)
                    ).cpu().numpy()  # [E]
                    frac_e = mse_e / en_v
                    dist_loop[l, :, bi] += alpha_v * frac_e
                    dist_unw[l, :, bi] += frac_e

        return {"stagesens_norm": dist_loop, "unweighted_norm": dist_unw}

    @torch.no_grad()
    def apply_quantization(
        self,
        bits_matrix: np.ndarray,
        *,
        awq_mode: str = "rtn",
        group_size: int = 128,
    ) -> float:
        """Apply per-(layer, expert) quantization using specified AWQ calibration mode."""
        self.restore_all()
        E, D, I = self.n_experts, self.D, self.I
        alphas = (0.0,) if awq_mode == "rtn" else (0.0, 0.25, 0.5, 0.75, 1.0)
        sc_key = "pool" if awq_mode == "rtn" else awq_mode

        for l, layer in enumerate(self.model.layers):
            em = layer.moe.experts
            dev = em.weight1.device
            lbits = bits_matrix[l]
            uniq = sorted(set(int(b) for b in lbits))
            if uniq == [16]:
                continue

            w1_v = em.weight1.view(E, I, D)
            w2_v = em.weight2.view(E, D, I)
            w3_v = em.weight3.view(E, I, D)
            a_in_all = torch.from_numpy(self.in_rms[sc_key][l]).to(dev)
            a_mid_all = torch.from_numpy(self.mid_rms[sc_key][l]).to(dev)
            # For stagesens AWQ, use stagesens RMS for both scale & objective; for loop0/pool/peakch use their own RMS
            for b in uniq:
                if b >= 16:
                    continue
                e_idxs = [e for e in range(E) if int(lbits[e]) == b]
                idx_t = torch.tensor(e_idxs, device=dev, dtype=torch.long)
                ai = a_in_all.index_select(0, idx_t)
                am = a_mid_all.index_select(0, idx_t)
                w1_v[idx_t], _ = awq_quantize_expert_bank(w1_v.index_select(0, idx_t), ai, ai, b, group_size, alphas=alphas)
                w3_v[idx_t], _ = awq_quantize_expert_bank(w3_v.index_select(0, idx_t), ai, ai, b, group_size, alphas=alphas)
                w2_v[idx_t], _ = awq_quantize_expert_bank(w2_v.index_select(0, idx_t), am, am, b, group_size, alphas=alphas)

        return float(np.mean(bits_matrix))


# ---------------------------------------------------------------------------
# Full Per-Model R2-B Execution
# ---------------------------------------------------------------------------

def run_model_r2b(
    model_key: str,
    mspec: dict[str, Any],
    *,
    model_dir: Path,
    dataset_dir: Path,
    mcq_raw: dict[str, list[dict[str, Any]]],
    cfg: dict[str, Any],
    device: torch.device,
    run_obj: Any,
) -> dict[str, Any]:
    artifact_dir = model_dir / mspec["local_dir"]
    seed = int(cfg.get("seed", 20261126))
    print(f"\n[r2b:{model_key}] Loading {artifact_dir} ...", flush=True)
    model, tokenizer, _mcfg = load_ifm_model_for_autograd(
        artifact_dir, device=str(device), attention_backend="sdpa"
    )
    n_layers = len(model.layers)
    n_experts = int(model.layers[0].moe.experts.n_local_experts)
    D = int(model.layers[0].moe.experts.model_dim)
    I = int(model.layers[0].moe.experts.inter_dim)
    topo = build_visit_topology(n_layers, int(model.loop_start_layer), int(model.loop_end_layers), int(model.loop_times))

    splits = build_disjoint_text_code_splits(
        dataset_dir,
        lambda t: tokenizer.encode(t, bos=True, eos=False),
        seq_len=int(cfg.get("seq_len", 256)),
        n_calib_a=int(cfg.get("n_calib", 16)),
        n_calib_b=4,
        n_adapt_train=4,
        n_eval_heldout=int(cfg.get("n_eval", 64)),
        seed=seed,
    )
    calib_seqs = splits["calib_a"]
    eval_seqs = splits["eval_heldout"]
    mcq_enc = {k: prepare_mcq_encoded(v, tokenizer) for k, v in mcq_raw.items()}
    raw_pad = getattr(tokenizer, "pad_id", None)
    if raw_pad is None or int(raw_pad) < 0:
        raw_pad = getattr(tokenizer, "eos_id", None)
    if raw_pad is None or int(raw_pad) < 0:
        raw_pad = getattr(tokenizer, "bos_id", 0)
    pad_id = max(0, int(raw_pad))

    # 1) Verify exactness of restore & right-padded batching
    with run_obj.phase(f"{model_key}_verify", total=1, unit="check") as pv:
        sample_seqs = [mcq_enc["arc_c"][i]["choices"][0][0][:-1] for i in range(4)]
        max_l = max(len(s) for s in sample_seqs)
        pad_in = torch.full((4, max_l), pad_id, device=device, dtype=torch.long)
        for bi, s in enumerate(sample_seqs):
            pad_in[bi, : len(s)] = torch.tensor(s, device=device, dtype=torch.long)
        with torch.inference_mode():
            batched = model(pad_in, multi_segments=False, cache=None)[0].float()
            pad_diff = 0.0
            for bi, s in enumerate(sample_seqs):
                one = model(torch.tensor([s], device=device, dtype=torch.long), multi_segments=False, cache=None)[0].float()
                pad_diff = max(pad_diff, float((one[0] - batched[bi, : len(s)]).abs().max().item()))
        print(f"[r2b:{model_key}] Exactness OK: pad_batch_diff={pad_diff:.2e}", flush=True)
        if pad_diff > 5e-2:
            raise RuntimeError(f"Batched MCQ parity failed: {pad_diff}")
        pv.step(1)

    # 2) Trajectory activation + sensitivity calibration + distortion tables
    with run_obj.phase(f"{model_key}_calib", total=1, unit="pass") as pc:
        calib = calibrate_trajectory_quant(model, topo, calib_seqs, device)
        qm = LoopQuantManager(model, topo, calib)
        # Precompute exact distortion tables for both Pool-AWQ and StageSens-AWQ and RTN
        dist_ss_awq = qm.precompute_distortion_tables(awq_mode="stagesens", group_size=128, device=device)
        dist_pool_awq = qm.precompute_distortion_tables(awq_mode="pool", group_size=128, device=device)
        dist_rtn = qm.precompute_distortion_tables(awq_mode="rtn", group_size=128, device=device)

        # Print per-stage skip-MoE sensitivity alpha summary
        stage_names = []
        for d in topo:
            if d["stage"] not in stage_names:
                stage_names.append(d["stage"])
        st_str = ", ".join(
            f"{st}={sum(calib['visit_sens'][d['layer_id'], d['loop_idx']] for d in topo if d['stage']==st):.3f}"
            for st in stage_names
        )
        print(f"[r2b:{model_key}] Calib stage skip-MoE sens sum (nats): {st_str}", flush=True)
        pc.step(1)

    # Frequency scores for ProQuant-style & Max-Stage-Freq baselines
    counts = calib["counts"]  # [L, max_v, E]
    global_freq = counts.sum(axis=1).astype(np.float64)  # [L, E]
    # Per-visit normalized frequency so every stage contributes equally to max_stage_freq
    v_totals = np.maximum(counts.sum(axis=2, keepdims=True), 1)
    max_stage_freq = (counts / v_totals).max(axis=1).astype(np.float64)

    # Construct all 23 conditions (BF16 + Uniform AWQ Ablations + Tri-level Mixed-Bit 2.25/2.50/3.00b)
    def uni(b: int) -> np.ndarray:
        return np.full((n_layers, n_experts), b, dtype=np.int32)

    conds: list[dict[str, Any]] = [
        {"name": "bf16_base", "bits": uni(16), "awq": "rtn", "gs": 128},
        # --- Uniform W3-g128: AWQ Calibration Comparison ---
        {"name": "w3_g128_rtn", "bits": uni(3), "awq": "rtn", "gs": 128},
        {"name": "w3_g128_loop0_awq", "bits": uni(3), "awq": "loop0", "gs": 128},
        {"name": "w3_g128_pool_awq", "bits": uni(3), "awq": "pool", "gs": 128},
        {"name": "w3_g128_stagesens_awq", "bits": uni(3), "awq": "stagesens", "gs": 128},
        # --- Uniform W2-g128: AWQ Calibration Comparison ---
        {"name": "w2_g128_rtn", "bits": uni(2), "awq": "rtn", "gs": 128},
        {"name": "w2_g128_loop0_awq", "bits": uni(2), "awq": "loop0", "gs": 128},
        {"name": "w2_g128_pool_awq", "bits": uni(2), "awq": "pool", "gs": 128},
        {"name": "w2_g128_stagesens_awq", "bits": uni(2), "awq": "stagesens", "gs": 128},
        {"name": "w2_g128_peakch_awq", "bits": uni(2), "awq": "peakch", "gs": 128},
        # --- Uniform W2-g64: AWQ Calibration Comparison ---
        {"name": "w2_g64_rtn", "bits": uni(2), "awq": "rtn", "gs": 64},
        {"name": "w2_g64_pool_awq", "bits": uni(2), "awq": "pool", "gs": 64},
        {"name": "w2_g64_stagesens_awq", "bits": uni(2), "awq": "stagesens", "gs": 64},
        # --- Iso-Bit 2.25b Mixed-Precision {2,3,4} (7.1x expert compression) ---
        {"name": "m225_global_freq_pool_awq", "bits": rank_to_trilevel_bits(global_freq, 2.25), "awq": "pool", "gs": 128},
        {"name": "m225_unw_mse_pool_awq", "bits": solve_trilevel_bit_allocation(dist_pool_awq["unweighted_norm"], 2.25), "awq": "pool", "gs": 128},
        {"name": "m225_loopquant_full", "bits": solve_trilevel_bit_allocation(dist_ss_awq["stagesens_norm"], 2.25), "awq": "stagesens", "gs": 128},
        # --- Iso-Bit 2.50b Mixed-Precision {2,3,4} (6.4x expert compression) ---
        {"name": "m250_global_freq_pool_awq", "bits": rank_to_trilevel_bits(global_freq, 2.50), "awq": "pool", "gs": 128},
        {"name": "m250_max_stage_freq_pool_awq", "bits": rank_to_trilevel_bits(max_stage_freq, 2.50), "awq": "pool", "gs": 128},
        {"name": "m250_unw_mse_pool_awq", "bits": solve_trilevel_bit_allocation(dist_pool_awq["unweighted_norm"], 2.50), "awq": "pool", "gs": 128},
        {"name": "m250_stagesens_knap_rtn", "bits": solve_trilevel_bit_allocation(dist_rtn["stagesens_norm"], 2.50), "awq": "rtn", "gs": 128},
        {"name": "m250_loopquant_full", "bits": solve_trilevel_bit_allocation(dist_ss_awq["stagesens_norm"], 2.50), "awq": "stagesens", "gs": 128},
        # --- Iso-Bit 3.00b Mixed-Precision {2,3,4} vs Uniform W3 ---
        {"name": "m300_global_freq_pool_awq", "bits": rank_to_trilevel_bits(global_freq, 3.00), "awq": "pool", "gs": 128},
        {"name": "m300_unw_mse_pool_awq", "bits": solve_trilevel_bit_allocation(dist_pool_awq["unweighted_norm"], 3.00), "awq": "pool", "gs": 128},
        {"name": "m300_loopquant_full", "bits": solve_trilevel_bit_allocation(dist_ss_awq["stagesens_norm"], 3.00), "awq": "stagesens", "gs": 128},
    ]

    results: dict[str, Any] = {}
    base_per_seq: np.ndarray | None = None

    with run_obj.phase(f"{model_key}_eval", total=len(conds), unit="conds") as pe:
        for idx_c, c in enumerate(conds, 1):
            t_c0 = time.time()
            avg_b = qm.apply_quantization(c["bits"], awq_mode=c["awq"], group_size=c["gs"])
            vram = compute_expert_vram_gb(c["bits"], D, I, group_size=c["gs"])
            nll_res = eval_nll_batched(model, eval_seqs, device, batch_size=8)

            mcq_accs: dict[str, float] = {}
            mcq_flags: dict[str, list[int]] = {}
            for ds_name, enc_items in mcq_enc.items():
                acc, fl = eval_mcq_batched(model, enc_items, device, batch_size=32, pad_token_id=pad_id)
                mcq_accs[ds_name] = acc
                mcq_flags[ds_name] = fl
            avg_mcq = float(np.mean(list(mcq_accs.values())))

            if c["name"] == "bf16_base":
                base_per_seq = np.asarray(nll_res["per_seq"], dtype=np.float64)
            assert base_per_seq is not None
            diffs = np.asarray(nll_res["per_seq"], dtype=np.float64) - base_per_seq
            dnll, dnll_lo, dnll_hi = paired_bootstrap_ci(diffs)
            elapsed_c = time.time() - t_c0

            b_hist = {str(b): int((c["bits"] == b).sum()) for b in (2, 3, 4, 16) if (c["bits"] == b).any()}
            results[c["name"]] = {
                "name": c["name"],
                "awq_mode": c["awq"],
                "group_size": c["gs"],
                "avg_bits": avg_b,
                "vram_gb": vram["expert_vram_gb"],
                "compression_ratio": vram["compression_ratio"],
                "bit_hist": b_hist,
                "nll": nll_res["nll"],
                "text_nll": nll_res["text_nll"],
                "code_nll": nll_res["code_nll"],
                "delta_nll": dnll,
                "delta_nll_ci": [dnll_lo, dnll_hi],
                "per_seq": nll_res["per_seq"],
                "arc_c": mcq_accs["arc_c"],
                "arc_e": mcq_accs["arc_e"],
                "obqa": mcq_accs["obqa"],
                "avg_mcq": avg_mcq,
                "mcq_flags": mcq_flags,
                "elapsed_s": elapsed_c,
            }
            print(
                f"[r2b:{model_key}] {c['name']:30s} b={avg_b:4.2f} ({vram['expert_vram_gb']:.2f}GB) | "
                f"NLL={nll_res['nll']:.4f} (Δ={dnll:+.4f}) | ARC-C={mcq_accs['arc_c']*100:5.1f}% "
                f"ARC-E={mcq_accs['arc_e']*100:5.1f}% OBQA={mcq_accs['obqa']*100:5.1f}% "
                f"Avg={avg_mcq*100:5.1f}% | t={elapsed_c:.1f}s",
                flush=True,
            )
            pe.row(
                f"{model_key}_results",
                condition=c["name"],
                avg_bits=avg_b,
                vram_gb=vram["expert_vram_gb"],
                delta_nll=dnll,
                arc_c=mcq_accs["arc_c"],
                arc_e=mcq_accs["arc_e"],
                obqa=mcq_accs["obqa"],
                avg_mcq=avg_mcq,
            )
            pe.step(idx_c, latest_avg_mcq=avg_mcq, latest_dnll=dnll)

    qm.restore_all()

    # Paired contrasts (positive = first method better)
    contrast_pairs = [
        ("w2_g128_pool_awq", "w2_g128_rtn"),
        ("w2_g128_stagesens_awq", "w2_g128_rtn"),
        ("w2_g128_stagesens_awq", "w2_g128_loop0_awq"),
        ("w2_g128_stagesens_awq", "w2_g128_pool_awq"),
        ("w2_g128_peakch_awq", "w2_g128_pool_awq"),
        ("w2_g64_stagesens_awq", "w2_g64_pool_awq"),
        ("w3_g128_stagesens_awq", "w3_g128_pool_awq"),
        ("w3_g128_stagesens_awq", "w3_g128_rtn"),
        ("m225_loopquant_full", "m225_global_freq_pool_awq"),
        ("m225_loopquant_full", "m225_unw_mse_pool_awq"),
        ("m250_loopquant_full", "m250_global_freq_pool_awq"),
        ("m250_loopquant_full", "m250_max_stage_freq_pool_awq"),
        ("m250_loopquant_full", "m250_unw_mse_pool_awq"),
        ("m250_loopquant_full", "m250_stagesens_knap_rtn"),
        ("m300_loopquant_full", "w3_g128_pool_awq"),
        ("m300_loopquant_full", "m300_global_freq_pool_awq"),
        ("m300_loopquant_full", "m300_unw_mse_pool_awq"),
    ]
    contrasts: dict[str, Any] = {}
    for a, b in contrast_pairs:
        if a in results and b in results:
            da = np.asarray(results[b]["per_seq"], dtype=np.float64) - np.asarray(results[a]["per_seq"], dtype=np.float64)
            m, lo, hi = paired_bootstrap_ci(da)
            fl_a = np.concatenate([results[a]["mcq_flags"][k] for k in ("arc_c", "arc_e", "obqa")])
            fl_b = np.concatenate([results[b]["mcq_flags"][k] for k in ("arc_c", "arc_e", "obqa")])
            mm, mlo, mhi = paired_bootstrap_ci(fl_a - fl_b)
            contrasts[f"{a}_vs_{b}"] = {
                "nll_gain": m,
                "nll_ci": [lo, hi],
                "nll_wins": int((da > 0).sum()),
                "n_seq": int(da.size),
                "mcq_gain_pp": mm * 100.0,
                "mcq_ci_pp": [mlo * 100.0, mhi * 100.0],
            }

    compact_results = {
        k: {kk: vv for kk, vv in v.items() if kk not in ("per_seq", "mcq_flags")}
        for k, v in results.items()
    }

    del qm, model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model_key": model_key,
        "results": compact_results,
        "contrasts": contrasts,
    }


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Round 2-B: LoopQuant-MoE — Trajectory-Calibrated AWQ & Stage-Sensitivity Mixed-Bit PTQ",
        "",
        f"- GPU: `{summary['gpu_index']}` ({summary['gpu_name']}), elapsed `{summary['elapsed_sec']:.1f}s`",
        f"- Benchmarks: Held-out NLL (`n={summary['n_eval']}`), ARC-Challenge (`n={summary['n_arc_c']}`), ARC-Easy (`n={summary['n_arc_e']}`), OpenBookQA (`n={summary['n_obqa']}`)",
        "",
    ]
    for mk, mres in summary["models"].items():
        lines.append(f"## Model: `{mk}`")
        lines.append("| Condition | Avg Bits | Expert VRAM | NLL | ΔNLL (95% CI) | ARC-C | ARC-E | OBQA | Avg MCQ |")
        lines.append("|---|---:|---:|---:|---|---:|---:|---:|---:|")
        for cname, r in mres["results"].items():
            lo, hi = r["delta_nll_ci"]
            lines.append(
                f"| `{cname}` | {r['avg_bits']:.2f} | {r['vram_gb']:.2f} GB ({r['compression_ratio']:.1f}x) | "
                f"{r['nll']:.4f} | {r['delta_nll']:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
                f"{r['arc_c']*100:.1f}% | {r['arc_e']*100:.1f}% | {r['obqa']*100:.1f}% | **{r['avg_mcq']*100:.1f}%** |"
            )
        lines.append("")
        lines.append("### Paired Bootstrap Contrasts (positive = first beats second)")
        for cname, c in mres["contrasts"].items():
            nlo, nhi = c["nll_ci"]
            mlo, mhi = c["mcq_ci_pp"]
            lines.append(
                f"- `{cname}`: NLL `{c['nll_gain']:+.4f}` CI `[{nlo:+.4f}, {nhi:+.4f}]` ({c['nll_wins']}/{c['n_seq']} wins) | "
                f"Pooled MCQ `{c['mcq_gain_pp']:+.2f} pp` CI `[{mlo:+.2f}, {mhi:+.2f}]`"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--xllm-loop-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    gpu_idx, gpu_lock, gpu_status = acquire_idle_gpu(cfg.get("preferred_gpu"))
    try:
        print(
            f"[r2b] Acquired idle GPU {gpu_idx}: {gpu_status.name} "
            f"(mem={gpu_status.mem_used_mib}MiB, util={gpu_status.util_pct}%)",
            flush=True,
        )
        boot = bootstrap_xllm(args.xllm_loop_dir, attention_backend="sdpa")
        print(f"[r2b] xllm bootstrap: {json.dumps(boot)}", flush=True)
        device = torch.device(f"cuda:{gpu_idx}")
        torch.cuda.set_device(device)

        seed = int(cfg.get("seed", 20261126))
        mcq_raw = {
            "arc_c": load_arc_challenge_samples(args.dataset_dir, n_samples=int(cfg.get("n_arc_c", 0)), seed=seed),
            "arc_e": load_arc_easy_samples(args.dataset_dir, n_samples=int(cfg.get("n_arc_e", 600)), seed=seed),
            "obqa": load_openbookqa_samples(args.dataset_dir, n_samples=int(cfg.get("n_obqa", 500)), seed=seed),
        }
        print(
            f"[r2b] Loaded MCQ suites: ARC-C={len(mcq_raw['arc_c'])}, ARC-E={len(mcq_raw['arc_e'])}, OBQA={len(mcq_raw['obqa'])}",
            flush=True,
        )

        plan = [
            {"phase": "setup", "of": 1, "unit": "env"},
            {"phase": "download", "of": 1, "unit": "assets"},
        ]
        for mk in cfg["models"]:
            plan.append({"phase": f"{mk}_verify", "of": 1, "unit": "check"})
            plan.append({"phase": f"{mk}_calib", "of": 1, "unit": "pass"})
            plan.append({"phase": f"{mk}_eval", "of": 24, "unit": "conds"})
        plan.append({"phase": "report", "of": 1, "unit": "file"})
        orx.plan(plan)
        run_obj = orx.progress(total=1, unit="run")

        models_out: dict[str, Any] = {}
        for mk, mspec in cfg["models"].items():
            models_out[mk] = run_model_r2b(
                mk,
                mspec,
                model_dir=Path(args.model_dir),
                dataset_dir=Path(args.dataset_dir),
                mcq_raw=mcq_raw,
                cfg=cfg,
                device=device,
                run_obj=run_obj,
            )

        summary = {
            "gpu_index": gpu_idx,
            "gpu_name": gpu_status.name,
            "elapsed_sec": time.time() - t0,
            "n_eval": int(cfg.get("n_eval", 64)),
            "n_arc_c": len(mcq_raw["arc_c"]),
            "n_arc_e": len(mcq_raw["arc_e"]),
            "n_obqa": len(mcq_raw["obqa"]),
            "models": models_out,
        }

        with run_obj.phase("report", total=1, unit="file") as pr:
            (out_dir / "summary_r2b_loopquant.json").write_text(json.dumps(summary, indent=2))
            rep = render_report(summary)
            (out_dir / "report_r2b_loopquant.md").write_text(rep)
            print("========== REPORT (report_r2b_loopquant.md) ==========", flush=True)
            print(rep, flush=True)
            print("========== END REPORT ==========", flush=True)

            m_out: dict[str, float] = {}
            for mk, mres in models_out.items():
                c = mres["contrasts"]
                m_out[f"{mk}_m250_vs_freq_nll"] = c["m250_loopquant_full_vs_m250_global_freq_pool_awq"]["nll_gain"]
                m_out[f"{mk}_m250_vs_freq_mcq"] = c["m250_loopquant_full_vs_m250_global_freq_pool_awq"]["mcq_gain_pp"]
                m_out[f"{mk}_m250_vs_unw_nll"] = c["m250_loopquant_full_vs_m250_unw_mse_pool_awq"]["nll_gain"]
                m_out[f"{mk}_w2_ss_vs_pool_nll"] = c["w2_g128_stagesens_awq_vs_w2_g128_pool_awq"]["nll_gain"]
                m_out[f"{mk}_m250_avg_mcq"] = mres["results"]["m250_loopquant_full"]["avg_mcq"]
            pr.step(1, **m_out)
    finally:
        gpu_lock.release()


if __name__ == "__main__":
    main()
