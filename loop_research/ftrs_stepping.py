"""Direction 1 Feasibility Pilot: Curvature-Calibrated Latent Trust-Region Stepping (FTRS).

Tests the core ICML 2027 Direction 1 hypotheses on ByteDance/Ouro-1.4B and
ByteDance/Ouro-2.6B-Thinking across GSM8K, MATH-500, and ARC-Challenge:
  0. Exactness verification: custom arbitrary-depth OuroCalibratedEngine vs native HF
     OuroForCausalLM (unpadded B=1, left-padded B=2, and KV-cached greedy decode).
  1. Probe 1A (Teacher-Forced Geometric Duality & Calibration):
     - Exact Proposition 3.4 check: Delta_U(1) == A_r - KL(p_{r-1} || p_r).
     - Reference-free readout Fisher curvature kappa_r = Var_{p_{r-1}}(Delta z_r) vs
       oracle optimal step alpha* in [0, 1.8] across transitions 1->2 .. 7->8.
     - Curvature quintiles (low-kappa extrapolation alpha*>1 vs high-kappa damping alpha*<1).
     - Low-margin stratification: where LoopCD fails (high kappa) vs succeeds (low kappa).
     - Index-free 5-feature FTRS-Gate trained strictly on GSM8K-train transitions and
       evaluated zero-shot on GSM8K-test and OOD MATH-500.
  2. Probe 1B (Closed-Loop Feedback & Depth Extrapolation R in {2, 4, 6, 8}):
     - ARC-Challenge (200 MC questions) + GSM8K free generation + MATH-500 free generation.
     - Side-by-side comparison of native, LoopCD-Logits (fixed & adaptive), LoopCD-Hidden,
       fixed damping (0.5), FTRS-Zero (Euclidean vs RMS-matched manifold feedback),
       FTRS-Unified (manifold feedback + curvature-bounded extrapolation), and FTRS-Gate.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from loop_research import orx_run as orx
from loop_research.data_assets import (
    extract_predicted_math_answer,
    load_arc_challenge_samples,
    load_gsm8k_split,
    load_math500_samples,
)
from loop_research.gpu_utils import acquire_idle_gpu


# ---------------------------------------------------------------------------
# Statistics Helpers
# ---------------------------------------------------------------------------

def spearman_corr(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[mask], ya[mask]
    if xa.size < 3 or np.std(xa) < 1e-12 or np.std(ya) < 1e-12:
        return 0.0
    rx = np.argsort(np.argsort(xa)).astype(np.float64)
    ry = np.argsort(np.argsort(ya)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / denom) if denom > 1e-12 else 0.0


def binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Exact rank-based AUROC (handles ties via average ranks)."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    mask = np.isfinite(s)
    s, y = s[mask], y[mask]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]
    ranks = np.empty(s.size, dtype=np.float64)
    i = 0
    while i < s.size:
        j = i + 1
        while j < s.size and s_sorted[j] == s_sorted[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[i:j] = avg_rank
        i = j
    sum_pos = float(ranks[y_sorted == 1].sum())
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def paired_bootstrap_ci(
    diffs: list[float] | np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 12345,
) -> tuple[float, float, float]:
    arr = np.asarray(diffs, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(arr.mean()), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Pure Geometry & Trust-Region Step Primitives (Unit-Testable on CPU!)
# ---------------------------------------------------------------------------

def compute_fisher_geometry(
    z_prev: torch.Tensor,
    z_curr: torch.Tensor,
    d_curr: torch.Tensor,
    d_prev: torch.Tensor | None,
    h_prev: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute reference-free Fisher curvature and index-free trajectory invariants.

    Args:
        z_prev, z_curr: [..., V] logits at loop r-1 and proposed loop r.
        d_curr: [..., D] proposed displacement H_r - H_{r-1}.
        d_prev: [..., D] previous displacement H_{r-1} - H_{r-2} (or None at r=1).
        h_prev: [..., D] state H_{r-1}.
    Returns:
        Dict of FP32 tensors of shape [...]: kappa, kl_fwd, margin, rel_norm, cos_mom.
    """
    zp = z_prev.float()
    zc = z_curr.float()
    log_p_prev = F.log_softmax(zp, dim=-1)
    log_p_curr = F.log_softmax(zc, dim=-1)
    p_prev = log_p_prev.exp()

    dz = zc - zp
    mean_dz = (p_prev * dz).sum(dim=-1)
    # Exact readout Fisher curvature: Var_{p_{r-1}}(Delta z_r)
    kappa = (p_prev * (dz - mean_dz.unsqueeze(-1)).pow(2)).sum(dim=-1).clamp_min(0.0)
    # Exact pathwise curvature integral under affine logits: KL(p_{r-1} || p_r)
    kl_fwd = (p_prev * (log_p_prev - log_p_curr)).sum(dim=-1).clamp_min(0.0)

    top2 = torch.topk(p_prev, k=2, dim=-1).values
    margin = (top2[..., 0] - top2[..., 1]).clamp(0.0, 1.0)

    d_f = d_curr.float()
    h_f = h_prev.float()
    rel_norm = d_f.norm(dim=-1) / (h_f.norm(dim=-1) + 1e-6)
    if d_prev is not None:
        cos_mom = F.cosine_similarity(d_f, d_prev.float(), dim=-1).clamp(-1.0, 1.0)
    else:
        cos_mom = torch.ones_like(kappa)

    return {
        "kappa": kappa,
        "kl_fwd": kl_fwd,
        "margin": margin,
        "rel_norm": rel_norm,
        "cos_mom": cos_mom,
        "mean_dz": mean_dz,
    }


def stack_index_free_features(geom: dict[str, torch.Tensor]) -> torch.Tensor:
    """Pack the 5 strictly index-free geometric invariants into shape [..., 5]."""
    return torch.stack(
        [
            torch.log(geom["kappa"].clamp_min(0.0) + 1e-6),
            torch.log(geom["kl_fwd"].clamp_min(0.0) + 1e-6),
            geom["margin"],
            geom["cos_mom"],
            torch.log1p(geom["rel_norm"].clamp_min(0.0)),
        ],
        dim=-1,
    )


def ftrs_zero_feedback_alpha(
    kappa: torch.Tensor,
    cos_mom: torch.Tensor,
    *,
    tau_fb: float = 0.15,
    use_momentum: bool = True,
) -> torch.Tensor:
    """Training-free, index-free Fisher trust-region feedback step alpha in [0, 1].

    - Local KL trust region: 0.5 * alpha^2 * kappa <= tau_fb  =>  alpha_tr = min(1, sqrt(2*tau/kappa)).
    - Limit-cycle / reversal damping: when cos(D_r, D_{r-1}) < 0, scale by (1 + cos) -> 0 at reversal.
    """
    alpha_tr = torch.sqrt((2.0 * tau_fb) / (kappa.clamp_min(0.0) + 1e-8)).clamp(0.0, 1.0)
    if use_momentum:
        mom_factor = (1.0 + cos_mom.clamp(max=0.0)).clamp(0.0, 1.0)
        return alpha_tr * mom_factor
    return alpha_tr


def apply_calibrated_step(
    h_prev: torch.Tensor,
    h_curr: torch.Tensor,
    alpha: torch.Tensor,
    *,
    manifold: str = "rms_match",
) -> torch.Tensor:
    """Apply H_cal = Step(H_prev, H_curr, alpha) with optional RMS-manifold preservation."""
    a = alpha.to(dtype=torch.float32).unsqueeze(-1)
    hp = h_prev.float()
    hc = h_curr.float()
    h_lin = hp + a * (hc - hp)
    if manifold == "rms_match":
        target_rms = torch.sqrt(hc.pow(2).mean(dim=-1, keepdim=True) + 1e-8)
        lin_rms = torch.sqrt(h_lin.pow(2).mean(dim=-1, keepdim=True) + 1e-8)
        h_out = h_lin * (target_rms / lin_rms)
    elif manifold == "euclidean":
        h_out = h_lin
    else:
        raise ValueError(f"Unknown manifold mode: {manifold}")
    return h_out.to(dtype=h_curr.dtype)


def unified_readout_logits(
    z_first: torch.Tensor,
    z_prev: torch.Tensor,
    z_curr: torch.Tensor,
    *,
    alpha_fb: torch.Tensor,
    kappa_step: torch.Tensor,
    cos_mom: torch.Tensor,
    omega_max: float = 0.75,
    tau_fb: float = 0.15,
    tau_ext: float = 0.25,
) -> torch.Tensor:
    """Unified FTRS readout combining overshoot damping (alpha < 1) and curvature-bounded LoopCD (alpha > 1).

    - When the last step is high-curvature or reversing (alpha_fb < 0.999), damp along (z_curr - z_prev)
      and disable outward extrapolation.
    - When the last step is well inside the trust region (low curvature & aligned momentum), allow
      outward extrapolation along (z_curr - z_first) capped by both LoopCD margin and Fisher trust region!
    """
    zf = z_first.float()
    zp = z_prev.float()
    zc = z_curr.float()
    a = alpha_fb.float().unsqueeze(-1)
    z_damped = zp + a * (zc - zp)

    # Curvature along the extrapolation ray (z_curr - z_first) at p_curr
    p_curr = F.softmax(zc, dim=-1)
    top2 = torch.topk(p_curr, k=2, dim=-1).values
    margin_curr = (top2[..., 0] - top2[..., 1]).clamp(0.0, 1.0)
    dz_ref = zc - zf
    mu_ref = (p_curr * dz_ref).sum(dim=-1, keepdim=True)
    kappa_ref = (p_curr * (dz_ref - mu_ref).pow(2)).sum(dim=-1).clamp_min(0.0)

    # Smooth gate turning off extrapolation as soon as step hits high curvature or negative momentum
    healthy_gate = ((alpha_fb.float() >= 0.95).float()) * cos_mom.float().clamp(0.0, 1.0)
    omega_tr = torch.sqrt((2.0 * tau_ext) / (kappa_ref + 1e-8)).clamp(0.0, omega_max)
    omega_margin = omega_max * (1.0 - margin_curr)
    omega_eff = torch.minimum(omega_margin, omega_tr) * healthy_gate

    return z_damped + omega_eff.unsqueeze(-1) * dz_ref


class FTRSGate(nn.Module):
    """Ultra-lightweight (~225 params), strictly index-free geometric step-size controller."""

    def __init__(self, in_dim: int = 5, hidden_dim: int = 32, max_alpha: float = 1.8) -> None:
        super().__init__()
        self.max_alpha = float(max_alpha)
        self.register_buffer("feat_mean", torch.zeros(in_dim))
        self.register_buffer("feat_std", torch.ones(in_dim))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize near alpha = 1.0 so untrained gate reproduces native recurrence
        nn.init.zeros_(self.net[-1].weight)
        init_logit = math.log((1.0 / self.max_alpha) / (1.0 - 1.0 / self.max_alpha))
        nn.init.constant_(self.net[-1].bias, init_logit)

    def set_norm_stats(self, feats: torch.Tensor) -> None:
        mean = feats.float().mean(dim=0)
        std = feats.float().std(dim=0).clamp_min(1e-4)
        self.feat_mean.copy_(mean)
        self.feat_std.copy_(std)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = (feats.float() - self.feat_mean) / self.feat_std
        raw = self.net(x).squeeze(-1)
        return self.max_alpha * torch.sigmoid(raw)


# ---------------------------------------------------------------------------
# Arbitrary-Depth Ouro Engine with Batched Left-Padded KV Cache & Calibration
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # q, k: [B, H, L, D]; cos, sin: [B, L, D] -> [B, 1, L, D]
    c = cos.unsqueeze(1)
    s = sin.unsqueeze(1)
    return (q * c) + (_rotate_half(q) * s), (k * c) + (_rotate_half(k) * s)


@dataclass
class StepPolicy:
    name: str
    n_loops: int
    feedback_mode: str = "none"  # none | fixed | ftrs_zero | ftrs_gate
    manifold: str = "rms_match"  # rms_match | euclidean
    readout_mode: str = "native"  # native | loopcd_logits_fixed | loopcd_logits_adapt | loopcd_hidden_fixed | ftrs_unified
    tau_fb: float = 0.15
    tau_ext: float = 0.25
    fixed_alpha: float = 0.5
    fixed_start_loop: int = 4  # 0-based loop index where fixed damping starts (4 => 5th loop, i.e. 4->5)
    omega: float = 0.5
    omega_max: float = 1.0
    use_momentum: bool = True
    gate_module: FTRSGate | None = None
    calibrate_prefill: bool = False  # Keep prompt representation clean by default; calibrate decode (or both)


class OuroCalibratedEngine:
    """Exact arbitrary-depth Ouro runner supporting batched left-padded KV decode and step calibration."""

    def __init__(self, model, device: torch.device) -> None:
        self.model = model
        self.core = model.model
        self.lm_head = model.lm_head
        self.device = device
        self.num_layers = int(model.config.num_hidden_layers)
        self.train_loops = int(getattr(model.config, "total_ut_steps", 4))
        self.num_heads = int(model.config.num_attention_heads)
        self.num_kv_heads = int(model.config.num_key_value_heads)
        self.head_dim = int(getattr(model.config, "head_dim", model.config.hidden_size // self.num_heads))
        self.enable_gqa = self.num_heads != self.num_kv_heads

    @torch.inference_mode()
    def _run_layer_prefill(
        self,
        layer,
        h: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor | None,
        *,
        return_kv: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        bsz, seq_len, _ = h.shape
        attn = layer.self_attn
        residual = h
        x = layer.input_layernorm(h)

        q = attn.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = attn.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = attn.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=(attn_mask is None),
            scale=attn.scaling,
            enable_gqa=self.enable_gqa,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.num_heads * self.head_dim)
        h = residual + layer.input_layernorm_2(attn.o_proj(out))
        h = h + layer.post_attention_layernorm_2(layer.mlp(layer.post_attention_layernorm(h)))
        return h, ((k, v) if return_kv else None)

    @torch.inference_mode()
    def _run_layer_decode(
        self,
        layer,
        h: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = h.shape[0]
        attn = layer.self_attn
        residual = h
        x = layer.input_layernorm(h)

        q = attn.q_proj(x).view(bsz, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k_new = attn.k_proj(x).view(bsz, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_new = attn.v_proj(x).view(bsz, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k_new = _apply_rope(q, k_new, cos, sin)

        k_cat = torch.cat([k_cache, k_new], dim=2)
        v_cat = torch.cat([v_cache, v_new], dim=2)

        out = F.scaled_dot_product_attention(
            q,
            k_cat,
            v_cat,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=attn.scaling,
            enable_gqa=self.enable_gqa,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, 1, self.num_heads * self.head_dim)
        h = residual + layer.input_layernorm_2(attn.o_proj(out))
        h = h + layer.post_attention_layernorm_2(layer.mlp(layer.post_attention_layernorm(h)))
        return h, k_cat, v_cat

    def _compute_step_alpha_and_state(
        self,
        ut: int,
        h_prev: torch.Tensor,
        h_prop: torch.Tensor,
        d_prev: torch.Tensor | None,
        z_prev: torch.Tensor | None,
        policy: StepPolicy,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor] | None]:
        """Compute feedback step alpha and calibrated state H_cal for transition (ut-1 -> ut)."""
        bsz_shape = h_prop.shape[:-1]
        if policy.feedback_mode == "none":
            return h_prop, torch.ones(bsz_shape, device=self.device, dtype=torch.float32), None, None

        if policy.feedback_mode == "fixed":
            val = policy.fixed_alpha if ut >= policy.fixed_start_loop else 1.0
            alpha = torch.full(bsz_shape, val, device=self.device, dtype=torch.float32)
            h_cal = apply_calibrated_step(h_prev, h_prop, alpha, manifold=policy.manifold) if val != 1.0 else h_prop
            return h_cal, alpha, None, None

        # FTRS modes need readout logits z_prev and z_prop
        if z_prev is None:
            z_prev = self.lm_head(h_prev)
        z_prop = self.lm_head(h_prop)
        d_curr = h_prop - h_prev
        geom = compute_fisher_geometry(z_prev, z_prop, d_curr, d_prev, h_prev)

        if policy.feedback_mode == "ftrs_zero":
            alpha = ftrs_zero_feedback_alpha(
                geom["kappa"], geom["cos_mom"], tau_fb=policy.tau_fb, use_momentum=policy.use_momentum
            )
        elif policy.feedback_mode == "ftrs_gate":
            assert policy.gate_module is not None
            feats = stack_index_free_features(geom)
            # Clamp feedback alpha to [0, 1.0] for stability when feeding back into next loop
            alpha = policy.gate_module(feats).clamp(0.0, 1.0)
        else:
            raise ValueError(f"Unknown feedback_mode: {policy.feedback_mode}")

        h_cal = apply_calibrated_step(h_prev, h_prop, alpha, manifold=policy.manifold)
        return h_cal, alpha, z_prop, geom

    def _compute_final_logits(
        self,
        h_first: torch.Tensor,
        h_prev: torch.Tensor,
        h_prop_last: torch.Tensor,
        h_cal_last: torch.Tensor,
        z_first: torch.Tensor | None,
        z_prev: torch.Tensor | None,
        z_prop_last: torch.Tensor | None,
        alpha_last: torch.Tensor,
        geom_last: dict[str, torch.Tensor] | None,
        d_prev_last: torch.Tensor | None,
        policy: StepPolicy,
    ) -> torch.Tensor:
        mode = policy.readout_mode
        if mode == "native":
            return self.lm_head(h_cal_last).float()
        if mode == "loopcd_hidden_fixed":
            h_ext = h_cal_last + policy.omega * (h_cal_last - h_first)
            return self.lm_head(h_ext).float()
        if mode == "loopcd_logits_fixed":
            zf = self.lm_head(h_first).float() if z_first is None else z_first.float()
            zl = self.lm_head(h_cal_last).float()
            return zl + policy.omega * (zl - zf)
        if mode == "loopcd_logits_adapt":
            zf = self.lm_head(h_first).float() if z_first is None else z_first.float()
            zl = self.lm_head(h_cal_last).float()
            p_l = F.softmax(zl, dim=-1)
            top2 = torch.topk(p_l, k=2, dim=-1).values
            margin = (top2[..., 0] - top2[..., 1]).clamp(0.0, 1.0)
            w = policy.omega_max * (1.0 - margin)
            return zl + w.unsqueeze(-1) * (zl - zf)
        if mode == "ftrs_unified":
            zf = self.lm_head(h_first).float() if z_first is None else z_first.float()
            zp = self.lm_head(h_prev).float() if z_prev is None else z_prev.float()
            zc = self.lm_head(h_prop_last).float() if z_prop_last is None else z_prop_last.float()
            if geom_last is None:
                geom_last = compute_fisher_geometry(zp, zc, h_prop_last - h_prev, d_prev_last, h_prev)
                alpha_last = ftrs_zero_feedback_alpha(
                    geom_last["kappa"], geom_last["cos_mom"], tau_fb=policy.tau_fb, use_momentum=policy.use_momentum
                )
            return unified_readout_logits(
                zf,
                zp,
                zc,
                alpha_fb=alpha_last,
                kappa_step=geom_last["kappa"],
                cos_mom=geom_last["cos_mom"],
                omega_max=policy.omega_max,
                tau_fb=policy.tau_fb,
                tau_ext=policy.tau_ext,
            )
        raise ValueError(f"Unknown readout_mode: {mode}")

    @torch.inference_mode()
    def forward_all_loop_hiddens(
        self,
        input_ids: torch.Tensor,
        max_loops: int = 8,
        attention_mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Run unmodified Ouro for max_loops and return [H_0, H_1, ..., H_{max_loops-1}] each [B, L, D]."""
        bsz, seq_len = input_ids.shape
        if attention_mask is None:
            pos_ids = torch.arange(seq_len, device=self.device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
            attn_mask = None
        else:
            pos_ids = (attention_mask.cumsum(dim=-1) - 1).clamp_min(0)
            valid_k = attention_mask[:, None, None, :].bool()
            causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=self.device))[None, None, :, :]
            attn_mask = causal & valid_k
            diag = torch.eye(seq_len, dtype=torch.bool, device=self.device)[None, None, :, :]
            attn_mask = attn_mask | diag

        h = self.core.embed_tokens(input_ids)
        cos, sin = self.core.rotary_emb(h, pos_ids)

        hiddens: list[torch.Tensor] = []
        for _ut in range(max_loops):
            for layer in self.core.layers:
                h, _ = self._run_layer_prefill(layer, h, cos, sin, attn_mask, return_kv=False)
            h = self.core.norm(h)
            hiddens.append(h)
        return hiddens

    @torch.inference_mode()
    def forward_policy_logits(
        self,
        input_ids: torch.Tensor,
        policy: StepPolicy,
        attention_mask: torch.Tensor | None = None,
        score_slice: slice | None = None,
    ) -> tuple[torch.Tensor, list[float]]:
        """Run full sequence forward under StepPolicy (for MC scoring & exactness verification)."""
        bsz, seq_len = input_ids.shape
        if attention_mask is None:
            pos_ids = torch.arange(seq_len, device=self.device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
            attn_mask = None
        else:
            pos_ids = (attention_mask.cumsum(dim=-1) - 1).clamp_min(0)
            valid_k = attention_mask[:, None, None, :].bool()
            causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=self.device))[None, None, :, :]
            attn_mask = causal & valid_k
            diag = torch.eye(seq_len, dtype=torch.bool, device=self.device)[None, None, :, :]
            attn_mask = attn_mask | diag

        h = self.core.embed_tokens(input_ids)
        cos, sin = self.core.rotary_emb(h, pos_ids)

        h_first = None
        h_prev = None
        d_prev = None
        h_prop_last = None
        alpha_last = torch.ones((bsz, seq_len), device=self.device, dtype=torch.float32)
        geom_last = None
        mean_alphas: list[float] = []

        for ut in range(policy.n_loops):
            h_in = h
            for layer in self.core.layers:
                h, _ = self._run_layer_prefill(layer, h, cos, sin, attn_mask, return_kv=False)
            h_prop = self.core.norm(h)

            if ut == 0:
                h_first = h_prop
                h_prev = h_prop
                h = h_prop
                mean_alphas.append(1.0)
            else:
                # Chunk lm_head if seq_len is large to keep VRAM low
                if policy.feedback_mode in ("ftrs_zero", "ftrs_gate") and seq_len > 128:
                    h_cal_chunks = []
                    alpha_chunks = []
                    for s0 in range(0, seq_len, 128):
                        s1 = min(seq_len, s0 + 128)
                        dp_c = d_prev[:, s0:s1] if d_prev is not None else None
                        hc_c, al_c, _, _ = self._compute_step_alpha_and_state(
                            ut, h_in[:, s0:s1], h_prop[:, s0:s1], dp_c, None, policy
                        )
                        h_cal_chunks.append(hc_c)
                        alpha_chunks.append(al_c)
                    h_cal = torch.cat(h_cal_chunks, dim=1)
                    alpha = torch.cat(alpha_chunks, dim=1)
                    geom = None
                else:
                    h_cal, alpha, _, geom = self._compute_step_alpha_and_state(
                        ut, h_in, h_prop, d_prev, None, policy
                    )

                mean_alphas.append(float(alpha.mean().item()))
                if ut == policy.n_loops - 1:
                    h_prev = h_in
                    h_prop_last = h_prop
                    alpha_last = alpha
                    geom_last = geom
                else:
                    d_prev = h_cal - h_in
                h = h_cal

        # Slice before final readout if only tail tokens are scored
        sl = score_slice if score_slice is not None else slice(None)
        if policy.n_loops == 1:
            logits = self.lm_head(h[:, sl]).float()
        else:
            d_prev_sl = d_prev[:, sl] if d_prev is not None else None
            gl_sl = None
            logits = self._compute_final_logits(
                h_first[:, sl],
                h_prev[:, sl],
                h_prop_last[:, sl],
                h[:, sl],
                None,
                None,
                None,
                alpha_last[:, sl],
                gl_sl,
                d_prev_sl,
                policy,
            )
        return logits, mean_alphas

    @torch.inference_mode()
    def generate_batched(
        self,
        prompt_ids_list: list[list[int]],
        policy: StepPolicy,
        *,
        max_new_tokens: int = 256,
        pad_token_id: int = 0,
        eos_token_ids: set[int] | None = None,
        stop_strings: list[str] | None = None,
        tokenizer: Any = None,
    ) -> tuple[list[list[int]], dict[str, Any]]:
        """Left-padded batched KV-cached greedy decoding at arbitrary depth R with step calibration."""
        bsz = len(prompt_ids_list)
        max_p = max(len(p) for p in prompt_ids_list)
        input_ids = torch.full((bsz, max_p), pad_token_id, device=self.device, dtype=torch.long)
        attn_mask_1d = torch.zeros((bsz, max_p), device=self.device, dtype=torch.long)
        for i, p in enumerate(prompt_ids_list):
            input_ids[i, max_p - len(p) :] = torch.tensor(p, device=self.device, dtype=torch.long)
            attn_mask_1d[i, max_p - len(p) :] = 1

        # Prefill: build per-(ut, layer) KV cache across policy.n_loops
        pos_ids = (attn_mask_1d.cumsum(dim=-1) - 1).clamp_min(0)
        valid_k = attn_mask_1d[:, None, None, :].bool()
        causal = torch.tril(torch.ones((max_p, max_p), dtype=torch.bool, device=self.device))[None, None, :, :]
        prefill_mask = (causal & valid_k) | torch.eye(max_p, dtype=torch.bool, device=self.device)[None, None, :, :]

        h = self.core.embed_tokens(input_ids)
        cos, sin = self.core.rotary_emb(h, pos_ids)

        all_k: list[list[torch.Tensor]] = []
        all_v: list[list[torch.Tensor]] = []
        h_first_last = None
        h_prev_last = None
        h_prop_last = None
        d_prev_last = None
        alpha_last = torch.ones((bsz,), device=self.device, dtype=torch.float32)
        geom_last = None
        z_first_last = None
        z_prev_last = None
        z_prop_last = None

        for ut in range(policy.n_loops):
            h_in_last = h[:, -1, :]
            k_ut: list[torch.Tensor] = []
            v_ut: list[torch.Tensor] = []
            for layer in self.core.layers:
                h, kv = self._run_layer_prefill(layer, h, cos, sin, prefill_mask, return_kv=True)
                assert kv is not None
                k_ut.append(kv[0])
                v_ut.append(kv[1])
            h_prop = self.core.norm(h)
            all_k.append(k_ut)
            all_v.append(v_ut)

            if ut == 0:
                h_first_last = h_prop[:, -1, :]
                h = h_prop
            else:
                # Calibrate the last prompt position (predicting first answer token)
                if policy.feedback_mode != "none":
                    h_cal_last, alpha_l, z_p_l, g_l = self._compute_step_alpha_and_state(
                        ut, h_in_last, h_prop[:, -1, :], d_prev_last, z_prev_last, policy
                    )
                    h_next = h_prop.clone()
                    h_next[:, -1, :] = h_cal_last
                else:
                    h_cal_last = h_prop[:, -1, :]
                    alpha_l = torch.ones((bsz,), device=self.device, dtype=torch.float32)
                    z_p_l = None
                    g_l = None
                    h_next = h_prop

                if ut == policy.n_loops - 1:
                    h_prev_last = h_in_last
                    h_prop_last = h_prop[:, -1, :]
                    alpha_last = alpha_l
                    geom_last = g_l
                    z_prop_last = z_p_l
                else:
                    d_prev_last = h_cal_last - h_in_last
                    z_prev_last = z_p_l
                h = h_next

        first_logits = self._compute_final_logits(
            h_first_last,
            h_prev_last,
            h_prop_last,
            h[:, -1, :],
            z_first_last,
            z_prev_last,
            z_prop_last,
            alpha_last,
            geom_last,
            d_prev_last,
            policy,
        )
        next_tok = first_logits.argmax(dim=-1)  # [B]

        eos_set = eos_token_ids or set()
        finished = torch.zeros(bsz, device=self.device, dtype=torch.bool)
        generated: list[list[int]] = [[] for _ in range(bsz)]
        for i in range(bsz):
            t_i = int(next_tok[i].item())
            generated[i].append(t_i)
            if t_i in eos_set:
                finished[i] = True

        # Track per-loop mean alpha during decode
        alpha_sums = np.zeros(policy.n_loops, dtype=np.float64)
        alpha_counts = np.zeros(policy.n_loops, dtype=np.int64)

        cur_valid_mask = attn_mask_1d  # [B, cur_len]
        next_pos = attn_mask_1d.sum(dim=-1, keepdim=True)  # [B, 1]

        for step_idx in range(1, max_new_tokens):
            if bool(finished.all().item()):
                break
            cur_valid_mask = torch.cat(
                [cur_valid_mask, torch.ones((bsz, 1), device=self.device, dtype=torch.long)], dim=1
            )
            dec_mask = cur_valid_mask[:, None, None, :].bool()  # [B, 1, 1, cur_len]

            h = self.core.embed_tokens(next_tok.unsqueeze(1))  # [B, 1, D]
            cos, sin = self.core.rotary_emb(h, next_pos)

            h_first = None
            h_prev = None
            h_prop_l = None
            d_prev = None
            z_first = None
            z_prev = None
            z_prop_l = None
            alpha_l = torch.ones((bsz,), device=self.device, dtype=torch.float32)
            geom_l = None

            for ut in range(policy.n_loops):
                h_in = h[:, 0, :]
                for l_idx, layer in enumerate(self.core.layers):
                    h, k_cat, v_cat = self._run_layer_decode(
                        layer, h, cos, sin, all_k[ut][l_idx], all_v[ut][l_idx], dec_mask
                    )
                    all_k[ut][l_idx] = k_cat
                    all_v[ut][l_idx] = v_cat
                h_prop = self.core.norm(h)
                h_prop_vec = h_prop[:, 0, :]

                if ut == 0:
                    h_first = h_prop_vec
                    if policy.feedback_mode in ("ftrs_zero", "ftrs_gate") or policy.readout_mode in (
                        "loopcd_logits_fixed",
                        "loopcd_logits_adapt",
                        "ftrs_unified",
                    ):
                        z_first = self.lm_head(h_first)
                        z_prev = z_first
                    h = h_prop
                    alpha_sums[0] += float((~finished).sum().item())
                    alpha_counts[0] += int((~finished).sum().item())
                else:
                    h_cal_vec, al_vec, z_p_vec, g_vec = self._compute_step_alpha_and_state(
                        ut, h_in, h_prop_vec, d_prev, z_prev, policy
                    )
                    active = ~finished
                    if bool(active.any().item()):
                        alpha_sums[ut] += float(al_vec[active].sum().item())
                        alpha_counts[ut] += int(active.sum().item())
                    if ut == policy.n_loops - 1:
                        h_prev = h_in
                        h_prop_l = h_prop_vec
                        alpha_l = al_vec
                        geom_l = g_vec
                        z_prop_l = z_p_vec
                    else:
                        d_prev = h_cal_vec - h_in
                        # Recompute logits at h_cal_vec if calibrated step changed state
                        if policy.feedback_mode in ("ftrs_zero", "ftrs_gate"):
                            if bool((al_vec < 0.999).any().item()):
                                z_prev = self.lm_head(h_cal_vec)
                            else:
                                z_prev = z_p_vec
                    h = h_cal_vec.unsqueeze(1)

            step_logits = self._compute_final_logits(
                h_first,
                h_prev,
                h_prop_l,
                h[:, 0, :],
                z_first,
                z_prev,
                z_prop_l,
                alpha_l,
                geom_l,
                d_prev,
                policy,
            )
            next_tok = step_logits.argmax(dim=-1)
            next_pos = next_pos + 1

            for i in range(bsz):
                if finished[i]:
                    continue
                t_i = int(next_tok[i].item())
                generated[i].append(t_i)
                if t_i in eos_set:
                    finished[i] = True

            # Periodic stop-string check for base 4-shot prompts (every 16 tokens)
            if stop_strings and tokenizer is not None and (step_idx % 16 == 0):
                for i in range(bsz):
                    if finished[i]:
                        continue
                    txt = tokenizer.decode(generated[i], skip_special_tokens=True)
                    if any(ss in txt for ss in stop_strings):
                        finished[i] = True

        mean_alpha_by_loop = [
            float(alpha_sums[u] / max(alpha_counts[u], 1)) for u in range(policy.n_loops)
        ]
        return generated, {"mean_alpha_by_loop": mean_alpha_by_loop}


# ---------------------------------------------------------------------------
# Stage 0: Exactness Verification
# ---------------------------------------------------------------------------

@torch.inference_mode()
def verify_engine_exactness(model, engine: OuroCalibratedEngine, tokenizer, device: torch.device) -> dict[str, Any]:
    """Verify OuroCalibratedEngine matches native HF OuroForCausalLM for B=1, left-padded B=2, and decode."""
    p1 = "Question: If a train travels 60 miles per hour for 3 hours, how far does it go?\nAnswer:"
    p2 = "Question: What is 17 + 25?\nAnswer:"
    ids1 = tokenizer.encode(p1, add_special_tokens=True)
    ids2 = tokenizer.encode(p2, add_special_tokens=True)

    # 1) B=1 exactness
    t1 = torch.tensor([ids1], device=device, dtype=torch.long)
    hf_logits_1 = model(input_ids=t1, use_cache=False).logits.float().cpu()
    eng_logits_1, _ = engine.forward_policy_logits(t1, StepPolicy(name="verify", n_loops=engine.train_loops))
    diff_1 = float((hf_logits_1 - eng_logits_1.cpu()).abs().max().item())

    # 2) Left-padded B=2 exactness
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    max_l = max(len(ids1), len(ids2))
    t2 = torch.full((2, max_l), pad_id, device=device, dtype=torch.long)
    m2 = torch.zeros((2, max_l), device=device, dtype=torch.long)
    for row, ids in enumerate([ids1, ids2]):
        t2[row, max_l - len(ids) :] = torch.tensor(ids, device=device, dtype=torch.long)
        m2[row, max_l - len(ids) :] = 1
    hf_logits_2 = model(input_ids=t2, attention_mask=m2, use_cache=False).logits[:, -1, :].float().cpu()
    eng_logits_2, _ = engine.forward_policy_logits(
        t2, StepPolicy(name="verify2", n_loops=engine.train_loops), attention_mask=m2
    )
    diff_2 = float((hf_logits_2 - eng_logits_2[:, -1, :].cpu()).abs().max().item())

    # 3) 4-step KV-cached greedy decode exactness
    gen_eng, _ = engine.generate_batched(
        [ids1, ids2],
        StepPolicy(name="verify_dec", n_loops=engine.train_loops),
        max_new_tokens=4,
        pad_token_id=pad_id,
    )
    hf_gen_1 = []
    cur = t1.clone()
    for _ in range(4):
        nt = int(model(input_ids=cur, use_cache=False).logits[0, -1].argmax().item())
        hf_gen_1.append(nt)
        cur = torch.cat([cur, torch.tensor([[nt]], device=device, dtype=torch.long)], dim=1)
    decode_exact = gen_eng[0] == hf_gen_1

    res = {
        "max_abs_diff_b1": diff_1,
        "max_abs_diff_b2_leftpad": diff_2,
        "decode_4tok_exact": decode_exact,
        "eng_toks": gen_eng[0],
        "hf_toks": hf_gen_1,
    }
    print(f"[verify] OuroCalibratedEngine exactness: {json.dumps(res)}")
    if not math.isfinite(diff_1) or diff_1 > 0.15 or diff_2 > 0.15:
        raise RuntimeError(f"Engine exactness check FAILED: {res}")
    return res


# ---------------------------------------------------------------------------
# Stage 1: Probe 1A (Teacher-Forced Geometric Duality & FTRS-Gate Calibration)
# ---------------------------------------------------------------------------

def _build_tf_example(item: dict[str, Any], tokenizer, *, is_thinking: bool, max_sol_tokens: int = 160) -> tuple[list[int], int]:
    """Build teacher-forced prompt + reference solution token IDs, returning (full_ids, sol_start_idx)."""
    if is_thinking:
        prompt_text = f"Solve the following math problem step by step and put your final answer in \\boxed{{}}.\n\n{item['question']}\n\nSolution:\n"
    else:
        prompt_text = f"Question: {item['question']}\nAnswer: "
    p_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
    s_ids = tokenizer.encode(item["solution"], add_special_tokens=False)[:max_sol_tokens]
    if len(s_ids) < 4:
        s_ids = tokenizer.encode(item["gold"], add_special_tokens=False)
    return p_ids + s_ids, len(p_ids)


@torch.inference_mode()
def collect_probe1a_transitions(
    engine: OuroCalibratedEngine,
    items: list[dict[str, Any]],
    tokenizer,
    *,
    is_thinking: bool,
    max_loops: int = 8,
    max_sol_tokens: int = 128,
    alpha_grid: list[float] | None = None,
    run_phase: Any = None,
    step_offset: int = 0,
) -> dict[str, Any]:
    """Collect exact per-transition (r-1 -> r) Fisher geometry, Prop 3.4 quantities, and oracle alpha*."""
    alphas = torch.tensor(
        alpha_grid if alpha_grid is not None else [round(0.1 * k, 2) for k in range(19)],
        device=engine.device,
        dtype=torch.float32,
    )  # [G] in [0.0, 1.8]
    idx_a1 = int(torch.argmin((alphas - 1.0).abs()).item())
    idx_a05 = int(torch.argmin((alphas - 0.5).abs()).item())
    idx_a15 = int(torch.argmin((alphas - 1.5).abs()).item())

    # Store compact CPU tensors for every transition across all solution tokens
    recs_feats: list[torch.Tensor] = []  # [N_tok, 5]
    recs_trans: list[np.ndarray] = []  # transition index u in 1..max_loops-1
    recs_kappa: list[np.ndarray] = []
    recs_kl: list[np.ndarray] = []
    recs_margin: list[np.ndarray] = []
    recs_cos: list[np.ndarray] = []
    recs_slope_a: list[np.ndarray] = []
    recs_du1: list[np.ndarray] = []
    recs_prop34_err: list[np.ndarray] = []
    recs_alpha_star: list[np.ndarray] = []
    recs_du_grid: list[np.ndarray] = []  # [N_tok, G] exact Delta_U(alpha) along affine logit ray!

    for idx_i, item in enumerate(items, 1):
        full_ids, sol_start = _build_tf_example(item, tokenizer, is_thinking=is_thinking, max_sol_tokens=max_sol_tokens)
        if len(full_ids) <= sol_start + 2:
            continue
        inp = torch.tensor([full_ids[:-1]], device=engine.device, dtype=torch.long)
        tgt = torch.tensor(full_ids[1:], device=engine.device, dtype=torch.long)
        # Predict solution tokens: positions from (sol_start - 1) to end
        sl_start = max(0, sol_start - 1)
        tgt_sl = tgt[sl_start:]  # [T]
        n_tok = int(tgt_sl.shape[0])
        if n_tok == 0:
            continue

        hiddens = engine.forward_all_loop_hiddens(inp, max_loops=max_loops)
        # Slice only solution-predicting positions for fast readout
        h_sl = [h[0, sl_start:, :] for h in hiddens]  # each [T, D]
        z_sl = [engine.lm_head(hs).float() for hs in h_sl]  # each [T, V]

        d_prev = None
        for u in range(1, max_loops):
            h_prev = h_sl[u - 1]
            h_curr = h_sl[u]
            z_prev = z_sl[u - 1]
            z_curr = z_sl[u]
            d_curr = h_curr - h_prev

            geom = compute_fisher_geometry(z_prev, z_curr, d_curr, d_prev, h_prev)
            feats = stack_index_free_features(geom).cpu()  # [T, 5]

            # Exact Proposition 3.4 slope A_r = (z_curr - z_prev)[y*] - E_{p_prev}[z_curr - z_prev]
            dz = z_curr - z_prev
            dz_gold = dz.gather(-1, tgt_sl.unsqueeze(-1)).squeeze(-1)  # [T]
            slope_a = dz_gold - geom["mean_dz"]

            # Exact Delta_U(alpha) across the 19-point grid [0.0 .. 1.8]
            log_p0 = F.log_softmax(z_prev, dim=-1)
            ll0 = log_p0.gather(-1, tgt_sl.unsqueeze(-1)).squeeze(-1)  # [T]

            du_cols = []
            for a_val in alphas.tolist():
                if a_val == 0.0:
                    du_cols.append(torch.zeros_like(ll0))
                else:
                    za = z_prev + a_val * dz
                    lpa = F.log_softmax(za, dim=-1)
                    lla = lpa.gather(-1, tgt_sl.unsqueeze(-1)).squeeze(-1)
                    du_cols.append(lla - ll0)
            du_mat = torch.stack(du_cols, dim=-1)  # [T, G]
            best_idx = du_mat.argmax(dim=-1)
            alpha_star = alphas[best_idx]
            du1 = du_mat[:, idx_a1]
            prop34_err = (du1 - (slope_a - geom["kl_fwd"])).abs()

            recs_feats.append(feats)
            recs_trans.append(np.full(n_tok, u, dtype=np.int32))
            recs_kappa.append(geom["kappa"].cpu().numpy())
            recs_kl.append(geom["kl_fwd"].cpu().numpy())
            recs_margin.append(geom["margin"].cpu().numpy())
            recs_cos.append(geom["cos_mom"].cpu().numpy())
            recs_slope_a.append(slope_a.cpu().numpy())
            recs_du1.append(du1.cpu().numpy())
            recs_prop34_err.append(prop34_err.cpu().numpy())
            recs_alpha_star.append(alpha_star.cpu().numpy())
            recs_du_grid.append(du_mat.cpu().numpy())

            d_prev = d_curr

        if run_phase is not None:
            run_phase.step(step_offset + idx_i)

    return {
        "alphas": alphas.cpu().numpy(),
        "idx_a1": idx_a1,
        "idx_a05": idx_a05,
        "idx_a15": idx_a15,
        "feats": torch.cat(recs_feats, dim=0),
        "trans": np.concatenate(recs_trans),
        "kappa": np.concatenate(recs_kappa),
        "kl": np.concatenate(recs_kl),
        "margin": np.concatenate(recs_margin),
        "cos": np.concatenate(recs_cos),
        "slope_a": np.concatenate(recs_slope_a),
        "du1": np.concatenate(recs_du1),
        "prop34_err": np.concatenate(recs_prop34_err),
        "alpha_star": np.concatenate(recs_alpha_star),
        "du_grid": np.concatenate(recs_du_grid, axis=0),
    }


def _interp_du_from_grid(du_grid: np.ndarray, alphas: np.ndarray, alpha_pred: np.ndarray) -> np.ndarray:
    """Linearly interpolate exact Delta_U(alpha_pred) from the dense 0.1-spaced grid [0, 1.8]."""
    step = float(alphas[1] - alphas[0])
    a_clip = np.clip(alpha_pred, float(alphas[0]), float(alphas[-1]))
    pos = (a_clip - float(alphas[0])) / step
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, len(alphas) - 1)
    frac = pos - lo
    rows = np.arange(du_grid.shape[0])
    return (1.0 - frac) * du_grid[rows, lo] + frac * du_grid[rows, hi]


def train_ftrs_gate_on_probe1a(
    train_data: dict[str, Any],
    device: torch.device,
    *,
    epochs: int = 25,
    batch_size: int = 2048,
    lr: float = 3e-3,
    seed: int = 20261124,
) -> tuple[FTRSGate, dict[str, Any]]:
    """Train index-free 5-feature FTRSGate directly on exact Delta_U(alpha) utility landscape!"""
    torch.manual_seed(seed)
    feats = train_data["feats"].to(device)  # [N, 5]
    du_grid = torch.tensor(train_data["du_grid"], device=device, dtype=torch.float32)  # [N, G]
    alphas = torch.tensor(train_data["alphas"], device=device, dtype=torch.float32)
    step = float(alphas[1] - alphas[0])
    n_grid = du_grid.shape[1]

    gate = FTRSGate(in_dim=5, hidden_dim=32, max_alpha=float(alphas[-1])).to(device)
    gate.set_norm_stats(feats)

    opt = torch.optim.AdamW(gate.parameters(), lr=lr, weight_decay=1e-4)
    n = feats.shape[0]
    row_arange = torch.arange(batch_size, device=device)

    for _ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for s0 in range(0, n, batch_size):
            idx = perm[s0 : s0 + batch_size]
            xb = feats[idx]
            gb = du_grid[idx]
            a_pred = gate(xb)  # [B] in [0, 1.8]
            pos = (a_pred / step).clamp(0.0, n_grid - 1.0001)
            lo = pos.long()
            hi = (lo + 1).clamp(max=n_grid - 1)
            frac = pos - lo.float()
            r_idx = row_arange[: idx.shape[0]]
            du_pred = (1.0 - frac) * gb[r_idx, lo] + frac * gb[r_idx, hi]
            # Maximize expected utility gain Delta_U(alpha)
            loss = -du_pred.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    gate.eval()
    return gate, {"n_train_tokens_x_loops": int(n), "trainable_params": sum(p.numel() for p in gate.parameters())}


def analyze_probe1a_split(
    split_name: str,
    data: dict[str, Any],
    *,
    tau_fb: float,
    gate: FTRSGate,
    device: torch.device,
    train_loops: int = 4,
) -> dict[str, Any]:
    """Compute decisive geometric duality diagnostics and policy utility gains on a held-out split."""
    alphas = data["alphas"]
    du_grid = data["du_grid"]
    kappa = data["kappa"]
    kl = data["kl"]
    margin = data["margin"]
    cos_m = data["cos"]
    slope_a = data["slope_a"]
    du1 = data["du1"]
    a_star = data["alpha_star"]
    trans = data["trans"]

    # Predict alphas for FTRS-Zero and FTRS-Gate
    a_ftrs_zero = ftrs_zero_feedback_alpha(
        torch.tensor(kappa, dtype=torch.float32),
        torch.tensor(cos_m, dtype=torch.float32),
        tau_fb=tau_fb,
        use_momentum=True,
    ).numpy()
    with torch.inference_mode():
        a_ftrs_gate = gate(data["feats"].to(device)).cpu().numpy()

    # LoopCD-style adaptive extrapolation: alpha = 1 + 0.5 * (1 - margin)
    a_loopcd = 1.0 + 0.5 * (1.0 - margin)
    a_fixed05 = np.full_like(a_ftrs_zero, 0.5)

    du_native = du1
    du_fixed05 = _interp_du_from_grid(du_grid, alphas, a_fixed05)
    du_loopcd = _interp_du_from_grid(du_grid, alphas, a_loopcd)
    du_ftrs_zero = _interp_du_from_grid(du_grid, alphas, a_ftrs_zero)
    du_ftrs_gate = _interp_du_from_grid(du_grid, alphas, a_ftrs_gate)
    du_oracle = du_grid.max(axis=1)

    # Overshoot masks:
    # Finite-step failure: A_r > 0 and Delta_U(1) < 0
    pos_slope = slope_a > 1e-6
    fsf_mask = pos_slope & (du1 < -1e-6)
    harmful_mask = du1 < -1e-6

    # Overall correlations & AUROCs
    sp_kappa_astar = spearman_corr(kappa, a_star)
    sp_kl_astar = spearman_corr(kl, a_star)
    auroc_fsf_given_pos = binary_auroc(kappa[pos_slope], fsf_mask[pos_slope].astype(np.int64))
    auroc_harmful_all = binary_auroc(kappa, harmful_mask.astype(np.int64))

    # Extrapolation transitions (r >= train_loops, i.e. 4->5, 5->6, 6->7, 7->8)
    ext_mask = trans >= train_loops
    in_mask = trans < train_loops

    # Curvature quintiles (Q1 lowest kappa -> Q5 highest kappa)
    q_edges = np.quantile(kappa, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    quintiles: list[dict[str, Any]] = []
    for qi in range(5):
        lo_e, hi_e = q_edges[qi], q_edges[qi + 1]
        qm = (kappa >= lo_e) & (kappa <= hi_e if qi == 4 else kappa < hi_e)
        if qm.sum() == 0:
            continue
        quintiles.append(
            {
                "quintile": f"Q{qi+1}",
                "n": int(qm.sum()),
                "kappa_median": float(np.median(kappa[qm])),
                "mean_alpha_star": float(np.mean(a_star[qm])),
                "frac_astar_gt1": float(np.mean(a_star[qm] > 1.05)),
                "frac_astar_lt1": float(np.mean(a_star[qm] < 0.95)),
                "frac_harmful": float(np.mean(harmful_mask[qm])),
                "du_native": float(np.mean(du_native[qm])),
                "du_loopcd": float(np.mean(du_loopcd[qm])),
                "du_fixed05": float(np.mean(du_fixed05[qm])),
                "du_ftrs_zero": float(np.mean(du_ftrs_zero[qm])),
                "du_ftrs_gate": float(np.mean(du_ftrs_gate[qm])),
                "du_oracle": float(np.mean(du_oracle[qm])),
            }
        )

    # Low-margin stratification (lowest 20% margin tokens split by median kappa within low-margin!)
    m_thresh = float(np.quantile(margin, 0.20))
    low_m = margin <= m_thresh
    km_med = float(np.median(kappa[low_m])) if low_m.sum() > 0 else 0.0
    lm_low_k = low_m & (kappa <= km_med)
    lm_high_k = low_m & (kappa > km_med)

    def _slice_stats(mask: np.ndarray) -> dict[str, float]:
        if mask.sum() == 0:
            return {"n": 0}
        return {
            "n": int(mask.sum()),
            "mean_alpha_star": float(np.mean(a_star[mask])),
            "du_native": float(np.mean(du_native[mask])),
            "du_loopcd": float(np.mean(du_loopcd[mask])),
            "du_fixed05": float(np.mean(du_fixed05[mask])),
            "du_ftrs_zero": float(np.mean(du_ftrs_zero[mask])),
            "du_ftrs_gate": float(np.mean(du_ftrs_gate[mask])),
            "du_oracle": float(np.mean(du_oracle[mask])),
        }

    per_trans: dict[str, dict[str, Any]] = {}
    for u in sorted(np.unique(trans).tolist()):
        tm = trans == u
        per_trans[f"{u}->{u+1}"] = {
            "n": int(tm.sum()),
            "kappa_median": float(np.median(kappa[tm])),
            "frac_fsf": float(np.mean(fsf_mask[tm])),
            "frac_harmful": float(np.mean(harmful_mask[tm])),
            "spearman_kappa_astar": spearman_corr(kappa[tm], a_star[tm]),
            "auroc_fsf": binary_auroc(kappa[tm & pos_slope], fsf_mask[tm & pos_slope].astype(np.int64))
            if (tm & pos_slope).sum() > 5
            else 0.5,
            "mean_alpha_star": float(np.mean(a_star[tm])),
            "du_native": float(np.mean(du_native[tm])),
            "du_loopcd": float(np.mean(du_loopcd[tm])),
            "du_fixed05": float(np.mean(du_fixed05[tm])),
            "du_ftrs_zero": float(np.mean(du_ftrs_zero[tm])),
            "du_ftrs_gate": float(np.mean(du_ftrs_gate[tm])),
        }

    return {
        "split": split_name,
        "n_total": int(len(kappa)),
        "prop34_max_err": float(np.max(data["prop34_err"])),
        "prop34_mean_err": float(np.mean(data["prop34_err"])),
        "spearman_kappa_astar_all": sp_kappa_astar,
        "spearman_kl_astar_all": sp_kl_astar,
        "auroc_fsf_given_pos_slope": auroc_fsf_given_pos,
        "auroc_harmful_all": auroc_harmful_all,
        "in_depth_stats": _slice_stats(in_mask),
        "extrap_depth_stats": _slice_stats(ext_mask),
        "low_margin_low_kappa": _slice_stats(lm_low_k),
        "low_margin_high_kappa": _slice_stats(lm_high_k),
        "quintiles": quintiles,
        "per_transition": per_trans,
    }


# ---------------------------------------------------------------------------
# Stage 2: Probe 1B (ARC-Challenge MC + GSM8K & MATH-500 Free Generation)
# ---------------------------------------------------------------------------

GSM8K_4SHOT = """Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.
#### 6

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.
#### 5

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.
#### 39

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.
#### 8

"""


@torch.inference_mode()
def evaluate_arc_challenge(
    engine: OuroCalibratedEngine,
    samples: list[dict[str, Any]],
    tokenizer,
    policy: StepPolicy,
) -> dict[str, Any]:
    """Evaluate ARC-Challenge exact length-normalized choice log-likelihood accuracy."""
    correct_flags: list[int] = []
    t0 = time.time()
    for item in samples:
        prompt = f"Question: {item['question']}\nAnswer:"
        p_ids = tokenizer.encode(prompt, add_special_tokens=True)
        scores = []
        for choice_txt in item["choice_texts"]:
            c_ids = tokenizer.encode(" " + choice_txt, add_special_tokens=False)
            if not c_ids:
                scores.append(-1e9)
                continue
            full = p_ids + c_ids
            inp = torch.tensor([full[:-1]], device=engine.device, dtype=torch.long)
            tgt = torch.tensor(full[len(p_ids) :], device=engine.device, dtype=torch.long)
            sl = slice(len(p_ids) - 1, len(full) - 1)
            logits, _ = engine.forward_policy_logits(inp, policy, score_slice=sl)
            lp = F.log_softmax(logits[0], dim=-1)
            tok_ll = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            scores.append(float(tok_ll.mean().item()))
        pred = int(np.argmax(scores))
        correct_flags.append(1 if pred == item["gold_idx"] else 0)
    return {
        "policy": policy.name,
        "n_loops": policy.n_loops,
        "accuracy": float(np.mean(correct_flags)),
        "n_correct": int(sum(correct_flags)),
        "n_total": len(correct_flags),
        "elapsed_sec": time.time() - t0,
        "correct_flags": correct_flags,
    }


def _format_gen_prompt(item: dict[str, Any], tokenizer, *, is_thinking: bool, task: str) -> list[int]:
    if is_thinking:
        if task == "gsm8k":
            user_msg = f"Solve this math problem step by step. Put your final numeric answer in \\boxed{{}}.\n\n{item['question']}"
        else:
            user_msg = f"Solve the following math problem step by step. Put your final answer inside \\boxed{{}}.\n\n{item['question']}"
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            prompt_str = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=False,
                add_generation_prompt=True,
            )
            return tokenizer.encode(prompt_str, add_special_tokens=False)
        return tokenizer.encode(f"User: {user_msg}\nAssistant:", add_special_tokens=True)
    else:
        prompt_str = f"{GSM8K_4SHOT}Question: {item['question']}\nAnswer:"
        return tokenizer.encode(prompt_str, add_special_tokens=True)


@torch.inference_mode()
def evaluate_math_generation(
    engine: OuroCalibratedEngine,
    items: list[dict[str, Any]],
    tokenizer,
    policy: StepPolicy,
    *,
    is_thinking: bool,
    task: str,
    max_new_tokens: int,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Run batched KV-cached free generation and score exact math accuracy + completion rate."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    eos_ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))
    for tok_str in ("<|im_end|>", "<|endoftext|>", "</s>"):
        tid = tokenizer.convert_tokens_to_ids(tok_str)
        if isinstance(tid, int) and tid >= 0:
            eos_ids.add(tid)

    stop_strings = None if is_thinking else ["\nQuestion:", "\n\nQuestion:"]

    prompts = [_format_gen_prompt(it, tokenizer, is_thinking=is_thinking, task=task) for it in items]
    correct_flags: list[int] = []
    finished_think_flags: list[int] = []
    gen_lens: list[int] = []
    alpha_accum = np.zeros(policy.n_loops, dtype=np.float64)
    n_batches = 0
    t0 = time.time()

    for s0 in range(0, len(items), batch_size):
        batch_items = items[s0 : s0 + batch_size]
        batch_prompts = prompts[s0 : s0 + batch_size]
        gen_ids_batch, stats = engine.generate_batched(
            batch_prompts,
            policy,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_id,
            eos_token_ids=eos_ids,
            stop_strings=stop_strings,
            tokenizer=tokenizer,
        )
        alpha_accum += np.asarray(stats["mean_alpha_by_loop"], dtype=np.float64)
        n_batches += 1

        for it, gids in zip(batch_items, gen_ids_batch):
            txt = tokenizer.decode(gids, skip_special_tokens=True)
            if not is_thinking and "\nQuestion:" in txt:
                txt = txt.split("\nQuestion:")[0]
            pred = extract_predicted_math_answer(txt, mode=task)
            is_ok = int(pred != "" and pred == it["gold"])
            correct_flags.append(is_ok)
            gen_lens.append(len(gids))
            if is_thinking:
                finished_think_flags.append(1 if ("</think>" in txt or len(gids) < max_new_tokens) else 0)
            else:
                finished_think_flags.append(1 if ("####" in txt or len(gids) < max_new_tokens) else 0)

    elapsed = time.time() - t0
    return {
        "policy": policy.name,
        "task": task,
        "n_loops": policy.n_loops,
        "accuracy": float(np.mean(correct_flags)),
        "n_correct": int(sum(correct_flags)),
        "n_total": len(correct_flags),
        "completion_rate": float(np.mean(finished_think_flags)),
        "mean_gen_tokens": float(np.mean(gen_lens)),
        "elapsed_sec": elapsed,
        "mean_alpha_by_loop": (alpha_accum / max(n_batches, 1)).tolist(),
        "correct_flags": correct_flags,
    }


# ---------------------------------------------------------------------------
# Per-Model Full Pipeline
# ---------------------------------------------------------------------------

def run_model_pilot(
    mspec: dict[str, Any],
    *,
    cfg: dict[str, Any],
    model_dir: Path,
    gsm_train: list[dict[str, Any]],
    gsm_test: list[dict[str, Any]],
    math_test: list[dict[str, Any]],
    math_gen_items: list[dict[str, Any]],
    arc_samples: list[dict[str, Any]],
    device: torch.device,
    run_obj: Any,
) -> dict[str, Any]:
    mkey = mspec["key"]
    is_thinking = bool(mspec["is_thinking"])
    train_loops = int(mspec["train_loops"])
    max_probe_loops = int(cfg.get("max_probe_loops", 8))
    local_path = model_dir / mspec["local_dir"]

    print(f"\n[ftrs:{mkey}] Loading {local_path} (is_thinking={is_thinking}) ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(local_path), trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(local_path),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    engine = OuroCalibratedEngine(model, device)

    # 0) Verify exactness
    with run_obj.phase(f"{mkey}_verify", total=1, unit="check") as pv:
        verify_res = verify_engine_exactness(model, engine, tokenizer, device)
        pv.step(1)

    # 1) Probe 1A: Teacher-Forced Geometric Duality & FTRS-Gate Calibration
    n_tf_total = len(gsm_train) + len(gsm_test) + len(math_test)
    with run_obj.phase(f"{mkey}_probe1a", total=n_tf_total, unit="seqs") as p1a:
        tr_data = collect_probe1a_transitions(
            engine, gsm_train, tokenizer, is_thinking=is_thinking, max_loops=max_probe_loops, run_phase=p1a, step_offset=0
        )
        te_gsm_data = collect_probe1a_transitions(
            engine, gsm_test, tokenizer, is_thinking=is_thinking, max_loops=max_probe_loops, run_phase=p1a, step_offset=len(gsm_train)
        )
        te_math_data = collect_probe1a_transitions(
            engine, math_test, tokenizer, is_thinking=is_thinking, max_loops=max_probe_loops, run_phase=p1a, step_offset=len(gsm_train) + len(gsm_test)
        )

    # Select tau_fb strictly on GSM8K-train (zero test leakage!)
    tau_grid = [float(x) for x in cfg.get("tau_fb_grid", [0.05, 0.15, 0.35])]
    best_tau = float(cfg.get("tau_fb_default", 0.15))
    best_tr_du = -1e9
    tau_scores: dict[str, float] = {}
    for tau_c in tau_grid:
        a_tr = ftrs_zero_feedback_alpha(
            torch.tensor(tr_data["kappa"], dtype=torch.float32),
            torch.tensor(tr_data["cos"], dtype=torch.float32),
            tau_fb=tau_c,
            use_momentum=True,
        ).numpy()
        du_tr = float(np.mean(_interp_du_from_grid(tr_data["du_grid"], tr_data["alphas"], a_tr)))
        tau_scores[str(tau_c)] = du_tr
        if du_tr > best_tr_du:
            best_tr_du = du_tr
            best_tau = tau_c
    print(f"[ftrs:{mkey}] Train-selected tau_fb={best_tau} (grid={json.dumps(tau_scores)})")

    # Train 5-feature index-free FTRS-Gate strictly on GSM8K-train
    gate, gate_info = train_ftrs_gate_on_probe1a(tr_data, device, seed=int(cfg.get("seed", 20261124)))

    probe1a_gsm = analyze_probe1a_split(
        "gsm8k_test", te_gsm_data, tau_fb=best_tau, gate=gate, device=device, train_loops=train_loops
    )
    probe1a_math = analyze_probe1a_split(
        "math500_test", te_math_data, tau_fb=best_tau, gate=gate, device=device, train_loops=train_loops
    )
    print(
        f"[ftrs:{mkey}] Probe 1A GSM8K: Spearman(kappa,alpha*)={probe1a_gsm['spearman_kappa_astar_all']:+.3f} "
        f"AUROC(FSF)={probe1a_gsm['auroc_fsf_given_pos_slope']:.3f} Prop3.4_err={probe1a_gsm['prop34_mean_err']:.2e}"
    )
    print(
        f"[ftrs:{mkey}] Probe 1A MATH500: Spearman(kappa,alpha*)={probe1a_math['spearman_kappa_astar_all']:+.3f} "
        f"AUROC(FSF)={probe1a_math['auroc_fsf_given_pos_slope']:.3f}"
    )
    del tr_data, te_gsm_data, te_math_data
    gc.collect()

    # Build clean comparison suite for Probe 1B
    w_log = float(mspec.get("loopcd_logits_omega", 0.5))
    w_hid = float(mspec.get("loopcd_hidden_omega", 0.25))
    w_max = float(mspec.get("loopcd_adapt_omega_max", 1.0))
    tau_ext = float(cfg.get("tau_ext_default", 0.25))

    arc_policies = [
        StepPolicy("native_R2", 2),
        StepPolicy("native_R4", 4),
        StepPolicy("native_R6", 6),
        StepPolicy("native_R8", 8),
        StepPolicy("loopcd_logits_R4", 4, readout_mode="loopcd_logits_fixed", omega=w_log),
        StepPolicy("loopcd_adapt_R4", 4, readout_mode="loopcd_logits_adapt", omega_max=w_max),
        StepPolicy("loopcd_adapt_R8", 8, readout_mode="loopcd_logits_adapt", omega_max=w_max),
        StepPolicy("loopcd_hidden_R4", 4, readout_mode="loopcd_hidden_fixed", omega=w_hid),
        StepPolicy("fixed_damp05_R6", 6, feedback_mode="fixed", fixed_alpha=0.5, fixed_start_loop=train_loops),
        StepPolicy("fixed_damp05_R8", 8, feedback_mode="fixed", fixed_alpha=0.5, fixed_start_loop=train_loops),
        StepPolicy("ftrs_zero_euclid_R8", 8, feedback_mode="ftrs_zero", manifold="euclidean", tau_fb=best_tau),
        StepPolicy("ftrs_zero_mfld_R4", 4, feedback_mode="ftrs_zero", manifold="rms_match", tau_fb=best_tau),
        StepPolicy("ftrs_zero_mfld_R6", 6, feedback_mode="ftrs_zero", manifold="rms_match", tau_fb=best_tau),
        StepPolicy("ftrs_zero_mfld_R8", 8, feedback_mode="ftrs_zero", manifold="rms_match", tau_fb=best_tau),
        StepPolicy("ftrs_unified_R2", 2, feedback_mode="ftrs_zero", manifold="rms_match", readout_mode="ftrs_unified", tau_fb=best_tau, tau_ext=tau_ext, omega_max=w_max),
        StepPolicy("ftrs_unified_R4", 4, feedback_mode="ftrs_zero", manifold="rms_match", readout_mode="ftrs_unified", tau_fb=best_tau, tau_ext=tau_ext, omega_max=w_max),
        StepPolicy("ftrs_unified_R6", 6, feedback_mode="ftrs_zero", manifold="rms_match", readout_mode="ftrs_unified", tau_fb=best_tau, tau_ext=tau_ext, omega_max=w_max),
        StepPolicy("ftrs_unified_R8", 8, feedback_mode="ftrs_zero", manifold="rms_match", readout_mode="ftrs_unified", tau_fb=best_tau, tau_ext=tau_ext, omega_max=w_max),
        StepPolicy("ftrs_gate_mfld_R6", 6, feedback_mode="ftrs_gate", manifold="rms_match", gate_module=gate),
        StepPolicy("ftrs_gate_mfld_R8", 8, feedback_mode="ftrs_gate", manifold="rms_match", gate_module=gate),
    ]

    # 2) ARC-Challenge evaluation
    arc_results: dict[str, dict[str, Any]] = {}
    with run_obj.phase(f"{mkey}_arc", total=len(arc_policies), unit="conds") as parc:
        for i_p, pol in enumerate(arc_policies, 1):
            r = evaluate_arc_challenge(engine, arc_samples, tokenizer, pol)
            arc_results[pol.name] = r
            print(f"[ftrs:{mkey}:ARC] {pol.name:20s} R={pol.n_loops} acc={r['accuracy']*100:.2f}% ({r['n_correct']}/{r['n_total']}) [{r['elapsed_sec']:.1f}s]")
            parc.row(f"{mkey}_arc", policy=pol.name, n_loops=pol.n_loops, acc=r["accuracy"])
            parc.step(i_p)

    # 3) Free-Generation GSM8K (+ MATH-500 for Thinking model!)
    gen_policies = [
        StepPolicy("native_R2", 2),
        StepPolicy("native_R4", 4),
        StepPolicy("native_R6", 6),
        StepPolicy("native_R8", 8),
        StepPolicy("loopcd_logits_R4", 4, readout_mode="loopcd_logits_fixed", omega=w_log),
        StepPolicy("loopcd_adapt_R4", 4, readout_mode="loopcd_logits_adapt", omega_max=w_max),
        StepPolicy("loopcd_adapt_R8", 8, readout_mode="loopcd_logits_adapt", omega_max=w_max),
        StepPolicy("fixed_damp05_R8", 8, feedback_mode="fixed", fixed_alpha=0.5, fixed_start_loop=train_loops),
        StepPolicy("ftrs_zero_euclid_R8", 8, feedback_mode="ftrs_zero", manifold="euclidean", tau_fb=best_tau),
        StepPolicy("ftrs_zero_mfld_R6", 6, feedback_mode="ftrs_zero", manifold="rms_match", tau_fb=best_tau),
        StepPolicy("ftrs_zero_mfld_R8", 8, feedback_mode="ftrs_zero", manifold="rms_match", tau_fb=best_tau),
        StepPolicy("ftrs_unified_R4", 4, feedback_mode="ftrs_zero", manifold="rms_match", readout_mode="ftrs_unified", tau_fb=best_tau, tau_ext=tau_ext, omega_max=w_max),
        StepPolicy("ftrs_unified_R8", 8, feedback_mode="ftrs_zero", manifold="rms_match", readout_mode="ftrs_unified", tau_fb=best_tau, tau_ext=tau_ext, omega_max=w_max),
        StepPolicy("ftrs_gate_mfld_R8", 8, feedback_mode="ftrs_gate", manifold="rms_match", gate_module=gate),
    ]

    n_gsm_gen = int(cfg["n_gsm_gen_think"] if is_thinking else cfg["n_gsm_gen_base"])
    max_tok_gsm = int(cfg["max_new_tokens_think_gsm"] if is_thinking else cfg["max_new_tokens_base"])
    bsz_dec = int(cfg.get("decode_batch_size", 16))
    gsm_gen_items = gsm_test[:n_gsm_gen]

    gsm_gen_results: dict[str, dict[str, Any]] = {}
    with run_obj.phase(f"{mkey}_gsm_gen", total=len(gen_policies), unit="conds") as pgsm:
        for i_p, pol in enumerate(gen_policies, 1):
            r = evaluate_math_generation(
                engine,
                gsm_gen_items,
                tokenizer,
                pol,
                is_thinking=is_thinking,
                task="gsm8k",
                max_new_tokens=max_tok_gsm,
                batch_size=bsz_dec,
            )
            gsm_gen_results[pol.name] = r
            al_str = ",".join(f"{x:.2f}" for x in r["mean_alpha_by_loop"])
            print(
                f"[ftrs:{mkey}:GSM8K] {pol.name:20s} R={pol.n_loops} acc={r['accuracy']*100:.2f}% "
                f"({r['n_correct']}/{r['n_total']}) comp={r['completion_rate']*100:.0f}% "
                f"tok={r['mean_gen_tokens']:.0f} alpha=[{al_str}] [{r['elapsed_sec']:.1f}s]"
            )
            pgsm.row(f"{mkey}_gsm", policy=pol.name, n_loops=pol.n_loops, acc=r["accuracy"])
            pgsm.step(i_p)

    math_gen_results: dict[str, dict[str, Any]] = {}
    if is_thinking and math_gen_items:
        math_policies = [
            StepPolicy("native_R4", 4),
            StepPolicy("native_R8", 8),
            StepPolicy("loopcd_adapt_R4", 4, readout_mode="loopcd_logits_adapt", omega_max=w_max),
            StepPolicy("loopcd_adapt_R8", 8, readout_mode="loopcd_logits_adapt", omega_max=w_max),
            StepPolicy("fixed_damp05_R8", 8, feedback_mode="fixed", fixed_alpha=0.5, fixed_start_loop=train_loops),
            StepPolicy("ftrs_zero_mfld_R8", 8, feedback_mode="ftrs_zero", manifold="rms_match", tau_fb=best_tau),
            StepPolicy("ftrs_unified_R4", 4, feedback_mode="ftrs_zero", manifold="rms_match", readout_mode="ftrs_unified", tau_fb=best_tau, tau_ext=tau_ext, omega_max=w_max),
            StepPolicy("ftrs_unified_R8", 8, feedback_mode="ftrs_zero", manifold="rms_match", readout_mode="ftrs_unified", tau_fb=best_tau, tau_ext=tau_ext, omega_max=w_max),
            StepPolicy("ftrs_gate_mfld_R8", 8, feedback_mode="ftrs_gate", manifold="rms_match", gate_module=gate),
        ]
        max_tok_math = int(cfg.get("max_new_tokens_think_math", 1280))
        with run_obj.phase(f"{mkey}_math_gen", total=len(math_policies), unit="conds") as pmath:
            for i_p, pol in enumerate(math_policies, 1):
                r = evaluate_math_generation(
                    engine,
                    math_gen_items,
                    tokenizer,
                    pol,
                    is_thinking=True,
                    task="math500",
                    max_new_tokens=max_tok_math,
                    batch_size=bsz_dec,
                )
                math_gen_results[pol.name] = r
                print(
                    f"[ftrs:{mkey}:MATH500] {pol.name:20s} R={pol.n_loops} acc={r['accuracy']*100:.2f}% "
                    f"({r['n_correct']}/{r['n_total']}) comp={r['completion_rate']*100:.0f}% "
                    f"tok={r['mean_gen_tokens']:.0f} [{r['elapsed_sec']:.1f}s]"
                )
                pmath.row(f"{mkey}_math", policy=pol.name, n_loops=pol.n_loops, acc=r["accuracy"])
                pmath.step(i_p)

    # Paired bootstrap contrasts
    def _paired(res_map: dict[str, dict[str, Any]], a: str, b: str) -> dict[str, Any] | None:
        if a not in res_map or b not in res_map:
            return None
        fa = np.asarray(res_map[a]["correct_flags"], dtype=np.float64)
        fb = np.asarray(res_map[b]["correct_flags"], dtype=np.float64)
        mean_d, lo, hi = paired_bootstrap_ci(fa - fb)
        return {"diff_pp": mean_d * 100.0, "ci95_pp": [lo * 100.0, hi * 100.0], "fixed": int(((fa == 1) & (fb == 0)).sum()), "broken": int(((fa == 0) & (fb == 1)).sum())}

    contrasts: dict[str, Any] = {}
    for bname, rmap in [("arc", arc_results), ("gsm8k", gsm_gen_results), ("math500", math_gen_results)]:
        if not rmap:
            continue
        contrasts[bname] = {
            k: v
            for k, v in {
                "native_R8_vs_R4": _paired(rmap, "native_R8", "native_R4"),
                "ftrs_zero_mfld_R8_vs_native_R8": _paired(rmap, "ftrs_zero_mfld_R8", "native_R8"),
                "ftrs_zero_mfld_R8_vs_native_R4": _paired(rmap, "ftrs_zero_mfld_R8", "native_R4"),
                "ftrs_zero_mfld_R8_vs_fixed05_R8": _paired(rmap, "ftrs_zero_mfld_R8", "fixed_damp05_R8"),
                "ftrs_zero_mfld_R8_vs_euclid_R8": _paired(rmap, "ftrs_zero_mfld_R8", "ftrs_zero_euclid_R8"),
                "ftrs_unified_R4_vs_native_R4": _paired(rmap, "ftrs_unified_R4", "native_R4"),
                "ftrs_unified_R4_vs_loopcd_adapt_R4": _paired(rmap, "ftrs_unified_R4", "loopcd_adapt_R4"),
                "ftrs_unified_R8_vs_native_R8": _paired(rmap, "ftrs_unified_R8", "native_R8"),
                "ftrs_unified_R8_vs_native_R4": _paired(rmap, "ftrs_unified_R8", "native_R4"),
                "ftrs_unified_R8_vs_loopcd_adapt_R8": _paired(rmap, "ftrs_unified_R8", "loopcd_adapt_R8"),
                "ftrs_gate_R8_vs_native_R8": _paired(rmap, "ftrs_gate_mfld_R8", "native_R8"),
                "ftrs_gate_R8_vs_native_R4": _paired(rmap, "ftrs_gate_mfld_R8", "native_R4"),
            }.items()
            if v is not None
        }

    def _strip(rmap: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {k: {ck: cv for ck, cv in v.items() if ck != "correct_flags"} for k, v in rmap.items()}

    del model, engine, gate
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model_key": mkey,
        "local_path": str(local_path),
        "is_thinking": is_thinking,
        "train_loops": train_loops,
        "verify": verify_res,
        "selected_tau_fb": best_tau,
        "tau_grid_train_du": tau_scores,
        "gate_info": gate_info,
        "probe1a_gsm8k": probe1a_gsm,
        "probe1a_math500": probe1a_math,
        "arc_results": _strip(arc_results),
        "gsm8k_gen_results": _strip(gsm_gen_results),
        "math500_gen_results": _strip(math_gen_results),
        "contrasts": contrasts,
    }


def render_report_md(summary: dict[str, Any]) -> str:
    lines = ["# Direction 1 Feasibility Report: Curvature-Calibrated Latent Trust-Region Stepping (FTRS)", ""]
    lines.append(f"- **Elapsed**: `{summary['elapsed_sec']:.1f}s` on GPU `{summary['gpu_index']}` ({summary['gpu_name']})")
    lines.append("")

    for mkey, md in summary["models"].items():
        lines.append(f"## Model: `{mkey}` (train_loops={md['train_loops']}, selected tau_fb={md['selected_tau_fb']})")
        lines.append(
            f"- **Exactness**: B=1 max_diff=`{md['verify']['max_abs_diff_b1']:.2e}`, "
            f"B=2 left-pad max_diff=`{md['verify']['max_abs_diff_b2_leftpad']:.2e}`, "
            f"4-tok decode exact=`{md['verify']['decode_4tok_exact']}`"
        )
        for pkey, plabel in [("probe1a_gsm8k", "GSM8K-test"), ("probe1a_math500", "MATH-500")]:
            p = md[pkey]
            lines.append(
                f"- **Probe 1A ({plabel}, n={p['n_total']})**: "
                f"Spearman(kappa, alpha*)=`{p['spearman_kappa_astar_all']:+.3f}`, "
                f"Spearman(KL, alpha*)=`{p['spearman_kl_astar_all']:+.3f}`, "
                f"AUROC(FSF|A>0)=`{p['auroc_fsf_given_pos_slope']:.3f}`, "
                f"AUROC(harmful)=`{p['auroc_harmful_all']:.3f}`, "
                f"Prop3.4 mean err=`{p['prop34_mean_err']:.2e}`"
            )
            lm_lo = p["low_margin_low_kappa"]
            lm_hi = p["low_margin_high_kappa"]
            lines.append(
                f"  - Low-margin / Low-kappa (n={lm_lo.get('n',0)}): alpha*=`{lm_lo.get('mean_alpha_star',0):.2f}`, "
                f"dU native=`{lm_lo.get('du_native',0):+.4f}`, LoopCD=`{lm_lo.get('du_loopcd',0):+.4f}`, "
                f"FTRS-Zero=`{lm_lo.get('du_ftrs_zero',0):+.4f}`, FTRS-Gate=`{lm_lo.get('du_ftrs_gate',0):+.4f}`"
            )
            lines.append(
                f"  - Low-margin / High-kappa (n={lm_hi.get('n',0)}): alpha*=`{lm_hi.get('mean_alpha_star',0):.2f}`, "
                f"dU native=`{lm_hi.get('du_native',0):+.4f}`, LoopCD=`{lm_hi.get('du_loopcd',0):+.4f}`, "
                f"FTRS-Zero=`{lm_hi.get('du_ftrs_zero',0):+.4f}`, FTRS-Gate=`{lm_hi.get('du_ftrs_gate',0):+.4f}`"
            )
        lines.append("")
        lines.append("### Probe 1A Curvature Quintiles (GSM8K-test)")
        lines.append("| Quintile | Median kappa | Mean alpha* | %>1 | %<1 | %Harmful | dU Native | dU LoopCD | dU Fixed0.5 | dU FTRS-Zero | dU FTRS-Gate |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for q in md["probe1a_gsm8k"]["quintiles"]:
            lines.append(
                f"| {q['quintile']} | {q['kappa_median']:.3e} | {q['mean_alpha_star']:.2f} | "
                f"{q['frac_astar_gt1']*100:.1f}% | {q['frac_astar_lt1']*100:.1f}% | {q['frac_harmful']*100:.1f}% | "
                f"{q['du_native']:+.4f} | {q['du_loopcd']:+.4f} | {q['du_fixed05']:+.4f} | "
                f"{q['du_ftrs_zero']:+.4f} | {q['du_ftrs_gate']:+.4f} |"
            )
        lines.append("")

        for btitle, bkey in [("ARC-Challenge", "arc_results"), ("GSM8K Free-Gen", "gsm8k_gen_results"), ("MATH-500 Free-Gen", "math500_gen_results")]:
            rmap = md.get(bkey, {})
            if not rmap:
                continue
            lines.append(f"### Probe 1B: {btitle}")
            lines.append("| Policy | R | Accuracy | Correct/Total | Comp% | Sec |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for pname, rv in rmap.items():
                comp = f"{rv['completion_rate']*100:.0f}%" if "completion_rate" in rv else "—"
                lines.append(
                    f"| `{pname}` | {rv['n_loops']} | {rv['accuracy']*100:.2f}% | "
                    f"{rv['n_correct']}/{rv['n_total']} | {comp} | {rv['elapsed_sec']:.1f} |"
                )
            lines.append("")

        lines.append("### Key Paired Contrasts (pp, 95% bootstrap CI)")
        for bname, cmap in md["contrasts"].items():
            lines.append(f"**{bname}**:")
            for ck, cv in cmap.items():
                lines.append(
                    f"- `{ck}`: `{cv['diff_pp']:+.2f} pp` CI `[{cv['ci95_pp'][0]:+.2f}, {cv['ci95_pp'][1]:+.2f}]` (+{cv['fixed']}/-{cv['broken']})"
                )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    seed = int(cfg.get("seed", 20261124))

    gpu_idx, gpu_lock, gpu_info = acquire_idle_gpu(cfg.get("preferred_gpu"))
    try:
        device = torch.device(f"cuda:{gpu_idx}")
        torch.cuda.set_device(device)
        print(f"[ftrs] Acquired idle GPU {gpu_idx}: {gpu_info.name} (mem={gpu_info.mem_used_mib}MiB, util={gpu_info.util_pct}%)")

        gsm_train = load_gsm8k_split(args.dataset_dir, "train", n_samples=int(cfg.get("n_probe1a_gsm_train", 100)), seed=seed)
        gsm_test = load_gsm8k_split(args.dataset_dir, "test", n_samples=int(cfg.get("n_probe1a_gsm_test", 100)), seed=seed + 1)
        math_test = load_math500_samples(args.dataset_dir, n_samples=int(cfg.get("n_probe1a_math_test", 100)), max_level=5, seed=seed + 2)
        math_gen = load_math500_samples(
            args.dataset_dir,
            n_samples=int(cfg.get("n_math_gen_think", 48)),
            max_level=int(cfg.get("math_gen_max_level", 3)),
            seed=seed + 3,
        )
        arc_samples = load_arc_challenge_samples(args.dataset_dir, n_samples=int(cfg.get("n_arc_eval", 200)), seed=seed + 4)

        plan_stages: list[dict[str, Any]] = [
            {"phase": "setup", "of": 1, "unit": "env"},
            {"phase": "download", "of": 1, "unit": "assets"},
        ]
        for mspec in cfg["models"]:
            mk = mspec["key"]
            plan_stages.append({"phase": f"{mk}_verify", "of": 1, "unit": "check"})
            plan_stages.append({"phase": f"{mk}_probe1a", "of": len(gsm_train) + len(gsm_test) + len(math_test), "unit": "seqs"})
            plan_stages.append({"phase": f"{mk}_arc", "of": 20, "unit": "conds"})
            plan_stages.append({"phase": f"{mk}_gsm_gen", "of": 14, "unit": "conds"})
            if mspec.get("is_thinking", False):
                plan_stages.append({"phase": f"{mk}_math_gen", "of": 9, "unit": "conds"})
        plan_stages.append({"phase": "report", "of": 1, "unit": "file"})
        orx.plan(plan_stages)

        run_obj = orx.progress()
        model_outputs: dict[str, Any] = {}
        for mspec in cfg["models"]:
            m_res = run_model_pilot(
                mspec,
                cfg=cfg,
                model_dir=Path(args.model_dir),
                gsm_train=gsm_train,
                gsm_test=gsm_test,
                math_test=math_test,
                math_gen_items=math_gen,
                arc_samples=arc_samples,
                device=device,
                run_obj=run_obj,
            )
            model_outputs[mspec["key"]] = m_res

        with run_obj.phase("report", total=1, unit="file") as prep:
            summary = {
                "idea": "new_dir1_ftrs",
                "gpu_index": gpu_idx,
                "gpu_name": gpu_info.name,
                "elapsed_sec": time.time() - t0,
                "config": cfg,
                "models": model_outputs,
            }
            (out_dir / "summary_ftrs.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
            report_md = render_report_md(summary)
            (out_dir / "report_ftrs.md").write_text(report_md)

            metrics: dict[str, float] = {}
            for mk, md in model_outputs.items():
                metrics[f"{mk}_spearman_k_astar"] = md["probe1a_gsm8k"]["spearman_kappa_astar_all"]
                metrics[f"{mk}_auroc_fsf"] = md["probe1a_gsm8k"]["auroc_fsf_given_pos_slope"]
                metrics[f"{mk}_arc_native_r4"] = md["arc_results"]["native_R4"]["accuracy"]
                metrics[f"{mk}_arc_native_r8"] = md["arc_results"]["native_R8"]["accuracy"]
                metrics[f"{mk}_arc_ftrs_uni_r8"] = md["arc_results"]["ftrs_unified_R8"]["accuracy"]
                metrics[f"{mk}_gsm_native_r4"] = md["gsm8k_gen_results"]["native_R4"]["accuracy"]
                metrics[f"{mk}_gsm_native_r8"] = md["gsm8k_gen_results"]["native_R8"]["accuracy"]
                metrics[f"{mk}_gsm_loopcd_r4"] = md["gsm8k_gen_results"]["loopcd_adapt_R4"]["accuracy"]
                metrics[f"{mk}_gsm_ftrs_uni_r4"] = md["gsm8k_gen_results"]["ftrs_unified_R4"]["accuracy"]
                metrics[f"{mk}_gsm_ftrs_uni_r8"] = md["gsm8k_gen_results"]["ftrs_unified_R8"]["accuracy"]
                metrics[f"{mk}_gsm_ftrs_gate_r8"] = md["gsm8k_gen_results"]["ftrs_gate_mfld_R8"]["accuracy"]
                if md["math500_gen_results"]:
                    metrics[f"{mk}_math_native_r4"] = md["math500_gen_results"]["native_R4"]["accuracy"]
                    metrics[f"{mk}_math_native_r8"] = md["math500_gen_results"]["native_R8"]["accuracy"]
                    metrics[f"{mk}_math_ftrs_uni_r8"] = md["math500_gen_results"]["ftrs_unified_R8"]["accuracy"]
            prep.step(1, **metrics)

            print("========== REPORT (report_ftrs.md) ==========")
            print(report_md)
            print("========== END REPORT ==========")
    finally:
        gpu_lock.release()


if __name__ == "__main__":
    main()
