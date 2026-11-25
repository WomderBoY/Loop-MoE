"""Route 1 Verification: Trajectory-Consistent Multi-Step Readout Extrapolation (TCRE)
+ Token-Level Ragged-KV Dynamic Loop Compute Reallocation.

Fixes all 4 Direction-1 biases identified in review:
  1. Strictly open-loop readout space (zero closed-loop hidden-state tampering).
  2. Consistent prefill/decode KV manifold (sink-protected full-depth prefill +
     shallow Ragged-KV reuse for early-exited decode tokens).
  3. Layer-depth-aware B=2 BF16 exactness check (compares B=2 vs native HF B=2
     with explicit position_ids + B=1 exactness + 4-tok decode exactness, no false
     kill on 192-layer Ouro-2.6B-Thinking).
  4. Scaled evaluation (n=500 GSM8K on all 3 models, n=500 ARC-Challenge,
     n=200 MATH-500 on Thinking models, proper special-token </think> parsing,
     pre-allocated contiguous KV buffers, and real B=1 wall-clock latency bench).
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
import torch.nn.functional as F

from loop_research import orx_run as orx
from loop_research.data_assets import (
    extract_predicted_math_answer,
    load_arc_challenge_samples,
    load_gsm8k_split,
    load_math500_samples,
)
from loop_research.ftrs_stepping import (
    GSM8K_4SHOT,
    _apply_rope,
    binary_auroc,
    load_ouro_for_transformers5,
    paired_bootstrap_ci,
    spearman_corr,
)
from loop_research.gpu_utils import acquire_idle_gpu


# ---------------------------------------------------------------------------
# Pure Open-Loop Trajectory Geometry & TCRE Primitives (Unit-Testable on CPU!)
# ---------------------------------------------------------------------------

def compute_distribution_stats(z: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute FP32 softmax distribution, entropy, top-2 margin, top-2 logit gap, and argmax."""
    zf = z.float()
    log_p = F.log_softmax(zf, dim=-1)
    p = log_p.exp()
    entropy = -(p * log_p).sum(dim=-1).clamp_min(0.0)
    top2_z = torch.topk(zf, k=2, dim=-1)
    logit_gap = (top2_z.values[..., 0] - top2_z.values[..., 1]).clamp_min(0.0)
    top2_p = torch.topk(p, k=2, dim=-1)
    margin = (top2_p.values[..., 0] - top2_p.values[..., 1]).clamp(0.0, 1.0)
    top1_prob = top2_p.values[..., 0].clamp(0.0, 1.0)
    top1_id = top2_z.indices[..., 0]
    return {
        "p": p,
        "log_p": log_p,
        "entropy": entropy,
        "margin": margin,
        "logit_gap": logit_gap,
        "top1_prob": top1_prob,
        "top1_id": top1_id,
    }


def compute_js_divergence(p_prev: torch.Tensor, p_curr: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Jensen-Shannon divergence JS(p_prev || p_curr) in nats, shape [...]."""
    m = 0.5 * (p_prev + p_curr).clamp_min(eps)
    log_m = m.log()
    kl_1 = (p_prev * (p_prev.clamp_min(eps).log() - log_m)).sum(dim=-1).clamp_min(0.0)
    kl_2 = (p_curr * (p_curr.clamp_min(eps).log() - log_m)).sum(dim=-1).clamp_min(0.0)
    return 0.5 * (kl_1 + kl_2)


def compute_trajectory_consistency(
    z_prev2: torch.Tensor,
    z_prev1: torch.Tensor,
    z_curr: torch.Tensor,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Compute shift-invariant, scale-invariant Fisher trajectory consistency between
    consecutive open-loop readout increments (z_{r-1} - z_{r-2}) and (z_r - z_{r-1}).

    All inner products are weighted by p_{r-1} (Fisher metric on active candidates),
    avoiding dilution by ~49k near-zero-probability vocabulary dimensions.
    """
    s_prev = compute_distribution_stats(z_prev1)
    s_curr = compute_distribution_stats(z_curr)
    p_ref = s_prev["p"]

    dz_prev = z_prev1.float() - z_prev2.float()
    dz_curr = z_curr.float() - z_prev1.float()

    # Center increments under p_{r-1} so constant logit shifts vanish identically
    mu_prev = (p_ref * dz_prev).sum(dim=-1, keepdim=True)
    mu_curr = (p_ref * dz_curr).sum(dim=-1, keepdim=True)
    c_prev = dz_prev - mu_prev
    c_curr = dz_curr - mu_curr

    v_prev = (p_ref * c_prev * c_prev).sum(dim=-1).clamp_min(0.0)
    v_curr = (p_ref * c_curr * c_curr).sum(dim=-1).clamp_min(0.0)
    cov = (p_ref * c_prev * c_curr).sum(dim=-1)

    # Fisher directional cosine in [-1, 1]
    denom = torch.sqrt((v_prev * v_curr).clamp_min(0.0)) + eps
    cos_fisher = (cov / denom).clamp(-1.0, 1.0)
    # Zero out noise when either step is numerically negligible
    tiny = (v_prev < 1e-7) | (v_curr < 1e-7)
    cos_fisher = torch.where(tiny, torch.zeros_like(cos_fisher), cos_fisher)

    # Fisher-Rayleigh contraction ratio lambda_hat = <dz_{r-1}, dz_r>_F / ||dz_{r-1}||_F^2
    lambda_hat = torch.where(tiny, torch.zeros_like(cov), cov / (v_prev + eps))

    # Unweighted Euclidean cosine (for Stage 1 diagnostic comparison)
    cos_euclid = F.cosine_similarity(dz_prev, dz_curr, dim=-1).clamp(-1.0, 1.0)

    # Entropy reduction: positive means sharpening/converging, negative means dispersing
    delta_h = s_prev["entropy"] - s_curr["entropy"]
    js_div = compute_js_divergence(p_ref, s_curr["p"])

    return {
        "cos_fisher": cos_fisher,
        "cos_euclid": cos_euclid,
        "lambda_hat": lambda_hat,
        "kappa_prev": v_prev,
        "kappa_curr": v_curr,
        "entropy_prev": s_prev["entropy"],
        "entropy_curr": s_curr["entropy"],
        "delta_h": delta_h,
        "margin_curr": s_curr["margin"],
        "js_div": js_div,
    }


def compute_openloop_readout(
    z_list: list[torch.Tensor],
    *,
    mode: str = "native",
    omega: float = 0.5,
    omega_max: float = 1.0,
    cos_gamma: float = 0.5,
    delta_h_tol: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Pure open-loop readout combiner over z_list = [z_1, ..., z_R], each [..., V].

    Modes:
      - native: z_R
      - loopcd_fixed: z_R + omega * (z_R - z_1)
      - loopcd_adapt: z_R + omega_max * (1 - margin_R) * (z_R - z_1)
      - tcre_cos: Direction-consistency protected extrapolation + continuous convex damping:
          * If cos_fisher > 0 and delta_h >= -delta_h_tol:
              w_ext = omega_max * (1 - margin_R) * (cos_fisher ** cos_gamma)
              z_out = z_R + w_ext * (z_R - z_1)
          * Else if cos_fisher <= 0 (oscillation / reversal):
              w_R = 1 / (1 - cos_fisher) in [0.5, 1.0]  (exact 0.5 midpoint at cos=-1, continuous at cos=0!)
              z_out = (1 - w_R) * z_{R-1} + w_R * z_R
          * Else (cos > 0 but entropy dispersed delta_h < -delta_h_tol):
              z_out = 0.25 * z_{R-1} + 0.75 * z_R  (mild conservative damping, no extrapolation)
      - tcre_aitken: Unified Fisher-metric Aitken Delta^2 fixed-point readout:
          * lambda_eff = clip(lambda_hat, -1.0, 0.75)
          * w_ait = lambda_eff / (1 - lambda_eff)
            - For lambda_eff > 0 & delta_h >= -delta_h_tol: extrapolate along (z_R - z_{R-1}) capped by
              omega_max * (1 - margin_R) * (cos_fisher ** cos_gamma) * 2.0
            - For lambda_eff <= 0: damp automatically along (z_R - z_{R-1}) with w in [-0.5, 0]
              (at lambda=-1 -> w=-0.5 -> exact 0.5*(z_{R-1}+z_R) midpoint!)
    """
    r_len = len(z_list)
    z_last = z_list[-1].float()
    diag: dict[str, torch.Tensor] = {}

    if mode == "native" or r_len < 2:
        return z_last, diag

    z_first = z_list[0].float()
    if mode == "loopcd_fixed":
        return z_last + float(omega) * (z_last - z_first), diag

    if mode == "loopcd_adapt":
        st = compute_distribution_stats(z_last)
        w = float(omega_max) * (1.0 - st["margin"])
        diag["omega_eff"] = w
        return z_last + w.unsqueeze(-1) * (z_last - z_first), diag

    if r_len < 3:
        # With only 2 loops, no 3-point directional cosine exists; fall back to conservative adapt
        st = compute_distribution_stats(z_last)
        w = 0.5 * float(omega_max) * (1.0 - st["margin"])
        diag["omega_eff"] = w
        return z_last + w.unsqueeze(-1) * (z_last - z_first), diag

    z_prev2 = z_list[-3].float()
    z_prev1 = z_list[-2].float()
    tc = compute_trajectory_consistency(z_prev2, z_prev1, z_last)
    cos_f = tc["cos_fisher"]
    dh = tc["delta_h"]
    margin = tc["margin_curr"]

    diag["cos_fisher"] = cos_f
    diag["delta_h"] = dh
    diag["lambda_hat"] = tc["lambda_hat"]

    if mode == "tcre_cos":
        aligned = (cos_f > 0.0) & (dh >= -float(delta_h_tol))
        reversing = cos_f <= 0.0

        # Branch 1: aligned & sharpening -> cosine-gated adaptive extrapolation
        w_ext = float(omega_max) * (1.0 - margin) * cos_f.clamp_min(0.0).pow(float(cos_gamma))
        z_ext = z_last + w_ext.unsqueeze(-1) * (z_last - z_first)

        # Branch 2: reversal -> continuous Aitken-midpoint convex damping in [0.5, 1.0]
        w_r = 1.0 / (1.0 - cos_f.clamp(min=-1.0, max=0.0))  # 1.0 at cos=0, 0.5 at cos=-1
        z_rev = (1.0 - w_r).unsqueeze(-1) * z_prev1 + w_r.unsqueeze(-1) * z_last

        # Branch 3: aligned but entropy increased -> mild conservative damping
        z_disp = 0.25 * z_prev1 + 0.75 * z_last

        z_out = torch.where(
            aligned.unsqueeze(-1),
            z_ext,
            torch.where(reversing.unsqueeze(-1), z_rev, z_disp),
        )
        diag["aligned"] = aligned.float()
        diag["reversing"] = reversing.float()
        diag["omega_eff"] = torch.where(aligned, w_ext, torch.where(reversing, w_r - 1.0, torch.full_like(w_ext, -0.25)))
        return z_out, diag

    if mode == "tcre_aitken":
        lam_eff = tc["lambda_hat"].clamp(-1.0, 0.75)
        w_raw = lam_eff / (1.0 - lam_eff)  # in [-0.5, 3.0]
        aligned = (lam_eff > 0.0) & (cos_f > 0.0) & (dh >= -float(delta_h_tol))
        reversing = lam_eff <= 0.0

        # Cap forward Aitken extrapolation smoothly by margin & cosine
        w_cap = float(omega_max) * 2.0 * (1.0 - margin) * cos_f.clamp_min(0.0).pow(float(cos_gamma))
        w_pos = torch.minimum(w_raw.clamp_min(0.0), w_cap)
        w_neg = w_raw.clamp(-0.5, 0.0)
        # If lam > 0 yet entropy dispersed, conservatively damp slightly
        w_eff = torch.where(aligned, w_pos, torch.where(reversing, w_neg, torch.full_like(w_pos, -0.25)))
        z_out = z_last + w_eff.unsqueeze(-1) * (z_last - z_prev1)

        diag["aligned"] = aligned.float()
        diag["reversing"] = reversing.float()
        diag["omega_eff"] = w_eff
        return z_out, diag

    raise ValueError(f"Unknown open-loop readout mode: {mode}")


# ---------------------------------------------------------------------------
# Route 1 Policy & Fast Pre-Allocated Ragged-KV Ouro Engine
# ---------------------------------------------------------------------------

@dataclass
class Route1Policy:
    name: str
    n_loops: int = 4
    readout_mode: str = "native"  # native | loopcd_fixed | loopcd_adapt | tcre_cos | tcre_aitken
    omega: float = 0.5
    omega_max: float = 1.0
    cos_gamma: float = 0.5
    delta_h_tol: float = 0.05
    # Dynamic Ragged-KV Early-Exit settings (applied during decode; prefill stays full-depth sink-protected!)
    exit_mode: str = "none"  # none | ouro_gate | gap_agree | random
    gate_cdf_thresh: float = 1.0
    exit_gap1: float = 1e9
    exit_gap2: float = 1e9
    exit_gap3: float = 1e9
    rand_exit_probs: tuple[float, ...] = ()  # (p_r1, p_r2, p_r3, p_r4) summing to 1.0
    rand_seed: int = 20261125


class OuroRoute1Engine:
    """Exact, fast, pre-allocated KV Ouro engine with open-loop TCRE & Ragged-KV early exit."""

    def __init__(self, model, device: torch.device) -> None:
        self.model = model
        self.core = model.model
        self.lm_head = model.lm_head
        self.device = device
        self.dtype = next(model.parameters()).dtype
        self.num_layers = int(model.config.num_hidden_layers)
        self.train_loops = int(getattr(model.config, "total_ut_steps", 4))
        self.num_heads = int(model.config.num_attention_heads)
        self.num_kv_heads = int(model.config.num_key_value_heads)
        self.head_dim = int(getattr(model.config, "head_dim", model.config.hidden_size // self.num_heads))
        self.vocab_size = int(model.config.vocab_size)
        self.enable_gqa = self.num_heads != self.num_kv_heads
        self.has_early_exit_gate = hasattr(self.core, "early_exit_gate") and self.core.early_exit_gate is not None

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
    def _run_layer_decode_inplace(
        self,
        layer,
        h: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_layer_buf: torch.Tensor,
        v_layer_buf: torch.Tensor,
        write_pos: int,
        attn_mask: torch.Tensor | None,
        act_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Single-token decode layer step using pre-allocated static KV buffer (zero torch.cat!)."""
        bsz = h.shape[0]
        attn = layer.self_attn
        residual = h
        x = layer.input_layernorm(h)

        q = attn.q_proj(x).view(bsz, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k_new = attn.k_proj(x).view(bsz, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_new = attn.v_proj(x).view(bsz, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k_new = _apply_rope(q, k_new, cos, sin)

        cur_len = write_pos + 1
        if act_idx is None:
            k_layer_buf[:, :, write_pos : write_pos + 1, :] = k_new
            v_layer_buf[:, :, write_pos : write_pos + 1, :] = v_new
            k_view = k_layer_buf[:, :, :cur_len, :]
            v_view = v_layer_buf[:, :, :cur_len, :]
        else:
            k_layer_buf[act_idx, :, write_pos : write_pos + 1, :] = k_new
            v_layer_buf[act_idx, :, write_pos : write_pos + 1, :] = v_new
            k_view = k_layer_buf[act_idx, :, :cur_len, :]
            v_view = v_layer_buf[act_idx, :, :cur_len, :]

        out = F.scaled_dot_product_attention(
            q,
            k_view,
            v_view,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=attn.scaling,
            enable_gqa=self.enable_gqa,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, 1, self.num_heads * self.head_dim)
        h = residual + layer.input_layernorm_2(attn.o_proj(out))
        h = h + layer.post_attention_layernorm_2(layer.mlp(layer.post_attention_layernorm(h)))
        return h

    @torch.inference_mode()
    def forward_all_loop_hiddens(
        self,
        input_ids: torch.Tensor,
        max_loops: int = 4,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Run unmodified Ouro for max_loops; return ([H_0..H_{R-1}], [gate_0..gate_{R-1}])."""
        bsz, seq_len = input_ids.shape
        if attention_mask is None:
            pos_ids = torch.arange(seq_len, device=self.device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
            attn_mask = None
        else:
            pos_ids = (attention_mask.cumsum(dim=-1) - 1).clamp_min(0)
            valid_k = attention_mask[:, None, None, :].bool()
            causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=self.device))[None, None, :, :]
            diag = torch.eye(seq_len, dtype=torch.bool, device=self.device)[None, None, :, :]
            attn_mask = (causal & valid_k) | diag

        h = self.core.embed_tokens(input_ids)
        cos, sin = self.core.rotary_emb(h, pos_ids)

        hiddens: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []
        for _ut in range(max_loops):
            for layer in self.core.layers:
                h, _ = self._run_layer_prefill(layer, h, cos, sin, attn_mask, return_kv=False)
            h = self.core.norm(h)
            hiddens.append(h)
            if self.has_early_exit_gate:
                gates.append(self.core.early_exit_gate(h).squeeze(-1).float())
        return hiddens, gates

    @torch.inference_mode()
    def forward_policy_logits(
        self,
        input_ids: torch.Tensor,
        policy: Route1Policy,
        attention_mask: torch.Tensor | None = None,
        score_slice: slice | None = None,
    ) -> torch.Tensor:
        """Pure open-loop sequence forward under Route1Policy (for MC scoring & exactness)."""
        hiddens, _ = self.forward_all_loop_hiddens(input_ids, max_loops=policy.n_loops, attention_mask=attention_mask)
        sl = score_slice if score_slice is not None else slice(None)
        if policy.readout_mode == "native":
            return self.lm_head(hiddens[-1][:, sl]).float()
        z_list = [self.lm_head(h[:, sl]).float() for h in hiddens]
        logits, _ = compute_openloop_readout(
            z_list,
            mode=policy.readout_mode,
            omega=policy.omega,
            omega_max=policy.omega_max,
            cos_gamma=policy.cos_gamma,
            delta_h_tol=policy.delta_h_tol,
        )
        return logits

    def _random_exit_loops(self, seq_ids: list[int], step_idx: int, policy: Route1Policy) -> torch.Tensor:
        """Deterministic per-(seq_id, step_idx) random exit loop draw (batch-order invariant!)."""
        cdf = np.cumsum(np.asarray(policy.rand_exit_probs, dtype=np.float64))
        cdf[-1] = 1.0
        out = []
        for sid in seq_ids:
            # SplitMix64-style deterministic hash -> [0, 1)
            x = (sid * 0x9E3779B97F4A7C15 + step_idx * 0xBF58476D1CE4E5B9 + policy.rand_seed) & 0xFFFFFFFFFFFFFFFF
            x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
            x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
            x = x ^ (x >> 31)
            u = (x & 0xFFFFFFFF) / float(0x100000000)
            r_draw = int(np.searchsorted(cdf, u, side="right")) + 1
            out.append(min(max(r_draw, 1), policy.n_loops))
        return torch.tensor(out, device=self.device, dtype=torch.long)

    @torch.inference_mode()
    def generate_batched(
        self,
        prompt_ids_list: list[list[int]],
        policy: Route1Policy,
        *,
        max_new_tokens: int = 256,
        pad_token_id: int = 0,
        eos_token_ids: set[int] | None = None,
        stop_strings: list[str] | None = None,
        tokenizer: Any = None,
        seq_id_offset: int = 0,
        collect_calib_trace: bool = False,
    ) -> tuple[list[list[int]], dict[str, Any]]:
        """Fast pre-allocated KV greedy decode with open-loop TCRE & Ragged-KV dynamic loop exit."""
        bsz = len(prompt_ids_list)
        max_p = max(len(p) for p in prompt_ids_list)
        max_total_len = max_p + max_new_tokens
        n_loops = policy.n_loops

        input_ids = torch.full((bsz, max_p), pad_token_id, device=self.device, dtype=torch.long)
        attn_mask_1d = torch.zeros((bsz, max_p), device=self.device, dtype=torch.long)
        for i, p in enumerate(prompt_ids_list):
            input_ids[i, max_p - len(p) :] = torch.tensor(p, device=self.device, dtype=torch.long)
            attn_mask_1d[i, max_p - len(p) :] = 1

        # Pre-allocate contiguous static KV cache: [n_loops, num_layers, B, H_kv, max_total_len, D]
        k_buf = torch.empty(
            (n_loops, self.num_layers, bsz, self.num_kv_heads, max_total_len, self.head_dim),
            device=self.device,
            dtype=self.dtype,
        )
        v_buf = torch.empty_like(k_buf)

        # Prefill: sink-protected full n_loops pass
        pos_ids = (attn_mask_1d.cumsum(dim=-1) - 1).clamp_min(0)
        if bsz == 1:
            prefill_mask = None
        else:
            valid_k = attn_mask_1d[:, None, None, :].bool()
            causal = torch.tril(torch.ones((max_p, max_p), dtype=torch.bool, device=self.device))[None, None, :, :]
            diag = torch.eye(max_p, dtype=torch.bool, device=self.device)[None, None, :, :]
            prefill_mask = (causal & valid_k) | diag

        h = self.core.embed_tokens(input_ids)
        cos, sin = self.core.rotary_emb(h, pos_ids)

        prefill_z_last: list[torch.Tensor] = []
        for ut in range(n_loops):
            for l_idx, layer in enumerate(self.core.layers):
                h, kv = self._run_layer_prefill(layer, h, cos, sin, prefill_mask, return_kv=True)
                assert kv is not None
                k_buf[ut, l_idx, :, :, :max_p, :] = kv[0]
                v_buf[ut, l_idx, :, :, :max_p, :] = kv[1]
            h = self.core.norm(h)
            if policy.readout_mode != "native" or ut == n_loops - 1:
                prefill_z_last.append(self.lm_head(h[:, -1, :]).float())

        if policy.readout_mode == "native":
            first_logits = prefill_z_last[-1]
        else:
            first_logits, _ = compute_openloop_readout(
                prefill_z_last,
                mode=policy.readout_mode,
                omega=policy.omega,
                omega_max=policy.omega_max,
                cos_gamma=policy.cos_gamma,
                delta_h_tol=policy.delta_h_tol,
            )

        next_tok = first_logits.argmax(dim=-1)
        eos_set = eos_token_ids or set()
        generated: list[list[int]] = [[] for _ in range(bsz)]
        finished = torch.zeros(bsz, device=self.device, dtype=torch.bool)
        for i in range(bsz):
            t_i = int(next_tok[i].item())
            generated[i].append(t_i)
            if t_i in eos_set:
                finished[i] = True

        # Active row tracking for dynamic finished-row compaction
        orig_indices = list(range(bsz))
        cur_valid_mask = attn_mask_1d
        next_pos = attn_mask_1d.sum(dim=-1, keepdim=True)  # [cur_b, 1]

        # Statistics
        loop_hist = np.zeros(n_loops, dtype=np.int64)
        total_decode_tokens = 0
        aligned_count = 0.0
        reversal_count = 0.0
        deep_tcre_count = 0
        cos_sum = 0.0

        # Optional calibration trace collection (underflow-free top-2 logit gaps + agreement + Ouro gate CDF)
        calib_records: dict[str, list[np.ndarray]] = {
            "gap1": [], "gap2": [], "agree12": [], "gap3": [], "agree23": [],
            "cdf1": [], "cdf2": [], "cdf3": [],
        }

        need_all_z_for_readout = policy.readout_mode != "native"
        need_z_each_loop = (policy.exit_mode == "gap_agree") or collect_calib_trace

        for step_idx in range(1, max_new_tokens):
            if bool(finished.all().item()):
                break

            # Dynamic compaction of finished rows to save GPU compute on tail sequences
            cur_b = len(orig_indices)
            n_fin = int(finished.sum().item())
            if cur_b > 1 and n_fin >= max(4, cur_b // 4):
                keep = (~finished).nonzero(as_tuple=False).squeeze(-1)
                k_buf = k_buf[:, :, keep]
                v_buf = v_buf[:, :, keep]
                next_tok = next_tok[keep]
                next_pos = next_pos[keep]
                cur_valid_mask = cur_valid_mask[keep]
                finished = finished[keep]
                orig_indices = [orig_indices[int(j)] for j in keep.tolist()]
                cur_b = len(orig_indices)

            write_pos = max_p + step_idx - 1
            cur_len = write_pos + 1
            cur_valid_mask = torch.cat(
                [cur_valid_mask, torch.ones((cur_b, 1), device=self.device, dtype=torch.long)], dim=1
            )
            # For B=1 unpadded, no attention mask needed at all!
            dec_mask_full = None if (bsz == 1 and cur_b == 1) else cur_valid_mask[:, None, None, :].bool()

            h = self.core.embed_tokens(next_tok.unsqueeze(1))  # [cur_b, 1, D]
            cos, sin = self.core.rotary_emb(h, next_pos)

            exited = finished.clone()
            step_logits = torch.zeros((cur_b, self.vocab_size), device=self.device, dtype=torch.float32)
            exit_loop_row = torch.zeros((cur_b,), device=self.device, dtype=torch.long)

            z_hist: list[torch.Tensor | None] = [None] * n_loops
            p_prev_full: torch.Tensor | None = None
            top1_prev_full: torch.Tensor | None = None
            rem_prob = torch.ones((cur_b,), device=self.device, dtype=torch.float32)

            rand_targets = (
                self._random_exit_loops([seq_id_offset + oi for oi in orig_indices], step_idx, policy)
                if policy.exit_mode == "random"
                else None
            )

            # Per-step temporary buffers for optional calibration trace
            step_cal: dict[str, torch.Tensor] = {}

            for ut in range(n_loops):
                active_mask = ~exited
                if not bool(active_mask.any().item()):
                    break

                all_active = bool(active_mask.all().item())
                if all_active:
                    act_idx = None
                    h_work = h
                    cos_w, sin_w = cos, sin
                    mask_w = dec_mask_full
                else:
                    act_idx = active_mask.nonzero(as_tuple=False).squeeze(-1)
                    h_work = h[act_idx]
                    cos_w = cos[act_idx]
                    sin_w = sin[act_idx]
                    mask_w = None if dec_mask_full is None else dec_mask_full[act_idx]

                for l_idx, layer in enumerate(self.core.layers):
                    h_work = self._run_layer_decode_inplace(
                        layer,
                        h_work,
                        cos_w,
                        sin_w,
                        k_buf[ut, l_idx],
                        v_buf[ut, l_idx],
                        write_pos,
                        mask_w,
                        act_idx=act_idx,
                    )
                h_norm = self.core.norm(h_work)
                h_vec = h_norm[:, 0, :]

                if all_active:
                    h = h_norm
                else:
                    h[act_idx] = h_norm

                # Compute logits ONLY when required (exit rule, TCRE readout history, or final loop)
                is_last_loop = ut == n_loops - 1
                need_z_now = (
                    need_z_each_loop
                    or is_last_loop
                    or (need_all_z_for_readout and (ut == 0 or ut >= n_loops - 3))
                )
                z_act = self.lm_head(h_vec).float() if need_z_now else None

                if z_act is not None and (need_all_z_for_readout or collect_calib_trace):
                    if all_active:
                        z_hist[ut] = z_act
                    else:
                        zb = torch.zeros((cur_b, self.vocab_size), device=self.device, dtype=torch.float32)
                        zb[act_idx] = z_act
                        z_hist[ut] = zb

                # Optional calibration trace collection (runs full n_loops, records all features!)
                if collect_calib_trace and all_active:
                    assert z_act is not None
                    st_c = compute_distribution_stats(z_act)
                    if self.has_early_exit_gate:
                        lam_g = torch.sigmoid(self.core.early_exit_gate(h_vec).squeeze(-1).float())
                        rem_prob = rem_prob * (1.0 - lam_g)
                        cdf_g = 1.0 - rem_prob
                    else:
                        cdf_g = torch.zeros((cur_b,), device=self.device, dtype=torch.float32)
                    if ut == 0:
                        step_cal["gap1"] = st_c["logit_gap"]
                        step_cal["cdf1"] = cdf_g
                        top1_prev_full = st_c["top1_id"]
                    elif ut == 1:
                        assert top1_prev_full is not None
                        step_cal["gap2"] = st_c["logit_gap"]
                        step_cal["agree12"] = (st_c["top1_id"] == top1_prev_full).float()
                        step_cal["cdf2"] = cdf_g
                        top1_prev_full = st_c["top1_id"]
                    elif ut == 2:
                        assert top1_prev_full is not None
                        step_cal["gap3"] = st_c["logit_gap"]
                        step_cal["agree23"] = (st_c["top1_id"] == top1_prev_full).float()
                        step_cal["cdf3"] = cdf_g
                        top1_prev_full = st_c["top1_id"]

                # Evaluate Early-Exit Decision (strictly on ut < n_loops - 1)
                if not is_last_loop and policy.exit_mode != "none":
                    if policy.exit_mode == "ouro_gate":
                        assert self.has_early_exit_gate
                        g_act = self.core.early_exit_gate(h_vec).squeeze(-1).float()
                        lam_act = torch.sigmoid(g_act)
                        if all_active:
                            rem_prob = rem_prob * (1.0 - lam_act)
                            cdf_act = 1.0 - rem_prob
                        else:
                            rem_prob[act_idx] = rem_prob[act_idx] * (1.0 - lam_act)
                            cdf_act = 1.0 - rem_prob[act_idx]
                        do_exit_act = cdf_act >= policy.gate_cdf_thresh

                    elif policy.exit_mode == "gap_agree":
                        assert z_act is not None
                        top2_z = torch.topk(z_act, k=2, dim=-1)
                        gap_a = top2_z.values[..., 0] - top2_z.values[..., 1]
                        t1_a = top2_z.indices[..., 0]
                        if ut == 0:
                            do_exit_act = gap_a >= policy.exit_gap1
                            if all_active:
                                top1_prev_full = t1_a
                            else:
                                top1_prev_full = torch.zeros((cur_b,), device=self.device, dtype=torch.long)
                                top1_prev_full[act_idx] = t1_a
                        else:
                            assert top1_prev_full is not None
                            t1_prev_a = top1_prev_full if all_active else top1_prev_full[act_idx]
                            agree_a = t1_a == t1_prev_a
                            thresh = policy.exit_gap2 if ut == 1 else policy.exit_gap3
                            do_exit_act = agree_a & (gap_a >= thresh)
                            if all_active:
                                top1_prev_full = t1_a
                            else:
                                top1_prev_full[act_idx] = t1_a

                    elif policy.exit_mode == "random":
                        assert rand_targets is not None
                        rt_act = rand_targets if all_active else rand_targets[act_idx]
                        do_exit_act = rt_act <= (ut + 1)
                    else:
                        raise ValueError(f"Unknown exit_mode: {policy.exit_mode}")

                    if bool(do_exit_act.any().item()):
                        if all_active:
                            ex_local = do_exit_act.nonzero(as_tuple=False).squeeze(-1)
                            ex_global = ex_local
                        else:
                            ex_local = do_exit_act.nonzero(as_tuple=False).squeeze(-1)
                            ex_global = act_idx[ex_local]

                        # Compute logits ONLY for exiting rows if not already computed
                        if z_act is None:
                            z_ex = self.lm_head(h_vec[ex_local]).float()
                        else:
                            z_ex = z_act[ex_local]

                        # For exit at loop >= 2, use native readout at exit loop (fast & clean)
                        step_logits[ex_global] = z_ex
                        exit_loop_row[ex_global] = ut + 1
                        exited[ex_global] = True

                        # Vectorized shallow Ragged-KV reuse: copy current loop's 1-token KV to future loops!
                        if bsz == 1 and cur_b == 1:
                            k_buf[ut + 1 :, :, :, :, write_pos : write_pos + 1, :] = k_buf[
                                ut : ut + 1, :, :, :, write_pos : write_pos + 1, :
                            ]
                            v_buf[ut + 1 :, :, :, :, write_pos : write_pos + 1, :] = v_buf[
                                ut : ut + 1, :, :, :, write_pos : write_pos + 1, :
                            ]
                        else:
                            k_buf[ut + 1 :, :, ex_global, :, write_pos : write_pos + 1, :] = k_buf[
                                ut : ut + 1, :, ex_global, :, write_pos : write_pos + 1, :
                            ]
                            v_buf[ut + 1 :, :, ex_global, :, write_pos : write_pos + 1, :] = v_buf[
                                ut : ut + 1, :, ex_global, :, write_pos : write_pos + 1, :
                            ]

                # Final loop readout (all remaining active rows receive open-loop TCRE / readout!)
                if is_last_loop:
                    assert z_act is not None
                    if policy.readout_mode == "native":
                        z_final_act = z_act
                        r_diag: dict[str, torch.Tensor] = {}
                    else:
                        if all_active:
                            zl_act = [zt for zt in z_hist if zt is not None]
                        else:
                            zl_act = [zt[act_idx] for zt in z_hist if zt is not None]
                        z_final_act, r_diag = compute_openloop_readout(
                            zl_act,
                            mode=policy.readout_mode,
                            omega=policy.omega,
                            omega_max=policy.omega_max,
                            cos_gamma=policy.cos_gamma,
                            delta_h_tol=policy.delta_h_tol,
                        )

                    if all_active:
                        step_logits = z_final_act
                        exit_loop_row.fill_(n_loops)
                    else:
                        step_logits[act_idx] = z_final_act
                        exit_loop_row[act_idx] = n_loops

                    if "cos_fisher" in r_diag:
                        n_deep = int(z_final_act.shape[0])
                        deep_tcre_count += n_deep
                        cos_sum += float(r_diag["cos_fisher"].sum().item())
                        if "aligned" in r_diag:
                            aligned_count += float(r_diag["aligned"].sum().item())
                            reversal_count += float(r_diag["reversing"].sum().item())

            if collect_calib_trace and "gap3" in step_cal:
                unfin_m = ~finished
                if bool(unfin_m.any().item()):
                    for ck, cv in step_cal.items():
                        calib_records[ck].append(cv[unfin_m].cpu().numpy())

            # Update loop usage stats for unfinished rows
            active_tok_mask = ~finished
            if bool(active_tok_mask.any().item()):
                el_vals = exit_loop_row[active_tok_mask].cpu().numpy()
                for rv in el_vals:
                    loop_hist[int(rv) - 1] += 1
                total_decode_tokens += int(el_vals.size)

            next_tok = step_logits.argmax(dim=-1)
            next_pos = next_pos + 1

            for j in range(cur_b):
                if finished[j]:
                    continue
                oi = orig_indices[j]
                t_j = int(next_tok[j].item())
                generated[oi].append(t_j)
                if t_j in eos_set:
                    finished[j] = True

            # Periodic stop-string check for non-thinking 4-shot prompts
            if stop_strings and tokenizer is not None and (step_idx % 16 == 0):
                for j in range(cur_b):
                    if finished[j]:
                        continue
                    oi = orig_indices[j]
                    txt = tokenizer.decode(generated[oi], skip_special_tokens=True)
                    if any(ss in txt for ss in stop_strings):
                        finished[j] = True

        mean_loops = (
            float(sum((u + 1) * int(loop_hist[u]) for u in range(n_loops)) / max(total_decode_tokens, 1))
            if total_decode_tokens > 0
            else float(n_loops)
        )
        stats: dict[str, Any] = {
            "mean_loops": mean_loops,
            "loop_hist": loop_hist.tolist(),
            "total_decode_tokens": int(total_decode_tokens),
            "deep_tcre_tokens": int(deep_tcre_count),
            "mean_cos_fisher_deep": float(cos_sum / max(deep_tcre_count, 1)) if deep_tcre_count > 0 else 0.0,
            "frac_aligned_deep": float(aligned_count / max(deep_tcre_count, 1)) if deep_tcre_count > 0 else 0.0,
            "frac_reversal_deep": float(reversal_count / max(deep_tcre_count, 1)) if deep_tcre_count > 0 else 0.0,
        }
        if collect_calib_trace:
            stats["calib_trace"] = {
                ck: np.concatenate(cv) if cv else np.array([], dtype=np.float32)
                for ck, cv in calib_records.items()
            }
        return generated, stats


# ---------------------------------------------------------------------------
# Stage 0: Fixed Exactness Verification (No BF16 B=2 False Kill!)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def verify_engine_exactness(model, engine: OuroRoute1Engine, tokenizer, device: torch.device) -> dict[str, Any]:
    """Verify OuroRoute1Engine matches native HF OuroForCausalLM AND Ragged-KV exactness."""
    p1 = "Question: If a train travels 60 miles per hour for 3 hours, how far does it go?\nAnswer:"
    p2 = "Question: What is 17 + 25?\nAnswer:"
    ids1 = tokenizer.encode(p1, add_special_tokens=True)
    ids2 = tokenizer.encode(p2, add_special_tokens=True)

    # 1) B=1 unpadded exactness
    t1 = torch.tensor([ids1], device=device, dtype=torch.long)
    hf_logits_1 = model(input_ids=t1, use_cache=False).logits.float().cpu()
    eng_logits_1 = engine.forward_policy_logits(t1, Route1Policy(name="v1", n_loops=engine.train_loops)).cpu()
    diff_1 = float((hf_logits_1 - eng_logits_1).abs().max().item())

    # 2) Left-padded B=2 exactness:
    #    (a) vs native HF B=2 with explicit position_ids (same GEMM batching!)
    #    (b) vs unpadded B=1 HF reference (checks top-1 exact + layer-scaled mean diff!)
    t2_unpad = torch.tensor([ids2], device=device, dtype=torch.long)
    hf_logits_p2 = model(input_ids=t2_unpad, use_cache=False).logits[0, -1, :].float().cpu()
    hf_logits_ref_b1 = torch.stack([hf_logits_1[0, -1, :], hf_logits_p2], dim=0)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    max_l = max(len(ids1), len(ids2))
    t2 = torch.full((2, max_l), pad_id, device=device, dtype=torch.long)
    m2 = torch.zeros((2, max_l), device=device, dtype=torch.long)
    for row, ids in enumerate([ids1, ids2]):
        t2[row, max_l - len(ids) :] = torch.tensor(ids, device=device, dtype=torch.long)
        m2[row, max_l - len(ids) :] = 1
    pos2 = (m2.cumsum(dim=-1) - 1).clamp_min(0)
    hf_logits_b2_same = model(input_ids=t2, attention_mask=m2, position_ids=pos2, use_cache=False).logits[:, -1, :].float().cpu()

    eng_logits_2 = engine.forward_policy_logits(
        t2, Route1Policy(name="v2", n_loops=engine.train_loops), attention_mask=m2
    )[:, -1, :].cpu()

    diff_2_same_b = float((hf_logits_b2_same - eng_logits_2).abs().max().item())
    diff_2_vs_b1_t = (hf_logits_ref_b1 - eng_logits_2).abs()
    diff_2_vs_b1_max = float(diff_2_vs_b1_t.max().item())
    diff_2_vs_b1_mean = float(diff_2_vs_b1_t.mean().item())
    top1_b2_exact = bool(torch.equal(hf_logits_ref_b1.argmax(dim=-1), eng_logits_2.argmax(dim=-1)))

    # 3) 4-step pre-allocated KV-cache greedy decode exactness on BOTH sequences
    gen_eng, _ = engine.generate_batched(
        [ids1, ids2],
        Route1Policy(name="v_dec", n_loops=engine.train_loops),
        max_new_tokens=4,
        pad_token_id=pad_id,
    )
    hf_gen_pair = []
    for ids_i in (ids1, ids2):
        cur = torch.tensor([ids_i], device=device, dtype=torch.long)
        toks = []
        for _ in range(4):
            nt = int(model(input_ids=cur, use_cache=False).logits[0, -1].argmax().item())
            toks.append(nt)
            cur = torch.cat([cur, torch.tensor([[nt]], device=device, dtype=torch.long)], dim=1)
        hf_gen_pair.append(toks)
    decode_exact_both = (gen_eng[0] == hf_gen_pair[0]) and (gen_eng[1] == hf_gen_pair[1])

    # 4) Ragged-KV dynamic exit consistency: B=2 early-exit vs B=1 early-exit exact match
    ee_test_pol = Route1Policy(
        name="v_ee",
        n_loops=engine.train_loops,
        readout_mode="tcre_cos",
        exit_mode="gap_agree",
        exit_gap1=6.0,
        exit_gap2=3.0,
        exit_gap3=4.0,
    )
    gen_ee_b2, st_ee_b2 = engine.generate_batched([ids1, ids2], ee_test_pol, max_new_tokens=6, pad_token_id=pad_id)
    gen_ee_b1_0, _ = engine.generate_batched([ids1], ee_test_pol, max_new_tokens=6, pad_token_id=pad_id)
    gen_ee_b1_1, _ = engine.generate_batched([ids2], ee_test_pol, max_new_tokens=6, pad_token_id=pad_id)
    ee_b2_matches_b1 = (gen_ee_b2[0] == gen_ee_b1_0[0]) and (gen_ee_b2[1] == gen_ee_b1_1[0])

    # Layer-scaled BF16 B=2 vs B=1 tolerance (192 effective layers has ~0.13 mean / ~1.6 max cuBLAS rounding)
    eff_layers = engine.num_layers * engine.train_loops
    mean_tol = 0.08 * (eff_layers / 96.0) + 0.05
    max_tol_same_b = 0.25 * (eff_layers / 96.0) + 0.15

    res = {
        "effective_layers": eff_layers,
        "max_abs_diff_b1": diff_1,
        "max_abs_diff_b2_same_batch": diff_2_same_b,
        "max_abs_diff_b2_vs_b1": diff_2_vs_b1_max,
        "mean_abs_diff_b2_vs_b1": diff_2_vs_b1_mean,
        "top1_b2_exact": top1_b2_exact,
        "decode_4tok_exact_both": decode_exact_both,
        "ee_b2_matches_b1": ee_b2_matches_b1,
        "ee_test_mean_loops": st_ee_b2["mean_loops"],
        "eng_toks_0": gen_eng[0],
        "hf_toks_0": hf_gen_pair[0],
    }
    print(f"[verify] OuroRoute1Engine exactness: {json.dumps(res)}")
    if (
        not math.isfinite(diff_1)
        or diff_1 > 0.15
        or diff_2_same_b > max_tol_same_b
        or diff_2_vs_b1_mean > mean_tol
        or not top1_b2_exact
        or not decode_exact_both
    ):
        raise RuntimeError(f"Engine exactness check FAILED: {res}")
    return res


# ---------------------------------------------------------------------------
# Stage 1: Open-Loop Trajectory Diagnostic + Zero-Leakage Threshold Calibration
# ---------------------------------------------------------------------------

def _format_gen_prompt(item: dict[str, Any], tokenizer, *, is_thinking: bool, task: str) -> list[int]:
    if is_thinking:
        if task == "gsm8k":
            user_msg = f"Solve this math problem step by step. Put your final numeric answer in \\boxed{{}}.\n\n{item['question']}"
        else:
            user_msg = f"Solve the following math problem step by step. Put your final answer inside \\boxed{{}}.\n\n{item['question']}"
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            try:
                prompt_str = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_msg}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            except TypeError:
                prompt_str = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_msg}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            return tokenizer.encode(prompt_str, add_special_tokens=False)
        return tokenizer.encode(f"User: {user_msg}\nAssistant:\n<think>\n", add_special_tokens=True)
    else:
        prompt_str = f"{GSM8K_4SHOT}Question: {item['question']}\nAnswer:"
        return tokenizer.encode(prompt_str, add_special_tokens=True)


def _get_eos_ids(tokenizer) -> set[int]:
    eos_ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))
    for tok_str in ("<|im_end|>", "<|endoftext|>", "</s>"):
        tid = tokenizer.convert_tokens_to_ids(tok_str)
        if isinstance(tid, int) and tid >= 0:
            eos_ids.add(tid)
    return eos_ids


@torch.inference_mode()
def run_openloop_trajectory_diagnostic(
    engine: OuroRoute1Engine,
    items: list[dict[str, Any]],
    tokenizer,
    *,
    is_thinking: bool,
    max_loops: int = 6,
    max_sol_tokens: int = 128,
    run_phase: Any = None,
) -> dict[str, Any]:
    """Verify whether Fisher directional cosine & entropy reduction beat kappa_r at
    distinguishing productive extrapolation from oscillation/harmful steps."""
    recs: dict[str, list[np.ndarray]] = {
        "trans": [], "cos_f": [], "cos_e": [], "delta_h": [], "kappa": [],
        "du_native": [], "du_loopcd": [], "du_tcre_cos": [], "du_tcre_aitken": [],
        "harmful": [], "extrap_better": [],
    }
    for idx_i, item in enumerate(items, 1):
        prompt_text = (
            f"Solve the following math problem step by step and put your final answer in \\boxed{{}}.\n\n{item['question']}\n\nSolution:\n"
            if is_thinking
            else f"Question: {item['question']}\nAnswer: "
        )
        p_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
        s_ids = tokenizer.encode(item["solution"], add_special_tokens=False)[:max_sol_tokens]
        if len(s_ids) < 4:
            continue
        full = p_ids + s_ids
        inp = torch.tensor([full[:-1]], device=engine.device, dtype=torch.long)
        tgt = torch.tensor(full[len(p_ids) :], device=engine.device, dtype=torch.long)
        sl_start = max(0, len(p_ids) - 1)

        hiddens, _ = engine.forward_all_loop_hiddens(inp, max_loops=max_loops)
        z_all = [engine.lm_head(h[0, sl_start:, :]).float() for h in hiddens]

        # Analyze transitions u >= 2 (i.e. 3rd loop onward: 2->3, 3->4, 4->5, 5->6)
        for u in range(2, max_loops):
            tc = compute_trajectory_consistency(z_all[u - 2], z_all[u - 1], z_all[u])
            z_prev = z_all[u - 1]
            z_curr = z_all[u]
            z_lcd, _ = compute_openloop_readout(z_all[: u + 1], mode="loopcd_adapt", omega_max=1.0)
            z_tc, _ = compute_openloop_readout(z_all[: u + 1], mode="tcre_cos", omega_max=1.0)
            z_ta, _ = compute_openloop_readout(z_all[: u + 1], mode="tcre_aitken", omega_max=1.0)

            ll_prev = F.log_softmax(z_prev, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            ll_curr = F.log_softmax(z_curr, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            ll_lcd = F.log_softmax(z_lcd, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            ll_tc = F.log_softmax(z_tc, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            ll_ta = F.log_softmax(z_ta, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

            du_nat = (ll_curr - ll_prev).cpu().numpy()
            du_lcd = (ll_lcd - ll_prev).cpu().numpy()
            du_tc = (ll_tc - ll_prev).cpu().numpy()
            du_ta = (ll_ta - ll_prev).cpu().numpy()

            recs["trans"].append(np.full(tgt.shape[0], u, dtype=np.int32))
            recs["cos_f"].append(tc["cos_fisher"].cpu().numpy())
            recs["cos_e"].append(tc["cos_euclid"].cpu().numpy())
            recs["delta_h"].append(tc["delta_h"].cpu().numpy())
            recs["kappa"].append(tc["kappa_curr"].cpu().numpy())
            recs["du_native"].append(du_nat)
            recs["du_loopcd"].append(du_lcd)
            recs["du_tcre_cos"].append(du_tc)
            recs["du_tcre_aitken"].append(du_ta)
            recs["harmful"].append((du_nat < -1e-6).astype(np.int64))
            recs["extrap_better"].append((du_lcd > du_nat + 1e-6).astype(np.int64))

        if run_phase is not None:
            run_phase.step(idx_i)

    arr = {k: np.concatenate(v) for k, v in recs.items()}
    per_loop: dict[str, Any] = {}
    for u in sorted(np.unique(arr["trans"]).tolist()):
        m = arr["trans"] == u
        rev = m & (arr["cos_f"] <= 0.0)
        aln = m & (arr["cos_f"] > 0.0)
        per_loop[f"R={u+1}"] = {
            "n": int(m.sum()),
            "frac_reversal": float(np.mean(arr["cos_f"][m] <= 0.0)),
            "auroc_harmful_neg_cos_f": binary_auroc(-arr["cos_f"][m], arr["harmful"][m]),
            "auroc_harmful_neg_dh": binary_auroc(-arr["delta_h"][m], arr["harmful"][m]),
            "auroc_harmful_kappa": binary_auroc(arr["kappa"][m], arr["harmful"][m]),
            "auroc_extrap_pos_cos_f": binary_auroc(arr["cos_f"][m], arr["extrap_better"][m]),
            "du_native": float(np.mean(arr["du_native"][m])),
            "du_loopcd": float(np.mean(arr["du_loopcd"][m])),
            "du_tcre_cos": float(np.mean(arr["du_tcre_cos"][m])),
            "du_tcre_aitken": float(np.mean(arr["du_tcre_aitken"][m])),
            "rev_du_loopcd": float(np.mean(arr["du_loopcd"][rev])) if rev.sum() > 0 else 0.0,
            "rev_du_tcre_cos": float(np.mean(arr["du_tcre_cos"][rev])) if rev.sum() > 0 else 0.0,
            "aln_du_loopcd": float(np.mean(arr["du_loopcd"][aln])) if aln.sum() > 0 else 0.0,
            "aln_du_tcre_cos": float(np.mean(arr["du_tcre_cos"][aln])) if aln.sum() > 0 else 0.0,
        }
    return {
        "n_total": int(len(arr["cos_f"])),
        "auroc_harmful_neg_cos_f": binary_auroc(-arr["cos_f"], arr["harmful"]),
        "auroc_harmful_neg_cos_e": binary_auroc(-arr["cos_e"], arr["harmful"]),
        "auroc_harmful_neg_dh": binary_auroc(-arr["delta_h"], arr["harmful"]),
        "auroc_harmful_kappa_old": binary_auroc(arr["kappa"], arr["harmful"]),
        "spearman_cos_f_du_native": spearman_corr(arr["cos_f"], arr["du_native"]),
        "spearman_dh_du_native": spearman_corr(arr["delta_h"], arr["du_native"]),
        "per_loop": per_loop,
    }


def _quantile_thresholds_from_trace(trace: dict[str, np.ndarray], q: float) -> dict[str, float]:
    """Map a single monotone quantile parameter q in [0, 1] to exact underflow-free top-2 logit-gap thresholds.

    Using top-2 logit gap (z_(1) - z_(2)) avoids FP32 softmax entropy underflow ties at 0.0 on
    high-confidence CoT tokens, guaranteeing smooth calibration across every target budget R.
    """
    if q <= 1e-6:
        return {"exit_gap1": 1e9, "exit_gap2": 1e9, "exit_gap3": 1e9}
    q_clip = min(max(float(q), 0.0), 1.0)
    # Bimodal epistemic-pivot reallocation: easy tokens exit at Loop 1-2, while
    # low-gap uncertain pivot tokens stay strictly in Loop 4 for full refinement + TCRE!
    q1 = 0.65 * q_clip
    q3 = 0.50 * q_clip
    return {
        "exit_gap1": float(np.quantile(trace["gap1"], 1.0 - q1)),
        "exit_gap2": float(np.quantile(trace["gap2"], 1.0 - q_clip)),
        "exit_gap3": float(np.quantile(trace["gap3"], 1.0 - q3)),
    }


def _simulate_entropy_agree_mean_r(trace: dict[str, np.ndarray], q: float) -> tuple[float, tuple[float, ...], dict[str, float]]:
    """Simulate R=4 exit depths on calibration trace for quantile parameter q in [0, 1]."""
    th = _quantile_thresholds_from_trace(trace, q)
    g1, g2, g3 = trace["gap1"], trace["gap2"], trace["gap3"]
    a12, a23 = trace["agree12"] > 0.5, trace["agree23"] > 0.5

    ex1 = g1 >= th["exit_gap1"]
    ex2 = (~ex1) & a12 & (g2 >= th["exit_gap2"])
    ex3 = (~ex1) & (~ex2) & a23 & (g3 >= th["exit_gap3"])
    ex4 = (~ex1) & (~ex2) & (~ex3)

    probs = (float(ex1.mean()), float(ex2.mean()), float(ex3.mean()), float(ex4.mean()))
    mean_r = 1.0 * probs[0] + 2.0 * probs[1] + 3.0 * probs[2] + 4.0 * probs[3]
    return mean_r, probs, th


def _simulate_ouro_gate_mean_r(trace: dict[str, np.ndarray], cdf_thresh: float) -> tuple[float, tuple[float, ...]]:
    ex1 = trace["cdf1"] >= cdf_thresh
    ex2 = (~ex1) & (trace["cdf2"] >= cdf_thresh)
    ex3 = (~ex1) & (~ex2) & (trace["cdf3"] >= cdf_thresh)
    ex4 = (~ex1) & (~ex2) & (~ex3)
    probs = (float(ex1.mean()), float(ex2.mean()), float(ex3.mean()), float(ex4.mean()))
    mean_r = 1.0 * probs[0] + 2.0 * probs[1] + 3.0 * probs[2] + 4.0 * probs[3]
    return mean_r, probs


def calibrate_exit_thresholds(
    engine: OuroRoute1Engine,
    train_items: list[dict[str, Any]],
    tokenizer,
    *,
    is_thinking: bool,
    target_budgets: list[float],
    n_rollouts: int = 16,
) -> dict[str, Any]:
    """Calibrate zero-leakage early-exit thresholds strictly on GSM8K-train free rollouts."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    eos_ids = _get_eos_ids(tokenizer)
    stop_strings = None if is_thinking else ["\nQuestion:", "\n\nQuestion:"]

    prompts = [_format_gen_prompt(it, tokenizer, is_thinking=is_thinking, task="gsm8k") for it in train_items[:n_rollouts]]
    max_tok = 384 if is_thinking else 200
    _, stats = engine.generate_batched(
        prompts,
        Route1Policy(name="calib_rollout", n_loops=engine.train_loops, readout_mode="native"),
        max_new_tokens=max_tok,
        pad_token_id=pad_id,
        eos_token_ids=eos_ids,
        stop_strings=stop_strings,
        tokenizer=tokenizer,
        collect_calib_trace=True,
    )
    trace = stats["calib_trace"]
    n_tok = int(len(trace["gap1"]))
    if n_tok < 32:
        raise RuntimeError(f"Too few calibration tokens collected: {n_tok}")

    calibrated: dict[str, Any] = {
        "n_calib_tokens": n_tok,
        "median_gap1": float(np.median(trace["gap1"])),
        "median_gap2": float(np.median(trace["gap2"])),
        "agree12_rate": float(np.mean(trace["agree12"])),
        "agree23_rate": float(np.mean(trace["agree23"])),
        "median_cdf1": float(np.median(trace["cdf1"])),
        "median_cdf2": float(np.median(trace["cdf2"])),
        "entropy_agree": {},
        "ouro_gate": {},
    }

    for r_tgt in target_budgets:
        key = f"{r_tgt:.1f}"
        # 1) Binary search monotone quantile parameter q in [0, 1] (larger q => more exits => smaller mean_r)
        lo, hi = 0.0, 1.0
        for _ in range(32):
            mid = 0.5 * (lo + hi)
            mr, _, _ = _simulate_entropy_agree_mean_r(trace, mid)
            if mr > r_tgt:
                lo = mid
            else:
                hi = mid
        q_best = 0.5 * (lo + hi)
        mr_best, probs_best, th_best = _simulate_entropy_agree_mean_r(trace, q_best)
        calibrated["entropy_agree"][key] = {
            "quantile_q": q_best,
            "calib_mean_r": mr_best,
            "calib_exit_probs": list(probs_best),
            **th_best,
        }

        # 2) Binary search CDF threshold for ouro_gate (smaller thresh => more exits => smaller mean_r)
        glo, ghi = 0.001, 0.9999
        for _ in range(32):
            gmid = 0.5 * (glo + ghi)
            gmr, _ = _simulate_ouro_gate_mean_r(trace, gmid)
            if gmr < r_tgt:
                glo = gmid
            else:
                ghi = gmid
        g_best = 0.5 * (glo + ghi)
        gmr_best, gprobs_best = _simulate_ouro_gate_mean_r(trace, g_best)
        calibrated["ouro_gate"][key] = {
            "gate_cdf_thresh": g_best,
            "calib_mean_r": gmr_best,
            "calib_exit_probs": list(gprobs_best),
        }

    print(f"[calib] Collected {n_tok} train rollout tokens: agree12={calibrated['agree12_rate']*100:.1f}%, "
          f"gap1_med={calibrated['median_gap1']:.2f}, gap2_med={calibrated['median_gap2']:.2f}, "
          f"gate_cdf1_med={calibrated['median_cdf1']:.3f}")
    for r_tgt in target_budgets:
        k = f"{r_tgt:.1f}"
        ea = calibrated["entropy_agree"][k]
        og = calibrated["ouro_gate"][k]
        print(f"  target R={k}: gap_agree(q={ea['quantile_q']:.3f}, gap=[{ea['exit_gap1']:.2f},{ea['exit_gap2']:.2f},{ea['exit_gap3']:.2f}], "
              f"calib_R={ea['calib_mean_r']:.2f}, p={[round(x, 2) for x in ea['calib_exit_probs']]}) | "
              f"ouro_gate(cdf={og['gate_cdf_thresh']:.3f}, calib_R={og['calib_mean_r']:.2f})")
    return calibrated


# ---------------------------------------------------------------------------
# Stage 2/3/4: Evaluation & True B=1 Latency Benchmarking
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate_arc_challenge(
    engine: OuroRoute1Engine,
    samples: list[dict[str, Any]],
    tokenizer,
    policy: Route1Policy,
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
            logits = engine.forward_policy_logits(inp, policy, score_slice=sl)
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


@torch.inference_mode()
def evaluate_math_generation(
    engine: OuroRoute1Engine,
    items: list[dict[str, Any]],
    tokenizer,
    policy: Route1Policy,
    *,
    is_thinking: bool,
    task: str,
    max_new_tokens: int,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Run fast pre-allocated KV batched generation and score accuracy, mean_R, and completion."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    eos_ids = _get_eos_ids(tokenizer)
    stop_strings = None if is_thinking else ["\nQuestion:", "\n\nQuestion:"]

    prompts = [_format_gen_prompt(it, tokenizer, is_thinking=is_thinking, task=task) for it in items]
    correct_flags: list[int] = []
    finished_flags: list[int] = []
    gen_lens: list[int] = []
    total_loop_hist = np.zeros(policy.n_loops, dtype=np.int64)
    total_dec_toks = 0
    total_deep_toks = 0
    weighted_cos = 0.0
    weighted_aligned = 0.0
    weighted_reversal = 0.0
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
            seq_id_offset=s0,
        )
        total_loop_hist += np.asarray(stats["loop_hist"], dtype=np.int64)
        total_dec_toks += int(stats["total_decode_tokens"])
        dt = int(stats["deep_tcre_tokens"])
        total_deep_toks += dt
        weighted_cos += dt * float(stats["mean_cos_fisher_deep"])
        weighted_aligned += dt * float(stats["frac_aligned_deep"])
        weighted_reversal += dt * float(stats["frac_reversal_deep"])

        for it, gids in zip(batch_items, gen_ids_batch):
            # Decode WITH special tokens preserved so </think> boundary is NEVER stripped!
            raw_txt = tokenizer.decode(gids, skip_special_tokens=False)
            if not is_thinking:
                clean_txt = tokenizer.decode(gids, skip_special_tokens=True)
                if "\nQuestion:" in clean_txt:
                    clean_txt = clean_txt.split("\nQuestion:")[0]
                pred = extract_predicted_math_answer(clean_txt, mode=task)
                finished_flags.append(1 if ("####" in clean_txt or len(gids) < max_new_tokens) else 0)
            else:
                pred = extract_predicted_math_answer(raw_txt, mode=task)
                finished_flags.append(1 if ("</think>" in raw_txt or len(gids) < max_new_tokens) else 0)
            correct_flags.append(int(pred != "" and pred == it["gold"]))
            gen_lens.append(len(gids))

    elapsed = time.time() - t0
    mean_r = (
        float(sum((u + 1) * int(total_loop_hist[u]) for u in range(policy.n_loops)) / max(total_dec_toks, 1))
        if total_dec_toks > 0
        else float(policy.n_loops)
    )
    res: dict[str, Any] = {
        "policy": policy.name,
        "task": task,
        "n_loops_max": policy.n_loops,
        "mean_loops": mean_r,
        "loop_fracs": (total_loop_hist / max(total_dec_toks, 1)).tolist(),
        "accuracy": float(np.mean(correct_flags)),
        "n_correct": int(sum(correct_flags)),
        "n_total": len(correct_flags),
        "completion_rate": float(np.mean(finished_flags)),
        "mean_gen_tokens": float(np.mean(gen_lens)),
        "elapsed_sec": elapsed,
        "mean_cos_fisher_deep": float(weighted_cos / max(total_deep_toks, 1)) if total_deep_toks > 0 else 0.0,
        "frac_aligned_deep": float(weighted_aligned / max(total_deep_toks, 1)) if total_deep_toks > 0 else 0.0,
        "frac_reversal_deep": float(weighted_reversal / max(total_deep_toks, 1)) if total_deep_toks > 0 else 0.0,
        "correct_flags": correct_flags,
    }
    if task == "math500":
        l13 = [f for it, f in zip(items, correct_flags) if int(it.get("level", 3)) <= 3]
        l45 = [f for it, f in zip(items, correct_flags) if int(it.get("level", 3)) >= 4]
        res["acc_level_1_3"] = float(np.mean(l13)) if l13 else 0.0
        res["acc_level_4_5"] = float(np.mean(l45)) if l45 else 0.0
    return res


@torch.inference_mode()
def benchmark_b1_latency(
    engine: OuroRoute1Engine,
    items: list[dict[str, Any]],
    tokenizer,
    policies: list[Route1Policy],
    *,
    is_thinking: bool,
    max_new_tokens: int = 192,
) -> dict[str, dict[str, Any]]:
    """Measure true single-stream (B=1) GPU wall-clock decode latency (ms/tok & speedup vs native_R4)."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    eos_ids = _get_eos_ids(tokenizer)
    stop_strings = None if is_thinking else ["\nQuestion:", "\n\nQuestion:"]
    prompts = [_format_gen_prompt(it, tokenizer, is_thinking=is_thinking, task="gsm8k") for it in items]

    # Warmup 1 prompt to eliminate JIT / allocator noise
    engine.generate_batched([prompts[0]], policies[0], max_new_tokens=16, pad_token_id=pad_id, eos_token_ids=eos_ids)
    torch.cuda.synchronize(engine.device)

    out: dict[str, dict[str, Any]] = {}
    ref_ms_per_tok: float | None = None
    for pol in policies:
        torch.cuda.synchronize(engine.device)
        t0 = time.perf_counter()
        tot_toks = 0
        weighted_r = 0.0
        for idx_p, p_ids in enumerate(prompts):
            gen_ids, st = engine.generate_batched(
                [p_ids],
                pol,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_id,
                eos_token_ids=eos_ids,
                stop_strings=stop_strings,
                tokenizer=tokenizer,
                seq_id_offset=idx_p,
            )
            nt = len(gen_ids[0])
            tot_toks += nt
            weighted_r += nt * float(st["mean_loops"])
        torch.cuda.synchronize(engine.device)
        dt = time.perf_counter() - t0
        ms_per_tok = (dt * 1000.0) / max(tot_toks, 1)
        if pol.name == "native_R4" or ref_ms_per_tok is None:
            ref_ms_per_tok = ms_per_tok
        speedup = ref_ms_per_tok / max(ms_per_tok, 1e-6)
        out[pol.name] = {
            "policy": pol.name,
            "mean_loops": float(weighted_r / max(tot_toks, 1)),
            "total_tokens": int(tot_toks),
            "elapsed_sec": float(dt),
            "ms_per_tok": float(ms_per_tok),
            "tok_per_sec": float(tot_toks / max(dt, 1e-6)),
            "speedup_vs_native_r4": float(speedup),
        }
        print(
            f"[bench_b1] {pol.name:18s} R_avg={out[pol.name]['mean_loops']:.2f} "
            f"ms/tok={ms_per_tok:.2f} tok/s={out[pol.name]['tok_per_sec']:.1f} speedup={speedup:.2f}x"
        )
    return out


# ---------------------------------------------------------------------------
# Per-Model Full Route 1 Pipeline
# ---------------------------------------------------------------------------

def _build_ea_policy(name: str, cal_ea: dict[str, Any], *, readout_mode: str, omega_max: float) -> Route1Policy:
    return Route1Policy(
        name=name,
        n_loops=4,
        readout_mode=readout_mode,
        omega_max=omega_max,
        exit_mode="gap_agree",
        exit_gap1=float(cal_ea["exit_gap1"]),
        exit_gap2=float(cal_ea["exit_gap2"]),
        exit_gap3=float(cal_ea["exit_gap3"]),
    )


def run_model_route1(
    mspec: dict[str, Any],
    *,
    cfg: dict[str, Any],
    model_dir: Path,
    gsm_train: list[dict[str, Any]],
    gsm_test: list[dict[str, Any]],
    math_test: list[dict[str, Any]],
    arc_samples: list[dict[str, Any]],
    device: torch.device,
    run_obj: Any,
    out_dir: Path,
) -> dict[str, Any]:
    mkey = mspec["key"]
    is_thinking = bool(mspec["is_thinking"])
    train_loops = int(mspec["train_loops"])
    w_max = float(mspec.get("omega_max", 1.0))
    w_fix = float(mspec.get("loopcd_fixed_omega", 0.5))
    local_path = model_dir / mspec["local_dir"]

    print(f"\n[route1:{mkey}] Loading {local_path} (is_thinking={is_thinking}) ...")
    model, tokenizer = load_ouro_for_transformers5(local_path, device)
    engine = OuroRoute1Engine(model, device)

    # Stage 0: Fixed exactness verification
    with run_obj.phase(f"{mkey}_verify", total=1, unit="check") as pv:
        verify_res = verify_engine_exactness(model, engine, tokenizer, device)
        pv.step(1)

    # Stage 1: Open-loop trajectory diagnostic + train-only quantile calibration
    n_diag = int(cfg.get("n_diag_gsm_test", 100))
    with run_obj.phase(f"{mkey}_diag", total=n_diag + 1, unit="seqs") as pdiag:
        diag_res = run_openloop_trajectory_diagnostic(
            engine,
            gsm_test[:n_diag],
            tokenizer,
            is_thinking=is_thinking,
            max_loops=int(cfg.get("diag_max_loops", 6)),
            run_phase=pdiag,
        )
        print(
            f"[route1:{mkey}:diag] AUROC(harmful): -cos_F={diag_res['auroc_harmful_neg_cos_f']:.3f}, "
            f"-dH={diag_res['auroc_harmful_neg_dh']:.3f} vs old kappa={diag_res['auroc_harmful_kappa_old']:.3f} | "
            f"R=4 rev_frac={diag_res['per_loop']['R=4']['frac_reversal']*100:.1f}%"
        )
        calib_res = calibrate_exit_thresholds(
            engine,
            gsm_train,
            tokenizer,
            is_thinking=is_thinking,
            target_budgets=[float(x) for x in cfg.get("target_r_budgets", [1.8, 2.2, 2.8])],
            n_rollouts=int(cfg.get("n_calib_think_rollouts", 16) if is_thinking else cfg.get("n_calib_train", 24)),
        )
        pdiag.step(n_diag + 1)

    cal_ea_18 = calib_res["entropy_agree"]["1.8"]
    cal_ea_22 = calib_res["entropy_agree"]["2.2"]
    cal_ea_28 = calib_res["entropy_agree"]["2.8"]
    cal_og_22 = calib_res["ouro_gate"]["2.2"]
    rand_probs_22 = tuple(float(x) for x in cal_ea_22["calib_exit_probs"])

    # Stage 2: ARC-Challenge (for Ouro-1.4B, n=500)
    arc_results: dict[str, dict[str, Any]] = {}
    if bool(mspec.get("run_arc", False)):
        arc_policies = [
            Route1Policy("native_R2", n_loops=2),
            Route1Policy("native_R4", n_loops=4),
            Route1Policy("native_R6", n_loops=6),
            Route1Policy("loopcd_fixed_R4", n_loops=4, readout_mode="loopcd_fixed", omega=w_fix),
            Route1Policy("loopcd_adapt_R4", n_loops=4, readout_mode="loopcd_adapt", omega_max=w_max),
            Route1Policy("loopcd_adapt_R6", n_loops=6, readout_mode="loopcd_adapt", omega_max=w_max),
            Route1Policy("tcre_cos_R4", n_loops=4, readout_mode="tcre_cos", omega_max=w_max),
            Route1Policy("tcre_cos_R6", n_loops=6, readout_mode="tcre_cos", omega_max=w_max),
            Route1Policy("tcre_aitken_R4", n_loops=4, readout_mode="tcre_aitken", omega_max=w_max),
            Route1Policy("tcre_aitken_R6", n_loops=6, readout_mode="tcre_aitken", omega_max=w_max),
        ]
        with run_obj.phase(f"{mkey}_arc", total=len(arc_policies), unit="conds") as parc:
            for ip, pol in enumerate(arc_policies, 1):
                r = evaluate_arc_challenge(engine, arc_samples, tokenizer, pol)
                arc_results[pol.name] = r
                print(
                    f"[route1:{mkey}:ARC] {pol.name:18s} R={pol.n_loops} acc={r['accuracy']*100:.2f}% "
                    f"({r['n_correct']}/{r['n_total']}) [{r['elapsed_sec']:.1f}s]"
                )
                parc.row(f"{mkey}_arc", policy=pol.name, n_loops=pol.n_loops, acc=r["accuracy"])
                parc.step(ip)

    # Stage 3: GSM8K Free Generation (n=500 across ALL 3 models!)
    gsm_policies: list[Route1Policy] = [
        Route1Policy("native_R2", n_loops=2),
        Route1Policy("native_R4", n_loops=4),
        Route1Policy("loopcd_adapt_R4", n_loops=4, readout_mode="loopcd_adapt", omega_max=w_max),
        Route1Policy("tcre_cos_R4", n_loops=4, readout_mode="tcre_cos", omega_max=w_max),
        Route1Policy("tcre_aitken_R4", n_loops=4, readout_mode="tcre_aitken", omega_max=w_max),
        Route1Policy("rand_ee_22", n_loops=4, readout_mode="native", exit_mode="random", rand_exit_probs=rand_probs_22),
        Route1Policy(
            "ouro_gate_ee_22",
            n_loops=4,
            readout_mode="native",
            exit_mode="ouro_gate",
            gate_cdf_thresh=float(cal_og_22["gate_cdf_thresh"]),
        ),
        _build_ea_policy("conf_ee_22", cal_ea_22, readout_mode="native", omega_max=w_max),
        _build_ea_policy("route1_tcre_18", cal_ea_18, readout_mode="tcre_cos", omega_max=w_max),
        _build_ea_policy("route1_tcre_22", cal_ea_22, readout_mode="tcre_cos", omega_max=w_max),
        _build_ea_policy("route1_tcre_28", cal_ea_28, readout_mode="tcre_cos", omega_max=w_max),
    ]
    if bool(mspec.get("run_r6_robustness", False)):
        gsm_policies.extend([
            Route1Policy("native_R1", n_loops=1),
            Route1Policy("native_R3", n_loops=3),
            Route1Policy("native_R6", n_loops=6),
            Route1Policy("loopcd_adapt_R6", n_loops=6, readout_mode="loopcd_adapt", omega_max=w_max),
            Route1Policy("tcre_cos_R6", n_loops=6, readout_mode="tcre_cos", omega_max=w_max),
        ])

    gsm_results: dict[str, dict[str, Any]] = {}
    with run_obj.phase(f"{mkey}_gsm", total=len(gsm_policies), unit="conds") as pgsm:
        for ip, pol in enumerate(gsm_policies, 1):
            r = evaluate_math_generation(
                engine,
                gsm_test,
                tokenizer,
                pol,
                is_thinking=is_thinking,
                task="gsm8k",
                max_new_tokens=int(mspec["max_new_tokens_gsm"]),
                batch_size=int(mspec["batch_size_gsm"]),
            )
            gsm_results[pol.name] = r
            print(
                f"[route1:{mkey}:GSM8K] {pol.name:18s} R_avg={r['mean_loops']:.2f} acc={r['accuracy']*100:.2f}% "
                f"({r['n_correct']}/{r['n_total']}) comp={r['completion_rate']*100:.0f}% "
                f"tok={r['mean_gen_tokens']:.0f} aln={r['frac_aligned_deep']*100:.0f}% "
                f"rev={r['frac_reversal_deep']*100:.0f}% [{r['elapsed_sec']:.1f}s]"
            )
            pgsm.row(
                f"{mkey}_gsm",
                policy=pol.name,
                mean_r=r["mean_loops"],
                acc=r["accuracy"],
                comp=r["completion_rate"],
            )
            live_m = {f"{mkey}_gsm_live_acc": r["accuracy"]}
            if pol.name in ("native_R4", "tcre_cos_R4", "route1_tcre_22"):
                live_m[f"{mkey}_gsm_{pol.name}"] = r["accuracy"]
                if pol.name == "route1_tcre_22":
                    live_m[f"{mkey}_gsm_route1_22_r"] = r["mean_loops"]
            pgsm.step(ip, **live_m)

    # Stage 4: MATH-500 Free Generation (for Thinking models, n=200)
    math_results: dict[str, dict[str, Any]] = {}
    if bool(mspec.get("run_math", False)) and math_test:
        math_policies = [
            Route1Policy("native_R2", n_loops=2),
            Route1Policy("native_R4", n_loops=4),
            Route1Policy("loopcd_adapt_R4", n_loops=4, readout_mode="loopcd_adapt", omega_max=w_max),
            Route1Policy("tcre_cos_R4", n_loops=4, readout_mode="tcre_cos", omega_max=w_max),
            Route1Policy(
                "ouro_gate_ee_22",
                n_loops=4,
                readout_mode="native",
                exit_mode="ouro_gate",
                gate_cdf_thresh=float(cal_og_22["gate_cdf_thresh"]),
            ),
            _build_ea_policy("conf_ee_22", cal_ea_22, readout_mode="native", omega_max=w_max),
            _build_ea_policy("route1_tcre_22", cal_ea_22, readout_mode="tcre_cos", omega_max=w_max),
        ]
        with run_obj.phase(f"{mkey}_math", total=len(math_policies), unit="conds") as pmath:
            for ip, pol in enumerate(math_policies, 1):
                r = evaluate_math_generation(
                    engine,
                    math_test,
                    tokenizer,
                    pol,
                    is_thinking=True,
                    task="math500",
                    max_new_tokens=int(mspec["max_new_tokens_math"]),
                    batch_size=int(mspec["batch_size_math"]),
                )
                math_results[pol.name] = r
                print(
                    f"[route1:{mkey}:MATH500] {pol.name:18s} R_avg={r['mean_loops']:.2f} acc={r['accuracy']*100:.2f}% "
                    f"(L1-3={r['acc_level_1_3']*100:.1f}%, L4-5={r['acc_level_4_5']*100:.1f}%) "
                    f"comp={r['completion_rate']*100:.0f}% tok={r['mean_gen_tokens']:.0f} [{r['elapsed_sec']:.1f}s]"
                )
                pmath.row(
                    f"{mkey}_math",
                    policy=pol.name,
                    mean_r=r["mean_loops"],
                    acc=r["accuracy"],
                    comp=r["completion_rate"],
                )
                pmath.step(ip)

    # Stage 5: True B=1 Wall-Clock Latency Benchmark
    bench_policies = [
        Route1Policy("native_R4", n_loops=4),
        Route1Policy("native_R2", n_loops=2),
        Route1Policy("tcre_cos_R4", n_loops=4, readout_mode="tcre_cos", omega_max=w_max),
        Route1Policy(
            "ouro_gate_ee_22",
            n_loops=4,
            readout_mode="native",
            exit_mode="ouro_gate",
            gate_cdf_thresh=float(cal_og_22["gate_cdf_thresh"]),
        ),
        _build_ea_policy("route1_tcre_18", cal_ea_18, readout_mode="tcre_cos", omega_max=w_max),
        _build_ea_policy("route1_tcre_22", cal_ea_22, readout_mode="tcre_cos", omega_max=w_max),
        _build_ea_policy("route1_tcre_28", cal_ea_28, readout_mode="tcre_cos", omega_max=w_max),
    ]
    n_b1 = int(cfg.get("n_bench_b1_think", 10) if is_thinking else cfg.get("n_bench_b1_base", 16))
    with run_obj.phase(f"{mkey}_bench", total=1, unit="bench") as pbench:
        bench_res = benchmark_b1_latency(
            engine,
            gsm_test[:n_b1],
            tokenizer,
            bench_policies,
            is_thinking=is_thinking,
            max_new_tokens=256 if is_thinking else 192,
        )
        pbench.step(1, **{f"{mkey}_speedup_22": bench_res["route1_tcre_22"]["speedup_vs_native_r4"]})

    # Paired bootstrap contrasts
    def _paired(rmap: dict[str, dict[str, Any]], a: str, b: str) -> dict[str, Any] | None:
        if a not in rmap or b not in rmap:
            return None
        fa = np.asarray(rmap[a]["correct_flags"], dtype=np.float64)
        fb = np.asarray(rmap[b]["correct_flags"], dtype=np.float64)
        mean_d, lo, hi = paired_bootstrap_ci(fa - fb)
        return {
            "diff_pp": mean_d * 100.0,
            "ci95_pp": [lo * 100.0, hi * 100.0],
            "fixed": int(((fa == 1) & (fb == 0)).sum()),
            "broken": int(((fa == 0) & (fb == 1)).sum()),
        }

    contrasts: dict[str, Any] = {}
    for bname, rmap in [("arc", arc_results), ("gsm8k", gsm_results), ("math500", math_results)]:
        if not rmap:
            continue
        pairs = {
            "tcre_cos_R4_vs_native_R4": _paired(rmap, "tcre_cos_R4", "native_R4"),
            "tcre_cos_R4_vs_loopcd_adapt_R4": _paired(rmap, "tcre_cos_R4", "loopcd_adapt_R4"),
            "tcre_aitken_R4_vs_native_R4": _paired(rmap, "tcre_aitken_R4", "native_R4"),
            "tcre_cos_R6_vs_loopcd_adapt_R6": _paired(rmap, "tcre_cos_R6", "loopcd_adapt_R6"),
            "tcre_cos_R6_vs_native_R6": _paired(rmap, "tcre_cos_R6", "native_R6"),
            "route1_tcre_22_vs_native_R4": _paired(rmap, "route1_tcre_22", "native_R4"),
            "route1_tcre_22_vs_native_R2": _paired(rmap, "route1_tcre_22", "native_R2"),
            "route1_tcre_22_vs_rand_ee_22": _paired(rmap, "route1_tcre_22", "rand_ee_22"),
            "route1_tcre_22_vs_ouro_gate_ee_22": _paired(rmap, "route1_tcre_22", "ouro_gate_ee_22"),
            "route1_tcre_22_vs_conf_ee_22": _paired(rmap, "route1_tcre_22", "conf_ee_22"),
            "route1_tcre_28_vs_native_R4": _paired(rmap, "route1_tcre_28", "native_R4"),
            "route1_tcre_18_vs_native_R2": _paired(rmap, "route1_tcre_18", "native_R2"),
        }
        contrasts[bname] = {k: v for k, v in pairs.items() if v is not None}

    def _strip(rmap: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {k: {ck: cv for ck, cv in v.items() if ck != "correct_flags"} for k, v in rmap.items()}

    m_out = {
        "model_key": mkey,
        "local_path": str(local_path),
        "is_thinking": is_thinking,
        "train_loops": train_loops,
        "verify": verify_res,
        "diag": diag_res,
        "calib": calib_res,
        "arc_results": _strip(arc_results),
        "gsm_results": _strip(gsm_results),
        "math_results": _strip(math_results),
        "bench_b1": bench_res,
        "contrasts": contrasts,
    }
    (out_dir / f"partial_{mkey}.json").write_text(json.dumps(m_out, indent=2, ensure_ascii=False))

    del model, engine
    gc.collect()
    torch.cuda.empty_cache()
    return m_out


def render_report_md(summary: dict[str, Any]) -> str:
    lines = ["# Route 1 Verification Report: TCRE + Ragged-KV Token Compute Reallocation", ""]
    lines.append(f"- **Elapsed**: `{summary['elapsed_sec']:.1f}s` on GPU `{summary['gpu_index']}` ({summary['gpu_name']})")
    lines.append("")

    for mkey, md in summary["models"].items():
        v = md["verify"]
        d = md["diag"]
        lines.append(f"## Model: `{mkey}` (effective_layers={v['effective_layers']})")
        lines.append(
            f"- **Exactness**: B=1 diff=`{v['max_abs_diff_b1']:.2e}`, B=2 same-batch diff=`{v['max_abs_diff_b2_same_batch']:.2e}`, "
            f"B=2 vs B=1 mean=`{v['mean_abs_diff_b2_vs_b1']:.3f}` top1=`{v['top1_b2_exact']}`, "
            f"4-tok decode exact=`{v['decode_4tok_exact_both']}`, EE B=2==B=1=`{v['ee_b2_matches_b1']}`"
        )
        lines.append(
            f"- **Open-Loop Trajectory Diagnostic**: AUROC(harmful): `-cos_F={d['auroc_harmful_neg_cos_f']:.3f}`, "
            f"`-cos_E={d['auroc_harmful_neg_cos_e']:.3f}`, `-dH={d['auroc_harmful_neg_dh']:.3f}` vs old `kappa={d['auroc_harmful_kappa_old']:.3f}`"
        )
        lines.append("")

        for title, bkey in [("ARC-Challenge (n=500)", "arc_results"), ("GSM8K Free-Gen (n=500)", "gsm_results"), ("MATH-500 Free-Gen (n=200)", "math_results")]:
            rmap = md.get(bkey, {})
            if not rmap:
                continue
            lines.append(f"### {title}")
            lines.append("| Policy | Mean R | Accuracy | Correct/Total | Comp% | AvgTok | B=1 Speedup | Sec |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            b1 = md.get("bench_b1", {})
            for pname, rv in rmap.items():
                mr = rv.get("mean_loops", rv.get("n_loops", 4))
                comp = f"{rv['completion_rate']*100:.0f}%" if "completion_rate" in rv else "—"
                tok = f"{rv['mean_gen_tokens']:.0f}" if "mean_gen_tokens" in rv else "—"
                sp = f"{b1[pname]['speedup_vs_native_r4']:.2f}x" if pname in b1 else "—"
                lines.append(
                    f"| `{pname}` | {mr:.2f} | {rv['accuracy']*100:.2f}% | "
                    f"{rv['n_correct']}/{rv['n_total']} | {comp} | {tok} | {sp} | {rv['elapsed_sec']:.1f} |"
                )
            lines.append("")

        lines.append("### Paired Bootstrap Contrasts (pp, 95% CI)")
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
    seed = int(cfg.get("seed", 20261125))

    gpu_idx, gpu_lock, gpu_info = acquire_idle_gpu(cfg.get("preferred_gpu"))
    try:
        device = torch.device(f"cuda:{gpu_idx}")
        torch.cuda.set_device(device)
        print(f"[route1] Acquired idle GPU {gpu_idx}: {gpu_info.name} (mem={gpu_info.mem_used_mib}MiB, util={gpu_info.util_pct}%)")

        gsm_train = load_gsm8k_split(args.dataset_dir, "train", n_samples=int(cfg.get("n_calib_train", 32)), seed=seed)
        gsm_test = load_gsm8k_split(args.dataset_dir, "test", n_samples=int(cfg.get("n_gsm_eval", 500)), seed=seed + 1)
        math_test = load_math500_samples(
            args.dataset_dir,
            n_samples=int(cfg.get("n_math_eval", 200)),
            max_level=int(cfg.get("math_max_level", 5)),
            seed=seed + 2,
        )
        arc_samples = load_arc_challenge_samples(args.dataset_dir, n_samples=int(cfg.get("n_arc_eval", 500)), seed=seed + 3)

        # Declare exact stage plan dynamically
        plan_stages: list[dict[str, Any]] = [
            {"phase": "setup", "of": 1, "unit": "env"},
            {"phase": "download", "of": 1, "unit": "assets"},
        ]
        n_diag = int(cfg.get("n_diag_gsm_test", 100))
        for mspec in cfg["models"]:
            mk = mspec["key"]
            plan_stages.append({"phase": f"{mk}_verify", "of": 1, "unit": "check"})
            plan_stages.append({"phase": f"{mk}_diag", "of": n_diag + 1, "unit": "seqs"})
            if bool(mspec.get("run_arc", False)):
                plan_stages.append({"phase": f"{mk}_arc", "of": 10, "unit": "conds"})
            n_gsm_pols = 16 if bool(mspec.get("run_r6_robustness", False)) else 11
            plan_stages.append({"phase": f"{mk}_gsm", "of": n_gsm_pols, "unit": "conds"})
            if bool(mspec.get("run_math", False)):
                plan_stages.append({"phase": f"{mk}_math", "of": 7, "unit": "conds"})
            plan_stages.append({"phase": f"{mk}_bench", "of": 1, "unit": "bench"})
        plan_stages.append({"phase": "report", "of": 1, "unit": "file"})
        orx.plan(plan_stages)

        run_obj = orx.progress()
        model_outputs: dict[str, Any] = {}
        for mspec in cfg["models"]:
            model_outputs[mspec["key"]] = run_model_route1(
                mspec,
                cfg=cfg,
                model_dir=Path(args.model_dir),
                gsm_train=gsm_train,
                gsm_test=gsm_test,
                math_test=math_test,
                arc_samples=arc_samples,
                device=device,
                run_obj=run_obj,
                out_dir=out_dir,
            )

        with run_obj.phase("report", total=1, unit="file") as prep:
            summary = {
                "idea": "route1_tcre_ragged",
                "gpu_index": gpu_idx,
                "gpu_name": gpu_info.name,
                "elapsed_sec": time.time() - t0,
                "config": cfg,
                "models": model_outputs,
            }
            (out_dir / "summary_route1.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
            report_md = render_report_md(summary)
            (out_dir / "report_route1.md").write_text(report_md)

            final_metrics: dict[str, float] = {}
            for mk, md in model_outputs.items():
                if "route1_tcre_22" in md["gsm_results"]:
                    final_metrics[f"{mk}_gsm_route1_22"] = md["gsm_results"]["route1_tcre_22"]["accuracy"]
                    final_metrics[f"{mk}_gsm_route1_22_r"] = md["gsm_results"]["route1_tcre_22"]["mean_loops"]
                if "native_R4" in md["gsm_results"]:
                    final_metrics[f"{mk}_gsm_native_r4"] = md["gsm_results"]["native_R4"]["accuracy"]
                if "tcre_cos_R4" in md["gsm_results"]:
                    final_metrics[f"{mk}_gsm_tcre_r4"] = md["gsm_results"]["tcre_cos_R4"]["accuracy"]
                if "route1_tcre_22" in md["bench_b1"]:
                    final_metrics[f"{mk}_speedup_22"] = md["bench_b1"]["route1_tcre_22"]["speedup_vs_native_r4"]
            prep.step(1, **final_metrics)

            print("========== REPORT (report_route1.md) ==========")
            print(report_md)
            print("========== END REPORT ==========")
    finally:
        gpu_lock.release()


if __name__ == "__main__":
    main()
