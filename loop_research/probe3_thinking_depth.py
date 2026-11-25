"""Round 1 Probe 3: Reasoning LoopLM Depth Scaling Anatomy, R=1 KV Poisoning Diagnosis,
Floor-Protected Adaptive Recurrence (R_min=2), and <think> Phase-Split Depth.

Evaluates cached ByteDance Ouro models:
  - ByteDance/Ouro-1.4B (GSM8K n=160)
  - ByteDance/Ouro-1.4B-Thinking (GSM8K n=100 & MATH-500 n=100)
  - ByteDance/Ouro-2.6B-Thinking (MATH-500 n=80)

Key scientific measurements:
  0. Exactness check vs native HF OuroForCausalLM (B=1 exact + 4-tok decode exact).
  1. Part A (Complete Depth Scaling Curve): R in {1, 2, 3, 4, 6} (+ R=8 on 1.4B)
     measuring accuracy, completion rate, mean thinking/output length, and wall-clock time.
  2. Part B (Diagnosing Route-1's Early-Exit Crash & Ragged-KV vs Full-KV Oracle):
     - gate_rmin1 (reproduces R=1 exit crash)
     - gate_rmin2 (floor R_min=2, Ouro native gate)
     - kl_rmin2 (floor R_min=2, training-free KL(p_{r-1}||p_r) exit)
     - gap_rmin2 (floor R_min=2, top-2 logit gap + agreement exit)
     - kl_rmin2_fullkv (oracle control: same exit decisions, full-depth KV preserved)
  3. Part C (Phase-Split Recurrence for Thinking Models):
     - phase_think2_ans4 (R=2 inside <think>, R=4 after </think>)
     - phase_think3_ans4 (R=3 inside <think>, R=4 after </think>)
     - phase_think_kl2_ans4 (kl_rmin2 inside <think>, R=4 after </think>)
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
    load_gsm8k_split,
    load_math500_samples,
)
from loop_research.ftrs_stepping import (
    GSM8K_4SHOT,
    _apply_rope,
    load_ouro_for_transformers5,
    paired_bootstrap_ci,
)
from loop_research.gpu_utils import acquire_idle_gpu


# ---------------------------------------------------------------------------
# Policy Spec & Pure-Logic Calibration Helpers (CPU Testable!)
# ---------------------------------------------------------------------------

@dataclass
class DepthPolicy:
    name: str
    n_loops: int = 4
    exit_mode: str = "fixed"  # fixed | ouro_gate | kl | gap_agree | phase_split | phase_kl
    r_min: int = 2
    thresh: float = 0.0
    think_loops: int = 2
    answer_loops: int = 4
    ragged_kv: bool = True


def compute_kl_div(p_prev: torch.Tensor, p_curr: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Compute KL(p_prev || p_curr) in nats over the last dimension."""
    p0 = p_prev.clamp_min(eps)
    p1 = p_curr.clamp_min(eps)
    return (p0 * (p0.log() - p1.log())).sum(dim=-1).clamp_min(0.0)


def calibrate_floor_threshold(
    values_by_loop: dict[int, np.ndarray],
    *,
    mode: str,  # "kl" (exit if val < tau) | "gate" (exit if cdf >= tau) | "gap" (exit if val > tau)
    r_min: int,
    n_loops: int,
    target_mean_r: float,
) -> tuple[float, float, list[float]]:
    """Binary-search a single scalar threshold on calibration trace to hit target_mean_r."""
    first_arr = next(iter(values_by_loop.values()))
    n_tok = len(first_arr)
    if n_tok == 0:
        return 0.5, float(n_loops), [0.0] * (n_loops - 1) + [1.0]

    def simulate(tau: float) -> tuple[float, list[float]]:
        exited = np.zeros(n_tok, dtype=bool)
        exit_r = np.full(n_tok, n_loops, dtype=np.int32)
        for r in range(r_min, n_loops):
            arr = values_by_loop[r]
            if mode == "kl":
                cond = (~exited) & (arr < tau)
            elif mode == "gate":
                cond = (~exited) & (arr >= tau)
            elif mode == "gap":
                cond = (~exited) & (arr > tau)
            else:
                raise ValueError(mode)
            exit_r[cond] = r
            exited |= cond
        probs = [float((exit_r == r).mean()) for r in range(1, n_loops + 1)]
        return float(exit_r.mean()), probs

    # Set search bounds so larger `t` always means HIGHER mean_r or LOWER mean_r monotonically
    if mode == "kl":
        # larger tau => MORE exits => LOWER mean_r
        lo, hi = 1e-6, 5.0
        for _ in range(36):
            mid = 0.5 * (lo + hi)
            mr, _ = simulate(mid)
            if mr > target_mean_r:
                lo = mid
            else:
                hi = mid
        best_tau = 0.5 * (lo + hi)
    elif mode == "gate":
        # larger tau => FEWER exits => HIGHER mean_r
        lo, hi = 1e-4, 0.9999
        for _ in range(36):
            mid = 0.5 * (lo + hi)
            mr, _ = simulate(mid)
            if mr < target_mean_r:
                lo = mid
            else:
                hi = mid
        best_tau = 0.5 * (lo + hi)
    elif mode == "gap":
        # larger tau => FEWER exits => HIGHER mean_r
        lo, hi = 0.0, 25.0
        for _ in range(36):
            mid = 0.5 * (lo + hi)
            mr, _ = simulate(mid)
            if mr < target_mean_r:
                lo = mid
            else:
                hi = mid
        best_tau = 0.5 * (lo + hi)
    else:
        raise ValueError(mode)

    mr, probs = simulate(best_tau)
    return best_tau, mr, probs


def resolve_think_end_ids(tokenizer) -> set[int]:
    """Identify token IDs corresponding to </think> if present in vocab."""
    out: set[int] = set()
    for s in ("</think>", "<|/think|>"):
        tid = tokenizer.convert_tokens_to_ids(s)
        if isinstance(tid, int) and tid >= 0:
            # Verify it isn't unk_token_id
            if tokenizer.unk_token_id is None or tid != tokenizer.unk_token_id:
                out.add(tid)
    return out


# ---------------------------------------------------------------------------
# OOM-Safe Exact Ouro KV-Cache Engine
# ---------------------------------------------------------------------------

class OuroDepthEngine:
    """Exact, OOM-safe batched KV-cache engine for Ouro-1.4B / 1.4B-Thinking / 2.6B-Thinking."""

    def __init__(self, model, device: torch.device) -> None:
        self.model = model
        self.core = model.model
        self.lm_head = model.lm_head
        self.device = device
        self.dtype = next(model.parameters()).dtype
        self.num_layers = len(self.core.layers)
        self.train_loops = int(getattr(model.config, "total_ut_steps", 4))
        self.num_heads = int(model.config.num_attention_heads)
        self.num_kv_heads = int(model.config.num_key_value_heads)
        self.head_dim = int(getattr(model.config, "head_dim", model.config.hidden_size // self.num_heads))
        self.enable_gqa = self.num_heads != self.num_kv_heads
        self.has_early_exit_gate = hasattr(self.core, "early_exit_gate") and self.core.early_exit_gate is not None
        self.vocab_size = int(self.lm_head.out_features)

    def _run_layer_prefill(self, layer, h, cos, sin, attn_mask):
        residual = h
        x = layer.input_layernorm(h)
        bsz, seq_len, _ = x.shape
        attn = layer.self_attn
        q = attn.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = attn.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = attn.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=(attn_mask is None),
            scale=attn.scaling,
            enable_gqa=self.enable_gqa,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.num_heads * self.head_dim)
        h = residual + layer.input_layernorm_2(attn.o_proj(out))
        h = h + layer.post_attention_layernorm_2(layer.mlp(layer.post_attention_layernorm(h)))
        return h, (k, v)

    def _run_layer_decode(self, layer, h, cos, sin, k_layer_buf, v_layer_buf, write_pos, attn_mask, act_idx=None):
        residual = h
        x = layer.input_layernorm(h)
        bsz = x.shape[0]
        attn = layer.self_attn
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
            q, k_view, v_view,
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
    def generate_batched(
        self,
        prompt_ids_list: list[list[int]],
        policy: DepthPolicy,
        *,
        max_new_tokens: int,
        pad_token_id: int,
        eos_token_ids: set[int],
        think_end_ids: set[int],
        stop_strings: list[str] | None = None,
        tokenizer: Any = None,
        collect_calib: bool = False,
    ) -> tuple[list[list[int]], dict[str, Any]]:
        bsz = len(prompt_ids_list)
        max_p = max(len(p) for p in prompt_ids_list)
        max_total_len = max_p + max_new_tokens
        n_loops = policy.n_loops

        input_ids = torch.full((bsz, max_p), pad_token_id, device=self.device, dtype=torch.long)
        attn_mask_1d = torch.zeros((bsz, max_p), device=self.device, dtype=torch.long)
        for i, p in enumerate(prompt_ids_list):
            input_ids[i, max_p - len(p) :] = torch.tensor(p, device=self.device, dtype=torch.long)
            attn_mask_1d[i, max_p - len(p) :] = 1

        k_buf = torch.empty(
            (n_loops, self.num_layers, bsz, self.num_kv_heads, max_total_len, self.head_dim),
            device=self.device,
            dtype=self.dtype,
        )
        v_buf = torch.empty_like(k_buf)

        pos_ids = (attn_mask_1d.cumsum(dim=-1) - 1).clamp_min(0)
        if bsz == 1:
            prefill_mask = None
        else:
            valid_k = attn_mask_1d[:, None, None, :].bool()
            causal = torch.tril(torch.ones((max_p, max_p), dtype=torch.bool, device=self.device))[None, None, :, :]
            diag = torch.eye(max_p, dtype=torch.bool, device=self.device)[None, None, :, :]
            prefill_mask = (causal & valid_k) | diag

        # Prefill: full n_loops pass
        h = self.core.embed_tokens(input_ids)
        cos, sin = self.core.rotary_emb(h, pos_ids)
        for ut in range(n_loops):
            for l_idx, layer in enumerate(self.core.layers):
                h, (k_p, v_p) = self._run_layer_prefill(layer, h, cos, sin, prefill_mask)
                k_buf[ut, l_idx, :, :, :max_p, :] = k_p
                v_buf[ut, l_idx, :, :, :max_p, :] = v_p
            h = self.core.norm(h)
        first_logits = self.lm_head(h[:, -1, :]).float()

        next_tok = first_logits.argmax(dim=-1)
        generated: list[list[int]] = [[] for _ in range(bsz)]
        finished = torch.zeros(bsz, device=self.device, dtype=torch.bool)
        in_answer = torch.zeros(bsz, device=self.device, dtype=torch.bool)
        for i in range(bsz):
            t_i = int(next_tok[i].item())
            generated[i].append(t_i)
            if t_i in eos_token_ids:
                finished[i] = True
            if t_i in think_end_ids:
                in_answer[i] = True

        cur_valid_mask = attn_mask_1d
        next_pos = attn_mask_1d.sum(dim=-1, keepdim=True)

        loop_hist = np.zeros(n_loops, dtype=np.int64)
        total_decode_tokens = 0
        calib_kl: dict[int, list[np.ndarray]] = {r: [] for r in range(2, n_loops)}
        calib_gate: dict[int, list[np.ndarray]] = {r: [] for r in range(1, n_loops)}
        calib_gap: dict[int, list[np.ndarray]] = {r: [] for r in range(1, n_loops)}

        need_probs = collect_calib or policy.exit_mode in ("kl", "phase_kl")
        need_gap = collect_calib or policy.exit_mode == "gap_agree"
        need_gate = (collect_calib or policy.exit_mode == "ouro_gate") and self.has_early_exit_gate

        for step_idx in range(1, max_new_tokens):
            if bool(finished.all().item()):
                break

            write_pos = max_p + step_idx - 1
            cur_valid_mask = torch.cat(
                [cur_valid_mask, torch.ones((bsz, 1), device=self.device, dtype=torch.long)], dim=1
            )
            dec_mask_full = None if bsz == 1 else cur_valid_mask[:, None, None, :].bool()

            h = self.core.embed_tokens(next_tok.unsqueeze(1))
            cos, sin = self.core.rotary_emb(h, next_pos)

            # Determine active rows for compute vs rows whose output token is already decided
            decided = finished.clone()
            compute_done = finished.clone()
            step_logits = torch.zeros((bsz, self.vocab_size), device=self.device, dtype=torch.float32)
            exit_loop_row = torch.zeros((bsz,), device=self.device, dtype=torch.long)

            p_prev_full: torch.Tensor | None = None
            top1_prev_full: torch.Tensor | None = None
            rem_prob = torch.ones((bsz,), device=self.device, dtype=torch.float32)
            cdf_acc = torch.zeros((bsz,), device=self.device, dtype=torch.float32)

            for ut in range(n_loops):
                loop_num = ut + 1  # 1-indexed
                active_comp = ~compute_done
                if not bool(active_comp.any().item()):
                    break

                all_active = bool(active_comp.all().item())
                if all_active:
                    act_idx = None
                    h_work = h
                    cos_w, sin_w = cos, sin
                    mask_w = dec_mask_full
                else:
                    act_idx = active_comp.nonzero(as_tuple=False).squeeze(-1)
                    h_work = h[act_idx]
                    cos_w, sin_w = cos[act_idx], sin[act_idx]
                    mask_w = None if dec_mask_full is None else dec_mask_full[act_idx]

                for l_idx, layer in enumerate(self.core.layers):
                    h_work = self._run_layer_decode(
                        layer, h_work, cos_w, sin_w,
                        k_buf[ut, l_idx], v_buf[ut, l_idx],
                        write_pos, mask_w, act_idx=act_idx,
                    )
                h_work = self.core.norm(h_work)

                # Write updated h back into full h tensor
                if all_active:
                    h = h_work
                else:
                    h = h.clone()
                    h[act_idx] = h_work

                is_last = loop_num == n_loops

                # Compute readout logits only when needed
                need_logits_now = (
                    is_last
                    or need_probs
                    or need_gap
                    or (policy.exit_mode == "ouro_gate" and loop_num >= policy.r_min)
                    or (policy.exit_mode == "phase_split" and loop_num == policy.think_loops)
                )
                z_act = self.lm_head(h_work[:, 0, :]).float() if need_logits_now else None

                # Ouro learned gate CDF update
                if need_gate and not is_last:
                    g_val = torch.sigmoid(self.core.early_exit_gate(h_work[:, 0, :]).squeeze(-1).float())
                    if all_active:
                        p_step = rem_prob * g_val
                        cdf_acc = cdf_acc + p_step
                        rem_prob = rem_prob * (1.0 - g_val)
                    else:
                        p_step = rem_prob[act_idx] * g_val
                        cdf_acc[act_idx] = cdf_acc[act_idx] + p_step
                        rem_prob[act_idx] = rem_prob[act_idx] * (1.0 - g_val)

                # Distribution / KL / gap stats
                p_curr_act = F.softmax(z_act, dim=-1) if (need_probs and z_act is not None) else None
                kl_act = None
                if need_probs and p_curr_act is not None:
                    if p_prev_full is not None:
                        pp = p_prev_full if all_active else p_prev_full[act_idx]
                        kl_act = compute_kl_div(pp, p_curr_act)
                    if all_active:
                        p_prev_full = p_curr_act
                    else:
                        if p_prev_full is None:
                            p_prev_full = torch.zeros((bsz, self.vocab_size), device=self.device, dtype=torch.float32)
                        p_prev_full[act_idx] = p_curr_act

                gap_act = None
                agree_act = None
                if need_gap and z_act is not None:
                    top2 = torch.topk(z_act, k=2, dim=-1)
                    gap_act = (top2.values[:, 0] - top2.values[:, 1]).clamp_min(0.0)
                    t1_act = top2.indices[:, 0]
                    if top1_prev_full is not None:
                        tp = top1_prev_full if all_active else top1_prev_full[act_idx]
                        agree_act = t1_act == tp
                    else:
                        agree_act = torch.ones_like(t1_act, dtype=torch.bool)
                    if all_active:
                        top1_prev_full = t1_act
                    else:
                        if top1_prev_full is None:
                            top1_prev_full = torch.zeros((bsz,), device=self.device, dtype=torch.long)
                        top1_prev_full[act_idx] = t1_act

                # Calibration recording (only during full fixed R=n_loops pass)
                if collect_calib and not is_last:
                    unfin = ~finished
                    if bool(unfin.any().item()):
                        if kl_act is not None and loop_num in calib_kl:
                            calib_kl[loop_num].append(kl_act[unfin].cpu().numpy())
                        if need_gate and loop_num in calib_gate:
                            calib_gate[loop_num].append(cdf_acc[unfin].cpu().numpy())
                        if gap_act is not None and loop_num in calib_gap:
                            # If loop_num >= 2, require agreement with previous loop (else gap=-1)
                            eff_gap = torch.where(agree_act, gap_act, torch.full_like(gap_act, -1.0)) if (loop_num >= 2 and agree_act is not None) else gap_act
                            calib_gap[loop_num].append(eff_gap[unfin].cpu().numpy())

                # Decide which undecided rows exit at this loop
                if not is_last and policy.exit_mode != "fixed":
                    if all_active:
                        undec_act = ~decided
                        in_ans_act = in_answer
                    else:
                        undec_act = ~decided[act_idx]
                        in_ans_act = in_answer[act_idx]

                    exit_act = torch.zeros_like(undec_act)
                    if policy.exit_mode == "ouro_gate" and loop_num >= policy.r_min:
                        cdf_a = cdf_acc if all_active else cdf_acc[act_idx]
                        exit_act = undec_act & (cdf_a >= policy.thresh)
                    elif policy.exit_mode == "kl" and loop_num >= policy.r_min and kl_act is not None:
                        exit_act = undec_act & (kl_act < policy.thresh)
                    elif policy.exit_mode == "gap_agree" and loop_num >= policy.r_min and gap_act is not None:
                        cond_g = (gap_act > policy.thresh) & (agree_act if (loop_num >= 2 and agree_act is not None) else True)
                        exit_act = undec_act & cond_g
                    elif policy.exit_mode == "phase_split" and loop_num == policy.think_loops:
                        exit_act = undec_act & (~in_ans_act)
                    elif policy.exit_mode == "phase_kl" and loop_num >= policy.r_min and kl_act is not None:
                        exit_act = undec_act & (~in_ans_act) & (kl_act < policy.thresh)

                    if bool(exit_act.any().item()):
                        assert z_act is not None
                        ex_local = exit_act.nonzero(as_tuple=False).squeeze(-1)
                        ex_global = ex_local if all_active else act_idx[ex_local]
                        step_logits[ex_global] = z_act[ex_local]
                        exit_loop_row[ex_global] = loop_num
                        decided[ex_global] = True

                        if policy.ragged_kv:
                            compute_done[ex_global] = True
                            k_buf[ut + 1 :, :, ex_global, :, write_pos : write_pos + 1, :] = k_buf[
                                ut : ut + 1, :, ex_global, :, write_pos : write_pos + 1, :
                            ]
                            v_buf[ut + 1 :, :, ex_global, :, write_pos : write_pos + 1, :] = v_buf[
                                ut : ut + 1, :, ex_global, :, write_pos : write_pos + 1, :
                            ]

                # Final loop assigns logits to all remaining undecided rows
                if is_last:
                    assert z_act is not None
                    if all_active:
                        undec_g = ~decided
                        if bool(undec_g.any().item()):
                            step_logits[undec_g] = z_act[undec_g]
                            exit_loop_row[undec_g] = n_loops
                    else:
                        undec_local = (~decided[act_idx]).nonzero(as_tuple=False).squeeze(-1)
                        if undec_local.numel() > 0:
                            ug = act_idx[undec_local]
                            step_logits[ug] = z_act[undec_local]
                            exit_loop_row[ug] = n_loops

            # Record loop histogram over unfinished rows
            unfin_mask = ~finished
            if bool(unfin_mask.any().item()):
                el_vals = exit_loop_row[unfin_mask].cpu().numpy()
                for rv in el_vals:
                    loop_hist[int(rv) - 1] += 1
                total_decode_tokens += int(el_vals.size)

            next_tok = step_logits.argmax(dim=-1)
            next_pos = next_pos + 1

            for j in range(bsz):
                if finished[j]:
                    continue
                t_j = int(next_tok[j].item())
                generated[j].append(t_j)
                if t_j in eos_token_ids:
                    finished[j] = True
                if t_j in think_end_ids:
                    in_answer[j] = True

            if stop_strings and tokenizer is not None and (step_idx % 16 == 0):
                for j in range(bsz):
                    if finished[j]:
                        continue
                    txt = tokenizer.decode(generated[j], skip_special_tokens=True)
                    if any(ss in txt for ss in stop_strings):
                        finished[j] = True

        del k_buf, v_buf
        mean_r = (
            float(sum((u + 1) * int(loop_hist[u]) for u in range(n_loops)) / max(total_decode_tokens, 1))
            if total_decode_tokens > 0
            else float(n_loops)
        )
        stats: dict[str, Any] = {
            "mean_loops": mean_r,
            "loop_hist": loop_hist.tolist(),
            "total_decode_tokens": int(total_decode_tokens),
        }
        if collect_calib:
            stats["calib_kl"] = {r: np.concatenate(v) if v else np.array([]) for r, v in calib_kl.items()}
            stats["calib_gate"] = {r: np.concatenate(v) if v else np.array([]) for r, v in calib_gate.items()}
            stats["calib_gap"] = {r: np.concatenate(v) if v else np.array([]) for r, v in calib_gap.items()}
        return generated, stats


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
def verify_engine_exactness(model, engine: OuroDepthEngine, tokenizer) -> dict[str, Any]:
    prompt = "Question: If 3x + 5 = 20, what is x?\nAnswer:"
    p_ids = tokenizer.encode(prompt, add_special_tokens=True)
    inp = torch.tensor([p_ids], device=engine.device, dtype=torch.long)

    # Native HF logits & 4-tok greedy with use_cache=False (safe under transformers 5.x)
    hf_logits = model(input_ids=inp, use_cache=False).logits[0, -1].float()
    hf_toks: list[int] = []
    cur = list(p_ids)
    for _ in range(4):
        lg = model(input_ids=torch.tensor([cur], device=engine.device, dtype=torch.long), use_cache=False).logits[0, -1].float()
        nxt = int(lg.argmax().item())
        hf_toks.append(nxt)
        cur.append(nxt)

    # Engine 4-tok greedy at R=train_loops
    pol = DepthPolicy(name="verify", n_loops=engine.train_loops, exit_mode="fixed")
    eng_gen, _ = engine.generate_batched(
        [p_ids], pol, max_new_tokens=4,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id or 0,
        eos_token_ids=set(), think_end_ids=set(),
    )
    eng_toks = eng_gen[0]

    # Compare 1st-step logits explicitly via single forward
    h = engine.core.embed_tokens(inp)
    pos_ids = torch.arange(len(p_ids), device=engine.device).unsqueeze(0)
    cos, sin = engine.core.rotary_emb(h, pos_ids)
    for _ in range(engine.train_loops):
        for layer in engine.core.layers:
            h, _ = engine._run_layer_prefill(layer, h, cos, sin, None)
        h = engine.core.norm(h)
    eng_logits = engine.lm_head(h[0, -1]).float()
    max_diff = float((hf_logits - eng_logits).abs().max().item())

    res = {
        "max_abs_diff_b1": max_diff,
        "decode_4tok_exact": eng_toks == hf_toks,
        "eng_toks": eng_toks,
        "hf_toks": hf_toks,
    }
    print(f"[verify] OuroDepthEngine exactness: {json.dumps(res)}", flush=True)
    if max_diff > 0.15 or not res["decode_4tok_exact"]:
        raise RuntimeError(f"Engine exactness FAILED: {res}")
    return res


@torch.inference_mode()
def evaluate_policy(
    engine: OuroDepthEngine,
    items: list[dict[str, Any]],
    tokenizer,
    policy: DepthPolicy,
    *,
    is_thinking: bool,
    task: str,
    max_new_tokens: int,
    batch_size: int,
) -> dict[str, Any]:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    eos_ids = _get_eos_ids(tokenizer)
    think_end_ids = resolve_think_end_ids(tokenizer) if is_thinking else set()
    stop_strings = None if is_thinking else ["\nQuestion:", "\n\nQuestion:"]
    prompts = [_format_gen_prompt(it, tokenizer, is_thinking=is_thinking, task=task) for it in items]

    correct_flags: list[int] = []
    finished_flags: list[int] = []
    gen_lens: list[int] = []
    think_lens: list[int] = []
    total_hist = np.zeros(policy.n_loops, dtype=np.int64)
    total_dec = 0
    t0 = time.time()

    for s0 in range(0, len(items), batch_size):
        b_items = items[s0 : s0 + batch_size]
        b_prompts = prompts[s0 : s0 + batch_size]
        gen_batch, st = engine.generate_batched(
            b_prompts, policy,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_id,
            eos_token_ids=eos_ids,
            think_end_ids=think_end_ids,
            stop_strings=stop_strings,
            tokenizer=tokenizer,
        )
        torch.cuda.empty_cache()
        total_hist += np.asarray(st["loop_hist"], dtype=np.int64)
        total_dec += int(st["total_decode_tokens"])

        for it, gids in zip(b_items, gen_batch):
            raw_txt = tokenizer.decode(gids, skip_special_tokens=False)
            if not is_thinking:
                clean_txt = tokenizer.decode(gids, skip_special_tokens=True)
                if "\nQuestion:" in clean_txt:
                    clean_txt = clean_txt.split("\nQuestion:")[0]
                pred = extract_predicted_math_answer(clean_txt, mode=task)
                finished_flags.append(1 if ("####" in clean_txt or len(gids) < max_new_tokens) else 0)
                think_lens.append(0)
            else:
                pred = extract_predicted_math_answer(raw_txt, mode=task)
                has_end = "</think>" in raw_txt
                finished_flags.append(1 if (has_end or len(gids) < max_new_tokens) else 0)
                # Estimate tokens before </think>
                t_len = len(gids)
                for idx_t, tid in enumerate(gids):
                    if tid in think_end_ids:
                        t_len = idx_t + 1
                        break
                think_lens.append(t_len)
            correct_flags.append(int(pred != "" and pred == it["gold"]))
            gen_lens.append(len(gids))

    elapsed = time.time() - t0
    mean_r = (
        float(sum((u + 1) * int(total_hist[u]) for u in range(policy.n_loops)) / max(total_dec, 1))
        if total_dec > 0
        else float(policy.n_loops)
    )
    mean_len = float(np.mean(gen_lens))
    return {
        "policy": policy.name,
        "task": task,
        "n_loops_max": policy.n_loops,
        "mean_loops": mean_r,
        "loop_fracs": (total_hist / max(total_dec, 1)).tolist(),
        "accuracy": float(np.mean(correct_flags)),
        "n_correct": int(sum(correct_flags)),
        "n_total": len(correct_flags),
        "completion_rate": float(np.mean(finished_flags)),
        "mean_gen_tokens": mean_len,
        "mean_think_tokens": float(np.mean(think_lens)),
        "loop_tokens_per_prob": mean_r * mean_len,
        "elapsed_sec": elapsed,
        "correct_flags": correct_flags,
    }


def run_model_probe3(
    mspec: dict[str, Any],
    *,
    model_dir: Path,
    dataset_dir: Path,
    cfg: dict[str, Any],
    device: torch.device,
    run_obj: Any,
) -> dict[str, Any]:
    mkey = mspec["key"]
    is_thinking = bool(mspec["is_thinking"])
    local_dir = model_dir / mspec["local_dir"]
    seed = int(cfg.get("seed", 20261126))

    print(f"\n[probe3:{mkey}] Loading {local_dir} (is_thinking={is_thinking}) ...", flush=True)
    model, tokenizer = load_ouro_for_transformers5(local_dir, device)
    engine = OuroDepthEngine(model, device)

    with run_obj.phase(f"{mkey}_verify", total=1, unit="check") as pv:
        verify_res = verify_engine_exactness(model, engine, tokenizer)
        pv.step(1)

    # Zero-leakage calibration on GSM8K train split (16 prompts)
    gsm_train = load_gsm8k_split(dataset_dir, split="train", n_samples=16, seed=seed)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    eos_ids = _get_eos_ids(tokenizer)
    think_end_ids = resolve_think_end_ids(tokenizer) if is_thinking else set()
    calib_prompts = [_format_gen_prompt(it, tokenizer, is_thinking=is_thinking, task="gsm8k") for it in gsm_train]

    with run_obj.phase(f"{mkey}_calib", total=1, unit="pass") as pc:
        cal_pol = DepthPolicy(name="calib", n_loops=4, exit_mode="fixed")
        cal_bsz = min(8, len(calib_prompts))
        kl_all: dict[int, list[np.ndarray]] = {2: [], 3: []}
        gate_all: dict[int, list[np.ndarray]] = {1: [], 2: [], 3: []}
        gap_all: dict[int, list[np.ndarray]] = {1: [], 2: [], 3: []}
        for s0 in range(0, len(calib_prompts), cal_bsz):
            _, st = engine.generate_batched(
                calib_prompts[s0 : s0 + cal_bsz], cal_pol,
                max_new_tokens=256 if not is_thinking else 512,
                pad_token_id=pad_id, eos_token_ids=eos_ids, think_end_ids=think_end_ids,
                stop_strings=None if is_thinking else ["\nQuestion:"], tokenizer=tokenizer,
                collect_calib=True,
            )
            torch.cuda.empty_cache()
            for r in (2, 3):
                if len(st["calib_kl"][r]) > 0:
                    kl_all[r].append(st["calib_kl"][r])
            for r in (1, 2, 3):
                if len(st["calib_gate"][r]) > 0:
                    gate_all[r].append(st["calib_gate"][r])
                if len(st["calib_gap"][r]) > 0:
                    gap_all[r].append(st["calib_gap"][r])

        kl_cat = {r: np.concatenate(v) if v else np.array([]) for r, v in kl_all.items()}
        gate_cat = {r: np.concatenate(v) if v else np.array([]) for r, v in gate_all.items()}
        gap_cat = {r: np.concatenate(v) if v else np.array([]) for r, v in gap_all.items()}

        tau_gate_rmin1, mr_g1, p_g1 = calibrate_floor_threshold(gate_cat, mode="gate", r_min=1, n_loops=4, target_mean_r=2.4)
        tau_gate_rmin2, mr_g2, p_g2 = calibrate_floor_threshold(gate_cat, mode="gate", r_min=2, n_loops=4, target_mean_r=2.5)
        tau_kl_25, mr_kl25, p_kl25 = calibrate_floor_threshold(kl_cat, mode="kl", r_min=2, n_loops=4, target_mean_r=2.5)
        tau_gap_25, mr_gap25, p_gap25 = calibrate_floor_threshold(gap_cat, mode="gap", r_min=2, n_loops=4, target_mean_r=2.5)
        calib_info = {
            "tau_gate_rmin1": tau_gate_rmin1, "calib_mr_g1": mr_g1, "p_g1": p_g1,
            "tau_gate_rmin2": tau_gate_rmin2, "calib_mr_g2": mr_g2, "p_g2": p_g2,
            "tau_kl_25": tau_kl_25, "calib_mr_kl25": mr_kl25, "p_kl25": p_kl25,
            "tau_gap_25": tau_gap_25, "calib_mr_gap25": mr_gap25, "p_gap25": p_gap25,
        }
        print(f"[probe3:{mkey}] Calib thresholds: {json.dumps({k: (round(v, 4) if isinstance(v, float) else [round(x, 2) for x in v]) for k, v in calib_info.items()})}", flush=True)
        pc.step(1)

    tasks_out: dict[str, dict[str, Any]] = {}
    for tspec in mspec["tasks"]:
        tname = tspec["task"]
        n_eval = int(tspec["n_eval"])
        max_new = int(tspec["max_new_tokens"])
        bsz = int(tspec["batch_size"])
        if tname == "gsm8k":
            items = load_gsm8k_split(dataset_dir, split="test", n_samples=n_eval, seed=seed)
        else:
            items = load_math500_samples(dataset_dir, n_samples=n_eval, seed=seed, max_level=5)

        policies: list[DepthPolicy] = [
            DepthPolicy(name="native_R1", n_loops=1, exit_mode="fixed"),
            DepthPolicy(name="native_R2", n_loops=2, exit_mode="fixed"),
            DepthPolicy(name="native_R3", n_loops=3, exit_mode="fixed"),
            DepthPolicy(name="native_R4", n_loops=4, exit_mode="fixed"),
            DepthPolicy(name="native_R6", n_loops=6, exit_mode="fixed"),
        ]
        if mkey == "ouro14b":
            policies.append(DepthPolicy(name="native_R8", n_loops=8, exit_mode="fixed"))

        policies.extend([
            DepthPolicy(name="gate_rmin1", n_loops=4, exit_mode="ouro_gate", r_min=1, thresh=tau_gate_rmin1, ragged_kv=True),
            DepthPolicy(name="gate_rmin2", n_loops=4, exit_mode="ouro_gate", r_min=2, thresh=tau_gate_rmin2, ragged_kv=True),
            DepthPolicy(name="kl_rmin2", n_loops=4, exit_mode="kl", r_min=2, thresh=tau_kl_25, ragged_kv=True),
            DepthPolicy(name="gap_rmin2", n_loops=4, exit_mode="gap_agree", r_min=2, thresh=tau_gap_25, ragged_kv=True),
        ])
        if mkey == "ouro14b":
            policies.append(
                DepthPolicy(name="kl_rmin2_fullkv", n_loops=4, exit_mode="kl", r_min=2, thresh=tau_kl_25, ragged_kv=False)
            )
        if is_thinking:
            policies.extend([
                DepthPolicy(name="phase_think2_ans4", n_loops=4, exit_mode="phase_split", think_loops=2, answer_loops=4, ragged_kv=True),
                DepthPolicy(name="phase_think3_ans4", n_loops=4, exit_mode="phase_split", think_loops=3, answer_loops=4, ragged_kv=True),
                DepthPolicy(name="phase_think_kl2_ans4", n_loops=4, exit_mode="phase_kl", r_min=2, thresh=tau_kl_25, ragged_kv=True),
            ])

        t_res: dict[str, Any] = {}
        with run_obj.phase(f"{mkey}_{tname}", total=len(policies), unit="conds") as pt:
            for idx_p, pol in enumerate(policies, 1):
                r = evaluate_policy(
                    engine, items, tokenizer, pol,
                    is_thinking=is_thinking, task=tname,
                    max_new_tokens=max_new, batch_size=bsz,
                )
                t_res[pol.name] = r
                print(
                    f"[probe3:{mkey}:{tname}] {pol.name:22s} R={r['mean_loops']:4.2f} "
                    f"acc={r['accuracy']*100:5.1f}% ({r['n_correct']:3d}/{r['n_total']}) "
                    f"comp={r['completion_rate']*100:3.0f}% tok={r['mean_gen_tokens']:5.0f} "
                    f"t={r['elapsed_sec']:5.1f}s",
                    flush=True,
                )
                pt.row(
                    f"{mkey}_{tname}",
                    policy=pol.name,
                    mean_r=r["mean_loops"],
                    acc=r["accuracy"],
                    mean_toks=r["mean_gen_tokens"],
                    elapsed_s=r["elapsed_sec"],
                )
                pt.step(idx_p)

        # Paired contrasts vs native_R4 and native_R2
        contrasts: dict[str, Any] = {}
        ref4 = np.asarray(t_res["native_R4"]["correct_flags"], dtype=np.float64)
        ref2 = np.asarray(t_res["native_R2"]["correct_flags"], dtype=np.float64)
        for pname, r in t_res.items():
            arr = np.asarray(r["correct_flags"], dtype=np.float64)
            m4, lo4, hi4 = paired_bootstrap_ci(arr - ref4)
            m2, lo2, hi2 = paired_bootstrap_ci(arr - ref2)
            contrasts[pname] = {
                "vs_R4_pp": [m4 * 100, lo4 * 100, hi4 * 100],
                "vs_R2_pp": [m2 * 100, lo2 * 100, hi2 * 100],
            }
        # Strip per-item flags from JSON to keep compact? Keep summary clean
        tasks_out[tname] = {"results": {k: {kk: vv for kk, vv in v.items() if kk != "correct_flags"} for k, v in t_res.items()}, "contrasts": contrasts}

    del engine, model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model_key": mkey,
        "is_thinking": is_thinking,
        "verify": verify_res,
        "calib": calib_info,
        "tasks": tasks_out,
    }


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Round 1 Probe 3: Ouro & Ouro-Thinking Depth Scaling and Floor-Protected Recurrence",
        "",
        f"- GPU: `{summary['gpu_index']}` ({summary['gpu_name']}), elapsed `{summary['elapsed_sec']:.1f}s`",
        "",
    ]
    for mkey, mres in summary["models"].items():
        lines.append(f"## Model: `{mkey}` (is_thinking={mres['is_thinking']})")
        for tname, tdata in mres["tasks"].items():
            lines.append(f"### Task: `{tname}`")
            lines.append("| Policy | Mean R | Acc (%) | Δ vs R=4 (95% CI pp) | Comp% | Mean Tok | Think Tok | Time (s) |")
            lines.append("|---|---:|---:|---|---:|---:|---:|---:|")
            for pname, r in tdata["results"].items():
                d4, lo4, hi4 = tdata["contrasts"][pname]["vs_R4_pp"]
                lines.append(
                    f"| `{pname}` | {r['mean_loops']:.2f} | {r['accuracy']*100:.1f}% ({r['n_correct']}/{r['n_total']}) | "
                    f"{d4:+.1f} [{lo4:+.1f}, {hi4:+.1f}] | {r['completion_rate']*100:.0f}% | "
                    f"{r['mean_gen_tokens']:.0f} | {r['mean_think_tokens']:.0f} | {r['elapsed_sec']:.1f} |"
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
    gpu_idx, gpu_lock, gpu_status = acquire_idle_gpu(cfg.get("preferred_gpu"))
    try:
        print(
            f"[probe3] Acquired idle GPU {gpu_idx}: {gpu_status.name} "
            f"(mem={gpu_status.mem_used_mib}MiB, util={gpu_status.util_pct}%)",
            flush=True,
        )
        device = torch.device(f"cuda:{gpu_idx}")
        torch.cuda.set_device(device)

        plan = [
            {"phase": "setup", "of": 1, "unit": "env"},
            {"phase": "download", "of": 1, "unit": "assets"},
        ]
        for mspec in cfg["models"]:
            mk = mspec["key"]
            plan.append({"phase": f"{mk}_verify", "of": 1, "unit": "check"})
            plan.append({"phase": f"{mk}_calib", "of": 1, "unit": "pass"})
            for tspec in mspec["tasks"]:
                n_pol = 11 if mk == "ouro14b" else 12
                plan.append({"phase": f"{mk}_{tspec['task']}", "of": n_pol, "unit": "conds"})
        plan.append({"phase": "report", "of": 1, "unit": "file"})
        orx.plan(plan)
        run_obj = orx.progress(total=1, unit="run")

        models_out: dict[str, Any] = {}
        for mspec in cfg["models"]:
            models_out[mspec["key"]] = run_model_probe3(
                mspec,
                model_dir=Path(args.model_dir),
                dataset_dir=Path(args.dataset_dir),
                cfg=cfg,
                device=device,
                run_obj=run_obj,
            )

        summary = {
            "idea": "probe3_thinking_depth",
            "gpu_index": gpu_idx,
            "gpu_name": gpu_status.name,
            "elapsed_sec": time.time() - t0,
            "config": cfg,
            "models": models_out,
        }
        with run_obj.phase("report", total=1, unit="file") as pr:
            (out_dir / "summary_probe3_depth.json").write_text(json.dumps(summary, indent=2))
            md = render_report(summary)
            (out_dir / "report_probe3_depth.md").write_text(md)
            print("========== REPORT (report_probe3_depth.md) ==========", flush=True)
            print(md, flush=True)
            print("========== END REPORT ==========", flush=True)

            headline: dict[str, float] = {}
            if "ouro14b" in models_out and "gsm8k" in models_out["ouro14b"]["tasks"]:
                r14 = models_out["ouro14b"]["tasks"]["gsm8k"]["results"]
                headline["o14_r4_acc"] = r14["native_R4"]["accuracy"]
                headline["o14_kl_rmin2_acc"] = r14["kl_rmin2"]["accuracy"]
                headline["o14_gate_rmin1_acc"] = r14["gate_rmin1"]["accuracy"]
            if "ouro14b_thinking" in models_out and "math500" in models_out["ouro14b_thinking"]["tasks"]:
                rm = models_out["ouro14b_thinking"]["tasks"]["math500"]["results"]
                headline["o14t_math_r4_acc"] = rm["native_R4"]["accuracy"]
                headline["o14t_math_r6_acc"] = rm["native_R6"]["accuracy"]
                headline["o14t_math_kl2_acc"] = rm["kl_rmin2"]["accuracy"]
            if "ouro26b_thinking" in models_out and "math500" in models_out["ouro26b_thinking"]["tasks"]:
                rm2 = models_out["ouro26b_thinking"]["tasks"]["math500"]["results"]
                headline["o26t_math_r4_acc"] = rm2["native_R4"]["accuracy"]
                headline["o26t_math_r6_acc"] = rm2["native_R6"]["accuracy"]
                headline["o26t_math_kl2_acc"] = rm2["kl_rmin2"]["accuracy"]
            pr.step(1, **headline)
    finally:
        gpu_lock.release()


if __name__ == "__main__":
    main()
