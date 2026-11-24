"""Direction 1 Feasibility Pilot: Which tokens deserve extra loop depth for future KV use?

Implements the exact preliminary study on frozen ByteDance/Ouro-1.4B (4 loops):
  0. Exactness verification: custom looped + Ragged-KV engine vs native HF OuroForCausalLM
     (both teacher-forced logits and KV-cached greedy decode).
  1. Probe 1A (Paired Synthetic Two-Hop QA): 48 base docs x 2 (orig + position-swapped twin)
     x 2 paired questions (Q_A, Q_B on identical prefix) = 192 QA evaluations.
     Exhaustive 28 combos (2 deep blocks of 8, 6 shallow at 2 loops w/ Ragged KV fallback,
     question+answer suffix always 4 loops) + all-shallow (0/8) + all-deep (8/8).
  2. Probe 1B (Natural Multi-Hop QA: framolfese/2WikiMultihopQA): 96 validation items,
     exhaustive 28 combos + all-shallow + all-deep, Answer NLL & Greedy EM/F1.
  3. Probe 1C (Current-Prediction Gain vs Future KV-Write Gain): 32 WikiText + 32 HumanEval
     Python chunks (L=512, 8 blocks of 64 tokens), measuring Delta_curr(b) vs Delta_write(b)
     at near (+1..32), mid (+33..128), far (+129..) horizons.
  4. Strictly causal online baselines & document-grouped CV linear selector on Probe 1A & 1B.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from loop_research import orx_run as orx
from loop_research.data_assets import (
    build_disjoint_text_code_splits,
    generate_synthetic_paired_twohop,
    load_2wiki_samples,
    map_char_spans_to_blocks,
    normalize_answer_text,
)
from loop_research.gpu_utils import acquire_idle_gpu


COMBOS_2_OF_8: list[tuple[int, int]] = list(itertools.combinations(range(8), 2))


def spearman_corr(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.size < 3 or np.std(xa) < 1e-12 or np.std(ya) < 1e-12:
        return 0.0
    rx = np.argsort(np.argsort(xa)).astype(np.float64)
    ry = np.argsort(np.argsort(ya)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / denom) if denom > 1e-12 else 0.0


def pearson_corr(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.size < 3 or np.std(xa) < 1e-12 or np.std(ya) < 1e-12:
        return 0.0
    xa = xa - xa.mean()
    ya = ya - ya.mean()
    denom = float(np.sqrt((xa * xa).sum() * (ya * ya).sum()))
    return float((xa * ya).sum() / denom) if denom > 1e-12 else 0.0


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


def split_prefix_into_blocks(prefix_len: int, n_blocks: int = 8) -> list[tuple[int, int]]:
    """Partition [0, prefix_len) into n_blocks contiguous nearly-equal token spans."""
    spans: list[tuple[int, int]] = []
    for b in range(n_blocks):
        s = (b * prefix_len) // n_blocks
        e = ((b + 1) * prefix_len) // n_blocks
        spans.append((s, e))
    return spans


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # q, k: [B, H, L, D]; cos, sin: [1, L, D] -> [1, 1, L, D]
    c = cos.unsqueeze(1)
    s = sin.unsqueeze(1)
    return (q * c) + (_rotate_half(q) * s), (k * c) + (_rotate_half(k) * s)


class OuroRaggedEngine:
    """Exact Ouro-1.4B looped runner with batched Ragged-KV fallback and cached 4-loop decode."""

    def __init__(self, model, device: torch.device) -> None:
        self.model = model
        self.core = model.model
        self.lm_head = model.lm_head
        self.device = device
        self.num_layers = int(model.config.num_hidden_layers)
        self.total_ut = int(getattr(model.config, "total_ut_steps", 4))
        self.num_heads = int(model.config.num_attention_heads)
        self.num_kv_heads = int(model.config.num_key_value_heads)
        self.head_dim = int(getattr(model.config, "head_dim", model.config.hidden_size // self.num_heads))
        self.enable_gqa = self.num_heads != self.num_kv_heads

    @torch.inference_mode()
    def _layer_step_prefill(
        self,
        layer,
        h: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        active_mask: torch.Tensor | None = None,
        fallback_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_kv: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Run one OuroDecoderLayer over [B, L, D] with optional Ragged-KV fallback for inactive tokens."""
        bsz, seq_len, _ = h.shape
        attn = layer.self_attn
        residual = h
        x = layer.input_layernorm(h)

        q = attn.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = attn.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = attn.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)

        if active_mask is not None and fallback_kv is not None:
            m4 = active_mask[:, None, :, None]
            k_eff = torch.where(m4, k, fallback_kv[0])
            v_eff = torch.where(m4, v, fallback_kv[1])
        else:
            k_eff, v_eff = k, v

        out = F.scaled_dot_product_attention(
            q,
            k_eff,
            v_eff,
            dropout_p=0.0,
            is_causal=True,
            scale=attn.scaling,
            enable_gqa=self.enable_gqa,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.num_heads * self.head_dim)
        out = attn.o_proj(out)
        h = residual + layer.input_layernorm_2(out)

        res2 = h
        h = res2 + layer.post_attention_layernorm_2(layer.mlp(layer.post_attention_layernorm(h)))
        return h, ((k_eff, v_eff) if return_kv else None)

    @torch.inference_mode()
    def run_shallow_loops(
        self,
        input_ids: torch.Tensor,
        shallow_loops: int = 2,
    ) -> dict[str, Any]:
        """Run ut=0..shallow_loops-1 once for [1, L], capturing hidden states, exit gates, and KV caches."""
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=self.device, dtype=torch.long).unsqueeze(0)
        h = self.core.embed_tokens(input_ids)
        cos, sin = self.core.rotary_emb(h, position_ids)

        loop_hiddens: list[torch.Tensor] = []
        loop_gates: list[torch.Tensor] = []
        loop_kvs: list[list[tuple[torch.Tensor, torch.Tensor]]] = []

        for _ut in range(shallow_loops):
            kvs_this_ut: list[tuple[torch.Tensor, torch.Tensor]] = []
            for layer in self.core.layers:
                h, kv = self._layer_step_prefill(layer, h, cos, sin, return_kv=True)
                kvs_this_ut.append(kv)
            h = self.core.norm(h)
            loop_hiddens.append(h)
            loop_gates.append(self.core.early_exit_gate(h).squeeze(-1))
            loop_kvs.append(kvs_this_ut)

        return {
            "cos": cos,
            "sin": sin,
            "loop_hiddens": loop_hiddens,
            "loop_gates": loop_gates,
            "loop_kvs": loop_kvs,
        }

    @torch.inference_mode()
    def run_late_loops_batched(
        self,
        shallow_state: dict[str, Any],
        active_masks: torch.Tensor,
        *,
        shallow_loops: int = 2,
        return_final_kv: bool = False,
    ) -> tuple[torch.Tensor, list[list[tuple[torch.Tensor, torch.Tensor]]] | None]:
        """From cached shallow_state, run ut=shallow_loops..total_ut-1 for a batch of [B, L] active masks.

        Inactive tokens (active_masks == False) freeze their hidden state and expose their
        loop-(shallow_loops-1) KV cache at the same physical layer (exact Ragged KV fallback).
        """
        bsz, seq_len = active_masks.shape
        cos = shallow_state["cos"]
        sin = shallow_state["sin"]
        h_shallow = shallow_state["loop_hiddens"][shallow_loops - 1]  # [1, L, D]
        fallback_kvs = shallow_state["loop_kvs"][shallow_loops - 1]

        h = h_shallow.expand(bsz, seq_len, -1).clone()
        mask3 = active_masks[:, :, None]
        late_kvs: list[list[tuple[torch.Tensor, torch.Tensor]]] = []

        for ut in range(shallow_loops, self.total_ut):
            kvs_this_ut: list[tuple[torch.Tensor, torch.Tensor]] = []
            for layer_idx, layer in enumerate(self.core.layers):
                h_next, kv = self._layer_step_prefill(
                    layer,
                    h,
                    cos,
                    sin,
                    active_mask=active_masks,
                    fallback_kv=fallback_kvs[layer_idx],
                    return_kv=return_final_kv,
                )
                # Freeze inactive tokens' state cleanly
                h = torch.where(mask3, h_next, h)
                if return_final_kv:
                    kvs_this_ut.append(kv)
            h_norm = self.core.norm(h)
            h = torch.where(mask3, h_norm, h_shallow)
            if return_final_kv:
                late_kvs.append(kvs_this_ut)

        return h, (late_kvs if return_final_kv else None)

    @torch.inference_mode()
    def greedy_decode_from_kv(
        self,
        first_logits: torch.Tensor,
        shallow_kvs: list[list[tuple[torch.Tensor, torch.Tensor]]],
        late_kvs: list[list[tuple[torch.Tensor, torch.Tensor]]],
        prompt_len: int,
        *,
        max_new_tokens: int = 10,
        eos_token_id: int | None = None,
    ) -> list[list[int]]:
        """Run exact 4-loop batched greedy decoding for [B] masks over their intervened prefix KV caches."""
        bsz = first_logits.shape[0]
        next_tok = first_logits.argmax(dim=-1)  # [B]
        generated = [next_tok.tolist()]

        # Build mutable per-(ut, layer) KV lists on GPU; expand shallow KVs from [1, H, L, D] -> [B, H, L, D]
        all_k: list[list[torch.Tensor]] = []
        all_v: list[list[torch.Tensor]] = []
        for ut_kvs in shallow_kvs:
            all_k.append([kv[0].expand(bsz, -1, -1, -1).contiguous() for kv in ut_kvs])
            all_v.append([kv[1].expand(bsz, -1, -1, -1).contiguous() for kv in ut_kvs])
        for ut_kvs in late_kvs:
            all_k.append([kv[0].contiguous() for kv in ut_kvs])
            all_v.append([kv[1].contiguous() for kv in ut_kvs])

        cur_pos = prompt_len
        for _step in range(max_new_tokens - 1):
            pos_ids = torch.tensor([[cur_pos]], device=self.device, dtype=torch.long)
            h = self.core.embed_tokens(next_tok.unsqueeze(1))  # [B, 1, D]
            cos, sin = self.core.rotary_emb(h, pos_ids)

            for ut in range(self.total_ut):
                for l_idx, layer in enumerate(self.core.layers):
                    attn = layer.self_attn
                    residual = h
                    x = layer.input_layernorm(h)
                    q = attn.q_proj(x).view(bsz, 1, self.num_heads, self.head_dim).transpose(1, 2)
                    k_new = attn.k_proj(x).view(bsz, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
                    v_new = attn.v_proj(x).view(bsz, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
                    q, k_new = _apply_rope(q, k_new, cos, sin)

                    k_cat = torch.cat([all_k[ut][l_idx], k_new], dim=2)
                    v_cat = torch.cat([all_v[ut][l_idx], v_new], dim=2)
                    all_k[ut][l_idx] = k_cat
                    all_v[ut][l_idx] = v_cat

                    out = F.scaled_dot_product_attention(
                        q, k_cat, v_cat, dropout_p=0.0, is_causal=False, scale=attn.scaling, enable_gqa=self.enable_gqa
                    )
                    out = out.transpose(1, 2).contiguous().view(bsz, 1, self.num_heads * self.head_dim)
                    h = residual + layer.input_layernorm_2(attn.o_proj(out))
                    h = h + layer.post_attention_layernorm_2(layer.mlp(layer.post_attention_layernorm(h)))
                h = self.core.norm(h)

            logits_step = self.lm_head(h[:, -1, :])  # [B, V]
            next_tok = logits_step.argmax(dim=-1)
            generated.append(next_tok.tolist())
            cur_pos += 1
            if eos_token_id is not None and all(t == eos_token_id for t in generated[-1]):
                break

        # Transpose [steps, B] -> [B, steps]
        return [[generated[s][b] for s in range(len(generated))] for b in range(bsz)]


@torch.inference_mode()
def verify_ouro_exactness(model, engine: OuroRaggedEngine, tokenizer, device: torch.device) -> dict[str, Any]:
    """Verify full-depth custom runner reproduces native HF OuroForCausalLM logits and greedy decode."""
    sample = "Record 01: Clara Vance oversees Project Cobalt, headquartered in Bergen.\nQuestion: City?\nAnswer:"
    ids = tokenizer.encode(sample, add_special_tokens=True)
    input_ids = torch.tensor([ids], device=device, dtype=torch.long)

    hf_out = model(input_ids=input_ids, use_cache=False)
    hf_logits = hf_out.logits.float().cpu()

    sh = engine.run_shallow_loops(input_ids, shallow_loops=2)
    full_mask = torch.ones((1, len(ids)), device=device, dtype=torch.bool)
    h_custom, late_kvs = engine.run_late_loops_batched(sh, full_mask, shallow_loops=2, return_final_kv=True)
    custom_logits = engine.lm_head(h_custom).float().cpu()

    diff = (hf_logits - custom_logits).abs()
    max_diff = float(diff.max().item())
    mean_diff = float(diff.mean().item())
    top1_match = bool((hf_logits[0, -1].argmax() == custom_logits[0, -1].argmax()).item())

    # Verify 4-token greedy decode matches HF step-by-step full forward
    gen_custom = engine.greedy_decode_from_kv(
        engine.lm_head(h_custom[:, -1, :]),
        sh["loop_kvs"],
        late_kvs,
        len(ids),
        max_new_tokens=4,
        eos_token_id=tokenizer.eos_token_id,
    )[0]
    hf_greedy: list[int] = []
    cur = input_ids.clone()
    for _ in range(len(gen_custom)):
        out_step = model(input_ids=cur, use_cache=False)
        nt = int(out_step.logits[0, -1].argmax().item())
        hf_greedy.append(nt)
        cur = torch.cat([cur, torch.tensor([[nt]], device=device)], dim=1)

    decode_exact = gen_custom == hf_greedy
    res = {
        "max_abs_logit_diff": max_diff,
        "mean_abs_logit_diff": mean_diff,
        "last_token_top1_match": top1_match,
        "greedy_4tok_exact_match": decode_exact,
        "custom_4tok": gen_custom,
        "hf_4tok": hf_greedy,
    }
    print(f"[verify] Ouro exactness check: {json.dumps(res)}")
    if max_diff > 5e-2 or not top1_match:
        raise RuntimeError(f"Ouro exactness check FAILED: {res}")
    return res


@torch.inference_mode()
def extract_causal_block_features(
    engine: OuroRaggedEngine,
    shallow_state: dict[str, Any],
    input_ids: torch.Tensor,
    block_spans: list[tuple[int, int]],
) -> list[dict[str, float]]:
    """Compute strictly causal features available at loop 1-2 for each of the 8 prefix blocks.

    Zero access to future blocks or the downstream question!
    """
    h0 = shallow_state["loop_hiddens"][0][0]  # [L, D] loop 1
    h1 = shallow_state["loop_hiddens"][1][0]  # [L, D] loop 2
    g1 = shallow_state["loop_gates"][1][0].float()  # [L] early exit gate logit after loop 2

    prefix_end = block_spans[-1][1]
    logits0 = engine.lm_head(h0[:prefix_end]).float()  # [P, V]
    logits1 = engine.lm_head(h1[:prefix_end]).float()  # [P, V]
    logp0 = F.log_softmax(logits0, dim=-1)
    logp1 = F.log_softmax(logits1, dim=-1)
    p1 = logp1.exp()

    # Strictly causal token-level signals at each position t inside prefix
    entropy1 = -(p1 * logp1).sum(dim=-1)  # [P]
    kl_1_0 = (p1 * (logp1 - logp0)).sum(dim=-1)  # KL(p_loop2 || p_loop1)
    state_delta = (h1[:prefix_end].float() - h0[:prefix_end].float()).norm(dim=-1) / (
        h0[:prefix_end].float().norm(dim=-1).clamp_min(1e-6)
    )
    h1_norm = h1[:prefix_end].float().norm(dim=-1)
    exit_prob = torch.sigmoid(g1[:prefix_end])

    # Causal NLL of observed tokens (predicting token t from t-1, strictly within/up to block)
    targets = input_ids[0, 1:prefix_end]
    nll0_tok = -logp0[:-1].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    nll1_tok = -logp1[:-1].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    nll_gain_1_2 = nll0_tok - nll1_tok  # positive = loop 2 helped predict this token

    feats: list[dict[str, float]] = []
    for b, (s, e) in enumerate(block_spans):
        e_eff = max(s + 1, e)
        nll_s = max(0, s - 1)
        nll_e = max(nll_s + 1, e_eff - 1)
        feats.append(
            {
                "block": float(b),
                "pos_norm": float(b) / 7.0,
                "entropy_mean": float(entropy1[s:e_eff].mean().item()),
                "entropy_max": float(entropy1[s:e_eff].max().item()),
                "kl_1_0_mean": float(kl_1_0[s:e_eff].mean().item()),
                "state_delta_mean": float(state_delta[s:e_eff].mean().item()),
                "exit_continue_score": float((1.0 - exit_prob[s:e_eff]).mean().item()),
                "h1_norm_mean": float(h1_norm[s:e_eff].mean().item()),
                "nll1_mean": float(nll1_tok[nll_s:nll_e].mean().item()),
                "curr_nll_gain_1_2": float(nll_gain_1_2[nll_s:nll_e].mean().item()),
            }
        )
    return feats


SYNTH_2SHOT_HEADER = (
    "Read the records and answer with only the exact city name.\n\n"
    "Example 1:\n"
    "Record 01: Coordinator Alice Chen was appointed to oversee Project Amber.\n"
    "Record 02: Project Amber maintains its primary headquarters in Lisbon.\n"
    "Question: In which city is the project overseen by Alice Chen headquartered?\n"
    "Answer: Lisbon\n\n"
    "Example 2:\n"
    "Record 01: Coordinator David Park was appointed to oversee Project Cedar.\n"
    "Record 02: Project Cedar maintains its primary headquarters in Zurich.\n"
    "Question: In which city is the project overseen by David Park headquartered?\n"
    "Answer: Zurich\n\n"
    "Records:\n"
)

WIKI_2SHOT_HEADER = (
    "Read the passages and give only the short exact answer.\n\n"
    "Example 1:\n"
    "[Riverdale] Riverdale is a film directed by John Ford.\n"
    "[John Ford] John Ford was born in Cape Elizabeth, Maine.\n"
    "Question: Where was the director of Riverdale born?\n"
    "Answer: Cape Elizabeth, Maine\n\n"
    "Example 2:\n"
    "[Alpha Tower] Alpha Tower was designed by architect Maria Lopez.\n"
    "[Maria Lopez] Maria Lopez graduated from Delft University.\n"
    "Question: Which university did the architect of Alpha Tower graduate from?\n"
    "Answer: Delft University\n\n"
    "Passages:\n"
)


def _build_30_masks(
    total_len: int,
    block_spans: list[tuple[int, int]],
    doc_start: int,
    doc_end: int,
    device: torch.device,
) -> tuple[list[str], torch.Tensor]:
    """Return names and [30, L] bool tensor: all_shallow, all_deep, and 28 2-of-8 combos.

    Header/sink tokens [0, doc_start) AND question+answer suffix [doc_end, total_len) are
    ALWAYS 4 loops across all 30 configs so BOS attention sinks never confound block selection.
    """
    names: list[str] = ["all_shallow", "all_deep"] + [f"b{i}_{j}" for i, j in COMBOS_2_OF_8]
    masks = torch.zeros((30, total_len), device=device, dtype=torch.bool)
    masks[:, :doc_start] = True
    masks[:, doc_end:] = True
    # Config 0: all_shallow (only header/sink + suffix active)
    # Config 1: all_deep (entire sequence active)
    masks[1, :] = True
    # Configs 2..29: header/sink + 2 deep document blocks + suffix
    for c_idx, (b1, b2) in enumerate(COMBOS_2_OF_8):
        s1, e1 = block_spans[b1]
        s2, e2 = block_spans[b2]
        masks[2 + c_idx, s1:e1] = True
        masks[2 + c_idx, s2:e2] = True
    return names, masks


def _check_em_and_f1(pred_text: str, gold_norm: str) -> tuple[float, float]:
    first_line = pred_text.strip().splitlines()[0] if pred_text.strip() else ""
    pred_norm = normalize_answer_text(first_line)
    em = 1.0 if (pred_norm == gold_norm or pred_norm.startswith(gold_norm + " ") or gold_norm in pred_norm.split()) else 0.0
    p_toks = pred_norm.split()
    g_toks = gold_norm.split()
    if not p_toks or not g_toks:
        return em, 0.0
    common = sum(min(p_toks.count(w), g_toks.count(w)) for w in set(g_toks))
    if common == 0:
        return em, 0.0
    prec = common / len(p_toks)
    rec = common / len(g_toks)
    f1 = 2 * prec * rec / (prec + rec)
    return em, f1


@torch.inference_mode()
def evaluate_qa_probe(
    engine: OuroRaggedEngine,
    tokenizer,
    samples: list[dict[str, Any]],
    *,
    probe_name: str,
    device: torch.device,
    batch_chunk: int = 15,
    max_new_tokens: int = 10,
    run_phase: Any = None,
) -> list[dict[str, Any]]:
    """Run exhaustive 30-config (all_shallow, all_deep, 28 2-of-8) evaluation on a QA probe."""
    results: list[dict[str, Any]] = []
    # Cache shallow prefix state when consecutive items share identical doc_id (paired Q_A / Q_B)
    prefix_cache_key: str | None = None
    cached_prefix_feats: list[dict[str, float]] | None = None

    header_text = SYNTH_2SHOT_HEADER if probe_name == "synth" else WIKI_2SHOT_HEADER
    header_ids = tokenizer.encode(header_text, add_special_tokens=True)

    for idx, item in enumerate(samples, 1):
        doc_ids = tokenizer.encode(item["prefix_text"], add_special_tokens=False)
        # Clean concise question suffix matching 2-shot examples
        raw_suffix = item["question_suffix"]
        q_pos = raw_suffix.find("Question:")
        clean_suffix = "\n" + raw_suffix[q_pos:] if q_pos >= 0 else raw_suffix
        suffix_ids = tokenizer.encode(clean_suffix, add_special_tokens=False)
        ans_ids = tokenizer.encode(item["gold_answer"], add_special_tokens=False)
        if not ans_ids or len(doc_ids) < 32:
            continue

        doc_start = len(header_ids)
        doc_end = doc_start + len(doc_ids)
        prefix_len = doc_end
        prompt_len = doc_end + len(suffix_ids)
        full_ids = header_ids + doc_ids + suffix_ids + ans_ids
        total_len = len(full_ids)
        rel_spans = split_prefix_into_blocks(len(doc_ids), 8)
        block_spans = [(doc_start + s, doc_start + e) for s, e in rel_spans]

        # Ground-truth evidence blocks
        if "evidence_slots" in item:
            ev_spans = [item["para_spans"][s] for s in item["evidence_slots"]]
        else:
            ev_spans = [item["para_spans"][s] for s in item.get("support_para_indices", [])]
        evidence_blocks = map_char_spans_to_blocks(item["prefix_text"], len(doc_ids), ev_spans, n_blocks=8)

        full_tensor = torch.tensor([full_ids], device=device, dtype=torch.long)
        sh_full = engine.run_shallow_loops(full_tensor, shallow_loops=2)

        # Causal prefix features depend ONLY on prefix tokens (identical for paired Q_A & Q_B!)
        if prefix_cache_key == item["doc_id"] and cached_prefix_feats is not None:
            causal_feats = cached_prefix_feats
        else:
            causal_feats = extract_causal_block_features(engine, sh_full, full_tensor, block_spans)
            prefix_cache_key = item["doc_id"]
            cached_prefix_feats = causal_feats

        # 1) Teacher-forced Answer NLL across all 30 masks
        mask_names, masks_full = _build_30_masks(total_len, block_spans, doc_start, doc_end, device)
        nll_by_cfg: list[float] = []
        target_ans = torch.tensor(ans_ids, device=device, dtype=torch.long)
        for start in range(0, 30, batch_chunk):
            m_sub = masks_full[start : start + batch_chunk]
            h_late, _ = engine.run_late_loops_batched(sh_full, m_sub, shallow_loops=2, return_final_kv=False)
            # Predict ans_ids from positions [prompt_len - 1 : total_len - 1]
            ans_logits = engine.lm_head(h_late[:, prompt_len - 1 : total_len - 1, :]).float()  # [B, A, V]
            log_probs = F.log_softmax(ans_logits, dim=-1)
            tok_nll = -log_probs.gather(-1, target_ans.view(1, -1, 1).expand(m_sub.shape[0], -1, 1)).squeeze(-1)
            nll_by_cfg.extend(tok_nll.mean(dim=-1).tolist())

        # 2) Greedy EM & F1 across all 30 masks (4-loop decode over intervened prefix KV!)
        # Slice sh_full KVs to prompt_len so we don't even re-run shallow loops!
        sh_prompt = {
            "cos": sh_full["cos"][:, :prompt_len, :],
            "sin": sh_full["sin"][:, :prompt_len, :],
            "loop_hiddens": [x[:, :prompt_len, :] for x in sh_full["loop_hiddens"]],
            "loop_gates": [x[:, :prompt_len] for x in sh_full["loop_gates"]],
            "loop_kvs": [[(k[:, :, :prompt_len, :], v[:, :, :prompt_len, :]) for k, v in ut_kv] for ut_kv in sh_full["loop_kvs"]],
        }
        masks_prompt = masks_full[:, :prompt_len]
        em_by_cfg: list[float] = []
        f1_by_cfg: list[float] = []
        preds_sample: dict[str, str] = {}
        for start in range(0, 30, batch_chunk):
            m_sub = masks_prompt[start : start + batch_chunk]
            h_late, late_kvs = engine.run_late_loops_batched(sh_prompt, m_sub, shallow_loops=2, return_final_kv=True)
            first_logits = engine.lm_head(h_late[:, -1, :])
            gen_ids_list = engine.greedy_decode_from_kv(
                first_logits,
                sh_prompt["loop_kvs"],
                late_kvs,
                prompt_len,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
            )
            for j, g_ids in enumerate(gen_ids_list):
                txt = tokenizer.decode(g_ids, skip_special_tokens=True)
                em, f1 = _check_em_and_f1(txt, item["gold_answer_norm"])
                em_by_cfg.append(em)
                f1_by_cfg.append(f1)
                cfg_i = start + j
                if cfg_i in (0, 1):
                    preds_sample[mask_names[cfg_i]] = txt.strip()[:40]

        combo_nlls = nll_by_cfg[2:]
        best_c = int(np.argmin(combo_nlls))
        worst_c = int(np.argmax(combo_nlls))
        rec = {
            "qa_id": item["qa_id"],
            "doc_id": item["doc_id"],
            "base_id": item.get("base_id"),
            "variant": item.get("variant"),
            "which_chain": item.get("which_chain"),
            "qtype": item.get("qtype"),
            "prefix_len": prefix_len,
            "prompt_len": prompt_len,
            "evidence_blocks": evidence_blocks[:2],
            "all_evidence_blocks": evidence_blocks,
            "nll_shallow": nll_by_cfg[0],
            "nll_deep": nll_by_cfg[1],
            "combo_nlls": combo_nlls,
            "em_shallow": em_by_cfg[0],
            "em_deep": em_by_cfg[1],
            "combo_ems": em_by_cfg[2:],
            "f1_shallow": f1_by_cfg[0],
            "f1_deep": f1_by_cfg[1],
            "combo_f1s": f1_by_cfg[2:],
            "oracle_combo_idx": best_c,
            "oracle_blocks": list(COMBOS_2_OF_8[best_c]),
            "worst_blocks": list(COMBOS_2_OF_8[worst_c]),
            "causal_block_features": causal_feats,
            "preds_sample": preds_sample,
        }
        results.append(rec)

        if run_phase is not None:
            mean_sh = float(np.mean([r["nll_shallow"] for r in results]))
            mean_dp = float(np.mean([r["nll_deep"] for r in results]))
            mean_or = float(np.mean([min(r["combo_nlls"]) for r in results]))
            gap = mean_sh - mean_dp
            recov = (mean_sh - mean_or) / gap if abs(gap) > 1e-6 else 0.0
            run_phase.step(
                idx,
                **{
                    f"{probe_name}_gap_nll": gap,
                    f"{probe_name}_oracle_recov": recov,
                    f"{probe_name}_em_deep": float(np.mean([r["em_deep"] for r in results])),
                },
            )
    return results


def _select_top2_combo(scores_8: list[float]) -> int:
    top2 = sorted(sorted(range(8), key=lambda b: scores_8[b], reverse=True)[:2])
    return COMBOS_2_OF_8.index((top2[0], top2[1]))


def analyze_qa_probe(records: list[dict[str, Any]], *, probe_name: str) -> dict[str, Any]:
    """Compute oracle recovery, all strictly causal baselines, paired switching, and CV linear selector."""
    n = len(records)
    sh_nll = np.array([r["nll_shallow"] for r in records])
    dp_nll = np.array([r["nll_deep"] for r in records])
    combo_nlls = np.array([r["combo_nlls"] for r in records])  # [N, 28]
    sh_em = np.array([r["em_shallow"] for r in records])
    dp_em = np.array([r["em_deep"] for r in records])
    combo_ems = np.array([r["combo_ems"] for r in records])  # [N, 28]
    combo_f1s = np.array([r["combo_f1s"] for r in records])

    gap_nll = float((sh_nll - dp_nll).mean())
    gap_em = float((dp_em - sh_em).mean())

    def _summarize_selection(sel_indices: np.ndarray, label: str) -> dict[str, Any]:
        picked_nll = combo_nlls[np.arange(n), sel_indices]
        picked_em = combo_ems[np.arange(n), sel_indices]
        picked_f1 = combo_f1s[np.arange(n), sel_indices]
        mean_nll = float(picked_nll.mean())
        recov = float((sh_nll.mean() - mean_nll) / gap_nll) if abs(gap_nll) > 1e-6 else 0.0
        # Paired CI vs random_2 mean per sample
        rand_per_sample = combo_nlls.mean(axis=1)
        improv_vs_rand, lo, hi = paired_bootstrap_ci(rand_per_sample - picked_nll)
        return {
            "policy": label,
            "nll": mean_nll,
            "nll_recovery": recov,
            "nll_improv_vs_random": improv_vs_rand,
            "nll_improv_vs_random_ci95": [lo, hi],
            "em": float(picked_em.mean()),
            "f1": float(picked_f1.mean()),
        }

    policies: dict[str, dict[str, Any]] = {}
    policies["all_shallow"] = {
        "policy": "all_shallow",
        "nll": float(sh_nll.mean()),
        "nll_recovery": 0.0,
        "em": float(sh_em.mean()),
        "f1": float(np.mean([r["f1_shallow"] for r in records])),
    }
    policies["all_deep"] = {
        "policy": "all_deep",
        "nll": float(dp_nll.mean()),
        "nll_recovery": 1.0,
        "em": float(dp_em.mean()),
        "f1": float(np.mean([r["f1_deep"] for r in records])),
    }
    rand_nll_per = combo_nlls.mean(axis=1)
    policies["random_2"] = {
        "policy": "random_2",
        "nll": float(rand_nll_per.mean()),
        "nll_recovery": float((sh_nll.mean() - rand_nll_per.mean()) / gap_nll) if abs(gap_nll) > 1e-6 else 0.0,
        "em": float(combo_ems.mean()),
        "f1": float(combo_f1s.mean()),
    }

    # Oracle (hindsight best of 28 per sample)
    oracle_idx = np.argmin(combo_nlls, axis=1)
    policies["oracle_2"] = _summarize_selection(oracle_idx, "oracle_2")

    # Worst of 28
    worst_idx = np.argmax(combo_nlls, axis=1)
    policies["worst_2"] = _summarize_selection(worst_idx, "worst_2")

    # Oracle ground-truth evidence blocks (when >=2 evidence blocks mapped)
    ev_indices = []
    ev_valid = 0
    for r in records:
        eb = r["evidence_blocks"]
        if len(eb) >= 2 and eb[0] != eb[1]:
            pair = tuple(sorted(eb[:2]))
            ev_indices.append(COMBOS_2_OF_8.index(pair))
            ev_valid += 1
        elif len(eb) == 1:
            b0 = eb[0]
            b1 = (b0 + 1) % 8
            ev_indices.append(COMBOS_2_OF_8.index(tuple(sorted((b0, b1)))))
        else:
            ev_indices.append(0)
    policies["gold_evidence_2"] = _summarize_selection(np.array(ev_indices), "gold_evidence_2")
    policies["gold_evidence_2"]["valid_2block_fraction"] = ev_valid / max(n, 1)

    # Fixed position baselines
    policies["first_2"] = _summarize_selection(np.full(n, COMBOS_2_OF_8.index((0, 1))), "first_2")
    policies["last_2"] = _summarize_selection(np.full(n, COMBOS_2_OF_8.index((6, 7))), "last_2")

    # Document-grouped 4-fold CV for best_fixed_pair AND causal linear ridge selector
    doc_ids = sorted(set(r["doc_id"] for r in records))
    doc_to_fold = {d: i % 4 for i, d in enumerate(doc_ids)}
    folds = np.array([doc_to_fold[r["doc_id"]] for r in records])

    # Best fixed pair (CV)
    best_fixed_cv = np.zeros(n, dtype=np.int64)
    for f in range(4):
        tr = folds != f
        te = folds == f
        best_c = int(np.argmin(combo_nlls[tr].mean(axis=0)))
        best_fixed_cv[te] = best_c
    policies["best_fixed_pair_cv"] = _summarize_selection(best_fixed_cv, "best_fixed_pair_cv")

    # Causal heuristic baselines (strictly loop-1/2 prefix-only!)
    heuristic_keys = [
        ("top2_entropy", "entropy_mean"),
        ("top2_kl_1_0", "kl_1_0_mean"),
        ("top2_state_delta_flashloop", "state_delta_mean"),
        ("top2_exit_gate", "exit_continue_score"),
        ("top2_curr_nll_gain", "curr_nll_gain_1_2"),
    ]
    for pname, fkey in heuristic_keys:
        sel = np.array(
            [_select_top2_combo([bf[fkey] for bf in r["causal_block_features"]]) for r in records],
            dtype=np.int64,
        )
        policies[pname] = _summarize_selection(sel, pname)

    # Document-grouped 4-fold CV ridge selector predicting marginal block benefit from causal features
    feat_cols = [
        "pos_norm",
        "entropy_mean",
        "entropy_max",
        "kl_1_0_mean",
        "state_delta_mean",
        "exit_continue_score",
        "h1_norm_mean",
        "nll1_mean",
        "curr_nll_gain_1_2",
    ]
    # Compute marginal block value y[i, b] = mean NLL of combos WITHOUT b minus mean NLL of combos WITH b
    block_has = np.zeros((8, 28), dtype=np.bool_)
    for c_i, (b1, b2) in enumerate(COMBOS_2_OF_8):
        block_has[b1, c_i] = True
        block_has[b2, c_i] = True

    X_all = np.zeros((n, 8, len(feat_cols)), dtype=np.float64)
    Y_all = np.zeros((n, 8), dtype=np.float64)
    for i, r in enumerate(records):
        for b in range(8):
            bf = r["causal_block_features"][b]
            for k, col in enumerate(feat_cols):
                X_all[i, b, k] = bf[col]
            Y_all[i, b] = float(combo_nlls[i, ~block_has[b]].mean() - combo_nlls[i, block_has[b]].mean())

    cv_sel = np.zeros(n, dtype=np.int64)
    for f in range(4):
        tr = folds != f
        te = folds == f
        X_tr = X_all[tr].reshape(-1, len(feat_cols))
        y_tr = Y_all[tr].reshape(-1)
        mu = X_tr.mean(axis=0, keepdims=True)
        sd = X_tr.std(axis=0, keepdims=True) + 1e-6
        X_tr_n = np.hstack([(X_tr - mu) / sd, np.ones((X_tr.shape[0], 1))])
        reg = 10.0 * np.eye(X_tr_n.shape[1])
        reg[-1, -1] = 0.0
        w = np.linalg.solve(X_tr_n.T @ X_tr_n + reg, X_tr_n.T @ y_tr)

        for idx_te in np.where(te)[0]:
            X_te_n = np.hstack([(X_all[idx_te] - mu) / sd, np.ones((8, 1))])
            preds_8 = (X_te_n @ w).tolist()
            cv_sel[idx_te] = _select_top2_combo(preds_8)
    policies["causal_ridge_cv"] = _summarize_selection(cv_sel, "causal_ridge_cv")

    # Oracle block frequency histogram (check position bias!)
    oracle_block_counts = [0] * 8
    oracle_ev_overlap = []
    for r in records:
        ob = set(r["oracle_blocks"])
        for b in ob:
            oracle_block_counts[b] += 1
        eb = set(r["evidence_blocks"])
        if eb:
            oracle_ev_overlap.append(len(ob & eb) / len(eb))

    # Paired-question analysis (when same doc_id has both Q_A and Q_B!)
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_doc.setdefault(r["doc_id"], []).append(r)
    paired_stats: dict[str, Any] = {}
    paired_docs = [v for v in by_doc.values() if len(v) == 2]
    if paired_docs:
        same_oracle_exact = 0
        oracle_jaccard = []
        cross_q_transfer_nll_loss = []
        oracle_hits_own_ev = []
        oracle_hits_other_ev = []
        for qa, qb in paired_docs:
            oa, ob = set(qa["oracle_blocks"]), set(qb["oracle_blocks"])
            if oa == ob:
                same_oracle_exact += 1
            oracle_jaccard.append(len(oa & ob) / max(len(oa | ob), 1))
            # If we apply Q_A's oracle blocks to Q_B (and vice versa), how much worse than own oracle?
            idx_a, idx_b = qa["oracle_combo_idx"], qb["oracle_combo_idx"]
            cross_q_transfer_nll_loss.append(
                0.5 * ((qa["combo_nlls"][idx_b] - qa["combo_nlls"][idx_a]) + (qb["combo_nlls"][idx_a] - qb["combo_nlls"][idx_b]))
            )
            eva, evb = set(qa["evidence_blocks"]), set(qb["evidence_blocks"])
            oracle_hits_own_ev.append(0.5 * (len(oa & eva) / 2.0 + len(ob & evb) / 2.0))
            oracle_hits_other_ev.append(0.5 * (len(oa & evb) / 2.0 + len(ob & eva) / 2.0))

        # Position-swap analysis: compare base_id orig vs swapped
        by_base_chain: dict[tuple[Any, Any], dict[str, dict[str, Any]]] = {}
        for r in records:
            if r.get("base_id") is not None and r.get("variant") in ("orig", "swapped"):
                by_base_chain.setdefault((r["base_id"], r["which_chain"]), {})[r["variant"]] = r
        swap_shifts = []
        for pair_map in by_base_chain.values():
            if "orig" in pair_map and "swapped" in pair_map:
                o_orig = set(pair_map["orig"]["oracle_blocks"])
                o_swap = set(pair_map["swapped"]["oracle_blocks"])
                ev_orig = set(pair_map["orig"]["evidence_blocks"])
                ev_swap = set(pair_map["swapped"]["evidence_blocks"])
                swap_shifts.append(
                    {
                        "moved_away_from_old_pos": 1.0 - len(o_swap & o_orig) / 2.0,
                        "orig_hit_ev": len(o_orig & ev_orig) / 2.0,
                        "swap_hit_ev": len(o_swap & ev_swap) / 2.0,
                    }
                )
        paired_stats = {
            "n_paired_docs": len(paired_docs),
            "same_oracle_exact_rate": same_oracle_exact / len(paired_docs),
            "oracle_jaccard_between_QA_QB": float(np.mean(oracle_jaccard)),
            "cross_question_transfer_penalty_nll": float(np.mean(cross_q_transfer_nll_loss)),
            "oracle_overlap_own_evidence": float(np.mean(oracle_hits_own_ev)),
            "oracle_overlap_other_evidence": float(np.mean(oracle_hits_other_ev)),
            "position_swap_moved_rate": float(np.mean([x["moved_away_from_old_pos"] for x in swap_shifts])) if swap_shifts else 0.0,
            "position_swap_hit_rate_orig": float(np.mean([x["orig_hit_ev"] for x in swap_shifts])) if swap_shifts else 0.0,
            "position_swap_hit_rate_swapped": float(np.mean([x["swap_hit_ev"] for x in swap_shifts])) if swap_shifts else 0.0,
        }

    return {
        "probe": probe_name,
        "n_items": n,
        "gap_shallow_minus_deep_nll": gap_nll,
        "gap_deep_minus_shallow_em": gap_em,
        "oracle_block_counts": oracle_block_counts,
        "oracle_evidence_overlap_mean": float(np.mean(oracle_ev_overlap)) if oracle_ev_overlap else 0.0,
        "policies": policies,
        "paired_control": paired_stats,
    }


@torch.inference_mode()
def run_probe_1c_decomposition(
    engine: OuroRaggedEngine,
    chunks: list[dict[str, Any]],
    *,
    device: torch.device,
    run_phase: Any = None,
) -> dict[str, Any]:
    """Probe 1C: Separate Current-Prediction Gain Delta_curr(b) vs Future KV-Write Gain Delta_write(b).

    For each chunk (L=512, 8 blocks of 64 tokens):
      - Full-depth baseline: all 8 blocks run 4 loops.
      - Delta_curr(b): under full-depth history, mean NLL reduction inside block b from loop-2 to loop-4.
      - Delta_write(b) (for b in 0..6): downgrade ONLY block b's late-loop KV (ut=2,3) to loop-2 KV
        while keeping all other blocks at 4 loops; measure NLL increase on subsequent tokens (> block b),
        broken down by near (+1..32), mid (+33..128), and far (+129..).
    """
    rows: list[dict[str, Any]] = []
    sink_tokens = 16
    for idx, ch in enumerate(chunks, 1):
        toks = ch["tokens"]
        seq_len = len(toks)
        rel_spans = split_prefix_into_blocks(seq_len - sink_tokens, 8)
        block_spans = [(sink_tokens + s, sink_tokens + e) for s, e in rel_spans]
        input_ids = torch.tensor([toks], device=device, dtype=torch.long)
        targets = input_ids[0, 1:]  # [L-1]

        sh = engine.run_shallow_loops(input_ids, shallow_loops=2)
        # 8 masks in ONE batch: mask 0 = full-depth (all True); mask 1..7 = downgrade block b in 0..6
        masks = torch.ones((8, seq_len), device=device, dtype=torch.bool)
        for b in range(7):
            s, e = block_spans[b]
            masks[1 + b, s:e] = False

        h_late, _ = engine.run_late_loops_batched(sh, masks, shallow_loops=2, return_final_kv=False)
        # Compute per-token NLL for full-depth (index 0) and each leave-one-block-shallow intervention (1..7)
        # Plus loop-2 readout NLL for Delta_curr
        h_loop2 = sh["loop_hiddens"][1][0]  # [L, D]
        logits_l2 = engine.lm_head(h_loop2[:-1]).float()
        nll_l2 = F.cross_entropy(logits_l2, targets, reduction="none")  # [L-1]

        nlls_late = []
        for m_i in range(8):
            logits_m = engine.lm_head(h_late[m_i, :-1]).float()
            nlls_late.append(F.cross_entropy(logits_m, targets, reduction="none"))
        nll_full = nlls_late[0]  # [L-1]

        causal_feats = extract_causal_block_features(engine, sh, input_ids, block_spans)

        for b in range(7):
            s, e = block_spans[b]
            # Current prediction gain inside block b (tokens predicted within [s, e))
            p_start = max(0, s - 1)
            p_end = max(p_start + 1, e - 1)
            delta_curr = float((nll_l2[p_start:p_end] - nll_full[p_start:p_end]).mean().item())

            # Future KV-write gain: NLL increase on tokens STRICTLY AFTER block b (positions >= e, predicted from >= e-1)
            # Note: predicting token e uses position e-1 (which is the last token of block b, so its hidden state was frozen if we use e-1!).
            # Per research-directions.md §1 ("从预测 x_{i+2} 开始计"), start from predicting token e+1 (pred index e) so ONLY KV-read effects count!
            fut_start = e
            fut_diff = nlls_late[1 + b][fut_start:] - nll_full[fut_start:]  # [L - 1 - e]
            n_fut = int(fut_diff.numel())
            if n_fut <= 0:
                continue
            delta_write_all = float(fut_diff.mean().item())
            delta_write_near = float(fut_diff[: min(32, n_fut)].mean().item())
            delta_write_mid = float(fut_diff[32 : min(128, n_fut)].mean().item()) if n_fut > 32 else None
            delta_write_far = float(fut_diff[128:].mean().item()) if n_fut > 128 else None

            rows.append(
                {
                    "chunk_idx": idx - 1,
                    "domain": ch["domain"],
                    "block": b,
                    "delta_curr": delta_curr,
                    "delta_write_all": delta_write_all,
                    "delta_write_near": delta_write_near,
                    "delta_write_mid": delta_write_mid,
                    "delta_write_far": delta_write_far,
                    "nll_full_block": float(nll_full[p_start:p_end].mean().item()),
                    "entropy_mean": causal_feats[b]["entropy_mean"],
                    "state_delta_mean": causal_feats[b]["state_delta_mean"],
                    "exit_continue_score": causal_feats[b]["exit_continue_score"],
                }
            )
        if run_phase is not None:
            run_phase.step(idx)

    # Compute overall & per-domain & within-block-position correlations!
    def _corr_block(sub_rows: list[dict[str, Any]]) -> dict[str, Any]:
        dc = [r["delta_curr"] for r in sub_rows]
        dw = [r["delta_write_all"] for r in sub_rows]
        dn = [r["delta_write_near"] for r in sub_rows]
        ent = [r["entropy_mean"] for r in sub_rows]
        sd = [r["state_delta_mean"] for r in sub_rows]
        # Quadrant analysis: low delta_curr (< median) AND high delta_write (> 75th percentile)
        dc_arr = np.array(dc)
        dw_arr = np.array(dw)
        low_curr_high_write = float(np.mean((dc_arr < np.median(dc_arr)) & (dw_arr > np.percentile(dw_arr, 75))))
        return {
            "n_blocks": len(sub_rows),
            "mean_delta_curr": float(dc_arr.mean()),
            "mean_delta_write_all": float(dw_arr.mean()),
            "mean_delta_write_near": float(np.mean(dn)),
            "spearman_curr_vs_write": spearman_corr(dc, dw),
            "pearson_curr_vs_write": pearson_corr(dc, dw),
            "spearman_entropy_vs_write": spearman_corr(ent, dw),
            "spearman_state_delta_vs_write": spearman_corr(sd, dw),
            "frac_low_curr_high_write_quadrant": low_curr_high_write,
        }

    # Position-controlled (residualized by block index b) correlation
    by_b: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        by_b.setdefault(r["block"], []).append(r)
    dc_resid, dw_resid = [], []
    for b, br in by_b.items():
        mc = float(np.mean([x["delta_curr"] for x in br]))
        mw = float(np.mean([x["delta_write_all"] for x in br]))
        for x in br:
            dc_resid.append(x["delta_curr"] - mc)
            dw_resid.append(x["delta_write_all"] - mw)

    summary = {
        "overall": _corr_block(rows),
        "text": _corr_block([r for r in rows if r["domain"] == "text"]),
        "code": _corr_block([r for r in rows if r["domain"] == "code"]),
        "position_residualized_spearman_curr_vs_write": spearman_corr(dc_resid, dw_resid),
        "position_residualized_pearson_curr_vs_write": pearson_corr(dc_resid, dw_resid),
        "per_block_means": {
            str(b): {
                "delta_curr": float(np.mean([x["delta_curr"] for x in br])),
                "delta_write_all": float(np.mean([x["delta_write_all"] for x in br])),
                "delta_write_near": float(np.mean([x["delta_write_near"] for x in br])),
                "delta_write_mid": float(np.mean([x["delta_write_mid"] for x in br if x["delta_write_mid"] is not None])) if any(x["delta_write_mid"] is not None for x in br) else None,
                "delta_write_far": float(np.mean([x["delta_write_far"] for x in br if x["delta_write_far"] is not None])) if any(x["delta_write_far"] is not None for x in br) else None,
            }
            for b, br in sorted(by_b.items())
        },
    }
    return {"summary": summary, "rows": rows}


def render_report_md(summary: dict[str, Any]) -> str:
    lines = ["# Direction 1 Feasibility Report: Token KV Depth Value (Ouro-1.4B)", ""]
    lines.append(f"- **Model**: `{summary['model_path']}` (bf16, 4 loops, Ragged KV fallback at ut=2,3)")
    lines.append(f"- **Exactness Check**: max abs logit diff = `{summary['verify']['max_abs_logit_diff']:.3e}`, top1 match = `{summary['verify']['last_token_top1_match']}`, 4-tok greedy exact = `{summary['verify']['greedy_4tok_exact_match']}`")
    lines.append(f"- **Elapsed**: `{summary['elapsed_sec']:.1f}s` on GPU `{summary['gpu_index']}` ({summary['gpu_name']})")
    lines.append("")

    for pkey in ("probe_1a_synth", "probe_1b_2wiki"):
        p = summary[pkey]
        lines.append(f"## {p['probe']} (n={p['n_items']})")
        lines.append(
            f"- Shallow→Deep NLL gap: **{p['gap_shallow_minus_deep_nll']:.4f}** | "
            f"EM gap: **{p['gap_deep_minus_shallow_em']*100:+.2f} pp** | "
            f"Oracle evidence overlap: **{p['oracle_evidence_overlap_mean']*100:.1f}%**"
        )
        lines.append(f"- Oracle block frequency (b0..b7): `{p['oracle_block_counts']}`")
        if p["paired_control"]:
            pc = p["paired_control"]
            lines.append(
                f"- **Paired Q_A/Q_B Control** (n_docs={pc['n_paired_docs']}): "
                f"same-oracle rate = `{pc['same_oracle_exact_rate']*100:.1f}%`, "
                f"Jaccard(Q_A,Q_B) = `{pc['oracle_jaccard_between_QA_QB']:.3f}`, "
                f"cross-Q transfer NLL penalty = `{pc['cross_question_transfer_penalty_nll']:+.4f}`, "
                f"oracle hit own-ev = `{pc['oracle_overlap_own_evidence']*100:.1f}%` vs other-ev = `{pc['oracle_overlap_other_evidence']*100:.1f}%`, "
                f"position-swap moved rate = `{pc['position_swap_moved_rate']*100:.1f}%`"
            )
        lines.append("")
        lines.append("| Policy | NLL (↓) | Recovery (↑) | Δ vs Rand (95% CI) | Greedy EM (↑) | Greedy F1 (↑) |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        order = [
            "all_shallow", "all_deep", "oracle_2", "gold_evidence_2", "random_2", "worst_2",
            "first_2", "last_2", "best_fixed_pair_cv",
            "top2_entropy", "top2_kl_1_0", "top2_state_delta_flashloop",
            "top2_exit_gate", "top2_curr_nll_gain", "causal_ridge_cv",
        ]
        for pol_name in order:
            if pol_name not in p["policies"]:
                continue
            d = p["policies"][pol_name]
            ci = d.get("nll_improv_vs_random_ci95")
            ci_str = f"{d['nll_improv_vs_random']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "—"
            lines.append(
                f"| `{pol_name}` | {d['nll']:.4f} | {d['nll_recovery']*100:6.1f}% | {ci_str} | {d['em']*100:5.1f}% | {d['f1']*100:5.1f}% |"
            )
        lines.append("")

    p1c = summary["probe_1c_decomp"]
    lines.append("## Probe 1C: Current Gain (Δ_curr) vs Future KV-Write Gain (Δ_write)")
    for dom in ("overall", "text", "code"):
        d = p1c[dom]
        lines.append(
            f"- **{dom}** (n={d['n_blocks']} blocks): mean Δ_curr=`{d['mean_delta_curr']:.4f}`, "
            f"mean Δ_write=`{d['mean_delta_write_all']:.4f}` (near=`{d['mean_delta_write_near']:.4f}`), "
            f"Spearman(Δ_curr, Δ_write)=`{d['spearman_curr_vs_write']:+.3f}`, "
            f"Pearson=`{d['pearson_curr_vs_write']:+.3f}`, "
            f"Spearman(entropy, Δ_write)=`{d['spearman_entropy_vs_write']:+.3f}`, "
            f"Spearman(state_delta, Δ_write)=`{d['spearman_state_delta_vs_write']:+.3f}`, "
            f"low-curr & top25%-write frac=`{d['frac_low_curr_high_write_quadrant']*100:.1f}%`"
        )
    lines.append(
        f"- **Position-residualized** Spearman(Δ_curr, Δ_write)=`{p1c['position_residualized_spearman_curr_vs_write']:+.3f}`, "
        f"Pearson=`{p1c['position_residualized_pearson_curr_vs_write']:+.3f}`"
    )
    lines.append("")
    return "\n".join(lines)


def load_ouro_for_transformers5(model_path: Path, device: torch.device):
    """Load ByteDance/Ouro-1.4B cleanly under both transformers 4.x and 5.x."""
    import inspect
    import sys
    from transformers import AutoConfig, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    def _default_rope_init(config, device=None, seq_len=None):
        base = getattr(config, "rope_theta", 1000000.0)
        dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, 1.0

    if "default" not in ROPE_INIT_FUNCTIONS:
        ROPE_INIT_FUNCTIONS["default"] = _default_rope_init

    cfg = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    if not hasattr(cfg, "pad_token_id"):
        cfg.pad_token_id = None

    cls = get_class_from_dynamic_module("modeling_ouro.OuroForCausalLM", str(model_path))
    mod = sys.modules[cls.__module__]
    if not hasattr(mod.OuroRotaryEmbedding, "compute_default_rope_parameters"):
        mod.OuroRotaryEmbedding.compute_default_rope_parameters = staticmethod(_default_rope_init)

    orig_ccm = mod.create_causal_mask
    sig = inspect.signature(orig_ccm)

    def _compat_ccm(**kw):
        if "input_embeds" in kw and "inputs_embeds" not in kw and "inputs_embeds" in sig.parameters:
            kw["inputs_embeds"] = kw.pop("input_embeds")
        filtered = {k: v for k, v in kw.items() if k in sig.parameters}
        return orig_ccm(**filtered)

    mod.create_causal_mask = _compat_ccm

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    try:
        model = cls.from_pretrained(
            str(model_path),
            config=cfg,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    except TypeError:
        model = cls.from_pretrained(
            str(model_path),
            config=cfg,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    model = model.to(device).eval()
    return model, tokenizer


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

    gpu_idx, gpu_lock, gpu_info = acquire_idle_gpu(cfg.get("preferred_gpu"))
    try:
        device = torch.device(f"cuda:{gpu_idx}")
        torch.cuda.set_device(device)
        print(f"[idea1] Acquired idle GPU {gpu_idx}: {gpu_info.name} (mem={gpu_info.mem_used_mib}MiB, util={gpu_info.util_pct}%)")

        model_path = Path(args.model_dir) / "ByteDance__Ouro-1.4B"
        print(f"[idea1] Loading Ouro-1.4B from {model_path} ...")
        model, tokenizer = load_ouro_for_transformers5(model_path, device)

        engine = OuroRaggedEngine(model, device)

        n_base_docs = int(cfg.get("n_synth_base_docs", 48))
        n_2wiki = int(cfg.get("n_2wiki_samples", 96))
        n_1c = int(cfg.get("n_decomp_chunks", 64))
        batch_chunk = int(cfg.get("batch_chunk", 15))
        max_new_tokens = int(cfg.get("max_new_tokens", 10))
        seed = int(cfg.get("seed", 20261124))

        synth_samples = generate_synthetic_paired_twohop(n_base_docs=n_base_docs, seed=seed)
        wiki_samples = load_2wiki_samples(
            args.dataset_dir,
            tokenizer,
            n_samples=n_2wiki,
            min_prefix_tokens=int(cfg.get("min_prefix_tokens", 240)),
            max_prefix_tokens=int(cfg.get("max_prefix_tokens", 900)),
            seed=seed,
        )
        splits_1c = build_disjoint_text_code_splits(
            args.dataset_dir,
            lambda t: tokenizer.encode(t, add_special_tokens=False),
            seq_len=int(cfg.get("decomp_seq_len", 512)),
            n_calib_a=n_1c // 2,
            n_calib_b=n_1c // 2,
            n_adapt_train=8,
            n_eval_heldout=8,
            seed=seed,
        )
        decomp_chunks = splits_1c["calib_a"] + splits_1c["calib_b"]

        orx.plan(
            [
                {"phase": "setup", "of": 1, "unit": "env"},
                {"phase": "download", "of": 1, "unit": "assets"},
                {"phase": "verify", "of": 1, "unit": "check"},
                {"phase": "probe_1a_synth", "of": len(synth_samples), "unit": "qas"},
                {"phase": "probe_1b_2wiki", "of": len(wiki_samples), "unit": "qas"},
                {"phase": "probe_1c_decomp", "of": len(decomp_chunks), "unit": "chunks"},
                {"phase": "report", "of": 1, "unit": "file"},
            ]
        )

        run = orx.progress(total=1, unit="check", phase="verify")
        verify_res = verify_ouro_exactness(model, engine, tokenizer, device)
        run.step(1, verify_max_diff=verify_res["max_abs_logit_diff"])

        with run.phase("probe_1a_synth", total=len(synth_samples), unit="qas") as p1a:
            synth_records = evaluate_qa_probe(
                engine,
                tokenizer,
                synth_samples,
                probe_name="synth",
                device=device,
                batch_chunk=batch_chunk,
                max_new_tokens=max_new_tokens,
                run_phase=p1a,
            )
        synth_summary = analyze_qa_probe(synth_records, probe_name="probe_1a_synth")

        with run.phase("probe_1b_2wiki", total=len(wiki_samples), unit="qas") as p1b:
            wiki_records = evaluate_qa_probe(
                engine,
                tokenizer,
                wiki_samples,
                probe_name="wiki",
                device=device,
                batch_chunk=batch_chunk,
                max_new_tokens=max_new_tokens,
                run_phase=p1b,
            )
        wiki_summary = analyze_qa_probe(wiki_records, probe_name="probe_1b_2wiki")

        with run.phase("probe_1c_decomp", total=len(decomp_chunks), unit="chunks") as p1c:
            decomp_out = run_probe_1c_decomposition(engine, decomp_chunks, device=device, run_phase=p1c)

        with run.phase("report", total=1, unit="file") as prep:
            summary = {
                "idea": "idea1_kv_depth",
                "model_path": str(model_path),
                "gpu_index": gpu_idx,
                "gpu_name": gpu_info.name,
                "elapsed_sec": time.time() - t0,
                "config": cfg,
                "verify": verify_res,
                "probe_1a_synth": synth_summary,
                "probe_1b_2wiki": wiki_summary,
                "probe_1c_decomp": decomp_out["summary"],
            }
            (out_dir / "summary_idea1.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
            (out_dir / "records_synth.json").write_text(json.dumps(synth_records, ensure_ascii=False))
            (out_dir / "records_2wiki.json").write_text(json.dumps(wiki_records, ensure_ascii=False))
            (out_dir / "decomp_rows.json").write_text(json.dumps(decomp_out["rows"], ensure_ascii=False))

            report_md = render_report_md(summary)
            (out_dir / "report_idea1.md").write_text(report_md)

            # Emit policy rows for the orx view table & headline summary signals
            for pname, pdata in synth_summary["policies"].items():
                prep.row(
                    "synth_policies",
                    policy=pname,
                    nll=pdata["nll"],
                    recovery=pdata["nll_recovery"],
                    em=pdata["em"],
                )
            for pname, pdata in wiki_summary["policies"].items():
                prep.row(
                    "wiki_policies",
                    policy=pname,
                    nll=pdata["nll"],
                    recovery=pdata["nll_recovery"],
                    em=pdata["em"],
                )
            prep.step(
                1,
                synth_oracle_recov=synth_summary["policies"]["oracle_2"]["nll_recovery"],
                synth_gold_ev_recov=synth_summary["policies"]["gold_evidence_2"]["nll_recovery"],
                synth_causal_cv_recov=synth_summary["policies"]["causal_ridge_cv"]["nll_recovery"],
                synth_best_fixed_recov=synth_summary["policies"]["best_fixed_pair_cv"]["nll_recovery"],
                synth_em_deep=synth_summary["policies"]["all_deep"]["em"],
                synth_em_shallow=synth_summary["policies"]["all_shallow"]["em"],
                wiki_oracle_recov=wiki_summary["policies"]["oracle_2"]["nll_recovery"],
                wiki_causal_cv_recov=wiki_summary["policies"]["causal_ridge_cv"]["nll_recovery"],
                wiki_best_fixed_recov=wiki_summary["policies"]["best_fixed_pair_cv"]["nll_recovery"],
                wiki_em_deep=wiki_summary["policies"]["all_deep"]["em"],
                wiki_em_shallow=wiki_summary["policies"]["all_shallow"]["em"],
                decomp_spearman=decomp_out["summary"]["overall"]["spearman_curr_vs_write"],
                decomp_resid_spearman=decomp_out["summary"]["position_residualized_spearman_curr_vs_write"],
            )

            print("========== REPORT (report_idea1.md) ==========")
            print(report_md)
            print("========== END REPORT ==========")
    finally:
        gpu_lock.release()


if __name__ == "__main__":
    main()
