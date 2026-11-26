"""Round 4-B: Generalized Top-K -> k_v LoopSched-MoE for thinkingmachines/Inkling-2-Mini in Production vLLM.

Supports Inkling-2-Mini (26.4B NVFP4/MXFP8 Looped Sparse MoE):
  - Topology: 24 physical layers -> 6 Prelude [0..5] + 12 Looped Core [6..17] x 3 passes + 6 Coda [18..23]
    = 48 logical visits!
  - MoE: 128 routed experts, Top-8 per token (K=8), 0 shared experts -> MoE dominates active compute!
  - Generalizes Orthogonal Expert Shrinkage to arbitrary Top-K -> k in {1, ..., 7}:
      s_k = sum_{j=1..k} w_j e_j,  r_k = sum_{j=k+1..K} w_j e_j
      gamma_k* = 1 + E[<s_k, r_k>] / E[||s_k||^2]
      MSE_renorm(k) / MSE_noren(k) = E[||(alpha_k - 1)*s_k - r_k||^2] / E[||r_k||^2]
  - Runs 100% natively in production vLLM (SM120 Blackwell NVFP4/MXFP8 Triton kernels + CUDA Graphs):
      Stage 1: Verify hook exactness & measure exact 48-visit x 7-level shrinkage geometry.
      Stage 2: Measure stage/visit sensitivity & solve non-additive IterGreedy over k_v in {1..8}
               for -25%, -37.5%, -50%, -62.5%, -75% (NO k=0 cliff needed!).
      Stage 3: Evaluate all 18 policies on Held-out NLL (64 seqs) + 2,272 MCQ (ARC-C/E + OBQA)
               with paired bootstrap 95% CIs.
      Stage 4: Measure real production vLLM CUDA-Graph end-to-end Decode (B=1/8/32) & Prefill
               (B=8x256) wall-clock speedups across the key operating points!
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from loop_research import orx_run as orx
from loop_research.data_assets import (
    build_disjoint_text_code_splits,
    load_arc_challenge_samples,
    load_arc_easy_samples,
    load_openbookqa_samples,
)
from loop_research.gpu_utils import acquire_idle_gpu, query_gpus
from loop_research.inkling2_vllm_hook import (
    Inkling2LoopSchedController,
    Inkling2VLLMClassPatch,
)
from loop_research.r3_loopsched_v2 import paired_bootstrap_ci


# ---------------------------------------------------------------------------
# 1. Inkling-2-Mini Visit Topology & Generalized Top-K -> k Algorithms
# ---------------------------------------------------------------------------

def build_inkling2_topology(
    num_hidden_layers: int = 24,
    core_start: int = 6,
    core_end: int = 18,
    num_passes: int = 3,
) -> list[dict[str, Any]]:
    """48 logical visits for Inkling-2-Mini: 6 prelude + 12*3 core passes + 6 coda."""
    visits: list[dict[str, Any]] = []
    for lid in range(0, core_start):
        visits.append({"vid": len(visits), "layer_id": lid, "pass_idx": 0, "stage": "prelude", "is_rec": False})
    for p in range(num_passes):
        for lid in range(core_start, core_end):
            visits.append({"vid": len(visits), "layer_id": lid, "pass_idx": p, "stage": f"pass_{p}", "is_rec": True})
    for lid in range(core_end, num_hidden_layers):
        visits.append({"vid": len(visits), "layer_id": lid, "pass_idx": 0, "stage": "coda", "is_rec": False})
    return visits


def generalized_shrinkage_stats(
    weights_sorted: torch.Tensor,
    expert_outs_sorted: torch.Tensor,
    k_keep: int,
) -> dict[str, float]:
    """Compute exact generalized Top-K -> k_keep optimal shrinkage gamma_k* and MSE_renorm/MSE_noren."""
    w = weights_sorted.float()
    e = expert_outs_sorted.float()
    K = w.shape[1]
    assert 1 <= k_keep < K

    s_k = (w[:, :k_keep, None] * e[:, :k_keep, :]).sum(dim=1)
    r_k = (w[:, k_keep:, None] * e[:, k_keep:, :]).sum(dim=1)
    w_tot = w.sum(dim=1, keepdim=True)
    w_kept = w[:, :k_keep].sum(dim=1, keepdim=True).clamp_min(1e-12)
    alpha = w_tot / w_kept

    dot_sr = (s_k * r_k).sum(dim=-1)
    s_sq = (s_k * s_k).sum(dim=-1)
    r_sq = (r_k * r_k).sum(dim=-1)
    cos_sr = (dot_sr / (s_sq.clamp_min(1e-12).sqrt() * r_sq.clamp_min(1e-12).sqrt())).mean().item()

    gamma_opt = 1.0 + float(dot_sr.sum().item()) / max(float(s_sq.sum().item()), 1e-12)
    err_ren = ((alpha - 1.0) * s_k - r_k).pow(2).sum(dim=-1).mean().item()
    err_noren = r_sq.mean().item()
    err_opt = ((gamma_opt - 1.0) * s_k - r_k).pow(2).sum(dim=-1).mean().item()
    return {
        "k_keep": int(k_keep),
        "cos_kept_dropped": float(cos_sr),
        "gamma_opt": float(gamma_opt),
        "mean_alpha_renorm": float(alpha.mean().item()),
        "mse_ratio_ren_vs_noren": float(err_ren / max(err_noren, 1e-12)),
        "mse_ratio_noren_vs_opt": float(err_noren / max(err_opt, 1e-12)),
    }


def solve_layer_only_topk_schedule(
    topo: list[dict[str, Any]],
    visit_sens: np.ndarray,
    K_max: int,
    target_cut_frac: float,
    *,
    min_k: int = 1,
) -> list[int]:
    """Ablation baseline: assign k_layer by physical layer_id ONLY (all passes of a core layer share the same k!)."""
    V = len(topo)
    total_slots = V * K_max
    target_slots = max(V * min_k, min(total_slots, int(round(total_slots * (1.0 - target_cut_frac)))))

    # Group visits by physical layer_id
    by_lid: dict[int, list[int]] = {}
    for t in topo:
        by_lid.setdefault(int(t["layer_id"]), []).append(int(t["vid"]))
    layer_sens = {
        lid: float(np.mean([visit_sens[v] for v in vids]))
        for lid, vids in by_lid.items()
    }
    lids_asc = sorted(by_lid.keys(), key=lambda lid: (layer_sens[lid], lid))

    vk = [K_max] * V
    # Greedily decrement least-sensitive physical layers round-robin while keeping all passes equal
    # until within < 3 slots of target, then adjust single-visit prelude/coda if needed for exact match
    cur = total_slots
    while cur > target_slots:
        moved = False
        for lid in lids_asc:
            vids = by_lid[lid]
            step_cost = len(vids)
            if cur - step_cost >= target_slots and vk[vids[0]] > min_k:
                for v in vids:
                    vk[v] -= 1
                cur -= step_cost
                moved = True
                if cur == target_slots:
                    break
        if not moved:
            # Fine-tune using single-visit layers (prelude/coda) to hit exact slot budget
            single_lids = [lid for lid in lids_asc if len(by_lid[lid]) == 1 and vk[by_lid[lid][0]] > min_k]
            if not single_lids:
                break
            for lid in single_lids:
                if cur <= target_slots:
                    break
                vk[by_lid[lid][0]] -= 1
                cur -= 1
            break
    return vk


def build_stage_schedule_inkling2(
    topo: list[dict[str, Any]],
    k_by_stage: dict[str, int],
) -> list[int]:
    """Deterministic per-stage Top-k schedule for Inkling-2-Mini."""
    return [int(k_by_stage[t["stage"]]) for t in topo]


def solve_stage_static_topk(
    topo: list[dict[str, Any]],
    stage_sens: dict[str, float],
    K_max: int,
    target_cut_frac: float,
    *,
    min_k: int = 1,
) -> list[int]:
    """Deterministic stage-level schedule: cut least-sensitive stages first while keeping every visit in a stage uniform."""
    V = len(topo)
    total_slots = V * K_max
    target_slots = max(V * min_k, min(total_slots, int(round(total_slots * (1.0 - target_cut_frac)))))
    by_stage: dict[str, list[int]] = {}
    for t in topo:
        by_stage.setdefault(t["stage"], []).append(int(t["vid"]))
    stages_asc = sorted(by_stage.keys(), key=lambda st: (float(stage_sens.get(st, 1e6)), st))

    # Proportional inverse-sensitivity allocation rounded to exact slot budget
    inv_w = {st: 1.0 / max(float(stage_sens.get(st, 1e-4)), 1e-4) ** 0.45 for st in stages_asc}
    # Start from all K_max and decrement whole stages in priority order scaled by (K_max - k) * sens
    k_stage = {st: K_max for st in stages_asc}
    cur = total_slots
    while cur > target_slots:
        best_st = None
        best_score = 1e30
        for st in stages_asc:
            sz = len(by_stage[st])
            if k_stage[st] > min_k and cur - sz >= target_slots:
                # Convex penalty so one stage doesn't immediately drop to 1 before others drop from 8->6
                dropped_already = K_max - k_stage[st]
                score = float(stage_sens.get(st, 1e-4)) * ((dropped_already + 1.0) ** 1.35) / (inv_w[st] ** 0.0 + 1.0)
                if score < best_score:
                    best_score = score
                    best_st = st
        if best_st is None:
            break
        k_stage[best_st] -= 1
        cur -= len(by_stage[best_st])
    return build_stage_schedule_inkling2(topo, k_stage)


def solve_general_topk_iter_greedy(
    eval_a_fn,
    topo: list[dict[str, Any]],
    K_max: int,
    target_cut_frac: float,
    *,
    stage_priority_order: list[str] | None = None,
    visit_sens: np.ndarray | None = None,
    min_k_boundary: int = 4,
    min_k_rec: int = 1,
    step_delta_k: int = 1,
    cand_pool_size: int = 12,
    start_vk: list[int] | None = None,
    batch_drop_per_round: int = 1,
) -> list[int]:
    """Non-additive iterative greedy solver over general per-visit k_v in {1, ..., K_max}.

    Unlike IFM Top-2 (where k=1->0 removes 100% of routed experts), Top-8 allows smooth
    fine-grained steps (8->7->...->2->1) protecting sensitive stages while aggressively thinning
    redundant passes with ZERO k=0 cliffs!
    """
    V = len(topo)
    total_slots = V * K_max
    target_slots = max(0, min(total_slots, int(round(total_slots * (1.0 - target_cut_frac)))))
    vk = list(start_vk) if start_vk is not None else [K_max] * V

    stage_rank = {}
    if stage_priority_order:
        for idx, st in enumerate(stage_priority_order):
            stage_rank[st] = idx

    def _visit_priority(v: int) -> tuple:
        # Favor visits with lower marginal sensitivity and more remaining budget
        sens = float(visit_sens[v]) if visit_sens is not None else 0.0
        return (
            0 if topo[v]["is_rec"] else 1,
            stage_rank.get(topo[v]["stage"], 99),
            sens,
            v,
        )

    base_order = sorted(range(V), key=_visit_priority)

    while sum(vk) > target_slots:
        rem = sum(vk) - target_slots
        dk = min(step_delta_k, rem)

        # Dynamic priority: scale sensitivity by how much already dropped so we explore diverse visits
        def _dyn_key(v: int) -> tuple:
            dropped_v = K_max - vk[v]
            sens = float(visit_sens[v]) if visit_sens is not None else float(stage_rank.get(topo[v]["stage"], 5) + 1)
            return (sens * ((dropped_v + 1.0) ** 0.75), base_order.index(v))

        eligible = [
            v for v in range(V)
            if vk[v] - dk >= (min_k_rec if topo[v]["is_rec"] else min_k_boundary)
        ]
        if not eligible:
            eligible = [v for v in range(V) if vk[v] >= dk]
        if not eligible:
            break
        eligible.sort(key=_dyn_key)
        cands = eligible[:cand_pool_size]

        scored: list[tuple[float, int]] = []
        for v in cands:
            vk_try = list(vk)
            vk_try[v] -= dk
            scored.append((float(eval_a_fn(vk_try)), v))
        scored.sort(key=lambda t: (t[0], t[1]))

        n_apply = min(batch_drop_per_round, len(scored), max(1, rem // dk))
        for _, v_pick in scored[:n_apply]:
            if sum(vk) - dk >= target_slots and vk[v_pick] >= dk:
                vk[v_pick] -= dk

    return vk


# ---------------------------------------------------------------------------
# 2. vLLM Runtime Probe, Prompt-Logprob NLL/MCQ Evaluators & CUDA-Graph Speed
# ---------------------------------------------------------------------------

def detect_inkling_runtime() -> dict[str, Any]:
    """Check whether vLLM or transformers supports inkling2_loop in current env."""
    info: dict[str, Any] = {
        "vllm_available": False,
        "vllm_version": None,
        "vllm_has_inkling2": False,
        "transformers_version": None,
        "transformers_has_inkling2": False,
    }
    try:
        import vllm  # type: ignore
        info["vllm_available"] = True
        info["vllm_version"] = getattr(vllm, "__version__", "unknown")
        try:
            from vllm.model_executor.models.registry import ModelRegistry  # type: ignore
            archs = ModelRegistry.get_supported_archs()
            info["vllm_has_inkling2"] = any("Inkling2Loop" in a for a in archs)
        except Exception:
            pass
    except ImportError:
        pass

    try:
        import transformers  # type: ignore
        info["transformers_version"] = getattr(transformers, "__version__", "unknown")
        try:
            from transformers.models.auto.configuration_auto import CONFIG_MAPPING  # type: ignore
            info["transformers_has_inkling2"] = "inkling2_loop" in CONFIG_MAPPING
        except Exception:
            pass
    except ImportError:
        pass
    return info


def _extract_prompt_token_logprobs(req_output: Any, token_ids: list[int]) -> list[float]:
    """Extract per-token logprobs for token_ids[1:] from vLLM RequestOutput.prompt_logprobs."""
    plps = req_output.prompt_logprobs
    assert plps is not None and len(plps) == len(token_ids), (
        f"Unexpected prompt_logprobs len={None if plps is None else len(plps)} vs {len(token_ids)}"
    )
    vals: list[float] = []
    for pos in range(1, len(token_ids)):
        tid = int(token_ids[pos])
        entry = plps[pos]
        lp_obj = entry[tid]
        vals.append(float(lp_obj.logprob if hasattr(lp_obj, "logprob") else lp_obj))
    return vals


def eval_nll_vllm(
    llm: Any,
    seq_items: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute exact held-out NLL using vLLM batched prompt_logprobs (NVFP4/MXFP8 exact)."""
    from vllm import SamplingParams, TokensPrompt  # type: ignore

    prompts = [TokensPrompt(prompt_token_ids=list(it["tokens"])) for it in seq_items]
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    outputs = llm.generate(prompts, sp, use_tqdm=False)

    per_seq: list[float] = []
    text_nlls: list[float] = []
    code_nlls: list[float] = []
    for it, out in zip(seq_items, outputs):
        lps = _extract_prompt_token_logprobs(out, it["tokens"])
        nll = float(-np.mean(lps))
        per_seq.append(nll)
        (text_nlls if it.get("domain") == "text" else code_nlls).append(nll)
    return {
        "nll": float(np.mean(per_seq)),
        "text_nll": float(np.mean(text_nlls)) if text_nlls else 0.0,
        "code_nll": float(np.mean(code_nlls)) if code_nlls else 0.0,
        "per_seq": per_seq,
    }


def prepare_mcq_encoded_hf(samples: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    """Encode MCQ prompts using Inkling-2's HF tokenizer with exact choice boundary index."""
    encoded: list[dict[str, Any]] = []
    for q_idx, item in enumerate(samples):
        prompt = f"Question: {item['question']}\nAnswer:"
        p_ids = list(tokenizer.encode(prompt, add_special_tokens=True))
        p_len = len(p_ids)
        choices_enc: list[tuple[list[int], int]] = []
        for c_txt in item["choice_texts"]:
            c_ids = list(tokenizer.encode(" " + c_txt, add_special_tokens=False))
            choices_enc.append((p_ids + c_ids, p_len) if c_ids else ([], p_len))
        encoded.append({"q_idx": q_idx, "gold_idx": int(item["gold_idx"]), "choices": choices_enc})
    return encoded


def eval_mcq_vllm(
    llm: Any,
    encoded_items: list[dict[str, Any]],
    *,
    chunk_size: int = 512,
) -> tuple[float, list[int]]:
    """Score MCQ suites via length-normalized choice completion log-probability in vLLM."""
    from vllm import SamplingParams, TokensPrompt  # type: ignore

    flat: list[tuple[int, int, list[int], int]] = []
    scores_by_q: list[list[float]] = [[-1e9] * len(it["choices"]) for it in encoded_items]
    for it in encoded_items:
        qi = it["q_idx"]
        for ci, (full_ids, p_len) in enumerate(it["choices"]):
            if len(full_ids) > p_len:
                flat.append((qi, ci, full_ids, p_len))

    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    for s0 in range(0, len(flat), chunk_size):
        chunk = flat[s0 : s0 + chunk_size]
        prompts = [TokensPrompt(prompt_token_ids=x[2]) for x in chunk]
        outs = llm.generate(prompts, sp, use_tqdm=False)
        for (qi, ci, full_ids, p_len), out in zip(chunk, outs):
            lps = _extract_prompt_token_logprobs(out, full_ids)
            # lps[k] is logprob of full_ids[k+1]; choice tokens start at index p_len -> lps[p_len-1:]
            choice_lps = lps[p_len - 1 :]
            scores_by_q[qi][ci] = float(np.mean(choice_lps))

    flags = [
        1 if int(np.argmax(scores_by_q[qi])) == encoded_items[qi]["gold_idx"] else 0
        for qi in range(len(encoded_items))
    ]
    return float(np.mean(flags)), flags


def _cleanup_vllm(llm: Any) -> None:
    """Thoroughly destroy a vLLM LLM instance and release all GPU VRAM."""
    try:
        if hasattr(llm, "llm_engine") and hasattr(llm.llm_engine, "engine_core"):
            ec = llm.llm_engine.engine_core
            if hasattr(ec, "shutdown"):
                ec.shutdown()
    except Exception:
        pass
    del llm
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment  # type: ignore
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def benchmark_vllm_cudagraph_schedule(
    model_path: str,
    controller: Inkling2LoopSchedController,
    schedule_name: str,
    visit_k: list[int],
    shrink_mode: str,
    visit_gamma: list[float] | None,
    *,
    max_model_len: int = 2048,
    gpu_memory_utilization: float = 0.72,
    decode_tokens: int = 32,
    warmup: int = 3,
    iters: int = 12,
    prompt_token_bank: list[list[int]],
) -> list[dict[str, Any]]:
    """Capture production vLLM CUDA Graphs for `visit_k` and time real end-to-end Decode & Prefill."""
    from vllm import LLM, SamplingParams, TokensPrompt  # type: ignore

    controller.set_policy(visit_k, shrink_mode=shrink_mode, visit_gamma=visit_gamma)
    cut_pct = 100.0 * (1.0 - sum(visit_k) / float(len(visit_k) * controller.k_max))
    rows: list[dict[str, Any]] = []

    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=False,           # REAL production vLLM CUDA Graphs ON!
        enable_prefix_caching=False,   # Zero cache contamination!
        limit_mm_per_prompt={"image": 0},  # Text-only saves vision encoder VRAM & capture time
    )
    try:
        sp_1 = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
        sp_dec = SamplingParams(temperature=0.0, max_tokens=decode_tokens + 1, ignore_eos=True)

        # 1. Decode B=1, B=8, B=32 (subtract 1-token prefill from (decode_tokens+1)-token run!)
        for bsz in (1, 8, 32):
            shape_name = f"decode_B{bsz}_L1"
            p64 = [TokensPrompt(prompt_token_ids=prompt_token_bank[i % len(prompt_token_bank)][:64]) for i in range(bsz)]
            # Warmup both prefill-1 and decode-(N+1) paths
            for _ in range(warmup):
                _ = llm.generate(p64, sp_1, use_tqdm=False)
                _ = llm.generate(p64, sp_dec, use_tqdm=False)
            torch.cuda.synchronize()

            t1_ms_list: list[float] = []
            tn_ms_list: list[float] = []
            for _ in range(iters):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = llm.generate(p64, sp_1, use_tqdm=False)
                torch.cuda.synchronize()
                t1_ms_list.append((time.perf_counter() - t0) * 1000.0)

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = llm.generate(p64, sp_dec, use_tqdm=False)
                torch.cuda.synchronize()
                tn_ms_list.append((time.perf_counter() - t0) * 1000.0)

            step_ms = [max(tn - t1, 0.1) / float(decode_tokens) for t1, tn in zip(t1_ms_list, tn_ms_list)]
            med_ms = float(np.median(step_ms))
            q25, q75 = np.percentile(step_ms, [25, 75])
            rows.append({
                "backend": "vllm_cg_nvfp4",
                "shape": shape_name,
                "schedule": schedule_name,
                "cut_pct": float(cut_pct),
                "mean_k": float(np.mean(visit_k)),
                "median_ms": med_ms,
                "iqr_ms": float(q75 - q25),
                "tok_per_sec": float(bsz * 1000.0 / max(med_ms, 1e-6)),
            })

        # 2. Prefill B=8 x L=256 (TTFT)
        p256 = [TokensPrompt(prompt_token_ids=prompt_token_bank[i % len(prompt_token_bank)][:256]) for i in range(8)]
        for _ in range(warmup):
            _ = llm.generate(p256, sp_1, use_tqdm=False)
        torch.cuda.synchronize()
        pf_ms: list[float] = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = llm.generate(p256, sp_1, use_tqdm=False)
            torch.cuda.synchronize()
            pf_ms.append((time.perf_counter() - t0) * 1000.0)
        med_pf = float(np.median(pf_ms))
        q25, q75 = np.percentile(pf_ms, [25, 75])
        rows.append({
            "backend": "vllm_cg_nvfp4",
            "shape": "prefill_B8_L256",
            "schedule": schedule_name,
            "cut_pct": float(cut_pct),
            "mean_k": float(np.mean(visit_k)),
            "median_ms": med_pf,
            "iqr_ms": float(q75 - q25),
            "tok_per_sec": float(8 * 256 * 1000.0 / max(med_pf, 1e-6)),
        })
    finally:
        _cleanup_vllm(llm)

    return rows


# ---------------------------------------------------------------------------
# 3. End-to-End Track B Experiment Driver
# ---------------------------------------------------------------------------

def _fmt_stage_k_summary(topo: list[dict[str, Any]], vk: list[int]) -> str:
    stages = ["prelude", "pass_0", "pass_1", "pass_2", "coda"]
    parts = []
    for st in stages:
        vals = [vk[t["vid"]] for t in topo if t["stage"] == st]
        if vals:
            parts.append(f"{st}={np.mean(vals):.1f}")
    return ", ".join(parts)


def main() -> None:
    # CRITICAL: run vLLM V1 EngineCore in-process so class-level MoE hook & controller live in same process!
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    t_start = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model_path = Path(args.model_dir) / cfg.get("local_dir", "thinkingmachines__Inkling-2-Mini")

    rt = detect_inkling_runtime()
    print(f"[r4b:inkling2] Model path: {model_path} (exists={model_path.exists()})", flush=True)
    print(f"[r4b:inkling2] Runtime capabilities: {json.dumps(rt)}", flush=True)

    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(
            f"Inkling-2-Mini not found at {model_path}. Run: ASSETS=inkling2_mini bash scripts/download_assets.sh"
        )
    if not rt["vllm_has_inkling2"]:
        raise RuntimeError(
            f"vLLM with Inkling2LoopForConditionalGeneration not found in this env ({rt}). "
            "Run `bash scripts/setup_inkling_env.sh` first."
        )

    mcfg = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    tcfg = mcfg["text_config"]
    lcfg = tcfg["loop_config"]
    K_max = int(tcfg["num_experts_per_tok"])
    topo = build_inkling2_topology(
        num_hidden_layers=int(tcfg["num_hidden_layers"]),
        core_start=int(lcfg["core_start"]),
        core_end=int(lcfg["core_end"]),
        num_passes=int(lcfg["num_passes"]),
    )
    V = len(topo)
    print(
        f"[r4b:inkling2] Verified topology: V={V} visits "
        f"(prelude={lcfg['core_start']}, core={lcfg['core_end']-lcfg['core_start']}x{lcfg['num_passes']}, "
        f"coda={tcfg['num_hidden_layers']-lcfg['core_end']}), "
        f"E={tcfg['n_routed_experts']}, K={K_max}, shared={tcfg['n_shared_experts']}",
        flush=True,
    )

    exit_if_busy = os.environ.get("EXIT_IF_NO_IDLE_GPU", "1") == "1"
    wait_sec = 10 if exit_if_busy else 300
    try:
        gpu_idx, gpu_lock, gpu_status = acquire_idle_gpu(cfg.get("preferred_gpu"), wait_timeout_sec=wait_sec)
    except RuntimeError as exc:
        print(f"[r4b:inkling2] No completely idle GPU available ({exc}). Current GPUs:", flush=True)
        for g in query_gpus():
            print(f"  GPU {g.index}: {g.name} mem={g.memory_used_mib}MiB util={g.utilization_pct}% pids={g.active_pids}", flush=True)
        if exit_if_busy:
            print("[r4b:inkling2] Exiting cleanly (EXIT_IF_NO_IDLE_GPU=1).", flush=True)
            return
        raise

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        print(f"[r4b:inkling2] Acquired idle GPU {gpu_idx}: {gpu_status.name} (mem={gpu_status.memory_used_mib}MiB, util={gpu_status.utilization_pct}%)", flush=True)

        # Load tokenizer + disjoint datasets (identical suites to R2-A / R3 / R4-A!)
        from transformers import AutoTokenizer  # type: ignore
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

        seq_len = int(cfg.get("seq_len", 256))
        encode_fn = lambda txt: list(tokenizer.encode(txt, add_special_tokens=False))
        splits = build_disjoint_text_code_splits(
            Path(args.dataset_dir),
            encode_fn,
            seq_len=seq_len,
            n_calib_a=int(cfg.get("n_calib_a", 12)),
            n_calib_b=int(cfg.get("n_calib_b", 6)),
            n_adapt_train=8,
            n_eval_heldout=int(cfg.get("n_eval", 64)),
            seed=int(cfg.get("seed", 20261126)),
        )
        calib_a = splits["calib_a"]
        calib_b = splits["calib_b"]
        eval_seqs = splits["eval_heldout"]
        diag_seqs = calib_a[: int(cfg.get("n_diag", 4))]

        mcq_limits = cfg.get("mcq_limits", {"arc_challenge": 1172, "arc_easy": 600, "openbookqa": 500})
        arc_c_enc = prepare_mcq_encoded_hf(load_arc_challenge_samples(Path(args.dataset_dir), limit=int(mcq_limits["arc_challenge"])), tokenizer)
        arc_e_enc = prepare_mcq_encoded_hf(load_arc_easy_samples(Path(args.dataset_dir), limit=int(mcq_limits["arc_easy"])), tokenizer)
        obqa_enc = prepare_mcq_encoded_hf(load_openbookqa_samples(Path(args.dataset_dir), limit=int(mcq_limits["openbookqa"])), tokenizer)
        print(f"[r4b:inkling2] Loaded splits: diag={len(diag_seqs)}, calib_a={len(calib_a)}, calib_b={len(calib_b)}, eval={len(eval_seqs)} | MCQ: ARC-C={len(arc_c_enc)}, ARC-E={len(arc_e_enc)}, OBQA={len(obqa_enc)}", flush=True)

        speed_schedule_names = [
            "nvfp4_base",
            "cut25_iter_loopsched",
            "cut38_iter_loopsched",
            "cut50_unif_k4_noren",
            "cut50_iter_loopsched",
            "cut62_iter_loopsched",
            "cut75_iter_loopsched",
        ]
        n_eval_policies = 18

        orx.plan([
            {"phase": "setup", "of": 1, "unit": "env"},
            {"phase": "download", "of": 1, "unit": "assets"},
            {"phase": "inkling_verify_geom", "of": 2, "unit": "checks"},
            {"phase": "inkling_calib", "of": 5, "unit": "steps"},
            {"phase": "inkling_eval", "of": n_eval_policies, "unit": "conds"},
            {"phase": "inkling_speed", "of": len(speed_schedule_names), "unit": "scheds"},
            {"phase": "report", "of": 1, "unit": "file"},
        ])
        run = orx.progress(total=1, unit="run")

        controller = Inkling2LoopSchedController(num_visits=V, k_max=K_max)
        max_model_len = int(cfg.get("max_model_len", 2048))
        gpu_mem_util = float(cfg.get("gpu_memory_utilization", 0.72))

        from vllm import LLM  # type: ignore

        with Inkling2VLLMClassPatch(controller):
            # ------------------------------------------------------------------
            # Stage 1: Load vLLM Eager (fast policy switching) + Verify + Geometry
            # ------------------------------------------------------------------
            with run.phase("inkling_verify_geom", total=2, unit="checks") as p_vg:
                print("[r4b:inkling2] Loading Inkling-2-Mini in production vLLM (NVFP4/MXFP8, eager for calibration/eval)...", flush=True)
                llm = LLM(
                    model=str(model_path),
                    tensor_parallel_size=1,
                    trust_remote_code=True,
                    max_model_len=max_model_len,
                    gpu_memory_utilization=gpu_mem_util,
                    enforce_eager=True,
                    enable_prefix_caching=False,
                    limit_mm_per_prompt={"image": 0},
                )

                # Step 1: Verify exactness of our Triton route_general (k=8, noren=True == renorm at k=8!) vs native vLLM!
                controller.reset_baseline()
                ref_check = eval_nll_vllm(llm, diag_seqs[:2])
                controller.set_policy([K_max] * V, shrink_mode="noren")
                hook_check = eval_nll_vllm(llm, diag_seqs[:2])
                nll_diff = abs(ref_check["nll"] - hook_check["nll"])
                print(
                    f"[r4b:inkling2] Hook Exactness Verify (k=8 native vs route_general): "
                    f"native_NLL={ref_check['nll']:.6f}, hook_NLL={hook_check['nll']:.6f}, abs_diff={nll_diff:.3e}",
                    flush=True,
                )
                assert nll_diff < 1e-2, f"Hook mismatch vs native vLLM: {nll_diff}"
                p_vg.step(1)

                # Step 2: Exact 48-visit x 7-level Generalized Orthogonal Shrinkage Geometry
                controller.start_diagnostics()
                _ = eval_nll_vllm(llm, diag_seqs)
                geom_by_visit = controller.stop_diagnostics()
                controller.reset_baseline()

                # Aggregate geometry by stage and by k_keep in 1..7
                stages = ["prelude", "pass_0", "pass_1", "pass_2", "coda"]
                geom_by_stage_k: dict[str, dict[int, dict[str, float]]] = {}
                for st in stages:
                    vids_st = [t["vid"] for t in topo if t["stage"] == st]
                    geom_by_stage_k[st] = {}
                    for k in range(1, K_max):
                        geom_by_stage_k[st][k] = {
                            "cos": float(np.mean([geom_by_visit[v][k]["cos_kept_dropped"] for v in vids_st])),
                            "gamma_opt": float(np.mean([geom_by_visit[v][k]["gamma_opt"] for v in vids_st])),
                            "alpha_ren": float(np.mean([geom_by_visit[v][k]["mean_alpha_renorm"] for v in vids_st])),
                            "mse_ren_over_noren": float(np.mean([geom_by_visit[v][k]["mse_ratio_ren_vs_noren"] for v in vids_st])),
                            "mse_noren_over_opt": float(np.mean([geom_by_visit[v][k]["mse_ratio_noren_vs_opt"] for v in vids_st])),
                        }

                print("[r4b:inkling2] Generalized Orthogonal Shrinkage Geometry (k=4, 50% cut):", flush=True)
                for st in stages:
                    g4 = geom_by_stage_k[st][4]
                    print(
                        f"  {st:8s} | cos(s4,r4)={g4['cos']:+.4f} | gamma_4*={g4['gamma_opt']:.3f} "
                        f"(alpha_ren={g4['alpha_ren']:.3f}) | MSE_ren/MSE_noren={g4['mse_ren_over_noren']:.2f}x",
                        flush=True,
                    )
                print("[r4b:inkling2] Global Geometry across k=1..7 (mean over 48 visits):", flush=True)
                geom_global_k: dict[int, dict[str, float]] = {}
                for k in range(1, K_max):
                    geom_global_k[k] = {
                        "cos": float(np.mean([geom_by_visit[v][k]["cos_kept_dropped"] for v in range(V)])),
                        "gamma_opt": float(np.mean([geom_by_visit[v][k]["gamma_opt"] for v in range(V)])),
                        "alpha_ren": float(np.mean([geom_by_visit[v][k]["mean_alpha_renorm"] for v in range(V)])),
                        "mse_ren_over_noren": float(np.mean([geom_by_visit[v][k]["mse_ratio_ren_vs_noren"] for v in range(V)])),
                    }
                    gk = geom_global_k[k]
                    print(
                        f"  k={k} (cut={(1-k/8)*100:4.1f}%) | cos={gk['cos']:+.4f} | gamma*={gk['gamma_opt']:.3f} "
                        f"(alpha={gk['alpha_ren']:.3f}) | MSE_ren/noren={gk['mse_ren_over_noren']:.2f}x",
                        flush=True,
                    )
                p_vg.step(2)

            # ------------------------------------------------------------------
            # Stage 2: Stage Sensitivity + Per-Visit Screening + Non-Additive IterGreedy
            # ------------------------------------------------------------------
            def eval_calib_a(vk: list[int], mode: str = "noren") -> float:
                controller.set_policy(vk, shrink_mode=mode)
                return float(eval_nll_vllm(llm, calib_a)["nll"])

            with run.phase("inkling_calib", total=5, unit="steps") as p_cal:
                t_cal0 = time.time()
                base_a = eval_calib_a([K_max] * V, mode="renorm")

                # Step 1: Joint stage-level sensitivity (k=4, k=2, k=0)
                stage_sens_k4: dict[str, float] = {}
                stage_sens_k2: dict[str, float] = {}
                stage_sens_k0: dict[str, float] = {}
                for st in stages:
                    for k_test, dst_map in ((4, stage_sens_k4), (2, stage_sens_k2), (0, stage_sens_k0)):
                        vk_try = [k_test if topo[v]["stage"] == st else K_max for v in range(V)]
                        dst_map[st] = max(0.0, eval_calib_a(vk_try, mode="noren") - base_a)
                print(
                    f"[r4b:inkling2] Stage Sensitivity (ΔNLL on calib_a): "
                    f"k=4:{ {k: round(v, 4) for k, v in stage_sens_k4.items()} } | "
                    f"k=2:{ {k: round(v, 4) for k, v in stage_sens_k2.items()} } | "
                    f"k=0:{ {k: round(v, 3) for k, v in stage_sens_k0.items()} }",
                    flush=True,
                )
                p_cal.step(1)

                # Step 2: Single-visit 8->2 sensitivity across all 48 visits (fast: 48 forwards!)
                visit_sens_k2 = np.zeros(V, dtype=np.float64)
                for v in range(V):
                    vk_try = [K_max] * V
                    vk_try[v] = 2
                    visit_sens_k2[v] = max(0.0, eval_calib_a(vk_try, mode="noren") - base_a)
                stage_priority = sorted(stages, key=lambda st: float(stage_sens_k2[st]))
                p_cal.step(2)

                # Step 3: Layer-only & Stage-static baselines + Non-additive IterGreedy (-25% and -37.5%)
                vk_38_layer = solve_layer_only_topk_schedule(topo, visit_sens_k2, K_max, 0.375, min_k=1)
                vk_50_layer = solve_layer_only_topk_schedule(topo, visit_sens_k2, K_max, 0.50, min_k=1)
                vk_50_stage = solve_stage_static_topk(topo, stage_sens_k2, K_max, 0.50, min_k=1)

                # Chain non-additive IterGreedy smoothly: 0% -> -25% -> -37.5% -> -50% -> -62.5% -> -75%
                # Drop 4 visits by delta_k=1 per round from pool of 10 -> fast (~30 rounds total across ALL 5 cuts!)
                vk_25_iter = solve_general_topk_iter_greedy(
                    eval_calib_a, topo, K_max, 0.25,
                    stage_priority_order=stage_priority, visit_sens=visit_sens_k2,
                    min_k_boundary=4, min_k_rec=1, step_delta_k=1, cand_pool_size=10,
                    batch_drop_per_round=4,
                )
                vk_38_iter = solve_general_topk_iter_greedy(
                    eval_calib_a, topo, K_max, 0.375,
                    stage_priority_order=stage_priority, visit_sens=visit_sens_k2,
                    min_k_boundary=4, min_k_rec=1, step_delta_k=1, cand_pool_size=10,
                    start_vk=vk_25_iter, batch_drop_per_round=4,
                )
                p_cal.step(3)

                # Step 4: -50% IterGreedy (high precision: batch_drop=2 from pool=10!)
                vk_50_iter = solve_general_topk_iter_greedy(
                    eval_calib_a, topo, K_max, 0.50,
                    stage_priority_order=stage_priority, visit_sens=visit_sens_k2,
                    min_k_boundary=3, min_k_rec=1, step_delta_k=1, cand_pool_size=10,
                    start_vk=vk_38_iter, batch_drop_per_round=2,
                )
                p_cal.step(4)

                # Step 5: -62.5% and -75% IterGreedy (min_k_boundary=2, min_k_rec=1 -> zero k=0 cliffs!)
                vk_62_iter = solve_general_topk_iter_greedy(
                    eval_calib_a, topo, K_max, 0.625,
                    stage_priority_order=stage_priority, visit_sens=visit_sens_k2,
                    min_k_boundary=2, min_k_rec=1, step_delta_k=1, cand_pool_size=10,
                    start_vk=vk_50_iter, batch_drop_per_round=4,
                )
                vk_75_iter = solve_general_topk_iter_greedy(
                    eval_calib_a, topo, K_max, 0.75,
                    stage_priority_order=stage_priority, visit_sens=visit_sens_k2,
                    min_k_boundary=1, min_k_rec=1, step_delta_k=1, cand_pool_size=10,
                    start_vk=vk_62_iter, batch_drop_per_round=4,
                )

                # Build exact calibrated per-visit gamma_v* vectors for opt policies
                def build_gamma_vec(vk: list[int]) -> list[float]:
                    gs = []
                    for v, kv in enumerate(vk):
                        if 1 <= kv < K_max:
                            g = float(geom_by_visit[v][kv]["gamma_opt"])
                            gs.append(float(np.clip(g, 0.85, 1.50)))
                        else:
                            gs.append(1.0)
                    return gs

                gammas_k4 = build_gamma_vec([4] * V)
                gammas_50_iter = build_gamma_vec(vk_50_iter)

                print(f"[r4b:inkling2] Calibration complete in {time.time()-t_cal0:.1f}s:", flush=True)
                for nm, vk in [
                    ("cut25_iter", vk_25_iter),
                    ("cut38_layer", vk_38_layer),
                    ("cut38_iter", vk_38_iter),
                    ("cut50_layer", vk_50_layer),
                    ("cut50_stage", vk_50_stage),
                    ("cut50_iter", vk_50_iter),
                    ("cut62_iter", vk_62_iter),
                    ("cut75_iter", vk_75_iter),
                ]:
                    print(f"  {nm:12s} (sum={sum(vk):3d}/384, cut={(1-sum(vk)/384)*100:4.1f}%): {_fmt_stage_k_summary(topo, vk)}", flush=True)
                p_cal.step(5)

            # ------------------------------------------------------------------
            # Stage 3: Full Held-Out Quality Evaluation (18 Policies x 64 NLL + 2,272 MCQ)
            # ------------------------------------------------------------------
            policies: list[tuple[str, list[int], str, list[float] | None]] = [
                ("nvfp4_base",            [8] * V,       "renorm", None),
                ("cut25_unif_k6_renorm",  [6] * V,       "renorm", None),
                ("cut25_unif_k6_noren",   [6] * V,       "noren",  None),
                ("cut25_iter_loopsched",  vk_25_iter,    "noren",  None),
                ("cut38_layer_only",      vk_38_layer,   "noren",  None),
                ("cut38_iter_loopsched",  vk_38_iter,    "noren",  None),
                ("cut50_unif_k4_renorm",  [4] * V,       "renorm", None),
                ("cut50_unif_k4_noren",   [4] * V,       "noren",  None),
                ("cut50_unif_k4_opt",     [4] * V,       "opt",    gammas_k4),
                ("cut50_layer_only",      vk_50_layer,   "noren",  None),
                ("cut50_stage_static",    vk_50_stage,   "noren",  None),
                ("cut50_iter_loopsched",  vk_50_iter,    "noren",  None),
                ("cut50_iter_opt",        vk_50_iter,    "opt",    gammas_50_iter),
                ("cut62_unif_k3_noren",   [3] * V,       "noren",  None),
                ("cut62_iter_loopsched",  vk_62_iter,    "noren",  None),
                ("cut75_unif_k2_renorm",  [2] * V,       "renorm", None),
                ("cut75_unif_k2_noren",   [2] * V,       "noren",  None),
                ("cut75_iter_loopsched",  vk_75_iter,    "noren",  None),
            ]
            policy_map = {p[0]: p for p in policies}

            eval_results: dict[str, dict[str, Any]] = {}
            per_seq_nlls: dict[str, list[float]] = {}
            pooled_mcq_flags: dict[str, list[int]] = {}
            base_nll = 0.0
            base_per_seq: list[float] = []

            with run.phase("inkling_eval", total=len(policies), unit="conds") as p_ev:
                for idx, (pname, vk, smode, vgamma) in enumerate(policies, 1):
                    t0 = time.time()
                    controller.set_policy(vk, shrink_mode=smode, visit_gamma=vgamma)
                    nll_res = eval_nll_vllm(llm, eval_seqs)
                    arc_c, fc = eval_mcq_vllm(llm, arc_c_enc)
                    arc_e, fe = eval_mcq_vllm(llm, arc_e_enc)
                    obqa, fo = eval_mcq_vllm(llm, obqa_enc)
                    avg_mcq = float(np.mean([arc_c, arc_e, obqa]))
                    cut_pct = 100.0 * (1.0 - sum(vk) / float(V * K_max))

                    if pname == "nvfp4_base":
                        base_nll = nll_res["nll"]
                        base_per_seq = list(nll_res["per_seq"])
                        dnll, ci_lo, ci_hi = 0.0, 0.0, 0.0
                    else:
                        diffs = [a - b for a, b in zip(nll_res["per_seq"], base_per_seq)]
                        dnll, ci_lo, ci_hi = paired_bootstrap_ci(diffs)

                    per_seq_nlls[pname] = list(nll_res["per_seq"])
                    pooled_mcq_flags[pname] = fc + fe + fo
                    el = time.time() - t0

                    rec = {
                        "name": pname,
                        "shrink_mode": smode,
                        "cut_pct": float(cut_pct),
                        "mean_k": float(np.mean(vk)),
                        "min_k": int(min(vk)),
                        "max_k": int(max(vk)),
                        "stage_k": _fmt_stage_k_summary(topo, vk),
                        "visit_k": list(vk),
                        "nll": float(nll_res["nll"]),
                        "delta_nll": float(dnll),
                        "delta_nll_ci": [float(ci_lo), float(ci_hi)],
                        "arc_c": float(arc_c),
                        "arc_e": float(arc_e),
                        "obqa": float(obqa),
                        "avg_mcq": float(avg_mcq),
                        "elapsed_s": float(el),
                    }
                    eval_results[pname] = rec
                    print(
                        f"[r4b:inkling2] {pname:22s} cut={cut_pct:5.1f}% (k={np.mean(vk):.2f}) | "
                        f"NLL={nll_res['nll']:.4f} (Δ={dnll:+.4f}) | "
                        f"ARC-C={arc_c*100:5.1f}% ARC-E={arc_e*100:5.1f}% OBQA={obqa*100:5.1f}% Avg={avg_mcq*100:5.1f}% | "
                        f"[{rec['stage_k']}] | t={el:.1f}s",
                        flush=True,
                    )
                    p_ev.row(
                        "inkling_results",
                        policy=pname,
                        cut_pct=cut_pct,
                        delta_nll=dnll,
                        arc_c=arc_c,
                        arc_e=arc_e,
                        obqa=obqa,
                        avg_mcq=avg_mcq,
                    )
                    p_ev.step(idx, latest_avg_mcq=avg_mcq, latest_dnll=dnll)

            # Free eager LLM before real CUDA-Graph speed benchmarks!
            _cleanup_vllm(llm)

            # ------------------------------------------------------------------
            # Stage 4: Real Production vLLM CUDA-Graph End-to-End Speedups!
            # ------------------------------------------------------------------
            prompt_token_bank = [list(it["tokens"]) for it in eval_seqs[:32]]
            all_speed_rows: list[dict[str, Any]] = []
            base_ms_by_shape: dict[str, float] = {}

            with run.phase("inkling_speed", total=len(speed_schedule_names), unit="scheds") as p_sp:
                for s_idx, sname in enumerate(speed_schedule_names, 1):
                    _, vk, smode, vgamma = policy_map[sname]
                    s_rows = benchmark_vllm_cudagraph_schedule(
                        str(model_path),
                        controller,
                        sname,
                        vk,
                        smode,
                        vgamma,
                        max_model_len=max_model_len,
                        gpu_memory_utilization=gpu_mem_util,
                        decode_tokens=int(cfg.get("speed_decode_tokens", 32)),
                        warmup=int(cfg.get("speed_warmup", 3)),
                        iters=int(cfg.get("speed_iters", 12)),
                        prompt_token_bank=prompt_token_bank,
                    )
                    b1_spd = 1.0
                    for r in s_rows:
                        if sname == "nvfp4_base":
                            base_ms_by_shape[r["shape"]] = r["median_ms"]
                        r["speedup"] = float(base_ms_by_shape[r["shape"]] / max(r["median_ms"], 1e-6))
                        if r["shape"] == "decode_B1_L1":
                            b1_spd = r["speedup"]
                        all_speed_rows.append(r)
                        print(
                            f"  [vllm_cg_nvfp4] {r['shape']:15s} | {sname:22s} (cut={r['cut_pct']:4.1f}%, k={r['mean_k']:.2f}) | "
                            f"{r['median_ms']:6.2f} ms (IQR {r['iqr_ms']:.2f}) | {r['speedup']:.3f}x | {r['tok_per_sec']:,.0f} tok/s",
                            flush=True,
                        )
                        p_sp.row(
                            "inkling_speed",
                            shape=r["shape"],
                            schedule=sname,
                            cut_pct=r["cut_pct"],
                            median_ms=r["median_ms"],
                            speedup=r["speedup"],
                            tok_per_sec=r["tok_per_sec"],
                        )
                    p_sp.step(s_idx, latest_b1_speedup=b1_spd)

        # ------------------------------------------------------------------
        # Stage 5: Paired Bootstrap Contrasts & Report
        # ------------------------------------------------------------------
        def contrast(first: str, second: str) -> dict[str, Any]:
            # Positive NLL gain = first has LOWER NLL than second!
            nll_diffs = [b - a for a, b in zip(per_seq_nlls[first], per_seq_nlls[second])]
            n_m, n_lo, n_hi = paired_bootstrap_ci(nll_diffs)
            n_wins = sum(1 for d in nll_diffs if d > 0)
            mcq_diffs = [(a - b) * 100.0 for a, b in zip(pooled_mcq_flags[first], pooled_mcq_flags[second])]
            m_m, m_lo, m_hi = paired_bootstrap_ci(mcq_diffs)
            return {
                "nll_gain": float(n_m),
                "nll_ci": [float(n_lo), float(n_hi)],
                "nll_wins": int(n_wins),
                "mcq_gain_pp": float(m_m),
                "mcq_ci_pp": [float(m_lo), float(m_hi)],
            }

        contrast_pairs = [
            ("cut25_unif_k6_noren", "cut25_unif_k6_renorm"),
            ("cut50_unif_k4_noren", "cut50_unif_k4_renorm"),
            ("cut50_unif_k4_opt", "cut50_unif_k4_noren"),
            ("cut75_unif_k2_noren", "cut75_unif_k2_renorm"),
            ("cut38_iter_loopsched", "cut38_layer_only"),
            ("cut50_iter_loopsched", "cut50_unif_k4_noren"),
            ("cut50_iter_loopsched", "cut50_layer_only"),
            ("cut50_iter_loopsched", "cut50_stage_static"),
            ("cut50_iter_loopsched", "cut50_unif_k4_renorm"),
            ("cut50_iter_opt", "cut50_iter_loopsched"),
            ("cut62_iter_loopsched", "cut62_unif_k3_noren"),
            ("cut75_iter_loopsched", "cut75_unif_k2_noren"),
            ("cut75_iter_loopsched", "cut75_unif_k2_renorm"),
        ]
        contrasts = {f"{a}_vs_{b}": contrast(a, b) for a, b in contrast_pairs}

        spd_idx = {(r["shape"], r["schedule"]): r for r in all_speed_rows}
        elapsed = time.time() - t_start

        payload = {
            "model": "thinkingmachines/Inkling-2-Mini",
            "gpu_index": gpu_idx,
            "gpu_name": gpu_status.name,
            "runtime": rt,
            "elapsed_s": float(elapsed),
            "hook_verify_abs_diff": float(nll_diff),
            "geometry_global_k": geom_global_k,
            "geometry_by_stage_k": geom_by_stage_k,
            "geometry_by_visit": geom_by_visit,
            "stage_sensitivity": {
                "k4": stage_sens_k4,
                "k2": stage_sens_k2,
                "k0": stage_sens_k0,
            },
            "visit_sens_k2": visit_sens_k2.tolist(),
            "results": eval_results,
            "speed_rows": all_speed_rows,
            "contrasts": contrasts,
        }
        (out_dir / "results_r4b_inkling2.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

        # Markdown summary report
        lines = [
            "# Round 4-B: Inkling-2-Mini (Top-8 / 128 Experts, 0 Shared) LoopSched-MoE in Production vLLM",
            "",
            f"- GPU: `{gpu_idx}` ({gpu_status.name}), vLLM `{rt['vllm_version']}`, elapsed `{elapsed:.1f}s`",
            f"- Hook Exactness Verify (`k=8` native vs `route_general`): `abs_diff = {nll_diff:.3e}`",
            "",
            "## 1. Generalized Orthogonal Shrinkage Geometry (48 visits, NVFP4/MXFP8 exact)",
            "| k_keep | Cut% | cos(s_k, r_k) | gamma_k* | alpha_renorm | MSE_ren / MSE_noren |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for k in range(1, K_max):
            gk = geom_global_k[k]
            lines.append(
                f"| {k} | {(1-k/K_max)*100:.1f}% | {gk['cos']:+.4f} | {gk['gamma_opt']:.3f} | "
                f"{gk['alpha_ren']:.3f} | **{gk['mse_ren_over_noren']:.2f}x** |"
            )

        lines += [
            "",
            "## 2. Quality & Real Production vLLM CUDA-Graph Speedup (64 NLL + 2,272 MCQ)",
            "| Policy | Cut% | Stage Mean k | ΔNLL (95% CI) | ARC-C | ARC-E | OBQA | Avg MCQ | CG Decode B1 | CG Decode B32 | Prefill B8×256 |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for pname, _, _, _ in policies:
            r = eval_results[pname]
            b1 = spd_idx.get(("decode_B1_L1", pname))
            b32 = spd_idx.get(("decode_B32_L1", pname))
            pf = spd_idx.get(("prefill_B8_L256", pname))
            b1_s = f"{b1['median_ms']:.2f}ms (**{b1['speedup']:.2f}x**)" if b1 else "—"
            b32_s = f"{b32['median_ms']:.2f}ms (**{b32['speedup']:.2f}x**)" if b32 else "—"
            pf_s = f"{pf['median_ms']:.1f}ms (**{pf['speedup']:.2f}x**)" if pf else "—"
            lines.append(
                f"| `{pname}` | {r['cut_pct']:.1f}% | {r['stage_k']} | "
                f"{r['delta_nll']:+.4f} [{r['delta_nll_ci'][0]:+.4f}, {r['delta_nll_ci'][1]:+.4f}] | "
                f"{r['arc_c']*100:.1f}% | {r['arc_e']*100:.1f}% | {r['obqa']*100:.1f}% | **{r['avg_mcq']*100:.1f}%** | "
                f"{b1_s} | {b32_s} | {pf_s} |"
            )

        lines += [
            "",
            "## 3. Full Production vLLM CUDA-Graph Speedup Table",
            "| Shape | Schedule | Cut% | Mean k | Median (ms) | IQR (ms) | Speedup | Tok/s |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in all_speed_rows:
            lines.append(
                f"| `{r['shape']}` | `{r['schedule']}` | {r['cut_pct']:.1f}% | {r['mean_k']:.2f} | "
                f"{r['median_ms']:.2f} | {r['iqr_ms']:.2f} | **{r['speedup']:.3f}x** | {r['tok_per_sec']:,.0f} |"
            )

        lines += ["", "## 4. Paired Bootstrap Contrasts (positive = first beats second)"]
        for cname, c in contrasts.items():
            lines.append(
                f"- `{cname}`: NLL `{c['nll_gain']:+.4f}` CI `[{c['nll_ci'][0]:+.4f}, {c['nll_ci'][1]:+.4f}]` "
                f"({c['nll_wins']}/{len(eval_seqs)} wins) | "
                f"Pooled MCQ `{c['mcq_gain_pp']:+.2f} pp` CI `[{c['mcq_ci_pp'][0]:+.2f}, {c['mcq_ci_pp'][1]:+.2f}]`"
            )

        report_md = "\n".join(lines) + "\n"
        (out_dir / "report_r4b_inkling2.md").write_text(report_md, encoding="utf-8")

        with run.phase("report", total=1, unit="file") as p_rep:
            print("\n========== REPORT (report_r4b_inkling2.md) ==========", flush=True)
            print(report_md, flush=True)
            print("========== END REPORT ==========", flush=True)
            p_rep.step(
                1,
                inkling_cg_b1_cut50_speedup=float(spd_idx[("decode_B1_L1", "cut50_iter_loopsched")]["speedup"]),
                inkling_cg_b32_cut50_speedup=float(spd_idx[("decode_B32_L1", "cut50_iter_loopsched")]["speedup"]),
                inkling_cg_b32_cut75_speedup=float(spd_idx[("decode_B32_L1", "cut75_iter_loopsched")]["speedup"]),
                inkling_cut50_noren_vs_renorm_nll=float(contrasts["cut50_unif_k4_noren_vs_cut50_unif_k4_renorm"]["nll_gain"]),
                inkling_cut50_iter_vs_unif_nll=float(contrasts["cut50_iter_loopsched_vs_cut50_unif_k4_noren"]["nll_gain"]),
                inkling_cut50_iter_vs_renorm_mcq=float(contrasts["cut50_iter_loopsched_vs_cut50_unif_k4_renorm"]["mcq_gain_pp"]),
            )
    finally:
        gpu_lock.release()


if __name__ == "__main__":
    main()
