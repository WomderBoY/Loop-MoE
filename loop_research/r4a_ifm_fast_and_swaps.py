"""Round 4-A: IFM CUDA-Graph + Triton Fused-MoE Acceleration, Stratified IterGreedy & m-Swap Pareto.

Direct child of R3 (51fb758b-8a03-4c56-b5f9-3d98fa3a49ed).
Key advances over R3:
  1. Zero-Conversion CUDA-Graph + Triton Fused-MoE Engine (loop_research/ifm_fast_engine.py):
     - Eliminates the ~240ms Eager Python dispatch overhead & index_select weight copies that caused
       -50% k=1 decode to look slower (0.94-0.97x) in R3!
     - Measures BOTH native_xllm baseline reference AND 100% fair apples-to-apples Triton+CUDAGraph
       decode (B=1, 8, 32) and fast grouped-GEMM prefill (B=8, L=256) wall-clock speedups!
  2. Stratified Candidate Pool Non-Additive IterGreedy:
     - Fixes R3's unstratified candidate pool bug (which pushed Huginn loop_0/loop_1 into top-8 ahead
       of ultra-redundant loop_2..5, stopping prematurely at 1 swap).
     - Draws k=0 candidates round-robin across the lowest-sensitivity recurrent stages with soft
       non-adjacency preference, verified on both calib_a and disjoint calib_b.
  3. Explicit m-Swap Latency-Quality Pareto Sweep at EXACT -50.0% (m in {0, 2, 4, 8, 12, 16}):
     - Evaluates each m-swap point on BOTH quality (64-seq Held-out NLL + 2,272 MCQ) AND real
       CUDA-Graph + Triton GPU latency, constructing the true 2D Latency-Quality Pareto frontier!
  4. Full-Spectrum Non-Additive Pure-k12 IterGreedy at -25% and -37.5%:
     - Directly tests whether visit-level non-additive pure-k12 search beats Layer-Only Static on
       Ouro-MoE as well as Huginn-MoE!
  5. Safe Shared-Server GPU Etiquette:
     - Honors EXIT_IF_NO_IDLE_GPU=1 (exits cleanly with GPU report after asset download if all GPUs
       are currently occupied by other users).
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
import torch.nn.functional as F

from loop_research import orx_run as orx
from loop_research.data_assets import (
    build_disjoint_text_code_splits,
    load_arc_challenge_samples,
    load_arc_easy_samples,
    load_openbookqa_samples,
)
from loop_research.gpu_utils import acquire_idle_gpu, query_gpus
from loop_research.ifm_fast_engine import (
    HAS_TRITON,
    IFMFastMoEExecutor,
    benchmark_ifm_cudagraph_triton,
    install_triton_rms_norm_shim,
    verify_fast_engine_exactness,
)
from loop_research.ifm_runtime import bootstrap_xllm, load_ifm_model_for_autograd
from loop_research.r3_loopsched_v2 import (
    LoopSchedV2Executor,
    build_visit_topology,
    calibrate_dynamic_threshold,
    compute_factorized_importance,
    eval_mcq_batched,
    eval_nll_batched,
    paired_bootstrap_ci,
    prepare_mcq_encoded,
    solve_pure_k12_schedule,
)


# ---------------------------------------------------------------------------
# 1. Stratified Non-Adjacent Candidate Pool & Multi-Swap Pareto Solver
# ---------------------------------------------------------------------------

def build_stratified_k0_order(
    topo: list[dict[str, Any]],
    stage_k0_map: dict[str, float],
    screen_k0_cost: np.ndarray,
    eligible_vids: list[int],
) -> list[int]:
    """Order eligible k=0 visits round-robin across the least-sensitive recurrent stages first.

    Prevents noisy single-visit screening from letting sensitive early loops (e.g. Huginn loop_0/1)
    crowd out ultra-redundant middle stages (loop_2..6).
    """
    by_stage: dict[str, list[int]] = {}
    for v in eligible_vids:
        st = topo[v]["stage"]
        by_stage.setdefault(st, []).append(v)
    for st in by_stage:
        by_stage[st].sort(key=lambda v: (float(screen_k0_cost[v]), v))

    # Sort stages ascending by noise-free stage-joint k=0 sensitivity
    stages_sorted = sorted(by_stage.keys(), key=lambda st: (float(stage_k0_map.get(st, 1e6)), st))

    # Tier stages: prioritize the lowest-sensitivity half (within 4x of minimum stage_k0) first!
    min_st = max(min(float(stage_k0_map.get(st, 1e6)) for st in stages_sorted), 1e-4) if stages_sorted else 1.0
    primary_stages = [st for st in stages_sorted if float(stage_k0_map.get(st, 1e6)) <= max(4.0 * min_st, 0.15)]
    if not primary_stages:
        primary_stages = stages_sorted[: max(1, len(stages_sorted) // 2)]
    secondary_stages = [st for st in stages_sorted if st not in primary_stages]

    ordered: list[int] = []
    for group in (primary_stages, secondary_stages):
        max_len = max((len(by_stage[st]) for st in group), default=0)
        for rank in range(max_len):
            for st in group:
                if rank < len(by_stage[st]):
                    ordered.append(by_stage[st][rank])
    return ordered


def _select_non_adjacent_pool(
    base_order: list[int],
    vk: list[int],
    *,
    exclude_vid: int = -1,
    pool_size: int = 10,
) -> list[int]:
    """Select top pool_size remaining k=1 candidates, prioritizing visits non-adjacent to existing k=0."""
    V = len(vk)
    non_adj: list[int] = []
    adj: list[int] = []
    for v in base_order:
        if v == exclude_vid or vk[v] != 1:
            continue
        has_adj_k0 = (v > 0 and vk[v - 1] == 0) or (v + 1 < V and vk[v + 1] == 0)
        if has_adj_k0:
            adj.append(v)
        else:
            non_adj.append(v)
    return (non_adj + adj)[:pool_size]


def solve_stratified_swap_pareto_and_full_spectrum(
    eval_a_fn,
    eval_b_fn,
    topo: list[dict[str, Any]],
    stage_k0_map: dict[str, float],
    importance: np.ndarray,
    screen_k0_cost: np.ndarray,
    eligible_k0_vids: list[int],
    *,
    max_swaps: int = 16,
    swap_checkpoints: tuple[int, ...] = (0, 2, 4, 8, 12, 16),
    cand_pool_size: int = 10,
    up_pool_size: int = 4,
) -> dict[str, Any]:
    """Run:
      1) Full-spectrum pure k in {1,2} non-additive IterGreedy for -25% (56 drops) & -37.5% (84 drops);
      2) Stratified non-adjacent m-swap Pareto sweep at EXACT -50.0% (m = 0..max_swaps), recording
         schedules at swap_checkpoints and selecting m_best strictly on disjoint calib_b;
      3) Stratified non-additive extension from m_best to -62.5% (84 active slots).
    """
    V = len(topo)

    # --- Part 1: Non-Additive Pure k in {1, 2} Greedy for -25% and -37.5% ---
    # Drop 4 visits per round from a candidate pool of 12 lowest-importance remaining k=2 visits
    vk_k12 = [2] * V
    imp_asc = sorted(range(V), key=lambda v: (float(importance[v]), v))
    target_k1_25 = int(round(2.0 * V * 0.25))    # 56 for V=112
    target_k1_38 = int(round(2.0 * V * 0.375))   # 84 for V=112
    vk_25_iter: list[int] | None = None
    vk_38_iter: list[int] | None = None
    dropped = 0
    while dropped < target_k1_38:
        rem_k2 = [v for v in imp_asc if vk_k12[v] == 2]
        # Candidate pool: lowest 12 remaining k=2 visits
        pool = rem_k2[:12]
        step_sz = min(4, target_k1_38 - dropped)
        if dropped < target_k1_25 and dropped + step_sz > target_k1_25:
            step_sz = target_k1_25 - dropped
        # Score each candidate individually on the CURRENT cumulative schedule (non-additive!)
        scored: list[tuple[float, int]] = []
        for v in pool:
            vk_try = list(vk_k12)
            vk_try[v] = 1
            scored.append((float(eval_a_fn(vk_try)), v))
        scored.sort(key=lambda t: (t[0], t[1]))
        for _, v_pick in scored[:step_sz]:
            vk_k12[v_pick] = 1
            dropped += 1
        if dropped == target_k1_25 and vk_25_iter is None:
            vk_25_iter = list(vk_k12)
    vk_38_iter = list(vk_k12)
    assert vk_25_iter is not None

    # --- Part 2: Stratified Non-Adjacent m-Swap Pareto Sweep at EXACT -50.0% ---
    strat_order = build_stratified_k0_order(topo, stage_k0_map, screen_k0_cost, eligible_k0_vids)
    up_order = sorted(range(V), key=lambda v: (-float(importance[v]), v))

    vk_cur = [1] * V
    a_0 = float(eval_a_fn(vk_cur))
    b_0 = float(eval_b_fn(vk_cur))

    swap_schedules: dict[int, list[int]] = {0: list(vk_cur)}
    swap_trace: list[dict[str, Any]] = [
        {"m": 0, "v_up": -1, "stage_up": "-", "v_dn": -1, "stage_dn": "-", "calib_a": a_0, "calib_b": b_0}
    ]
    best_m = 0
    best_b = b_0
    best_vk_50 = list(vk_cur)

    for m in range(1, max_swaps + 1):
        # Pick TRUE highest-marginal-gain v_up among top up_pool_size remaining k=1 visits!
        cands_up = [v for v in up_order if vk_cur[v] == 1][:up_pool_size]
        if not cands_up:
            break
        best_up = cands_up[0]
        if len(cands_up) > 1:
            best_up_a = 1e18
            for vu in cands_up:
                vk_try = list(vk_cur)
                vk_try[vu] = 2
                va = float(eval_a_fn(vk_try))
                if va < best_up_a:
                    best_up_a = va
                    best_up = vu

        cands_dn = _select_non_adjacent_pool(strat_order, vk_cur, exclude_vid=best_up, pool_size=cand_pool_size)
        if not cands_dn:
            break

        best_dn = -1
        best_cand_a = 1e18
        for vd in cands_dn:
            vk_try = list(vk_cur)
            vk_try[best_up] = 2
            vk_try[vd] = 0
            va = float(eval_a_fn(vk_try))
            if va < best_cand_a:
                best_cand_a = va
                best_dn = vd

        vk_cur[best_up] = 2
        vk_cur[best_dn] = 0
        b_m = float(eval_b_fn(vk_cur))

        swap_trace.append(
            {
                "m": m,
                "v_up": int(best_up),
                "stage_up": topo[best_up]["stage"],
                "v_dn": int(best_dn),
                "stage_dn": topo[best_dn]["stage"],
                "calib_a": best_cand_a,
                "calib_b": b_m,
            }
        )
        if m in swap_checkpoints:
            swap_schedules[m] = list(vk_cur)
        if b_m < best_b - 1e-4:
            best_b = b_m
            best_m = m
            best_vk_50 = list(vk_cur)

    # --- Part 3: Stratified Extension from best_vk_50 to -62.5% ---
    target_slots_62 = int(round(2.0 * V * (1.0 - 0.625)))
    vk_62 = list(best_vk_50)
    while sum(vk_62) > target_slots_62:
        cands = _select_non_adjacent_pool(strat_order, vk_62, pool_size=cand_pool_size)
        # Also allow downgrading lowest-importance k=2 visit if cheaper than adding k=0
        k2_vids = sorted([v for v in range(V) if vk_62[v] == 2], key=lambda v: (float(importance[v]), v))
        for vk2 in k2_vids[:2]:
            if vk2 not in cands:
                cands.append(vk2)
        if not cands:
            cands = sorted([v for v in range(V) if vk_62[v] > 0], key=lambda v: (float(importance[v]), v))[:cand_pool_size]

        best_v = -1
        best_a = 1e18
        for v in cands:
            vk_try = list(vk_62)
            vk_try[v] -= 1
            va = float(eval_a_fn(vk_try))
            if va < best_a:
                best_a = va
                best_v = v
        vk_62[best_v] -= 1

    return {
        "vk_25_iter": vk_25_iter,
        "vk_38_iter": vk_38_iter,
        "swap_schedules": swap_schedules,
        "best_m": best_m,
        "best_calib_b": best_b,
        "vk_50_best": best_vk_50,
        "vk_62_strat": vk_62,
        "swap_trace": swap_trace,
    }


# ---------------------------------------------------------------------------
# 2. Calibration + Full Per-Model R4-A Execution
# ---------------------------------------------------------------------------

@torch.inference_mode()
def calibrate_r4a(
    model: Any,
    topo: list[dict[str, Any]],
    calib_a: list[dict[str, Any]],
    calib_b: list[dict[str, Any]],
    device: torch.device,
    *,
    batch_size: int = 8,
    phase_ctx: Any = None,
) -> dict[str, Any]:
    V = len(topo)
    n_layers = len(model.layers)
    topo_map: dict[tuple[int, int], int] = {}
    seen = [0] * n_layers
    for t in topo:
        lid = t["layer_id"]
        topo_map[(lid, seen[lid])] = t["vid"]
        seen[lid] += 1

    # Collect calib ratios for cut38_unif_renorm baseline threshold
    visit_ratios_lists: list[list[np.ndarray]] = [[] for _ in range(V)]
    visits_cursor = [0] * n_layers
    orig_fwds: list[tuple[Any, Any]] = []
    for lid, layer in enumerate(model.layers):
        moe = layer.moe
        orig = moe.forward
        orig_fwds.append((moe, orig))

        def make_probe(lid_c: int, moe_c: Any, fn_c: Any):
            def wrapped(x_in, load_balancing_type=None):
                vn = visits_cursor[lid_c]
                visits_cursor[lid_c] = vn + 1
                vid = topo_map[(lid_c, vn)]
                x = moe_c.norm(x_in)
                r_scores, _r_idx, _tpe, _aux = moe_c.router(x, load_balancing_type)
                w_f = r_scores.reshape(-1, r_scores.shape[-1]).float()
                w_min = w_f.min(dim=-1).values
                w_max = w_f.max(dim=-1).values
                visit_ratios_lists[vid].append((w_min / w_max.clamp_min(1e-12)).cpu().numpy().astype(np.float32))
                return fn_c(x_in, load_balancing_type)
            return wrapped

        moe.forward = make_probe(lid, moe, orig)

    try:
        for s0 in range(0, len(calib_a), batch_size):
            batch = calib_a[s0 : s0 + batch_size]
            toks = torch.tensor([b["tokens"] for b in batch], device=device, dtype=torch.long)
            for i in range(n_layers):
                visits_cursor[i] = 0
            _ = model(toks[:, :-1], multi_segments=False, cache=None)
    finally:
        for moe, orig in orig_fwds:
            moe.forward = orig

    visit_ratios = [
        np.concatenate(visit_ratios_lists[v], axis=0) if visit_ratios_lists[v] else np.zeros(0, dtype=np.float32)
        for v in range(V)
    ]
    if phase_ctx is not None:
        phase_ctx.step(1)

    calib_fast = calib_a[: min(8, len(calib_a))]
    ex_base = LoopSchedV2Executor(model, topo, mode="native", track_stats=False)
    base_a = eval_nll_batched(model, calib_fast, ex_base, device, batch_size=batch_size)["nll"]

    def eval_a(vk: list[int]) -> float:
        ex = LoopSchedV2Executor(model, topo, mode="static", visit_k=vk, renorm_on_drop2=False, track_stats=False)
        return eval_nll_batched(model, calib_fast, ex, device, batch_size=batch_size)["nll"]

    def eval_b(vk: list[int]) -> float:
        ex = LoopSchedV2Executor(model, topo, mode="static", visit_k=vk, renorm_on_drop2=False, track_stats=False)
        return eval_nll_batched(model, calib_b, ex, device, batch_size=batch_size)["nll"]

    stage_order: list[str] = []
    for t in topo:
        if t["stage"] not in stage_order:
            stage_order.append(t["stage"])

    stage_per_visit_sens: dict[str, float] = {}
    stage_k0_map: dict[str, float] = {}
    for st in stage_order:
        vids_s = [t["vid"] for t in topo if t["stage"] == st]
        d1 = eval_a([1 if v in vids_s else 2 for v in range(V)]) - base_a
        d0 = eval_a([0 if v in vids_s else 2 for v in range(V)]) - base_a
        stage_per_visit_sens[st] = max(float(d1), 1e-4) / float(len(vids_s))
        stage_k0_map[st] = float(d0)

    layer_per_visit_sens = np.zeros(n_layers, dtype=np.float64)
    for lid in range(n_layers):
        vids_l = [t["vid"] for t in topo if t["layer_id"] == lid]
        d_l = eval_a([1 if v in vids_l else 2 for v in range(V)]) - base_a
        layer_per_visit_sens[lid] = max(float(d_l), 1e-4) / float(len(vids_l))
    if phase_ctx is not None:
        phase_ctx.step(2)

    # Factorized importance (alpha=0.5 balanced prior for candidate ordering; IterGreedy uses TRUE cumulative NLL!)
    imp_fact = compute_factorized_importance(topo, stage_per_visit_sens, layer_per_visit_sens, alpha=0.5)
    imp_layer = compute_factorized_importance(topo, stage_per_visit_sens, layer_per_visit_sens, alpha=0.0)

    # Screen single-visit k=0 on eligible non-boundary recurrent visits
    eligible_k0_vids = [t["vid"] for t in topo if t["is_rec"] and stage_k0_map[t["stage"]] < 2.0]
    if len(eligible_k0_vids) < 32:
        sorted_st = sorted(stage_order, key=lambda s: stage_k0_map[s])
        keep_st = set(sorted_st[:-2]) if len(sorted_st) > 2 else set(sorted_st)
        eligible_k0_vids = [t["vid"] for t in topo if t["stage"] in keep_st]

    screen_k0_cost = np.full(V, 1e6, dtype=np.float64)
    for vid in eligible_k0_vids:
        vk_s = [2] * V
        vk_s[vid] = 0
        screen_k0_cost[vid] = max(0.0, eval_a(vk_s) - base_a)
    if phase_ctx is not None:
        phase_ctx.step(3)

    solver_out = solve_stratified_swap_pareto_and_full_spectrum(
        eval_a_fn=eval_a,
        eval_b_fn=eval_b,
        topo=topo,
        stage_k0_map=stage_k0_map,
        importance=imp_fact,
        screen_k0_cost=screen_k0_cost,
        eligible_k0_vids=eligible_k0_vids,
        max_swaps=16,
        swap_checkpoints=(0, 2, 4, 8, 12, 16),
        cand_pool_size=10,
        up_pool_size=4,
    )
    if phase_ctx is not None:
        phase_ctx.step(4)

    gc.collect()
    torch.cuda.empty_cache()

    return {
        "stage_k0_map": stage_k0_map,
        "stage_per_visit_sens": stage_per_visit_sens,
        "layer_per_visit_sens": layer_per_visit_sens,
        "imp_layer": imp_layer,
        "visit_ratios": visit_ratios,
        "solver": solver_out,
    }


def run_model_r4a(
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
    print(f"\n[r4a:{model_key}] Loading {artifact_dir} ...", flush=True)
    model, tokenizer, _mcfg = load_ifm_model_for_autograd(
        artifact_dir, device=str(device), attention_backend="sdpa"
    )
    n_layers = len(model.layers)
    topo = build_visit_topology(
        n_layers, int(model.loop_start_layer), int(model.loop_end_layers), int(model.loop_times)
    )
    V = len(topo)

    splits = build_disjoint_text_code_splits(
        dataset_dir,
        lambda t: tokenizer.encode(t, bos=True, eos=False),
        seq_len=int(cfg.get("seq_len", 256)),
        n_calib_a=int(cfg.get("n_calib_a", 16)),
        n_calib_b=int(cfg.get("n_calib_b", 8)),
        n_adapt_train=4,
        n_eval_heldout=int(cfg.get("n_eval", 64)),
        seed=seed,
    )
    calib_a, calib_b, eval_seqs = splits["calib_a"], splits["calib_b"], splits["eval_heldout"]
    mcq_enc = {k: prepare_mcq_encoded(v, tokenizer) for k, v in mcq_raw.items()}
    raw_pad = getattr(tokenizer, "pad_id", None)
    if raw_pad is None or int(raw_pad) < 0:
        raw_pad = getattr(tokenizer, "eos_id", None)
    if raw_pad is None or int(raw_pad) < 0:
        raw_pad = getattr(tokenizer, "bos_id", 0)
    pad_id = max(0, int(raw_pad))

    # 1) Verify Fast Engine (Triton RMSNorm + Triton Zero-Copy MoE + Static CUDA Graph) vs Native xLLM
    with run_obj.phase(f"{model_key}_verify", total=1, unit="check") as pv:
        # Measure unmodified native xLLM decode_B1_L1 reference latency BEFORE Triton norm upgrade
        with torch.inference_mode():
            inp_ref_b1 = torch.randint(10, 5000, (1, 1), device=device, dtype=torch.long)
            for _ in range(5):
                _ = model(inp_ref_b1, multi_segments=False, cache=None)
            torch.cuda.synchronize(device)
            nat_ms_list: list[float] = []
            for _ in range(20):
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                _ = model(inp_ref_b1, multi_segments=False, cache=None)
                e1.record()
                torch.cuda.synchronize(device)
                nat_ms_list.append(float(e0.elapsed_time(e1)))
            native_b1_ms = float(np.median(nat_ms_list))

        used_triton_norm = install_triton_rms_norm_shim()
        vinfo = verify_fast_engine_exactness(model, topo, eval_seqs[0]["tokens"], device)
        print(
            f"[r4a:{model_key}] FastEngine Verify: has_triton={vinfo['has_triton']}, triton_norm={used_triton_norm}, "
            f"max_diff(B1)={vinfo['max_diff_b1']:.3e} (mean={vinfo['mean_diff_b1']:.3e}), "
            f"max_diff(B8)={vinfo['max_diff_b8']:.3e}, top1_match={vinfo['top1_match_b1']}, "
            f"cudagraph_ok={vinfo['cudagraph_ok']} (replay_diff={vinfo['cudagraph_replay_diff']:.2e}, "
            f"err={vinfo['cudagraph_error']!r}) | native_xllm B1={native_b1_ms:.2f}ms",
            flush=True,
        )
        if not vinfo["top1_match_b1"] or vinfo["mean_diff_b1"] > 0.05:
            raise RuntimeError(f"Fast engine numerical mismatch: {vinfo}")
        pv.step(1)

    gc.collect()
    torch.cuda.empty_cache()

    # 2) Calibration: Stratified IterGreedy + Full-Spectrum Pure-k12 + m-Swap Trajectory
    with run_obj.phase(f"{model_key}_calib", total=4, unit="steps") as pc:
        t0_c = time.time()
        calib = calibrate_r4a(model, topo, calib_a, calib_b, device, batch_size=8, phase_ctx=pc)
        sol = calib["solver"]
        print(
            f"[r4a:{model_key}] Calibration done in {time.time() - t0_c:.1f}s | "
            f"stage_k0={ {k: round(v, 3) for k, v in calib['stage_k0_map'].items()} }",
            flush=True,
        )
        print(f"[r4a:{model_key}] Stratified -50% Swap Trajectory (best_m={sol['best_m']}, calib_b={sol['best_calib_b']:.4f}):", flush=True)
        for sw in sol["swap_trace"]:
            print(
                f"    m={sw['m']:2d}: +{sw['v_up']:3d}({sw['stage_up']:7s}) / -{sw['v_dn']:3d}({sw['stage_dn']:7s}) "
                f"-> calib_a={sw['calib_a']:.4f}, calib_b={sw['calib_b']:.4f}",
                flush=True,
            )

    # 3) Build Conditions (Deduplicating best_m if already in {0, 2, 4, 8, 12, 16})
    vg_one = [1.0] * V
    vk25_layer = solve_pure_k12_schedule(calib["imp_layer"], 0.25)
    vk38_layer = solve_pure_k12_schedule(calib["imp_layer"], 0.375)
    tau38_ren = calibrate_dynamic_threshold(calib["visit_ratios"], [2] * V, np.ones(V), 0.375)

    def mk_st(name: str, vk: list[int], renorm: bool = False) -> dict[str, Any]:
        return {"name": name, "mode": "static", "vk": vk, "renorm": renorm, "w": None, "tau": -1.0}

    conditions: list[dict[str, Any]] = [
        {"name": "bf16_base", "mode": "native", "vk": [2] * V, "renorm": False, "w": None, "tau": -1.0},
        mk_st("cut25_layer_only_static", vk25_layer),
        mk_st("cut25_iter_k12", sol["vk_25_iter"]),
        {"name": "cut38_unif_renorm", "mode": "dynamic", "vk": [2] * V, "renorm": True, "w": np.ones(V), "tau": tau38_ren},
        mk_st("cut38_layer_only_static", vk38_layer),
        mk_st("cut38_iter_k12", sol["vk_38_iter"]),
        mk_st("cut50_all_top1_renorm", [1] * V, renorm=True),
        mk_st("cut50_swap_m0_noren", sol["swap_schedules"][0]),
        mk_st("cut50_swap_m2", sol["swap_schedules"][2]),
        mk_st("cut50_swap_m4", sol["swap_schedules"][4]),
        mk_st("cut50_swap_m8", sol["swap_schedules"][8]),
        mk_st("cut50_swap_m12", sol["swap_schedules"][12]),
        mk_st("cut50_swap_m16", sol["swap_schedules"][16]),
    ]
    best_m = int(sol["best_m"])
    if best_m not in (0, 2, 4, 8, 12, 16):
        conditions.append(mk_st(f"cut50_strat_best_m{best_m}", sol["vk_50_best"]))
    conditions.append(mk_st("cut62_strat_iter", sol["vk_62_strat"]))

    best_50_name = (
        "cut50_swap_m0_noren"
        if best_m == 0
        else (f"cut50_swap_m{best_m}" if best_m in (2, 4, 8, 12, 16) else f"cut50_strat_best_m{best_m}")
    )

    # 4) Real GPU Wall-Clock Speedup Benchmark FIRST (under CUDA-Graph + Triton Fused-MoE!)
    bench_scheds: dict[str, tuple[list[int], list[float]]] = {
        "bf16_base": ([2] * V, vg_one),
        "cut25_iter_k12": (sol["vk_25_iter"], vg_one),
        "cut38_iter_k12": (sol["vk_38_iter"], vg_one),
        "cut50_swap_m0_noren": (sol["swap_schedules"][0], vg_one),
        "cut50_swap_m2": (sol["swap_schedules"][2], vg_one),
        "cut50_swap_m4": (sol["swap_schedules"][4], vg_one),
        "cut50_swap_m8": (sol["swap_schedules"][8], vg_one),
        "cut50_swap_m12": (sol["swap_schedules"][12], vg_one),
        "cut50_swap_m16": (sol["swap_schedules"][16], vg_one),
        "cut62_strat_iter": (sol["vk_62_strat"], vg_one),
    }
    if best_m not in (0, 2, 4, 8, 12, 16):
        bench_scheds[best_50_name] = (sol["vk_50_best"], vg_one)

    with run_obj.phase(f"{model_key}_speed", total=1, unit="bench") as ps:
        speed_rows = benchmark_ifm_cudagraph_triton(
            model, topo, bench_scheds, device, use_cudagraph=bool(vinfo["cudagraph_ok"])
        )
        print(
            f"\n[r4a:{model_key}] REAL GPU Wall-Clock Speedup (CUDA-Graph + Zero-Copy Triton Fused-MoE; "
            f"Native Eager xLLM B1 ref={native_b1_ms:.2f}ms):",
            flush=True,
        )
        for sr in speed_rows:
            print(
                f"  [{sr['backend']:17s}] {sr['shape']:15s} | {sr['schedule']:22s} "
                f"({sr['k0_count']:2d}/{sr['k1_count']:3d}/{sr['k2_count']:3d}, cut={sr['cut_pct']:4.1f}%) | "
                f"{sr['median_ms']:6.2f} ms (IQR {sr['iqr_ms']:.2f}) | {sr['speedup']:.3f}x | {sr['tok_per_sec']:,.0f} tok/s",
                flush=True,
            )
            ps.row(
                f"{model_key}_speed",
                backend=sr["backend"],
                shape=sr["shape"],
                schedule=sr["schedule"],
                cut_pct=sr["cut_pct"],
                median_ms=sr["median_ms"],
                speedup=sr["speedup"],
            )
        b1_base_fast = next(r["median_ms"] for r in speed_rows if r["shape"] == "decode_B1_L1" and r["schedule"] == "bf16_base")
        b1_m0_sp = next(r["speedup"] for r in speed_rows if r["shape"] == "decode_B1_L1" and r["schedule"] == "cut50_swap_m0_noren")
        b1_m8_sp = next(r["speedup"] for r in speed_rows if r["shape"] == "decode_B1_L1" and r["schedule"] == "cut50_swap_m8")
        print(
            f"[r4a:{model_key}] Engine speedup vs eager xLLM: {native_b1_ms:.2f}ms -> {b1_base_fast:.2f}ms "
            f"({native_b1_ms / max(b1_base_fast, 1e-6):.2f}x)! | Fair CG B1 speedup: m0={b1_m0_sp:.3f}x, m8={b1_m8_sp:.3f}x",
            flush=True,
        )
        ps.step(1, latest_b1_m0_speedup=b1_m0_sp, latest_b1_m8_speedup=b1_m8_sp)

    # 5) Full Quality Evaluation (64-seq Held-out NLL + 2,272 MCQ items)
    results: dict[str, dict[str, Any]] = {}
    base_per_seq: np.ndarray | None = None
    with run_obj.phase(f"{model_key}_eval", total=len(conditions), unit="conds") as pe:
        for idx_c, c in enumerate(conditions, start=1):
            t_c0 = time.time()
            ex = LoopSchedV2Executor(
                model, topo, mode=c["mode"], visit_k=c["vk"], visit_gamma=vg_one,
                renorm_on_drop2=c["renorm"], visit_weights=c["w"], dyn_threshold=c["tau"],
            )
            nll_res = eval_nll_batched(model, eval_seqs, ex, device, batch_size=8)
            mcq_accs: dict[str, float] = {}
            mcq_flags: dict[str, list[int]] = {}
            for ds_name, enc_items in mcq_enc.items():
                acc, fl = eval_mcq_batched(model, enc_items, ex, device, batch_size=32, pad_token_id=pad_id)
                mcq_accs[ds_name] = acc
                mcq_flags[ds_name] = fl
            avg_mcq = float(np.mean(list(mcq_accs.values())))

            if c["name"] == "bf16_base":
                base_per_seq = np.asarray(nll_res["per_seq"], dtype=np.float64)
            assert base_per_seq is not None
            dnll, dlo, dhi = paired_bootstrap_ci(np.asarray(nll_res["per_seq"], dtype=np.float64) - base_per_seq)
            elapsed_c = time.time() - t_c0

            rec = {
                "name": c["name"],
                "cut_pct": nll_res["cut_pct"],
                "nll": nll_res["nll"],
                "delta_nll": dnll,
                "delta_nll_ci": [dlo, dhi],
                "per_seq": nll_res["per_seq"],
                "arc_c": mcq_accs["arc_c"],
                "arc_e": mcq_accs["arc_e"],
                "obqa": mcq_accs["obqa"],
                "avg_mcq": avg_mcq,
                "mcq_flags": mcq_flags,
                "k0_count": int(sum(1 for k in c["vk"] if k == 0)),
                "k1_count": int(sum(1 for k in c["vk"] if k == 1)),
                "k2_count": int(sum(1 for k in c["vk"] if k == 2)),
                "elapsed_s": elapsed_c,
            }
            results[c["name"]] = rec
            print(
                f"[r4a:{model_key}] {c['name']:24s} cut={rec['cut_pct']:5.1f}% | NLL={rec['nll']:.4f} (Δ={dnll:+.4f}) "
                f"| ARC-C={mcq_accs['arc_c']*100:5.1f}% ARC-E={mcq_accs['arc_e']*100:5.1f}% OBQA={mcq_accs['obqa']*100:5.1f}% "
                f"Avg={avg_mcq*100:5.1f}% | ({rec['k0_count']}/{rec['k1_count']}/{rec['k2_count']}) | t={elapsed_c:.1f}s",
                flush=True,
            )
            pe.row(
                f"{model_key}_results",
                policy=c["name"],
                cut_pct=rec["cut_pct"],
                delta_nll=dnll,
                arc_c=mcq_accs["arc_c"],
                arc_e=mcq_accs["arc_e"],
                obqa=mcq_accs["obqa"],
                avg_mcq=avg_mcq,
            )
            pe.step(idx_c, latest_avg_mcq=avg_mcq, latest_dnll=dnll)

    # Paired bootstrap contrasts
    def contrast(a: str, b: str) -> dict[str, Any]:
        ra, rb = results[a], results[b]
        nll_gain = np.asarray(rb["per_seq"], dtype=np.float64) - np.asarray(ra["per_seq"], dtype=np.float64)
        ng_m, ng_lo, ng_hi = paired_bootstrap_ci(nll_gain)
        fa = np.concatenate([np.asarray(ra["mcq_flags"][k], dtype=np.float64) for k in ("arc_c", "arc_e", "obqa")])
        fb = np.concatenate([np.asarray(rb["mcq_flags"][k], dtype=np.float64) for k in ("arc_c", "arc_e", "obqa")])
        mg_m, mg_lo, mg_hi = paired_bootstrap_ci((fa - fb) * 100.0)
        return {
            "nll_gain": ng_m,
            "nll_ci": [ng_lo, ng_hi],
            "nll_wins": f"{int(np.sum(nll_gain > 0))}/{len(nll_gain)}",
            "mcq_pooled_gain_pp": mg_m,
            "mcq_pooled_ci_pp": [mg_lo, mg_hi],
        }

    contrasts = {
        "cut25_iter_k12_vs_layer_only": contrast("cut25_iter_k12", "cut25_layer_only_static"),
        "cut38_iter_k12_vs_layer_only": contrast("cut38_iter_k12", "cut38_layer_only_static"),
        "cut38_iter_k12_vs_unif_renorm": contrast("cut38_iter_k12", "cut38_unif_renorm"),
        "cut50_best_vs_m0_noren": contrast(best_50_name, "cut50_swap_m0_noren"),
        "cut50_best_vs_all_top1_renorm": contrast(best_50_name, "cut50_all_top1_renorm"),
        "cut50_m4_vs_m0_noren": contrast("cut50_swap_m4", "cut50_swap_m0_noren"),
        "cut50_m8_vs_m0_noren": contrast("cut50_swap_m8", "cut50_swap_m0_noren"),
        "cut50_m8_vs_all_top1_renorm": contrast("cut50_swap_m8", "cut50_all_top1_renorm"),
    }

    del model
    gc.collect()
    torch.cuda.empty_cache()

    compact_results = {
        k: {kk: vv for kk, vv in v.items() if kk not in ("mcq_flags", "per_seq")}
        for k, v in results.items()
    }
    return {
        "model_key": model_key,
        "native_xllm_b1_ms": native_b1_ms,
        "verify_info": vinfo,
        "best_m": best_m,
        "best_50_name": best_50_name,
        "swap_trace": sol["swap_trace"],
        "results": compact_results,
        "speed_rows": speed_rows,
        "contrasts": contrasts,
    }


def write_r4a_report(all_out: dict[str, Any], out_path: Path) -> str:
    lines: list[str] = []
    lines.append("# Round 4-A: IFM CUDA-Graph + Triton Fused-MoE & Stratified m-Swap Pareto\n")
    lines.append(f"- GPU: `{all_out['gpu_index']}` ({all_out['gpu_name']}), elapsed `{all_out['elapsed_s']:.1f}s`\n")
    for mk, mo in all_out["models"].items():
        b1_fast = next(r["median_ms"] for r in mo["speed_rows"] if r["shape"] == "decode_B1_L1" and r["schedule"] == "bf16_base")
        lines.append(
            f"\n## Model: `{mk}` (Native Eager B1=`{mo['native_xllm_b1_ms']:.2f}ms` -> "
            f"Triton+CG B1=`{b1_fast:.2f}ms` = **{mo['native_xllm_b1_ms']/max(b1_fast,1e-6):.2f}x**; "
            f"calib_b optimal `best_m={mo['best_m']}`)\n"
        )
        lines.append("### 1. Quality vs Real CUDA-Graph+Triton Decode Latency (m-Swap Pareto + Full Spectrum)")
        lines.append("| Policy | Cut% | (k0/k1/k2) | ΔNLL (95% CI) | ARC-C | ARC-E | OBQA | Avg MCQ | CG Decode B1 | CG Decode B8 | Prefill B8×256 |")
        lines.append("|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|")
        sp_map = {(r["schedule"], r["shape"]): r for r in mo["speed_rows"]}
        for r in mo["results"].values():
            lo, hi = r["delta_nll_ci"]
            b1 = sp_map.get((r["name"], "decode_B1_L1"))
            b8 = sp_map.get((r["name"], "decode_B8_L1"))
            pf = sp_map.get((r["name"], "prefill_B8_L256"))
            s_b1 = f"{b1['median_ms']:.1f}ms (**{b1['speedup']:.2f}x**)" if b1 else "—"
            s_b8 = f"{b8['median_ms']:.1f}ms (**{b8['speedup']:.2f}x**)" if b8 else "—"
            s_pf = f"{pf['median_ms']:.1f}ms (**{pf['speedup']:.2f}x**)" if pf else "—"
            lines.append(
                f"| `{r['name']}` | {r['cut_pct']:.1f}% | {r['k0_count']}/{r['k1_count']}/{r['k2_count']} | "
                f"{r['delta_nll']:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
                f"{r['arc_c']*100:.1f}% | {r['arc_e']*100:.1f}% | {r['obqa']*100:.1f}% | **{r['avg_mcq']*100:.1f}%** | "
                f"{s_b1} | {s_b8} | {s_pf} |"
            )

        lines.append("\n### 2. Full Speedup Table (Decode B1/B8/B32 + Prefill B8×256)")
        lines.append("| Backend | Shape | Schedule | (k0/k1/k2) | Median (ms) | IQR (ms) | Speedup | Tok/s |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|")
        for sr in mo["speed_rows"]:
            lines.append(
                f"| `{sr['backend']}` | `{sr['shape']}` | `{sr['schedule']}` | "
                f"{sr['k0_count']}/{sr['k1_count']}/{sr['k2_count']} | {sr['median_ms']:.2f} | {sr['iqr_ms']:.2f} | "
                f"**{sr['speedup']:.3f}x** | {sr['tok_per_sec']:,.0f} |"
            )

        lines.append("\n### 3. Paired Bootstrap Contrasts (positive = first beats second)")
        for cn, cv in mo["contrasts"].items():
            nlo, nhi = cv["nll_ci"]
            mlo, mhi = cv["mcq_pooled_ci_pp"]
            lines.append(
                f"- `{cn}`: NLL `{cv['nll_gain']:+.4f}` CI `[{nlo:+.4f}, {nhi:+.4f}]` ({cv['nll_wins']} wins) | "
                f"Pooled MCQ `{cv['mcq_pooled_gain_pp']:+.2f} pp` CI `[{mlo:+.2f}, {mhi:+.2f}]`"
            )

    md = "\n".join(lines) + "\n"
    out_path.write_text(md, encoding="utf-8")
    return md


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--xllm-loop-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    exit_if_busy = os.environ.get("EXIT_IF_NO_IDLE_GPU", "0") == "1"
    wait_sec = 10 if exit_if_busy else int(os.environ.get("GPU_WAIT_SEC", "300"))
    try:
        gpu_idx, gpu_lock, gpu_status = acquire_idle_gpu(cfg.get("preferred_gpu"), wait_timeout_sec=wait_sec)
    except RuntimeError as exc:
        gpus = query_gpus()
        print("[r4a] No idle GPU currently available (all GPUs occupied by other jobs):", flush=True)
        for g in gpus:
            print(
                f"  GPU {g.index} ({g.name}): mem={g.mem_used_mib}/{g.mem_total_mib}MiB, "
                f"util={g.util_pct}%, pids={g.active_pids}",
                flush=True,
            )
        if exit_if_busy:
            print("[r4a] EXIT_IF_NO_IDLE_GPU=1 set -> assets downloaded & code synced cleanly; exiting 0 to await user notice.", flush=True)
            (out_dir / "gpu_busy_status.json").write_text(
                json.dumps({"status": "waiting_for_idle_gpu", "reason": str(exc)}, indent=2)
            )
            return
        raise

    try:
        device = torch.device(f"cuda:{gpu_idx}")
        torch.cuda.set_device(device)
        print(
            f"[r4a] Acquired idle GPU {gpu_idx}: {gpu_status.name} "
            f"(mem={gpu_status.mem_used_mib}MiB, util={gpu_status.util_pct}%)",
            flush=True,
        )
        boot = bootstrap_xllm(args.xllm_loop_dir, attention_backend="sdpa")
        print(f"[r4a] xllm bootstrap: {json.dumps(boot)}", flush=True)

        seed = int(cfg.get("seed", 20261126))
        mcq_raw = {
            "arc_c": load_arc_challenge_samples(args.dataset_dir, n_samples=int(cfg.get("n_arc_c", 0)), seed=seed),
            "arc_e": load_arc_easy_samples(args.dataset_dir, n_samples=int(cfg.get("n_arc_e", 600)), seed=seed),
            "obqa": load_openbookqa_samples(args.dataset_dir, n_samples=int(cfg.get("n_obqa", 500)), seed=seed),
        }
        print(f"[r4a] Loaded MCQ suites: ARC-C={len(mcq_raw['arc_c'])}, ARC-E={len(mcq_raw['arc_e'])}, OBQA={len(mcq_raw['obqa'])}", flush=True)

        model_specs = cfg["models"]
        plan = [
            {"phase": "setup", "of": 1, "unit": "env"},
            {"phase": "download", "of": 1, "unit": "assets"},
        ]
        for mk in model_specs:
            plan.extend(
                [
                    {"phase": f"{mk}_verify", "of": 1, "unit": "check"},
                    {"phase": f"{mk}_calib", "of": 4, "unit": "steps"},
                    {"phase": f"{mk}_speed", "of": 1, "unit": "bench"},
                    {"phase": f"{mk}_eval", "of": 15, "unit": "conds"},
                ]
            )
        plan.append({"phase": "report", "of": 1, "unit": "file"})
        orx.plan(plan)
        run_obj = orx.progress(total=1, unit="run")

        all_out: dict[str, Any] = {
            "gpu_index": gpu_idx,
            "gpu_name": gpu_status.name,
            "models": {},
        }
        for mk, mspec in model_specs.items():
            all_out["models"][mk] = run_model_r4a(
                mk, mspec,
                model_dir=Path(args.model_dir),
                dataset_dir=Path(args.dataset_dir),
                mcq_raw=mcq_raw,
                cfg=cfg,
                device=device,
                run_obj=run_obj,
            )

        all_out["elapsed_s"] = time.time() - t0
        with run_obj.phase("report", total=1, unit="file") as pr:
            (out_dir / "results_r4a_ifm_fast.json").write_text(json.dumps(all_out, indent=2), encoding="utf-8")
            md = write_r4a_report(all_out, out_dir / "report_r4a_ifm_fast.md")
            print("\n========== REPORT (report_r4a_ifm_fast.md) ==========")
            print(md, flush=True)
            print("========== END REPORT ==========", flush=True)

            summary_m: dict[str, float] = {}
            for mk, mo in all_out["models"].items():
                sp_map = {(r["schedule"], r["shape"]): r for r in mo["speed_rows"]}
                summary_m[f"{mk}_cg_b1_m0_speedup"] = sp_map[("cut50_swap_m0_noren", "decode_B1_L1")]["speedup"]
                summary_m[f"{mk}_cg_b1_m8_speedup"] = sp_map[("cut50_swap_m8", "decode_B1_L1")]["speedup"]
                summary_m[f"{mk}_cg_b1_cut62_speedup"] = sp_map[("cut62_strat_iter", "decode_B1_L1")]["speedup"]
                summary_m[f"{mk}_prefill_m8_speedup"] = sp_map[("cut50_swap_m8", "prefill_B8_L256")]["speedup"]
                summary_m[f"{mk}_cut38_iter_vs_layer_nll"] = mo["contrasts"]["cut38_iter_k12_vs_layer_only"]["nll_gain"]
                summary_m[f"{mk}_cut50_best_vs_m0_nll"] = mo["contrasts"]["cut50_best_vs_m0_noren"]["nll_gain"]
                summary_m[f"{mk}_cut50_best_vs_renorm_mcq"] = mo["contrasts"]["cut50_best_vs_all_top1_renorm"]["mcq_pooled_gain_pp"]
                summary_m[f"{mk}_best_m"] = float(mo["best_m"])
            pr.step(1, **summary_m)
    finally:
        gpu_lock.release()


if __name__ == "__main__":
    main()
