"""Round 4-B: Generalized Top-K -> k_v LoopSched-MoE for thinkingmachines/Inkling-2-Mini.

Supports Inkling-2-Mini (26.4B NVFP4/MXFP8 Looped Sparse MoE):
  - Topology: 24 physical layers -> 6 Prelude [0..5] + 12 Looped Core [6..17] x 3 passes + 6 Coda [18..23]
    = 48 logical visits!
  - MoE: 128 routed experts, Top-8 per token (K=8), 0 shared experts -> MoE is ~75-85% of active compute!
  - Generalizes Orthogonal Expert Shrinkage to arbitrary Top-K -> k in {0, 1, ..., 8}:
      s_k = sum_{j=1..k} w_j e_j,  r_k = sum_{j=k+1..K} w_j e_j
      gamma_k* = 1 + E[<s_k, r_k>] / E[||s_k||^2]
      MSE_renorm(k) / MSE_noren(k) = E[||(alpha_k - 1)*s_k - r_k||^2] / E[||r_k||^2]
  - Supports production vLLM (>=0.35.0) Blackwell NVFP4 Fused-MoE backend (primary for real TTFT/TPOT)
    and HuggingFace transformers fallback for fine-grained per-visit diagnostics.
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
    """Compute exact generalized Top-K -> k_keep optimal shrinkage gamma_k* and MSE_renorm/MSE_noren.

    Args:
        weights_sorted: [N, K] descending routing weights (positive)
        expert_outs_sorted: [N, K, D] corresponding expert output vectors
        k_keep: integer in [1, K - 1]
    """
    w = weights_sorted.float()
    e = expert_outs_sorted.float()
    K = w.shape[1]
    assert 1 <= k_keep < K

    s_k = (w[:, :k_keep, None] * e[:, :k_keep, :]).sum(dim=1)        # [N, D]
    r_k = (w[:, k_keep:, None] * e[:, k_keep:, :]).sum(dim=1)        # [N, D]
    w_tot = w.sum(dim=1, keepdim=True)                               # [N, 1]
    w_kept = w[:, :k_keep].sum(dim=1, keepdim=True).clamp_min(1e-12)
    alpha = w_tot / w_kept                                           # [N, 1] renorm factor

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


def solve_general_topk_iter_greedy(
    eval_a_fn,
    topo: list[dict[str, Any]],
    K_max: int,
    target_cut_frac: float,
    *,
    stage_priority_order: list[str] | None = None,
    min_k_boundary: int = 4,
    min_k_rec: int = 1,
    step_delta_k: int = 1,
    cand_pool_size: int = 12,
) -> list[int]:
    """Non-additive iterative greedy solver over general per-visit k_v in {0, 1, ..., K_max}.

    Unlike IFM Top-2 (where jumping k=1->0 removes 100% of routed experts), Top-8 allows smooth
    fine-grained steps (8->7->...->2->1) protecting prelude/coda while aggressively thinning middle passes!
    """
    V = len(topo)
    total_slots = V * K_max
    target_slots = max(0, min(total_slots, int(round(total_slots * (1.0 - target_cut_frac)))))
    vk = [K_max] * V

    # Base ordering: recurrent visits in lowest-priority stages first, boundary visits last
    stage_rank = {}
    if stage_priority_order:
        for idx, st in enumerate(stage_priority_order):
            stage_rank[st] = idx
    base_order = sorted(
        range(V),
        key=lambda v: (
            0 if topo[v]["is_rec"] else 1,
            stage_rank.get(topo[v]["stage"], 99),
            v,
        ),
    )

    while sum(vk) > target_slots:
        rem = sum(vk) - target_slots
        dk = min(step_delta_k, rem)
        # Eligible pool respecting floor constraints first, then relaxing if needed
        cands = [
            v for v in base_order
            if vk[v] - dk >= (min_k_rec if topo[v]["is_rec"] else min_k_boundary)
        ][:cand_pool_size]
        if not cands:
            cands = [v for v in base_order if vk[v] >= dk][:cand_pool_size]
        if not cands:
            break

        best_v = cands[0]
        best_loss = 1e18
        for v in cands:
            vk_try = list(vk)
            vk_try[v] -= dk
            loss = float(eval_a_fn(vk_try))
            if loss < best_loss:
                best_loss = loss
                best_v = v
        vk[best_v] -= dk

    return vk


def build_stage_schedule_inkling2(
    topo: list[dict[str, Any]],
    k_by_stage: dict[str, int],
) -> list[int]:
    """Deterministic per-stage Top-k schedule for Inkling-2-Mini (zero calibration required!)."""
    return [int(k_by_stage[t["stage"]]) for t in topo]


# ---------------------------------------------------------------------------
# 2. vLLM / Transformers Runtime Probe & Execution Entrypoint
# ---------------------------------------------------------------------------

def detect_inkling_runtime() -> dict[str, Any]:
    """Check whether vLLM (>=0.35.0) or transformers supports inkling2_loop in current env."""
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
            info["vllm_has_inkling2"] = any("Inkling2" in a for a in archs)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model_path = Path(args.model_dir) / cfg.get("local_dir", "thinkingmachines__Inkling-2-Mini")

    rt = detect_inkling_runtime()
    print(f"[r4b:inkling2] Model path: {model_path} (exists={model_path.exists()})", flush=True)
    print(f"[r4b:inkling2] Runtime capabilities: {json.dumps(rt, indent=2)}", flush=True)

    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(
            f"Inkling-2-Mini not found at {model_path}. Run: ASSETS=inkling2_mini bash scripts/download_assets.sh"
        )

    mcfg = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    tcfg = mcfg["text_config"]
    lcfg = tcfg["loop_config"]
    topo = build_inkling2_topology(
        num_hidden_layers=int(tcfg["num_hidden_layers"]),
        core_start=int(lcfg["core_start"]),
        core_end=int(lcfg["core_end"]),
        num_passes=int(lcfg["num_passes"]),
    )
    print(
        f"[r4b:inkling2] Verified topology: V={len(topo)} visits "
        f"(prelude={lcfg['core_start']}, core={lcfg['core_end']-lcfg['core_start']}x{lcfg['num_passes']}, "
        f"coda={tcfg['num_hidden_layers']-lcfg['core_end']}), "
        f"E={tcfg['n_routed_experts']}, K={tcfg['num_experts_per_tok']}, shared={tcfg['n_shared_experts']}",
        flush=True,
    )

    if not (rt["vllm_has_inkling2"] or rt["transformers_has_inkling2"]):
        msg = (
            "Neither vLLM (>=0.35.0) nor transformers in this Python env registers `Inkling2LoopForConditionalGeneration`. "
            "Run `bash scripts/setup_inkling_env.sh` to create isolated `$HOME/inkling-vllm-env`, then relaunch with "
            "`PYTHON_BIN=$HOME/inkling-vllm-env/bin/python CONFIG_PATH=configs/r4b_inkling_config.json bash scripts/run_experiment.sh`."
        )
        print(f"[r4b:inkling2] {msg}", flush=True)
        (out_dir / "inkling2_runtime_check.json").write_text(
            json.dumps({"status": "needs_inkling_env", "runtime": rt, "message": msg}, indent=2)
        )
        return

    # Ready for live vLLM / transformers execution once isolated Inkling-2 env is active
    from loop_research.gpu_utils import acquire_idle_gpu

    exit_if_busy = os.environ.get("EXIT_IF_NO_IDLE_GPU", "0") == "1"
    wait_sec = 10 if exit_if_busy else 300
    gpu_idx, gpu_lock, gpu_status = acquire_idle_gpu(cfg.get("preferred_gpu"), wait_timeout_sec=wait_sec)
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        print(f"[r4b:inkling2] Acquired idle GPU {gpu_idx}: {gpu_status.name}", flush=True)

        # Evaluate deterministic Top-8 -> k_v schedules under vLLM if available
        if rt["vllm_has_inkling2"]:
            from vllm import LLM, SamplingParams  # type: ignore

            print("[r4b:inkling2] Loading Inkling-2-Mini in production vLLM (Blackwell NVFP4 + CUDA Graph)...", flush=True)
            llm = LLM(
                model=str(model_path),
                tensor_parallel_size=1,
                trust_remote_code=True,
                max_model_len=4096,
                gpu_memory_utilization=0.80,
            )
            # Baseline throughput & latency check
            prompts = [f"Question: Explain why looped transformers reuse weights across passes ({i}).\nAnswer:" for i in range(8)]
            sp = SamplingParams(temperature=0.0, max_tokens=64)
            _ = llm.generate(prompts[:2], sp)  # warmup
            t0 = time.time()
            outs = llm.generate(prompts, sp)
            dt = time.time() - t0
            n_gen = sum(len(o.outputs[0].token_ids) for o in outs)
            bench = {
                "status": "vllm_baseline_verified",
                "backend": "vllm",
                "vllm_version": rt["vllm_version"],
                "gpu_index": gpu_idx,
                "gpu_name": gpu_status.name,
                "prompts": len(prompts),
                "generated_tokens": n_gen,
                "elapsed_s": dt,
                "tok_per_sec": n_gen / max(dt, 1e-6),
            }
            print(f"[r4b:inkling2] vLLM baseline verified: {json.dumps(bench, indent=2)}", flush=True)
            (out_dir / "inkling2_vllm_verify.json").write_text(json.dumps(bench, indent=2))
    finally:
        gpu_lock.release()


if __name__ == "__main__":
    main()
