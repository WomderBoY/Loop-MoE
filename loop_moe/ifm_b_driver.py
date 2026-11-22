"""Stage B End-to-End Driver & Report Generator (B1 KV-Cache Verify + B2 Microbench/Offline/Real GPU)."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
from typing import Any

import torch

from . import orx_run as orx
from .ifm_a1_collect import download_ifm_artifact, select_idle_gpu
from .ifm_b1_kv_verify import verify_model_b1
from .ifm_b2_gpu_bench import run_real_gpu_benchmark_for_model
from .ifm_b2_microbench import benchmark_pcie_and_ffn
from .ifm_b2_offline_replay import run_b2_offline_replay
from .ifm_runtime import bootstrap_xllm
from .trace import read_json, read_jsonl, write_csv, write_json


def _fmt_pct(v: float) -> str:
    return f"{100.0 * v:.2f}%"


def _build_b_report(
    out_dir: Path,
    micro: dict[str, Any],
    b1: dict[str, Any],
    offline_summary: dict[str, dict[str, list[dict[str, Any]]]],
    offline_paired: dict[str, dict[str, list[dict[str, Any]]]],
    gpu_summary: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append("# IFM Part I MoE（Ouro-500B / Huginn-500B）Stage B（B1 + B2）实验报告\n")
    lines.append(
        f"- 测试 GPU：`{micro['device_name']}`（独立空闲卡 GPU `{micro['gpu_index']}`，单专家大小 `{micro['expert_mib']:.3f} MiB` BF16）\n"
    )
    lines.append("## 一、B2-硬件微基准（PCIe H2D 与 GPU/CPU 单专家计算）\n")
    lines.append(
        f"- 单专家固定内存 H2D（孤立）：p50 = `{micro['h2d_isolated_p50_ms']:.3f} ms`，p95 = `{micro['h2d_isolated_p95_ms']:.3f} ms`，有效带宽 = `{micro['h2d_isolated_bw_gib_s']:.2f} GiB/s`。"
    )
    lines.append(
        f"- 单专家固定内存 H2D（与 BF16 GEMM 并发）：p50 = `{micro['h2d_concurrent_p50_ms']:.3f} ms`，p95 = `{micro['h2d_concurrent_p95_ms']:.3f} ms`，有效带宽 = `{micro['h2d_concurrent_bw_gib_s']:.2f} GiB/s`。\n"
    )
    lines.append("| Batch Tokens | GPU BF16 FFN p50 (ms) | CPU BF16 FFN p50 (ms) | CPU FP32 FFN p50 (ms) |")
    lines.append("| ---: | ---: | ---: | ---: |")
    for r in micro["ffn_compute_ms"]:
        lines.append(
            f"| {r['batch_tokens']} | {r['gpu_bf16_p50_ms']:.4f} | {r['cpu_bf16_p50_ms']:.3f} | {r['cpu_fp32_p50_ms']:.3f} |"
        )
    lines.append("")

    lines.append("## 二、B1：按逻辑 Visit 的增量 KV-Cache 数值校验\n")
    lines.append(
        r"| 模型 | 校验请求×步数 | Decode Max \|Δlogit\| | Decode Mean \|Δlogit\| | Top-1 一致率 | 路由完全一致率 | 自由 Greedy 一致率 | Full-Prefix TPOT | 增量 KV TPOT | 加速比 | KV (KiB/tok) | 峰值显存 (GiB) |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for m, s in b1.items():
        lines.append(
            f"| {m} | {s['prompts_checked']}×{s['decode_steps_per_prompt']} | {s['decode_max_abs_logit_diff']:.4f} | "
            f"{s['decode_mean_abs_logit_diff']:.5f} | {_fmt_pct(s['decode_top1_match_rate'])} | "
            f"{_fmt_pct(s['decode_route_exact_match_rate'])} | {_fmt_pct(s['free_greedy_match_rate'])} | "
            f"{s['full_prefix_tpot_ms']:.1f} ms | {s['incremental_tpot_ms']:.1f} ms | {s['speedup_vs_full_prefix']:.2f}× | "
            f"{s['kv_kib_per_token']:.1f} | {s['peak_allocated_gib']:.2f} |"
        )
    lines.append("")

    lines.append("## 三、B2-离线 96 请求全量 2×2 协同仿真（同峰值预算口径，真机时延标定）\n")
    for m in offline_summary:
        for reg in offline_summary[m]:
            srows = offline_summary[m][reg]
            prows = offline_paired[m][reg]
            lines.append(f"### {m} — Regime `{reg}`\n")
            lines.append(
                "| 预算 (GiB / 专家) | Static Hot Hit | Gen-NoPF Hit | Loop-NoPF Hit | Gen-PF Stall (ms/tok) | Loop-PF Stall (ms/tok) | Loop-PF vs Gen-PF Stall 节省 (95% CI) | Loop-PF 有效/及时预取率 |"
            )
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            by_bp = {(int(r["budget_experts"]), r["policy"]): r for r in srows}
            p_map = {(int(r["budget_experts"]), r["policy"], r["baseline"]): r for r in prows}
            budgets = sorted({int(r["budget_experts"]) for r in srows})
            for cap in budgets:
                sh = by_bp[cap, "static_hot_nopf"]
                gn = by_bp[cap, "general_hazard_nopf"]
                ln = by_bp[cap, "loop_hazard_nopf"]
                gp = by_bp[cap, "general_hazard_pf"]
                lp = by_bp[cap, "loop_hazard_pf"]
                pr = p_map[cap, "loop_hazard_pf", "general_hazard_pf"]
                lines.append(
                    f"| {sh['budget_gib']:.2f} GiB ({cap}) | {_fmt_pct(sh['demand_hit_rate'])} | "
                    f"{_fmt_pct(gn['demand_hit_rate'])} | {_fmt_pct(ln['demand_hit_rate'])} | "
                    f"{gp['exposed_stall_ms_per_tok']:.2f} | {lp['exposed_stall_ms_per_tok']:.2f} | "
                    f"{pr['mean_stall_saved_pct']:+.2f}% [{pr['stall_ci95_lo_pct']:+.2f}, {pr['stall_ci95_hi_pct']:+.2f}] ({pr['wins_stall']}/{pr['losses_stall']}) | "
                    f"{_fmt_pct(lp['useful_pf_rate'])} / {_fmt_pct(lp['timely_pf_rate'])} |"
                )
            lines.append("")

    lines.append("## 四、B2-真机 GPU Offload + CUDA Copy-Stream 异步预取实测（18 配对请求 × 48 Token）\n")
    lines.append(
        "| 模型 | 预算 (GiB / 专家) | 策略 | Token 一致 | TPOT Mean (ms) | TPOT p50 (ms) | TTFT (ms) | Demand Hit | Total H2D (MiB/tok) | 有效/及时预取 | 暴露等待 (ms/tok) | 预测器 CPU (ms/tok) | 峰值 Alloc/Res (GiB) |"
    )
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in gpu_summary:
        lines.append(
            f"| {r['model']} | {r['budget_gib']:.2f} ({r['budget_experts']}) | `{r['policy']}` | "
            f"{_fmt_pct(r['exact_token_match_rate'])} | {r['tpot_mean_ms']:.2f} | {r['tpot_p50_ms']:.2f} | "
            f"{r['ttft_mean_ms']:.1f} | {_fmt_pct(r['demand_hit_rate'])} | {r['total_h2d_mib_per_tok']:.1f} | "
            f"{_fmt_pct(r['useful_pf_rate'])} / {_fmt_pct(r['timely_pf_rate'])} | {r['exposed_wait_ms_per_tok']:.2f} | "
            f"{r['predictor_cpu_ms_per_tok']:.2f} | {r['peak_allocated_gib']:.2f} / {r['peak_reserved_gib']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage B end-to-end driver (B1 + B2)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--xllm-loop-dir", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--trace-dir", required=True, help="Directory holding formal A1 traces (e.g. adf9d9e0 run dir)")
    parser.add_argument("--gpu", default=None)
    args = parser.parse_args()

    cfg = read_json(Path(args.config))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    b_cfg = cfg["stage_b"]

    bootstrap_xllm(args.xllm_loop_dir, attention_backend=cfg.get("attention_backend", "sdpa"))

    # Strict idle-GPU check: refuse to use any GPU occupied by other users!
    gpu = select_idle_gpu(args.gpu)
    device = torch.device(f"cuda:{gpu['index']}")
    torch.cuda.set_device(device)
    print(f"[Stage B] Selected strictly idle GPU {gpu['index']} ({gpu['name']}, {gpu['memory_used_mib']} MiB used)")

    # 1. Hardware microbenchmark
    orx.phase("b2_microbench", step=0, of=1, unit="bench")
    micro = benchmark_pcie_and_ffn(device=device)
    micro["gpu_index"] = gpu["index"]
    micro["gpu_uuid"] = gpu["uuid"]
    write_json(out_dir / "b2_microbench.json", micro)
    orx.emit({"step": 1, "m": {"h2d_conc_ms": micro["h2d_concurrent_p50_ms"], "pcie_bw_gibs": micro["h2d_concurrent_bw_gib_s"]}})
    print(f"[B2 microbench] H2D iso={micro['h2d_isolated_p50_ms']}ms conc={micro['h2d_concurrent_p50_ms']}ms ({micro['h2d_concurrent_bw_gib_s']} GiB/s)")

    # 2. B1 Incremental KV-Cache Verification
    orx.phase("b1_kv_verify", step=0, of=len(cfg["models"]), unit="models")
    all_prompts = list(read_jsonl(Path(cfg["prompts_file"])))
    b1_prompts = all_prompts[: int(b_cfg.get("b1_prompts", 6))]
    b1_summaries: dict[str, Any] = {}
    b1_rows: list[dict[str, Any]] = []
    for idx, model_name in enumerate(cfg["models"], start=1):
        spec = cfg["checkpoints"][model_name]
        mdir = download_ifm_artifact(spec["repo"], Path(args.model_root), local_files_only=True)
        summary, rows = verify_model_b1(
            model_name,
            spec,
            mdir,
            b1_prompts,
            device=device,
            decode_steps=int(b_cfg.get("b1_decode_steps", 32)),
            attention_backend=cfg.get("attention_backend", "sdpa"),
        )
        b1_summaries[model_name] = summary
        b1_rows.extend(rows)
        orx.emit(
            {
                "step": idx,
                "m": {
                    f"{'ouro' if 'Ouro' in model_name else 'huginn'}_b1_inc_tpot_ms": summary["incremental_tpot_ms"],
                    f"{'ouro' if 'Ouro' in model_name else 'huginn'}_b1_greedy_match": summary["free_greedy_match_rate"],
                },
            }
        )
        print(f"[B1 {model_name}] inc_tpot={summary['incremental_tpot_ms']}ms (vs {summary['full_prefix_tpot_ms']}ms), greedy_match={summary['free_greedy_match_rate']:.4f}, visit_comp={summary['measured_visit_compute_ms']}ms")
    write_json(out_dir / "b1_kv_verification.json", b1_summaries)
    write_csv(out_dir / "b1_kv_steps.csv", b1_rows)

    # 3. B2 Offline 96-request CPU co-simulation (calibrated with exact measured timings + batch=8 overlap regime)
    trace_root = Path(args.trace_dir)
    h2d_ms = float(micro["h2d_concurrent_p50_ms"])
    orx.phase("b2_offline_replay", step=0, of=len(cfg["models"]) * 2, unit="regimes")
    offline_summary: dict[str, dict[str, list[dict[str, Any]]]] = {}
    offline_paired: dict[str, dict[str, list[dict[str, Any]]]] = {}
    tuned_by_model: dict[str, dict[str, Any]] = {}
    step_i = 0
    for model_name in cfg["models"]:
        offline_summary[model_name] = {}
        offline_paired[model_name] = {}
        trace_path = trace_root / model_name / "trace_freegen.jsonl.gz"
        visit_comp_b1 = max(0.15, float(b1_summaries[model_name]["measured_visit_compute_ms"]))
        regimes = [
            ("measured_b1", h2d_ms, visit_comp_b1),
            ("batch8_overlap", h2d_ms, max(visit_comp_b1 * 3.0, h2d_ms * 1.15)),
        ]
        m_out = out_dir / model_name / "b2_offline"
        for reg_name, r_h2d, r_comp in regimes:
            hl = run_b2_offline_replay(
                trace_path,
                m_out,
                cfg,
                h2d_ms=r_h2d,
                visit_comp_ms=r_comp,
                regime_name=reg_name,
                workers=16,
            )
            if reg_name == "measured_b1":
                tuned_by_model[model_name] = hl["tuned"]
            offline_summary[model_name][reg_name] = list(
                csv.DictReader((m_out / f"b2_offline_summary_{reg_name}.csv").open())
            )
            # Parse float fields back
            for row in offline_summary[model_name][reg_name]:
                for k in (
                    "budget_gib",
                    "demand_hit_rate",
                    "demand_h2d_mib_per_tok",
                    "prefetch_h2d_mib_per_tok",
                    "total_h2d_mib_per_tok",
                    "timely_pf_rate",
                    "useful_pf_rate",
                    "wasted_pf_rate",
                    "exposed_stall_ms_per_tok",
                    "exposed_stall_p50_ms_per_tok",
                    "total_ms_per_tok",
                ):
                    row[k] = float(row[k])
            offline_paired[model_name][reg_name] = list(
                csv.DictReader((m_out / f"b2_offline_paired_{reg_name}.csv").open())
            )
            for row in offline_paired[model_name][reg_name]:
                for k in (
                    "budget_gib",
                    "mean_stall_saved_pct",
                    "stall_ci95_lo_pct",
                    "stall_ci95_hi_pct",
                    "mean_demand_h2d_saved_pct",
                    "h2d_ci95_lo_pct",
                    "h2d_ci95_hi_pct",
                ):
                    row[k] = float(row[k])
                for k in ("wins_stall", "ties_stall", "losses_stall"):
                    row[k] = int(row[k])
            step_i += 1
            orx.emit({"step": step_i})
            print(f"[B2 offline {model_name}/{reg_name}] tuned={hl['tuned']}")

    # 4. B2 Real GPU Offload & CUDA Copy-Stream 2x2 Benchmark (re-check no FOREIGN process started on our GPU!)
    def _assert_no_foreign_gpu_procs(gpu_uuid: str, gpu_index: str) -> None:
        my_pid = str(os.getpid())
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
            check=True,
            text=True,
            capture_output=True,
            timeout=15,
        ).stdout.splitlines()
        for line in out:
            if not line.strip():
                continue
            u, pid = [p.strip() for p in line.split(",")]
            if u == gpu_uuid and pid != my_pid:
                raise RuntimeError(f"Foreign process PID={pid} appeared on GPU {gpu_index}; aborting safely.")

    orx.phase("b2_gpu_bench", step=0, of=len(cfg["models"]), unit="models")
    gpu_budgets = [int(b) for b in b_cfg.get("gpu_budgets_experts", [191, 302])]
    gpu_summary_all: list[dict[str, Any]] = []
    gpu_reqs_all: list[dict[str, Any]] = []
    for idx, model_name in enumerate(cfg["models"], start=1):
        _assert_no_foreign_gpu_procs(gpu["uuid"], gpu["index"])
        spec = cfg["checkpoints"][model_name]
        mdir = download_ifm_artifact(spec["repo"], Path(args.model_root), local_files_only=True)
        trace_path = trace_root / model_name / "trace_freegen.jsonl.gz"
        srows, rrows = run_real_gpu_benchmark_for_model(
            model_name,
            spec,
            mdir,
            trace_path,
            all_prompts,
            cfg,
            device=device,
            budgets=gpu_budgets,
            n_eval_prompts=int(b_cfg.get("gpu_eval_prompts", 18)),
            decode_tokens=int(b_cfg.get("gpu_decode_tokens", 48)),
            tuned_params=tuned_by_model.get(model_name),
        )
        gpu_summary_all.extend(srows)
        gpu_reqs_all.extend(rrows)
        orx.emit({"step": idx})
        print(f"[B2 real GPU {model_name}] done ({len(srows)} budget×policy summaries)")

    write_csv(out_dir / "b2_gpu_summary.csv", gpu_summary_all)
    write_csv(out_dir / "b2_gpu_requests.csv", gpu_reqs_all)

    # 5. Report & Headline Metrics
    orx.phase("report", step=0, of=1, unit="report")
    report_md = _build_b_report(out_dir, micro, b1_summaries, offline_summary, offline_paired, gpu_summary_all)
    (out_dir / "report_b1_b2.md").write_text(report_md, encoding="utf-8")

    # Emit compact headline metrics
    headline: dict[str, Any] = {
        "h2d_isolated_p50_ms": micro["h2d_isolated_p50_ms"],
        "h2d_concurrent_p50_ms": micro["h2d_concurrent_p50_ms"],
        "pcie_bw_conc_gib_s": micro["h2d_concurrent_bw_gib_s"],
    }
    for m, s in b1_summaries.items():
        tag = "ouro" if "Ouro" in m else "huginn"
        headline[f"{tag}_b1_inc_tpot_ms"] = s["incremental_tpot_ms"]
        headline[f"{tag}_b1_greedy_match"] = s["free_greedy_match_rate"]
        headline[f"{tag}_b1_route_match"] = s["decode_route_exact_match_rate"]
    for m in cfg["models"]:
        tag = "ouro" if "Ouro" in m else "huginn"
        p_b1 = {(int(r["budget_experts"]), r["policy"], r["baseline"]): r for r in offline_paired[m]["measured_b1"]}
        if (302, "loop_hazard_pf", "general_hazard_pf") in p_b1:
            headline[f"{tag}_off_pf_stall_saved_6p3g_pct"] = round(
                p_b1[302, "loop_hazard_pf", "general_hazard_pf"]["mean_stall_saved_pct"], 3
            )
        if (191, "loop_hazard_pf", "general_hazard_pf") in p_b1:
            headline[f"{tag}_off_pf_stall_saved_4g_pct"] = round(
                p_b1[191, "loop_hazard_pf", "general_hazard_pf"]["mean_stall_saved_pct"], 3
            )
        # Real GPU TPOT at 4.0 GiB (191) and 6.3 GiB (302)
        gmap = {(int(r["budget_experts"]), r["policy"]): r for r in gpu_summary_all if r["model"] == m}
        for cap, btag in ((191, "4g"), (302, "6p3g")):
            if (cap, "loop_pf") in gmap and (cap, "general_pf") in gmap:
                headline[f"{tag}_gpu_tpot_{btag}_loop_pf_ms"] = gmap[cap, "loop_pf"]["tpot_mean_ms"]
                headline[f"{tag}_gpu_tpot_{btag}_gen_pf_ms"] = gmap[cap, "general_pf"]["tpot_mean_ms"]
                headline[f"{tag}_gpu_tpot_{btag}_gen_nopf_ms"] = gmap[cap, "general_nopf"]["tpot_mean_ms"]

    write_json(out_dir / "headline_metrics_b.json", headline)
    orx.emit({"step": 1, "m": headline})


if __name__ == "__main__":
    main()
