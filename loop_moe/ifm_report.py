"""Run A2 & A3 offline analyses and synthesize the A1-A3 Chinese Markdown report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .ifm_a2_predict import run_a2
from .ifm_a3_replay import run_a3
from . import orx_run as orx
from .trace import read_json


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _pct(val: float | str | None) -> str:
    if val is None or val == "":
        return "—"
    return f"{100.0 * float(val):.2f}%"


def _pp(val: float | str | None) -> str:
    if val is None or val == "":
        return "—"
    return f"{100.0 * float(val):+.2f} pp"


def generate_report(cfg: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    models = cfg["models"]
    lines = [
        f"# IFM Part I MoE（Ouro-500B / Huginn-500B）A1–A3 实验报告（{cfg.get('mode', 'run')}）",
        "",
        "本报告自动汇总 A1（生成质量与路由轨迹采集）、A2（相邻层/相邻 loop/相邻 token 预测价值分解）与 A3（静态驻留到 Belady 远见同预算回放）。",
        "",
    ]

    headline_metrics: dict[str, float] = {}

    for model_name in models:
        m_dir = out_dir / model_name
        a1 = read_json(m_dir / "a1_quality_summary.json")
        a2_fg = read_json(m_dir / "a2_freegen" / "a2_headline.json")
        a2_fx = read_json(m_dir / "a2_fixed" / "a2_headline.json")
        a3_fg = read_json(m_dir / "a3_freegen" / "a3_headline.json")
        a3_fx = read_json(m_dir / "a3_fixed" / "a3_headline.json")

        short = "ouro" if "ouro" in model_name.lower() else "huginn"
        if a2_fg.get("mean_top1_switch_rate") is not None:
            headline_metrics[f"{short}_top1_switch"] = float(a2_fg["mean_top1_switch_rate"])
        if a2_fg.get("all_delta_combined_vs_general_r2") is not None:
            headline_metrics[f"{short}_a2_delta_r2"] = float(a2_fg["all_delta_combined_vs_general_r2"])
        if "302" in a3_fg.get("focus", {}):
            f302 = a3_fg["focus"]["302"]
            headline_metrics[f"{short}_hit_6p3g_loop"] = float(f302["loop_a2_distance_hit_rate"])
            headline_metrics[f"{short}_hit_6p3g_pin"] = float(f302["pin_recurrent_lru_hit_rate"])
            headline_metrics[f"{short}_hit_6p3g_belady"] = float(f302["belady_hit_rate"])
            headline_metrics[f"{short}_saved_h2d_6p3g_pct"] = float(
                f302["loop_saved_h2d_pct_vs_strongest_non_loop"]
            )
        if "191" in a3_fg.get("focus", {}):
            f191 = a3_fg["focus"]["191"]
            headline_metrics[f"{short}_saved_h2d_4g_pct"] = float(
                f191["loop_saved_h2d_pct_vs_strongest_non_loop"]
            )

        attn = a1.get("attention_verification") or {}
        sw = a1.get("fixed_stepwise_verification") or {}
        lines += [
            f"## {model_name}",
            "",
            "### A1 生成质量与数值校验",
            "",
            f"- 逻辑访问/位置：`{a1['expected_visits_per_pos']}`；平均输出长度：`{a1['mean_output_tokens']:.1f}` token；"
            f"速度：`{a1['mean_ms_per_step']:.1f}` ms/step（全前缀重算）。",
            f"- 停止分布：自然 EOS `{_pct(a1['eos_rate'])}`，任务停止串 `{_pct(a1['stop_string_rate'])}`，截断 `{_pct(a1['truncation_rate'])}`；"
            f"4-gram 重复率 `{_pct(a1['mean_rep4_rate'])}`。",
            f"- 客观题校验：GSM8K `{_pct(a1['math_acc'])}` (n={a1['math_n']})，MBPP pass@1 `{_pct(a1['code_pass1'])}` (n={a1['code_n']})。",
            f"- 注意力后端校对（SDPA vs exact FP32-softmax）：max |Δlogit| = `{attn.get('max_abs_logit_diff', '—')}`, "
            f"top-1 token 一致 = `{attn.get('last_token_top1_match', '—')}`, "
            f"路由完全一致率 = `{_pct(attn.get('topk_route_exact_match_rate'))}`；"
            f"固定文本单次前向 vs 逐步重算路由一致率 = `{_pct(sw.get('exact_route_match_rate'))}`。",
            "",
            "### A2 路由预测分解（自由生成 vs 固定文本对照）",
            "",
            f"- 相邻轮次（自由生成）：top-1 换人率 `{_pct(a2_fg['mean_top1_switch_rate'])}`（固定文本 `{_pct(a2_fx['mean_top1_switch_rate'])}`），"
            f"相邻轮次 overlap `{_pct(a2_fg['mean_adjacent_loop_overlap'])}` vs 边缘独立基线 `{_pct(a2_fg['mean_marginal_null_overlap'])}`。",
            f"- 充分条件检验（‖z'−z‖∞ < m/2）：满足样本 `{a2_fg['suff_cond_total']}`，违背数 `{a2_fg['suff_cond_violations']}`（理论应为 0）。",
            "",
            "| 预测器 | 自由生成全层 R@2 | 自由生成 R@4 | 相对通用历史 ΔR@2 (95% CI) | 循环区 R@2 | 固定文本全层 R@2 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        s_fg = {(r["region"], r["predictor"]): r for r in _read_csv(m_dir / "a2_freegen" / "a2_summary.csv")}
        s_fx = {(r["region"], r["predictor"]): r for r in _read_csv(m_dir / "a2_fixed" / "a2_summary.csv")}
        for pred in (
            "pool_stage_pop",
            "prev_token_same_stage",
            "prev_loop_copy",
            "prev_loop_transition",
            "prev_phys_layer",
            "combined_no_prev_layer",
            "combined_loop",
            "general_history",
            "shuffled_cross_loop",
        ):
            r_all = s_fg["all", pred]
            r_rec = s_fg.get(("recurrent", pred))
            r_fx = s_fx["all", pred]
            ci = f"[{100*float(r_all['ci95_lo_r2']):+.2f}, {100*float(r_all['ci95_hi_r2']):+.2f}]"
            lines.append(
                f"| `{pred}` | {_pct(r_all['recall_at_2'])} | {_pct(r_all['recall_at_4'])} | "
                f"{_pp(r_all['delta_vs_general_r2'])} {ci} | "
                f"{_pct(r_rec['recall_at_2']) if r_rec else '—'} | {_pct(r_fx['recall_at_2'])} |"
            )

        # Retain vs Replace
        rr_path = m_dir / "a2_freegen" / "a2_retain_vs_replace.csv"
        if rr_path.exists():
            rr = {(r["predictor"], r["expert_kind"]): r for r in _read_csv(rr_path)}
            lines += [
                "",
                "**保留（Retained）vs 替换（A→B Replaced）拆分（自由生成，r > 0）：**",
                "",
                "| 预测器 | Retained R@2 | Retained R@4 | Replaced (A→B) R@2 | Replaced (A→B) R@4 |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
            for pred in ("pool_stage_pop", "prev_loop_copy", "prev_loop_transition", "prev_phys_layer", "combined_loop", "general_history", "shuffled_cross_loop"):
                ret = rr[pred, "retained"]
                rep = rr[pred, "replaced_new"]
                lines.append(
                    f"| `{pred}` | {_pct(ret['recall_at_2'])} | {_pct(ret['recall_at_4'])} | "
                    f"{_pct(rep['recall_at_2'])} | {_pct(rep['recall_at_4'])} |"
                )

        # A3 Table
        lines += [
            "",
            "### A3 同预算驻留回放（Peak 预算口径，自由生成）",
            "",
            "| 预算 (GiB / 专家) | LRU | Static Hot | OLED Adap | Pin-Rec LRU | Gen+Dist | Loop+Dist | Loop-Hier+Dist | Loop-Rem+Dist | Pin-Rec+Loop | Belady |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        a3_rows = {(int(r["budget_experts"]), r["policy"]): r for r in _read_csv(m_dir / "a3_freegen" / "a3_summary.csv")}
        for cap in cfg["a3"]["budgets_experts"]:
            if (cap, "belady") not in a3_rows:
                continue
            gib = float(a3_rows[cap, "belady"]["budget_gib"])
            cols = [
                _pct(a3_rows[cap, p]["hit_rate_by_bytes"]) if (cap, p) in a3_rows else "—"
                for p in (
                    "lru",
                    "static_hot",
                    "oled_adaptive",
                    "pin_recurrent_lru",
                    "general_history_distance",
                    "loop_a2_distance",
                    "loop_hier_distance",
                    "loop_rem_distance",
                    "pin_recurrent_loop",
                    "belady",
                )
            ]
            lines.append(f"| {gib:.2f} GiB ({cap}) | " + " | ".join(cols) + " |")

        lines += [
            "",
            "**A3 固定文本对照关键预算命中率：**",
            "",
            "| 预算 (GiB / 专家) | Static Hot | Pin-Rec LRU | Gen+Dist | Loop-Hier+Dist | Loop-Rem+Dist | Pin-Rec+Loop | Belady |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        a3_fx_rows = {(int(r["budget_experts"]), r["policy"]): r for r in _read_csv(m_dir / "a3_fixed" / "a3_summary.csv")}
        for cap in (191, 287, 302, 383):
            if (cap, "belady") not in a3_fx_rows:
                continue
            gib = float(a3_fx_rows[cap, "belady"]["budget_gib"])
            cols = [
                _pct(a3_fx_rows[cap, p]["hit_rate_by_bytes"]) if (cap, p) in a3_fx_rows else "—"
                for p in (
                    "static_hot",
                    "pin_recurrent_lru",
                    "general_history_distance",
                    "loop_hier_distance",
                    "loop_rem_distance",
                    "pin_recurrent_loop",
                    "belady",
                )
            ]
            lines.append(f"| {gib:.2f} GiB ({cap}) | " + " | ".join(cols) + " |")
        lines.append("")

    report_path = out_dir / "report_a1_a3.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "headline_metrics.json").write_text(json.dumps(headline_metrics, indent=2) + "\n", encoding="utf-8")
    return headline_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ifm_a1_a3.json"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    cfg = read_json(args.config)
    models = cfg["models"]

    # Stage A2
    p_a2 = orx.progress(total=len(models) * 2, unit="traces", phase="a2_predict")
    done = 0
    for model_name in models:
        m_dir = args.out_dir / model_name
        for mode in ("freegen", "fixed"):
            trace = m_dir / f"trace_{mode}.jsonl.gz"
            manifest = read_json(m_dir / f"requests_manifest_{mode}.json")
            h = run_a2(trace, manifest, cfg["a2"], m_dir / f"a2_{mode}")
            done += 1
            p_a2.step(done)
            print(f"[A2 {model_name}/{mode}] {json.dumps(h)}", flush=True)

    # Stage A3
    p_a3 = orx.progress(total=len(models) * 2, unit="traces", phase="a3_replay")
    done = 0
    for model_name in models:
        m_dir = args.out_dir / model_name
        for mode in ("freegen", "fixed"):
            trace = m_dir / f"trace_{mode}.jsonl.gz"
            manifest = read_json(m_dir / f"requests_manifest_{mode}.json")
            h = run_a3(trace, manifest, cfg["a2"], cfg["a3"], m_dir / f"a3_{mode}")
            done += 1
            p_a3.step(done)
            print(f"[A3 {model_name}/{mode}] {json.dumps(h)}", flush=True)

    # Report
    p_rep = orx.progress(total=1, unit="report", phase="report")
    metrics = generate_report(cfg, args.out_dir)
    p_rep.step(1, **metrics)
    print("A1_A3_REPORT_READY", json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
