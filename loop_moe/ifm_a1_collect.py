"""A1: Collect real generation + fixed-text routing traces on IFM Ouro-500B & Huginn-500B."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

from .ifm_prompts import (
    evaluate_mbpp_code,
    format_ifm_prompt,
    grade_gsm8k,
    ngram_repetition_rate,
    split_requests_three_way,
)
from . import orx_run as orx
from .trace import read_json, read_jsonl, write_line


def select_idle_gpu(preferred_index: str | None = None) -> dict[str, Any]:
    """Strictly pick an idle GPU (0 foreign compute processes, <= 32 MiB used)."""
    inv_raw = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        capture_output=True,
        timeout=15,
    ).stdout.splitlines()
    proc_raw = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        capture_output=True,
        timeout=15,
    ).stdout.splitlines()
    busy_uuids = {line.split(",")[0].strip() for line in proc_raw if line.strip()}
    gpus = []
    for line in inv_raw:
        if not line.strip():
            continue
        idx, uuid, name, mem_used, mem_total = [p.strip() for p in line.split(",")]
        gpus.append(
            {
                "index": idx,
                "uuid": uuid,
                "name": name,
                "memory_used_mib": int(mem_used),
                "memory_total_mib": int(mem_total),
                "busy": (uuid in busy_uuids) or (int(mem_used) > 32),
            }
        )
    if preferred_index is not None:
        match = next((g for g in gpus if g["index"] == str(preferred_index)), None)
        if match is None:
            raise RuntimeError(f"Requested GPU {preferred_index} not found")
        if match["busy"]:
            raise RuntimeError(
                f"Refusing to use occupied GPU {match['index']} ({match['memory_used_mib']} MiB used)"
            )
        return match
    idle = [g for g in gpus if not g["busy"]]
    if not idle:
        raise RuntimeError("No idle GPU available on this server; refusing to use occupied GPUs.")
    return idle[0]


def _artifact_files_complete(target_dir: Path) -> bool:
    manifest_path = target_dir / "artifact_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = read_json(manifest_path)
        files = manifest.get("files", [])
        if not files:
            return False
        for rec in files:
            fpath = target_dir / rec["path"]
            if not fpath.is_file() or fpath.stat().st_size != int(rec["size_bytes"]):
                return False
        return True
    except Exception:
        return False


def download_ifm_artifact(repo_id: str, model_root: Path, local_files_only: bool = False) -> Path:
    from huggingface_hub import snapshot_download

    target_dir = model_root / repo_id.replace("/", "__")
    target_dir.mkdir(parents=True, exist_ok=True)
    if _artifact_files_complete(target_dir):
        print(f"[download] Verified complete local artifact: {target_dir}", flush=True)
        return target_dir
    if local_files_only:
        raise FileNotFoundError(f"Incomplete or missing local artifact: {target_dir}")
    print(f"[download] Downloading {repo_id} -> {target_dir} ...", flush=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        local_files_only=False,
    )
    if not _artifact_files_complete(target_dir):
        raise RuntimeError(f"Downloaded artifact failed completeness check: {target_dir}")
    return target_dir


def _collect_freegen_request(
    torch,
    model,
    tokenizer,
    forward_capture,
    item: dict[str, Any],
    *,
    max_prompt_len: int,
    max_new_tokens: int,
    device: str,
) -> dict[str, Any]:
    prompt_text, stop_strings = format_ifm_prompt(item)
    prompt_ids = tokenizer.encode(prompt_text, bos=True, eos=False)
    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:]
    if not prompt_ids:
        raise ValueError(f"Empty prompt tokens for {item['id']}")

    eos_id = tokenizer.eos_id
    tokens = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    generated_ids: list[int] = []
    steps: list[dict[str, Any]] = []
    stop_reason = "max_new_tokens"
    t0 = time.monotonic()

    with torch.inference_mode():
        for step_idx in range(max_new_tokens):
            seq_len = tokens.shape[1]
            # Exact full-prefix recomputation; record ONLY the new position (seq_len - 1)
            # so older positions are never double-counted in A2/A3.
            logits, pos_visits = forward_capture(tokens, start=seq_len - 1, stop=seq_len)
            next_id = int(logits[0, -1].argmax(dim=-1).item())
            steps.append(
                {
                    "step": step_idx,
                    "token_pos": seq_len - 1,
                    "visits": pos_visits[0],
                }
            )
            generated_ids.append(next_id)
            if next_id == eos_id:
                stop_reason = "eos"
                break
            # Check author-style task stop strings on decoded suffix
            if stop_strings and step_idx >= 3:
                cur_text = tokenizer.decode(generated_ids)
                if any(s in cur_text for s in stop_strings):
                    stop_reason = "stop_string"
                    break
            if step_idx + 1 < max_new_tokens:
                next_tok = torch.tensor([[next_id]], device=device, dtype=torch.long)
                tokens = torch.cat([tokens, next_tok], dim=1)

    elapsed_s = time.monotonic() - t0
    raw_text = tokenizer.decode(generated_ids)
    # Trim after stop string for clean grading
    clean_text = raw_text
    for s in stop_strings:
        if s in clean_text:
            clean_text = clean_text.split(s)[0]

    rep4 = ngram_repetition_rate(generated_ids, n=4)
    quality: dict[str, Any] = {
        "eos": stop_reason == "eos",
        "stop_string": stop_reason == "stop_string",
        "truncated": stop_reason == "max_new_tokens",
        "rep4_rate": round(rep4, 4),
        "output_tokens": len(generated_ids),
        "elapsed_s": round(elapsed_s, 3),
        "ms_per_step": round(1000.0 * elapsed_s / max(1, len(steps)), 2),
        "correct": None,
        "detail": "",
    }
    if item["category"] == "math" and "gold_answer" in item:
        ok, pred = grade_gsm8k(clean_text, item["gold_answer"])
        quality["correct"] = ok
        quality["detail"] = f"pred={pred}, gold={item['gold_answer']}"
    elif item["category"] == "code" and "test_list" in item:
        ok, msg = evaluate_mbpp_code(clean_text, item["test_list"])
        quality["correct"] = ok
        quality["detail"] = msg

    return {
        "request_id": item["id"],
        "category": item["category"],
        "mode": "freegen",
        "prompt_len": len(prompt_ids),
        "generated_ids": generated_ids,
        "generated_text": clean_text,
        "raw_generated_text": raw_text,
        "stop_reason": stop_reason,
        "quality": quality,
        "steps": steps,
    }


def _collect_fixed_request(
    torch,
    tokenizer,
    forward_capture,
    item: dict[str, Any],
    *,
    continuation_tokens: int,
    device: str,
    verify_stepwise: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    prefix_ids = tokenizer.encode(item["prefix"], bos=True, eos=False)
    cont_ids = tokenizer.encode(item["continuation"], bos=False, eos=False)[:continuation_tokens]
    if len(cont_ids) < 16:
        raise ValueError(f"Fixed passage {item['id']} continuation too short ({len(cont_ids)} tokens)")
    full_ids = prefix_ids + cont_ids
    tokens = torch.tensor([full_ids], device=device, dtype=torch.long)
    start = len(prefix_ids)
    stop = len(full_ids)

    with torch.inference_mode():
        _, pos_visits = forward_capture(tokens, start=start, stop=stop)

        stepwise_check = None
        if verify_stepwise:
            # Verify that single-pass causal forward produces identical routing to step-by-step recomputation
            matches = 0
            total = 0
            for offset in range(min(4, len(cont_ids))):
                pos = start + offset
                _, single_v = forward_capture(tokens[:, : pos + 1], start=pos, stop=pos + 1)
                for va, vb in zip(pos_visits[offset], single_v[0]):
                    total += 1
                    if va[0] == vb[0]:
                        matches += 1
            stepwise_check = {
                "checked_positions": min(4, len(cont_ids)),
                "checked_visits": total,
                "exact_route_match_rate": matches / total if total else 1.0,
            }

    steps = [
        {"step": i, "token_pos": start + i, "visits": visits}
        for i, visits in enumerate(pos_visits)
    ]
    return (
        {
            "request_id": item["id"],
            "category": item["category"],
            "mode": "fixed",
            "prompt_len": len(prefix_ids),
            "generated_ids": cont_ids,
            "generated_text": tokenizer.decode(cont_ids),
            "stop_reason": "fixed_text",
            "quality": {
                "eos": False,
                "stop_string": False,
                "truncated": False,
                "rep4_rate": round(ngram_repetition_rate(cont_ids, n=4), 4),
                "output_tokens": len(cont_ids),
                "correct": None,
                "detail": "fixed_teacher_forcing",
            },
            "steps": steps,
        },
        stepwise_check,
    )


def _write_samples_markdown(model_name: str, requests: list[dict[str, Any]], out_path: Path) -> None:
    lines = [f"# {model_name} 生成样例与质量复核", ""]
    for req in requests:
        q = req["quality"]
        lines.append(
            f"## {req['request_id']} ({req['category']}) — "
            f"len={q['output_tokens']}, stop={req['stop_reason']}, rep4={q['rep4_rate']:.3f}, "
            f"correct={q['correct']} ({q['detail']})"
        )
        lines.append("")
        lines.append("```text")
        lines.append(req["generated_text"][:800])
        lines.append("```")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def collect_for_model(
    cfg: dict[str, Any],
    model_name: str,
    *,
    xllm_loop_dir: Path,
    model_root: Path,
    out_dir: Path,
    local_files_only: bool,
    progress_cb=None,
) -> dict[str, Any]:
    import torch
    from .ifm_runtime import bootstrap_xllm, ifm_route_collector, load_ifm_model, verify_attention_backend

    spec = cfg["checkpoints"][model_name]
    runtime_info = bootstrap_xllm(xllm_loop_dir, attention_backend=cfg.get("attention_backend", "sdpa"))
    artifact_dir = download_ifm_artifact(spec["repo"], model_root, local_files_only=local_files_only)

    manifest_raw = read_json(artifact_dir / "artifact_manifest.json")
    model, tokenizer, model_conf = load_ifm_model(
        artifact_dir,
        device="cuda:0",
        attention_backend=cfg.get("attention_backend", "sdpa"),
    )

    attn_check = None
    if cfg.get("verify_exact_attention", True):
        attn_check = verify_attention_backend(model, tokenizer, spec, device="cuda:0")
        print(f"[{model_name}] attention verification: {json.dumps(attn_check)}", flush=True)

    prompts_path = Path(cfg["prompts_file"])
    fixed_path = Path(cfg["fixed_prompts_file"])
    prompts = list(read_jsonl(prompts_path))
    fixed_items = list(read_jsonl(fixed_path))

    model_out = out_dir / model_name
    model_out.mkdir(parents=True, exist_ok=True)

    split_cfg = cfg["split"]
    split_freegen = split_requests_three_way(
        prompts,
        fit_ratio=split_cfg["fit_ratio"],
        val_ratio=split_cfg["val_ratio"],
        seed=split_cfg["seed"],
    )
    split_fixed = split_requests_three_way(
        fixed_items,
        fit_ratio=0.5,
        val_ratio=0.0,
        seed=split_cfg["seed"],
    )

    freegen_trace_path = model_out / "trace_freegen.jsonl.gz"
    fixed_trace_path = model_out / "trace_fixed.jsonl.gz"

    with ifm_route_collector(model, spec, top_candidates=int(cfg.get("top_candidates", 6))) as (
        pools,
        order,
        fwd_capture,
    ):
        base_meta = {
            "schema_version": 2,
            "trace_scope": "ifm_a1_new_positions",
            "model": model_name,
            "repo": spec["repo"],
            "arch": spec["arch"],
            "manifest_sha256": manifest_raw.get("manifest_sha256"),
            "checkpoint_id": manifest_raw.get("source", {}).get("checkpoint_id"),
            "pools": pools,
            "execution_order": [list(x) for x in order],
            "top_k": spec["top_k"],
            "top_candidates": int(cfg.get("top_candidates", 6)),
            "dtype": "bfloat16",
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "runtime": runtime_info,
            "attention_verification": attn_check,
        }

        freegen_records: list[dict[str, Any]] = []
        with gzip.open(freegen_trace_path, "wt", encoding="utf-8") as stream:
            write_line(
                stream,
                {
                    **base_meta,
                    "mode": "freegen",
                    "request_count": len(prompts),
                    "prompts_sha256": hashlib.sha256(prompts_path.read_bytes()).hexdigest(),
                    "max_new_tokens": int(cfg["max_new_tokens"]),
                },
            )
            for idx, item in enumerate(prompts, start=1):
                rec = _collect_freegen_request(
                    torch,
                    model,
                    tokenizer,
                    fwd_capture,
                    item,
                    max_prompt_len=int(cfg["max_prompt_len"]),
                    max_new_tokens=int(cfg["max_new_tokens"]),
                    device="cuda:0",
                )
                write_line(stream, rec)
                freegen_records.append(rec)
                q = rec["quality"]
                print(
                    f"[{model_name} freegen {idx}/{len(prompts)}] {item['id']} "
                    f"toks={q['output_tokens']} stop={rec['stop_reason']} "
                    f"rep4={q['rep4_rate']:.3f} ok={q['correct']} ({q['ms_per_step']:.1f} ms/tok)",
                    flush=True,
                )
                if progress_cb is not None:
                    progress_cb(model_name, idx, len(prompts), q)

        fixed_records: list[dict[str, Any]] = []
        stepwise_check = None
        with gzip.open(fixed_trace_path, "wt", encoding="utf-8") as stream:
            write_line(
                stream,
                {
                    **base_meta,
                    "mode": "fixed",
                    "request_count": len(fixed_items),
                    "prompts_sha256": hashlib.sha256(fixed_path.read_bytes()).hexdigest(),
                    "continuation_tokens": int(cfg["fixed_continuation_tokens"]),
                },
            )
            for idx, item in enumerate(fixed_items, start=1):
                rec, sw = _collect_fixed_request(
                    torch,
                    tokenizer,
                    fwd_capture,
                    item,
                    continuation_tokens=int(cfg["fixed_continuation_tokens"]),
                    device="cuda:0",
                    verify_stepwise=(idx == 1),
                )
                if sw is not None:
                    stepwise_check = sw
                    print(f"[{model_name}] fixed single-pass vs stepwise check: {json.dumps(sw)}", flush=True)
                write_line(stream, rec)
                fixed_records.append(rec)

    # Free GPU memory cleanly before next model
    del model
    torch.cuda.empty_cache()

    # Quality summary
    math_reqs = [r for r in freegen_records if r["category"] == "math" and r["quality"]["correct"] is not None]
    code_reqs = [r for r in freegen_records if r["category"] == "code" and r["quality"]["correct"] is not None]
    quality_summary = {
        "model": model_name,
        "freegen_requests": len(freegen_records),
        "fixed_requests": len(fixed_records),
        "mean_output_tokens": sum(r["quality"]["output_tokens"] for r in freegen_records) / len(freegen_records),
        "mean_ms_per_step": sum(r["quality"]["ms_per_step"] for r in freegen_records) / len(freegen_records),
        "eos_rate": sum(r["quality"]["eos"] for r in freegen_records) / len(freegen_records),
        "stop_string_rate": sum(r["quality"]["stop_string"] for r in freegen_records) / len(freegen_records),
        "truncation_rate": sum(r["quality"]["truncated"] for r in freegen_records) / len(freegen_records),
        "mean_rep4_rate": sum(r["quality"]["rep4_rate"] for r in freegen_records) / len(freegen_records),
        "math_acc": sum(bool(r["quality"]["correct"]) for r in math_reqs) / len(math_reqs) if math_reqs else None,
        "math_n": len(math_reqs),
        "code_pass1": sum(bool(r["quality"]["correct"]) for r in code_reqs) / len(code_reqs) if code_reqs else None,
        "code_n": len(code_reqs),
        "attention_verification": attn_check,
        "fixed_stepwise_verification": stepwise_check,
        "expected_visits_per_pos": len(order),
    }

    (model_out / "requests_manifest_freegen.json").write_text(
        json.dumps(split_freegen, indent=2) + "\n", encoding="utf-8"
    )
    (model_out / "requests_manifest_fixed.json").write_text(
        json.dumps(split_fixed, indent=2) + "\n", encoding="utf-8"
    )
    (model_out / "a1_quality_summary.json").write_text(
        json.dumps(quality_summary, indent=2) + "\n", encoding="utf-8"
    )
    _write_samples_markdown(model_name, freegen_records, model_out / "generation_samples.md")

    # Compact per-request table without heavy visits
    req_table = [
        {
            "request_id": r["request_id"],
            "category": r["category"],
            "split": (
                "fit"
                if r["request_id"] in split_freegen["fit_request_ids"]
                else "val"
                if r["request_id"] in split_freegen["val_request_ids"]
                else "eval"
            ),
            "prompt_len": r["prompt_len"],
            "output_tokens": r["quality"]["output_tokens"],
            "stop_reason": r["stop_reason"],
            "rep4_rate": r["quality"]["rep4_rate"],
            "ms_per_step": r["quality"]["ms_per_step"],
            "correct": r["quality"]["correct"],
            "detail": r["quality"]["detail"],
        }
        for r in freegen_records
    ]
    (model_out / "requests_summary.json").write_text(json.dumps(req_table, indent=2) + "\n", encoding="utf-8")
    return quality_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ifm_a1_a3.json"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--xllm-loop-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--gpu", type=str, default=None, help="Explicit GPU index; default auto-picks idle GPU")
    parser.add_argument("--model", action="append", help="Subset of models to run")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()

    cfg = read_json(args.config)
    models = args.model or cfg["models"]

    if args.download_only:
        dl_prog = orx.progress(total=len(models), unit="models")
        for idx, name in enumerate(models, start=1):
            path = download_ifm_artifact(cfg["checkpoints"][name]["repo"], args.model_root, args.local_files_only)
            print(f"Downloaded {name} -> {path}", flush=True)
            dl_prog.step(idx)
        return

    # Strict GPU safety check BEFORE importing torch/CUDA
    gpu = select_idle_gpu(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
    print(
        f"Selected idle GPU {gpu['index']} ({gpu['name']}, {gpu['memory_used_mib']} MiB used, {gpu['uuid']})",
        flush=True,
    )

    prompts = list(read_jsonl(Path(cfg["prompts_file"])))
    total_reqs = len(models) * len(prompts)
    prog = orx.progress(total=total_reqs, unit="requests")
    done_count = 0

    def _cb(model_name: str, _idx: int, _n: int, q: dict[str, Any]) -> None:
        nonlocal done_count
        done_count += 1
        prog.step(done_count, ms_per_step=q["ms_per_step"], rep4_rate=q["rep4_rate"])

    summaries = {}
    for name in models:
        summaries[name] = collect_for_model(
            cfg,
            name,
            xllm_loop_dir=args.xllm_loop_dir,
            model_root=args.model_root,
            out_dir=args.out_dir,
            local_files_only=args.local_files_only,
            progress_cb=_cb,
        )
    (args.out_dir / "a1_all_summaries.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print("A1_COMPLETE", json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
