"""Run a sharded routing workload on at most two explicitly selected idle GPUs."""

from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

from .trace import read_json, read_jsonl, read_trace, write_line


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def gpu_inventory():
    result = subprocess.run([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits",
    ], check=True, text=True, capture_output=True, timeout=15)
    return [{"index": index.strip(), "uuid": uuid.strip(), "memory_mib": int(memory)}
            for index, uuid, memory in (line.split(",") for line in result.stdout.splitlines())]


def check_gpu(gpu, own_pid=None, *, own_pids=()):
    allowed = set(own_pids)
    if own_pid is not None:
        allowed.add(own_pid)
    result = subprocess.run([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits",
    ], check=True, text=True, capture_output=True, timeout=15)
    foreign = [line.strip() for line in result.stdout.splitlines()
               if line.split(",")[0].strip() == gpu["uuid"]
               and int(line.split(",")[1]) not in allowed]
    if foreign:
        raise RuntimeError(f"GPU {gpu['index']} is occupied by another process: {foreign}; stopping our jobs")
    if not allowed:
        current = next(row for row in gpu_inventory() if row["uuid"] == gpu["uuid"])
        if current["memory_mib"] > 32:
            raise RuntimeError(f"GPU {gpu['index']} has {current['memory_mib']} MiB in use; refusing to start")


def worker_plan(models, gpus, parallel_models):
    """Give each concurrent model its own lane; never run two copies of one model."""
    if parallel_models:
        return [(gpus[index % len(gpus)], model) for index, model in enumerate(models)]
    return [(gpu, None) for gpu in gpus]


def prepare(config_path, out_dir):
    config = read_json(config_path)
    prompts_path = (config_path.parent / config["prompts"]).resolve()
    prompts = list(read_jsonl(prompts_path))
    if len({p["id"] for p in prompts}) != len(prompts) or not prompts:
        raise ValueError("Prompts must be nonempty with unique IDs")
    shard_size = config["shard_size"]
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    files = [*ROOT.glob("loop_moe/*.py"), ROOT / "configs/residency.json"]
    manifest = {
        "config": config, "prompts_sha256": sha256(prompts_path), "request_count": len(prompts),
        "code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sorted(files)},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        if read_json(manifest_path) != manifest:
            raise ValueError("Run inputs/code changed; use a new output directory")
    else:
        save_json(manifest_path, manifest)
    (out_dir / "logs").mkdir(exist_ok=True)
    prompt_dir = out_dir / "prompts"
    prompt_dir.mkdir(exist_ok=True)
    shards = []
    for start in range(0, len(prompts), shard_size):
        path = prompt_dir / f"{start // shard_size:03d}.jsonl"
        content = "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in prompts[start:start + shard_size])
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Shard input changed: {path}")
        path.write_text(content, encoding="utf-8")
        shards.append(path)
    return config, prompts_path, prompts, shards


def validate_shard(path, config, model, prompts_path):
    meta, requests = read_trace(path)
    expected = config["checkpoints"][model]
    fields = ("repo", "revision", "architecture", "dtype")
    if any(meta[key] != expected[key] for key in fields) or meta["model"] != model:
        raise ValueError(f"Wrong model/checkpoint in {path}")
    if meta["max_new_tokens"] != config["max_new_tokens"] or meta["prompts_sha256"] != sha256(prompts_path):
        raise ValueError(f"Wrong workload in {path}")
    if [r["request_id"] for r in requests] != [p["id"] for p in read_jsonl(prompts_path)]:
        raise ValueError(f"Request IDs/order differ in {path}")
    return meta, requests


def preflight(config, prompts):
    # Tokenization only: no model weights or CUDA context are loaded here.
    from transformers import AutoTokenizer
    from .collect import download_model

    rows = []
    for model in config["models"]:
        model_config = config["checkpoints"][model]
        checkpoint = download_model(model_config, True)
        raw_config = read_json(checkpoint / "config.json")
        is_loopie = model_config["architecture"] == "loopie"
        context = raw_config["max_position_embeddings" if is_loopie else "context_length"]
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True,
                                                 trust_remote_code=not is_loopie)
        lengths = []
        for prompt in prompts:
            length = (tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt["text"]}], add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt",
            ).input_ids.shape[-1] if is_loopie else len(tokenizer(prompt["text"]).input_ids))
            if not length or length + config["max_new_tokens"] > context:
                raise ValueError(f"{model}: {prompt['id']} exceeds context {context}")
            lengths.append(length)
        rows.append({"model": model, "requests": len(prompts), "min_prompt_tokens": min(lengths),
                     "max_prompt_tokens": max(lengths), "context_length": context,
                     "max_new_tokens": config["max_new_tokens"]})
    return rows


def merge_model(config, model, shards, prompts_path, out_dir):
    target = out_dir / model / "trace.jsonl.gz"
    expected_ids = [p["id"] for p in read_jsonl(prompts_path)]
    if target.exists():
        validate_shard(target, config, model, prompts_path)
        return target
    target.parent.mkdir(exist_ok=True)
    with NamedTemporaryFile(dir=target.parent, prefix=".merge-", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        base_meta = None
        seen = []
        with gzip.open(temporary, "wt", encoding="utf-8") as output:
            for shard in shards:
                path = out_dir / "shards" / shard.stem / model / "trace.jsonl.gz"
                meta, requests = validate_shard(path, config, model, shard)
                signature = {k: v for k, v in meta.items() if k not in ("request_count", "prompts_sha256")}
                if base_meta is None:
                    base_meta = signature
                    write_line(output, dict(meta, request_count=len(expected_ids),
                                            prompts_sha256=sha256(prompts_path)))
                elif signature != base_meta:
                    raise ValueError(f"Shard metadata differs for {model}: {shard.stem}")
                for request in requests:
                    seen.append(request["request_id"])
                    write_line(output, request)
        if seen != expected_ids:
            raise ValueError("Merged request IDs/order differ from the original prompts")
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def finish(config_path, config, shards, prompts_path, out_dir):
    for model in config["models"]:
        trace = merge_model(config, model, shards, prompts_path, out_dir)
        model_dir = trace.parent
        with (out_dir / "logs" / f"postprocess-{model}.log").open("a", encoding="utf-8") as log:
            jobs = [
                ("analysis", [sys.executable, "-u", "-m", "loop_moe.analyze", str(trace),
                              "--out-dir", str(model_dir / "analysis")]),
                ("replay", [sys.executable, "-u", "-m", "loop_moe.replay", str(trace),
                            "--config", str(config_path), "--out-dir", str(model_dir / "replay")]),
                ("residency", [sys.executable, "-u", "-m", "loop_moe.residency", str(trace),
                               "--config", str(ROOT / "configs/residency.json"),
                               "--out-dir", str(model_dir / "residency")]),
            ]
            for name, command in jobs:
                marker = model_dir / f".{name}-complete.json"
                if not marker.exists():
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                    save_json(marker, {"completed_at": timestamp()})


def run(args):
    args.config = args.config.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / ".run.lock").open("w") as run_lock:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config, prompts_path, prompts, shards = prepare(args.config, args.out_dir)
        checks = preflight(config, prompts)
        save_json(args.out_dir / "preflight.json", checks)
        print(json.dumps(checks, indent=2), flush=True)
        if args.preflight_only:
            return
        if not args.gpus or len(args.gpus) > 2 or len(set(args.gpus)) != len(args.gpus):
            raise ValueError("Specify one or two distinct GPUs via --gpus")
        if args.exclusive_gpu and len(args.gpus) != 1:
            raise ValueError("--exclusive-gpu is only for one dedicated, allocated GPU")
        inventory = gpu_inventory()
        selected = [next(row for row in inventory if gpu in (row["index"], row["uuid"])) for gpu in args.gpus]
        if len({row["uuid"] for row in selected}) != len(selected):
            raise ValueError("GPU aliases refer to the same GPU")
        locks = []
        try:
            for gpu in selected:
                lock = open(f"/tmp/loop-moe-{gpu['uuid']}.lock", "w")
                locks.append(lock)
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                check_gpu(gpu)
            if args.parallel_models:
                task_queues = {model: queue.Queue() for model in config["models"]}
            else:
                task_queues = {None: queue.Queue()}
            for shard in shards:
                for model in config["models"]:
                    path = args.out_dir / "shards" / shard.stem / model / "trace.jsonl.gz"
                    if path.exists():
                        validate_shard(path, config, model, shard)
                    else:
                        task_queues[model if args.parallel_models else None].put((model, shard))
            def pending_count():
                return sum(tasks.qsize() for tasks in task_queues.values())

            gpu_locks = {gpu["uuid"]: threading.Lock() for gpu in selected}
            owned_pids = {gpu["uuid"]: set() for gpu in selected}
            plan = worker_plan(config["models"], selected, args.parallel_models)
            stop = threading.Event()
            state_lock = threading.Lock()
            state = {"started_at": timestamp(), "phase": "collecting", "gpus": selected,
                     "parallel_models": args.parallel_models, "worker_count": len(plan),
                     "exclusive_gpu": args.exclusive_gpu,
                     "pending_shards": pending_count(), "active": {}, "completed_this_run": []}
            save_json(args.out_dir / "status.json", state)

            def worker(gpu, lane):
                tasks = task_queues[lane]
                worker_key = f"{gpu['index']}:{lane}" if lane else gpu["index"]
                gpu_lock = gpu_locks[gpu["uuid"]]
                gpu_pids = owned_pids[gpu["uuid"]]
                try:
                    while not stop.is_set():
                        try:
                            model, shard = tasks.get_nowait()
                        except queue.Empty:
                            return
                        command = [sys.executable, "-u", "-m", "loop_moe.collect", "--config", str(args.config),
                                   "--model", model, "--prompts", str(shard), "--local-files-only",
                                   "--out-dir", str(args.out_dir / "shards" / shard.stem)]
                        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu["uuid"], HF_HUB_OFFLINE="1",
                                   TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
                        log_path = args.out_dir / "logs" / f"{model}-{shard.stem}.log"
                        with log_path.open("a", encoding="utf-8") as log:
                            log.write(f"\nStarted {timestamp()} on GPU {gpu['index']} ({gpu['uuid']})\n")
                            log.flush()
                            # Serialize the check/spawn/register sequence on each GPU.
                            # Shared servers allow only this driver's registered PIDs.
                            # Dedicated containers check idle before starting and
                            # rely on their allocated device while workers run.
                            with gpu_lock:
                                if not args.exclusive_gpu or not gpu_pids:
                                    check_gpu(gpu, own_pids=gpu_pids)
                                if stop.is_set():
                                    return
                                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                                           stderr=subprocess.STDOUT)
                                gpu_pids.add(process.pid)
                            with state_lock:
                                state["active"][worker_key] = {"model": model, "shard": shard.stem,
                                                               "gpu_index": gpu["index"],
                                                               "pid": process.pid, "started_at": timestamp()}
                                state["pending_shards"] = pending_count()
                                save_json(args.out_dir / "status.json", state)
                            try:
                                while process.poll() is None:
                                    if stop.wait(5):
                                        raise RuntimeError("Another worker stopped; halting our collection")
                                    if not args.exclusive_gpu:
                                        with gpu_lock:
                                            check_gpu(gpu, own_pids=gpu_pids)
                                if process.returncode:
                                    raise RuntimeError(f"{model} shard {shard.stem} failed: see {log_path}")
                            except BaseException:
                                if process.poll() is None:
                                    process.terminate()
                                    try:
                                        process.wait(timeout=30)
                                    except subprocess.TimeoutExpired:
                                        process.kill()
                                        process.wait()
                                raise
                            finally:
                                with gpu_lock:
                                    gpu_pids.discard(process.pid)
                        validate_shard(args.out_dir / "shards" / shard.stem / model / "trace.jsonl.gz",
                                       config, model, shard)
                        with state_lock:
                            state["active"].pop(worker_key)
                            state["completed_this_run"].append({"model": model, "shard": shard.stem})
                            save_json(args.out_dir / "status.json", state)
                        print(f"Completed {model}/{shard.stem} on GPU {gpu['index']}", flush=True)
                except BaseException:
                    stop.set()
                    raise

            with ThreadPoolExecutor(max_workers=len(plan)) as pool:
                futures = [pool.submit(worker, gpu, lane) for gpu, lane in plan]
                errors = []
                for future in futures:
                    try:
                        future.result()
                    except BaseException as error:
                        errors.append(str(error))
            state["active"] = {}
            if errors:
                state.update(phase="stopped", errors=errors, stopped_at=timestamp())
                save_json(args.out_dir / "status.json", state)
                raise RuntimeError("; ".join(errors))
            state.update(phase="postprocessing", collection_completed_at=timestamp())
            save_json(args.out_dir / "status.json", state)
        finally:
            for lock in locks:
                lock.close()
        # Collection subprocesses have exited; CPU analysis no longer occupies GPUs.
        try:
            finish(args.config, config, shards, prompts_path, args.out_dir)
        except BaseException as error:
            state.update(phase="postprocessing_failed", errors=[str(error)], stopped_at=timestamp())
            save_json(args.out_dir / "status.json", state)
            raise
        state.update(phase="complete", completed_at=timestamp())
        save_json(args.out_dir / "status.json", state)
        print(f"Complete: {args.out_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/expanded.json"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+")
    parser.add_argument("--parallel-models", action="store_true",
                        help="Run one worker per model; --gpus 0 uses one GPU for all four models")
    parser.add_argument("--exclusive-gpu", action="store_true",
                        help="Dedicated rental GPU: check idle before starting; rely on allocation isolation "
                             "while workers run, since container and NVML PIDs can differ")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
