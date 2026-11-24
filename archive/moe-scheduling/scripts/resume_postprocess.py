"""Resume completed route collection on CPU, without checking or initializing CUDA."""

import argparse
import fcntl
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loop_moe.expanded import ROOT, finish, save_json, sha256, timestamp
from loop_moe.trace import SCHEMA_VERSION, TRACE_SCOPE, read_json, read_jsonl


def validate_inputs(config_path, out_dir):
    config_path, out_dir = Path(config_path).resolve(), Path(out_dir).resolve()
    config = read_json(config_path)
    prompts_path = (config_path.parent / config["prompts"]).resolve()
    prompts = list(read_jsonl(prompts_path))
    files = [*ROOT.glob("loop_moe/*.py"), ROOT / "configs/residency.json"]
    expected = {
        "config": config, "prompts_sha256": sha256(prompts_path), "request_count": len(prompts),
        "code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sorted(files)},
    }
    if read_json(out_dir / "manifest.json") != expected:
        raise ValueError("Configuration, prompts or experiment code differ from the collection manifest")
    shards = []
    for start in range(0, len(prompts), config["shard_size"]):
        shard = out_dir / "prompts" / f"{start // config['shard_size']:03d}.jsonl"
        selected = prompts[start:start + config["shard_size"]]
        if list(read_jsonl(shard)) != selected:
            raise ValueError(f"Prompt shard differs: {shard}")
        for model in config["models"]:
            trace = out_dir / "shards" / shard.stem / model / "trace.jsonl.gz"
            if not trace.is_file():
                raise FileNotFoundError(f"Collection is incomplete: {trace}")
            with gzip.open(trace, "rt", encoding="utf-8") as stream:
                meta = json.loads(next(stream))
            checkpoint = config["checkpoints"][model]
            if any(meta.get(key) != checkpoint[key] for key in ("repo", "revision", "architecture", "dtype")):
                raise ValueError(f"Checkpoint differs: {trace}")
            if (meta.get("schema_version"), meta.get("trace_scope"), meta.get("model"),
                    meta.get("request_count"), meta.get("prompts_sha256"), meta.get("max_new_tokens")) != (
                    SCHEMA_VERSION, TRACE_SCOPE, model, len(selected), sha256(shard), config["max_new_tokens"]):
                raise ValueError(f"Trace header differs: {trace}")
        shards.append(shard)
    # Full request/body validation is performed by merge_model before publishing
    # merged traces. The read-only handoff check verifies finalized shard headers.
    return config, prompts_path, shards


def run(args):
    config_path, out_dir = args.config.resolve(), args.out_dir.resolve()
    config, prompts_path, shards = validate_inputs(config_path, out_dir)
    print(f"CPU handoff check passed: {len(config['models'])} models, "
          f"{len(config['models']) * len(shards)} finalized shards", flush=True)
    if args.check_only:
        return
    with (out_dir / ".run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = out_dir / "status.json"
        state = read_json(state_path) if state_path.exists() else {}
        state.update(phase="postprocessing", postprocess_only=True, active={},
                     postprocessing_resumed_at=timestamp())
        state.pop("errors", None)
        state.pop("stopped_at", None)
        save_json(state_path, state)
        try:
            finish(config_path, config, shards, prompts_path, out_dir)
        except BaseException as error:
            state.update(phase="postprocessing_interrupted" if isinstance(error, KeyboardInterrupt)
                         else "postprocessing_failed", errors=[str(error)], stopped_at=timestamp())
            save_json(state_path, state)
            raise
        state.update(phase="complete", completed_at=timestamp())
        save_json(state_path, state)
        print(f"All CPU postprocessing complete: {out_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/expanded.json"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
