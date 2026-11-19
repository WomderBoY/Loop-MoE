"""Collect expert routes during greedy, cache-free generation on one GPU."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import platform
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile

from .adapters import route_collector
from .trace import SCHEMA_VERSION, TRACE_SCOPE, read_json, read_jsonl, write_line


FOIL_MODELS = ("Base", "Foil-1", "Foil-2", "Foil-3")


def capture_visits(torch, points, order, start, stop):
    selected_ids, candidate_ids, prefix_ids, statistics, visits = [], [], [], [], []
    previous_residuals = {}
    for depth, ((pool, loop), point) in enumerate(zip(order, points, strict=True)):
        probs = point.probs[0, start:stop].float()
        expert_ids = point.ids[0, start:stop]
        top_k = expert_ids.size(-1)
        sorted_probs, candidates = probs.topk(min(2 * top_k, probs.size(-1)), dim=-1)
        margin = sorted_probs[:, top_k - 1] - sorted_probs[:, top_k]
        entropy = -(probs * probs.clamp_min(1e-30).log()).sum(dim=-1)
        residual = point.residual[0, start:stop].float()
        has_previous_loop = pool in previous_residuals
        if has_previous_loop:
            cosine = torch.nn.functional.cosine_similarity(previous_residuals[pool], residual, dim=-1)
        else:
            cosine = torch.zeros_like(margin)
        if loop is not None:
            previous_residuals[pool] = residual
        stats = torch.cat(
            (probs.gather(-1, expert_ids), point.weights[0, start:stop].float(),
             torch.stack((margin, entropy, cosine), dim=-1), sorted_probs),
            dim=-1,
        )
        selected_ids.append(expert_ids)
        candidate_ids.append(candidates)
        prefix_ids.append(point.ids[0, :stop].flatten())
        statistics.append(stats)
        visits.append({
            "pool": pool,
            "depth": depth,
            "loop": loop,
            "has_previous_loop": has_previous_loop,
        })

    prefix_ids = torch.stack(prefix_ids)
    expert_count = max(point.probs.size(-1) for point in points)
    prefix_counts = prefix_ids.new_zeros((len(visits), expert_count))
    prefix_counts.scatter_add_(1, prefix_ids, torch.ones_like(prefix_ids))
    for visit, experts, candidates, stats, counts in zip(
        visits, torch.stack(selected_ids).tolist(), torch.stack(candidate_ids).tolist(),
        torch.stack(statistics).tolist(), prefix_counts.tolist(),
    ):
        top_k = len(experts[0])
        offset = 2 * top_k
        visit["routes"] = [
            [ids, values[:top_k], values[top_k:offset], values[offset], values[offset + 1],
             values[offset + 2] if visit["has_previous_loop"] else None,
             list(zip(candidates_at_position, values[offset + 3:]))]
            for ids, candidates_at_position, values in zip(experts, candidates, stats)
        ]
        visit["prefix_expert_token_counts"] = [(expert, count) for expert, count in enumerate(counts) if count]
        del visit["has_previous_loop"]
    return visits


def collect_request(torch, model, tokenizer, prompt, order, forward, model_config, max_new_tokens, vocab_size):
    if model_config["architecture"] == "loopie":
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt["text"]}], add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt",
        ).input_ids
    else:
        input_ids = tokenizer(prompt["text"], return_tensors="pt").input_ids
    prompt_ids = input_ids[0].tolist()
    context_length = model.config.max_position_embeddings if model_config["architecture"] == "loopie" else (
        model.config.context_length
    )
    if not prompt_ids or len(prompt_ids) + max_new_tokens > context_length:
        raise ValueError(f"Prompt {prompt['id']} is empty or prompt + generation exceeds the model context")
    input_ids = input_ids.to("cuda:0")
    eos = getattr(model.generation_config, "eos_token_id", None)
    if eos is None:
        eos = tokenizer.eos_token_id
    stop_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
    steps, generated_ids = [], []

    with torch.inference_mode():
        for step in range(max_new_tokens):
            output, points = forward(input_ids)
            next_token = output.logits[:, -1, :vocab_size].argmax(dim=-1)
            del output
            prefix_length = input_ids.size(1)
            first_position = 0 if step == 0 else prefix_length - 1
            steps.append({
                "step": step,
                "phase": "prefill" if step == 0 else "decode",
                "prefix_length": prefix_length,
                "first_position": first_position,
                "visits": capture_visits(torch, points, order, first_position, prefix_length),
            })
            del points
            generated_ids.append(next_token.item())
            if generated_ids[-1] in stop_ids or step + 1 == max_new_tokens:
                break
            input_ids = torch.cat((input_ids, next_token[:, None]), dim=1)

    return {
        "request_id": prompt["id"],
        "category": prompt["category"],
        "prompt_ids": prompt_ids,
        "generated_ids": generated_ids,
        "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "steps": steps,
    }


def download_model(config, local_files_only):
    from huggingface_hub import snapshot_download

    subfolder = config.get("subfolder")
    patterns = ["*.json", "*.py", "*.safetensors", "*.jinja", "merges.txt", "*.model"]
    if subfolder:
        patterns = [f"{subfolder}/{pattern}" for pattern in patterns]
    snapshot = Path(snapshot_download(
        repo_id=config["repo"], revision=config["revision"], allow_patterns=patterns,
        local_files_only=local_files_only,
    ))
    return snapshot / subfolder if subfolder else snapshot


def collect_model(args, config, variant, prompts, prompts_path, max_new_tokens):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = args.out_dir / variant / "trace.jsonl.gz"
    if path.exists():
        raise FileExistsError(f"{path} already exists; use a new --out-dir")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable. Check nvidia-smi and the installed PyTorch CUDA build.")
    capability = torch.cuda.get_device_capability(0)
    if config["dtype"] == "bfloat16" and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError(f"BF16 requires native CUDA support; detected compute capability {capability}")
    if capability[0] >= 12 and tuple(map(int, torch.version.cuda.split("."))) < (12, 8):
        raise RuntimeError("Blackwell requires a newer PyTorch CUDA build. Use scripts/setup_gpu.sh cu130.")

    checkpoint = download_model(config, args.local_files_only)
    architecture = config["architecture"]
    load_options = {"trust_remote_code": architecture != "loopie", "local_files_only": True}
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, **load_options)
    model_options = {}
    vocab_size = None
    if architecture == "loopie":
        vocab_size = read_json(checkpoint / "config.json")["vocab_size"]
        model_options["experts_implementation"] = "grouped_mm"
        # Load the embedding and LM head at their stored size. Limit greedy
        # decoding to the vocabulary declared by the upstream model config.
        model_options["vocab_size"] = config.get("checkpoint_vocab_size", vocab_size)
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        checkpoint, dtype=getattr(torch, config["dtype"]), output_loading_info=True, use_safetensors=True,
        **load_options, **model_options,
    )
    if any(loading_info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Checkpoint did not load cleanly: {loading_info}")
    model = model.to("cuda:0").eval()
    meta = {
        "schema_version": SCHEMA_VERSION,
        "trace_scope": TRACE_SCOPE,
        "model": variant,
        "repo": config["repo"],
        "revision": config["revision"],
        "request_count": len(prompts),
        "prompts_sha256": hashlib.sha256(prompts_path.read_bytes()).hexdigest(),
        "max_new_tokens": max_new_tokens,
        "architecture": architecture,
        "subfolder": config.get("subfolder"),
        "prompt_format": "chat_template" if architecture == "loopie" else "raw_text",
        "dtype": config["dtype"],
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": capability,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": version("transformers"),
    }
    if vocab_size is not None:
        meta["vocab_size"] = vocab_size
        meta["checkpoint_vocab_size"] = model.config.vocab_size
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=".trace-", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with route_collector(torch, model, architecture) as (pools, order, forward), \
                gzip.open(temporary_path, "wt", encoding="utf-8") as stream:
            meta["pools"] = pools
            meta["execution_order"] = order
            meta["loop_count"] = 1 + max((loop for _, loop in order if loop is not None), default=-1)
            write_line(stream, meta)
            for index, prompt in enumerate(prompts, start=1):
                result = collect_request(torch, model, tokenizer, prompt, order, forward, config, max_new_tokens,
                                         vocab_size)
                write_line(stream, result)
                print(f"[{variant} {index}/{len(prompts)}] {prompt['id']}: {len(result['generated_ids'])} tokens", flush=True)
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/foil_pilot.json"))
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--model", action="append")
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--download-only", action="store_true", help="Download checkpoints without using a GPU")
    args = parser.parse_args()
    config = read_json(args.config)
    models = config.get("checkpoints")
    if models is None:
        models = {name: {"repo": config["repo"], "revision": config["revision"], "subfolder": name,
                         "architecture": "foil", "dtype": "bfloat16"} for name in FOIL_MODELS}
    selected = args.model or config["models"]
    for name in selected:
        if name not in models:
            parser.error(f"Unknown model {name!r}. Available: {', '.join(models)}")
    if args.download_only:
        for name in selected:
            print(download_model(models[name], args.local_files_only), flush=True)
        return
    if args.out_dir is None:
        parser.error("--out-dir is required for route collection")
    prompts_path = args.prompts or args.config.parent / config["prompts"]
    prompts = list(read_jsonl(prompts_path))
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        prompts = prompts[:args.limit]
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else config["max_new_tokens"]
    if max_new_tokens < 1 or not prompts:
        parser.error("At least one prompt and one generation step are required")
    if any(not all(isinstance(prompt.get(key), str) and prompt[key].strip() for key in ("id", "category", "text"))
           for prompt in prompts):
        parser.error("Each prompt needs nonempty id, category and text strings")
    if len({prompt["id"] for prompt in prompts}) != len(prompts):
        parser.error("Prompt IDs must be unique")
    for variant in selected:
        collect_model(args, models[variant], variant, prompts, prompts_path, max_new_tokens)


if __name__ == "__main__":
    main()
