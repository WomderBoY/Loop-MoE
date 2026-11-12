"""Collect Foil routes during greedy, cache-free generation on one GPU."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import platform
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile

from .trace import SCHEMA_VERSION, TRACE_SCOPE, read_json, read_jsonl, write_line


def expert_layout(model):
    core = model.model
    pools = {}
    for block, layers in (("head", core.head_layers), ("loop", core.stack.layers), ("tail", core.tail_layers)):
        for index, layer in enumerate(layers):
            weights = (layer.experts_w1, layer.experts_w2, layer.experts_w3)
            pools[f"{block}:{index}"] = {
                "experts": layer.num_experts,
                "bytes_per_expert": sum(weight[0].numel() * weight.element_size() for weight in weights),
            }
    visits = [f"head:{index}" for index in range(core.num_head_layers)]
    visits += [f"loop:{index}" for _ in range(core.num_stacks) for index in range(len(core.stack.layers))]
    visits += [f"tail:{index}" for index in range(core.num_tail_layers)]
    return pools, visits


def capture_visits(torch, trace_sink, visit_pools, start, stop):
    if sorted(trace_sink) != list(range(len(visit_pools))):
        raise RuntimeError("Foil trace does not match the model's unrolled depth")
    selected_ids, candidate_ids, prefix_ids, statistics, visits = [], [], [], [], []
    previous_residuals = {}
    for depth, pool in enumerate(visit_pools):
        point = trace_sink[depth]
        block, _, layer = pool.partition(":")
        if point.block != block or (block == "loop" and point.layer_idx != int(layer)):
            raise RuntimeError(f"Foil trace does not match the physical expert pool at depth {depth}")
        probs = point.router_probs[0, start:stop].float()
        expert_ids = point.topk_idx[0, start:stop]
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
        if block == "loop":
            previous_residuals[pool] = residual
        stats = torch.cat(
            (probs.gather(-1, expert_ids), point.topk_weights[0, start:stop].float(),
             torch.stack((margin, entropy, cosine), dim=-1), sorted_probs),
            dim=-1,
        )
        selected_ids.append(expert_ids)
        candidate_ids.append(candidates)
        prefix_ids.append(point.topk_idx[0, :stop].flatten())
        statistics.append(stats)
        visits.append({
            "pool": pool,
            "depth": depth,
            "loop": point.loop_step if block == "loop" else None,
            "has_previous_loop": has_previous_loop,
        })

    prefix_ids = torch.stack(prefix_ids)
    expert_count = max(point.num_experts for point in trace_sink.values())
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


def collect_request(torch, model, tokenizer, prompt, visit_pools, max_new_tokens):
    input_ids = tokenizer(prompt["text"], return_tensors="pt").input_ids
    prompt_ids = input_ids[0].tolist()
    if not prompt_ids or len(prompt_ids) + max_new_tokens > model.config.context_length:
        raise ValueError(f"Prompt {prompt['id']} is empty or prompt + generation exceeds the model context")
    input_ids = input_ids.to("cuda:0")
    eos = getattr(model.generation_config, "eos_token_id", None)
    if eos is None:
        eos = tokenizer.eos_token_id
    stop_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
    steps, generated_ids = [], []

    with torch.inference_mode():
        for step in range(max_new_tokens):
            sink = {}
            output = model(input_ids=input_ids, trace_sink=sink)
            next_token = output.logits[:, -1, :].argmax(dim=-1)
            del output
            prefix_length = input_ids.size(1)
            first_position = 0 if step == 0 else prefix_length - 1
            steps.append({
                "step": step,
                "phase": "prefill" if step == 0 else "decode",
                "prefix_length": prefix_length,
                "first_position": first_position,
                "visits": capture_visits(torch, sink, visit_pools, first_position, prefix_length),
            })
            sink.clear()
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


def collect_model(args, config, variant, prompts, prompts_path, max_new_tokens):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = args.out_dir / variant / "trace.jsonl.gz"
    if path.exists():
        raise FileExistsError(f"{path} already exists; use a new --out-dir")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable. Check nvidia-smi and the installed PyTorch CUDA build.")
    capability = torch.cuda.get_device_capability(0)
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError(f"Foil requires native CUDA BF16 support; detected compute capability {capability}")
    if capability[0] >= 12 and tuple(map(int, torch.version.cuda.split("."))) < (12, 8):
        raise RuntimeError("Blackwell requires a newer PyTorch CUDA build. Use scripts/setup_gpu.sh cu130.")

    snapshot = snapshot_download(
        repo_id=config["repo"], revision=config["revision"], allow_patterns=f"{variant}/*",
        local_files_only=args.local_files_only,
    )
    checkpoint = Path(snapshot) / variant
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, trust_remote_code=True, local_files_only=True, dtype=torch.bfloat16,
    ).to("cuda:0").eval()
    pools, visit_pools = expert_layout(model)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "trace_scope": TRACE_SCOPE,
        "model": variant,
        "repo": config["repo"],
        "revision": config["revision"],
        "request_count": len(prompts),
        "prompts_sha256": hashlib.sha256(prompts_path.read_bytes()).hexdigest(),
        "max_new_tokens": max_new_tokens,
        "loop_count": model.model.num_stacks,
        "pools": pools,
        "dtype": "bfloat16",
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": capability,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": version("transformers"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=".trace-", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with gzip.open(temporary_path, "wt", encoding="utf-8") as stream:
            write_line(stream, meta)
            for index, prompt in enumerate(prompts, start=1):
                result = collect_request(torch, model, tokenizer, prompt, visit_pools, max_new_tokens)
                write_line(stream, result)
                print(f"[{variant} {index}/{len(prompts)}] {prompt['id']}: {len(result['generated_ids'])} tokens", flush=True)
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/foil_pilot.json"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", action="append", choices=("Base", "Foil-1", "Foil-2", "Foil-3"))
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
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
    for variant in args.model or config["models"]:
        collect_model(args, config, variant, prompts, prompts_path, max_new_tokens)


if __name__ == "__main__":
    main()
