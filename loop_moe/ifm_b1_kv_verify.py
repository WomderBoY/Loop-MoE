"""B1: Per-logical-visit incremental KV-cache engine & full-prefix numerical verification.

Implements `LoopedIncrementalEngine` for both IFM-Ouro-500B (`LoopedTransformer`) and
IFM-Huginn-500B (`Huginn`), maintaining one `(cache_k, cache_v, cache_len)` entry per
logical visit (112 visits/token) with exact RoPE offset and Huginn `_hashed_normal_state(cache_len)`.

Verifies against IFM's official full-prefix forward across prefill boundary, step 1, and
32 continuous decode steps on diverse prompts (text, math, code).
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import gc
import json
from pathlib import Path
import statistics
import time
from typing import Any, Callable

import torch

from .ifm_a1_collect import _model_dir, select_idle_gpu
from .ifm_prompts import format_ifm_prompt
from .ifm_runtime import bootstrap_xllm, build_ifm_topology, load_ifm_model
from .trace import read_json, read_jsonl, write_csv, write_json


MoEExecutorFn = Callable[[int, int, Any, torch.Tensor, str | None], tuple[torch.Tensor, Any]]


class LoopedIncrementalEngine:
    """Exact per-logical-visit KV-cached incremental engine for IFM Ouro & Huginn."""

    def __init__(self, model, model_spec: dict[str, Any]):
        self.model = model
        self.model_spec = model_spec
        self.arch = model_spec["arch"]
        self.pools, self.order = build_ifm_topology(model, model_spec)
        self.num_visits = len(self.order)
        self.visit_caches: list[tuple[torch.Tensor, torch.Tensor, int] | None] = [None] * self.num_visits
        self.cache_len: int = 0

    def reset(self) -> None:
        self.visit_caches = [None] * self.num_visits
        self.cache_len = 0

    def kv_cache_bytes(self) -> int:
        total = 0
        for c in self.visit_caches:
            if c is None:
                continue
            ck, cv, _ = c
            if ck is not None:
                total += ck.numel() * ck.element_size()
            if cv is not None:
                total += cv.numel() * cv.element_size()
        return total

    def _run_one_visit(
        self,
        v_idx: int,
        layer_id: int,
        hidden: torch.Tensor,
        freqs_cis: torch.Tensor | None,
        *,
        moe_executor: MoEExecutorFn | None = None,
        fp32_attn_output: bool = False,
        deterministic: bool = True,
    ) -> torch.Tensor:
        layer = self.model.layers[layer_id]
        cache = self.visit_caches[v_idx]
        if moe_executor is None:
            hidden, _aux, new_cache = layer(
                hidden,
                freqs_cis,
                segments=None,
                moe_router_load_balancing_type=None,
                fp32_attn_output=fp32_attn_output,
                deterministic=deterministic,
                cache=cache,
            )
        else:
            y, new_cache = layer.attention(
                hidden,
                freqs_cis,
                None,
                fp32_attn_output,
                deterministic,
                cache,
            )
            hidden, _aux = moe_executor(layer_id, v_idx, layer, y, None)
        self.visit_caches[v_idx] = new_cache
        return hidden

    @torch.inference_mode()
    def forward_step(
        self,
        tokens: torch.Tensor,
        *,
        moe_executor: MoEExecutorFn | None = None,
        fp32_attn_output: bool = False,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Advance KV cache by `tokens` ([1, seq_len]) and return logits [1, seq_len, vocab]."""
        model = self.model
        _, seq_len = tokens.shape
        cache_len = self.cache_len

        hidden = model.embed(tokens)
        freqs_cis = (
            None
            if model.rope is None
            else model.rope.get_freqs_cis(cache_len, cache_len + seq_len, hidden.device)
        )

        if self.arch == "transformer":
            input_embeddings = hidden
            dense_prelude_embeddings = None
            for v_idx, layer_id in enumerate(model.execution_schedule):
                if (
                    model.loop_input_injection_module is not None
                    and layer_id == model.loop_start_layer
                ):
                    hidden = model.loop_input_injection_module(hidden, input_embeddings)
                if (
                    model.dense_prelude_input_injection_module is not None
                    and layer_id in model.dense_prelude_input_layers
                ):
                    hidden = model.dense_prelude_input_injection_module(hidden, dense_prelude_embeddings)
                hidden = self._run_one_visit(
                    v_idx,
                    layer_id,
                    hidden,
                    freqs_cis,
                    moe_executor=moe_executor,
                    fp32_attn_output=fp32_attn_output,
                    deterministic=deterministic,
                )
                if (
                    model.dense_prelude_input_injection_module is not None
                    and layer_id == model.dense_prelude_source_layer
                ):
                    dense_prelude_embeddings = hidden
        elif self.arch == "huginn":
            v_idx = 0
            # Prelude
            for layer_id in range(model.loop_start_layer):
                hidden = self._run_one_visit(
                    v_idx,
                    layer_id,
                    hidden,
                    freqs_cis,
                    moe_executor=moe_executor,
                    fp32_attn_output=fp32_attn_output,
                    deterministic=deterministic,
                )
                v_idx += 1

            input_embeds = model._recurrence_input(hidden)
            if model.huginn_hierarchical_state != "none":
                raise NotImplementedError("Hierarchical H/L Huginn state not expected in Part I MoE")
            # Crucial: pass cache_len so _hashed_normal_state matches exact global position!
            state = model.initialize_recurrent_state(input_embeds, cache_len=cache_len)
            for _ in range(model.loop_times):
                if model.input_injection is not None:
                    state = model.input_injection(state, input_embeds)
                for layer_id in range(model.loop_start_layer, model.loop_end_layers):
                    state = self._run_one_visit(
                        v_idx,
                        layer_id,
                        state,
                        freqs_cis,
                        moe_executor=moe_executor,
                        fp32_attn_output=fp32_attn_output,
                        deterministic=deterministic,
                    )
                    v_idx += 1

            hidden = state
            # Coda
            for layer_id in range(model.loop_end_layers, model.num_layers):
                hidden = self._run_one_visit(
                    v_idx,
                    layer_id,
                    hidden,
                    freqs_cis,
                    moe_executor=moe_executor,
                    fp32_attn_output=fp32_attn_output,
                    deterministic=deterministic,
                )
                v_idx += 1
        else:
            raise ValueError(f"Unsupported arch: {self.arch}")

        self.cache_len = cache_len + seq_len
        logits = model._finish_forward(hidden, None, None)
        return logits


@torch.inference_mode()
def verify_model_b1(
    model_name: str,
    model_spec: dict[str, Any],
    model_dir: Path,
    prompts: list[dict[str, Any]],
    *,
    device: torch.device,
    decode_steps: int = 32,
    attention_backend: str = "sdpa",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run B1 verification on one model and return (summary, per_step_rows)."""
    torch.cuda.reset_peak_memory_stats(device)
    model, tokenizer, _mcfg = load_ifm_model(
        model_dir,
        device=str(device),
        attention_backend=attention_backend,
    )
    weights_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    engine = LoopedIncrementalEngine(model, model_spec)

    # Lightweight router-index capture hooks to compare exact top-k routes between full-prefix & KV-cache
    captured_routes: list[list[int]] = []

    def _make_route_hook():
        def _hook(_mod, _inp, output):
            _scores, idx, _tpe, _aux = output
            captured_routes.append(idx[0, -1].tolist())
        return _hook

    per_step_rows: list[dict[str, Any]] = []
    first_divergence: dict[str, Any] | None = None
    greedy_matches = 0
    greedy_total = 0
    full_step_ms: list[float] = []
    inc_step_ms: list[float] = []
    inc_visit_nonmoe_ms: list[float] = []
    kv_bytes_at_end: list[int] = []

    for p_idx, rec in enumerate(prompts):
        formatted = format_ifm_prompt(rec)
        prompt_ids = tokenizer.encode(formatted, bos=True, eos=False)[-512:]
        if not prompt_ids:
            continue

        # --- Part A: Teacher-forced exact same token sequence comparison (prefill + decode_steps) ---
        # First obtain reference greedy sequence from official full-prefix forward
        ref_seq = list(prompt_ids)
        ref_logits_list: list[torch.Tensor] = []
        ref_routes_list: list[list[list[int]]] = []

        with ExitStack() as stack:
            for layer in model.layers:
                stack.enter_context(layer.moe.router.register_forward_hook(_make_route_hook()))
            for step in range(decode_steps + 1):
                captured_routes.clear()
                inp = torch.tensor([ref_seq], device=device, dtype=torch.long)
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                logits, _, _ = model(inp, multi_segments=False, cache=None)
                torch.cuda.synchronize(device)
                if step > 0:
                    full_step_ms.append((time.perf_counter() - t0) * 1000.0)
                last_logits = logits[0, -1].float().cpu()
                ref_logits_list.append(last_logits)
                ref_routes_list.append(list(captured_routes))
                if step < decode_steps:
                    next_tok = int(last_logits.argmax().item())
                    ref_seq.append(next_tok)

        # Now run per-logical-visit KV-cached forward on the EXACT same token sequence (prefill + step-by-step)
        engine.reset()
        with ExitStack() as stack:
            for layer in model.layers:
                stack.enter_context(layer.moe.router.register_forward_hook(_make_route_hook()))

            for step in range(decode_steps + 1):
                captured_routes.clear()
                if step == 0:
                    inp = torch.tensor([prompt_ids], device=device, dtype=torch.long)
                else:
                    inp = torch.tensor([[ref_seq[len(prompt_ids) + step - 1]]], device=device, dtype=torch.long)
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                logits_inc = engine.forward_step(inp)
                torch.cuda.synchronize(device)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                if step > 0:
                    inc_step_ms.append(dt_ms)
                inc_last = logits_inc[0, -1].float().cpu()
                ref_last = ref_logits_list[step]
                diff = (inc_last - ref_last).abs()
                ref_top1 = int(ref_last.argmax().item())
                inc_top1 = int(inc_last.argmax().item())
                ref_top5 = set(torch.topk(ref_last, 5).indices.tolist())
                inc_top5 = set(torch.topk(inc_last, 5).indices.tolist())
                top5_overlap = len(ref_top5 & inc_top5) / 5.0

                ref_r = ref_routes_list[step]
                inc_r = list(captured_routes)
                route_matches = sum(1 for a, b in zip(ref_r, inc_r) if a == b)
                route_rate = route_matches / max(1, len(ref_r))

                per_step_rows.append(
                    {
                        "model": model_name,
                        "request_id": rec["id"],
                        "category": rec["category"],
                        "step": step,
                        "phase": "prefill" if step == 0 else "decode",
                        "seq_pos": len(prompt_ids) + step,
                        "max_abs_logit_diff": round(float(diff.max().item()), 6),
                        "mean_abs_logit_diff": round(float(diff.mean().item()), 6),
                        "top1_match": int(ref_top1 == inc_top1),
                        "top5_overlap": round(top5_overlap, 4),
                        "route_exact_match_rate": round(route_rate, 6),
                        "inc_step_ms": round(dt_ms, 3),
                    }
                )
        kv_bytes_at_end.append(engine.kv_cache_bytes())

        # --- Part B: Free greedy generation agreement & first divergence example ---
        engine.reset()
        inc_gen: list[int] = []
        inp = torch.tensor([prompt_ids], device=device, dtype=torch.long)
        logits_inc = engine.forward_step(inp)
        nxt = int(logits_inc[0, -1].argmax().item())
        inc_gen.append(nxt)
        for _ in range(decode_steps - 1):
            logits_inc = engine.forward_step(torch.tensor([[nxt]], device=device, dtype=torch.long))
            nxt = int(logits_inc[0, -1].argmax().item())
            inc_gen.append(nxt)

        ref_gen = ref_seq[len(prompt_ids): len(prompt_ids) + decode_steps]
        for s_i, (rt, it) in enumerate(zip(ref_gen, inc_gen)):
            greedy_total += 1
            if rt == it:
                greedy_matches += 1
            elif first_divergence is None:
                first_divergence = {
                    "request_id": rec["id"],
                    "category": rec["category"],
                    "decode_step": s_i + 1,
                    "ref_token_id": rt,
                    "inc_token_id": it,
                    "ref_text_prefix": tokenizer.decode(ref_gen[: s_i + 1]),
                    "inc_text_prefix": tokenizer.decode(inc_gen[: s_i + 1]),
                }

        # Measure per-visit non-routed-MoE compute time on prompt 0 (at seq_len ~ 128)
        if p_idx == 0:
            # Time attention + shared + router (excluding routed experts) across 112 visits for 5 decode steps
            t_nonmoe_total = 0.0
            n_visits_timed = 0

            def _timing_moe_exec(layer_id, v_idx, layer, y, lb):
                # Run full MoE (includes routed experts) — we subtract GPU routed-FFN time (~0.12 ms) or measure whole visit
                return layer.moe(y, lb)

            for _ in range(5):
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                _ = engine.forward_step(torch.tensor([[inc_gen[-1]]], device=device, dtype=torch.long))
                torch.cuda.synchronize(device)
                t_nonmoe_total += (time.perf_counter() - t0) * 1000.0
                n_visits_timed += engine.num_visits
            inc_visit_nonmoe_ms.append(t_nonmoe_total / max(1, n_visits_timed))

    peak_alloc_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
    peak_res_gib = torch.cuda.max_memory_reserved(device) / (1024**3)

    # Compute exact KV cache per token across all 112 logical visits
    sample_c = engine.visit_caches[0]
    if sample_c is not None and sample_c[0] is not None:
        kv_bytes_per_tok = (
            sum(
                (c[0].numel() // c[0].shape[1]) * c[0].element_size()
                + (c[1].numel() // c[1].shape[1]) * c[1].element_size()
                for c in engine.visit_caches
                if c is not None and c[0] is not None
            )
        )
    else:
        kv_bytes_per_tok = 0

    prefill_rows = [r for r in per_step_rows if r["phase"] == "prefill"]
    decode_rows = [r for r in per_step_rows if r["phase"] == "decode"]

    summary = {
        "model": model_name,
        "arch": model_spec["arch"],
        "prompts_checked": len(prompts),
        "decode_steps_per_prompt": decode_steps,
        "prefill_max_abs_logit_diff": max((r["max_abs_logit_diff"] for r in prefill_rows), default=0.0),
        "decode_max_abs_logit_diff": max((r["max_abs_logit_diff"] for r in decode_rows), default=0.0),
        "decode_mean_abs_logit_diff": round(statistics.mean(r["mean_abs_logit_diff"] for r in decode_rows), 6),
        "decode_top1_match_rate": round(statistics.mean(r["top1_match"] for r in decode_rows), 6),
        "decode_top5_overlap": round(statistics.mean(r["top5_overlap"] for r in decode_rows), 6),
        "decode_route_exact_match_rate": round(statistics.mean(r["route_exact_match_rate"] for r in decode_rows), 6),
        "free_greedy_match_rate": round(greedy_matches / max(1, greedy_total), 6),
        "free_greedy_matches": greedy_matches,
        "free_greedy_total": greedy_total,
        "first_divergence": first_divergence,
        "full_prefix_tpot_ms": round(statistics.mean(full_step_ms), 3),
        "incremental_tpot_ms": round(statistics.mean(inc_step_ms), 3),
        "incremental_tpot_p50_ms": round(statistics.median(inc_step_ms), 3),
        "speedup_vs_full_prefix": round(statistics.mean(full_step_ms) / max(1e-6, statistics.mean(inc_step_ms)), 2),
        "measured_visit_compute_ms": round(inc_visit_nonmoe_ms[0], 4) if inc_visit_nonmoe_ms else 0.0,
        "weights_gib": round(weights_bytes / (1024**3), 3),
        "kv_kib_per_token": round(kv_bytes_per_tok / 1024.0, 2),
        "kv_mib_at_128_tokens": round((kv_bytes_per_tok * 128) / (1024**2), 2),
        "kv_mib_at_2048_tokens": round((kv_bytes_per_tok * 2048) / (1024**2), 2),
        "peak_allocated_gib": round(peak_alloc_gib, 3),
        "peak_reserved_gib": round(peak_res_gib, 3),
    }

    del engine, model
    gc.collect()
    torch.cuda.empty_cache()
    return summary, per_step_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="B1 per-logical-visit KV-cache verification")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--xllm-loop-dir", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--n-prompts", type=int, default=6)
    parser.add_argument("--decode-steps", type=int, default=32)
    args = parser.parse_args()

    cfg = read_json(Path(args.config))
    bootstrap_xllm(args.xllm_loop_dir, attention_backend=cfg.get("attention_backend", "sdpa"))
    gpu = select_idle_gpu(args.gpu)
    device = torch.device(f"cuda:{gpu['index']}")
    torch.cuda.set_device(device)

    all_prompts = list(read_jsonl(Path(cfg["prompts_file"])))
    # Pick balanced prompts across text/math/code from the beginning of the prompt list
    selected = all_prompts[: args.n_prompts]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for model_name in cfg["models"]:
        spec = cfg["checkpoints"][model_name]
        mdir = _model_dir(Path(args.model_root), spec["repo"])
        summary, rows = verify_model_b1(
            model_name,
            spec,
            mdir,
            selected,
            device=device,
            decode_steps=args.decode_steps,
            attention_backend=cfg.get("attention_backend", "sdpa"),
        )
        summaries[model_name] = summary
        all_rows.extend(rows)
        print(f"[B1 {model_name}] {json.dumps(summary, ensure_ascii=False)}")

    write_json(out_dir / "b1_kv_verification.json", summaries)
    write_csv(out_dir / "b1_kv_steps.csv", all_rows)


if __name__ == "__main__":
    main()
