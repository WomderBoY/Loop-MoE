"""Official xLLM-Loop runtime loader, pure-PyTorch SM120 shims, and exact MoE route hooks."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
import types
from typing import Any, Iterator

import torch
import torch.nn.functional as F


def _install_xllm_extension_shim() -> bool:
    """Install pure-PyTorch xllm_extension shim if native CUDA extension is absent.

    Matches xllm/csrc/ops/group_rms_norm_kernel.cu arithmetic: FP32 accumulation
    with rsqrt(mean(x^2) + eps) and gamma applied in FP32 before casting to input dtype.
    """
    try:
        import xllm_extension  # noqa: F401
        return False
    except ImportError:
        pass

    ext = types.ModuleType("xllm_extension")
    ops = types.ModuleType("xllm_extension.ops")
    te = types.ModuleType("xllm_extension.te")
    ext.__file__ = __file__
    ops.__file__ = __file__
    te.__file__ = __file__

    def group_rms_norm_fwd_affine(
        x: torch.Tensor,
        num_features: int,
        num_groups: int,
        gamma: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        group_dim = num_features // num_groups
        x_f = x.float().reshape(*orig_shape[:-1], num_groups, group_dim)
        rstd = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
        g_f = gamma.float().reshape(num_groups, group_dim)
        y = (x_f * rstd * g_f).reshape(orig_shape).to(x.dtype)
        return y, rstd.reshape(-1, num_groups)

    def group_rms_norm_fwd(
        x: torch.Tensor,
        num_features: int,
        num_groups: int,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        group_dim = num_features // num_groups
        x_f = x.float().reshape(*orig_shape[:-1], num_groups, group_dim)
        rstd = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
        y = (x_f * rstd).reshape(orig_shape).to(x.dtype)
        return y, rstd.reshape(-1, num_groups)

    def group_layer_norm_fwd_affine(
        x: torch.Tensor,
        num_features: int,
        num_groups: int,
        gamma: torch.Tensor,
        beta: torch.Tensor | None,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        group_dim = num_features // num_groups
        x_f = x.float().reshape(*orig_shape[:-1], num_groups, group_dim)
        mean = x_f.mean(dim=-1, keepdim=True)
        var = (x_f - mean).pow(2).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + eps)
        y = (x_f - mean) * rstd * gamma.float().reshape(num_groups, group_dim)
        if beta is not None:
            y = y + beta.float().reshape(num_groups, group_dim)
        return y.reshape(orig_shape).to(x.dtype), mean.reshape(-1, num_groups), rstd.reshape(-1, num_groups)

    ops.group_rms_norm_fwd_affine = group_rms_norm_fwd_affine
    ops.group_rms_norm_fwd = group_rms_norm_fwd
    ops.group_layer_norm_fwd_affine = group_layer_norm_fwd_affine

    def _missing_op(name: str):
        def _stub(*_args, **_kwargs):
            raise NotImplementedError(f"xllm_extension.ops.{name} is not implemented in inference shim")

        return _stub

    def _ops_getattr(name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _missing_op(name)

    def _te_getattr(name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _missing_op(f"te.{name}")

    ops.__getattr__ = _ops_getattr  # type: ignore[attr-defined]

    class _DType:
        kByte = 0
        kFloat8E4M3 = 1
        kFloat8E5M2 = 2
        kInt32 = 3
        kFloat32 = 4
        kFloat16 = 5
        kBFloat16 = 6

    te.DType = _DType
    te.__getattr__ = _te_getattr  # type: ignore[attr-defined]

    ext.ops = ops
    ext.te = te
    sys.modules["xllm_extension"] = ext
    sys.modules["xllm_extension.ops"] = ops
    sys.modules["xllm_extension.te"] = te
    return True


def _sdpa_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    max_seqlen_q: int | None,
    max_seqlen_k: int | None,
    scale: float,
    dropout: float = 0.0,
    use_causal_mask: bool = True,
    deterministic: bool = True,
    training: bool = False,
) -> torch.Tensor:
    """Drop-in SDPA replacement for xllm flash_attention during single-segment eval."""
    if cu_seqlens_q is not None or cu_seqlens_k is not None:
        raise NotImplementedError("SDPA shim supports non-varlen inference only")
    # q, k, v arrive as [B, L, H, D]; PyTorch SDPA expects [B, H, L, D].
    qt = q.transpose(1, 2)
    kt = k.transpose(1, 2)
    vt = v.transpose(1, 2)
    n_heads = qt.shape[1]
    n_kv = kt.shape[1]
    l_q = qt.shape[2]
    l_k = kt.shape[2]
    attn_mask = None
    is_causal = use_causal_mask
    if use_causal_mask and l_q != l_k:
        # FlashAttention aligns causal masks to the bottom-right corner; PyTorch SDPA's
        # is_causal=True aligns to top-left. Handle incremental KV-cache decoding explicitly.
        is_causal = False
        if l_q > 1:
            attn_mask = torch.ones((l_q, l_k), dtype=torch.bool, device=qt.device).tril(diagonal=l_k - l_q)
    out = F.scaled_dot_product_attention(
        qt,
        kt,
        vt,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=(n_heads != n_kv),
    )
    return out.transpose(1, 2).contiguous()


def bootstrap_xllm(xllm_loop_dir: str | Path, attention_backend: str = "sdpa") -> dict[str, Any]:
    """Add xllm-loop to sys.path and install SM120-safe pure-PyTorch shims if needed."""
    xllm_path = str(Path(xllm_loop_dir).resolve())
    if xllm_path not in sys.path:
        sys.path.insert(0, xllm_path)

    os.environ.setdefault("ENABLE_FLASH_ATTENTION_3", "false")
    os.environ.setdefault("ENABLE_FLASH_ATTENTION_4", "false")

    used_ext_shim = _install_xllm_extension_shim()

    used_fa_shim = False
    try:
        import flash_attn_2_cuda  # noqa: F401
    except ImportError:
        stub = types.ModuleType("flash_attn_2_cuda")
        stub.__file__ = __file__
        sys.modules["flash_attn_2_cuda"] = stub
        used_fa_shim = True

    import xllm.modules.causal_attention as ca_mod

    if attention_backend == "sdpa" or used_fa_shim:
        ca_mod.flash_attention = _sdpa_flash_attention

    return {
        "xllm_loop_dir": xllm_path,
        "used_xllm_extension_shim": used_ext_shim,
        "used_flash_attn_sdpa_shim": attention_backend == "sdpa" or used_fa_shim,
        "attention_backend": attention_backend,
    }


def load_ifm_model(
    artifact_dir: str | Path,
    *,
    device: str = "cuda:0",
    attention_backend: str = "sdpa",
):
    """Load an IFM Part I artifact with official manifest verification and strict state dict."""
    from xllm.paper_part1.native_inference import load_native_model

    model, tokenizer, model_config = load_native_model(artifact_dir, device=device)
    if attention_backend == "exact":
        for layer in model.layers:
            layer.attention.causal_attention.backend = None
    return model, tokenizer, model_config


def build_ifm_topology(model, model_spec: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[tuple[str, int | None]]]:
    """Validate and return (pools_meta, execution_order) for IFM Ouro or Huginn."""
    prelude_start, prelude_end = model_spec["prelude_layers"]
    rec_start, rec_end = model_spec["recurrent_layers"]
    coda_start, coda_end = model_spec["coda_layers"]

    pools: dict[str, dict[str, Any]] = {}
    for layer_id, layer in enumerate(model.layers):
        if not hasattr(layer, "moe"):
            raise ValueError(f"Expected MoE layer at layer_id={layer_id}")
        experts = layer.moe.experts
        n_exp = int(experts.n_experts)
        # weight1: [n*inter, d], weight2: [n*d, inter], weight3: [n*inter, d]
        total_expert_bytes = sum(
            w.numel() * w.element_size() for w in (experts.weight1, experts.weight2, experts.weight3)
        )
        if total_expert_bytes % n_exp != 0:
            raise ValueError(f"Non-divisible expert weight bytes at layer {layer_id}")
        bpe = total_expert_bytes // n_exp
        if prelude_start <= layer_id < prelude_end:
            region = "prelude"
        elif rec_start <= layer_id < rec_end:
            region = "recurrent"
        elif coda_start <= layer_id < coda_end:
            region = "coda"
        else:
            raise ValueError(f"Layer {layer_id} outside prelude/recurrent/coda partition")
        pools[f"layer:{layer_id:02d}"] = {
            "layer_id": layer_id,
            "region": region,
            "experts": n_exp,
            "bytes_per_expert": bpe,
        }

    if len(pools) != model_spec["expected_pools"]:
        raise ValueError(f"Pool count mismatch: {len(pools)} != {model_spec['expected_pools']}")

    # Build execution_order directly from model.execution_schedule
    visit_counter: dict[int, int] = {}
    order: list[tuple[str, int | None]] = []
    for layer_id in model.execution_schedule:
        pool_name = f"layer:{layer_id:02d}"
        region = pools[pool_name]["region"]
        if region == "recurrent":
            loop_idx = visit_counter.get(layer_id, 0)
            visit_counter[layer_id] = loop_idx + 1
            order.append((pool_name, loop_idx))
        else:
            if layer_id in visit_counter:
                raise ValueError(f"Non-recurrent layer {layer_id} visited multiple times")
            visit_counter[layer_id] = 1
            order.append((pool_name, None))

    if len(order) != model_spec["expected_visits"]:
        raise ValueError(f"Logical visit count mismatch: {len(order)} != {model_spec['expected_visits']}")
    max_loops = max((lp for _, lp in order if lp is not None), default=-1) + 1
    if max_loops != model_spec["expected_loops"]:
        raise ValueError(f"Loop count mismatch: {max_loops} != {model_spec['expected_loops']}")
    for pinfo in pools.values():
        if pinfo["experts"] != model_spec["num_experts"]:
            raise ValueError(f"Expert count mismatch: {pinfo['experts']} != {model_spec['num_experts']}")
        if pinfo["bytes_per_expert"] != model_spec["bytes_per_expert"]:
            raise ValueError(
                f"Bytes/expert mismatch: {pinfo['bytes_per_expert']} != {model_spec['bytes_per_expert']}"
            )
    return pools, order


@dataclass
class _CapturedSlice:
    pool: str
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    topk_raw_scores: torch.Tensor
    top_cand_ids: torch.Tensor
    top_cand_scores: torch.Tensor
    top_cand_logits: torch.Tensor
    logit_margin: torch.Tensor
    sigmoid_margin: torch.Tensor
    entropy: torch.Tensor
    router_input: torch.Tensor
    raw_logits: torch.Tensor


@contextmanager
def ifm_route_collector(
    model,
    model_spec: dict[str, Any],
    *,
    top_candidates: int = 6,
) -> Iterator[tuple[dict[str, dict[str, Any]], list[tuple[str, int | None]], Any]]:
    """Attach zero-side-effect hooks to every MoE TopKRouter in exact execution order."""
    pools, order = build_ifm_topology(model, model_spec)
    state: dict[str, Any] = {"start": 0, "stop": None, "records": []}

    def _make_hook(pool_name: str, router):
        def _hook(_module, inputs, output):
            x = inputs[0]  # [B, L, D] normalized router input
            routing_scores, routing_indices, _tpe, _aux = output
            start = state["start"]
            stop = x.shape[1] if state["stop"] is None else state["stop"]
            x_sl = x[0, start:stop]  # [S, D]
            idx_sl = routing_indices[0, start:stop]  # [S, K]
            w_sl = routing_scores[0, start:stop].float()  # [S, K]

            # Match TopKRouting.forward exactly: run F.linear on full x before slicing
            # so cuBLAS uses the identical GEMM kernel/reduction order as the model.
            raw_logits = F.linear(x, router.weight).to(torch.float32)[0, start:stop]  # [S, E]
            if router.score_func == "sigmoid":
                raw_scores = torch.sigmoid(raw_logits)
            elif router.score_func == "softmax":
                raw_scores = F.softmax(raw_logits, dim=-1, dtype=torch.float32)
            else:
                raise ValueError(f"Unsupported router score_func: {router.score_func}")
            rank_scores = raw_scores + router.bias.to(raw_scores) if router.bias is not None else raw_scores

            k = int(router.topk)
            check_vals, check_ids = torch.topk(rank_scores, k, dim=-1)
            if not torch.equal(check_ids, idx_sl):
                # Allow exact score ties (within 1e-5) while always recording model's actual idx_sl
                model_vals = rank_scores.gather(-1, idx_sl)
                max_gap = float((check_vals.min(dim=-1).values - model_vals.min(dim=-1).values).abs().max().item())
                if max_gap > 1e-4:
                    raise RuntimeError(
                        f"Hook router top-k mismatch at {pool_name}: max_score_gap={max_gap:.6g}"
                    )

            c_k = min(top_candidates, raw_scores.shape[-1])
            cand_scores, cand_ids = torch.topk(rank_scores, c_k, dim=-1)
            cand_logits = raw_logits.gather(-1, cand_ids)
            topk_raw = raw_scores.gather(-1, idx_sl)

            # Margins between rank k and rank k+1 in raw logits (and sigmoid scores)
            logit_margin = cand_logits[:, k - 1] - cand_logits[:, k]
            sigmoid_margin = cand_scores[:, k - 1] - cand_scores[:, k]

            # Normalized routing distribution entropy over experts
            probs = raw_scores / raw_scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            entropy = -(probs * probs.clamp_min(1e-30).log()).sum(dim=-1) / math.log(raw_scores.shape[-1])

            state["records"].append(
                _CapturedSlice(
                    pool=pool_name,
                    topk_ids=idx_sl.detach(),
                    topk_weights=w_sl.detach(),
                    topk_raw_scores=topk_raw.detach(),
                    top_cand_ids=cand_ids.detach(),
                    top_cand_scores=cand_scores.detach(),
                    top_cand_logits=cand_logits.detach(),
                    logit_margin=logit_margin.detach(),
                    sigmoid_margin=sigmoid_margin.detach(),
                    entropy=entropy.detach(),
                    router_input=x_sl.detach().float(),
                    raw_logits=raw_logits.detach(),
                )
            )

        return _hook

    def _forward_and_capture(tokens: torch.Tensor, *, start: int, stop: int | None = None):
        state["start"] = start
        state["stop"] = stop
        state["records"] = []
        logits, _, cache = model(tokens, multi_segments=False, cache=None)
        if cache is not None:
            raise RuntimeError("Unexpected non-None cache in full-prefix forward")
        records: list[_CapturedSlice] = state["records"]
        if [r.pool for r in records] != [p for p, _ in order]:
            raise RuntimeError(
                f"Execution order mismatch: captured {len(records)} visits vs expected {len(order)}"
            )

        # Convert captured tensors into per-position compact visit lists
        n_pos = records[0].topk_ids.shape[0]
        prev_inputs: dict[str, torch.Tensor] = {}
        prev_logits: dict[str, torch.Tensor] = {}

        per_visit_cos: list[list[float | None]] = []
        per_visit_linf: list[list[float | None]] = []
        for rec, (pool_name, loop_idx) in zip(records, order):
            if loop_idx is not None and pool_name in prev_inputs:
                cos = F.cosine_similarity(prev_inputs[pool_name], rec.router_input, dim=-1).tolist()
                linf = (rec.raw_logits - prev_logits[pool_name]).abs().max(dim=-1).values.tolist()
            else:
                cos = [None] * n_pos
                linf = [None] * n_pos
            if loop_idx is not None:
                prev_inputs[pool_name] = rec.router_input
                prev_logits[pool_name] = rec.raw_logits
            per_visit_cos.append(cos)
            per_visit_linf.append(linf)

        # Bulk transfer to CPU lists
        cpu_visits = [
            (
                r.topk_ids.tolist(),
                r.topk_weights.tolist(),
                r.topk_raw_scores.tolist(),
                r.top_cand_ids.tolist(),
                r.top_cand_scores.tolist(),
                r.top_cand_logits.tolist(),
                r.logit_margin.tolist(),
                r.sigmoid_margin.tolist(),
                r.entropy.tolist(),
                cos_list,
                linf_list,
            )
            for r, cos_list, linf_list in zip(records, per_visit_cos, per_visit_linf)
        ]

        steps_visits: list[list[list[Any]]] = []
        for pos in range(n_pos):
            pos_visits = [
                [
                    v[0][pos],
                    [round(x, 6) for x in v[1][pos]],
                    [round(x, 6) for x in v[2][pos]],
                    v[3][pos],
                    [round(x, 6) for x in v[4][pos]],
                    [round(x, 5) for x in v[5][pos]],
                    round(float(v[6][pos]), 5),
                    round(float(v[7][pos]), 6),
                    round(float(v[8][pos]), 6),
                    None if v[9][pos] is None else round(float(v[9][pos]), 6),
                    None if v[10][pos] is None else round(float(v[10][pos]), 5),
                ]
                for v in cpu_visits
            ]
            steps_visits.append(pos_visits)
        state["records"] = []
        return logits, steps_visits

    with ExitStack() as stack:
        for layer_id, layer in enumerate(model.layers):
            pool_name = f"layer:{layer_id:02d}"
            stack.enter_context(layer.moe.router.register_forward_hook(_make_hook(pool_name, layer.moe.router)))
        yield pools, order, _forward_and_capture


@torch.inference_mode()
def verify_attention_backend(model, tokenizer, model_spec: dict[str, Any], device: str = "cuda:0") -> dict[str, Any]:
    """Compare SDPA attention vs official causal_attn_backend=None exact attention."""
    sample = (
        "Modern computer processors rely on a hierarchy of caches to bridge the gap "
        "between arithmetic speed and main memory latency."
    )
    ids = tokenizer.encode(sample, bos=True, eos=False)[:32]
    tokens = torch.tensor([ids], device=device, dtype=torch.long)

    orig_backends = [layer.attention.causal_attention.backend for layer in model.layers]
    try:
        with ifm_route_collector(model, model_spec, top_candidates=4) as (_pools, _order, fwd):
            for layer in model.layers:
                layer.attention.causal_attention.backend = "flash"
            logits_sdpa, visits_sdpa = fwd(tokens, start=0, stop=len(ids))
            logits_sdpa_f = logits_sdpa.float().cpu()

            for layer in model.layers:
                layer.attention.causal_attention.backend = None
            logits_exact, visits_exact = fwd(tokens, start=0, stop=len(ids))
            logits_exact_f = logits_exact.float().cpu()
    finally:
        for layer, b in zip(model.layers, orig_backends):
            layer.attention.causal_attention.backend = b

    diff = (logits_sdpa_f - logits_exact_f).abs()
    top1_match = bool((logits_sdpa_f[:, -1].argmax(dim=-1) == logits_exact_f[:, -1].argmax(dim=-1)).all().item())
    total_visits = 0
    exact_route_matches = 0
    for pos_a, pos_b in zip(visits_sdpa, visits_exact):
        for va, vb in zip(pos_a, pos_b):
            total_visits += 1
            if va[0] == vb[0]:
                exact_route_matches += 1
    return {
        "seq_len": len(ids),
        "max_abs_logit_diff": float(diff.max().item()),
        "mean_abs_logit_diff": float(diff.mean().item()),
        "last_token_top1_match": top1_match,
        "topk_route_exact_match_rate": exact_route_matches / total_visits if total_visits else 1.0,
        "total_checked_visits": total_visits,
    }
