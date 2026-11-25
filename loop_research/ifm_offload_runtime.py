"""Minimal vendored IFM KV/expert runtime from the frozen scheduling archive.

Only the verified engine, topology validation, packed expert store and direct
GPU executor are copied. The archived CLI modules depend on files removed
when the project was consolidated, so importing their full CLI is unsuitable.
Source functions/classes are copied verbatim; original snapshot: 20e9669.
"""
from __future__ import annotations
from typing import Any, Callable
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
MoEExecutorFn = Callable[[int, int, Any, torch.Tensor, str | None], tuple[torch.Tensor, Any]]


# Copied from archive/moe-scheduling/ifm_phase/loop_moe/ifm_runtime.py; sha256=3468ec13a1bd6ba3d50e7151ef445ad4bda1f716f2cdf5e2c53b7c6f779ac39d

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


# Copied from archive/moe-scheduling/ifm_phase/loop_moe/ifm_b1_kv_verify.py; sha256=91a0e4b0f8e784209b1a9409a1775162f3cee0c9f1ae1f6015b7eda6d1d15997

class LoopedIncrementalEngine:
    """Exact per-logical-visit KV-cached incremental engine for IFM Ouro & Huginn."""

    def __init__(self, model, model_spec: dict[str, Any]):
        self.model = model
        self.model_spec = model_spec
        self.arch = model_spec["arch"]
        self.pools, self.order = build_ifm_topology(model, model_spec)
        self.num_visits = len(self.order)
        # xllm CausalAttention only creates/updates KV cache when cache is not None:
        # pass (None, None, 0) on prefill so it returns (cache_k, cache_v, seq_len).
        self.visit_caches: list[tuple[torch.Tensor | None, torch.Tensor | None, int]] = [
            (None, None, 0) for _ in range(self.num_visits)
        ]
        self.cache_len: int = 0

    def reset(self) -> None:
        self.visit_caches = [(None, None, 0) for _ in range(self.num_visits)]
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


# Copied from archive/moe-scheduling/ifm_phase/loop_moe/ifm_b2_gpu_fast.py; sha256=90fad631ce8970062c5b538cb99a7023e3cc339687f2c78fd8bdadeff3a9bb62

class FastPinnedExpertStore:
    """Packs all 700 routed experts into pinned host RAM (indexed as [p][e]) and frees GPU copies."""

    def __init__(self, model, device: torch.device):
        self.device = device
        self.num_layers = len(model.layers)
        first_exp = model.layers[0].moe.experts
        self.model_dim = int(first_exp.model_dim)
        self.inter_dim = int(first_exp.inter_dim)
        self.n_experts = int(first_exp.n_experts)
        self.w1_numel = self.inter_dim * self.model_dim
        self.w2_numel = self.model_dim * self.inter_dim
        self.w3_numel = self.inter_dim * self.model_dim
        self.expert_numel = self.w1_numel + self.w2_numel + self.w3_numel
        self.bytes_per_expert = self.expert_numel * 2  # bf16

        self.host_experts: list[list[torch.Tensor]] = []
        with torch.no_grad():
            for layer_id, layer in enumerate(model.layers):
                exp = layer.moe.experts
                w1 = exp.weight1.detach().view(self.n_experts, self.inter_dim, self.model_dim).cpu()
                w2 = exp.weight2.detach().view(self.n_experts, self.model_dim, self.inter_dim).cpu()
                w3 = exp.weight3.detach().view(self.n_experts, self.inter_dim, self.model_dim).cpu()
                layer_list: list[torch.Tensor] = []
                o1 = self.w1_numel
                o2 = o1 + self.w2_numel
                for e in range(self.n_experts):
                    flat = torch.empty(self.expert_numel, dtype=torch.bfloat16, pin_memory=True)
                    flat[:o1].copy_(w1[e].reshape(-1))
                    flat[o1:o2].copy_(w2[e].reshape(-1))
                    flat[o2:].copy_(w3[e].reshape(-1))
                    layer_list.append(flat)
                self.host_experts.append(layer_list)
                exp.weight1 = nn.Parameter(torch.empty(0, device=device, dtype=torch.bfloat16), requires_grad=False)
                exp.weight2 = nn.Parameter(torch.empty(0, device=device, dtype=torch.bfloat16), requires_grad=False)
                exp.weight3 = nn.Parameter(torch.empty(0, device=device, dtype=torch.bfloat16), requires_grad=False)

        gc.collect()
        torch.cuda.empty_cache()

    def expert_views(self, slot_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        o1 = self.w1_numel
        o2 = o1 + self.w2_numel
        return (
            slot_tensor[:o1].view(self.inter_dim, self.model_dim),
            slot_tensor[o1:o2].view(self.model_dim, self.inter_dim),
            slot_tensor[o2:].view(self.inter_dim, self.model_dim),
        )

class FullResidentFastExecutor:
    """Apples-to-apples full-resident GPU executor using the exact same active-2-expert F.linear path."""

    def __init__(self, model):
        self.w1: list[torch.Tensor] = []
        self.w2: list[torch.Tensor] = []
        self.w3: list[torch.Tensor] = []
        for layer in model.layers:
            exp = layer.moe.experts
            n_e = int(exp.n_experts)
            d = int(exp.model_dim)
            inter = int(exp.inter_dim)
            self.w1.append(exp.weight1.detach().view(n_e, inter, d))
            self.w2.append(exp.weight2.detach().view(n_e, d, inter))
            self.w3.append(exp.weight3.detach().view(n_e, inter, d))

    def execute_moe(
        self,
        layer_id: int,
        v_idx: int,
        layer: Any,
        x_in: torch.Tensor,
        load_balancing_type: str | None,
    ) -> tuple[torch.Tensor, Any]:
        moe = layer.moe
        bsz, slen, dim = x_in.shape
        residual = x_in
        x = moe.norm(x_in)
        if moe.fc1 is not None:
            y1 = moe.fc2(F.silu(moe.fc1(x)) * moe.fc3(x))
        else:
            y1 = None
        routing_scores, routing_indices, _tpe, aux_loss = moe.router(x, load_balancing_type)
        x_2d = x.view(bsz * slen, dim)
        w1_l = self.w1[layer_id]
        w2_l = self.w2[layer_id]
        w3_l = self.w3[layer_id]

        if slen == 1:
            e0, e1 = int(routing_indices[0, 0, 0].item()), int(routing_indices[0, 0, 1].item())
            w_scores = routing_scores[0, 0]
            y0 = F.linear(F.silu(F.linear(x_2d, w1_l[e0])) * F.linear(x_2d, w3_l[e0]), w2_l[e0])
            y_b = F.linear(F.silu(F.linear(x_2d, w1_l[e1])) * F.linear(x_2d, w3_l[e1]), w2_l[e1])
            y2 = (y0 * w_scores[0].to(y0.dtype) + y_b * w_scores[1].to(y_b.dtype)).view(bsz, slen, dim)
        else:
            idx_flat = routing_indices[0]
            w_flat = routing_scores[0]
            y2 = torch.zeros_like(x_2d)
            for e in torch.unique(idx_flat).tolist():
                mask = idx_flat == e
                tok_pos, k_pos = torch.nonzero(mask, as_tuple=True)
                xe = x_2d.index_select(0, tok_pos)
                ye = F.linear(F.silu(F.linear(xe, w1_l[e])) * F.linear(xe, w3_l[e]), w2_l[e])
                y2.index_add_(0, tok_pos, ye * w_flat[tok_pos, k_pos].unsqueeze(-1).to(ye.dtype))
            y2 = y2.view(bsz, slen, dim)

        y = y1 + y2 if y1 is not None else y2
        return moe.residual(y, residual, c=y), aux_loss
