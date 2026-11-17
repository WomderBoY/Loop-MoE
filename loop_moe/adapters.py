"""Expose actual router decisions in physical expert-pool order."""

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass


@dataclass
class RoutePoint:
    probs: object
    weights: object
    ids: object
    residual: object


def expert_pool(weights, count):
    if any(weight.shape[0] != count for weight in weights):
        raise ValueError("Expert weight tensors must be indexed by physical expert ID")
    return {
        "experts": count,
        "bytes_per_expert": sum(weight[0].numel() * weight.element_size() for weight in weights),
    }


def stacked_pool(layer):
    return expert_pool((layer.experts_w1, layer.experts_w2, layer.experts_w3), layer.num_experts)


def foil_collector(model):
    core = model.model
    head = {f"head:{i}": stacked_pool(layer) for i, layer in enumerate(core.head_layers)}
    loop = {f"loop:{i}": stacked_pool(layer) for i, layer in enumerate(core.stack.layers)}
    tail = {f"tail:{i}": stacked_pool(layer) for i, layer in enumerate(core.tail_layers)}
    order = (
        [(pool, None) for pool in head]
        + [(pool, step) for step in range(core.num_stacks) for pool in loop]
        + [(pool, None) for pool in tail]
    )

    def forward(input_ids):
        sink = {}
        output = model(input_ids=input_ids, trace_sink=sink)
        points = []
        if sorted(sink) != list(range(len(order))):
            raise RuntimeError("Foil trace does not match the model's unrolled depth")
        for depth, (pool, step) in enumerate(order):
            point = sink[depth]
            block, _, layer = pool.partition(":")
            if point.block != block or (block == "loop" and
                                        (point.layer_idx != int(layer) or point.loop_step != step)):
                raise RuntimeError(f"Foil trace does not match the physical expert pool at depth {depth}")
            points.append(RoutePoint(point.router_probs, point.topk_weights, point.topk_idx, point.residual))
        return output, points

    return {**head, **loop, **tail}, order, forward


def layer_loop(torch, core, steps):
    # The stock Qwen3-MoE model applies each layer once. Repeat the same
    # module with the same positions, with KV caching disabled.
    class RepeatedLayer(torch.nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, hidden_states, **kwargs):
            if kwargs.get("use_cache") or kwargs.get("past_key_values") is not None:
                raise ValueError("Loopie collection requires a cache-free forward")
            for _ in range(steps):
                hidden_states = self.layer(hidden_states, **kwargs)
            return hidden_states

    core.layers = torch.nn.ModuleList(RepeatedLayer(layer) for layer in core.layers)


@contextmanager
def route_collector(torch, model, architecture):
    if architecture == "foil":
        yield foil_collector(model)
        return

    if architecture == "loopie":
        core = model.model
        layers = list(core.layers)
        loop_count = model.config.layer_loop_times
        if loop_count < 2 or model.config.model_loop_times != 1:
            raise ValueError("Expected Loopie with layer-wise recurrence")
        pools = {f"loop:{i}": expert_pool((layer.mlp.experts.gate_up_proj, layer.mlp.experts.down_proj),
                                         layer.mlp.experts.num_experts)
                 for i, layer in enumerate(layers)}
        order = [(pool, step) for pool in pools for step in range(loop_count)]
        routers = [layer.mlp.gate for layer in layers]
        layer_loop(torch, core, loop_count)
        forward_options = {"use_cache": False, "logits_to_keep": 1}
    else:
        core = model.model
        layers = list(core.stack.layers)
        pools = {f"loop:{i}": stacked_pool(layer) for i, layer in enumerate(layers)}
        order = [(pool, step) for step in range(core.num_stacks) for pool in pools]
        routers = [layer.router for layer in layers]
        forward_options = {"use_cache": False}

    pending = {}
    visits = []

    def capture_router(pool):
        def hook(_module, _inputs, output):
            if pool in pending:
                raise RuntimeError(f"Router fired again before block output for {pool}")
            pending[pool] = output
        return hook

    def capture_block(pool):
        def hook(_module, _inputs, output):
            if architecture == "loopie":
                logits, weights, ids = pending.pop(pool)
                probs = logits.softmax(dim=-1, dtype=torch.float32)
                point = RoutePoint(probs.unsqueeze(0), weights.unsqueeze(0), ids.unsqueeze(0), output)
            else:
                _, probs, weights, ids = pending.pop(pool)
                point = RoutePoint(probs, weights, ids, output[0])
            visits.append((pool, point))
        return hook

    def forward(input_ids):
        visits.clear()
        output = model(input_ids=input_ids, **forward_options)
        if pending or [pool for pool, _ in visits] != [pool for pool, _ in order]:
            raise RuntimeError(f"{architecture} route hooks did not match the expected execution order")
        points = [point for _, point in visits]
        visits.clear()
        return output, points

    with ExitStack() as hooks:
        for pool, layer, router in zip(pools, layers, routers):
            hooks.enter_context(router.register_forward_hook(capture_router(pool)))
            hooks.enter_context(layer.register_forward_hook(capture_block(pool)))
        yield pools, order, forward
