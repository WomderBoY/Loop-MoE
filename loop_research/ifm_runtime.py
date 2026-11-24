"""Official xLLM-Loop runtime loader with exact pure-PyTorch forward+backward SM120 shims."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _group_rms_norm_fwd_affine(
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


def _group_rms_norm_fwd(
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


def _group_rms_norm_bwd_affine(
    y_grad: torch.Tensor,
    x_or_y: torch.Tensor,
    num_features: int,
    num_groups: int,
    rstd: torch.Tensor,
    gamma: torch.Tensor,
    memory_efficient: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact FP32-accumulated backward for GroupRMSNorm with affine gamma."""
    orig_shape = y_grad.shape
    group_dim = num_features // num_groups
    dy_f = y_grad.float().reshape(-1, num_groups, group_dim)
    g_f = gamma.float().reshape(1, num_groups, group_dim)
    rstd_f = rstd.float().reshape(-1, num_groups, 1)

    if memory_efficient:
        y_f = x_or_y.float().reshape(-1, num_groups, group_dim)
        x_hat = y_f / g_f.clamp_min(1e-12)
    else:
        x_f = x_or_y.float().reshape(-1, num_groups, group_dim)
        x_hat = x_f * rstd_f

    gamma_grad = (dy_f * x_hat).sum(dim=0).reshape(gamma.shape).to(gamma.dtype)
    dy_g = dy_f * g_f
    proj = (dy_g * x_hat).mean(dim=-1, keepdim=True)
    x_grad = (rstd_f * (dy_g - x_hat * proj)).reshape(orig_shape).to(y_grad.dtype)
    return x_grad, gamma_grad


def _group_rms_norm_bwd(
    y_grad: torch.Tensor,
    x_or_y: torch.Tensor,
    num_features: int,
    num_groups: int,
    rstd: torch.Tensor,
    memory_efficient: bool = False,
) -> torch.Tensor:
    orig_shape = y_grad.shape
    group_dim = num_features // num_groups
    dy_f = y_grad.float().reshape(-1, num_groups, group_dim)
    rstd_f = rstd.float().reshape(-1, num_groups, 1)
    if memory_efficient:
        x_hat = x_or_y.float().reshape(-1, num_groups, group_dim)
    else:
        x_hat = x_or_y.float().reshape(-1, num_groups, group_dim) * rstd_f
    proj = (dy_f * x_hat).mean(dim=-1, keepdim=True)
    x_grad = (rstd_f * (dy_f - x_hat * proj)).reshape(orig_shape).to(y_grad.dtype)
    return x_grad


def _group_layer_norm_fwd_affine(
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


def _group_layer_norm_bwd_affine(
    y_grad: torch.Tensor,
    x_or_y: torch.Tensor,
    num_features: int,
    num_groups: int,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor | None,
    memory_efficient: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    orig_shape = y_grad.shape
    group_dim = num_features // num_groups
    dy_f = y_grad.float().reshape(-1, num_groups, group_dim)
    g_f = gamma.float().reshape(1, num_groups, group_dim)
    rstd_f = rstd.float().reshape(-1, num_groups, 1)
    if memory_efficient:
        y_f = x_or_y.float().reshape(-1, num_groups, group_dim)
        if beta is not None:
            y_f = y_f - beta.float().reshape(1, num_groups, group_dim)
        x_hat = y_f / g_f.clamp_min(1e-12)
    else:
        x_f = x_or_y.float().reshape(-1, num_groups, group_dim)
        mean_f = mean.float().reshape(-1, num_groups, 1)
        x_hat = (x_f - mean_f) * rstd_f

    gamma_grad = (dy_f * x_hat).sum(dim=0).reshape(gamma.shape).to(gamma.dtype)
    beta_grad = dy_f.sum(dim=0).reshape(beta.shape).to(beta.dtype) if beta is not None else None
    dy_g = dy_f * g_f
    c1 = dy_g.mean(dim=-1, keepdim=True)
    c2 = (dy_g * x_hat).mean(dim=-1, keepdim=True)
    x_grad = (rstd_f * (dy_g - c1 - x_hat * c2)).reshape(orig_shape).to(y_grad.dtype)
    return x_grad, gamma_grad, beta_grad


def install_xllm_extension_shim() -> bool:
    """Install pure-PyTorch xllm_extension shim with both forward and backward ops."""
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

    ops.group_rms_norm_fwd_affine = _group_rms_norm_fwd_affine
    ops.group_rms_norm_fwd = _group_rms_norm_fwd
    ops.group_rms_norm_bwd_affine = _group_rms_norm_bwd_affine
    ops.group_rms_norm_bwd = _group_rms_norm_bwd
    ops.group_layer_norm_fwd_affine = _group_layer_norm_fwd_affine
    ops.group_layer_norm_bwd_affine = _group_layer_norm_bwd_affine

    def _missing_op(name: str):
        def _stub(*_args, **_kwargs):
            raise NotImplementedError(f"xllm_extension.ops.{name} is not implemented in pure-PyTorch shim")
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
    """Autograd-compatible SDPA replacement for xllm flash_attention."""
    if cu_seqlens_q is not None or cu_seqlens_k is not None:
        raise NotImplementedError("SDPA shim supports non-varlen sequences only")
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
        is_causal = False
        if l_q > 1:
            attn_mask = torch.ones((l_q, l_k), dtype=torch.bool, device=qt.device).tril(diagonal=l_k - l_q)
    out = F.scaled_dot_product_attention(
        qt,
        kt,
        vt,
        attn_mask=attn_mask,
        dropout_p=dropout if training else 0.0,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=(n_heads != n_kv),
    )
    return out.transpose(1, 2).contiguous()


def bootstrap_xllm(xllm_loop_dir: str | Path, attention_backend: str = "sdpa") -> dict[str, Any]:
    """Add xllm-loop to sys.path and install forward+backward pure-PyTorch shims."""
    xllm_path = str(Path(xllm_loop_dir).resolve())
    if xllm_path not in sys.path:
        sys.path.insert(0, xllm_path)

    os.environ.setdefault("ENABLE_FLASH_ATTENTION_3", "false")
    os.environ.setdefault("ENABLE_FLASH_ATTENTION_4", "false")

    used_ext_shim = install_xllm_extension_shim()

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


def load_ifm_model_for_autograd(
    artifact_dir: str | Path,
    *,
    device: str = "cuda:0",
    attention_backend: str = "sdpa",
):
    """Load IFM Part I model and disable custom fused_block/fused_output so pure PyTorch autograd runs."""
    from xllm.paper_part1.native_inference import load_native_model

    model, tokenizer, model_config = load_native_model(artifact_dir, device=device)
    for layer in model.layers:
        if hasattr(layer, "fused_block"):
            layer.fused_block = False
        if attention_backend == "exact":
            layer.attention.causal_attention.backend = None
    if hasattr(model, "output") and hasattr(model.output, "fused_output_layer"):
        model.output.fused_output_layer = False
    model.eval()
    return model, tokenizer, model_config
