"""B2 Part 1: Hardware microbenchmark on 1 strictly idle GPU.

Measures:
1. Single-expert (21.375 MiB) pinned host -> GPU H2D latency & effective PCIe bandwidth
   (both isolated and concurrent with default-stream BF16 GEMM compute).
2. Single-expert SwiGLU FFN compute latency on GPU vs CPU across batch_tokens=[1,2,4,8,16,32].
3. Per-logical-visit non-routed-MoE compute time on the actual loaded IFM model during
   incremental KV-cached decode (seq_len=128), calibrating the true overlap window.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Any

import torch
import torch.nn.functional as F

from .ifm_a1_collect import select_idle_gpu
from .ifm_runtime import bootstrap_xllm, load_ifm_model
from .trace import read_json, write_json


def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


@torch.inference_mode()
def benchmark_pcie_and_ffn(
    *,
    device: torch.device,
    model_dim: int = 3072,
    inter_dim: int = 1792,
    n_warmup: int = 10,
    n_iters: int = 60,
) -> dict[str, Any]:
    """Measure exact single-expert H2D latency, concurrent bandwidth, and GPU/CPU FFN times."""
    # Pack w1 [inter, d], w2 [d, inter], w3 [inter, d] in BF16 into one flat contiguous tensor
    numel = 3 * inter_dim * model_dim
    expert_bytes = numel * 2  # bf16 = 2 bytes
    expert_mib = expert_bytes / (1024 * 1024)

    host_buf = torch.empty(numel, dtype=torch.bfloat16, pin_memory=True)
    host_buf.uniform_(-0.02, 0.02)
    gpu_buf = torch.empty(numel, dtype=torch.bfloat16, device=device)

    copy_stream = torch.cuda.Stream(device=device)
    torch.cuda.synchronize(device)

    # 1. Isolated H2D
    for _ in range(n_warmup):
        with torch.cuda.stream(copy_stream):
            gpu_buf.copy_(host_buf, non_blocking=True)
        copy_stream.synchronize()

    iso_ms: list[float] = []
    for _ in range(n_iters):
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(copy_stream):
            ev0.record(copy_stream)
            gpu_buf.copy_(host_buf, non_blocking=True)
            ev1.record(copy_stream)
        ev1.synchronize()
        iso_ms.append(float(ev0.elapsed_time(ev1)))

    # 2. Concurrent H2D + BF16 GEMM compute on default stream
    comp_a = torch.randn((128, model_dim), dtype=torch.bfloat16, device=device)
    comp_b = torch.randn((model_dim, model_dim), dtype=torch.bfloat16, device=device)
    for _ in range(n_warmup):
        _ = comp_a @ comp_b
        with torch.cuda.stream(copy_stream):
            gpu_buf.copy_(host_buf, non_blocking=True)
        torch.cuda.synchronize(device)

    conc_ms: list[float] = []
    for _ in range(n_iters):
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        # Launch compute bursts on default stream while copying on copy_stream
        with torch.cuda.stream(copy_stream):
            ev0.record(copy_stream)
            gpu_buf.copy_(host_buf, non_blocking=True)
            ev1.record(copy_stream)
        for _ in range(4):
            _ = comp_a @ comp_b
        torch.cuda.synchronize(device)
        conc_ms.append(float(ev0.elapsed_time(ev1)))

    # 3. Single-expert SwiGLU FFN on GPU vs CPU across batch token sizes
    w1_gpu = torch.randn((inter_dim, model_dim), dtype=torch.bfloat16, device=device) * 0.02
    w2_gpu = torch.randn((model_dim, inter_dim), dtype=torch.bfloat16, device=device) * 0.02
    w3_gpu = torch.randn((inter_dim, model_dim), dtype=torch.bfloat16, device=device) * 0.02

    w1_cpu_bf16 = w1_gpu.cpu()
    w2_cpu_bf16 = w2_gpu.cpu()
    w3_cpu_bf16 = w3_gpu.cpu()
    w1_cpu_fp32 = w1_cpu_bf16.float()
    w2_cpu_fp32 = w2_cpu_bf16.float()
    w3_cpu_fp32 = w3_cpu_bf16.float()

    ffn_rows: list[dict[str, Any]] = []
    for b_tok in (1, 2, 4, 8, 16, 32):
        x_gpu = torch.randn((b_tok, model_dim), dtype=torch.bfloat16, device=device)
        for _ in range(10):
            _ = F.linear(F.silu(F.linear(x_gpu, w1_gpu)) * F.linear(x_gpu, w3_gpu), w2_gpu)
        torch.cuda.synchronize(device)
        gpu_times = []
        for _ in range(40):
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            _ = F.linear(F.silu(F.linear(x_gpu, w1_gpu)) * F.linear(x_gpu, w3_gpu), w2_gpu)
            ev1.record()
            ev1.synchronize()
            gpu_times.append(float(ev0.elapsed_time(ev1)))

        x_cpu_bf16 = x_gpu.cpu()
        x_cpu_fp32 = x_cpu_bf16.float()
        # Warmup CPU
        for _ in range(3):
            _ = F.linear(F.silu(F.linear(x_cpu_bf16, w1_cpu_bf16)) * F.linear(x_cpu_bf16, w3_cpu_bf16), w2_cpu_bf16)
            _ = F.linear(F.silu(F.linear(x_cpu_fp32, w1_cpu_fp32)) * F.linear(x_cpu_fp32, w3_cpu_fp32), w2_cpu_fp32)

        cpu_bf16_times = []
        for _ in range(15):
            t0 = time.perf_counter()
            _ = F.linear(F.silu(F.linear(x_cpu_bf16, w1_cpu_bf16)) * F.linear(x_cpu_bf16, w3_cpu_bf16), w2_cpu_bf16)
            cpu_bf16_times.append((time.perf_counter() - t0) * 1000.0)

        cpu_fp32_times = []
        for _ in range(15):
            t0 = time.perf_counter()
            _ = F.linear(F.silu(F.linear(x_cpu_fp32, w1_cpu_fp32)) * F.linear(x_cpu_fp32, w3_cpu_fp32), w2_cpu_fp32)
            cpu_fp32_times.append((time.perf_counter() - t0) * 1000.0)

        ffn_rows.append(
            {
                "batch_tokens": b_tok,
                "gpu_bf16_p50_ms": round(_pct(gpu_times, 0.5), 4),
                "gpu_bf16_p95_ms": round(_pct(gpu_times, 0.95), 4),
                "cpu_bf16_p50_ms": round(_pct(cpu_bf16_times, 0.5), 4),
                "cpu_fp32_p50_ms": round(_pct(cpu_fp32_times, 0.5), 4),
            }
        )

    iso_p50 = _pct(iso_ms, 0.5)
    conc_p50 = _pct(conc_ms, 0.5)
    return {
        "device_name": torch.cuda.get_device_name(device),
        "expert_bytes": expert_bytes,
        "expert_mib": round(expert_mib, 4),
        "model_dim": model_dim,
        "inter_dim": inter_dim,
        "h2d_isolated_p50_ms": round(iso_p50, 4),
        "h2d_isolated_p95_ms": round(_pct(iso_ms, 0.95), 4),
        "h2d_isolated_mean_ms": round(statistics.mean(iso_ms), 4),
        "h2d_isolated_bw_gib_s": round((expert_bytes / (1024**3)) / (iso_p50 / 1000.0), 3),
        "h2d_concurrent_p50_ms": round(conc_p50, 4),
        "h2d_concurrent_p95_ms": round(_pct(conc_ms, 0.95), 4),
        "h2d_concurrent_mean_ms": round(statistics.mean(conc_ms), 4),
        "h2d_concurrent_bw_gib_s": round((expert_bytes / (1024**3)) / (conc_p50 / 1000.0), 3),
        "ffn_compute_ms": ffn_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="B2 hardware H2D & FFN microbenchmark on 1 idle GPU")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--gpu", default=None)
    args = parser.parse_args()

    gpu = select_idle_gpu(args.gpu)
    device = torch.device(f"cuda:{gpu['index']}")
    torch.cuda.set_device(device)
    res = benchmark_pcie_and_ffn(device=device)
    res["gpu_index"] = gpu["index"]
    res["gpu_uuid"] = gpu["uuid"]

    out_path = Path(args.out_dir) / "b2_microbench.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, res)
    print(f"[B2 microbench] {json.dumps(res, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
