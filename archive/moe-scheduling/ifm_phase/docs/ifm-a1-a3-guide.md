# IFM Part I MoE（Ouro-500B / Huginn-500B）A1–A3 实验运行指南

本指南对应实验清单中的 **A1（生成与轨迹采集）→ A2（相邻层/相邻 loop/相邻 token 预测分解）→ A3（静态驻留到 Belady 远见同预算回放）**。

## 1. 环境配置与模型下载（GPU 服务器，如 `pku_pro5000`）

默认优先使用服务器上已有的 Python 3.11 + PyTorch CUDA 环境（`~/loop-moe-env/bin/python`），并将 `xllm-loop` 源码、Hugging Face 权重与实验输出存放在大容量共享盘 `/data1/shared/$USER/`（可通过 `XLLM_LOOP_DIR` / `IFM_MODEL_DIR` / `OUT_DIR` 覆盖）。国内网络默认启用 `HF_ENDPOINT=https://hf-mirror.com` 与清华 PyPI 镜像。

```bash
# 1) 配置环境并固定官方 ifm-ai/xllm-loop 到 commit 755c7b3a91081fe0f83500eabd53bd2490b8b745
bash scripts/setup_ifm_env.sh

# 2) 下载并校验 IFM/LoopedLM-P1-moe-ouro-500b 与 IFM/LoopedLM-P1-moe-huginn-500b（各约 16.4 GiB BF16）
bash scripts/download_ifm_models.sh
```

> 说明：在 RTX PRO 5000（Blackwell SM120 / CUDA 13.0）上若未安装 Hopper 专用 FA3 与 C++ `xllm_extension`，`loop_moe/ifm_runtime.py` 自动启用数值等价的纯 PyTorch `GroupRMSNorm`（FP32 累加 + BF16 输出，严格对齐 `group_rms_norm_kernel.cu`）与 PyTorch SDPA 注意力，并在 A1 开头自动与官方 `causal_attn_backend=None`（精确 FP32-softmax）逐位置核对最大 logit 误差与 112 层路由完全一致率。

## 2. 一键运行 A1 → A2 → A3

脚本启动前通过 `nvidia-smi` 严格检测空闲 GPU（要求 0 个外部进程且显存占用 ≤ 32 MiB），绝不占用他人显卡；也可显式指定 `IFM_GPU_INDEX=5`：

```bash
bash scripts/run_ifm_a1_a3.sh
```

## 3. 主要产物

在输出目录 `OUT_DIR` 下生成：
- `<Model>/trace_freegen.jsonl.gz`、`<Model>/trace_fixed.jsonl.gz`：A1 自由生成与固定文本对照路由轨迹（含 top-k、top-6 候选与原始 logit、margin、router 输入 cosine、logit $L_\infty$ 漂移）。
- `<Model>/generation_samples.md`、`<Model>/a1_quality_summary.json`、`<Model>/requests_manifest_*.json`：生成样例、GSM8K/MBPP 客观题校验、重复率与 fit/val/eval 划分。
- `<Model>/a2_{freegen,fixed}/`：`a2_summary.csv`、`a2_by_layer_loop.csv`、`a2_retain_vs_replace.csv`、`a2_geometry_margin.csv`、`a2_requests.csv`。
- `<Model>/a3_{freegen,fixed}/`：`a3_summary.csv`、`a3_paired_summary.csv`、`a3_requests.csv`、`a3_paired_requests.csv`（含 2–14.61 GiB 预算、Huginn 循环区固定驻留策略与 Belady 离线远见）。
- `report_a1_a3.md`：中文综合汇总报告。
