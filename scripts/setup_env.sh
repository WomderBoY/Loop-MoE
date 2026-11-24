#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$HOME/loop-moe-env/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

if [[ -d /data1/shared && -w /data1/shared ]]; then
  STORAGE_BASE="/data1/shared/${USER}"
else
  STORAGE_BASE="$HOME/ifm_storage"
fi
mkdir -p "$STORAGE_BASE"

XLLM_LOOP_DIR="${XLLM_LOOP_DIR:-$STORAGE_BASE/xllm-loop}"
XLLM_COMMIT="${XLLM_COMMIT:-755c7b3a91081fe0f83500eabd53bd2490b8b745}"

echo "[setup] PYTHON_BIN=$PYTHON_BIN"
echo "[setup] STORAGE_BASE=$STORAGE_BASE"

if ! "$PYTHON_BIN" -c 'import torch, transformers, huggingface_hub, pyarrow, numpy, einops, safetensors' 2>/dev/null; then
  echo "[setup] Installing missing lightweight Python deps via Tsinghua mirror..."
  "$PYTHON_BIN" -m pip install -q --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    transformers huggingface_hub pyarrow numpy einops safetensors \
    tiktoken==0.8.0 sentencepiece blobfile "typeguard>=4.5.0" fire submitit nvidia-ml-py
fi

# Ensure xllm-loop exists when needed for IFM MoE experiments
if [[ "${NEED_XLLM:-0}" == "1" ]]; then
  if [[ ! -d "$XLLM_LOOP_DIR/.git" ]]; then
    git clone --no-checkout https://github.com/ifm-ai/xllm-loop.git "$XLLM_LOOP_DIR"
  fi
  if [[ "$(git -C "$XLLM_LOOP_DIR" rev-parse HEAD 2>/dev/null || true)" != "$XLLM_COMMIT" ]]; then
    if ! git -C "$XLLM_LOOP_DIR" cat-file -e "${XLLM_COMMIT}^{commit}" 2>/dev/null; then
      git -C "$XLLM_LOOP_DIR" fetch --depth 1 origin "$XLLM_COMMIT" 2>/dev/null || git -C "$XLLM_LOOP_DIR" fetch origin
    fi
    git -C "$XLLM_LOOP_DIR" checkout -q "$XLLM_COMMIT"
  fi
  echo "[setup] xllm-loop pinned at $(git -C "$XLLM_LOOP_DIR" rev-parse HEAD)"
fi

echo "[setup] Environment check OK."
