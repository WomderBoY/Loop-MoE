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

IFM_MODEL_DIR="${IFM_MODEL_DIR:-$STORAGE_BASE/hf_models}"
XLLM_LOOP_DIR="${XLLM_LOOP_DIR:-$STORAGE_BASE/xllm-loop}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/ifm_a1_a3.json}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HOME="${HF_HOME:-$STORAGE_BASE/hf_cache}"
mkdir -p "$IFM_MODEL_DIR" "$HF_HOME"

echo "[download] HF_ENDPOINT=$HF_ENDPOINT"
echo "[download] IFM_MODEL_DIR=$IFM_MODEL_DIR"

cd "$ROOT_DIR"
PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m loop_moe.ifm_a1_collect \
  --config "$CONFIG_PATH" \
  --out-dir "$STORAGE_BASE/tmp_download" \
  --xllm-loop-dir "$XLLM_LOOP_DIR" \
  --model-root "$IFM_MODEL_DIR" \
  --download-only
