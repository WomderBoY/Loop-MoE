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
echo "[setup] XLLM_LOOP_DIR=$XLLM_LOOP_DIR"

# Install lightweight pure-Python packages needed by xllm-loop if missing
"$PYTHON_BIN" - <<'PY' || "$PYTHON_BIN" -m pip install -q --no-cache-dir \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  einops tiktoken==0.8.0 sentencepiece blobfile "typeguard>=4.5.0" fire pyarrow
import einops, tiktoken, sentencepiece, blobfile, typeguard, fire, pyarrow
print("[setup] All Python deps already present.")
PY

# Clone and pin official ifm-ai/xllm-loop
if [[ ! -d "$XLLM_LOOP_DIR/.git" ]]; then
  git clone --no-checkout https://github.com/ifm-ai/xllm-loop.git "$XLLM_LOOP_DIR"
fi
git -C "$XLLM_LOOP_DIR" fetch --depth 1 origin "$XLLM_COMMIT" 2>/dev/null || git -C "$XLLM_LOOP_DIR" fetch origin
git -C "$XLLM_LOOP_DIR" checkout -q "$XLLM_COMMIT"
echo "[setup] xllm-loop pinned at $(git -C "$XLLM_LOOP_DIR" rev-parse HEAD)"

# Quick import sanity check with pure-PyTorch shims (no GPU touched)
PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" - "$XLLM_LOOP_DIR" <<'PY'
import sys
from loop_moe.ifm_runtime import bootstrap_xllm
info = bootstrap_xllm(sys.argv[1], attention_backend="sdpa")
from xllm.paper_part1.native_inference import load_native_model
from xllm.models.looped import LoopedTransformer, Huginn
print("[setup] xllm import OK:", info)
PY
