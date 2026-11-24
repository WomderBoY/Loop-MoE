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

MODEL_DIR="${MODEL_DIR:-$STORAGE_BASE/hf_models}"
DATASET_DIR="${DATASET_DIR:-$STORAGE_BASE/datasets}"
ASSETS="${ASSETS:-ouro14b,datasets}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$STORAGE_BASE/hf_cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
mkdir -p "$MODEL_DIR" "$DATASET_DIR" "$HF_HOME"

echo "[download] HF_ENDPOINT=$HF_ENDPOINT"
echo "[download] ASSETS=$ASSETS"
echo "[download] MODEL_DIR=$MODEL_DIR DATASET_DIR=$DATASET_DIR"

cd "$ROOT_DIR"
PYTHONPATH="$ROOT_DIR" PYTHONUNBUFFERED=1 "$PYTHON_BIN" - "$ASSETS" "$MODEL_DIR" "$DATASET_DIR" <<'PY'
import json
import sys
from loop_research.data_assets import download_requested_assets

assets = [a.strip() for a in sys.argv[1].split(",") if a.strip()]
res = download_requested_assets(assets, sys.argv[2], sys.argv[3])
print("[download] Resolved:", json.dumps(res, indent=2))
PY
