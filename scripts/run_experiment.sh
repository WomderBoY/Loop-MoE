#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

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

export STORAGE_BASE
export XLLM_LOOP_DIR="${XLLM_LOOP_DIR:-$STORAGE_BASE/xllm-loop}"
export MODEL_DIR="${MODEL_DIR:-$STORAGE_BASE/hf_models}"
export DATASET_DIR="${DATASET_DIR:-$STORAGE_BASE/datasets}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$STORAGE_BASE/hf_cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

RUN_TAG="${ORX_RUN_ID:-manual-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-$STORAGE_BASE/loop-research-runs/$RUN_TAG}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/experiment.json}"
mkdir -p "$OUT_DIR" "$MODEL_DIR" "$DATASET_DIR" "$HF_HOME"

report() { printf '::orx:: %s\n' "$1"; }

IDEA="$("$PYTHON_BIN" -c "import json,sys; print(json.load(open(sys.argv[1]))['idea'])" "$CONFIG_PATH")"
ASSETS="$("$PYTHON_BIN" -c "import json,sys; print(','.join(json.load(open(sys.argv[1])).get('assets', [])))" "$CONFIG_PATH")"
NEED_XLLM="$("$PYTHON_BIN" -c "import json,sys; print(1 if json.load(open(sys.argv[1])).get('need_xllm', False) else 0)" "$CONFIG_PATH")"

echo "[run] ROOT_DIR=$ROOT_DIR"
echo "[run] CONFIG_PATH=$CONFIG_PATH"
echo "[run] IDEA=$IDEA ASSETS=$ASSETS NEED_XLLM=$NEED_XLLM"
echo "[run] OUT_DIR=$OUT_DIR"
echo "[run] GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo snapshot)"

report '{"phase":"setup","step":0,"of":1,"unit":"env"}'
NEED_XLLM="$NEED_XLLM" bash scripts/setup_env.sh
report '{"step":1}'

report '{"phase":"download","step":0,"of":1,"unit":"assets"}'
ASSETS="$ASSETS" bash scripts/download_assets.sh
report '{"step":1}'

if [[ "$IDEA" == "idea1_kv_depth" ]]; then
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m loop_research.idea1_kv_depth \
    --config "$CONFIG_PATH" \
    --model-dir "$MODEL_DIR" \
    --dataset-dir "$DATASET_DIR" \
    --out-dir "$OUT_DIR"
elif [[ "$IDEA" == "idea2_expert_unbinding" ]]; then
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m loop_research.idea2_expert_unbinding \
    --config "$CONFIG_PATH" \
    --model-dir "$MODEL_DIR" \
    --dataset-dir "$DATASET_DIR" \
    --xllm-loop-dir "$XLLM_LOOP_DIR" \
    --out-dir "$OUT_DIR"
elif [[ "$IDEA" == "new_dir1_ftrs" ]]; then
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m loop_research.ftrs_stepping \
    --config "$CONFIG_PATH" \
    --model-dir "$MODEL_DIR" \
    --dataset-dir "$DATASET_DIR" \
    --out-dir "$OUT_DIR"
elif [[ "$IDEA" == "route1_tcre_ragged" ]]; then
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m loop_research.route1_tcre_ragged \
    --config "$CONFIG_PATH" \
    --model-dir "$MODEL_DIR" \
    --dataset-dir "$DATASET_DIR" \
    --out-dir "$OUT_DIR"
else
  echo "[run] Unknown idea in $CONFIG_PATH: $IDEA" >&2
  exit 1
fi

echo "OUT_DIR=$OUT_DIR"
