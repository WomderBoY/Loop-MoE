#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

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
# Default: gracefully exit after downloading assets if all shared GPUs are occupied
export EXIT_IF_NO_IDLE_GPU="${EXIT_IF_NO_IDLE_GPU:-1}"

RUN_TAG="${ORX_RUN_ID:-manual-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-$STORAGE_BASE/loop-research-runs/$RUN_TAG}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/experiment.json}"
mkdir -p "$OUT_DIR" "$MODEL_DIR" "$DATASET_DIR" "$HF_HOME"

report() { printf '::orx:: %s\n' "$1"; }

# Parse config with system python3 before selecting the experiment-specific venv
SYS_PY="$(command -v python3)"
IDEA="$("$SYS_PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['idea'])" "$CONFIG_PATH")"
ASSETS="$("$SYS_PY" -c "import json,sys; print(','.join(json.load(open(sys.argv[1])).get('assets', [])))" "$CONFIG_PATH")"
NEED_XLLM="$("$SYS_PY" -c "import json,sys; print(1 if json.load(open(sys.argv[1])).get('need_xllm', False) else 0)" "$CONFIG_PATH")"

echo "[run] ROOT_DIR=$ROOT_DIR"
echo "[run] CONFIG_PATH=$CONFIG_PATH"
echo "[run] IDEA=$IDEA ASSETS=$ASSETS NEED_XLLM=$NEED_XLLM EXIT_IF_NO_IDLE_GPU=$EXIT_IF_NO_IDLE_GPU"
echo "[run] OUT_DIR=$OUT_DIR"
echo "[run] GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo snapshot)"

report '{"phase":"setup","step":0,"of":1,"unit":"env"}'
if [[ "$IDEA" == "r4b_inkling2_loopsched" ]]; then
  INKLING_ENV_DIR="${INKLING_ENV_DIR:-$HOME/inkling-vllm-env}"
  bash scripts/setup_inkling_env.sh
  PYTHON_BIN="${PYTHON_BIN:-$INKLING_ENV_DIR/bin/python}"
  export PYTHON_BIN
else
  PYTHON_BIN="${PYTHON_BIN:-$HOME/loop-moe-env/bin/python}"
  if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$SYS_PY"
  fi
  export PYTHON_BIN
  NEED_XLLM="$NEED_XLLM" bash scripts/setup_env.sh
fi
report '{"step":1}'

report '{"phase":"download","step":0,"of":1,"unit":"assets"}'
ASSETS="$ASSETS" PYTHON_BIN="$PYTHON_BIN" bash scripts/download_assets.sh
report '{"step":1}'

if [[ "$IDEA" == "r2a_loopsched_moe" ]]; then
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m unittest tests/test_r2a_loopsched.py -v
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m loop_research.r2a_loopsched_moe \
    --config "$CONFIG_PATH" \
    --model-dir "$MODEL_DIR" \
    --dataset-dir "$DATASET_DIR" \
    --xllm-loop-dir "$XLLM_LOOP_DIR" \
    --out-dir "$OUT_DIR"
elif [[ "$IDEA" == "r3_loopsched_v2" ]]; then
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m unittest tests/test_r3_loopsched_v2.py -v
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m loop_research.r3_loopsched_v2 \
    --config "$CONFIG_PATH" \
    --model-dir "$MODEL_DIR" \
    --dataset-dir "$DATASET_DIR" \
    --xllm-loop-dir "$XLLM_LOOP_DIR" \
    --out-dir "$OUT_DIR"
elif [[ "$IDEA" == "r4a_ifm_fast_and_swaps" ]]; then
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m unittest tests/test_r4_fast_and_inkling.py -v
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m loop_research.r4a_ifm_fast_and_swaps \
    --config "$CONFIG_PATH" \
    --model-dir "$MODEL_DIR" \
    --dataset-dir "$DATASET_DIR" \
    --xllm-loop-dir "$XLLM_LOOP_DIR" \
    --out-dir "$OUT_DIR"
elif [[ "$IDEA" == "r4b_inkling2_loopsched" ]]; then
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m unittest tests/test_r4_fast_and_inkling.py -v
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m loop_research.r4b_inkling2_loopsched \
    --config "$CONFIG_PATH" \
    --model-dir "$MODEL_DIR" \
    --dataset-dir "$DATASET_DIR" \
    --out-dir "$OUT_DIR"
else
  echo "[run] Unknown idea in $CONFIG_PATH: $IDEA" >&2
  exit 1
fi

echo "OUT_DIR=$OUT_DIR"
