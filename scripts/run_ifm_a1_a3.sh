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

export XLLM_LOOP_DIR="${XLLM_LOOP_DIR:-$STORAGE_BASE/xllm-loop}"
export IFM_MODEL_DIR="${IFM_MODEL_DIR:-$STORAGE_BASE/hf_models}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$STORAGE_BASE/hf_cache}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

RUN_TAG="${ORX_RUN_ID:-manual-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-$STORAGE_BASE/loop-moe-runs/$RUN_TAG}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/ifm_a1_a3.json}"
mkdir -p "$OUT_DIR"

report() { printf '::orx:: %s\n' "$1"; }

echo "[run] ROOT_DIR=$ROOT_DIR"
echo "[run] CONFIG_PATH=$CONFIG_PATH"
echo "[run] OUT_DIR=$OUT_DIR"
echo "[run] GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo snapshot)"

report '{"phase":"setup","step":0,"of":1,"unit":"env"}'
bash scripts/setup_ifm_env.sh
report '{"step":1}'

report '{"phase":"download","step":0,"of":2,"unit":"models"}'
bash scripts/download_ifm_models.sh
report '{"step":2}'

STAGE="$("$PYTHON_BIN" -c "import json,sys; print(json.load(open(sys.argv[1])).get('stage','a1_a3'))" "$CONFIG_PATH")"
echo "[run] STAGE=$STAGE"

GPU_ARGS=()
if [[ -n "${IFM_GPU_INDEX:-}" ]]; then
  GPU_ARGS=(--gpu "$IFM_GPU_INDEX")
fi

if [[ "$STAGE" == "b2_gpu_fast" ]]; then
  TRACE_DIR="${IFM_TRACE_DIR:-$("$PYTHON_BIN" -c "import json,sys; print(json.load(open(sys.argv[1]))['stage_b2_fast']['trace_dir'])" "$CONFIG_PATH")}"
  echo "[run] TRACE_DIR=$TRACE_DIR"
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m loop_moe.ifm_b2_gpu_fast \
    --config "$CONFIG_PATH" \
    --out-dir "$OUT_DIR" \
    --xllm-loop-dir "$XLLM_LOOP_DIR" \
    --model-root "$IFM_MODEL_DIR" \
    --trace-dir "$TRACE_DIR" \
    "${GPU_ARGS[@]}"

  echo "========== REPORT ($OUT_DIR/report_b2_fast.md) =========="
  cat "$OUT_DIR/report_b2_fast.md"
  echo "========== END REPORT =========="
elif [[ "$STAGE" == "b1_b2" ]]; then
  TRACE_DIR="${IFM_TRACE_DIR:-$("$PYTHON_BIN" -c "import json,sys; print(json.load(open(sys.argv[1]))['stage_b']['trace_dir'])" "$CONFIG_PATH")}"
  echo "[run] TRACE_DIR=$TRACE_DIR"
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m loop_moe.ifm_b_driver \
    --config "$CONFIG_PATH" \
    --out-dir "$OUT_DIR" \
    --xllm-loop-dir "$XLLM_LOOP_DIR" \
    --model-root "$IFM_MODEL_DIR" \
    --trace-dir "$TRACE_DIR" \
    "${GPU_ARGS[@]}"

  echo "========== REPORT ($OUT_DIR/report_b1_b2.md) =========="
  cat "$OUT_DIR/report_b1_b2.md"
  echo "========== END REPORT =========="
else
  report '{"phase":"a1_collect"}'
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m loop_moe.ifm_a1_collect \
    --config "$CONFIG_PATH" \
    --out-dir "$OUT_DIR" \
    --xllm-loop-dir "$XLLM_LOOP_DIR" \
    --model-root "$IFM_MODEL_DIR" \
    --local-files-only \
    "${GPU_ARGS[@]}"

  PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m loop_moe.ifm_report \
    --config "$CONFIG_PATH" \
    --out-dir "$OUT_DIR"

  echo "========== REPORT ($OUT_DIR/report_a1_a3.md) =========="
  cat "$OUT_DIR/report_a1_a3.md"
  echo "========== END REPORT =========="
fi
echo "OUT_DIR=$OUT_DIR"
