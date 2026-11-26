#!/usr/bin/env bash
# Configure an isolated Python venv for thinkingmachines/Inkling-2-Mini (vLLM >=0.35.0 / transformers)
# Never modifies the primary IFM $HOME/loop-moe-env environment!
set -euo pipefail

ENV_DIR="${INKLING_ENV_DIR:-$HOME/inkling-vllm-env}"
BASE_PYTHON="${BASE_PYTHON:-$(command -v python3.12 || command -v python3.11 || command -v python3)}"
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

echo "[inkling-setup] Creating isolated venv at $ENV_DIR using $BASE_PYTHON"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  "$BASE_PYTHON" -m venv "$ENV_DIR"
fi

"$ENV_DIR/bin/pip" install -U pip wheel -i "$PIP_INDEX"

echo "[inkling-setup] Installing latest vLLM (supporting inkling2_loop + Blackwell NVFP4)..."
if ! "$ENV_DIR/bin/pip" install -U "vllm>=0.35.0" -i "$PIP_INDEX"; then
  echo "[inkling-setup] Stable PyPI mirror did not have vllm>=0.35.0; trying vLLM nightly wheel index..."
  "$ENV_DIR/bin/pip" install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
fi

"$ENV_DIR/bin/pip" install -U -i "$PIP_INDEX" \
  transformers huggingface_hub pyarrow numpy safetensors nvidia-ml-py

echo "[inkling-setup] Verifying Inkling-2-Mini runtime registration..."
"$ENV_DIR/bin/python" - <<'PY'
import json
from loop_research.r4b_inkling2_loopsched import detect_inkling_runtime
print(json.dumps(detect_inkling_runtime(), indent=2))
PY
echo "[inkling-setup] Done. Launch R4-B with:"
echo "  PYTHON_BIN=$ENV_DIR/bin/python CONFIG_PATH=configs/r4b_inkling_config.json bash scripts/run_experiment.sh"
