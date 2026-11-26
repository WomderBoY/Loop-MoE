#!/usr/bin/env bash
# Configure an isolated Python venv for thinkingmachines/Inkling-2-Mini (vLLM nightly / >=0.35.0 + Blackwell SM120)
# Never modifies the primary IFM $HOME/loop-moe-env environment!
set -euo pipefail

ENV_DIR="${INKLING_ENV_DIR:-$HOME/inkling-vllm-env}"
if [[ -n "${BASE_PYTHON:-}" ]]; then
  :
elif [[ -x "$HOME/loop-moe-env/bin/python" ]]; then
  BASE_PYTHON="$HOME/loop-moe-env/bin/python"
else
  BASE_PYTHON="$(command -v python3.12 || command -v python3.11 || command -v python3)"
fi
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
VLLM_NIGHTLY_INDEX="${VLLM_NIGHTLY_INDEX:-https://wheels.vllm.ai/nightly}"

echo "[inkling-setup] Target isolated venv: $ENV_DIR (base=$BASE_PYTHON)"
if [[ -x "$ENV_DIR/bin/python" ]]; then
  if "$ENV_DIR/bin/python" -c "
import vllm
from vllm.model_executor.models.registry import ModelRegistry
assert any('Inkling2Loop' in a for a in ModelRegistry.get_supported_archs())
from vllm.models.inkling2.nvidia import loop, moe
print('[inkling-setup] Cached env OK: vllm=' + str(vllm.__version__))
" 2>/dev/null; then
    exit 0
  fi
fi

if [[ ! -x "$ENV_DIR/bin/python" || ! -x "$ENV_DIR/bin/pip" ]]; then
  rm -rf "$ENV_DIR"
  echo "[inkling-setup] Creating isolated venv at $ENV_DIR ..."
  if ! "$BASE_PYTHON" -m venv "$ENV_DIR"; then
    echo "[inkling-setup] ensurepip unavailable; creating --without-pip venv and bootstrapping pip..."
    rm -rf "$ENV_DIR"
    "$BASE_PYTHON" -m venv --without-pip "$ENV_DIR"
    curl -sL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    "$ENV_DIR/bin/python" /tmp/get-pip.py -i "$PIP_INDEX"
  fi
fi

"$ENV_DIR/bin/pip" install -U pip wheel uv -i "$PIP_INDEX"

echo "[inkling-setup] Installing vLLM nightly (with Inkling2LoopForConditionalGeneration + SM120 NVFP4/MXFP8 kernels)..."
# Use uv pip for 10x faster parallel dependency resolution & wheel download
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-4}"
UV_BIN="$ENV_DIR/bin/uv"
if [[ -x "$UV_BIN" ]]; then
  if ! "$UV_BIN" pip install --python "$ENV_DIR/bin/python" \
      "vllm>=0.35.0" "transformers>=5.24.0" huggingface_hub pyarrow numpy safetensors nvidia-ml-py \
      --index-url "$PIP_INDEX" 2>/dev/null; then
    echo "[inkling-setup] Stable mirror <0.35.0; installing vLLM nightly via uv..."
    "$UV_BIN" pip install --python "$ENV_DIR/bin/python" --prerelease=allow \
      --index-strategy unsafe-best-match \
      vllm "transformers>=5.24.0" huggingface_hub pyarrow numpy safetensors nvidia-ml-py \
      --index-url "$PIP_INDEX" \
      --extra-index-url "$VLLM_NIGHTLY_INDEX" \
      --extra-index-url "https://download.pytorch.org/whl/cu130"
  fi
else
  if ! "$ENV_DIR/bin/pip" install -U "vllm>=0.35.0" -i "$PIP_INDEX" 2>/dev/null; then
    "$ENV_DIR/bin/pip" install -U --pre vllm \
      -i "$PIP_INDEX" \
      --extra-index-url "$VLLM_NIGHTLY_INDEX" \
      --extra-index-url "https://download.pytorch.org/whl/cu130"
  fi
  "$ENV_DIR/bin/pip" install -U -i "$PIP_INDEX" \
    "transformers>=5.24.0" huggingface_hub pyarrow numpy safetensors nvidia-ml-py
fi

echo "[inkling-setup] Verifying Inkling-2-Mini vLLM registration..."
"$ENV_DIR/bin/python" - <<'PY'
import json, torch, transformers, vllm
from vllm.model_executor.models.registry import ModelRegistry
archs = [a for a in ModelRegistry.get_supported_archs() if "Inkling2" in a]
from vllm.models.inkling2.nvidia import loop, moe
print(json.dumps({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "vllm": vllm.__version__,
    "inkling2_archs": archs,
}, indent=2))
assert any("Inkling2Loop" in a for a in archs), f"Inkling2Loop missing: {archs}"
PY
echo "[inkling-setup] Environment ready at $ENV_DIR"
