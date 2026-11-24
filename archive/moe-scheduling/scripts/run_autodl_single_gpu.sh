#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/autodl_env.sh
out_dir=${1:-runs/autodl-a800-4models}

# This command is used only after the instance is booted with an A800 GPU.
# Downloading and tokenizer validation are separate CPU-only preparation steps.
gpu_id=$("$MOE_ENV_PREFIX/bin/python" - "${GPU_ID:-}" <<'PY'
import subprocess
import sys

rows = subprocess.run([
    "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free", "--format=csv,noheader,nounits",
], check=True, text=True, capture_output=True).stdout.splitlines()
matches = ([row.split(",") for row in rows
            if sys.argv[1] in [value.strip() for value in row.split(",")[:2]]]
           if sys.argv[1] else [row.split(",") for row in rows])
if len(matches) != 1:
    raise SystemExit("Selected GPU is missing or ambiguous; check GPU_ID and nvidia-smi")
index, uuid, name, total, free = [value.strip() for value in matches[0]]
if "A800" not in name or int(total) < 75 * 1024 or int(free) < 60 * 1024:
    raise SystemExit("Four-model launcher requires an A800 80GB with at least 60 GiB currently free")
print(f"Four-model launch on GPU {index}: {name}, free={int(free) / 1024:.1f} GiB", file=sys.stderr, flush=True)
print(uuid)
PY
)

export HF_HUB_OFFLINE=1
exec "$MOE_ENV_PREFIX/bin/python" -u -m loop_moe.expanded \
  --out-dir "$out_dir" --gpus "$gpu_id" --parallel-models --exclusive-gpu
