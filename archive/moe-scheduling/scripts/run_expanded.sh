#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo 'Usage: PYTHON=~/loop-moe-env/bin/python bash scripts/run_expanded.sh OUT_DIR --gpus 5 7' >&2
  echo 'Preflight: bash scripts/run_expanded.sh OUT_DIR --preflight-only' >&2
  exit 2
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
out_dir=$1
shift
exec "${PYTHON:-python3}" -u -m loop_moe.expanded --out-dir "$out_dir" "$@"
