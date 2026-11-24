#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/autodl_env.sh
export HF_HUB_OFFLINE=1
out_dir=${1:-runs/autodl-a800-4models}
if [[ $# -gt 0 ]]; then shift; fi
exec "$MOE_ENV_PREFIX/bin/python" -u scripts/resume_postprocess.py --out-dir "$out_dir" "$@"
