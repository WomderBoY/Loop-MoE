#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo 'Usage: bash scripts/run_residency.sh OUT_DIR TRACE [TRACE ...]' >&2
  echo 'Optional: PYTHON=python3 RESIDENCY_CONFIG=configs/residency.json RESIDENCY_SPLIT=/path/to/split.json' >&2
  exit 2
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
out_dir=$1
shift
config=${RESIDENCY_CONFIG:-configs/residency.json}
python_bin=${PYTHON:-python3}
extra=()
if [[ -n ${RESIDENCY_SPLIT:-} ]]; then
  extra+=(--split "$RESIDENCY_SPLIT")
fi
for trace in "$@"; do
  model=$(basename "$(dirname "$trace")")
  if [[ -e "$out_dir/$model/residency_config.json" ]]; then
    echo "Refusing to overwrite completed results: $out_dir/$model" >&2
    exit 2
  fi
  "$python_bin" -u -m loop_moe.residency "$trace" --config "$config" \
    --out-dir "$out_dir/$model" "${extra[@]}"
done
