#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo 'Usage: bash scripts/run_pilot.sh OUT_DIR [collector options]' >&2
  exit 2
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."
out_dir=$1
shift
config=${FOIL_CONFIG:-configs/foil_pilot.json}
python -m loop_moe.collect --config "$config" "$@" --out-dir "$out_dir"
for trace in "$out_dir"/*/trace.jsonl.gz; do
  result_dir=${trace%/*}
  python -m loop_moe.analyze "$trace" --out-dir "$result_dir/analysis"
  python -m loop_moe.replay "$trace" --config "$config" --out-dir "$result_dir/replay"
done
