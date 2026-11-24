#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

config=${MOE_CONFIG:-${FOIL_CONFIG:-configs/foil_pilot.json}}
python -m loop_moe.collect --config "$config" "$@" --download-only
