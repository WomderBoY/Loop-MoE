#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

cuda_build=${1:-cu130}
case "$cuda_build" in
  cu126|cu130) ;;
  *) echo 'Usage: bash scripts/setup_gpu.sh [cu130|cu126]' >&2; exit 2 ;;
esac

python -c 'import sys; assert sys.version_info[:2] == (3, 11), "Activate a Python 3.11 environment first"'
python -m pip install --upgrade 'pip==26.2.1'
python -m pip install "torch==2.13.0+$cuda_build" --index-url "https://download.pytorch.org/whl/$cuda_build"
python -m pip install -r requirements-gpu.txt
python -m pip check
