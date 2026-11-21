#!/usr/bin/env bash
# Source this file from the repository root or another deployment script.
export MOE_DATA_ROOT=${MOE_DATA_ROOT:-/root/autodl-tmp}
export MOE_ENV_PREFIX=${MOE_ENV_PREFIX:-/root/miniconda3/envs/loop-moe}
export PATH="$MOE_ENV_PREFIX/bin:$PATH"
export HF_HOME=${HF_HOME:-$MOE_DATA_ROOT/huggingface}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=30
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
