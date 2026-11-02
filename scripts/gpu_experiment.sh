#!/usr/bin/env bash
# Explicit GPU experiment only. Example: bash scripts/gpu_experiment.sh smoke 1
set -euo pipefail
[[ $# == 2 && ( "$1" == smoke || "$1" == train ) ]] || { echo 'Usage: bash scripts/gpu_experiment.sh smoke|train GPU_INDEX'; exit 2; }
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${PI05_WORKSPACE:-$HOME/output/personal/k1deler/fold_towel_pi05_20261102}"
ACT="${ACT_WORKSPACE:-$HOME/output/personal/k1deler/fold_towel_20261101}"
PY="${PI05_PYTHON:-$HOME/output/envs/yam-openpi/bin/python}"
OPENPI="$WORKSPACE/vendor/openpi"
export PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export PYTHONPATH="$CODE:$OPENPI/src:$OPENPI/packages/openpi-client/src"
cd "$CODE"
exec "$PY" -u -m fold_pi05.train --mode "$1" --gpu "$2" --execute \
 --data "$ACT/converted" --workspace "$WORKSPACE" --settings "$CODE/configs/pi05_lora.json" \
 --openpi-root "$OPENPI" --base-params "$HOME/output/data/pretrained/pi05_base/params"
