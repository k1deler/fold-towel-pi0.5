#!/usr/bin/env bash
# CPU-only actual-data preparation. No model optimizer, GPU allocation, or robot I/O.
set -euo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${PI05_WORKSPACE:-$HOME/output/personal/k1deler/fold_towel_pi05_20261102}"
ACT="${ACT_WORKSPACE:-$HOME/output/personal/k1deler/fold_towel_20261101}"
PY="${PI05_PYTHON:-$HOME/output/envs/yam-openpi/bin/python}"
OPENPI="$WORKSPACE/vendor/openpi"
BASE="$HOME/output/data/pretrained/pi05_base/params"
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export PYTHONPATH="$CODE:$OPENPI/src:$OPENPI/packages/openpi-client/src"
cd "$CODE"
"$PY" -m fold_pi05.prepare --data "$ACT/converted" --workspace "$WORKSPACE" \
 --settings "$CODE/configs/pi05_lora.json" --act-audit "$ACT/reports/converted_audit.json" \
 --act-config "$ACT/runs/act_delta_seed42/config.yaml" --openpi-root "$OPENPI" --base-params "$BASE"
"$PY" -m fold_pi05.preflight --data "$ACT/converted" --workspace "$WORKSPACE" \
 --settings "$CODE/configs/pi05_lora.json" --openpi-root "$OPENPI" --base-params "$BASE"
