#!/usr/bin/env bash
# Inference only on the training server. No device initialization.
set -euo pipefail
ROOT="${PI05_WORKSPACE:-$HOME/output/personal/k1deler/fold_towel_pi05_20261102}"
PY="$HOME/output/envs/yam-openpi/bin/python"
GPU="${1:?Usage: bash scripts/serve_model.sh GPU_INDEX}"
[[ "$GPU" =~ ^[0-9]+$ ]] || exit 2
export CUDA_VISIBLE_DEVICES="$GPU" JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.34
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/code:$ROOT/vendor/openpi/src:$ROOT/vendor/openpi/packages/openpi-client/src"
"$PY" - "$GPU" <<'PY'
import subprocess,sys
index=int(sys.argv[1])
rows=[[int(x.strip()) for x in line.split(',')] for line in subprocess.check_output(
    ['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],text=True).splitlines()]
row=next(r for r in rows if r[0]==index)
if row[2]-row[1]<40960:raise SystemExit('Need 40GiB free VRAM; no foreign process is stopped')
PY
cd "$ROOT/code"
exec "$PY" -u -m deploy_pi05.service --workspace "$ROOT" --port 8778
