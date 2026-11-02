#!/usr/bin/env bash
# Snapshot a pinned public source commit from the already available checkout.
# No environment installation, pretrained-parameter download, or GPU computation.
set -euo pipefail
WORKSPACE="${PI05_WORKSPACE:-$HOME/output/personal/k1deler/fold_towel_pi05_20261102}"
SOURCE="${OPENPI_SOURCE:-$HOME/output/code/yam_vla_training/openpi}"
DEST="$WORKSPACE/vendor/openpi"
REV=215abfb217dbac7d5f1273282331b9b1866c0479
[[ -d "$SOURCE/.git" ]] || { echo 'Existing pinned OpenPI Git checkout missing; set OPENPI_SOURCE'; exit 1; }
[[ -x "${PI05_PYTHON:-$HOME/output/envs/yam-openpi/bin/python}" ]] || { echo 'Existing OpenPI Python environment missing; set PI05_PYTHON'; exit 1; }
git -C "$SOURCE" cat-file -e "$REV^{commit}"
if [[ ! -e "$DEST" ]]; then
    mkdir -p "$DEST"
    git -C "$SOURCE" archive "$REV" | tar -x -C "$DEST"
    printf '%s\n' "$REV" > "$WORKSPACE/vendor/openpi_revision.txt"
else
    [[ -f "$DEST/src/openpi/training/config.py" ]] || { echo 'Incomplete vendor copy; preserved for inspection'; exit 1; }
    echo 'Pinned snapshot already exists; preflight will fingerprint its code.'
fi
echo 'OPENPI_SOURCE_SNAPSHOT_READY; no shared source or environment modified'
