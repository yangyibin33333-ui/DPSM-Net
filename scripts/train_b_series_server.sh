#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DEEPCRACK_ROOT="${DEEPCRACK_ROOT:-$PWD/DeepCrack}"
python -B scripts/b_series_study.py --mode preflight --data-root "$DEEPCRACK_ROOT"
trap 'python -B scripts/report_b_series.py' EXIT
python -B scripts/b_series_study.py --mode all --data-root "$DEEPCRACK_ROOT" --execute
