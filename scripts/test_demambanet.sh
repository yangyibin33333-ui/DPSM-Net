#!/usr/bin/env bash
set -e

CHECKPOINT="${1:?Usage: bash scripts/test_demambanet.sh <checkpoint-path>}"

python test.py \
  --config configs/demambanet.yaml \
  --checkpoint "$CHECKPOINT" \
  --save_pred
