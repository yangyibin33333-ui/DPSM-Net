#!/usr/bin/env bash
set -e

python train.py \
  --config configs/demambanet.yaml \
  --data_root ./DeepCrack \
  --save_dir ./runs/demambanet
