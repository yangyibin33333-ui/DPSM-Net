# Server commands: Paper Main Table A

Run these only on `demamba21` with RTX 4090 D and the real `mamba_ssm`
environment.  All commands use random initialization and their YAMLs contain
separate output directories.

Before submitting all twelve tasks, run a server-side protocol parse/short
validation for A0 and A5 of each dataset.  Check the logged resolved config,
effective batch, target updates, initial checksum, and real-Mamba call pattern:
A0 uses original Stage3 and Stage4 direction cores; A1--A3 only original Stage4
direction cores; A4 four independent full-width Stage4 direction cores; A5 one
batch-packed shared 80-channel Stage4 core.  Do not use that validation's test
result for any selection.

First run the static registry audit:

```bash
python tests/test_paper_main_table_a_configs.py
```

## DeepCrack

```bash
python train.py \
  --config configs/experiments/paper_main_table_a/deepcrack/A0_C40_S34.yaml \
  --seed 42 \
  --device cuda

python train.py \
  --config configs/experiments/paper_main_table_a/deepcrack/A1_C40_S4.yaml \
  --seed 42 \
  --device cuda

python train.py \
  --config configs/experiments/paper_main_table_a/deepcrack/A2_C40_S4_LSSCF.yaml \
  --seed 42 \
  --device cuda

python train.py \
  --config configs/experiments/paper_main_table_a/deepcrack/A3_C40_S4_LSSCF_DPER.yaml \
  --seed 42 \
  --device cuda

python train.py \
  --config configs/experiments/paper_main_table_a/deepcrack/A4_C40_S4_LSSCF_DPER_RSDM.yaml \
  --seed 42 \
  --device cuda

python train.py \
  --config configs/experiments/paper_main_table_a/deepcrack/A5_C40_S4_LSSCF_DPER_RSDM_DPSM80.yaml \
  --seed 42 \
  --device cuda
```

## Crack500

```bash
python train.py \
  --config configs/experiments/paper_main_table_a/crack500/A0_C40_S34.yaml \
  --seed 42 \
  --device cuda

python train.py \
  --config configs/experiments/paper_main_table_a/crack500/A1_C40_S4.yaml \
  --seed 42 \
  --device cuda

python train.py \
  --config configs/experiments/paper_main_table_a/crack500/A2_C40_S4_LSSCF.yaml \
  --seed 42 \
  --device cuda

python train.py \
  --config configs/experiments/paper_main_table_a/crack500/A3_C40_S4_LSSCF_DPER.yaml \
  --seed 42 \
  --device cuda

python train.py \
  --config configs/experiments/paper_main_table_a/crack500/A4_C40_S4_LSSCF_DPER_RSDM.yaml \
  --seed 42 \
  --device cuda

python train.py \
  --config configs/experiments/paper_main_table_a/crack500/A5_C40_S4_LSSCF_DPER_RSDM_DPSM80.yaml \
  --seed 42 \
  --device cuda
```
