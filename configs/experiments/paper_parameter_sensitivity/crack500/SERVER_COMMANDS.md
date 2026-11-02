# demamba21: Crack500 final-model parameter sensitivity

From the repository root, first run only audit/regression checks (these are not
formal training):

```bash
python tests/test_paper_parameter_sensitivity_configs.py && \
python tests/test_rift_structural_directional_modeling.py && \
python tests/test_detail_preserving_early_rebase.py && \
python tests/test_paper_dependency_aware_ablation_configs.py && \
python tests/test_paper_three_factor_ablation_configs.py
```

Before each submitted run, verify in its log: identical Crack500 dataset
fingerprint; resolved 352x640 input; effective batch 8; 18,816 target updates;
all M5 structure flags; the intended DPER width/RSDM kernel/dilation/loss
weight; exactly one real `mamba_ssm` Stage4 core; a unique save directory; and
no automatic checkpoint resume.  Each run starts randomly initialized and uses
validation structural score only; test is never used to choose a setting.

Run the preregistered sequence as one continuous shell command:

```bash
python train.py --config configs/experiments/paper_parameter_sensitivity/crack500/PS_DW08.yaml --seed 42 --device cuda && \
python train.py --config configs/experiments/paper_parameter_sensitivity/crack500/PS_DW12.yaml --seed 42 --device cuda && \
python train.py --config configs/experiments/paper_parameter_sensitivity/crack500/PS_DW20.yaml --seed 42 --device cuda && \
python train.py --config configs/experiments/paper_parameter_sensitivity/crack500/PS_DW24.yaml --seed 42 --device cuda && \
python train.py --config configs/experiments/paper_parameter_sensitivity/crack500/PS_K09.yaml --seed 42 --device cuda && \
python train.py --config configs/experiments/paper_parameter_sensitivity/crack500/PS_K17.yaml --seed 42 --device cuda && \
python train.py --config configs/experiments/paper_parameter_sensitivity/crack500/PS_R01.yaml --seed 42 --device cuda && \
python train.py --config configs/experiments/paper_parameter_sensitivity/crack500/PS_R05.yaml --seed 42 --device cuda && \
python train.py --config configs/experiments/paper_parameter_sensitivity/crack500/PS_LT010.yaml --seed 42 --device cuda && \
python train.py --config configs/experiments/paper_parameter_sensitivity/crack500/PS_LT030.yaml --seed 42 --device cuda && \
python train.py --config configs/experiments/paper_parameter_sensitivity/crack500/PS_LB0125.yaml --seed 42 --device cuda && \
python train.py --config configs/experiments/paper_parameter_sensitivity/crack500/PS_LB0500.yaml --seed 42 --device cuda
```

Center values intentionally do not create duplicate jobs: width 16, kernel 13,
dilation 3, tangent 0.2, and BPCM 0.025 reuse M5; tangent 0 reuses D2 and BPCM
0 reuses D1.  Do not use test outcomes to add, remove, or reorder a value.
