# TUT dependency-aware confirmation commands (RTX 4090 D / demamba21)

First run the configuration/build audit. It does not construct a dataloader, perform an Auto Protocol dry-run, optimizer update, validation, test evaluation, or formal training.

```bash
python tests/test_paper_dependency_aware_ablation_tut_configs.py
```

After it passes, run the four seed-42 jobs. Each command random-initializes, uses its own timestamped subdirectory below the configured unique `save_dir`, and performs exactly one final test only after validation structural-score checkpoint selection. Do not add `--skip-final-test`.

```bash
python train.py --config configs/experiments/paper_dependency_aware_ablation/tut/T0_FrameworkOff_DPSMOff_StructSupOff.yaml --device cuda --seed 42
python train.py --config configs/experiments/paper_dependency_aware_ablation/tut/T1_FrameworkOn_DPSMOff_StructSupOff.yaml --device cuda --seed 42
python train.py --config configs/experiments/paper_dependency_aware_ablation/tut/T2_FrameworkOn_DPSMOn_StructSupOff.yaml --device cuda --seed 42
python train.py --config configs/experiments/paper_dependency_aware_ablation/tut/T3_FrameworkOn_DPSMOn_StructSupOn.yaml --device cuda --seed 42
```

Before each submission, confirm the resolved config reports 640x640, effective batch 8, target 9,225 updates, the initialization checksum, and the independent save path. T0/T1 must log `NoDPSMTopoBlock`, route branch absent, and no Stage4 Mamba backend. T2/T3 must log DPSM80's four 80-channel groups, one packed shared `mamba_ssm` core, a route head, and `route_s4`. Do not use test results to choose seeds, checkpoints, or follow-up rows.
