# Crack500 R0 supervised-route control

First run the read-only audits on demamba21. They do not access the dataset or
start a training run.

```bash
python tests/test_paper_scan_route_ablation_configs.py
python tests/test_paper_supervised_route_ablation_configs.py
```

Before formal scheduling, confirm the R0 build/log shows `scan_mode:
four_direction`, `route_mode: uniform`, tangent `0.2000`, BPCM `0.0000`, four
80-channel groups (width 320), one shared Mamba core, initial checksum
`3f7cd9a6...`, and one `DPSM80 first actual execution` entry per run. Confirm
Auto Protocol V2 resolves native 352×640, micro-batch 8 / accumulation 1 /
effective batch 8, 18,816 updates and normal validation checkpoint selection.

After those checks, the pre-registered three-seed R0 set is:

```bash
python train.py --config configs/experiments/paper_supervised_route_ablation/crack500/R0_FourDirection_Uniform_TangentOn.yaml --device cuda --seed 42
python train.py --config configs/experiments/paper_supervised_route_ablation/crack500/R0_FourDirection_Uniform_TangentOn.yaml --device cuda --seed 123
python train.py --config configs/experiments/paper_supervised_route_ablation/crack500/R0_FourDirection_Uniform_TangentOn.yaml --device cuda --seed 3407
```

Run and report all three once scheduling begins; do not use seed 42 to decide
whether to keep seeds 123/3407. R1 is the existing
`paper_dependency_aware_ablation/.../loss_2x2/D1_TangentOn_BPCMOff.yaml`
three-seed record and is not rerun. No checkpoint from another ablation is
loaded.
