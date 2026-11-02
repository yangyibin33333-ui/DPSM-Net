# Crack500 C0–C2 server commands

On demamba21 with the real `mamba_ssm` environment, first audit the source and
new registrations. This is read-only and does not train.

```bash
python tests/test_paper_scan_route_ablation_configs.py
```

After confirming the resolved Auto Protocol V2 reports native 352×640,
micro-batch 8 / accumulation 1 / effective batch 8, target 18,816 updates and
one real `mamba_ssm` Stage4 core, run the three new seed-42 jobs:

```bash
python train.py --config configs/experiments/paper_scan_route_ablation/crack500/C0_AxialRepeated_Uniform.yaml --device cuda --seed 42
python train.py --config configs/experiments/paper_scan_route_ablation/crack500/C1_AxialRepeated_Learned.yaml --device cuda --seed 42
python train.py --config configs/experiments/paper_scan_route_ablation/crack500/C2_FourDirection_Uniform.yaml --device cuda --seed 42
```

For each run, verify the log records `g0`–`g3` actual scan assignments, total
width 320 = 4×80, route scale 4, first-real-forward packed shape/call count,
initial checksum, and the independent save directory under
`runs/paper/scan_route_ablation_strict_v2/`. Uniform rows must state that the
route head is retained for raw logits/state compatibility but does not modulate
mask features. Do not alter a run if the server resolves a different micro
batch; pause the formal comparison and report it.

C3 is the existing M3 result:

```text
configs/experiments/paper_dependency_aware_ablation/crack500/M3_FrameworkOn_DPSMOn_StructSupOff.yaml
```

It is not rerun or assigned a duplicate save directory if the default
equivalence audit passes. After all four rows, a three-seed table must add
both seed 123 and seed 3407 for each of C0, C1 and C2; do not choose rows or
seeds from test results.
