# Server commands (RTX 4090 D / demamba21)

Run validation first. No command below uses test data for selection.

```bash
python tests/test_dpsm_lightweight_ablation.py && \
python tests/test_direction_partitioned_shared_mamba.py && \
python tests/test_paper_dependency_aware_ablation_configs.py
```

Do not use `--auto-protocol-dry-run` for the completed seed-42 L1/L2 runs:
it performs an optimizer update. The read-only tests above are the required
preflight for the remaining seeds.

Seed-42 logs for L1 and L2 are already complete (`train (96).log` and
`train (97).log`) and must not be overwritten. L0 quality reuses M1's
42/123/3407 runs, L3 quality reuses M3's 42/123/3407 runs, and L4 remains a
single-seed heavy reference. The only remaining formal quality jobs are L1
and L2 at seeds 123 and 3407; leave final testing enabled and do not append
`--skip-final-test`:

```bash
python train.py --config configs/experiments/dpsm_lightweight_ablation/crack500/L1_SingleRasterMamba80.yaml --device cuda --seed 123 && \
python train.py --config configs/experiments/dpsm_lightweight_ablation/crack500/L1_SingleRasterMamba80.yaml --device cuda --seed 3407 && \
python train.py --config configs/experiments/dpsm_lightweight_ablation/crack500/L2_FourIndependentMamba80.yaml --device cuda --seed 123 && \
python train.py --config configs/experiments/dpsm_lightweight_ablation/crack500/L2_FourIndependentMamba80.yaml --device cuda --seed 3407
```

All four runs use the existing YAML and therefore create independent timestamped directories under their existing unique save directories. Verify resolved 352×640 protocol, effective batch 8, 18,816 updates, real `mamba_ssm`, initialization checksum, and the selected validation checkpoint before launch; final test runs once only after selection.

Main 512×512 paper efficiency protocol (mask-only, three rounds, FP16, 100
warmups, 300 timed forwards; no loader/preprocessing/metric work).  The
orchestrator creates a new timestamped directory, audits the actual Stage4
topology and backend, performs full-mask/mask-only equivalence before timing,
and stops with an incomplete summary on the first failed variant:

```bash
python scripts/measure_dpsm_lightweight_efficiency.py --device cuda
```

Its fixed manifest is `L0=0`, `L1=1`, `L2=4`, `L3=1`, `L4=4` expected
Stage4 Mamba cores. It invokes the per-model tool with
`--expected-mamba-cores` and `--require-mamba-ssm`; L0's empty backend is only
accepted after the real no-route/no-`route_s4` topology is verified.

The resulting timestamped `summary.csv`/`summary.json`/`summary.md` data are
paper-admissible only when every variant is `complete`.  FLOPs are explicitly
reported as `standard_flops_lower_bound`, `mamba_flops_estimated`, and
`total_flops_estimated`; never cite the lower bound alone as total FLOPs.

Measure the unique main-table structures at the same 512×512 FLOP input only:

```bash
for cfg in configs/experiments/paper_dependency_aware_ablation/crack500/M0_FrameworkOff_DPSMOff_StructSupOff.yaml configs/experiments/paper_dependency_aware_ablation/crack500/M1_FrameworkOn_DPSMOff_StructSupOff.yaml configs/experiments/paper_dependency_aware_ablation/crack500/M2_FrameworkOff_DPSMOn_StructSupOff.yaml configs/experiments/paper_dependency_aware_ablation/crack500/M3_FrameworkOn_DPSMOn_StructSupOff.yaml; do python scripts/measure_model_complexity.py --config "$cfg" --img_size 512 512 --batch_size 1 --device cuda --amp fp16 --warmup 100 --iterations 300 --benchmark_repeats 3 --inference_path full --require_mamba_ssm; done
```
