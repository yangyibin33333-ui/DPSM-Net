# Strict Stage4 T0 / T2 acceptance report

No training, optimizer update, dataset evaluation, or historical output modification was performed.

## Changed source and new configuration

- `models/de_mambanet.py`: opt-in `stage4_variant=None | local_only | local_dpsm80`; historical defaults unchanged. Only the explicitly validated new paths bypass the historical RSDM/NoDPSM/DPSM Transformer guards.
- `models/blocks.py`: two strict Stage4 containers. T0 registers local, 320→320 fuse and refine. T2 registers local, the original DirectionPartitionedSharedMamba object, 640→320 fuse and refine. Neither registers a global branch or an Identity placeholder.
- `configs/experiments/stage4_compact_ablation/crack500/T0_LocalOnly_StrictV2.yaml`
- `configs/experiments/stage4_compact_ablation/crack500/T2_LocalDPSM80_StrictV2.yaml`
- `tests/test_stage4_compact_strict_v2.py`: synthetic acceptance and configuration/checkpoint audit.
- `tests/measure_stage4_compact.py`: subsequent GPU-only efficiency measurement entry point, using existing repository measurement functions.

The historical Stage4 scaffold is temporarily constructed to preserve initialization of retained modules. It is then replaced before the model can run. Deleted branches are unregistered, absent from named_modules/named_parameters/state_dict, and never execute. Local DWConvBlock→StripConvBlock, outside residual and refine are reused exactly. T2 retains the same formal four-direction, four 80-channel groups, packed shared core, route modulation and outputs as T3. `models/mamba_blocks.py` was not edited.

## Executed checks

All T0/T1/T2/T3 models were instantiated and checked using actual module types and parameters. Each passed batch-1 352×640 and 512×512 forward/backward in FP32. Every output and gradient was finite, every trainable parameter received a gradient, and input gradients were finite and nonzero. T2/T3 executed exactly one core call with [4,L,80] input. These are code checks, not performance evidence.

Environment: Windows, PyTorch 2.14.1+cpu, no CUDA or mamba_ssm. T2/T3 execution used the repository fallback. This does not validate official fused Mamba/CUDA execution or AMP training. Dependencies were installed only under `tmp/stage4_runtime`; historical source snapshots used for regression are under `tmp/stage4_audit`.

| Model | Actual desktop Params | Formal architecture Params |
|---|---:|---:|
| T0 | 1,681,107 | 1,681,107 |
| T1 | 2,605,267 | 2,605,267 |
| T2 | 2,260,871 | 2,269,591 |
| T3 | 3,185,031 | 3,193,751 |

Formal T2/T3 counts are checkpoint-schema tensor counts, not claims of formal-backend instantiation: historical T3 core parameters = 48,800, desktop fallback core parameters = 40,080, difference = 8,720. Non-core checkpoint keys/shapes were checked equal. The T2 container retains exactly the T3 routed module.

| Model / mask-only | 352×640 | 512×512 | Meaning |
|---|---:|---:|---|
| T0 | 19.954401280 G | 23.219666944 G | fvcore-counted operations; unsupported ops omitted |
| T2 desktop fallback | 22.420654080 G | 26.089488384 G | diagnostic partial count only; NOT formal DPSM80 FLOPs |

Convention: one multiply-add counts as one FLOP. Full unsupported-operator lists are in `report.json`. Formal T2 Estimated FLOPs, FPS and latency remain unmeasured. No speed values are claimed.

## Compatibility and configuration

- T1/T3 current vs pre-change model: every state_dict key, shape and initialized tensor is identical at seed 42. T0 retained 230 identical tensors relative to T1; T2 retained 260 relative to T3, excluding only their required new fusion.
- Historical T1 checkpoint: entire schema equal and strict load succeeded.
- Historical T3 checkpoint: all non-backend keys/shapes equal. Its Mamba backend schema differs from desktop fallback exactly as expected. Strict loading into real mamba_ssm is pending GPU preflight; same-backend historical construction was preserved.
- Existing `test_direction_partitioned_shared_mamba.py`: all 8 checks passed, covering scan/inverse, guards, original RSDM, structure, initialization, modulation, packing/isolation, and full-model backward.
- Syntax compilation passed for both changed model files and both new Python test files. The efficiency helper's CLI help was executed; GPU timing itself was not run.
- Five invalid selector combinations were rejected. No historical guard was globally removed.
- Six historical resolved configs (M1/M3, seeds 42/123/3407) were read. The real V2 resolver was exercised with historical train-size/foreground metadata substituted for data reads. This is a deterministic historical protocol replay, not a fresh data audit or GPU micro-batch probe.
- All six historical runs match data/train/loss/evaluation settings and all compared planned budgets. All six pairwise T0/T1/T2/T3 resolved differences are restricted to Stage4 selection/legacy architecture fields and isolated save paths. `report.json` contains every differing field and both values; `resolved_configs.json` contains the full replay-resolved configs.
- Resolved plan: 352×640, effective batch 8, 64 epochs, 294 steps/epoch, 18,816 updates, SCNP warmup 941 updates, validation every 400 updates, earliest stopping 10,349 updates, patience 2,823 updates. AdamW LR 0.00015, decay 0.0001, cosine, EMA 0.999, edge 0.3, SCNP 0.2, tangent/BPCM 0. The shared BCE+Dice, selection and evaluation implementation was untouched.
- SHA256 hashes for train.py, test.py, models/mamba_blocks.py, all loss and metric files were unchanged across the audit and are recorded in `report.json`. Historical M1/M3 configs/results were never written.

## Next steps and commands (not executed)

Code/configuration acceptance is complete for the desktop backend. Full formal-training readiness requires a successful real Mamba/CUDA preflight in the intended training environment. Keep the same package versions as the historical experiment environment. The preflight performs synthetic forward/backward only:

```bash
python tests/test_stage4_compact_strict_v2.py --device cuda --require-mamba-ssm
```

After that passes, these commands launch the six planned experiments in isolated directories. The seed is supplied through the actual train.py CLI, not an ignored YAML seed field:

```bash
for variant in T0_LocalOnly_StrictV2 T2_LocalDPSM80_StrictV2; do
  for seed in 42 123 3407; do
    python train.py \
      --config "configs/experiments/stage4_compact_ablation/crack500/${variant}.yaml" \
      --seed "$seed" \
      --save_dir "runs/paper/stage4_compact_ablation_strict_v2/crack500/${variant}_seed${seed}"
  done
done
```

The existing measure_model_complexity.py CLI itself has frozen Transformer/class guards that reject the new strict containers. Its compatible low-level functions were reused by the new test helper without changing the historical script. Subsequent formal measurements (512×512, batch 1, FP16, mask-only, 100 warmups, 300 iterations, 3 repetitions):

```bash
python tests/measure_stage4_compact.py --config configs/experiments/stage4_compact_ablation/crack500/T0_LocalOnly_StrictV2.yaml --img-size 512 512
python tests/measure_stage4_compact.py --config configs/experiments/stage4_compact_ablation/crack500/T2_LocalDPSM80_StrictV2.yaml --img-size 512 512
```

Each creates a fresh output directory. Add `--img-size 352 640` for deployment-size measurements. The helper rejects CPU/fallback and requires valid fused-Mamba boundary evidence for T2 FLOPs. It records unsupported operations and the estimation convention.
