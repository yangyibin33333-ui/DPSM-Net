# TUT dependency-aware confirmation (T0--T3)

This directory is the pre-registered TUT, seed-42 confirmation of the Crack500 dependency-aware main chain. It is not the earlier `P0`--`P2` Stage4-order pilot: it uses the frozen parallel model only, has independent save directories, and leaves final testing enabled. Test data are never used for seed, parameter, epoch, threshold, or checkpoint selection; each run tests the validation-selected structural-score checkpoint exactly once after training.

| ID | Crack500 mapping | Framework | Stage4 | StructSup |
|---|---|---:|---|---:|
| T0 | M0 | off | `NoDPSMTopoBlock` | off |
| T1 | M1 | on: LSSCF+DPER+RSDM | `NoDPSMTopoBlock` | off |
| T2 | M3 | on: LSSCF+DPER+RSDM | DPSM80 | off |
| T3 | M5 | on: LSSCF+DPER+RSDM | DPSM80 | on |

Framework-on keeps RSDM exclusively on DPER's half-resolution 40-channel semantic stream. Framework-off never runs RSDM. DPSM-off is the existing topology-matched local Stage4: it has no Mamba core, route head, or `route_s4`. DPSM-on has four 80-channel directional groups packed as `(4*B, H*W, 80)` through one shared `mamba_ssm` core, while retaining the Stage4 Transformer.

All four files freeze Auto Protocol V2 and differ only in the registered Framework fields, DPSM fields, or the two structural-supervision weights. On TUT the frozen large regime resolves native input to 640x640, effective batch 8, 75 target epochs, and 9,225 target optimizer updates. SCNP/BPCM phases, validation every 400 updates, significant structural-score checkpoint choice, and update-clock early stopping are identical across rows. Training starts from random initialization; no adjacent-ablation checkpoint is loaded.

Run the read-only audit before server submission:

```bash
python tests/test_paper_dependency_aware_ablation_tut_configs.py
```

Do not add `--skip-final-test` to the formal commands. The flag belongs only to the historical position diagnostic and would suppress the one allowed final test evaluation for this confirmation.
