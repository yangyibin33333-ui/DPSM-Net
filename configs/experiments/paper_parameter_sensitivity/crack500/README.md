# Crack500 final-model OFAT parameter sensitivity

This registry contains the preregistered one-factor-at-a-time (OFAT) sensitivity
experiments for the frozen final DPER–RSDM–DPSM80 model.  Every row starts from
`M5_FrameworkOn_DPSMOn_StructSupOn.yaml`: C40, LSSCF, DPER, RSDM, DPSM80
(four 80-channel groups, one batch-packed shared Mamba core), four-direction
scan, learned route modulation, and Frozen Auto Protocol V2 are retained.

The 12 new Crack500 experiments all use seed 42, the native resolved 352x640
input, effective batch 8, 64 target epochs, and 18,816 target optimizer updates.
Training starts from random initialization.  Validation `structural_score`
selects the checkpoint; the test split is run once after training and never
selects a value, a seed, or a checkpoint.

| Family | New registrations | Sole changed factor | Center/value reuse |
|---|---|---|---|
| DPER detail width | `PS_DW08`, `PS_DW12`, `PS_DW20`, `PS_DW24` | `model.early_detail_channels` | 16 reuses M5 |
| RSDM strip kernel | `PS_K09`, `PS_K17` | `model.rsdm_kernel_size` | 13 reuses M5 |
| RSDM diagonal dilation | `PS_R01`, `PS_R05` | `model.rsdm_dilation` | 3 reuses M5 |
| tangent weight | `PS_LT010`, `PS_LT030` | `loss.lambda_tangent` | 0 reuses D2; 0.2 reuses M5 |
| BPCM weight | `PS_LB0125`, `PS_LB0500` | `loss.lambda_bpcm` | 0 reuses D1; 0.025 reuses M5 |

`rsdm_kernel_size` and `rsdm_dilation` default to 13 and 3.  They are written
only in non-center RSDM YAMLs so a strict config diff records exactly one
experimental factor.  The final default's state-dict names/shapes, 3,193,751
parameters, initialization, forward path, DPSM80 mathematics, route mechanism,
loss definitions, data split, augmentation, EMA, and Auto Protocol V2 are
frozen.

These are single-seed sensitivity trends, not statistical-significance claims.
All values are preregistered: no row may be dropped or selected from test
metrics, and a stronger single-seed score cannot revise the frozen model.  If
three-seed analysis is later needed, seed 123 and seed 3407 must be added for
the complete registered value set rather than only favorable values.  The
structure knobs belong in the main appendix; loss-weight scans are supplementary
material.

Run the config audit and the listed regressions on demamba21 before submitting
jobs.  The formal backend is real `mamba_ssm` on RTX 4090 D; a CPU fallback is
not formal-model evidence.
