# Paper Main Table A: frozen A0--A5 chain

This registry prepares the twelve formal seed-42 runs for DeepCrack and
Crack500.  Every YAML is copied from the frozen Auto Protocol V2 base and has a
unique `save.save_dir`.  Runs are random initialization only: no weights may be
loaded from an adjacent ablation, and test results never choose parameters,
seeds, or checkpoints.

| ID | Sole change from the preceding row |
| --- | --- |
| A0 | Original C40: `mamba_stages=[3,4]`, ordinary decoder, LSSCF/DPER/RSDM/DPSM off. |
| A1 | Only `mamba_stages=[4]`: the C40-S4Mamba bridge baseline. |
| A2 | Only turns on LSSCF. |
| A3 | Only turns on DPER. |
| A4 | Only turns on RSDM; Stage4 remains the original four independent full-width directional Mamba paths. |
| A5 | Only turns on DPSM: four 80-channel direction groups batch-packed into one shared Mamba core. |

A1 is mandatory because A0 to A2 changes both Stage3 Mamba and the decoder.
Consequently A0 must not be compared directly with A2 and attributed to LSSCF;
the controlled LSSCF contrast is A1 to A2.

`early_detail_channels=16` and `stage4_direction_group_channels=80` are
retained in every row as inert parameters when their associated modules are
off.  All historical failure/candidate branches (including CP-DPSM, FWRMH,
DAFU1, DSMH, NTD/PNTD, SPCR, HTPR, FJSE, ESCS, PFRS, MSDR, LSAF, CFPS, and
distillation) are explicitly disabled.

Frozen Auto Protocol V2 targets are: DeepCrack `384x544`, effective batch 8,
200 epochs, and 9200 updates; Crack500 `352x640`, effective batch 8, 64 epochs,
and 18816 updates.  Actual micro-batch may differ only with gradient
accumulation preserving effective batch 8.  The seed is supplied as `--seed 42`
in the server commands.

Validation structural score controls significant checkpoints and early stopping;
the test split runs only once after training/checkpoint selection is complete.
Reported FLOPs are an `fvcore` lower bound, not an exhaustive count of every
fusion operation.  All formal runs must execute on `demamba21` (RTX 4090 D) in
the real `mamba_ssm` environment, never through a local fallback backend.
