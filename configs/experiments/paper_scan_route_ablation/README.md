# Crack500 DPSM80 scan × route ablation

This is a pre-registered 2×2 mechanism table inside the frozen DPSM80 Stage4
path.  It does not alter the decoder, Framework (LSSCF + DPER + RSDM), 320-wide
Stage4 interface, Transformer, parallel fusion, refinement, output heads,
training protocol, or evaluation/checkpoint rules.

| ID | g0/g1/g2/g3 scan assignment | route modulation | tangent/BPCM | source |
|---|---|---|---|---|
| C0 | H, V, H, V | uniform (1/4 each) | 0 / 0 | `C0_AxialRepeated_Uniform.yaml` |
| C1 | H, V, H, V | learned softmax | 0 / 0 | `C1_AxialRepeated_Learned.yaml` |
| C2 | H, V, D, AD | uniform (1/4 each) | 0 / 0 | `C2_FourDirection_Uniform.yaml` |
| C3 | H, V, D, AD | learned softmax | 0 / 0 | existing `paper_dependency_aware_ablation/.../M3_FrameworkOn_DPSMOn_StructSupOff.yaml` |

Each row retains four independently computed 80-channel group sequences,
packed as `[4B, Hs×Ws, 80]` through exactly one shared Mamba core. `H` and `V`
are the existing full-grid snake traversals. The axial-repeated rows do not use
two 160-channel branches, reverse scans, padding, a new projection, or a
smaller sequence budget; g2 and g3 consume their own features.

`uniform` retains `route_head` and emits its raw logits, preserving parameter
registration and output API. It uses fixed weights 1/4 and the original route
scale 4, hence each group has unit feature gain. Its `route_head` therefore has
no gradient from the mask path; this is a mechanism ablation, not a claim that
all registered parameters remain functionally trainable. Tangent/BPCM are
fixed to zero for non-default mechanisms because axial g2/g3 are not the
original diagonal tangent classes.

The comparison pairs are C0→C1 and C2→C3 (learned modulation), and C0→C2 and
C1→C3 (diagonal scan assignment). They do not measure a two-scan efficiency
reduction and cannot establish learned pixel routing over a global learned
alternative, which is not registered here.

All three new rows use the frozen Crack500 Auto Protocol V2: native 352×640,
effective batch 8, target 18,816 updates / 64 epochs, EMA, validation every
400 updates, and the existing significant-best checkpoint and early-stopping
rules. Seed is supplied by each command. C3 reuses the completed M3 result
only after the default-equivalence audit passes; no duplicate C3 YAML or run
directory is created.

Run the audit before any server training:

```bash
python tests/test_paper_scan_route_ablation_configs.py
```

Formal server runs require the real `mamba_ssm` backend. Confirm the resolved
config, native input, micro-batch/accumulation (expected 8/1), effective batch
8, 18,816 updates, one shared backend call with `[4B,Hs×Ws,80]`, initial
checksum, and each unique save directory. Test remains a single final
evaluation after validation chooses the checkpoint; it is never used to select
a seed, configuration, or checkpoint.
