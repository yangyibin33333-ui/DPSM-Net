# Crack500 tangent-supervised DPSM80 route-modulation ablation

This two-row causal control asks one narrow question: when the same four-way
route head receives frozen tangent supervision, should predicted softmax weights
modulate the four DPSM80 feature groups?

| ID | scans | raw `route_s4` / tangent / BPCM | DPSM feature modulation | source |
|---|---|---|---|---|
| R0 | H,V,D,AD | retained / 0.2 / 0 | uniform 1/4; existing `4×weight` gives unit gain | new YAML |
| R1 | H,V,D,AD | retained / 0.2 / 0 | learned `softmax(route_logits)` with existing `4×weight` | historic D1 |

R1 is exactly
`configs/experiments/paper_dependency_aware_ablation/crack500/loss_2x2/D1_TangentOn_BPCMOff.yaml`.
It has no duplicate R1 YAML, run directory, checkpoint, or rerun. After default
normalization, R0 differs from D1 only in `model.dpsm_route_mode` and `save`.

Uniform retains normal route-head execution and emits unmodified `route_s4`.
TangentRouteLoss can therefore update its parameters, while mask/edge losses
cannot reach it through DPSM feature modulation. This is not a tangent-loss,
scan-count, global-route, or efficiency comparison.

Both rows retain Framework/LSSCF/DPER/RSDM, 320=4×80 Stage4 groups, one packed
shared Mamba core, H/V/D/AD scans, parallel Stage4 topology, route scale four,
data, augmentation, optimizer, EMA, evaluation and checkpoint rules. Crack500
Auto Protocol V2 remains native 352×640, effective batch 8, 18,816 updates and
64 epochs. Test follows validation checkpoint selection once only.

Run read-only audits before scheduling R0. CPU fallback is logic-only; checksum
and backend evidence require demamba21 CUDA + mamba_ssm.
