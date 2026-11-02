# DPSM80 lightweight ablation (Crack500)

All rows retain the frozen Framework-on encoder (LSSCF + DPER + RSDM), Stage4
Local/Transformer/fusion/refine, decoder, loss protocol, and Auto Protocol V2
from `M3_FrameworkOn_DPSMOn_StructSupOff`. `lambda_tangent` and `lambda_bpcm`
are both zero. Only the Stage4 routed/Mamba branch changes.

| Row | Stage4 routed branch | cores / width / directions | training |
|---|---|---|---|
| L0 | `NoDPSMTopoBlock`: no route/Mamba, 2-way local+Transformer fusion | 0 / - / 0 | reuse strict_v2 M1 seed 42 |
| L1 | one raster-order 320→80→Mamba→320 residual | 1 / 80 / raster | new seed 42 |
| L2 | four independent tangent-routed directional cores | 4 / 80 / 4 | new seed 42 |
| L3 | frozen DPSM80, 4×80 partition and packed shared core | 1 / 80 / 4 | reuse strict_v2 M3 seed 42 |
| L4 | original tangent-routed full-width reference | 4 / 320 / 4 | new seed 42 |

L1's zero `[B,4,H,W]` route tensor is an API-only compatibility output and
never affects features; its tangent loss is forcibly zero by configuration.
L1/L2 first construct the normal full-width `TopoRouteBlock`, then replace only
`route_mamba` after shared components are initialized. Therefore Local,
Transformer, fusion, and refine match L3 element-wise under the same seed.

Main efficiency comparisons use **512×512**, batch 1, FP16, eval/inference
mode, CUDA Events, 100 warmups, 300 timed forwards, and three independent
rounds. Native 352×640 results, if recorded, are labelled deployment-only and
never mixed with the main table. The formal entry point first proves that the
eval-only mask path is numerically equivalent to the full-forward mask (FP16
`atol=rtol=1e-3`). It writes a new timestamped directory and refuses a complete
table if any row fails its expected Mamba-core/backend audit.

The formal tool saves the server's actual `Mamba.forward` source and a
single-core fvcore `by_operator`/`by_module` trace before reporting FLOPs. It
uses `fvcore_counted_flops`, `mamba_fused_missing_flops_estimated`, and
`corrected_flops_estimated`; only components proved to be inside the unsupported
`MambaInnerFn` are added. The convention is one fused multiply-add equals one
FLOP. `Estimated FLOPs†` means “fvcore-counted operations plus an analytical
estimate for the fused Mamba operator”; remaining unsupported elementwise,
reduction, and data-rearrangement operations remain disclosed, so it is not an
exact-FLOPs claim.

L2 and L3 intentionally differ in more than parameter sharing. L2 has four
independent calls and padded diagonal traversals; L3 uses one shared core with
four packed, padding-free ordered sequences. The machine-readable report and
Markdown summary disclose cores, calls, token slots, diagonal-padding token
slots, packed execution, and every captured Mamba input shape. DPSM80 therefore
combines core sharing, packed directional execution, and padding-free ordered
traversals; no speed comparison is attributed to sharing alone.
