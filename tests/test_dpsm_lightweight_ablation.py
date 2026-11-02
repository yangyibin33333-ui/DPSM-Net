"""Audit L0--L4 DPSM80 lightweight structures; no training or data loader.

Run: ``python tests/test_dpsm_lightweight_ablation.py``.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
except ModuleNotFoundError:
    torch = None
else:
    from models import build_model
    from models.blocks import NoDPSMTopoBlock, TopoRouteBlock
    from models.mamba_blocks import DirectionPartitionedSharedMamba, SingleRasterMamba80, TangentRoutedMambaBlock, SequenceMambaCore
    from scripts.measure_model_complexity import (
        audit_stage4_mamba,
        capture_stage4_execution,
        diagnose_mamba_fused_boundary,
        estimate_mamba_inner_flops,
        measure_flops,
        _mamba_component_costs,
        validate_mask_only_equivalence,
    )


REGISTRY = ROOT / "configs" / "experiments" / "dpsm_lightweight_ablation" / "crack500"
ROWS = {"L0": "L0_NoMamba", "L1": "L1_SingleRasterMamba80", "L2": "L2_FourIndependentMamba80", "L3": "L3_DPSM80", "L4": "L4_FourIndependentMamba320"}


def load(identifier: str) -> dict:
    with (REGISTRY / f"{ROWS[identifier]}.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build(identifier: str):
    cfg = copy.deepcopy(load(identifier)["model"]); name = cfg.pop("name")
    cfg["use_mamba_ssm"] = bool(torch.cuda.is_available())
    return build_model(name, **cfg).to("cuda" if torch.cuda.is_available() else "cpu").eval()


def test_yaml_registry_and_protocol() -> None:
    assert sorted(path.stem for path in REGISTRY.glob("L*.yaml")) == sorted(ROWS.values())
    configs = {key: load(key) for key in ROWS}
    reference = copy.deepcopy(configs["L3"]); reference.pop("model"); reference.pop("save")
    for key, cfg in configs.items():
        protocol = copy.deepcopy(cfg); protocol.pop("model"); protocol.pop("save")
        assert protocol == reference, key
        assert cfg["data"]["data_root"] == "./Crack500" and cfg["data"]["img_size"] is None
        assert cfg["loss"]["lambda_tangent"] == cfg["loss"]["lambda_bpcm"] == 0.0
        assert cfg["model"]["stage4_hybrid_order"] == "parallel"
    assert configs["L0"]["model"]["use_topology_matched_no_dpsm_stage4"]
    assert configs["L1"]["model"]["stage4_routed_branch"] == "single_raster_mamba80"
    assert configs["L2"]["model"]["stage4_routed_branch"] == "four_independent_mamba80"
    assert configs["L3"]["model"]["use_direction_partitioned_shared_mamba"]
    assert configs["L4"]["model"]["stage4_mamba_channels"] is None


def test_structure_shared_initialization_and_smoke() -> None:
    if torch is None:
        print("SKIP: torch unavailable; YAML audit completed")
        return
    models = {}
    for key in ROWS:
        torch.manual_seed(3407); models[key] = build(key)
    l0, l1, l2, l3, l4 = (models[key] for key in ROWS)
    assert isinstance(l0.stage4, NoDPSMTopoBlock) and not l0.route_stage4 and l0.stage4.fuse.block[0].in_channels == 640
    assert not hasattr(l0.stage4, "route_mamba")
    assert isinstance(l1.stage4, TopoRouteBlock) and isinstance(l1.stage4.route_mamba, SingleRasterMamba80)
    assert l1.stage4.route_mamba.instantiated_core_count == l1.stage4.route_mamba.active_mamba_path_count == 1
    assert not hasattr(l1.stage4.route_mamba, "route_head") and not hasattr(l1.stage4.route_mamba, "scans") and not hasattr(l1.stage4.route_mamba, "local_compensations")
    assert isinstance(l2.stage4.route_mamba, TangentRoutedMambaBlock) and l2.stage4.route_mamba.mamba_channels == 80
    assert l2.stage4.route_mamba.instantiated_core_count == l2.stage4.route_mamba.active_mamba_path_count == 4
    assert l2.stage4.route_mamba.local_compensations is None
    assert isinstance(l3.stage4.route_mamba, DirectionPartitionedSharedMamba) and l3.stage4.route_mamba.instantiated_core_count == 1
    assert isinstance(l4.stage4.route_mamba, TangentRoutedMambaBlock) and l4.stage4.route_mamba.mamba_channels == 320
    assert l4.stage4.route_mamba.instantiated_core_count == l4.stage4.route_mamba.active_mamba_path_count == 4 and l4.stage4.route_mamba.local_compensations is None
    for candidate in (l1, l2, l3, l4):
        for name in ("local", "global_branch", "fuse", "refine"):
            expected, actual = l3.stage4.state_dict(), candidate.stage4.state_dict()
            for key, value in expected.items():
                if key.startswith(f"{name}."):
                    assert key in actual and torch.equal(value, actual[key]), (candidate.stage4.route_mamba.__class__.__name__, key)
    device = next(l3.parameters()).device
    image = torch.randn(1, 3, 64, 96, device=device)
    for key, model in models.items():
        output = model(image); assert output["mask"].shape[-2:] == image.shape[-2:] and torch.isfinite(output["mask"]).all(), key
        if key == "L0": assert "route_s4" not in output
        else: assert output["route_s4"].shape[1] == 4
    assert l1.stage4.route_mamba.last_raster_sequence_shape == (1, 8 * 12, 80) and l1.stage4.route_mamba.last_core_call_count == 1
    assert l3.stage4.route_mamba.last_packed_shape == (4, 8 * 12, 80) and l3.stage4.route_mamba.last_shared_mamba_call_count == 1


def test_formal_efficiency_audit_and_mask_only_contract() -> None:
    if torch is None:
        print("SKIP: torch unavailable; formal efficiency audit not executed")
        return
    models = {}
    for key in ROWS:
        torch.manual_seed(42); models[key] = build(key)
    device = next(models["L0"].parameters()).device
    image = torch.randn(1, 3, 64, 96, device=device)
    expected = {"L0": 0, "L1": 1, "L2": 4, "L3": 1, "L4": 4}
    for key, model in models.items():
        with torch.inference_mode():
            outputs, captured = capture_stage4_execution(model, image)
        if key == "L0":
            audit = audit_stage4_mamba(
                model, variant=key, expected_cores=0, outputs=outputs, captured_inputs=captured, formal_backend=True
            )
            assert audit["mamba_instantiated_core_count"] == 0
            assert audit["mamba_forward_call_count"] == audit["mamba_routed_sequence_count"] == 0
            assert estimate_mamba_inner_flops(model, captured, None)[0] == 0.0
        mask = validate_mask_only_equivalence(model, image, device=device, amp="off")
        assert mask["mask_only_allclose"], key
        if key == "L3":
            assert audit_stage4_mamba(
                model, variant=key, expected_cores=None, outputs=outputs, captured_inputs=captured, formal_backend=False
            )["mamba_routed_sequence_count"] == 4
            assert len(captured[0]) == 1
    # A zero-core Stage4 cannot satisfy a positive-core formal contract.
    with torch.inference_mode():
        l0_output, l0_captured = capture_stage4_execution(models["L0"], image)
    try:
        audit_stage4_mamba(
            models["L0"], variant="L0", expected_cores=1, outputs=l0_output, captured_inputs=l0_captured, formal_backend=True
        )
    except RuntimeError as error:
        assert "expected=1" in str(error) and "actual=0" in str(error)
    else:
        raise AssertionError("L0 incorrectly passed expected_mamba_cores=1")
    # Missing metadata is not allowed to masquerade as the valid L0 empty state.
    l1_core = models["L1"].stage4.route_mamba.raster_mamba
    original_backend = l1_core.backend; l1_core.backend = "missing"
    try:
        with torch.inference_mode():
            l1_output, l1_captured = capture_stage4_execution(models["L1"], image)
        audit_stage4_mamba(
            models["L1"], variant="missing", expected_cores=1, outputs=l1_output,
            captured_inputs=l1_captured, formal_backend=True,
        )
    except RuntimeError as error:
        assert "mamba_ssm" in str(error)
    else:
        raise AssertionError("missing Stage4 backend metadata incorrectly passed formal audit")
    finally:
        l1_core.backend = original_backend
    # Explicit fallback must never be accepted for a positive-core formal run.
    cfg = copy.deepcopy(load("L1")["model"]); name = cfg.pop("name"); cfg["use_mamba_ssm"] = False
    fallback = build_model(name, **cfg).to(device).eval()
    with torch.inference_mode():
        outputs, captured = capture_stage4_execution(fallback, image)
    try:
        audit_stage4_mamba(
            fallback, variant="fallback", expected_cores=1, outputs=outputs, captured_inputs=captured, formal_backend=True
        )
    except RuntimeError as error:
        assert "mamba_ssm" in str(error)
    else:
        raise AssertionError("fallback Stage4 Mamba incorrectly passed formal audit")
    mamba_cores = [
        core
        for model in models.values()
        for core in getattr(getattr(model, "stage4", None), "route_mamba", torch.nn.Identity()).modules()
        if isinstance(core, SequenceMambaCore)
    ]
    if not torch.cuda.is_available() or any(core.backend != "mamba_ssm" for core in mamba_cores):
        print("SKIP: mamba_ssm CUDA backend unavailable; formal positive-core checks require the server environment")
        return
    for key in ("L1", "L2", "L3", "L4"):
        model = models[key]
        with torch.inference_mode():
            outputs, captured = capture_stage4_execution(model, image)
        audit = audit_stage4_mamba(
            model, variant=key, expected_cores=expected[key], outputs=outputs, captured_inputs=captured, formal_backend=True
        )
        evidence = diagnose_mamba_fused_boundary(model.stage4.route_mamba.raster_mamba if key == "L1" else next(
            module for module in model.stage4.route_mamba.modules() if isinstance(module, SequenceMambaCore)
        ), captured[0][0])
        estimate, method, components = estimate_mamba_inner_flops(model, captured, evidence)
        if estimate is None:
            raise AssertionError(
                f"{key} fused-boundary evidence is not paper-admissible: method={method}; "
                f"status={evidence.get('flops_status')}; boundary={evidence.get('mamba_component_boundary')}; "
                f"unknown={evidence.get('unknown_unsupported_ops')}; unsupported={evidence.get('fvcore_unsupported')}"
            )
        assert audit["mamba_instantiated_core_count"] == expected[key] and estimate > 0.0
        if evidence["mamba_component_boundary"]["in_proj"] == "fvcore_counted_outside_fused":
            assert "in_proj" not in components
            assert evidence["fvcore_counted_component_flops"]["in_proj"] > 0.0
        else:
            assert evidence["mamba_component_boundary"]["in_proj"] == "inside_MambaInnerFn_missing"
    try:
        import fvcore  # noqa: F401
    except ModuleNotFoundError:
        print("SKIP: fvcore unavailable; formal FLOP tracer assertions require the server environment")
        return
    for key, model in models.items():
        with torch.inference_mode():
            _, captured = capture_stage4_execution(model, image)
        lower_bound, unsupported = measure_flops(model, image, inference_path="mask-only")
        if key == "L0":
            estimated, _, _ = estimate_mamba_inner_flops(model, captured, None)
        else:
            core = next(module for module in model.stage4.route_mamba.modules() if isinstance(module, SequenceMambaCore))
            evidence = diagnose_mamba_fused_boundary(core, captured[0][0])
            estimated, _, _ = estimate_mamba_inner_flops(model, captured, evidence)
        if key == "L0":
            assert estimated == 0.0
        else:
            assert estimated is not None and estimated > 0.0 and lower_bound + estimated > lower_bound
            assert any("MambaInnerFn" in operator for operator in unsupported), (key, unsupported)


def test_mamba_component_formula_matches_manual_small_case() -> None:
    if torch is None:
        print("SKIP: torch unavailable; component formula is static")
        return
    core = SequenceMambaCore(4, use_mamba_ssm=False, d_state=3, d_conv=5, expand=2)
    assert _mamba_component_costs(core) == {
        "in_proj": 64, "conv": 40, "x_proj": 56, "dt_proj": 8, "out_proj": 32,
        "selective_scan_and_skip": 160,
    }


def check() -> None:
    test_yaml_registry_and_protocol(); print("PASS: test_yaml_registry_and_protocol")
    test_structure_shared_initialization_and_smoke(); print("PASS: test_structure_shared_initialization_and_smoke")
    test_formal_efficiency_audit_and_mask_only_contract(); print("PASS: test_formal_efficiency_audit_and_mask_only_contract")
    test_mamba_component_formula_matches_manual_small_case(); print("PASS: test_mamba_component_formula_matches_manual_small_case")


if __name__ == "__main__": check()
