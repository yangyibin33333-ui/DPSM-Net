"""Audit the frozen Crack500 Framework × DPSM × StructSup registry.

Run from the repository root with:
    python tests/test_paper_three_factor_ablation_configs.py

The configuration audit is read-only.  The small model-equivalence probe
explicitly disables mamba_ssm, uses no dataset, and never executes an
optimizer update or Auto Protocol dry-run.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - depends on the local validation environment
    torch = None
    TORCH_AVAILABLE = False
else:
    TORCH_AVAILABLE = True
    from models import build_model
    from models.blocks import TopoRouteBlock
    from models.mamba_blocks import DirectionPartitionedSharedMamba

PROBE_DEVICE = torch.device("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None


REGISTRY = ROOT / "configs" / "experiments" / "paper_three_factor_ablation" / "crack500"
OLD = ROOT / "configs" / "experiments" / "paper_main_table_a" / "crack500"
ROWS = {
    "G0": "G0_FrameworkOff_DPSMOff_StructSupOff",
    "G1": "G1_FrameworkOn_DPSMOff_StructSupOff",
    "G2": "G2_FrameworkOff_DPSMOn_StructSupOff",
    "G3": "G3_FrameworkOff_DPSMOff_StructSupOn",
    "G4": "G4_FrameworkOn_DPSMOn_StructSupOff",
    "G5": "G5_FrameworkOn_DPSMOff_StructSupOn",
    "G6": "G6_FrameworkOff_DPSMOn_StructSupOn",
    "G7": "G7_FrameworkOn_DPSMOn_StructSupOn",
}
OLD_EQUIVALENTS = {
    "G3": "A1_C40_S4",
    "G5": "A4_C40_S4_LSSCF_DPER_RSDM",
    "G7": "A5_C40_S4_LSSCF_DPER_RSDM_DPSM80",
}
FRAMEWORK_FIELDS = {
    ("model", "use_low_resolution_subpixel_separable_concat_fusion"),
    ("model", "use_detail_preserving_early_rebase"),
    ("model", "use_rsdm_stage1"),
}
DPSM_FIELD = {("model", "use_direction_partitioned_shared_mamba")}
STRUCT_SUP_FIELDS = {("loss", "lambda_tangent"), ("loss", "lambda_bpcm")}
FAILED_MODEL_FLAGS = (
    "use_common_private_direction_partitioned_shared_mamba",
    "use_full_width_residual_mask_head",
    "use_difference_aware_up1_fusion",
    "use_depthwise_separable_mask_head",
    "use_narrow_transport_decoder",
    "use_partial_narrow_transport_decoder",
    "use_subpixel_phase_consistency_refiner",
    "use_haar_topology_preserving_rebase",
    "use_full_resolution_joint_exchange",
    "use_early_serpentine_connectivity_ssm",
    "use_partial_full_resolution_stage1",
    "use_multiscale_directional_reallocation",
    "use_semantic_conditioned_local_kernel_residual",
    "use_residual_branch_competition_fusion",
)


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    assert isinstance(loaded, dict), path
    return loaded


def _row(identifier: str) -> dict:
    return _load(REGISTRY / f"{ROWS[identifier]}.yaml")


def _flatten(value: object, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], object]:
    if isinstance(value, dict):
        flattened: dict[tuple[str, ...], object] = {}
        for key, child in value.items():
            flattened.update(_flatten(child, prefix + (str(key),)))
        return flattened
    return {prefix: value}


def _diff(left: dict, right: dict) -> set[tuple[str, ...]]:
    left_flat, right_flat = _flatten(left), _flatten(right)
    return {key for key in left_flat.keys() | right_flat.keys() if left_flat.get(key) != right_flat.get(key)}


def _without_save(config: dict) -> dict:
    value = copy.deepcopy(config)
    value.pop("save", None)
    return value


def _assert_only_differences(left: dict, right: dict, allowed: set[tuple[str, ...]], label: str) -> None:
    actual = _diff(_without_save(left), _without_save(right))
    assert actual == allowed, f"{label}: expected {sorted(allowed)}, got {sorted(actual)}"


def _build(config: dict) -> torch.nn.Module:
    assert TORCH_AVAILABLE
    model_cfg = copy.deepcopy(config["model"])
    name = model_cfg.pop("name")
    # CPU-only environments use the repository fallback solely for topology and
    # equality.  demamba21 uses its real CUDA mamba_ssm backend and additionally
    # checks the frozen parameter regressions below.
    model_cfg["use_mamba_ssm"] = bool(torch.cuda.is_available())
    return build_model(name, **model_cfg).to(PROBE_DEVICE).eval()


def test_yaml_protocol_and_factor_matrix() -> None:
    configs = {identifier: _row(identifier) for identifier in ROWS}
    for identifier, config in configs.items():
        assert config["data"]["data_root"] == "./Crack500", identifier
        assert config["data"]["img_size"] is None, identifier
        assert config["save"]["save_dir"] == f"runs/paper/three_factor_ablation/crack500/{ROWS[identifier]}", identifier
        assert config["model"]["base_channels"] == 40
        assert config["model"]["mamba_stages"] == [4]
        assert config["model"]["transformer_stages"] == [4]
        assert config["model"]["route_bins"] == 4
        assert config["model"]["stage4_direction_group_channels"] == 80
        assert all(config["model"].get(flag, False) is False for flag in FAILED_MODEL_FLAGS), identifier
        assert config["distillation"]["enabled"] is False
        auto = config["auto_protocol"]
        assert auto["version"] == 2 and auto["enabled"] is True
        assert auto["native_input"] == {"enabled": True, "align_multiple": 32, "max_pixels": 409600, "size_statistic": "mode"}
        assert auto["regimes"][-1]["name"] == "xlarge"
        assert auto["regimes"][-1]["effective_batch_size"] == 8
        assert auto["regimes"][-1]["target_epochs"] == 64
        assert auto["loss_phases"] == {
            "use_update_clock": True,
            "scnp_warmup_ratio": 0.05,
            "bpcm_start_ratio": 0.30,
            "bpcm_ramp_ratio": 0.20,
        }
        assert auto["validation"]["interval_updates"] == 400
        assert auto["early_stopping"] == {
            "enabled": True,
            "selection_metric": "structural_score",
            "min_training_ratio": 0.55,
            "patience_ratio": 0.15,
            "min_delta": 0.0005,
        }
    # Frozen Crack500 resolved targets: 2,352 train samples / effective batch 8 × 64.
    target_updates = 2352 // 8 * 64
    assert target_updates == 18816
    assert round(target_updates * 0.05) == 941
    assert round(target_updates * 0.30) == 5645
    assert round(target_updates * 0.20) == 3763
    assert int(target_updates * 0.55 + 0.999999) == 10349
    assert int(target_updates * 0.15 + 0.999999) == 2823

    for identifier in ("G0", "G2", "G3", "G6"):
        model = configs[identifier]["model"]
        assert model["use_low_resolution_subpixel_separable_concat_fusion"] is False
        assert model["use_detail_preserving_early_rebase"] is False
        assert model["use_rsdm_stage1"] is False
    for identifier in ("G1", "G4", "G5", "G7"):
        model = configs[identifier]["model"]
        assert model["use_low_resolution_subpixel_separable_concat_fusion"] is True
        assert model["use_detail_preserving_early_rebase"] is True
        assert model["use_rsdm_stage1"] is True
    for identifier in ("G0", "G1", "G3", "G5"):
        assert configs[identifier]["model"]["use_direction_partitioned_shared_mamba"] is False
    for identifier in ("G2", "G4", "G6", "G7"):
        assert configs[identifier]["model"]["use_direction_partitioned_shared_mamba"] is True
    for identifier in ("G0", "G1", "G2", "G4"):
        assert configs[identifier]["loss"]["lambda_tangent"] == 0.0
        assert configs[identifier]["loss"]["lambda_bpcm"] == 0.0
    for identifier in ("G3", "G5", "G6", "G7"):
        assert configs[identifier]["loss"]["lambda_tangent"] == 0.2
        assert configs[identifier]["loss"]["lambda_bpcm"] == 0.025


def test_registered_differences_and_old_equivalence() -> None:
    configs = {identifier: _row(identifier) for identifier in ROWS}
    for identifier, old_stem in OLD_EQUIVALENTS.items():
        old = _load(OLD / f"{old_stem}.yaml")
        assert _diff(_without_save(configs[identifier]), _without_save(old)) == set(), identifier

    for left, right in (("G0", "G3"), ("G1", "G5"), ("G2", "G6"), ("G4", "G7")):
        _assert_only_differences(configs[left], configs[right], STRUCT_SUP_FIELDS, f"{left}->{right}")
    for left, right in (("G0", "G1"), ("G2", "G4"), ("G3", "G5"), ("G6", "G7")):
        _assert_only_differences(configs[left], configs[right], FRAMEWORK_FIELDS, f"{left}->{right}")
    for left, right in (("G0", "G2"), ("G1", "G4"), ("G3", "G6"), ("G5", "G7")):
        _assert_only_differences(configs[left], configs[right], DPSM_FIELD, f"{left}->{right}")

    g7 = configs["G7"]
    d1 = _load(REGISTRY / "loss_2x2" / "D1_TangentOn_BPCMOff.yaml")
    d2 = _load(REGISTRY / "loss_2x2" / "D2_TangentOff_BPCMOn.yaml")
    _assert_only_differences(d1, g7, {("loss", "lambda_bpcm")}, "D1->G7")
    _assert_only_differences(d2, g7, {("loss", "lambda_tangent")}, "D2->G7")
    assert configs["G4"]["loss"]["lambda_tangent"] == configs["G4"]["loss"]["lambda_bpcm"] == 0.0
    assert g7["loss"]["lambda_tangent"] == 0.2 and g7["loss"]["lambda_bpcm"] == 0.025


def test_model_topology_parameter_pairs_and_g7_a5_equivalence() -> None:
    if not TORCH_AVAILABLE:
        print("SKIP: PyTorch unavailable; YAML/diff audit completed, run the topology probe on demamba21.")
        return
    configs = {identifier: _row(identifier) for identifier in ROWS}
    models = {identifier: _build(config) for identifier, config in configs.items()}
    parameter_counts = {identifier: sum(parameter.numel() for parameter in model.parameters()) for identifier, model in models.items()}
    assert parameter_counts["G0"] == parameter_counts["G3"]
    assert parameter_counts["G1"] == parameter_counts["G5"]
    assert parameter_counts["G4"] == parameter_counts["G7"]
    assert parameter_counts["G2"] == parameter_counts["G6"]
    if PROBE_DEVICE.type == "cuda":
        assert parameter_counts["G3"] == 6_152_767
        assert parameter_counts["G5"] == 5_747_511
        assert parameter_counts["G7"] == 3_193_751
    for identifier in ("G0", "G1", "G3", "G5"):
        assert isinstance(models[identifier].stage4, TopoRouteBlock)
        assert len(models[identifier].mamba_backend_info()["stage4"]) == 4
        if PROBE_DEVICE.type == "cuda":
            assert models[identifier].mamba_backend_info()["stage4"] == ["mamba_ssm"] * 4
    for identifier in ("G2", "G4", "G6", "G7"):
        assert isinstance(models[identifier].stage4, TopoRouteBlock)
        dpsm = models[identifier].stage4.route_mamba
        assert isinstance(dpsm, DirectionPartitionedSharedMamba)
        assert dpsm.dim == 320 and dpsm.group_channels == 80 and dpsm.instantiated_core_count == 1
        assert len(models[identifier].mamba_backend_info()["stage4"]) == 1
        if PROBE_DEVICE.type == "cuda":
            assert models[identifier].mamba_backend_info()["stage4"] == ["mamba_ssm"]

    old_a5 = _load(OLD / "A5_C40_S4_LSSCF_DPER_RSDM_DPSM80.yaml")
    torch.manual_seed(2908)
    expected = _build(old_a5)
    torch.manual_seed(2908)
    candidate = _build(configs["G7"])
    expected_state, candidate_state = expected.state_dict(), candidate.state_dict()
    assert expected_state.keys() == candidate_state.keys()
    assert all(torch.equal(value, candidate_state[key]) for key, value in expected_state.items())
    image = torch.randn(1, 3, 64, 96, device=PROBE_DEVICE)
    with torch.inference_mode():
        expected_output, candidate_output = expected(image), candidate(image)
    assert expected_output.keys() == candidate_output.keys()
    for key, value in expected_output.items():
        torch.testing.assert_close(value, candidate_output[key], rtol=1e-5, atol=1e-6)
    assert candidate_output["mask"].shape == candidate_output["edge"].shape == (1, 1, 64, 96)
    assert candidate_output["route_s4"].shape == (1, 4, 8, 12)
    print(f"{PROBE_DEVICE.type} topology parameter counts:", parameter_counts)


def test() -> None:
    checks = (
        test_yaml_protocol_and_factor_matrix,
        test_registered_differences_and_old_equivalence,
        test_model_topology_parameter_pairs_and_g7_a5_equivalence,
    )
    for check in checks:
        check()
        print(f"PASS: {check.__name__}")


if __name__ == "__main__":
    test()
