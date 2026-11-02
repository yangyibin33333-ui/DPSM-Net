"""Read-only audit for the formal TUT dependency-aware confirmation.

Run from repository root:
    python tests/test_paper_dependency_aware_ablation_tut_configs.py

No dataloader, Auto Protocol dry-run, optimizer update, validation, test split,
or formal training is used here.
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
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    torch = None
    TORCH_AVAILABLE = False
else:
    TORCH_AVAILABLE = True
    from models import build_model
    from models.blocks import NoDPSMTopoBlock, TopoRouteBlock
    from models.mamba_blocks import DirectionPartitionedSharedMamba, SequenceMambaCore
    from train import validate_no_dpsm_topology_config


REGISTRY = ROOT / "configs" / "experiments" / "paper_dependency_aware_ablation" / "tut"
CRACK500 = ROOT / "configs" / "experiments" / "paper_dependency_aware_ablation" / "crack500"
ROWS = {
    "T0": ("T0_FrameworkOff_DPSMOff_StructSupOff", "M0_FrameworkOff_DPSMOff_StructSupOff"),
    "T1": ("T1_FrameworkOn_DPSMOff_StructSupOff", "M1_FrameworkOn_DPSMOff_StructSupOff"),
    "T2": ("T2_FrameworkOn_DPSMOn_StructSupOff", "M3_FrameworkOn_DPSMOn_StructSupOff"),
    "T3": ("T3_FrameworkOn_DPSMOn_StructSupOn", "M5_FrameworkOn_DPSMOn_StructSupOn"),
}
SAVE_ROOT = "runs/paper/dependency_aware_ablation_strict_v2/tut"
FRAMEWORK_FIELDS = {
    ("model", "use_low_resolution_subpixel_separable_concat_fusion"),
    ("model", "use_detail_preserving_early_rebase"),
    ("model", "use_rsdm_stage1"),
}
DPSM_FIELDS = {
    ("model", "mamba_stages"),
    ("model", "use_topology_matched_no_dpsm_stage4"),
    ("model", "use_direction_partitioned_shared_mamba"),
}
STRUCT_SUP_FIELDS = {("loss", "lambda_tangent"), ("loss", "lambda_bpcm")}
PROBE_DEVICE = torch.device("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    assert isinstance(config, dict), path
    return config


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
    result = copy.deepcopy(config)
    result.pop("save", None)
    return result


def _row(identifier: str) -> dict:
    return _load(REGISTRY / f"{ROWS[identifier][0]}.yaml")


def _build(config: dict):
    assert TORCH_AVAILABLE
    model_config = copy.deepcopy(config["model"])
    name = model_config.pop("name")
    # CPU smoke tests use the existing fallback only. The YAML still requires
    # mamba_ssm, and the CUDA branch below verifies it on demamba21.
    model_config["use_mamba_ssm"] = bool(torch.cuda.is_available())
    return build_model(name, **model_config).to(PROBE_DEVICE).eval()


def _only(left: dict, right: dict, allowed: set[tuple[str, ...]], label: str) -> None:
    actual = _diff(_without_save(left), _without_save(right))
    assert actual == allowed, f"{label}: expected {sorted(allowed)}, got {sorted(actual)}"


def test_registry_mapping_and_frozen_protocol() -> None:
    expected_files = sorted(name for name, _ in ROWS.values())
    assert sorted(path.stem for path in REGISTRY.glob("T*.yaml")) == expected_files
    configs = {identifier: _row(identifier) for identifier in ROWS}
    common = copy.deepcopy(configs["T0"])
    for section in ("model", "loss", "save"):
        common.pop(section)
    for identifier, config in configs.items():
        non_factor = copy.deepcopy(config)
        for section in ("model", "loss", "save"):
            non_factor.pop(section)
        assert non_factor == common, identifier
        model, loss, auto = config["model"], config["loss"], config["auto_protocol"]
        framework_on = identifier in {"T1", "T2", "T3"}
        dpsm_on = identifier in {"T2", "T3"}
        struct_on = identifier == "T3"
        assert config["data"]["data_root"] == "./TUT" and config["data"]["img_size"] is None
        assert model["base_channels"] == 40 and model["route_bins"] == 4
        assert model["transformer_stages"] == [4] and model["use_transformer"] is True
        assert model["use_mamba_ssm"] is True and model["stage4_direction_group_channels"] == 80
        for key in FRAMEWORK_FIELDS:
            assert bool(model[key[1]]) is framework_on
        assert bool(model["use_direction_partitioned_shared_mamba"]) is dpsm_on
        assert bool(model["use_topology_matched_no_dpsm_stage4"]) is (not dpsm_on)
        assert model["mamba_stages"] == ([4] if dpsm_on else [])
        assert loss["lambda_tangent"] == (0.2 if struct_on else 0.0)
        assert loss["lambda_bpcm"] == (0.025 if struct_on else 0.0)
        assert not (not dpsm_on and loss["lambda_tangent"] > 0.0)
        assert config["distillation"]["enabled"] is False
        flat_keys = {".".join(key).lower() for key in _flatten(config)}
        assert not any(token in key for key in flat_keys for token in (
            "resume", "pretrained", "initial_weights", "init_checkpoint", "load_checkpoint"))
        assert config["save"]["save_dir"] == f"{SAVE_ROOT}/{ROWS[identifier][0]}"
        assert "stage4_order_ablation" not in config["save"]["save_dir"]
        assert auto["version"] == 2 and auto["enabled"] is True and auto["dry_run"] is False
        large = next(regime for regime in auto["regimes"] if regime["name"] == "large")
        assert large["effective_batch_size"] == 8 and large["target_epochs"] == 75
        assert auto["loss_phases"] == {"use_update_clock": True, "scnp_warmup_ratio": 0.05, "bpcm_start_ratio": 0.30, "bpcm_ramp_ratio": 0.20}
        assert auto["validation"]["interval_updates"] == 400
        assert auto["early_stopping"] == {"enabled": True, "selection_metric": "structural_score", "min_training_ratio": 0.55, "patience_ratio": 0.15, "min_delta": 0.0005}
    assert (987 // 8) * 75 == 9225


def test_mapping_and_registered_factor_diffs() -> None:
    configs = {identifier: _row(identifier) for identifier in ROWS}
    for identifier, (_, source_name) in ROWS.items():
        source = _load(CRACK500 / f"{source_name}.yaml")
        expected = copy.deepcopy(source)
        expected["data"]["data_root"] = "./TUT"
        expected["save"] = configs[identifier]["save"]
        assert configs[identifier] == expected, f"{identifier} no longer maps exactly to {source_name}"
    _only(configs["T0"], configs["T1"], FRAMEWORK_FIELDS, "T0->T1")
    _only(configs["T1"], configs["T2"], DPSM_FIELDS, "T1->T2")
    _only(configs["T2"], configs["T3"], STRUCT_SUP_FIELDS, "T2->T3")


def test_stage4_topology_and_rsdm_dependency() -> None:
    if not TORCH_AVAILABLE:
        print("SKIP: PyTorch unavailable; run topology checks on demamba21.")
        return
    models = {identifier: _build(_row(identifier)) for identifier in ROWS}
    for identifier in ("T0", "T1"):
        model = models[identifier]
        assert isinstance(model.stage4, NoDPSMTopoBlock)
        assert model.route_stage4 is False and model.mamba_backend_info() == {}
        assert sum(isinstance(module, SequenceMambaCore) for module in model.modules()) == 0
        assert not hasattr(model.stage4, "route_mamba") and not hasattr(model.stage4, "route_head")
        with torch.inference_mode():
            outputs = model(torch.randn(1, 3, 64, 96, device=PROBE_DEVICE))
        assert outputs.get("route_s4") is None
    # The frozen model declares the optional slot as ``None`` before deciding
    # whether to register the real block; absence is not the enabled-state
    # contract.
    assert models["T0"].rsdm_stage1 is None
    for identifier in ("T1", "T2", "T3"):
        assert models[identifier].rsdm_stage1 is not None, identifier
    for identifier in ("T2", "T3"):
        model = models[identifier]
        routed = model.stage4.route_mamba
        assert isinstance(model.stage4, TopoRouteBlock)
        assert isinstance(routed, DirectionPartitionedSharedMamba)
        assert model.stage4.hybrid_order == "parallel"
        assert routed.group_channels == 80 and routed.instantiated_core_count == 1
        assert tuple(routed.directions) == ("horizontal", "vertical", "diagonal", "anti_diagonal")
        with torch.inference_mode():
            outputs = model(torch.randn(1, 3, 64, 96, device=PROBE_DEVICE))
        assert outputs["route_s4"].shape == (1, 4, 8, 12)
        assert routed.last_packed_shape == (4, 96, 80)
        assert routed.last_shared_mamba_call_count == 1
        if PROBE_DEVICE.type == "cuda":
            assert model.mamba_backend_info()["stage4"] == ["mamba_ssm"]
    illegal = copy.deepcopy(_row("T0"))
    illegal["loss"]["lambda_tangent"] = 0.2
    try:
        validate_no_dpsm_topology_config(illegal)
    except ValueError as error:
        assert "route_s4" in str(error)
    else:
        raise AssertionError("NoDPSMTopoBlock accepted tangent supervision")


def test() -> None:
    for check in (test_registry_mapping_and_frozen_protocol, test_mapping_and_registered_factor_diffs, test_stage4_topology_and_rsdm_dependency):
        check()
        print(f"PASS: {check.__name__}")


if __name__ == "__main__":
    test()
