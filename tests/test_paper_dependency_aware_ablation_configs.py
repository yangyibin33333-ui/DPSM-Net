"""Read-only audit for Crack500 strict_v2 dependency-aware ablations.

Run from repository root:
    python tests/test_paper_dependency_aware_ablation_configs.py

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
    from losses import DEMambaLoss
    from models import build_model
    from models.blocks import NoDPSMTopoBlock, TopoRouteBlock, TransformerGlobalBlock
    from models.mamba_blocks import DirectionPartitionedSharedMamba, SequenceMambaCore
    from train import validate_no_dpsm_topology_config
    from utils import model_checksum


REGISTRY = ROOT / "configs" / "experiments" / "paper_dependency_aware_ablation" / "crack500"
OLD_A5 = ROOT / "configs" / "experiments" / "paper_main_table_a" / "crack500" / "A5_C40_S4_LSSCF_DPER_RSDM_DPSM80.yaml"
ROWS = {
    "M0": "M0_FrameworkOff_DPSMOff_StructSupOff",
    "M1": "M1_FrameworkOn_DPSMOff_StructSupOff",
    "M2": "M2_FrameworkOff_DPSMOn_StructSupOff",
    "M3": "M3_FrameworkOn_DPSMOn_StructSupOff",
    "M4": "M4_FrameworkOff_DPSMOn_StructSupOn",
    "M5": "M5_FrameworkOn_DPSMOn_StructSupOn",
}
STRICT_ROOT = "runs/paper/dependency_aware_ablation_strict_v2/crack500"
FRAMEWORK_FIELDS = {
    ("model", "use_low_resolution_subpixel_separable_concat_fusion"),
    ("model", "use_detail_preserving_early_rebase"),
    ("model", "use_rsdm_stage1"),
}
DPSM_FACTOR_FIELDS = {
    ("model", "mamba_stages"),
    ("model", "use_topology_matched_no_dpsm_stage4"),
    ("model", "use_direction_partitioned_shared_mamba"),
}
STRUCT_SUP_FIELDS = {("loss", "lambda_tangent"), ("loss", "lambda_bpcm")}
EXPECTED_DPSM_CHECKSUMS = {
    "M2": "052f1ab0155641d2c9ab4b0eb953f0fc51fcb1785eaeea2beedbca14c6f6085f",
    "M3": "3f7cd9a6ef3e73c2ddb51ceac09d64fbd9353ac336b97fad3cc1990e11246a4a",
}
PROBE_DEVICE = torch.device("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    assert isinstance(value, dict), path
    return value


def _row(identifier: str) -> dict:
    return _load(REGISTRY / f"{ROWS[identifier]}.yaml")


def _flatten(value: object, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], object]:
    if isinstance(value, dict):
        result: dict[tuple[str, ...], object] = {}
        for key, child in value.items():
            result.update(_flatten(child, prefix + (str(key),)))
        return result
    return {prefix: value}


def _without_save(config: dict, *, permit_new_default: bool = False) -> dict:
    result = copy.deepcopy(config)
    result.pop("save", None)
    if permit_new_default and result.get("model", {}).get("use_topology_matched_no_dpsm_stage4") is False:
        result["model"].pop("use_topology_matched_no_dpsm_stage4")
    return result


def _diff(left: dict, right: dict) -> set[tuple[str, ...]]:
    left_flat, right_flat = _flatten(left), _flatten(right)
    return {key for key in left_flat.keys() | right_flat.keys() if left_flat.get(key) != right_flat.get(key)}


def _only(left: dict, right: dict, allowed: set[tuple[str, ...]], label: str) -> None:
    actual = _diff(_without_save(left), _without_save(right))
    assert actual == allowed, f"{label}: expected {sorted(allowed)}, got {sorted(actual)}"


def _build(config: dict) -> torch.nn.Module:
    assert TORCH_AVAILABLE
    model_cfg = copy.deepcopy(config["model"])
    name = model_cfg.pop("name")
    model_cfg["use_mamba_ssm"] = bool(torch.cuda.is_available())
    return build_model(name, **model_cfg).to(PROBE_DEVICE).eval()


def _criterion(config: dict) -> DEMambaLoss:
    assert TORCH_AVAILABLE
    loss_cfg = copy.deepcopy(config["loss"])
    # Crack500 xlarge V2 resolves this to its pre-registered fixed value.
    if loss_cfg.get("mask_pos_weight") == "auto":
        loss_cfg["mask_pos_weight"] = 1.0
    return DEMambaLoss(**loss_cfg).to(PROBE_DEVICE)


def _batch(height: int = 64, width: int = 96) -> dict[str, torch.Tensor]:
    assert TORCH_AVAILABLE
    direction = torch.randint(0, 4, (1, 1, height, width), device=PROBE_DEVICE)
    tangent = torch.zeros(1, 4, height, width, device=PROBE_DEVICE)
    tangent.scatter_(1, direction, 1.0)
    return {
        "image": torch.randn(1, 3, height, width, device=PROBE_DEVICE),
        "mask": (torch.rand(1, 1, height, width, device=PROBE_DEVICE) < 0.1).float(),
        "edge_gt": (torch.rand(1, 1, height, width, device=PROBE_DEVICE) < 0.04).float(),
        "tangent_gt": tangent,
        "tangent_valid": torch.ones(1, 1, height, width, device=PROBE_DEVICE),
    }


def _assert_protocol(config: dict, identifier: str) -> None:
    model, loss, auto = config["model"], config["loss"], config["auto_protocol"]
    framework_on = identifier in {"M1", "M3", "M5"}
    dpsm_on = identifier in {"M2", "M3", "M4", "M5"}
    struct_sup_on = identifier in {"M4", "M5"}
    strict_no_dpsm = identifier in {"M0", "M1"}
    assert model["base_channels"] == 40 and model["route_bins"] == 4
    assert model["transformer_stages"] == [4] and model["use_transformer"] is True
    assert model["stage4_direction_group_channels"] == 80 and model["use_mamba_ssm"] is True
    assert bool(model["use_low_resolution_subpixel_separable_concat_fusion"]) is framework_on
    assert bool(model["use_detail_preserving_early_rebase"]) is framework_on
    assert bool(model["use_rsdm_stage1"]) is framework_on
    assert bool(model["use_topology_matched_no_dpsm_stage4"]) is strict_no_dpsm
    assert bool(model["use_direction_partitioned_shared_mamba"]) is dpsm_on
    assert model["mamba_stages"] == ([4] if dpsm_on else [])
    assert loss["lambda_tangent"] == (0.2 if struct_sup_on else 0.0)
    assert loss["lambda_bpcm"] == (0.025 if struct_sup_on else 0.0)
    assert strict_no_dpsm or dpsm_on
    assert config["data"]["data_root"] == "./Crack500" and config["data"]["img_size"] is None
    assert config["distillation"]["enabled"] is False
    assert auto["version"] == 2 and auto["enabled"] is True
    assert auto["regimes"][-1]["effective_batch_size"] == 8
    assert auto["regimes"][-1]["target_epochs"] == 64
    assert auto["loss_phases"] == {
        "use_update_clock": True,
        "scnp_warmup_ratio": 0.05,
        "bpcm_start_ratio": 0.30,
        "bpcm_ramp_ratio": 0.20,
    }
    assert auto["validation"]["interval_updates"] == 400
    assert auto["early_stopping"]["min_delta"] == 0.0005
    assert config["save"]["save_dir"] == f"{STRICT_ROOT}/{ROWS[identifier]}"


def test_registry_protocol_and_dependency_rules() -> None:
    active_yaml = sorted(path.stem for path in REGISTRY.glob("M*.yaml"))
    assert active_yaml == sorted(ROWS.values())
    configs = {identifier: _row(identifier) for identifier in ROWS}
    for identifier, config in configs.items():
        _assert_protocol(config, identifier)
    d1 = _load(REGISTRY / "loss_2x2" / "D1_TangentOn_BPCMOff.yaml")
    d2 = _load(REGISTRY / "loss_2x2" / "D2_TangentOff_BPCMOn.yaml")
    assert d1["save"]["save_dir"] == f"{STRICT_ROOT}/loss_2x2/D1_TangentOn_BPCMOff"
    assert d2["save"]["save_dir"] == f"{STRICT_ROOT}/loss_2x2/D2_TangentOff_BPCMOn"
    assert d1["model"]["use_topology_matched_no_dpsm_stage4"] is False
    assert d2["model"]["use_topology_matched_no_dpsm_stage4"] is False
    # Frozen Crack500 V2 update clock.
    assert (2357 // 8) * 64 == 18816
    assert round(18816 * 0.05) == 941 and round(18816 * 0.30) == 5645 and round(18816 * 0.20) == 3763

    if TORCH_AVAILABLE:
        illegal = copy.deepcopy(configs["M0"])
        illegal["loss"]["lambda_tangent"] = 0.2
        try:
            validate_no_dpsm_topology_config(illegal)
        except ValueError as error:
            assert "route_s4" in str(error)
        else:
            raise AssertionError("NoDPSMTopoBlock tangent supervision was accepted")
        illegal["loss"].update(lambda_tangent=0.0, lambda_bpcm=0.025)
        try:
            validate_no_dpsm_topology_config(illegal)
        except ValueError as error:
            assert "StructSup" in str(error)
        else:
            raise AssertionError("NoDPSMTopoBlock BPCM StructSup was accepted")


def test_registered_factor_diffs_and_dpsm_regression_contract() -> None:
    configs = {identifier: _row(identifier) for identifier in ROWS}
    _only(configs["M0"], configs["M1"], FRAMEWORK_FIELDS, "M0->M1")
    _only(configs["M0"], configs["M2"], DPSM_FACTOR_FIELDS, "M0->M2")
    _only(configs["M1"], configs["M3"], DPSM_FACTOR_FIELDS, "M1->M3")
    _only(configs["M2"], configs["M4"], STRUCT_SUP_FIELDS, "M2->M4")
    _only(configs["M3"], configs["M5"], STRUCT_SUP_FIELDS, "M3->M5")
    old_a5 = _load(OLD_A5)
    assert _diff(_without_save(configs["M5"], permit_new_default=True), _without_save(old_a5)) == set()
    d1 = _load(REGISTRY / "loss_2x2" / "D1_TangentOn_BPCMOff.yaml")
    d2 = _load(REGISTRY / "loss_2x2" / "D2_TangentOff_BPCMOn.yaml")
    _only(d1, configs["M3"], {("loss", "lambda_tangent")}, "M3->D1")
    _only(d2, configs["M3"], {("loss", "lambda_bpcm")}, "M3->D2")
    assert d1["loss"]["lambda_tangent"] == 0.2 and d1["loss"]["lambda_bpcm"] == 0.0
    assert d2["loss"]["lambda_tangent"] == 0.0 and d2["loss"]["lambda_bpcm"] == 0.025


def test_no_dpsm_model_guards() -> None:
    if not TORCH_AVAILABLE:
        print("SKIP: PyTorch unavailable; run NoDPSMTopoBlock guards on demamba21.")
        return
    config = _row("M0")
    invalid_overrides = (
        ({"use_direction_partitioned_shared_mamba": True}, "mutually exclusive"),
        ({"use_common_private_direction_partitioned_shared_mamba": True}, "mutually exclusive"),
        ({"mamba_stages": [4]}, "mamba_stages"),
        ({"use_transformer": False}, "TransformerGlobalBlock"),
        ({"transformer_stages": []}, "TransformerGlobalBlock"),
    )
    for overrides, expected in invalid_overrides:
        invalid = copy.deepcopy(config)
        invalid["model"].update(overrides)
        try:
            _build(invalid)
        except ValueError as error:
            assert expected.lower() in str(error).lower(), (overrides, error)
        else:
            raise AssertionError(f"NoDPSMTopoBlock accepted invalid override: {overrides}")


def _assert_shared_initialization(left: torch.nn.Module, right: torch.nn.Module, label: str) -> None:
    left_state, right_state = left.state_dict(), right.state_dict()
    common = [key for key in left_state.keys() & right_state.keys() if not key.startswith("stage4.fuse")]
    assert common, label
    for key in common:
        assert torch.equal(left_state[key], right_state[key]), f"{label}: shared tensor changed: {key}"
    for prefix in ("stem.", "stage1.", "stage2.", "stage3.", "decoder.", "stage4.local.", "stage4.global_branch.", "stage4.refine."):
        assert any(key.startswith(prefix) for key in common), f"{label}: missing shared prefix {prefix}"


def test_stage4_topology_parameter_registration_and_initialization() -> None:
    if not TORCH_AVAILABLE:
        print("SKIP: PyTorch unavailable; run topology checks on demamba21.")
        return
    configs = {identifier: _row(identifier) for identifier in ROWS}
    models = {identifier: _build(config) for identifier, config in configs.items()}
    counts = {identifier: sum(parameter.numel() for parameter in model.parameters()) for identifier, model in models.items()}
    for identifier in ("M0", "M1"):
        model, state = models[identifier], models[identifier].state_dict()
        assert isinstance(model.stage4, NoDPSMTopoBlock)
        assert model.route_stage4 is False and model.mamba_backend_info() == {}
        assert isinstance(model.stage4.global_branch, TransformerGlobalBlock)
        assert model.stage4.fuse.block[0].in_channels == 640 and model.stage4.fuse.block[0].out_channels == 320
        assert not any(token in key.lower() for key in state for token in ("route_mamba", "route_head", "shared_mamba", "mamba"))
        assert sum(isinstance(module, SequenceMambaCore) for module in model.modules()) == 0
    for identifier in ("M2", "M3", "M4", "M5"):
        model = models[identifier]
        assert isinstance(model.stage4, TopoRouteBlock) and model.route_stage4 is True
        assert isinstance(model.stage4.route_mamba, DirectionPartitionedSharedMamba)
        assert model.stage4.route_mamba.group_channels == 80
        assert model.stage4.route_mamba.instantiated_core_count == 1
        assert len(model.mamba_backend_info()["stage4"]) == 1
        if PROBE_DEVICE.type == "cuda":
            assert model.mamba_backend_info()["stage4"] == ["mamba_ssm"]
    assert counts["M0"] == 3_010_523
    assert counts["M1"] == 2_605_267
    assert counts["M2"] == counts["M4"] == 3_599_007
    assert counts["M3"] == counts["M5"] == 3_193_751

    for left_id, right_id in (("M0", "M2"), ("M1", "M3")):
        torch.manual_seed(2909)
        left = _build(configs[left_id])
        torch.manual_seed(2909)
        right = _build(configs[right_id])
        _assert_shared_initialization(left, right, f"{left_id}<->{right_id}")
    print(f"{PROBE_DEVICE.type} strict_v2 parameter counts: {counts}")


def test_dpsm_on_state_output_and_checksum_regression() -> None:
    if not TORCH_AVAILABLE:
        print("SKIP: PyTorch unavailable; run DPSM regression checks on demamba21.")
        return
    configs = {identifier: _row(identifier) for identifier in ROWS}
    d1 = _load(REGISTRY / "loss_2x2" / "D1_TangentOn_BPCMOff.yaml")
    d2 = _load(REGISTRY / "loss_2x2" / "D2_TangentOff_BPCMOn.yaml")
    for group in ((configs["M2"], configs["M4"]), (configs["M3"], configs["M5"], d1, d2)):
        states = []
        for config in group:
            torch.manual_seed(42)
            states.append(_build(config).state_dict())
        assert all(states[0].keys() == state.keys() for state in states[1:])
        for key, value in states[0].items():
            assert all(torch.equal(value, state[key]) for state in states[1:]), key
    old_a5 = _load(OLD_A5)
    torch.manual_seed(42)
    expected = _build(old_a5)
    torch.manual_seed(42)
    candidate = _build(configs["M5"])
    assert expected.state_dict().keys() == candidate.state_dict().keys()
    assert all(torch.equal(value, candidate.state_dict()[key]) for key, value in expected.state_dict().items())
    image = torch.randn(1, 3, 64, 96, device=PROBE_DEVICE)
    with torch.inference_mode():
        expected_out, candidate_out = expected(image), candidate(image)
    assert expected_out.keys() == candidate_out.keys()
    for key, value in expected_out.items():
        torch.testing.assert_close(value, candidate_out[key], rtol=1e-5, atol=1e-6)
    if PROBE_DEVICE.type == "cuda":
        for identifier in ("M2", "M3"):
            torch.manual_seed(42)
            assert model_checksum(_build(configs[identifier])) == EXPECTED_DPSM_CHECKSUMS[identifier]


def test_forward_and_loss_semantics() -> None:
    if not TORCH_AVAILABLE:
        print("SKIP: PyTorch unavailable; run forward/loss checks on demamba21.")
        return
    configs = {identifier: _row(identifier) for identifier in ROWS}
    d1 = _load(REGISTRY / "loss_2x2" / "D1_TangentOn_BPCMOff.yaml")
    d2 = _load(REGISTRY / "loss_2x2" / "D2_TangentOff_BPCMOn.yaml")
    batch = _batch()
    for identifier in ROWS:
        outputs = _build(configs[identifier])(batch["image"])
        loss, items = _criterion(configs[identifier])(outputs, batch, epoch=100)
        assert torch.isfinite(loss) and torch.isfinite(outputs["mask"]).all(), identifier
        if identifier in {"M0", "M1"}:
            assert outputs.get("route_s4") is None
            assert items["tangent_loss"] == 0.0 and items["bpcm_loss"] == 0.0
        elif identifier in {"M2", "M3"}:
            assert outputs["route_s4"].shape == (1, 4, 8, 12)
            assert items["tangent_loss"] == 0.0 and items["bpcm_loss"] == 0.0
        else:
            assert outputs["route_s4"].shape == (1, 4, 8, 12)
            assert items["tangent_loss"] > 0.0 and items["bpcm_loss"] > 0.0
    for label, config, tangent_on, bpcm_on in (("D1", d1, True, False), ("D2", d2, False, True)):
        outputs = _build(config)(batch["image"])
        loss, items = _criterion(config)(outputs, batch, epoch=100)
        assert torch.isfinite(loss), label
        assert (items["tangent_loss"] > 0.0) is tangent_on, label
        assert (items["bpcm_loss"] > 0.0) is bpcm_on, label


def test() -> None:
    checks = (
        test_registry_protocol_and_dependency_rules,
        test_registered_factor_diffs_and_dpsm_regression_contract,
        test_no_dpsm_model_guards,
        test_stage4_topology_parameter_registration_and_initialization,
        test_dpsm_on_state_output_and_checksum_regression,
        test_forward_and_loss_semantics,
    )
    for check in checks:
        check()
        print(f"PASS: {check.__name__}")


if __name__ == "__main__":
    test()
