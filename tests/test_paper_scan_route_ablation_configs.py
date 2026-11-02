"""Read-only audit for the Crack500 DPSM80 scan × route C0--C3 table.

Run from repository root:
    python tests/test_paper_scan_route_ablation_configs.py

This test never opens the dataset, resolves an Auto Protocol dry-run, creates
an optimizer, executes validation/test, or starts formal training.  CPU runs
exercise the fallback Mamba logic only; real ``mamba_ssm`` checks belong on
demamba21.
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
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    torch = None
    nn = None
    TORCH_AVAILABLE = False
else:
    TORCH_AVAILABLE = True
    from losses import DEMambaLoss
    from models import build_model
    from models.mamba_blocks import DirectionalGridPermutation, DirectionPartitionedSharedMamba, SequenceMambaCore
    from train import validate_dpsm_scan_route_config
    from utils import model_checksum


REGISTRY = ROOT / "configs" / "experiments" / "paper_scan_route_ablation" / "crack500"
M3_PATH = (
    ROOT
    / "configs"
    / "experiments"
    / "paper_dependency_aware_ablation"
    / "crack500"
    / "M3_FrameworkOn_DPSMOn_StructSupOff.yaml"
)
M5_PATH = (
    ROOT
    / "configs"
    / "experiments"
    / "paper_dependency_aware_ablation"
    / "crack500"
    / "M5_FrameworkOn_DPSMOn_StructSupOn.yaml"
)
L3_PATH = ROOT / "configs" / "experiments" / "dpsm_lightweight_ablation" / "crack500" / "L3_DPSM80.yaml"
ROWS = {
    "C0": REGISTRY / "C0_AxialRepeated_Uniform.yaml",
    "C1": REGISTRY / "C1_AxialRepeated_Learned.yaml",
    "C2": REGISTRY / "C2_FourDirection_Uniform.yaml",
    "C3": M3_PATH,
}
EXPECTED_MODES = {
    "C0": ("axial_repeated", "uniform"),
    "C1": ("axial_repeated", "learned"),
    "C2": ("four_direction", "uniform"),
    "C3": ("four_direction", "learned"),
}
EXPECTED_GROUPS = {
    "axial_repeated": ("horizontal", "vertical", "horizontal", "vertical"),
    "four_direction": ("horizontal", "vertical", "diagonal", "anti_diagonal"),
}
EXPECTED_CHECKSUM = "3f7cd9a6ef3e73c2ddb51ceac09d64fbd9353ac336b97fad3cc1990e11246a4a"
STRICT_ROOT = "runs/paper/scan_route_ablation_strict_v2/crack500"
DEVICE = torch.device("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    assert isinstance(config, dict), path
    return config


def _configs() -> dict[str, dict]:
    return {identifier: _load(path) for identifier, path in ROWS.items()}


def _normalized(config: dict) -> dict:
    result = copy.deepcopy(config)
    result["model"].setdefault("dpsm_scan_mode", "four_direction")
    result["model"].setdefault("dpsm_route_mode", "learned")
    return result


def _flatten(value: object, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], object]:
    if isinstance(value, dict):
        result: dict[tuple[str, ...], object] = {}
        for key, child in value.items():
            result.update(_flatten(child, prefix + (str(key),)))
        return result
    return {prefix: value}


def _diff(left: dict, right: dict) -> set[tuple[str, ...]]:
    left_flat, right_flat = _flatten(left), _flatten(right)
    return {key for key in left_flat.keys() | right_flat.keys() if left_flat.get(key) != right_flat.get(key)}


def _without_save(config: dict) -> dict:
    result = _normalized(config)
    result.pop("save", None)
    return result


def _build(config: dict) -> torch.nn.Module:
    assert TORCH_AVAILABLE
    model_cfg = copy.deepcopy(_normalized(config)["model"])
    model_name = model_cfg.pop("name")
    # CPU validates the same topology/logic but is not asserted as real Mamba.
    model_cfg["use_mamba_ssm"] = bool(torch.cuda.is_available())
    return build_model(model_name, **model_cfg).to(DEVICE)


def _criterion(config: dict) -> DEMambaLoss:
    assert TORCH_AVAILABLE
    loss_cfg = copy.deepcopy(config["loss"])
    # Crack500 Auto Protocol V2 resolves this before formal loss construction.
    if loss_cfg.get("mask_pos_weight") == "auto":
        loss_cfg["mask_pos_weight"] = 1.0
    return DEMambaLoss(**loss_cfg).to(DEVICE)


def _batch(height: int = 64, width: int = 96) -> dict[str, torch.Tensor]:
    assert TORCH_AVAILABLE
    direction = torch.randint(0, 4, (1, 1, height, width), device=DEVICE)
    tangent = torch.zeros(1, 4, height, width, device=DEVICE)
    tangent.scatter_(1, direction, 1.0)
    return {
        "image": torch.randn(1, 3, height, width, device=DEVICE),
        "mask": (torch.rand(1, 1, height, width, device=DEVICE) < 0.1).float(),
        "edge_gt": (torch.rand(1, 1, height, width, device=DEVICE) < 0.04).float(),
        "tangent_gt": tangent,
        "tangent_valid": torch.ones(1, 1, height, width, device=DEVICE),
    }


def _state_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor], label: str) -> None:
    assert left.keys() == right.keys(), label
    for key, value in left.items():
        assert torch.equal(value, right[key]), f"{label}: {key}"


def test_yaml_matrix_and_frozen_protocol() -> None:
    assert sorted(path.name for path in REGISTRY.glob("C*.yaml")) == [
        "C0_AxialRepeated_Uniform.yaml",
        "C1_AxialRepeated_Learned.yaml",
        "C2_FourDirection_Uniform.yaml",
    ]
    configs = _configs()
    for identifier, config in configs.items():
        normalized = _normalized(config)
        model, loss, auto = normalized["model"], normalized["loss"], normalized["auto_protocol"]
        expected_scan, expected_route = EXPECTED_MODES[identifier]
        assert model["dpsm_scan_mode"] == expected_scan
        assert model["dpsm_route_mode"] == expected_route
        assert model["base_channels"] == 40 and model["route_bins"] == 4
        assert model["mamba_stages"] == [4] and model["transformer_stages"] == [4]
        assert model["use_transformer"] is True and model["use_direction_partitioned_shared_mamba"] is True
        assert model["stage4_direction_group_channels"] == 80
        assert model["use_low_resolution_subpixel_separable_concat_fusion"] is True
        assert model["use_detail_preserving_early_rebase"] is True and model["use_rsdm_stage1"] is True
        assert loss["lambda_tangent"] == 0.0 and loss["lambda_bpcm"] == 0.0
        assert normalized["data"]["data_root"] == "./Crack500" and normalized["data"]["img_size"] is None
        assert normalized["distillation"]["enabled"] is False
        assert auto["version"] == 2 and auto["enabled"] is True
        assert auto["regimes"][-1]["effective_batch_size"] == 8
        assert auto["regimes"][-1]["target_epochs"] == 64
        assert auto["validation"]["interval_updates"] == 400
        assert auto["early_stopping"]["min_delta"] == 0.0005
    for identifier in ("C0", "C1", "C2"):
        assert configs[identifier]["save"]["save_dir"] == f"{STRICT_ROOT}/{ROWS[identifier].stem}"
    assert (2357 // 8) * 64 == 18_816


def test_normalized_factor_diffs() -> None:
    configs = _configs()
    mechanisms = {
        ("model", "dpsm_scan_mode"),
        ("model", "dpsm_route_mode"),
    }
    shared = _without_save(configs["C0"])
    for identifier in ("C1", "C2", "C3"):
        actual = _diff(shared, _without_save(configs[identifier]))
        assert actual <= mechanisms, (identifier, actual)
    expected_diffs = {
        ("C0", "C1"): {("model", "dpsm_route_mode")},
        ("C0", "C2"): {("model", "dpsm_scan_mode")},
        ("C1", "C3"): {("model", "dpsm_scan_mode")},
        ("C2", "C3"): {("model", "dpsm_route_mode")},
    }
    for (left_id, right_id), expected in expected_diffs.items():
        actual = _diff(_without_save(configs[left_id]), _without_save(configs[right_id]))
        assert actual == expected, f"{left_id}->{right_id}: {actual}"


def test_default_registration_and_initialization_regression() -> None:
    if not TORCH_AVAILABLE:
        print("SKIP: PyTorch unavailable; run DPSM80 construction regression on demamba21.")
        return
    configs = _configs()
    implicit = configs["C3"]
    explicit = _normalized(configs["C3"])
    torch.manual_seed(42)
    implicit_model = _build(implicit).eval()
    torch.manual_seed(42)
    explicit_model = _build(explicit).eval()
    _state_equal(implicit_model.state_dict(), explicit_model.state_dict(), "implicit vs explicit frozen default")
    assert sum(parameter.numel() for parameter in implicit_model.parameters()) == 3_193_751
    image = torch.randn(1, 3, 64, 96, device=DEVICE)
    with torch.inference_mode():
        implicit_output, explicit_output = implicit_model(image), explicit_model(image)
    assert implicit_output.keys() == explicit_output.keys()
    for key, value in implicit_output.items():
        torch.testing.assert_close(value, explicit_output[key], rtol=0.0, atol=0.0)

    # The historic checksum is a CUDA/mamba_ssm regression anchor.  CPU uses
    # the intentional fallback and must not be represented as this evidence.
    if DEVICE.type == "cuda":
        torch.manual_seed(42)
        assert model_checksum(_build(implicit)) == EXPECTED_CHECKSUM
    else:
        print("SKIP: CUDA mamba_ssm checksum anchor requires demamba21; CPU fallback is logic-only.")

    # M5 (loss-only extension) and L3 (efficiency registry) keep the omitted
    # default mechanism fields, and must not change DPSM80 initialization.
    for label, path in (("M5", M5_PATH), ("L3", L3_PATH)):
        torch.manual_seed(42)
        candidate = _build(_load(path)).eval()
        _state_equal(implicit_model.state_dict(), candidate.state_dict(), f"C3/M3 default vs {label}")

    models = {}
    for identifier, config in configs.items():
        torch.manual_seed(3407)
        models[identifier] = _build(config).eval()
    reference = models["C0"].state_dict()
    for identifier, model in models.items():
        _state_equal(reference, model.state_dict(), f"C0 vs {identifier} initialization")
        assert sum(parameter.numel() for parameter in model.parameters()) == 3_193_751


def test_scan_coverage_reversal_and_group_execution() -> None:
    if not TORCH_AVAILABLE:
        print("SKIP: PyTorch unavailable; run scan/inverse checks on demamba21.")
        return
    for height, width in ((8, 12), (7, 11), (5, 5)):
        source = torch.randn(2, 80, height, width, device=DEVICE)
        for direction in DirectionalGridPermutation.directions:
            scan = DirectionalGridPermutation(direction).to(DEVICE)
            order, inverse = scan.indices(height, width, DEVICE)
            assert order.numel() == height * width and torch.unique(order).numel() == height * width
            assert torch.equal(order[inverse], torch.arange(height * width, device=DEVICE))
            torch.testing.assert_close(scan.inverse(scan(source), height, width), source, rtol=0.0, atol=0.0)

    configs = _configs()
    x = torch.randn(2, 320, 8, 12, device=DEVICE)
    for identifier, config in configs.items():
        model = _build(config).eval()
        dpsm = model.stage4.route_mamba
        assert isinstance(dpsm, DirectionPartitionedSharedMamba)
        assert dpsm.group_scan_directions == EXPECTED_GROUPS[EXPECTED_MODES[identifier][0]]
        assert dpsm.group_scan_mapping == {
            f"g{index}": direction for index, direction in enumerate(dpsm.group_scan_directions)
        }
        with torch.inference_mode():
            routed, route_logits = dpsm(x)
        assert routed.shape == x.shape and route_logits.shape == (2, 4, 8, 12)
        assert dpsm.last_packed_shape == (8, 96, 80)
        assert dpsm.last_shared_mamba_call_count == 1 and dpsm.instantiated_core_count == 1
        assert set(dpsm.last_group_sequence_shapes) == {"g0", "g1", "g2", "g3"}
        assert set(dpsm.last_group_restored_shapes) == {"g0", "g1", "g2", "g3"}
        assert all(shape == (2, 96, 80) for shape in dpsm.last_group_sequence_shapes.values())
        assert all(shape == (2, 80, 8, 12) for shape in dpsm.last_group_restored_shapes.values())
        expected_diagnostic_keys = (
            set(DirectionalGridPermutation.directions)
            if EXPECTED_MODES[identifier][0] == "four_direction"
            else {"g0", "g1", "g2", "g3"}
        )
        assert set(dpsm.last_direction_sequence_shapes) == expected_diagnostic_keys
        assert set(dpsm.last_restored_shapes) == expected_diagnostic_keys
        assert sum(isinstance(module, SequenceMambaCore) for module in dpsm.modules()) == 1
        counts = {direction: (scan.reorder_call_count, scan.inverse_call_count) for direction, scan in dpsm.scans.items()}
        if EXPECTED_MODES[identifier][0] == "axial_repeated":
            assert counts["horizontal"] == (2, 2) and counts["vertical"] == (2, 2)
            assert counts["diagonal"] == (0, 0) and counts["anti_diagonal"] == (0, 0)
        else:
            assert all(count == (1, 1) for count in counts.values())

    # Remove pre-mixing only in this direct unit probe.  Independent g2/g3
    # perturbations must reach their own packed slots, not reuse g0/g1 output.
    torch.manual_seed(11)
    probe = DirectionPartitionedSharedMamba(320, nn.Conv2d(320, 4, 1), use_mamba_ssm=False, scan_mode="axial_repeated").to(DEVICE).eval()
    probe.pre_mix = nn.Identity()
    packed_inputs: list[torch.Tensor] = []
    handle = probe.shared_mamba.register_forward_pre_hook(lambda _module, args: packed_inputs.append(args[0].detach().clone()))
    try:
        input0 = torch.randn(1, 320, 8, 12, device=DEVICE)
        with torch.inference_mode():
            probe(input0)
        input2 = input0.clone(); input2[:, 160:240].add_(3.0)
        with torch.inference_mode():
            probe(input2)
        input3 = input0.clone(); input3[:, 240:320].sub_(5.0)
        with torch.inference_mode():
            probe(input3)
    finally:
        handle.remove()
    assert len(packed_inputs) == 3
    torch.testing.assert_close(packed_inputs[0][0:2], packed_inputs[1][0:2], rtol=0.0, atol=0.0)
    assert not torch.equal(packed_inputs[0][2:3], packed_inputs[1][2:3])
    torch.testing.assert_close(packed_inputs[0][3:4], packed_inputs[1][3:4], rtol=0.0, atol=0.0)
    torch.testing.assert_close(packed_inputs[0][0:3], packed_inputs[2][0:3], rtol=0.0, atol=0.0)
    assert not torch.equal(packed_inputs[0][3:4], packed_inputs[2][3:4])


def test_route_semantics_gradients_and_mask_only() -> None:
    if not TORCH_AVAILABLE:
        print("SKIP: PyTorch unavailable; run route/forward/backward checks on demamba21.")
        return
    configs = _configs()
    batch = _batch()
    for identifier, config in configs.items():
        model = _build(config).eval()
        with torch.inference_mode():
            full = model(batch["image"])
            mask_only = model.forward_mask_only(batch["image"])
        assert full["mask"].shape == mask_only["mask"].shape == (1, 1, 64, 96)
        assert full["edge"].shape == (1, 1, 64, 96) and full["route_s4"].shape == (1, 4, 8, 12)
        assert all(torch.isfinite(value).all() for value in full.values())
        torch.testing.assert_close(full["mask"], mask_only["mask"], rtol=1e-5, atol=1e-6)
        loss, items = _criterion(config)(full, batch, epoch=100)
        assert torch.isfinite(loss) and items["tangent_loss"] == 0.0 and items["bpcm_loss"] == 0.0

    # Uniform leaves raw logits live but makes the mask path exactly invariant
    # to route-head parameter changes in eval mode.
    uniform = _build(configs["C0"]).eval()
    with torch.inference_mode():
        before = uniform(batch["image"])
        for parameter in uniform.stage4.route_mamba.route_head.parameters():
            parameter.add_(17.0)
        after = uniform(batch["image"])
    assert not torch.equal(before["route_s4"], after["route_s4"])
    torch.testing.assert_close(before["mask"], after["mask"], rtol=0.0, atol=0.0)

    restored = torch.ones(4, 1, 80, 2, 3, device=DEVICE)
    weights = torch.tensor([[[[0.7]], [[0.2]], [[0.08]], [[0.02]]]], device=DEVICE).expand(1, 4, 2, 3)
    weighted = DirectionPartitionedSharedMamba.apply_route_weights(restored, weights)
    for group in range(4):
        torch.testing.assert_close(weighted[group], torch.full_like(weighted[group], 4.0 * weights[0, group, 0, 0]))
    uniform_weights = torch.full((1, 4, 2, 3), 0.25, device=DEVICE)
    torch.testing.assert_close(
        DirectionPartitionedSharedMamba.apply_route_weights(restored, uniform_weights), restored, rtol=0.0, atol=0.0
    )

    for identifier, expect_route_gradient in (("C0", False), ("C3", True)):
        torch.manual_seed(99)
        model = _build(configs[identifier]).train()
        outputs = model(torch.randn(1, 3, 64, 96, device=DEVICE))
        (outputs["mask"].square().mean() + outputs["edge"].square().mean()).backward()
        dpsm = model.stage4.route_mamba
        route_grads = [parameter.grad for parameter in dpsm.route_head.parameters()]
        has_route_gradient = any(gradient is not None and torch.count_nonzero(gradient).item() > 0 for gradient in route_grads)
        assert has_route_gradient is expect_route_gradient, identifier
        for module, label in ((dpsm.pre_mix, "pre_mix"), (dpsm.shared_mamba, "shared_mamba"), (dpsm.output_mix, "output_mix")):
            gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
            assert gradients and any(gradient is not None and torch.count_nonzero(gradient).item() > 0 for gradient in gradients), label


def test_illegal_scan_route_registrations_fail_before_training() -> None:
    if not TORCH_AVAILABLE:
        print("SKIP: PyTorch unavailable; run constructor/training guards on demamba21.")
        return
    c0 = _configs()["C0"]
    illegal_model_cases = (
        ({"dpsm_scan_mode": "raster"}, "dpsm_scan_mode"),
        ({"dpsm_route_mode": "global"}, "dpsm_route_mode"),
        ({"use_direction_partitioned_shared_mamba": False}, "require"),
        ({"use_direction_partitioned_shared_mamba": False, "use_topology_matched_no_dpsm_stage4": True, "mamba_stages": []}, "NoDPSMTopoBlock"),
        ({"stage4_routed_branch": "single_raster_mamba80"}, "stage4_routed_branch"),
    )
    for overrides, expected in illegal_model_cases:
        invalid = copy.deepcopy(c0)
        invalid["model"].update(overrides)
        try:
            _build(invalid)
        except ValueError as error:
            assert expected.lower() in str(error).lower(), (overrides, error)
        else:
            raise AssertionError(f"illegal scan/route model accepted: {overrides}")
    # R0's four-direction uniform route retains raw logits, so tangent
    # supervision is valid even though mask features use unit-gain weights.
    supervised_four_direction = copy.deepcopy(_configs()["C2"])
    supervised_four_direction["loss"]["lambda_tangent"] = 0.2
    validate_dpsm_scan_route_config(supervised_four_direction)

    invalid_loss = copy.deepcopy(c0)
    invalid_loss["loss"]["lambda_tangent"] = 0.2
    try:
        validate_dpsm_scan_route_config(invalid_loss)
    except ValueError as error:
        assert "axial_repeated" in str(error) and "four_direction" in str(error)
    else:
        raise AssertionError("axial-repeated tangent supervision was accepted")
    invalid_loss = copy.deepcopy(supervised_four_direction)
    invalid_loss["loss"]["lambda_bpcm"] = 0.025
    try:
        validate_dpsm_scan_route_config(invalid_loss)
    except ValueError as error:
        assert "lambda_bpcm=0" in str(error)
    else:
        raise AssertionError("non-default DPSM scan/route BPCM was accepted")
    # Existing M5 is the frozen default mechanism and remains legal.
    validate_dpsm_scan_route_config(_load(M5_PATH))


def test() -> None:
    checks = (
        test_yaml_matrix_and_frozen_protocol,
        test_normalized_factor_diffs,
        test_default_registration_and_initialization_regression,
        test_scan_coverage_reversal_and_group_execution,
        test_route_semantics_gradients_and_mask_only,
        test_illegal_scan_route_registrations_fail_before_training,
    )
    for check in checks:
        check()
        print(f"PASS: {check.__name__}")


if __name__ == "__main__":
    test()
