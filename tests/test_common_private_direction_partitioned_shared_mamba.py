"""Regression tests for isolated DPER-RSDM-CPDPSM64."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from losses import DEMambaLoss
from models import build_model
from models.blocks import RIFTStructuralDirectionalModelingBlock, TopoRouteBlock
from models.decoder import LowResolutionSubpixelSeparableConcatUpBlock
from models.mamba_blocks import (
    CommonPrivateDirectionPartitionedSharedMamba,
    DirectionalGridPermutation,
    DirectionPartitionedSharedMamba,
    SequenceMambaCore,
)
from models.transformer_blocks import TransformerGlobalBlock


DPSM_CONFIG = (
    ROOT / "configs/experiments/lightweight_c40_s4mamba_lsscf_dper_rsdm_dpsm"
    / "C3_C40_S4MAMBA_LSSCF_DPER_RSDM_DPSM80_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
CPDPSM_CONFIG = (
    ROOT / "configs/experiments/lightweight_c40_s4mamba_lsscf_dper_rsdm_cpdpsm"
    / "C3_C40_S4MAMBA_LSSCF_DPER_RSDM_CPDPSM64_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIRECTIONS = ("horizontal", "vertical", "diagonal", "anti_diagonal")


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build(path: Path, **overrides: object) -> nn.Module:
    model_cfg = copy.deepcopy(_load(path)["model"])
    model_cfg.update(overrides)
    name = model_cfg.pop("name")
    if DEVICE.type != "cuda":
        model_cfg["use_mamba_ssm"] = False
    return build_model(name, **model_cfg).to(DEVICE)


def _batch(height: int = 64, width: int = 80) -> dict[str, torch.Tensor]:
    direction = torch.randint(0, 4, (1, 1, height, width), device=DEVICE)
    tangent = torch.zeros(1, 4, height, width, device=DEVICE)
    tangent.scatter_(1, direction, 1.0)
    return {
        "image": torch.randn(1, 3, height, width, device=DEVICE),
        "mask": (torch.rand(1, 1, height, width, device=DEVICE) < 0.08).float(),
        "edge_gt": (torch.rand(1, 1, height, width, device=DEVICE) < 0.04).float(),
        "tangent_gt": tangent,
        "tangent_valid": torch.ones(1, 1, height, width, device=DEVICE),
    }


def test_cpdpsm_config_is_an_isolated_dpsm_replacement() -> None:
    baseline, candidate = _load(DPSM_CONFIG), _load(CPDPSM_CONFIG)
    for section in ("data", "train", "loss", "evaluation"):
        assert candidate[section] == baseline[section], section
    expected = copy.deepcopy(baseline["model"])
    expected.update(
        use_direction_partitioned_shared_mamba=False,
        use_common_private_direction_partitioned_shared_mamba=True,
        stage4_common_direction_channels=64,
        stage4_private_direction_channels=64,
    )
    assert candidate["model"] == expected
    assert candidate["save"] != baseline["save"]
    assert "distillation" not in candidate
    assert candidate["loss"].get("lambda_cfps", 0.0) == 0.0


def test_guards_and_original_dpsm80_compatibility() -> None:
    invalid = (
        ({"use_direction_partitioned_shared_mamba": True}, "mutually exclusive"),
        ({"stage4_common_direction_channels": 32}, "64 common"),
        ({"stage4_private_direction_channels": 80}, "64 common"),
        ({"stage4_mamba_spatial_reduction": 2}, "spatial reduction"),
        # RSDM's pre-existing full-channel guard executes before CP-DPSM's
        # more specific global-bottleneck guard; both reject this input.
        ({"stage4_mamba_channels": 160}, "full-channel"),
        ({"transformer_stages": ()}, "Transformer"),
        ({"use_rsdm_stage1": False}, "RSDM"),
        ({"use_narrow_transport_decoder": True}, "NTD"),
    )
    for override, expected in invalid:
        try:
            _build(CPDPSM_CONFIG, **override)
        except ValueError as error:
            assert expected.lower() in str(error).lower(), (override, error)
        else:
            raise AssertionError(f"CP-DPSM64 guard accepted {override}")

    torch.manual_seed(2401)
    baseline = _build(DPSM_CONFIG).eval()
    torch.manual_seed(2401)
    explicit_false = _build(DPSM_CONFIG, use_common_private_direction_partitioned_shared_mamba=False).eval()
    assert baseline.state_dict().keys() == explicit_false.state_dict().keys()
    assert all(torch.equal(value, explicit_false.state_dict()[name]) for name, value in baseline.state_dict().items())


def test_common_private_structure_route_formula_and_packing() -> None:
    model = _build(CPDPSM_CONFIG).eval()
    assert isinstance(model.stage4, TopoRouteBlock)
    route = model.stage4.route_mamba
    assert isinstance(route, CommonPrivateDirectionPartitionedSharedMamba)
    assert not isinstance(route, DirectionPartitionedSharedMamba)
    assert route.dim == 320 and route.shared_channels == route.private_channels == 64
    assert route.per_direction_channels == 128 and route.shared_channels + 4 * route.private_channels == 320
    assert route.instantiated_core_count == 1
    assert sum(isinstance(module, SequenceMambaCore) for module in route.modules()) == 1
    assert not hasattr(route, "paths") and not hasattr(route, "local_compensations")
    assert isinstance(model.stage4.global_branch, TransformerGlobalBlock)
    assert sum(isinstance(module, RIFTStructuralDirectionalModelingBlock) for module in model.modules()) == 1
    assert all(isinstance(getattr(model.decoder, name), LowResolutionSubpixelSeparableConcatUpBlock) for name in ("up3", "up2", "up1"))
    assert model.decoder.narrow_transport_decoder is None and model.decoder.partial_narrow_transport_decoder is None
    assert not any("teacher" in name.lower() or "cfps" in name.lower() for name in model.state_dict())

    common = torch.stack([torch.full((2, 64, 3, 5), float(index + 1), device=DEVICE) for index in range(4)])
    private = torch.stack([torch.full((2, 64, 3, 5), float(index + 5), device=DEVICE) for index in range(4)])
    weights = torch.softmax(torch.randn(2, 4, 3, 5, device=DEVICE), dim=1)
    fused_common, weighted_private = route.apply_route_weights(common, private, weights)
    expected_common = sum(common[index] * weights[:, index : index + 1] for index in range(4))
    assert torch.allclose(fused_common, expected_common)
    for index in range(4):
        assert torch.equal(weighted_private[index], private[index] * (4.0 * weights[:, index : index + 1]))

    captured: dict[str, torch.Tensor] = {}
    handles = [
        route.pre_mix.register_forward_hook(lambda _module, _inputs, output: captured.setdefault("mixed", output)),
        route.shared_mamba.register_forward_pre_hook(lambda _module, inputs: captured.setdefault("packed", inputs[0])),
    ]
    handles.extend(
        route.scans[direction].register_forward_pre_hook(
            lambda _module, inputs, direction=direction: captured.setdefault(f"scan_{direction}", inputs[0])
        )
        for direction in DIRECTIONS
    )
    try:
        with torch.no_grad():
            feature = torch.randn(2, 320, 5, 7, device=DEVICE)
            output, route_logits = route(feature)
    finally:
        for handle in handles:
            handle.remove()
    assert output.shape == feature.shape and route_logits.shape == (2, 4, 5, 7)
    assert route.last_shared_mamba_call_count == 1
    assert route.last_packed_shape == (8, 35, 128)
    assert tuple(captured["packed"].shape) == (8, 35, 128)
    assert route.last_common_output_shape == (2, 64, 5, 7)
    assert route.last_private_output_shape == (2, 256, 5, 7)
    assert route.last_combined_shape == (2, 320, 5, 7)
    common_input, *private_inputs = captured["mixed"].split((64, 64, 64, 64, 64), dim=1)
    for direction, private_input in zip(DIRECTIONS, private_inputs, strict=True):
        assert route.last_direction_input_shapes[direction] == (2, 128, 5, 7)
        sequence = route.last_direction_sequence_shapes[direction]
        assert sequence == (2, 35, 128)
        assert route.last_restored_shapes[direction] == (2, 128, 5, 7)
        direction_input = captured[f"scan_{direction}"]
        assert torch.equal(direction_input[:, :64], common_input)
        assert torch.equal(direction_input[:, 64:], private_input)
    assert all(route.scans[name].reorder_call_count == 1 and route.scans[name].inverse_call_count == 1 for name in DIRECTIONS)
    if DEVICE.type == "cuda":
        assert model.mamba_backend_info()["stage4"] == ["mamba_ssm"]


def test_reorders_batch_isolation_gradients_and_encoder_contract() -> None:
    for direction in DIRECTIONS:
        scan = DirectionalGridPermutation(direction).to(DEVICE)
        source = torch.arange(2 * 128 * 5 * 7, device=DEVICE).reshape(2, 128, 5, 7)
        sequence = scan(source)
        assert torch.equal(scan.inverse(sequence, 5, 7), source)

    model = _build(CPDPSM_CONFIG).eval()
    route = model.stage4.route_mamba
    assert isinstance(route, CommonPrivateDirectionPartitionedSharedMamba)
    captured: dict[str, torch.Tensor] = {}

    def retain_pre_mix(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        output.retain_grad()
        captured["mixed"] = output

    handle = route.pre_mix.register_forward_hook(retain_pre_mix)
    try:
        batch = _batch()
        outputs = model(batch["image"])
        loss, _ = DEMambaLoss(lambda_bpcm=0.025, bpcm_start_epoch=60, bpcm_ramp_epochs=40)(outputs, batch, epoch=100)
        loss.backward()
    finally:
        handle.remove()
    assert set(outputs) >= {"mask", "edge", "reliability", "route_s4"}
    assert outputs["mask"].shape == batch["mask"].shape
    assert captured["mixed"].grad is not None and torch.isfinite(captured["mixed"].grad).all()
    for group in captured["mixed"].grad.split((64, 64, 64, 64, 64), dim=1):
        assert torch.count_nonzero(group).item() > 0
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in route.shared_mamba.parameters())

    model.eval()
    image = torch.randn(1, 3, 64, 80, device=DEVICE)
    with torch.no_grad():
        full_mask = model(image)["mask"]
        mask_only = model.forward_mask_only(image)
    assert set(mask_only) == {"mask"} and torch.equal(full_mask, mask_only["mask"])
    if DEVICE.type == "cuda":
        amp_model = _build(CPDPSM_CONFIG).train()
        amp_batch = _batch()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            amp_outputs = amp_model(amp_batch["image"])
            amp_loss, _ = DEMambaLoss(lambda_edge=0.0, lambda_tangent=0.0, scnp_weight=0.0)(amp_outputs, amp_batch, epoch=1)
        amp_loss.backward()
        assert torch.isfinite(amp_loss)


def run_all() -> None:
    checks = (
        test_cpdpsm_config_is_an_isolated_dpsm_replacement,
        test_guards_and_original_dpsm80_compatibility,
        test_common_private_structure_route_formula_and_packing,
        test_reorders_batch_isolation_gradients_and_encoder_contract,
    )
    for check in checks:
        check()
        print(f"PASS: {check.__name__}")


if __name__ == "__main__":
    run_all()
