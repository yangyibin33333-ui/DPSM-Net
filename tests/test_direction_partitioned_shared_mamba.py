"""Regression tests for isolated DPER-RSDM-DPSM80."""

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
    DirectionalGridPermutation,
    DirectionPartitionedSharedMamba,
    DirectionalMambaPath,
    SequenceMambaCore,
)
from models.transformer_blocks import TransformerGlobalBlock


RSDM_CONFIG = (
    ROOT / "configs" / "experiments" / "lightweight_c40_s4mamba_lsscf_dper_rsdm"
    / "C3_C40_S4MAMBA_LSSCF_DPER_RSDM_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
DPSM_CONFIG = (
    ROOT / "configs" / "experiments" / "lightweight_c40_s4mamba_lsscf_dper_rsdm_dpsm"
    / "C3_C40_S4MAMBA_LSSCF_DPER_RSDM_DPSM80_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
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


def _assert_gradient(module: nn.Module, label: str) -> None:
    gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
    assert gradients and all(gradient is not None for gradient in gradients), f"{label}: missing gradient"
    assert all(torch.isfinite(gradient).all() for gradient in gradients), f"{label}: non-finite gradient"
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients), f"{label}: zero gradient"


def test_dpsm_config_is_a_strict_rsdm_single_variable_change() -> None:
    baseline, candidate = _load(RSDM_CONFIG), _load(DPSM_CONFIG)
    assert {key: value for key, value in candidate.items() if key not in {"model", "save"}} == {
        key: value for key, value in baseline.items() if key not in {"model", "save"}
    }
    expected = copy.deepcopy(baseline["model"])
    expected.update(use_direction_partitioned_shared_mamba=True, stage4_direction_group_channels=80)
    assert candidate["model"] == expected
    assert "rsdm_dpsm" in candidate["save"]["save_dir"]


def test_scan_orders_are_unique_and_inverse_on_non_square_grids() -> None:
    source = torch.arange(2 * 80 * 5 * 7, device=DEVICE).reshape(2, 80, 5, 7)
    for direction in DIRECTIONS:
        scan = DirectionalGridPermutation(direction).to(DEVICE)
        order, inverse = scan.indices(5, 7, DEVICE)
        assert order.shape == inverse.shape == (35,)
        assert torch.equal(torch.sort(order).values, torch.arange(35, device=DEVICE))
        sequence = scan(source)
        restored = scan.inverse(sequence, 5, 7)
        assert torch.equal(source, restored), direction
        assert scan.reorder_call_count == scan.inverse_call_count == 1


def test_dpsm_guards_and_original_rsdm_compatibility() -> None:
    invalid = (
        ({"stage4_direction_group_channels": 64}, "four 80-channel"),
        ({"stage4_mamba_spatial_reduction": 2}, "spatial reduction"),
        ({"stage4_mamba_channels": 160}, "full-channel"),
        ({"transformer_stages": ()}, "Transformer"),
        ({"stage4_active_mamba_directions": ("horizontal", "vertical")}, "four-direction"),
        ({"use_multiscale_directional_reallocation": True}, "MSDR"),
        ({"use_narrow_transport_decoder": True}, "NTD"),
        ({"use_partial_narrow_transport_decoder": True}, "PNTD"),
        ({"use_depthwise_separable_mask_head": True}, "DSMH"),
    )
    for override, expected in invalid:
        try:
            _build(DPSM_CONFIG, **override)
        except ValueError as error:
            assert expected.lower() in str(error).lower(), (override, error)
        else:
            raise AssertionError(f"DPSM guard accepted {override}")

    torch.manual_seed(1700)
    baseline = _build(RSDM_CONFIG).eval()
    torch.manual_seed(1700)
    explicit_false = _build(RSDM_CONFIG, use_direction_partitioned_shared_mamba=False).eval()
    assert baseline.state_dict().keys() == explicit_false.state_dict().keys()
    assert all(torch.equal(value, explicit_false.state_dict()[key]) for key, value in baseline.state_dict().items())


def test_dpsm_supports_the_framework_off_stage4_topology() -> None:
    model = _build(
        DPSM_CONFIG,
        use_low_resolution_subpixel_separable_concat_fusion=False,
        use_detail_preserving_early_rebase=False,
        use_rsdm_stage1=False,
    ).eval()
    assert isinstance(model.stage4, TopoRouteBlock)
    dpsm = model.stage4.route_mamba
    assert isinstance(dpsm, DirectionPartitionedSharedMamba)
    assert dpsm.dim == 320 and dpsm.group_channels == 80
    assert dpsm.instantiated_core_count == 1
    assert model.rsdm_stage1 is None


def test_dpsm_structure_registration_and_shared_initialization() -> None:
    torch.manual_seed(1701)
    baseline = _build(RSDM_CONFIG).eval()
    torch.manual_seed(1701)
    model = _build(DPSM_CONFIG).eval()
    assert isinstance(model.stage4, TopoRouteBlock)
    dpsm = model.stage4.route_mamba
    assert isinstance(dpsm, DirectionPartitionedSharedMamba)
    assert dpsm.dim == 320 and dpsm.group_channels == 80
    assert dpsm.active_mamba_directions == DIRECTIONS
    assert dpsm.spatial_reduction == 1 and dpsm.mamba_channels == 320
    assert dpsm.instantiated_core_count == 1
    assert sum(isinstance(module, SequenceMambaCore) for module in dpsm.modules()) == 1
    assert not any(isinstance(module, DirectionalMambaPath) for module in dpsm.modules())
    assert not hasattr(dpsm, "paths") and not hasattr(dpsm, "local_compensations")
    assert not hasattr(dpsm, "input_projection") and not hasattr(dpsm, "out_proj")
    assert isinstance(model.stage4.global_branch, TransformerGlobalBlock)
    assert model.stage4.local is not None and model.stage4.fuse is not None and model.stage4.refine is not None
    assert isinstance(model.rsdm_stage1, RIFTStructuralDirectionalModelingBlock)
    assert sum(isinstance(module, RIFTStructuralDirectionalModelingBlock) for module in model.modules()) == 1
    assert isinstance(model.decoder.up3, LowResolutionSubpixelSeparableConcatUpBlock)
    assert isinstance(model.decoder.up2, LowResolutionSubpixelSeparableConcatUpBlock)
    assert isinstance(model.decoder.up1, LowResolutionSubpixelSeparableConcatUpBlock)
    assert model.decoder.narrow_transport_decoder is None
    assert model.decoder.partial_narrow_transport_decoder is None
    dense_mask = model.decoder.mask_head[0].block[0]
    assert dense_mask.in_channels == dense_mask.out_channels == 40
    assert dense_mask.kernel_size == (3, 3) and dense_mask.groups == 1

    baseline_state, candidate_state = baseline.state_dict(), model.state_dict()
    for key, value in baseline_state.items():
        if not key.startswith("stage4.route_mamba.") or key.startswith("stage4.route_mamba.route_head."):
            assert key in candidate_state and torch.equal(value, candidate_state[key]), key
    forbidden = (
        "stage4.route_mamba.paths.",
        "stage4.route_mamba.input_projection.",
        "stage4.route_mamba.out_proj.",
        "stage4.route_mamba.local_compensations.",
    )
    assert not any(key.startswith(forbidden) for key in candidate_state)
    assert sum(parameter.numel() for parameter in model.parameters()) < sum(
        parameter.numel() for parameter in baseline.parameters()
    )
    if DEVICE.type == "cuda":
        assert sum(parameter.numel() for parameter in model.parameters()) <= 5_500_000
        assert model.mamba_backend_info()["stage4"] == ["mamba_ssm"]


def test_route_modulation_is_groupwise_and_has_no_global_bottleneck() -> None:
    restored = torch.stack(
        [torch.full((2, 80, 3, 5), float(index + 1), device=DEVICE) for index in range(4)], dim=0
    )
    weights = torch.softmax(torch.randn(2, 4, 3, 5, device=DEVICE), dim=1)
    weighted = DirectionPartitionedSharedMamba.apply_route_weights(restored, weights)
    assert weighted.shape == (4, 2, 80, 3, 5)
    for index in range(4):
        expected = restored[index] * (4.0 * weights[:, index : index + 1])
        assert torch.equal(weighted[index], expected)
    changed = weights.clone()
    changed[:, 2] = changed[:, 2] + 0.25
    changed_output = DirectionPartitionedSharedMamba.apply_route_weights(restored, changed)
    assert torch.equal(weighted[0], changed_output[0])
    assert torch.equal(weighted[1], changed_output[1])
    assert not torch.equal(weighted[2], changed_output[2])
    assert torch.equal(weighted[3], changed_output[3])


def test_dpsm_forward_packing_calls_and_batch_isolation() -> None:
    torch.manual_seed(1702)
    model = _build(DPSM_CONFIG).eval()
    dpsm = model.stage4.route_mamba
    assert isinstance(dpsm, DirectionPartitionedSharedMamba)
    observed: dict[str, object] = {"shared_calls": 0, "transformer_calls": 0}

    def shared_hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
        observed["shared_calls"] = int(observed["shared_calls"]) + 1
        observed["packed"] = tuple(inputs[0].shape)

    handles = [
        dpsm.shared_mamba.register_forward_hook(shared_hook),
        model.stage4.global_branch.register_forward_hook(
            lambda _module, _inputs, _output: observed.__setitem__(
                "transformer_calls", int(observed["transformer_calls"]) + 1
            )
        ),
    ]
    stage4_input = torch.randn(2, 320, 7, 9, device=DEVICE)
    try:
        with torch.inference_mode():
            stage4_output, route_logits = model.stage4(stage4_input)
    finally:
        for handle in handles:
            handle.remove()
    assert stage4_output.shape == (2, 320, 7, 9)
    assert route_logits.shape == (2, 4, 7, 9)
    assert observed["packed"] == (8, 63, 80)
    assert dpsm.last_packed_shape == (8, 63, 80)
    assert dpsm.last_shared_mamba_call_count == 1
    assert observed["shared_calls"] == 1
    assert observed["transformer_calls"] == 1

    # CUDA Mamba kernels may choose numerically different implementations for
    # batch=8 and batch=4, so comparing the real core across batch sizes is not
    # a valid sample-isolation test. Keep the exact production scan/pack/
    # unpack/route path, and substitute only a deterministic per-sequence core
    # that provably has no batch reduction. Equality then detects any indexing
    # or reshape that mixes samples or directions.
    class _PerSequenceAffine(nn.Module):
        def forward(self, sequence: torch.Tensor) -> torch.Tensor:
            return sequence * 1.25 + 0.5

    real_core = dpsm.shared_mamba
    dpsm.shared_mamba = _PerSequenceAffine().to(DEVICE)
    alternate_input = stage4_input.clone()
    alternate_input[1] = torch.randn_like(alternate_input[1])
    try:
        with torch.inference_mode():
            isolated_batch, _ = dpsm(stage4_input)
            isolated_alternate, _ = dpsm(alternate_input)
    finally:
        dpsm.shared_mamba = real_core
    assert torch.equal(isolated_batch[:1], isolated_alternate[:1])
    assert dpsm.last_packed_shape == (8, 63, 80)
    assert dpsm.last_shared_mamba_call_count == 1
    for direction in DIRECTIONS:
        scan = dpsm.scans[direction]
        assert scan.reorder_call_count == scan.inverse_call_count == 3


def test_full_model_contract_mask_identity_and_backward_gradients() -> None:
    torch.manual_seed(1703)
    model = _build(DPSM_CONFIG).eval()
    image = torch.randn(1, 3, 64, 80, device=DEVICE)
    with torch.inference_mode():
        full = model(image)
        mask_only = model.forward_mask_only(image)
    assert {"mask", "edge", "reliability", "route_s4"}.issubset(full)
    assert full["mask"].shape == full["edge"].shape == full["reliability"].shape == (1, 1, 64, 80)
    assert full["route_s4"].shape == (1, 4, 8, 10)
    assert set(mask_only) == {"mask"}
    assert torch.equal(full["mask"], mask_only["mask"])
    dpsm = model.stage4.route_mamba
    assert dpsm.last_packed_shape == (4, 80, 80)
    assert dpsm.last_shared_mamba_call_count == 1

    model.train()
    batch = _batch()
    captured: dict[str, torch.Tensor] = {}

    def capture_mixed(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        output.retain_grad()
        captured["mixed"] = output

    handle = dpsm.pre_mix.register_forward_hook(capture_mixed)
    try:
        outputs = model(batch["image"])
        loss, _ = DEMambaLoss(**_load(DPSM_CONFIG)["loss"]).to(DEVICE)(outputs, batch, epoch=0)
        assert torch.isfinite(loss)
        loss.backward()
    finally:
        handle.remove()
    mixed_grad = captured["mixed"].grad
    assert mixed_grad is not None and torch.isfinite(mixed_grad).all()
    assert all(torch.count_nonzero(group).item() > 0 for group in mixed_grad.split(80, dim=1))
    for module, label in (
        (dpsm.pre_mix, "pre_mix"),
        (dpsm.shared_mamba, "shared_mamba"),
        (dpsm.output_mix, "output_mix"),
        (dpsm.route_head, "route_head"),
        (model.stage4.global_branch, "Transformer"),
        (model.stage4.local, "local"),
        (model.stage4.fuse, "fuse"),
        (model.stage4.refine, "refine"),
        (model.rsdm_stage1, "RSDM"),
    ):
        _assert_gradient(module, label)


def test() -> None:
    checks = (
        test_dpsm_config_is_a_strict_rsdm_single_variable_change,
        test_scan_orders_are_unique_and_inverse_on_non_square_grids,
        test_dpsm_guards_and_original_rsdm_compatibility,
        test_dpsm_supports_the_framework_off_stage4_topology,
        test_dpsm_structure_registration_and_shared_initialization,
        test_route_modulation_is_groupwise_and_has_no_global_bottleneck,
        test_dpsm_forward_packing_calls_and_batch_isolation,
        test_full_model_contract_mask_identity_and_backward_gradients,
    )
    for check in checks:
        check()
        print(f"PASS: {check.__name__}")


if __name__ == "__main__":
    test()
