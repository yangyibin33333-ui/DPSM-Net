"""Regression checks for full-resolution bottleneck routed Mamba (FR-BRM).

Run with: ``python tests/test_stage4_full_resolution_bottleneck_routed_mamba.py``.
"""

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

from models import build_model
from models.blocks import LocalCrackBlock, TopoRouteBlock, TransformerGlobalBlock
from models.decoder import LowResolutionSubpixelSeparableConcatUpBlock
from models.mamba_blocks import TangentRoutedMambaBlock


C40_CONFIG = ROOT / "configs" / "experiments" / "C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
S4MAMBA_CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "lightweight_c40_s4mamba"
    / "C3_C40_S4MAMBA_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
LSAF_CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "lightweight_c40_s4mamba_lsaf"
    / "C3_C40_S4MAMBA_LSAF_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
LSSCF_CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "lightweight_c40_s4mamba_lsscf"
    / "C3_C40_S4MAMBA_LSSCF_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
FRBRM_CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "lightweight_c40_s4mamba_lsscf_frbrm"
    / "C3_C40_S4MAMBA_LSSCF_FRBRM160_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
ALL_DIRECTIONS = ("horizontal", "vertical", "diagonal", "anti_diagonal")
RUNTIME_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build_from_config(path: Path) -> torch.nn.Module:
    model_cfg = copy.deepcopy(_load_config(path)["model"])
    model_name = model_cfg.pop("name")
    if RUNTIME_DEVICE.type != "cuda":
        model_cfg["use_mamba_ssm"] = False
    return build_model(model_name, **model_cfg).to(RUNTIME_DEVICE)


def _fallback_block(*, mamba_channels: int | None = None) -> TangentRoutedMambaBlock:
    return TangentRoutedMambaBlock(32, use_mamba_ssm=False, mamba_channels=mamba_channels)


def test_default_and_full_channel_modes_are_bitwise_compatible() -> None:
    torch.manual_seed(11)
    default = _fallback_block().eval()
    torch.manual_seed(11)
    explicit_none = _fallback_block(mamba_channels=None).eval()
    torch.manual_seed(11)
    explicit_full = _fallback_block(mamba_channels=32).eval()
    assert isinstance(default.input_projection, nn.Identity)
    assert isinstance(explicit_full.input_projection, nn.Identity)
    assert set(default.state_dict()) == set(explicit_none.state_dict()) == set(explicit_full.state_dict())
    strict_result = explicit_full.load_state_dict(default.state_dict(), strict=True)
    assert not strict_result.missing_keys and not strict_result.unexpected_keys

    image = torch.randn(1, 32, 7, 9)
    with torch.inference_mode():
        default_output = default(image)
        none_output = explicit_none(image)
        full_output = explicit_full(image)
    assert torch.equal(default_output[0], none_output[0])
    assert torch.equal(default_output[1], none_output[1])
    assert torch.equal(default_output[0], full_output[0])
    assert torch.equal(default_output[1], full_output[1])


def test_bottleneck_block_uses_all_input_channels_and_four_full_resolution_paths() -> None:
    block = TangentRoutedMambaBlock(320, use_mamba_ssm=False, mamba_channels=160).eval()
    assert isinstance(block.input_projection, nn.Sequential)
    assert isinstance(block.input_projection[0], nn.Conv2d)
    assert block.input_projection[0].in_channels == 320
    assert block.input_projection[0].out_channels == 160
    assert isinstance(block.out_proj[0], nn.Conv2d)
    assert block.out_proj[0].in_channels == 160
    assert block.out_proj[0].out_channels == 320
    assert block.spatial_reduction == 1
    assert block.active_mamba_directions == ALL_DIRECTIONS
    assert block.active_mamba_path_count == 4
    assert block.local_compensations is None
    assert all(path.core.norm.normalized_shape == (160,) for path in block.paths)

    route_head_channels: list[int] = []
    projection_channels: list[int] = []
    path_shapes: list[tuple[int, int, int]] = []
    calls = [0, 0, 0, 0]
    handles: list[torch.utils.hooks.RemovableHandle] = []
    handles.append(
        block.route_head.register_forward_pre_hook(
            lambda _module, inputs: route_head_channels.append(inputs[0].shape[1])
        )
    )
    handles.append(
        block.input_projection.register_forward_pre_hook(
            lambda _module, inputs: projection_channels.append(inputs[0].shape[1])
        )
    )

    def make_path_hook(index: int):
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            calls[index] += 1
            feature = inputs[0]
            path_shapes.append((feature.shape[1], feature.shape[-2], feature.shape[-1]))

        return hook

    for index, path in enumerate(block.paths):
        handles.append(path.register_forward_pre_hook(make_path_hook(index)))
    image = torch.randn(1, 320, 7, 9)
    try:
        with torch.inference_mode():
            routed, route_logits = block(image)
    finally:
        for handle in handles:
            handle.remove()
    assert route_head_channels == [320]
    assert projection_channels == [320]
    assert calls == [1, 1, 1, 1]
    assert path_shapes == [(160, 7, 9)] * 4
    assert route_logits.shape == (1, 4, 7, 9)
    assert routed.shape == (1, 320, 7, 9)


def test_bottleneck_and_spatial_reduction_are_mutually_exclusive() -> None:
    try:
        TangentRoutedMambaBlock(320, use_mamba_ssm=False, mamba_channels=160, spatial_reduction=2)
    except ValueError as error:
        assert "cannot be combined" in str(error)
    else:
        raise AssertionError("FR-BRM and spatial reduction must be mutually exclusive.")


def test_frbrm_config_is_single_variable_change() -> None:
    baseline = _load_config(LSSCF_CONFIG)
    candidate = _load_config(FRBRM_CONFIG)
    assert {key: value for key, value in candidate.items() if key not in {"model", "save"}} == {
        key: value for key, value in baseline.items() if key not in {"model", "save"}
    }
    expected_model = copy.deepcopy(baseline["model"])
    expected_model["stage4_mamba_spatial_reduction"] = 1
    expected_model["stage4_mamba_channels"] = 160
    assert candidate["model"] == expected_model
    assert candidate["save"]["save_dir"] == (
        "./runs/pilot/lightweight_c40_s4mamba_lsscf_frbrm/"
        "C3_C40_S4MAMBA_LSSCF_FRBRM160_LR150_PHOTO_NATIVE384x544_EMA_BPCM"
    )


def test_frbrm_model_contract_and_lsscf_decoder_stability() -> None:
    model = _build_from_config(FRBRM_CONFIG).train()
    assert isinstance(model.stage3, LocalCrackBlock)
    assert isinstance(model.stage4, TopoRouteBlock)
    route_mamba = model.stage4.route_mamba
    assert route_mamba.mamba_channels == 160
    assert route_mamba.spatial_reduction == 1
    assert route_mamba.active_mamba_directions == ALL_DIRECTIONS
    assert route_mamba.local_compensations is None
    assert isinstance(model.stage4.global_branch, TransformerGlobalBlock)
    assert isinstance(model.decoder.up3, LowResolutionSubpixelSeparableConcatUpBlock)

    local_sizes: list[tuple[int, int]] = []
    transformer_sizes: list[tuple[int, int]] = []
    mamba_shapes: list[tuple[int, int, int]] = []
    handles = [
        model.stage4.local.register_forward_pre_hook(
            lambda _module, inputs: local_sizes.append(tuple(inputs[0].shape[-2:]))
        ),
        model.stage4.global_branch.register_forward_pre_hook(
            lambda _module, inputs: transformer_sizes.append(tuple(inputs[0].shape[-2:]))
        ),
    ]
    for path in route_mamba.paths:
        handles.append(
            path.register_forward_pre_hook(
                lambda _module, inputs: mamba_shapes.append(
                    (inputs[0].shape[1], inputs[0].shape[-2], inputs[0].shape[-1])
                )
            )
        )
    try:
        image = torch.randn(1, 3, 64, 80, device=RUNTIME_DEVICE)
        outputs = model(image)
    finally:
        for handle in handles:
            handle.remove()
    stage4_size = tuple(outputs["route_s4"].shape[-2:])
    assert local_sizes == [stage4_size]
    assert transformer_sizes == [stage4_size]
    assert mamba_shapes == [(160, *stage4_size)] * 4
    assert {"mask", "edge", "reliability", "route_s4"}.issubset(outputs)
    assert "route_s3" not in outputs
    assert outputs["mask"].shape == (1, 1, 64, 80)
    loss = outputs["mask"].mean() + outputs["edge"].mean() + outputs["reliability"].mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)

    model.eval()
    with torch.inference_mode():
        mask_only = model.forward_mask_only(image)
    assert set(mask_only) == {"mask"}
    assert mask_only["mask"].shape == (1, 1, 64, 80)


def test_existing_configurations_keep_full_mamba_channels() -> None:
    for config_path in (C40_CONFIG, S4MAMBA_CONFIG, LSAF_CONFIG, LSSCF_CONFIG):
        model = _build_from_config(config_path)
        assert model.stage4.route_mamba.mamba_channels == 320
        assert isinstance(model.stage4.route_mamba.input_projection, nn.Identity)


def test() -> None:
    tests = (
        test_default_and_full_channel_modes_are_bitwise_compatible,
        test_bottleneck_block_uses_all_input_channels_and_four_full_resolution_paths,
        test_bottleneck_and_spatial_reduction_are_mutually_exclusive,
        test_frbrm_config_is_single_variable_change,
        test_frbrm_model_contract_and_lsscf_decoder_stability,
        test_existing_configurations_keep_full_mamba_channels,
    )
    for check in tests:
        check()
        print(f"PASS: {check.__name__}")


if __name__ == "__main__":
    test()
