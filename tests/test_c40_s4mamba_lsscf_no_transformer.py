"""Regression checks for C40-S4Mamba-LSSCF without the Stage4 Transformer.

Run with: ``python tests/test_c40_s4mamba_lsscf_no_transformer.py``.
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
from models.mamba_blocks import LocalDirectionalCompensation, SequenceMambaCore


C40_CONFIG = ROOT / "configs" / "experiments" / "C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
LSSCF_CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "lightweight_c40_s4mamba_lsscf"
    / "C3_C40_S4MAMBA_LSSCF_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
NOTR_CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "lightweight_c40_s4mamba_lsscf_no_transformer"
    / "C3_C40_S4MAMBA_LSSCF_NOTR_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
ALL_DIRECTIONS = ("horizontal", "vertical", "diagonal", "anti_diagonal")
RUNTIME_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build_from_config(path: Path) -> torch.nn.Module:
    model_cfg = copy.deepcopy(_load_config(path)["model"])
    model_name = model_cfg.pop("name")
    # The direct regression test remains usable on CPU-only hosts while
    # preserving the configured CUDA Mamba topology on actual training hosts.
    if RUNTIME_DEVICE.type != "cuda":
        model_cfg["use_mamba_ssm"] = False
    return build_model(model_name, **model_cfg).to(RUNTIME_DEVICE)


def test_notr_config_changes_only_transformer_switch_and_save_dir() -> None:
    baseline = _load_config(LSSCF_CONFIG)
    candidate = _load_config(NOTR_CONFIG)
    assert {key: value for key, value in candidate.items() if key not in {"model", "save"}} == {
        key: value for key, value in baseline.items() if key not in {"model", "save"}
    }
    expected_model = copy.deepcopy(baseline["model"])
    expected_model["transformer_stages"] = []
    assert candidate["model"] == expected_model
    assert candidate["save"]["save_dir"] == (
        "./runs/pilot/lightweight_c40_s4mamba_lsscf_no_transformer/"
        "C3_C40_S4MAMBA_LSSCF_NOTR_LR150_PHOTO_NATIVE384x544_EMA_BPCM"
    )
    # No S4-LRRM reduction or FR-BRM channel bottleneck is inherited.
    assert "stage4_mamba_spatial_reduction" not in candidate["model"]
    assert "stage4_mamba_channels" not in candidate["model"]


def test_no_transformer_stage4_keeps_full_four_direction_mamba_and_lsscf() -> None:
    torch.manual_seed(42)
    model = _build_from_config(NOTR_CONFIG).train()
    stage4 = model.stage4
    route_mamba = stage4.route_mamba
    assert isinstance(model.stage3, LocalCrackBlock)
    assert not any(isinstance(module, SequenceMambaCore) for module in model.stage3.modules())
    assert isinstance(stage4, TopoRouteBlock)
    assert stage4.channels == 320
    assert isinstance(stage4.global_branch, nn.Identity)
    assert not any(isinstance(module, TransformerGlobalBlock) for module in stage4.modules())
    assert route_mamba.spatial_reduction == 1
    assert route_mamba.mamba_channels == 320
    assert route_mamba.active_mamba_directions == ALL_DIRECTIONS
    assert route_mamba.active_mamba_path_count == len(ALL_DIRECTIONS)
    assert len(route_mamba.paths) == len(ALL_DIRECTIONS)
    assert route_mamba.local_compensations is None
    assert not any(isinstance(module, LocalDirectionalCompensation) for module in stage4.modules())
    assert isinstance(model.decoder.up3, LowResolutionSubpixelSeparableConcatUpBlock)
    assert isinstance(model.decoder.up2, LowResolutionSubpixelSeparableConcatUpBlock)
    assert isinstance(model.decoder.up1, LowResolutionSubpixelSeparableConcatUpBlock)

    path_shapes: list[tuple[int, int, int]] = []
    path_calls = [0, 0, 0, 0]
    identity_inputs: list[torch.Tensor] = []
    identity_outputs: list[torch.Tensor] = []
    fuse_channels: list[int] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_path_hook(index: int):
        def _hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            path_calls[index] += 1
            feature = inputs[0]
            path_shapes.append((feature.shape[1], feature.shape[-2], feature.shape[-1]))

        return _hook

    for index, path in enumerate(route_mamba.paths):
        handles.append(path.register_forward_pre_hook(_make_path_hook(index)))
    handles.append(
        stage4.global_branch.register_forward_pre_hook(
            lambda _module, inputs: identity_inputs.append(inputs[0])
        )
    )
    handles.append(
        stage4.global_branch.register_forward_hook(
            lambda _module, _inputs, output: identity_outputs.append(output)
        )
    )
    handles.append(
        stage4.fuse.register_forward_pre_hook(
            lambda _module, inputs: fuse_channels.append(inputs[0].shape[1])
        )
    )
    image = torch.randn(1, 3, 64, 80, device=RUNTIME_DEVICE)
    try:
        outputs = model(image)
    finally:
        for handle in handles:
            handle.remove()

    stage4_size = tuple(outputs["route_s4"].shape[-2:])
    assert path_calls == [1, 1, 1, 1]
    assert path_shapes == [(320, *stage4_size)] * 4
    assert outputs["route_s4"].shape == (1, 4, *stage4_size)
    assert len(identity_inputs) == len(identity_outputs) == 1
    assert identity_inputs[0].shape == (1, 320, *stage4_size)
    assert identity_outputs[0] is identity_inputs[0]
    # The identity feature remains the third branch; fuse remains 3C -> C.
    assert fuse_channels == [3 * stage4.channels]
    assert stage4.fuse.block[0].in_channels == 3 * stage4.channels
    assert {"mask", "edge", "reliability", "route_s4"}.issubset(outputs)
    assert "route_s3" not in outputs
    assert outputs["mask"].shape == (1, 1, 64, 80)
    loss = outputs["mask"].mean() + outputs["edge"].mean() + outputs["reliability"].mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


def test_mask_only_and_existing_transformer_configurations_remain_intact() -> None:
    model = _build_from_config(NOTR_CONFIG).eval()
    image = torch.randn(1, 3, 64, 80, device=RUNTIME_DEVICE)
    with torch.inference_mode():
        outputs = model.forward_mask_only(image)
    assert set(outputs) == {"mask"}
    assert outputs["mask"].shape == (1, 1, 64, 80)

    lsscf = _build_from_config(LSSCF_CONFIG)
    c40 = _build_from_config(C40_CONFIG)
    assert isinstance(lsscf.stage4.global_branch, TransformerGlobalBlock)
    assert any(isinstance(module, TransformerGlobalBlock) for module in lsscf.stage4.modules())
    assert isinstance(c40.stage4.global_branch, TransformerGlobalBlock)
    assert isinstance(c40.decoder.up3, nn.Module)


def test() -> None:
    tests = (
        test_notr_config_changes_only_transformer_switch_and_save_dir,
        test_no_transformer_stage4_keeps_full_four_direction_mamba_and_lsscf,
        test_mask_only_and_existing_transformer_configurations_remain_intact,
    )
    for check in tests:
        check()
        print(f"PASS: {check.__name__}")


if __name__ == "__main__":
    test()
