"""Regression checks for the DPER early-encoder ablation.

Run with: ``python tests/test_detail_preserving_early_rebase.py``.
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
import models.blocks as block_modules
from models.blocks import (
    CNNEdgeStem,
    DetailPreservingHalfStem,
    HalfResolutionStage1Block,
    LocalCrackBlock,
    SubpixelDetailReconstruction,
    TopoRouteBlock,
    TransformerGlobalBlock,
)
from models.decoder import LowResolutionSubpixelSeparableConcatUpBlock


LSSCF_CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "lightweight_c40_s4mamba_lsscf"
    / "C3_C40_S4MAMBA_LSSCF_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
DPER_CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "lightweight_c40_s4mamba_lsscf_dper"
    / "C3_C40_S4MAMBA_LSSCF_DPER_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
ALL_DIRECTIONS = ("horizontal", "vertical", "diagonal", "anti_diagonal")
RUNTIME_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build(path: Path, **overrides: object) -> torch.nn.Module:
    model_cfg = copy.deepcopy(_load_config(path)["model"])
    model_cfg.update(overrides)
    model_name = model_cfg.pop("name")
    # Selective scan is CUDA-only.  CPU regression runs retain the four-path
    # topology but exercise the repository's established fallback backend.
    if RUNTIME_DEVICE.type != "cuda":
        model_cfg["use_mamba_ssm"] = False
    return build_model(model_name, **model_cfg).to(RUNTIME_DEVICE)


def _assert_equal_state(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> None:
    assert set(left) == set(right)
    for key in left:
        assert torch.equal(left[key], right[key]), key


def test_dper_config_is_strictly_isolated() -> None:
    baseline = _load_config(LSSCF_CONFIG)
    candidate = _load_config(DPER_CONFIG)
    assert {key: value for key, value in candidate.items() if key not in {"model", "save"}} == {
        key: value for key, value in baseline.items() if key not in {"model", "save"}
    }
    expected_model = copy.deepcopy(baseline["model"])
    expected_model.update(
        {
            "use_detail_preserving_early_rebase": True,
            "early_detail_channels": 16,
        }
    )
    assert candidate["model"] == expected_model
    assert candidate["save"]["save_dir"] == (
        "./runs/pilot/lightweight_c40_s4mamba_lsscf_dper/"
        "C3_C40_S4MAMBA_LSSCF_DPER_LR150_PHOTO_NATIVE384x544_EMA_BPCM"
    )


def test_false_flag_is_bitwise_lsscf_compatible() -> None:
    torch.manual_seed(101)
    baseline = _build(LSSCF_CONFIG).eval()
    torch.manual_seed(101)
    disabled = _build(
        LSSCF_CONFIG,
        use_detail_preserving_early_rebase=False,
        early_detail_channels=16,
    ).eval()
    _assert_equal_state(baseline.state_dict(), disabled.state_dict())
    image = torch.randn(1, 3, 64, 80, device=RUNTIME_DEVICE)
    with torch.inference_mode():
        baseline_outputs = baseline(image)
        disabled_outputs = disabled(image)
    assert set(baseline_outputs) == set(disabled_outputs)
    for key in baseline_outputs:
        assert torch.equal(baseline_outputs[key], disabled_outputs[key]), key


def test_dper_shapes_pixel_phases_and_stage4_contract() -> None:
    torch.manual_seed(102)
    model = _build(DPER_CONFIG).eval()
    assert model.use_detail_preserving_early_rebase
    assert model.early_detail_channels == 16
    assert isinstance(model.stem, DetailPreservingHalfStem)
    assert isinstance(model.stage1, HalfResolutionStage1Block)
    assert isinstance(model.f1_reconstruction, SubpixelDetailReconstruction)
    assert not isinstance(model.stem, CNNEdgeStem)
    assert not isinstance(model.stage1, LocalCrackBlock)
    assert isinstance(model.decoder.up3, LowResolutionSubpixelSeparableConcatUpBlock)
    assert isinstance(model.decoder.up2, LowResolutionSubpixelSeparableConcatUpBlock)
    assert isinstance(model.decoder.up1, LowResolutionSubpixelSeparableConcatUpBlock)
    assert isinstance(model.stage4, TopoRouteBlock)
    assert isinstance(model.stage4.global_branch, TransformerGlobalBlock)
    assert model.stage4.route_mamba.spatial_reduction == 1
    assert model.stage4.route_mamba.mamba_channels == 320
    assert model.stage4.route_mamba.active_mamba_directions == ALL_DIRECTIONS
    assert model.stage4.route_mamba.active_mamba_path_count == 4

    reconstruction = model.f1_reconstruction
    phase_weights = reconstruction.semantic_projection.weight.view(24, 4, 40, 1, 1)
    assert torch.equal(phase_weights[:, 0], phase_weights[:, 1])
    assert torch.equal(phase_weights[:, 0], phase_weights[:, 2])
    assert torch.equal(phase_weights[:, 0], phase_weights[:, 3])

    stage1_inputs: list[tuple[int, int]] = []
    f1_sizes: list[tuple[int, int, int]] = []
    path_calls = [0, 0, 0, 0]
    handles: list[torch.utils.hooks.RemovableHandle] = [
        model.stage1.register_forward_pre_hook(
            lambda _module, inputs: stage1_inputs.append(tuple(inputs[0].shape[-2:]))
        ),
        reconstruction.register_forward_hook(
            lambda _module, _inputs, output: f1_sizes.append(
                (output.shape[1], output.shape[-2], output.shape[-1])
            )
        ),
    ]

    def _path_hook(index: int):
        def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...]) -> None:
            path_calls[index] += 1

        return hook

    for index, path in enumerate(model.stage4.route_mamba.paths):
        handles.append(path.register_forward_pre_hook(_path_hook(index)))
    image = torch.randn(1, 3, 64, 80, device=RUNTIME_DEVICE)
    original_sobel = block_modules.sobel_edge
    sobel_calls = 0

    def _counted_sobel(value: torch.Tensor) -> torch.Tensor:
        nonlocal sobel_calls
        sobel_calls += 1
        return original_sobel(value)

    block_modules.sobel_edge = _counted_sobel
    try:
        with torch.inference_mode():
            features = model.extract_encoder_features(image)
            outputs = model(image)
    finally:
        block_modules.sobel_edge = original_sobel
        for handle in handles:
            handle.remove()

    # Both encoder-only and full forward intentionally invoke the DPER stem
    # once; no single encoder pass duplicates Sobel computation.
    assert sobel_calls == 2
    assert stage1_inputs == [(32, 40), (32, 40)]
    assert f1_sizes == [(40, 64, 80), (40, 64, 80)]
    assert features["f1"].shape == (1, 40, 64, 80)
    assert features["f2"].shape == (1, 80, 32, 40)
    assert features["f3"].shape == (1, 160, 16, 20)
    assert features["f4"].shape == (1, 320, 8, 10)
    assert outputs["route_s4"].shape == (1, 4, 8, 10)
    assert path_calls == [2, 2, 2, 2]
    assert {"mask", "edge", "reliability", "route_s4"}.issubset(outputs)
    assert outputs["mask"].shape == (1, 1, 64, 80)

    evidence = torch.cat([image, original_sobel(image)], dim=1)
    shuffled = model.stem.pixel_unshuffle(evidence)
    assert shuffled.shape == (1, 16, 32, 40)
    assert torch.equal(nn.PixelShuffle(2)(shuffled), evidence)


def test_dper_native_forward_mask_only_and_backward() -> None:
    torch.manual_seed(103)
    model = _build(DPER_CONFIG).eval()
    small_image = torch.randn(1, 3, 64, 80, device=RUNTIME_DEVICE)
    native_image = torch.randn(1, 3, 384, 544, device=RUNTIME_DEVICE)
    with torch.inference_mode():
        full_outputs = model(small_image)
        mask_only = model.forward_mask_only(small_image)
        native_outputs = model(native_image)
    assert torch.equal(full_outputs["mask"], mask_only["mask"])
    assert native_outputs["mask"].shape == (1, 1, 384, 544)
    assert native_outputs["edge"].shape == (1, 1, 384, 544)
    assert native_outputs["reliability"].shape == (1, 1, 384, 544)

    model.train()
    outputs = model(small_image)
    (outputs["mask"].mean() + outputs["edge"].mean() + outputs["reliability"].mean()).backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


def test_dper_preserves_shared_initialization_and_removes_old_modules() -> None:
    torch.manual_seed(104)
    baseline = _build(LSSCF_CONFIG)
    torch.manual_seed(104)
    dper = _build(DPER_CONFIG)
    baseline_state = baseline.state_dict()
    dper_state = dper.state_dict()
    shared_prefixes = ("stage2.", "down2.", "stage3.", "down3.", "stage4.", "decoder.")
    for key in baseline_state:
        if key.startswith(shared_prefixes):
            assert key in dper_state, key
            assert torch.equal(baseline_state[key], dper_state[key]), key
    assert not any(key.startswith("stem.stem.") for key in dper_state)
    assert not any(key.startswith("stage1.local.") for key in dper_state)
    assert not any(key.startswith("stage1.refine.") for key in dper_state)
    assert not any(key.startswith("stage1.global_branch.") for key in dper_state)
    assert not any(key.startswith("stage1.project.") for key in dper_state)
    assert isinstance(dper.down1, nn.Module)
    assert dper.down1.block[0].kernel_size == (1, 1)
    assert dper.down1.block[0].stride == (1, 1)


def test() -> None:
    tests = (
        test_dper_config_is_strictly_isolated,
        test_false_flag_is_bitwise_lsscf_compatible,
        test_dper_shapes_pixel_phases_and_stage4_contract,
        test_dper_native_forward_mask_only_and_backward,
        test_dper_preserves_shared_initialization_and_removes_old_modules,
    )
    for check in tests:
        check()
        print(f"PASS: {check.__name__}")


if __name__ == "__main__":
    test()
