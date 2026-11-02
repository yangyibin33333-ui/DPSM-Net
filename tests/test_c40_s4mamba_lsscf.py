"""Regression checks for the C40-S4Mamba LSSCF decoder experiment.

Run with: ``python tests/test_c40_s4mamba_lsscf.py``.
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
from models.decoder import (
    BoundaryAwareDecoder,
    LowResolutionSubpixelAdditiveUpBlock,
    LowResolutionSubpixelSeparableConcatUpBlock,
)
from models.mamba_blocks import SequenceMambaCore


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
ALL_DIRECTIONS = ("horizontal", "vertical", "diagonal", "anti_diagonal")
RUNTIME_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build_from_config(path: Path) -> torch.nn.Module:
    model_cfg = copy.deepcopy(_load_config(path)["model"])
    model_name = model_cfg.pop("name")
    # mamba_ssm's selective scan is CUDA-only; retain the topology and use the
    # established fallback only for direct CPU-host regression execution.
    if RUNTIME_DEVICE.type != "cuda":
        model_cfg["use_mamba_ssm"] = False
    return build_model(model_name, **model_cfg).to(RUNTIME_DEVICE)


def test_lsscf_block_contract_and_invalid_inputs() -> None:
    block = LowResolutionSubpixelSeparableConcatUpBlock(16, 12, 8)
    decoder_feature = torch.randn(2, 16, 5, 7)
    skip_feature = torch.randn(2, 12, 10, 14)
    output = block(decoder_feature, skip_feature)
    assert output.shape == (2, 8, 10, 14)

    phase_weights = block.decoder_projection.weight.view(8, 4, 16, 1, 1)
    assert torch.equal(phase_weights[:, 0], phase_weights[:, 1])
    assert torch.equal(phase_weights[:, 0], phase_weights[:, 2])
    assert torch.equal(phase_weights[:, 0], phase_weights[:, 3])
    assert block.depthwise.kernel_size == (3, 3)
    assert block.depthwise.in_channels == 16 and block.depthwise.out_channels == 16
    assert block.depthwise.groups == block.depthwise.in_channels
    assert block.pointwise.kernel_size == (1, 1)
    assert block.pointwise.in_channels == 16 and block.pointwise.out_channels == 8
    assert not any(
        isinstance(module, nn.Conv2d)
        and module.kernel_size == (3, 3)
        and module.groups == 1
        for module in block.modules()
    )

    try:
        block(torch.randn(1, 16, 5, 7), torch.randn(1, 12, 9, 14))
    except ValueError as error:
        assert "exactly two times" in str(error)
    else:
        raise AssertionError("LSSCF must reject a non-2x skip spatial size.")
    try:
        block(torch.randn(1, 15, 5, 7), torch.randn(1, 12, 10, 14))
    except ValueError as error:
        assert "decoder channels" in str(error)
    else:
        raise AssertionError("LSSCF must reject incorrect decoder channels.")
    try:
        block(torch.randn(1, 16, 5, 7), torch.randn(1, 11, 10, 14))
    except ValueError as error:
        assert "skip channels" in str(error)
    else:
        raise AssertionError("LSSCF must reject incorrect skip channels.")


def test_config_changes_only_decoder_fusion_and_save_dir() -> None:
    lsaf = _load_config(LSAF_CONFIG)
    lsscf = _load_config(LSSCF_CONFIG)
    assert {key: value for key, value in lsscf.items() if key not in {"model", "save"}} == {
        key: value for key, value in lsaf.items() if key not in {"model", "save"}
    }
    expected_model = copy.deepcopy(lsaf["model"])
    expected_model["use_low_resolution_subpixel_additive_fusion"] = False
    expected_model["use_low_resolution_subpixel_separable_concat_fusion"] = True
    assert lsscf["model"] == expected_model
    assert lsscf["save"]["save_dir"] == (
        "./runs/pilot/lightweight_c40_s4mamba_lsscf/"
        "C3_C40_S4MAMBA_LSSCF_LR150_PHOTO_NATIVE384x544_EMA_BPCM"
    )


def test_lsscf_decoder_structure_and_training_contract() -> None:
    torch.manual_seed(42)
    model = _build_from_config(LSSCF_CONFIG).train()
    decoder = model.decoder
    assert decoder.proj4 is None
    assert isinstance(decoder.up3, LowResolutionSubpixelSeparableConcatUpBlock)
    assert isinstance(decoder.up2, LowResolutionSubpixelSeparableConcatUpBlock)
    assert isinstance(decoder.up1, LowResolutionSubpixelSeparableConcatUpBlock)
    assert not decoder.use_low_resolution_subpixel_additive_fusion
    assert decoder.use_low_resolution_subpixel_separable_concat_fusion
    assert isinstance(model.stage3, LocalCrackBlock)
    assert not any(isinstance(module, SequenceMambaCore) for module in model.stage3.modules())
    assert isinstance(model.stage4, TopoRouteBlock)
    assert model.stage4.route_mamba.active_mamba_directions == ALL_DIRECTIONS
    assert model.stage4.route_mamba.active_mamba_path_count == len(ALL_DIRECTIONS)
    assert isinstance(model.stage4.global_branch, TransformerGlobalBlock)

    projection_sizes: list[tuple[int, int]] = []
    shuffle_input_sizes: list[tuple[int, int]] = []
    shuffle_output_sizes: list[tuple[int, int]] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for block in (decoder.up3, decoder.up2, decoder.up1):
        handles.append(
            block.decoder_projection.register_forward_pre_hook(
                lambda _module, inputs: projection_sizes.append(tuple(inputs[0].shape[-2:]))
            )
        )
        handles.append(
            block.pixel_shuffle.register_forward_pre_hook(
                lambda _module, inputs: shuffle_input_sizes.append(tuple(inputs[0].shape[-2:]))
            )
        )
        handles.append(
            block.pixel_shuffle.register_forward_hook(
                lambda _module, _inputs, output: shuffle_output_sizes.append(tuple(output.shape[-2:]))
            )
        )
    try:
        image = torch.randn(1, 3, 64, 80, device=RUNTIME_DEVICE)
        outputs = model(image)
    finally:
        for handle in handles:
            handle.remove()
    assert len(projection_sizes) == len(shuffle_input_sizes) == len(shuffle_output_sizes) == 3
    assert projection_sizes == shuffle_input_sizes
    assert all(output_h == 2 * input_h and output_w == 2 * input_w for (input_h, input_w), (output_h, output_w) in zip(shuffle_input_sizes, shuffle_output_sizes, strict=True))
    assert {"mask", "edge", "reliability"}.issubset(outputs)
    assert outputs["mask"].shape == (1, 1, 64, 80)
    assert "route_s3" not in outputs and "route_s4" in outputs
    loss = outputs["mask"].mean() + outputs["edge"].mean() + outputs["reliability"].mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


def test_lsscf_mask_only_and_existing_decoder_modes_remain_unchanged() -> None:
    model = _build_from_config(LSSCF_CONFIG).eval()
    image = torch.randn(1, 3, 64, 80, device=RUNTIME_DEVICE)
    auxiliary_head_calls = {"edge": 0, "reliability": 0}
    edge_handle = model.decoder.edge_head.register_forward_hook(
        lambda *_args: auxiliary_head_calls.__setitem__("edge", auxiliary_head_calls["edge"] + 1)
    )
    reliability_handle = model.decoder.reliability_head.register_forward_hook(
        lambda *_args: auxiliary_head_calls.__setitem__(
            "reliability", auxiliary_head_calls["reliability"] + 1
        )
    )
    try:
        with torch.inference_mode():
            outputs = model.forward_mask_only(image)
    finally:
        edge_handle.remove()
        reliability_handle.remove()
    assert set(outputs) == {"mask"}
    assert outputs["mask"].shape == (1, 1, 64, 80)
    assert auxiliary_head_calls == {"edge": 0, "reliability": 0}

    c40 = _build_from_config(C40_CONFIG)
    s4mamba = _build_from_config(S4MAMBA_CONFIG)
    lsaf = _build_from_config(LSAF_CONFIG)
    assert not c40.decoder.uses_low_resolution_subpixel_fusion
    assert not s4mamba.decoder.uses_low_resolution_subpixel_fusion
    assert isinstance(lsaf.decoder.up3, LowResolutionSubpixelAdditiveUpBlock)
    assert lsaf.decoder.use_low_resolution_subpixel_additive_fusion
    assert not lsaf.decoder.use_low_resolution_subpixel_separable_concat_fusion


def test_lsscf_and_lsaf_are_mutually_exclusive_and_incompatible_options_fail() -> None:
    try:
        BoundaryAwareDecoder(
            [40, 80, 160, 320],
            40,
            use_low_resolution_subpixel_additive_fusion=True,
            use_low_resolution_subpixel_separable_concat_fusion=True,
        )
    except ValueError as error:
        assert "mutually exclusive" in str(error)
    else:
        raise AssertionError("LSAF and LSSCF must not be enabled together.")

    incompatible_options = (
        "use_frequency_gate",
        "use_semantic_skip_gate",
        "use_detail_residual_gate",
        "use_structure_harmonic_fusion",
        "use_boundary_logit_calibration",
        "use_sdf_head",
        "use_edge_mask_consistency_refiner",
        "use_deep_supervision",
        "use_route_guided_directional_refiner",
        "use_route_guided_anisotropic_diffusion",
        "use_local_semantic_continuity_calibrator",
        "use_local_pixel_reassembly_residual",
        "use_local_spectral_confidence_rejector",
        "use_high_resolution_spectral_detail_gate",
    )
    for option in incompatible_options:
        try:
            BoundaryAwareDecoder(
                [40, 80, 160, 320],
                40,
                use_low_resolution_subpixel_separable_concat_fusion=True,
                **{option: True},
            )
        except ValueError as error:
            assert "LSSCF" in str(error)
        else:
            raise AssertionError(f"LSSCF must reject {option}.")


def test() -> None:
    tests = (
        test_lsscf_block_contract_and_invalid_inputs,
        test_config_changes_only_decoder_fusion_and_save_dir,
        test_lsscf_decoder_structure_and_training_contract,
        test_lsscf_mask_only_and_existing_decoder_modes_remain_unchanged,
        test_lsscf_and_lsaf_are_mutually_exclusive_and_incompatible_options_fail,
    )
    for check in tests:
        check()
        print(f"PASS: {check.__name__}")


if __name__ == "__main__":
    test()
