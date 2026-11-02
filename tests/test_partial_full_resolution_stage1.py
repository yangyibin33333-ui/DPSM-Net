"""Regression checks for the strict DPER-PFRS single-variable candidate.

Run with either ``python tests/test_partial_full_resolution_stage1.py`` or
``python -m pytest tests/test_partial_full_resolution_stage1.py -q``.
"""

from __future__ import annotations

import copy
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import build_model
from models.blocks import (
    DetailPreservingHalfStem,
    DWConvBlock,
    HalfResolutionStage1Block,
    LocalCrackBlock,
    PartialFullResolutionStage1,
    StripConvBlock,
    TopoRouteBlock,
    TransformerGlobalBlock,
)
from models.decoder import LowResolutionSubpixelSeparableConcatUpBlock
from models.mamba_blocks import SequenceMambaCore
from train import make_optimizer


DPER_CONFIG = (
    ROOT / "configs" / "experiments" / "lightweight_c40_s4mamba_lsscf_dper"
    / "C3_C40_S4MAMBA_LSSCF_DPER_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
PFRS_CONFIG = (
    ROOT / "configs" / "experiments" / "lightweight_c40_s4mamba_lsscf_dper_pfrs"
    / "C3_C40_S4MAMBA_LSSCF_DPER_PFRS_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
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
    # CPU verifies tensor/dataflow mechanics through the established fallback.
    # Formal PFRS acceptance requires --require_mamba_ssm on demamba21.
    if RUNTIME_DEVICE.type != "cuda":
        model_cfg["use_mamba_ssm"] = False
    return build_model(model_name, **model_cfg).to(RUNTIME_DEVICE)


def _assert_equal_state(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> None:
    assert set(left) == set(right)
    for key, value in left.items():
        assert torch.equal(value, right[key]), key


def test_pfrs_config_is_a_strict_dper_single_variable_change() -> None:
    dper = _load_config(DPER_CONFIG)
    pfrs = _load_config(PFRS_CONFIG)
    assert {key: value for key, value in pfrs.items() if key not in {"model", "save"}} == {
        key: value for key, value in dper.items() if key not in {"model", "save"}
    }
    expected_model = copy.deepcopy(dper["model"])
    expected_model["use_partial_full_resolution_stage1"] = True
    assert pfrs["model"] == expected_model
    assert pfrs["save"]["save_dir"] == (
        "./runs/pilot/lightweight_c40_s4mamba_lsscf_dper_pfrs/"
        "C3_C40_S4MAMBA_LSSCF_DPER_PFRS_LR150_PHOTO_NATIVE384x544_EMA_BPCM"
    )


def test_pfrs_guards_and_disabled_dper_equivalence() -> None:
    invalid = (
        ({"use_detail_preserving_early_rebase": False, "use_partial_full_resolution_stage1": True}, "requires"),
        ({"use_subpixel_phase_consistency_refiner": True, "use_partial_full_resolution_stage1": True}, "cannot"),
        ({"use_haar_topology_preserving_rebase": True, "use_partial_full_resolution_stage1": True}, "cannot"),
        ({"use_narrow_dual_resolution_stage1": True, "use_partial_full_resolution_stage1": True}, "cannot"),
        ({"use_full_resolution_joint_exchange": True, "use_partial_full_resolution_stage1": True}, "cannot"),
        ({"use_early_serpentine_connectivity_ssm": True, "use_partial_full_resolution_stage1": True}, "cannot"),
    )
    for overrides, expected_text in invalid:
        try:
            _build(DPER_CONFIG, **overrides)
        except ValueError as error:
            assert expected_text in str(error)
        else:
            raise AssertionError("PFRS dependency validation did not reject an invalid configuration.")

    torch.manual_seed(1101)
    dper = _build(DPER_CONFIG).eval()
    torch.manual_seed(1101)
    disabled = _build(DPER_CONFIG, use_partial_full_resolution_stage1=False).eval()
    assert disabled.partial_full_resolution_stage1 is None
    _assert_equal_state(dper.state_dict(), disabled.state_dict())
    image = torch.randn(1, 3, 64, 80, device=RUNTIME_DEVICE)
    with torch.inference_mode():
        reference = dper(image)
        actual = disabled(image)
    assert set(reference) == set(actual)
    for key in reference:
        assert torch.equal(reference[key], actual[key]), key


def test_pfrs_structure_residual_exchange_and_encoder_dataflow() -> None:
    torch.manual_seed(1102)
    dper = _build(DPER_CONFIG).eval()
    torch.manual_seed(1102)
    model = _build(PFRS_CONFIG).eval()
    assert model.use_detail_preserving_early_rebase
    assert model.use_partial_full_resolution_stage1
    assert not model.use_subpixel_phase_consistency_refiner
    assert not model.use_haar_topology_preserving_rebase
    assert not model.use_narrow_dual_resolution_stage1
    assert not model.use_full_resolution_joint_exchange
    assert not model.use_early_serpentine_connectivity_ssm
    assert isinstance(model.stem, DetailPreservingHalfStem)
    assert isinstance(model.stage1, HalfResolutionStage1Block)
    assert isinstance(model.partial_full_resolution_stage1, PartialFullResolutionStage1)
    assert model.early_serpentine_connectivity_ssm is None
    pfrs = model.partial_full_resolution_stage1
    assert pfrs.detail_channels == 16 and pfrs.semantic_channels == 40
    assert isinstance(pfrs.full_stage1, LocalCrackBlock)
    assert isinstance(pfrs.full_stage1.project, nn.Identity)
    assert isinstance(pfrs.full_stage1.global_branch, nn.Identity)
    assert isinstance(pfrs.full_stage1.local[0], DWConvBlock)
    assert isinstance(pfrs.full_stage1.local[1], StripConvBlock)
    assert isinstance(pfrs.full_stage1.refine, DWConvBlock)
    assert pfrs.full_stage1.local[1].horizontal.kernel_size == (1, 7)
    assert pfrs.full_stage1.local[1].vertical.kernel_size == (7, 1)
    assert isinstance(pfrs.phase_unshuffle, nn.PixelUnshuffle) and pfrs.phase_unshuffle.downscale_factor == 2
    assert isinstance(pfrs.exchange_projection, nn.Sequential) and len(pfrs.exchange_projection) == 2
    exchange_conv, exchange_norm = pfrs.exchange_projection
    assert isinstance(exchange_conv, nn.Conv2d) and exchange_conv.in_channels == 64 and exchange_conv.out_channels == 40
    assert exchange_conv.bias is None and isinstance(exchange_norm, nn.BatchNorm2d)
    assert not any(isinstance(module, (SequenceMambaCore, TransformerGlobalBlock)) for module in pfrs.modules())
    assert not any("gate" in name or "alpha" in name for name, _ in pfrs.named_parameters())

    dper_state, pfrs_state = dper.state_dict(), model.state_dict()
    assert set(pfrs_state) - set(dper_state) == {
        key for key in pfrs_state if key.startswith("partial_full_resolution_stage1.")
    }
    for key, value in dper_state.items():
        assert key in pfrs_state, key
        assert torch.equal(value, pfrs_state[key]), key

    assert isinstance(model.decoder.up3, LowResolutionSubpixelSeparableConcatUpBlock)
    assert isinstance(model.decoder.up2, LowResolutionSubpixelSeparableConcatUpBlock)
    assert isinstance(model.decoder.up1, LowResolutionSubpixelSeparableConcatUpBlock)
    assert isinstance(model.stage4, TopoRouteBlock)
    assert isinstance(model.stage4.global_branch, TransformerGlobalBlock)
    assert model.stage4.route_mamba.active_mamba_directions == ALL_DIRECTIONS
    assert model.stage4.route_mamba.active_mamba_path_count == 4
    assert len(model.mamba_backend_info()["stage4"]) == 4
    if RUNTIME_DEVICE.type == "cuda":
        assert model.mamba_backend_info()["stage4"] == ["mamba_ssm"] * 4

    observed: dict[str, torch.Tensor] = {}
    calls = {"stage1": 0, "pfrs": 0}

    def stage1_hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        calls["stage1"] += 1
        observed["semantic_half"] = inputs[0].detach()

    def stage1_out_hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        observed["stage1_output"] = output.detach()

    def pfrs_hook(
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        calls["pfrs"] += 1
        observed["full_detail"] = inputs[0].detach()
        observed["semantic_stage1"] = inputs[1].detach()
        observed["detail_stage1"] = output[0].detach()
        observed["enhanced_half"] = output[1].detach()

    def unshuffle_hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        observed["detail_delta"] = inputs[0].detach()

    def unshuffle_out_hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        observed["unshuffle_delta"] = output.detach()

    handles = [
        model.stage1.register_forward_pre_hook(stage1_hook),
        model.stage1.register_forward_hook(stage1_out_hook),
        pfrs.register_forward_hook(pfrs_hook),
        pfrs.phase_unshuffle.register_forward_pre_hook(unshuffle_hook),
        pfrs.phase_unshuffle.register_forward_hook(unshuffle_out_hook),
    ]
    image = torch.randn(1, 3, 64, 80, device=RUNTIME_DEVICE)
    try:
        with torch.inference_mode():
            features = model.extract_encoder_features(image)
    finally:
        for handle in handles:
            handle.remove()

    assert calls == {"stage1": 1, "pfrs": 1}
    assert observed["full_detail"].shape == observed["detail_stage1"].shape == observed["detail_delta"].shape == (1, 16, 64, 80)
    assert observed["semantic_half"].shape == observed["semantic_stage1"].shape == observed["enhanced_half"].shape == (1, 40, 32, 40)
    assert observed["unshuffle_delta"].shape == (1, 64, 32, 40)
    assert torch.equal(observed["detail_delta"], observed["detail_stage1"] - observed["full_detail"])
    assert torch.equal(observed["semantic_stage1"], observed["stage1_output"])
    assert features["f1"].shape == (1, 40, 64, 80)
    assert features["f2"].shape == (1, 80, 32, 40)
    assert features["f3"].shape == (1, 160, 16, 20)
    assert features["f4"].shape == (1, 320, 8, 10)


def test_pfrs_all_entrypoints_shapes_gradients_and_fused_adamw() -> None:
    torch.manual_seed(1103)
    model = _build(PFRS_CONFIG).eval()
    assert isinstance(model.partial_full_resolution_stage1, PartialFullResolutionStage1)
    pfrs = model.partial_full_resolution_stage1
    calls = 0

    def count_call(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...]) -> None:
        nonlocal calls
        calls += 1

    handle = pfrs.register_forward_pre_hook(count_call)
    small = torch.randn(1, 3, 64, 80, device=RUNTIME_DEVICE)
    try:
        with torch.inference_mode():
            features = model.extract_encoder_features(small)
            full = model(small)
            mask_only = model.forward_mask_only(small)
    finally:
        handle.remove()
    assert calls == 3
    assert features["f1"].shape == (1, 40, 64, 80)
    assert full["mask"].shape == full["edge"].shape == full["reliability"].shape == (1, 1, 64, 80)
    assert torch.equal(full["mask"], mask_only["mask"])

    for height, width in ((384, 544), (512, 512), (128, 96)):
        with torch.inference_mode():
            output = model.forward_mask_only(torch.randn(1, 3, height, width, device=RUNTIME_DEVICE))
        assert output["mask"].shape == (1, 1, height, width)

    model.train()
    optimizer = make_optimizer(model, _load_config(PFRS_CONFIG)["train"], RUNTIME_DEVICE, logging.getLogger(__name__))
    assert all(group["fused"] for group in optimizer.param_groups) if RUNTIME_DEVICE.type == "cuda" else True
    # Validate the encoder branches with a direct structural objective.  A
    # simple mean of three logits is not a faithful training loss and can fail
    # to exercise shallow skip gradients; f1/f2 energy covers both the full-
    # resolution LocalCrackBlock and residual-only exchange explicitly.
    optimizer.zero_grad(set_to_none=True)
    training_features = model.extract_encoder_features(small)
    structural_probe = training_features["f1"].float().square().mean() + training_features["f2"].float().square().mean()
    structural_probe.backward()
    required_gradients = {
        "full_stage1.local.0.block.0.block.0.weight",
        "full_stage1.local.1.horizontal.weight",
        "full_stage1.local.1.vertical.weight",
        "full_stage1.refine.block.0.block.0.weight",
        "exchange_projection.0.weight",
    }
    pfrs_parameters = dict(pfrs.named_parameters())
    for name, parameter in pfrs_parameters.items():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    for name in required_gradients:
        gradient = pfrs_parameters[name].grad
        assert gradient is not None and torch.isfinite(gradient).all(), name
        assert torch.count_nonzero(gradient).item() > 0, name
    stage1_gradient = model.stage1.residual.block[0].weight.grad
    assert stage1_gradient is not None and torch.isfinite(stage1_gradient).all()
    assert torch.count_nonzero(stage1_gradient).item() > 0
    # Exercise the repository's actual AMP/fused-AdamW mechanics separately.
    optimizer.zero_grad(set_to_none=True)
    if RUNTIME_DEVICE.type == "cuda":
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(small)
            loss = output["mask"].square().mean() + output["edge"].square().mean() + output["reliability"].square().mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        output = model(small)
        (output["mask"].square().mean() + output["edge"].square().mean() + output["reliability"].square().mean()).backward()
        optimizer.step()
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test() -> None:
    tests = (
        test_pfrs_config_is_a_strict_dper_single_variable_change,
        test_pfrs_guards_and_disabled_dper_equivalence,
        test_pfrs_structure_residual_exchange_and_encoder_dataflow,
        test_pfrs_all_entrypoints_shapes_gradients_and_fused_adamw,
    )
    for check in tests:
        check()
        print(f"PASS: {check.__name__}")


if __name__ == "__main__":
    test()
