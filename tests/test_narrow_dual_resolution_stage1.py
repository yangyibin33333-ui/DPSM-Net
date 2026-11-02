"""Regression checks for the independent DPER-NDRS single-variable candidate.

Run with either ``python tests/test_narrow_dual_resolution_stage1.py`` or
``python -m pytest tests/test_narrow_dual_resolution_stage1.py -q``.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import build_model
from models.blocks import (
    DetailPreservingHalfStem,
    HalfResolutionStage1Block,
    NarrowDualResolutionStage1,
    SubpixelPhaseConsistencyRefiner,
    TopoRouteBlock,
    TransformerGlobalBlock,
)
from models.decoder import LowResolutionSubpixelSeparableConcatUpBlock


DPER_CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "lightweight_c40_s4mamba_lsscf_dper"
    / "C3_C40_S4MAMBA_LSSCF_DPER_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
NDRS_CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "lightweight_c40_s4mamba_lsscf_dper_ndrs"
    / "C3_C40_S4MAMBA_LSSCF_DPER_NDRS_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
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
    # CPU supports the regression graph through the established fallback while
    # retaining all four Stage4 paths. Formal measurements require mamba_ssm.
    if RUNTIME_DEVICE.type != "cuda":
        model_cfg["use_mamba_ssm"] = False
    return build_model(model_name, **model_cfg).to(RUNTIME_DEVICE)


def _assert_equal_state(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> None:
    assert set(left) == set(right)
    for key, value in left.items():
        assert torch.equal(value, right[key]), key


def test_ndrs_config_is_a_strict_dper_single_variable_change() -> None:
    dper = _load_config(DPER_CONFIG)
    ndrs = _load_config(NDRS_CONFIG)
    assert {key: value for key, value in ndrs.items() if key not in {"model", "save"}} == {
        key: value for key, value in dper.items() if key not in {"model", "save"}
    }
    expected_model = copy.deepcopy(dper["model"])
    expected_model["use_narrow_dual_resolution_stage1"] = True
    assert ndrs["model"] == expected_model
    assert ndrs["save"]["save_dir"] == (
        "./runs/pilot/lightweight_c40_s4mamba_lsscf_dper_ndrs/"
        "C3_C40_S4MAMBA_LSSCF_DPER_NDRS_LR150_PHOTO_NATIVE384x544_EMA_BPCM"
    )


def test_ndrs_guards_and_false_flag_are_bitwise_dper_compatible() -> None:
    invalid = (
        ({"use_detail_preserving_early_rebase": False, "use_narrow_dual_resolution_stage1": True}, "requires"),
        ({"use_subpixel_phase_consistency_refiner": True, "use_narrow_dual_resolution_stage1": True}, "cannot"),
        ({"use_haar_topology_preserving_rebase": True, "use_narrow_dual_resolution_stage1": True}, "cannot"),
    )
    for overrides, text in invalid:
        try:
            _build(DPER_CONFIG, **overrides)
        except ValueError as error:
            assert text in str(error)
        else:
            raise AssertionError("NDRS dependency validation did not reject an invalid configuration.")

    torch.manual_seed(401)
    dper = _build(DPER_CONFIG).eval()
    torch.manual_seed(401)
    disabled = _build(DPER_CONFIG, use_narrow_dual_resolution_stage1=False).eval()
    _assert_equal_state(dper.state_dict(), disabled.state_dict())
    image = torch.randn(1, 3, 64, 80, device=RUNTIME_DEVICE)
    with torch.inference_mode():
        reference = dper(image)
        actual = disabled(image)
    assert set(reference) == set(actual)
    for key in reference:
        assert torch.equal(reference[key], actual[key]), key


def test_ndrs_path_shapes_stage1_exchange_and_preserved_topology() -> None:
    torch.manual_seed(402)
    dper = _build(DPER_CONFIG).eval()
    torch.manual_seed(402)
    model = _build(NDRS_CONFIG).eval()
    assert model.use_detail_preserving_early_rebase
    assert model.use_narrow_dual_resolution_stage1
    assert not model.use_subpixel_phase_consistency_refiner
    assert not model.use_haar_topology_preserving_rebase
    assert isinstance(model.stem, DetailPreservingHalfStem)
    assert isinstance(model.stage1, HalfResolutionStage1Block)
    assert isinstance(model.narrow_dual_resolution_stage1, NarrowDualResolutionStage1)
    assert model.f1_reconstruction is not None and model.f1_reconstruction.phase_refiner is None
    assert not any(isinstance(module, SubpixelPhaseConsistencyRefiner) for module in model.modules())
    ndrs = model.narrow_dual_resolution_stage1
    assert ndrs.detail_channels == 16 and ndrs.semantic_channels == 40
    assert ndrs.local_depthwise.groups == 16
    assert ndrs.local_depthwise.kernel_size == (3, 3)
    assert ndrs.local_depthwise.stride == (1, 1)
    assert ndrs.exchange_depthwise.groups == 16
    assert ndrs.exchange_depthwise.kernel_size == (3, 3)
    assert ndrs.exchange_depthwise.stride == (2, 2)
    assert ndrs.local_pointwise.in_channels == ndrs.local_pointwise.out_channels == 16
    assert ndrs.exchange_projection.in_channels == 16 and ndrs.exchange_projection.out_channels == 40

    dper_state = dper.state_dict()
    ndrs_state = model.state_dict()
    assert set(ndrs_state) - set(dper_state) == {
        key for key in ndrs_state if key.startswith("narrow_dual_resolution_stage1.")
    }
    for key, value in dper_state.items():
        assert key in ndrs_state, key
        assert torch.equal(value, ndrs_state[key]), key

    assert isinstance(model.decoder.up3, LowResolutionSubpixelSeparableConcatUpBlock)
    assert isinstance(model.decoder.up2, LowResolutionSubpixelSeparableConcatUpBlock)
    assert isinstance(model.decoder.up1, LowResolutionSubpixelSeparableConcatUpBlock)
    assert isinstance(model.stage4, TopoRouteBlock)
    assert isinstance(model.stage4.global_branch, TransformerGlobalBlock)
    assert model.stage4.route_mamba.active_mamba_directions == ALL_DIRECTIONS
    assert model.stage4.route_mamba.active_mamba_path_count == 4
    assert len(model.mamba_backend_info()["stage4"]) == 4

    calls = {"ndrs": 0, "stage1": 0}
    observed: dict[str, torch.Tensor] = {}

    def ndrs_hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: tuple[torch.Tensor, torch.Tensor]) -> None:
        calls["ndrs"] += 1
        observed["full_detail"] = inputs[0].detach()
        observed["local_detail"] = output[0].detach()
        observed["enhanced_half"] = output[1].detach()

    def stage1_hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        calls["stage1"] += 1
        observed["stage1_input"] = inputs[0].detach()

    handles = [
        ndrs.register_forward_hook(ndrs_hook),
        model.stage1.register_forward_pre_hook(stage1_hook),
    ]
    image = torch.randn(1, 3, 64, 80, device=RUNTIME_DEVICE)
    try:
        with torch.inference_mode():
            features = model.extract_encoder_features(image)
    finally:
        for handle in handles:
            handle.remove()

    assert calls == {"ndrs": 1, "stage1": 1}
    assert observed["full_detail"].shape == observed["local_detail"].shape == (1, 16, 64, 80)
    assert observed["enhanced_half"].shape == (1, 40, 32, 40)
    assert torch.equal(observed["stage1_input"], observed["enhanced_half"])
    assert features["f1"].shape == (1, 40, 64, 80)
    assert features["f2"].shape == (1, 80, 32, 40)
    assert features["f3"].shape == (1, 160, 16, 20)
    assert features["f4"].shape == (1, 320, 8, 10)


def test_ndrs_shared_encoder_paths_mask_identity_and_gradients() -> None:
    torch.manual_seed(403)
    model = _build(NDRS_CONFIG).eval()
    assert isinstance(model.narrow_dual_resolution_stage1, NarrowDualResolutionStage1)
    ndrs = model.narrow_dual_resolution_stage1
    calls = 0

    def count_call(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...]) -> None:
        nonlocal calls
        calls += 1

    handle = ndrs.register_forward_pre_hook(count_call)
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

    with torch.inference_mode():
        native = model(torch.randn(1, 3, 384, 544, device=RUNTIME_DEVICE))
    assert native["mask"].shape == native["edge"].shape == native["reliability"].shape == (1, 1, 384, 544)

    model.train()
    output = model(small)
    (output["mask"].mean() + output["edge"].mean() + output["reliability"].mean()).backward()
    ndrs_gradients = dict(ndrs.named_parameters())
    assert ndrs_gradients
    for name, parameter in ndrs_gradients.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test() -> None:
    tests = (
        test_ndrs_config_is_a_strict_dper_single_variable_change,
        test_ndrs_guards_and_false_flag_are_bitwise_dper_compatible,
        test_ndrs_path_shapes_stage1_exchange_and_preserved_topology,
        test_ndrs_shared_encoder_paths_mask_identity_and_gradients,
    )
    for check in tests:
        check()
        print(f"PASS: {check.__name__}")


if __name__ == "__main__":
    test()
