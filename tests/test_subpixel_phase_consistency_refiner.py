"""Regression checks for the DPER-SPCR single-variable ablation.

Run with: ``python tests/test_subpixel_phase_consistency_refiner.py``.
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
from models.blocks import SubpixelPhaseConsistencyRefiner, TopoRouteBlock, TransformerGlobalBlock
from models.decoder import LowResolutionSubpixelSeparableConcatUpBlock


DPER_CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "lightweight_c40_s4mamba_lsscf_dper"
    / "C3_C40_S4MAMBA_LSSCF_DPER_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
SPCR_CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "lightweight_c40_s4mamba_lsscf_dper_spcr"
    / "C3_C40_S4MAMBA_LSSCF_DPER_SPCR_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
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
    if RUNTIME_DEVICE.type != "cuda":
        model_cfg["use_mamba_ssm"] = False
    return build_model(model_name, **model_cfg).to(RUNTIME_DEVICE)


def test_spcr_config_is_a_single_variable_change() -> None:
    dper = _load_config(DPER_CONFIG)
    spcr = _load_config(SPCR_CONFIG)
    assert {key: value for key, value in spcr.items() if key not in {"model", "save"}} == {
        key: value for key, value in dper.items() if key not in {"model", "save"}
    }
    expected_model = copy.deepcopy(dper["model"])
    expected_model["use_subpixel_phase_consistency_refiner"] = True
    assert spcr["model"] == expected_model
    assert spcr["save"]["save_dir"] == (
        "./runs/pilot/lightweight_c40_s4mamba_lsscf_dper_spcr/"
        "C3_C40_S4MAMBA_LSSCF_DPER_SPCR_LR150_PHOTO_NATIVE384x544_EMA_BPCM"
    )


def test_spcr_is_strict_identity_then_learns_a_nonzero_residual() -> None:
    torch.manual_seed(201)
    refiner = SubpixelPhaseConsistencyRefiner(24).train()
    assert refiner.depthwise.groups == 24
    assert refiner.depthwise.bias is None and refiner.pointwise.bias is None
    assert torch.equal(refiner.norm2.weight, torch.zeros_like(refiner.norm2.weight))
    assert torch.equal(refiner.norm2.bias, torch.zeros_like(refiner.norm2.bias))
    feature = torch.randn(2, 24, 7, 9)
    initial = refiner(feature)
    assert initial.shape == feature.shape
    assert torch.equal(initial, feature)

    optimizer = torch.optim.SGD(refiner.parameters(), lr=0.1)
    optimizer.zero_grad(set_to_none=True)
    loss = refiner(feature).square().mean()
    loss.backward()
    assert refiner.norm2.weight.grad is not None
    assert refiner.norm2.bias.grad is not None
    assert torch.isfinite(refiner.norm2.weight.grad).all()
    assert torch.isfinite(refiner.norm2.bias.grad).all()
    optimizer.step()
    with torch.no_grad():
        residual = refiner(feature) - feature
    assert torch.isfinite(residual).all()
    assert torch.count_nonzero(residual).item() > 0


def test_spcr_requires_dper_and_keeps_false_flag_bitwise_compatible() -> None:
    try:
        _build(DPER_CONFIG, use_detail_preserving_early_rebase=False, use_subpixel_phase_consistency_refiner=True)
    except ValueError as error:
        assert "requires" in str(error)
    else:
        raise AssertionError("SPCR must reject a non-DPER configuration.")

    torch.manual_seed(202)
    dper = _build(DPER_CONFIG).eval()
    torch.manual_seed(202)
    false_flag = _build(DPER_CONFIG, use_subpixel_phase_consistency_refiner=False).eval()
    assert set(dper.state_dict()) == set(false_flag.state_dict())
    for key, value in dper.state_dict().items():
        assert torch.equal(value, false_flag.state_dict()[key]), key
    image = torch.randn(1, 3, 64, 80, device=RUNTIME_DEVICE)
    with torch.inference_mode():
        expected = dper(image)
        actual = false_flag(image)
    assert set(expected) == set(actual)
    for key in expected:
        assert torch.equal(expected[key], actual[key]), key


def test_spcr_model_contract_shared_initialization_and_mask_only() -> None:
    torch.manual_seed(203)
    dper = _build(DPER_CONFIG).eval()
    torch.manual_seed(203)
    spcr = _build(SPCR_CONFIG).eval()
    assert spcr.use_detail_preserving_early_rebase
    assert spcr.use_subpixel_phase_consistency_refiner
    reconstruction = spcr.f1_reconstruction
    assert reconstruction is not None
    assert isinstance(reconstruction.phase_refiner, SubpixelPhaseConsistencyRefiner)
    assert reconstruction.reconstructed_channels == 24
    assert reconstruction.phase_refiner.channels == 24

    dper_state = dper.state_dict()
    spcr_state = spcr.state_dict()
    added_keys = {key for key in spcr_state if key not in dper_state}
    assert added_keys and all(key.startswith("f1_reconstruction.phase_refiner.") for key in added_keys)
    for key, value in dper_state.items():
        assert key in spcr_state, key
        assert torch.equal(value, spcr_state[key]), key

    assert isinstance(spcr.stage4, TopoRouteBlock)
    assert isinstance(spcr.stage4.global_branch, TransformerGlobalBlock)
    assert spcr.stage4.channels == 320
    assert spcr.stage4.route_mamba.spatial_reduction == 1
    assert spcr.stage4.route_mamba.mamba_channels == 320
    assert spcr.stage4.route_mamba.active_mamba_directions == ALL_DIRECTIONS
    assert spcr.stage4.route_mamba.active_mamba_path_count == 4
    assert isinstance(spcr.decoder.up3, LowResolutionSubpixelSeparableConcatUpBlock)
    assert isinstance(spcr.decoder.up2, LowResolutionSubpixelSeparableConcatUpBlock)
    assert isinstance(spcr.decoder.up1, LowResolutionSubpixelSeparableConcatUpBlock)

    calls = 0

    def _count(_module: torch.nn.Module, _inputs: tuple[object, ...], _output: object) -> None:
        nonlocal calls
        calls += 1

    count_handle = reconstruction.phase_refiner.register_forward_hook(_count)
    image = torch.randn(1, 3, 64, 80, device=RUNTIME_DEVICE)
    try:
        with torch.inference_mode():
            dper_outputs = dper(image)
            full_outputs = spcr(image)
            mask_only = spcr.forward_mask_only(image)
            features = spcr.extract_encoder_features(image)
    finally:
        count_handle.remove()
    # A zero-initialized SPCR must leave original DPER output exactly intact.
    for key in dper_outputs:
        assert torch.equal(dper_outputs[key], full_outputs[key]), key
    assert torch.equal(full_outputs["mask"], mask_only["mask"])
    assert features["f1"].shape == (1, 40, 64, 80)
    assert features["f2"].shape == (1, 80, 32, 40)
    assert features["f3"].shape == (1, 160, 16, 20)
    assert features["f4"].shape == (1, 320, 8, 10)
    # full, mask-only, and explicit encoder extraction each execute SPCR once.
    assert calls == 3
    assert not any(key.startswith("stem.stem.") for key in spcr_state)
    assert not any(key.startswith("stage1.local.") for key in spcr_state)


def test() -> None:
    tests = (
        test_spcr_config_is_a_single_variable_change,
        test_spcr_is_strict_identity_then_learns_a_nonzero_residual,
        test_spcr_requires_dper_and_keeps_false_flag_bitwise_compatible,
        test_spcr_model_contract_shared_initialization_and_mask_only,
    )
    for check in tests:
        check()
        print(f"PASS: {check.__name__}")


if __name__ == "__main__":
    test()
