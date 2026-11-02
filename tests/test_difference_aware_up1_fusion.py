"""Regression checks for the isolated DPSM80-DAFU1 Up1 experiment."""

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

from losses import DEMambaLoss
from models import build_model
from models.decoder import (
    DifferenceAwareSubpixelUpBlock,
    LowResolutionSubpixelSeparableConcatUpBlock,
)
from models.mamba_blocks import DirectionPartitionedSharedMamba
from models.transformer_blocks import TransformerGlobalBlock
from train import make_optimizer


DPSM_CONFIG = (
    ROOT / "configs" / "experiments" / "lightweight_c40_s4mamba_lsscf_dper_rsdm_dpsm"
    / "C3_C40_S4MAMBA_LSSCF_DPER_RSDM_DPSM80_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
DAFU1_CONFIG = (
    ROOT / "configs" / "experiments" / "lightweight_c40_s4mamba_lsscf_dper_rsdm_dpsm_dafu1"
    / "C3_C40_S4MAMBA_LSSCF_DPER_RSDM_DPSM80_DAFU1_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build(path: Path, **overrides: object) -> torch.nn.Module:
    model_cfg = copy.deepcopy(_load(path)["model"])
    model_cfg.update(overrides)
    name = model_cfg.pop("name")
    if DEVICE.type != "cuda":
        model_cfg["use_mamba_ssm"] = False
    return build_model(name, **model_cfg).to(DEVICE)


def _batch(batch_size: int, height: int, width: int) -> dict[str, torch.Tensor]:
    direction = torch.randint(0, 4, (batch_size, 1, height, width), device=DEVICE)
    tangent = torch.zeros(batch_size, 4, height, width, device=DEVICE)
    tangent.scatter_(1, direction, 1.0)
    return {
        "image": torch.randn(batch_size, 3, height, width, device=DEVICE),
        "mask": (torch.rand(batch_size, 1, height, width, device=DEVICE) < 0.1).float(),
        "edge_gt": (torch.rand(batch_size, 1, height, width, device=DEVICE) < 0.05).float(),
        "tangent_gt": tangent,
        "tangent_valid": torch.ones(batch_size, 1, height, width, device=DEVICE),
    }


def _assert_dict_equal(left: dict, right: dict) -> None:
    assert set(left) == set(right)
    for key, value in left.items():
        assert torch.equal(value, right[key]), key


def test_dafu1_config_is_a_strict_dpsm80_single_variable_change() -> None:
    baseline, candidate = _load(DPSM_CONFIG), _load(DAFU1_CONFIG)
    assert {key: value for key, value in candidate.items() if key not in {"model", "save"}} == {
        key: value for key, value in baseline.items() if key not in {"model", "save"}
    }
    expected_model = copy.deepcopy(baseline["model"])
    expected_model["use_difference_aware_up1_fusion"] = True
    assert candidate["model"] == expected_model
    assert candidate["save"]["save_dir"] == (
        "./runs/pilot/lightweight_c40_s4mamba_lsscf_dper_rsdm_dpsm_dafu1/"
        "C3_C40_S4MAMBA_LSSCF_DPER_RSDM_DPSM80_DAFU1_LR150_PHOTO_NATIVE384x544_EMA_BPCM"
    )


def test_dafu1_guards_and_disabled_dpsm80_compatibility() -> None:
    invalid = (
        ({"use_direction_partitioned_shared_mamba": False}, "DPSM80"),
        ({"use_common_private_direction_partitioned_shared_mamba": True}, "mutually exclusive"),
        ({"use_depthwise_separable_mask_head": True}, "DSMH"),
        ({"use_full_width_residual_mask_head": True}, "FWRMH"),
        ({"use_low_resolution_subpixel_separable_concat_fusion": False}, "LSSCF"),
        ({"use_narrow_transport_decoder": True}, "NTD"),
        # The pre-existing RSDM decoder-isolation guard runs before the
        # DAFU1-specific guard and correctly identifies this as LSAF.
        ({"use_low_resolution_subpixel_additive_fusion": True}, "LSAF"),
        # RSDM's original decoder-isolation guard is intentionally earlier
        # than DAFU1 and names this legacy decoder module explicitly.
        ({"use_edge_mask_consistency_refiner": True}, "edge-mask consistency refiner"),
        ({"stage4_mamba_spatial_reduction": 2}, "full-channel"),
    )
    for override, expected in invalid:
        try:
            _build(DAFU1_CONFIG, **override)
        except ValueError as error:
            assert expected.lower() in str(error).lower(), (override, error)
        else:
            raise AssertionError(f"DAFU1 guard accepted {override}")

    torch.manual_seed(1811)
    baseline = _build(DPSM_CONFIG).eval()
    torch.manual_seed(1811)
    disabled = _build(DPSM_CONFIG, use_difference_aware_up1_fusion=False).eval()
    _assert_dict_equal(baseline.state_dict(), disabled.state_dict())
    image = torch.randn(1, 3, 64, 80, device=DEVICE)
    with torch.inference_mode():
        _assert_dict_equal(baseline(image), disabled(image))
        _assert_dict_equal(baseline.forward_mask_only(image), disabled.forward_mask_only(image))


def test_dafu1_topology_shapes_and_shared_initialization() -> None:
    torch.manual_seed(1812)
    baseline = _build(DPSM_CONFIG).eval()
    torch.manual_seed(1812)
    model = _build(DAFU1_CONFIG).eval()
    decoder = model.decoder
    assert model.use_difference_aware_up1_fusion and decoder.use_difference_aware_up1_fusion
    assert not model.use_full_width_residual_mask_head and not model.use_depthwise_separable_mask_head
    assert isinstance(decoder.up1, DifferenceAwareSubpixelUpBlock)
    assert isinstance(decoder.up2, LowResolutionSubpixelSeparableConcatUpBlock)
    assert isinstance(decoder.up3, LowResolutionSubpixelSeparableConcatUpBlock)
    assert decoder.up1.in_channels == 80 and decoder.up1.skip_channels == decoder.up1.out_channels == 40
    assert decoder.up1.delta_depthwise.groups == 40
    assert decoder.up1.delta_depthwise.kernel_size == (3, 3)
    assert decoder.up1.delta_pointwise.in_channels == decoder.up1.delta_pointwise.out_channels == 40
    assert not hasattr(decoder.up1, "depthwise") and not hasattr(decoder.up1, "pointwise")
    assert isinstance(decoder.mask_head, nn.Sequential)
    dense = decoder.mask_head[0].block[0]
    assert isinstance(dense, nn.Conv2d)
    assert dense.in_channels == dense.out_channels == 40 and dense.kernel_size == (3, 3) and dense.groups == 1
    assert isinstance(model.stage4.route_mamba, DirectionPartitionedSharedMamba)
    assert isinstance(model.stage4.global_branch, TransformerGlobalBlock)

    baseline_state, candidate_state = baseline.state_dict(), model.state_dict()
    removed = ("decoder.up1.depthwise.", "decoder.up1.depthwise_norm.", "decoder.up1.pointwise.", "decoder.up1.pointwise_norm.")
    for key, value in baseline_state.items():
        if not key.startswith(removed):
            assert key in candidate_state and torch.equal(value, candidate_state[key]), key
    new_keys = set(candidate_state) - set(baseline_state)
    assert new_keys and all(key.startswith("decoder.up1.delta_") for key in new_keys)

    decoder.up1.set_debug_shapes(True)
    image = torch.randn(1, 3, 64, 80, device=DEVICE)
    with torch.inference_mode():
        output = model(image)
    expected_shapes = {
        "input": (1, 80, 32, 40), "skip": (1, 40, 64, 80), "u": (1, 40, 64, 80),
        "s": (1, 40, 64, 80), "base": (1, 40, 64, 80), "delta": (1, 40, 64, 80), "out": (1, 40, 64, 80),
    }
    assert decoder.up1.last_debug_shapes == expected_shapes
    assert output["mask"].shape == (1, 1, 64, 80)


def test_dafu1_forward_mask_only_and_backward() -> None:
    torch.manual_seed(1813)
    model = _build(DAFU1_CONFIG).eval()
    sizes = [(1, 64, 80), (3, 128, 96)]
    if DEVICE.type == "cuda":
        sizes.append((1, 384, 544))
    for batch_size, height, width in sizes:
        image = torch.randn(batch_size, 3, height, width, device=DEVICE)
        calls = {"edge": 0, "reliability": 0}
        handles = [
            model.decoder.edge_head.register_forward_pre_hook(
                lambda _module, _inputs: calls.__setitem__("edge", calls["edge"] + 1)
            ),
            model.decoder.reliability_head.register_forward_pre_hook(
                lambda _module, _inputs: calls.__setitem__("reliability", calls["reliability"] + 1)
            ),
        ]
        try:
            with torch.inference_mode():
                full = model(image)
                mask_only = model.forward_mask_only(image)
        finally:
            for handle in handles:
                handle.remove()
        assert calls == {"edge": 1, "reliability": 1}
        assert full["mask"].shape == full["edge"].shape == full["reliability"].shape == (batch_size, 1, height, width)
        assert torch.equal(full["mask"], mask_only["mask"])

    model.train()
    batch = _batch(1, 64, 80)
    loss, _ = DEMambaLoss(**_load(DAFU1_CONFIG)["loss"]).to(DEVICE)(model(batch["image"]), batch, epoch=0)
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.decoder.up1.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad).item() > 0, name
    optimizer = make_optimizer(model, _load(DAFU1_CONFIG)["train"], DEVICE, logging.getLogger(__name__))
    parameter_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    optimizer_ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    assert len(optimizer_ids) == len(set(optimizer_ids))
    assert set(optimizer_ids) == parameter_ids
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test() -> None:
    checks = (
        test_dafu1_config_is_a_strict_dpsm80_single_variable_change,
        test_dafu1_guards_and_disabled_dpsm80_compatibility,
        test_dafu1_topology_shapes_and_shared_initialization,
        test_dafu1_forward_mask_only_and_backward,
    )
    for check in checks:
        check()
        print(f"PASS: {check.__name__}")


if __name__ == "__main__":
    test()
