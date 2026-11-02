"""Regression checks for the isolated DPSM80-FWRMH mask-head experiment.

Run from the repository root with
``python tests/test_full_width_residual_mask_head.py``.
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

from losses import DEMambaLoss
from models import build_model
from models.decoder import FullWidthResidualMaskHead, LowResolutionSubpixelSeparableConcatUpBlock
from models.mamba_blocks import DirectionPartitionedSharedMamba
from models.transformer_blocks import TransformerGlobalBlock
from train import make_optimizer


DPSM_CONFIG = (
    ROOT / "configs" / "experiments" / "lightweight_c40_s4mamba_lsscf_dper_rsdm_dpsm"
    / "C3_C40_S4MAMBA_LSSCF_DPER_RSDM_DPSM80_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
FWRMH_CONFIG = (
    ROOT / "configs" / "experiments" / "lightweight_c40_s4mamba_lsscf_dper_rsdm_dpsm_fwrmh"
    / "C3_C40_S4MAMBA_LSSCF_DPER_RSDM_DPSM80_FWRMH_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
RUNTIME_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build(path: Path, **overrides: object) -> torch.nn.Module:
    config = copy.deepcopy(_load(path)["model"])
    config.update(overrides)
    name = config.pop("name")
    if RUNTIME_DEVICE.type != "cuda":
        config["use_mamba_ssm"] = False
    return build_model(name, **config).to(RUNTIME_DEVICE)


def _batch(batch_size: int, height: int, width: int) -> dict[str, torch.Tensor]:
    direction = torch.randint(0, 4, (batch_size, 1, height, width), device=RUNTIME_DEVICE)
    tangent = torch.zeros(batch_size, 4, height, width, device=RUNTIME_DEVICE)
    tangent.scatter_(1, direction, 1.0)
    return {
        "image": torch.randn(batch_size, 3, height, width, device=RUNTIME_DEVICE),
        "mask": (torch.rand(batch_size, 1, height, width, device=RUNTIME_DEVICE) < 0.1).float(),
        "edge_gt": (torch.rand(batch_size, 1, height, width, device=RUNTIME_DEVICE) < 0.05).float(),
        "tangent_gt": tangent,
        "tangent_valid": torch.ones(batch_size, 1, height, width, device=RUNTIME_DEVICE),
    }


def _assert_dict_equal(left: dict, right: dict) -> None:
    assert set(left) == set(right)
    for key, value in left.items():
        assert torch.equal(value, right[key]), key


def test_fwrmh_config_is_a_strict_dpsm80_single_variable_change() -> None:
    baseline, candidate = _load(DPSM_CONFIG), _load(FWRMH_CONFIG)
    assert {key: value for key, value in candidate.items() if key not in {"model", "save"}} == {
        key: value for key, value in baseline.items() if key not in {"model", "save"}
    }
    expected_model = copy.deepcopy(baseline["model"])
    expected_model["use_full_width_residual_mask_head"] = True
    assert candidate["model"] == expected_model
    assert candidate["save"]["save_dir"] == (
        "./runs/pilot/lightweight_c40_s4mamba_lsscf_dper_rsdm_dpsm_fwrmh/"
        "C3_C40_S4MAMBA_LSSCF_DPER_RSDM_DPSM80_FWRMH_LR150_PHOTO_NATIVE384x544_EMA_BPCM"
    )


def test_fwrmh_guards_and_disabled_dpsm80_compatibility() -> None:
    invalid = (
        ({"use_depthwise_separable_mask_head": True}, "DSMH"),
        ({"use_direction_partitioned_shared_mamba": False}, "DPSM80"),
        ({"use_common_private_direction_partitioned_shared_mamba": True}, "mutually exclusive"),
        ({"use_low_resolution_subpixel_separable_concat_fusion": False}, "LSSCF"),
        ({"base_channels": 32}, "C40"),
        ({"use_subpixel_phase_consistency_refiner": True}, "SPCR"),
        ({"use_narrow_transport_decoder": True}, "NTD"),
        ({"use_frequency_gate": True}, "optional decoder"),
        ({"stage4_mamba_spatial_reduction": 2}, "full-channel"),
    )
    for override, expected in invalid:
        try:
            _build(FWRMH_CONFIG, **override)
        except ValueError as error:
            assert expected.lower() in str(error).lower(), (override, error)
        else:
            raise AssertionError(f"FWRMH guard accepted {override}")

    torch.manual_seed(1711)
    baseline = _build(DPSM_CONFIG).eval()
    torch.manual_seed(1711)
    disabled = _build(DPSM_CONFIG, use_full_width_residual_mask_head=False).eval()
    _assert_dict_equal(baseline.state_dict(), disabled.state_dict())
    image = torch.randn(1, 3, 64, 80, device=RUNTIME_DEVICE)
    with torch.inference_mode():
        _assert_dict_equal(baseline(image), disabled(image))
        _assert_dict_equal(baseline.forward_mask_only(image), disabled.forward_mask_only(image))


def test_fwrmh_structure_shared_initialization_and_checkpoint_contract() -> None:
    torch.manual_seed(1712)
    baseline = _build(DPSM_CONFIG).eval()
    torch.manual_seed(1712)
    model = _build(FWRMH_CONFIG).eval()
    head = model.decoder.mask_head
    assert model.use_full_width_residual_mask_head and model.decoder.use_full_width_residual_mask_head
    assert not model.use_depthwise_separable_mask_head
    assert isinstance(head, FullWidthResidualMaskHead)
    assert head.channels == 40 and head.hidden_channels == 16
    assert isinstance(head.reduce.block[0], nn.Conv2d)
    assert head.reduce.block[0].in_channels == 40 and head.reduce.block[0].out_channels == 16
    assert head.reduce.block[0].kernel_size == (1, 1) and head.reduce.block[0].groups == 1
    assert isinstance(head.spatial.block[0], nn.Conv2d)
    assert head.spatial.block[0].in_channels == head.spatial.block[0].out_channels == head.spatial.block[0].groups == 16
    assert head.spatial.block[0].kernel_size == (3, 3)
    assert isinstance(head.expand[0], nn.Conv2d)
    assert head.expand[0].in_channels == 16 and head.expand[0].out_channels == 40 and head.expand[0].groups == 1
    assert isinstance(head.classifier, nn.Conv2d)
    assert head.classifier.in_channels == 40 and head.classifier.out_channels == 1
    assert not any(
        isinstance(module, nn.Conv2d)
        and module.in_channels == module.out_channels == 40
        and module.kernel_size == (3, 3)
        and module.groups == 1
        for module in head.modules()
    )
    assert isinstance(model.decoder.up1, LowResolutionSubpixelSeparableConcatUpBlock)
    assert isinstance(model.stage4.route_mamba, DirectionPartitionedSharedMamba)
    assert isinstance(model.stage4.global_branch, TransformerGlobalBlock)

    baseline_state, candidate_state = baseline.state_dict(), model.state_dict()
    for key, value in baseline_state.items():
        if not key.startswith("decoder.mask_head."):
            assert key in candidate_state and torch.equal(value, candidate_state[key]), key
    assert all(key.startswith("decoder.mask_head.") for key in set(candidate_state) - set(baseline_state))
    try:
        model.load_state_dict(baseline_state, strict=True)
    except RuntimeError as error:
        message = str(error)
        assert "decoder.mask_head" in message and "Missing key" in message and "Unexpected key" in message
    else:
        raise AssertionError("A historical dense-head checkpoint must not strict-load into FWRMH silently.")


def test_fwrmh_forward_shapes_mask_only_and_batch_isolation() -> None:
    torch.manual_seed(1713)
    model = _build(FWRMH_CONFIG).eval()
    head = model.decoder.mask_head
    assert isinstance(head, FullWidthResidualMaskHead)
    sizes = [(1, 64, 80), (3, 128, 96)]
    if RUNTIME_DEVICE.type == "cuda":
        sizes.append((1, 384, 544))
    for batch_size, height, width in sizes:
        image = torch.randn(batch_size, 3, height, width, device=RUNTIME_DEVICE)
        calls = {"mask": 0, "edge": 0, "reliability": 0}
        features: list[tuple[int, ...]] = []

        def capture_mask(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            calls["mask"] += 1
            features.append(tuple(inputs[0].shape))

        handles = [
            head.register_forward_pre_hook(capture_mask),
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
        assert calls == {"mask": 2, "edge": 1, "reliability": 1}
        assert features == [(batch_size, 40, height, width), (batch_size, 40, height, width)]
        assert full["mask"].shape == full["edge"].shape == full["reliability"].shape == (batch_size, 1, height, width)
        assert torch.equal(full["mask"], mask_only["mask"])


def test_fwrmh_backward_and_optimizer_coverage() -> None:
    torch.manual_seed(1714)
    model = _build(FWRMH_CONFIG).train()
    config = _load(FWRMH_CONFIG)
    batch = _batch(1, 64, 80)
    outputs = model(batch["image"])
    loss, _ = DEMambaLoss(**config["loss"]).to(RUNTIME_DEVICE)(outputs, batch, epoch=0)
    assert torch.isfinite(loss)
    loss.backward()
    head = model.decoder.mask_head
    assert isinstance(head, FullWidthResidualMaskHead)
    for name, parameter in head.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad).item() > 0, name
    optimizer = make_optimizer(model, config["train"], RUNTIME_DEVICE, logging.getLogger(__name__))
    parameter_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    optimizer_ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    assert len(optimizer_ids) == len(set(optimizer_ids))
    assert set(optimizer_ids) == parameter_ids


def test() -> None:
    checks = (
        test_fwrmh_config_is_a_strict_dpsm80_single_variable_change,
        test_fwrmh_guards_and_disabled_dpsm80_compatibility,
        test_fwrmh_structure_shared_initialization_and_checkpoint_contract,
        test_fwrmh_forward_shapes_mask_only_and_batch_isolation,
        test_fwrmh_backward_and_optimizer_coverage,
    )
    for check in checks:
        check()
        print(f"PASS: {check.__name__}")


if __name__ == "__main__":
    test()
