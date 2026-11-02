"""Regression tests for the isolated DPER-RSDM-PNTD24 candidate."""

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
from models.blocks import RIFTStructuralDirectionalModelingBlock
from models.decoder import LowResolutionSubpixelSeparableConcatUpBlock, PartialNarrowTransportDecoder
from train import make_optimizer


RSDM_CONFIG = (
    ROOT / "configs" / "experiments" / "lightweight_c40_s4mamba_lsscf_dper_rsdm"
    / "C3_C40_S4MAMBA_LSSCF_DPER_RSDM_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
PNTD_CONFIG = (
    ROOT / "configs" / "experiments" / "lightweight_c40_s4mamba_lsscf_dper_rsdm_pntd"
    / "C3_C40_S4MAMBA_LSSCF_DPER_RSDM_PNTD24_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
RUNTIME_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build(path: Path, **overrides: object) -> nn.Module:
    config = copy.deepcopy(_load(path)["model"])
    config.update(overrides)
    name = config.pop("name")
    if RUNTIME_DEVICE.type != "cuda":
        config["use_mamba_ssm"] = False
    return build_model(name, **config).to(RUNTIME_DEVICE)


def _synthetic_batch(height: int = 64, width: int = 80) -> dict[str, torch.Tensor]:
    direction = torch.randint(0, 4, (1, 1, height, width), device=RUNTIME_DEVICE)
    tangent = torch.zeros(1, 4, height, width, device=RUNTIME_DEVICE)
    tangent.scatter_(1, direction, 1.0)
    return {
        "image": torch.randn(1, 3, height, width, device=RUNTIME_DEVICE),
        "mask": (torch.rand(1, 1, height, width, device=RUNTIME_DEVICE) < 0.08).float(),
        "edge_gt": (torch.rand(1, 1, height, width, device=RUNTIME_DEVICE) < 0.04).float(),
        "tangent_gt": tangent,
        "tangent_valid": torch.ones(1, 1, height, width, device=RUNTIME_DEVICE),
    }


def _assert_nonzero_finite_gradients(module: nn.Module, label: str) -> None:
    gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
    assert gradients and all(gradient is not None for gradient in gradients), f"{label}: missing gradient"
    assert all(torch.isfinite(gradient).all() for gradient in gradients), f"{label}: non-finite gradient"
    assert all(torch.count_nonzero(gradient).item() > 0 for gradient in gradients), f"{label}: zero gradient"


def test_pntd_config_is_strictly_isolated() -> None:
    baseline, candidate = _load(RSDM_CONFIG), _load(PNTD_CONFIG)
    assert {key: value for key, value in candidate.items() if key not in {"model", "save"}} == {
        key: value for key, value in baseline.items() if key not in {"model", "save"}
    }
    expected = copy.deepcopy(baseline["model"])
    expected.update(use_partial_narrow_transport_decoder=True, partial_transport_dim=24)
    assert candidate["model"] == expected
    assert "rsdm_pntd" in candidate["save"]["save_dir"]


def test_pntd_guards_and_rsdm_disabled_compatibility() -> None:
    invalid = (
        ({"use_rsdm_stage1": False}, "RSDM"),
        ({"use_narrow_transport_decoder": True}, "mutually exclusive"),
        ({"partial_transport_dim": 32}, "fixed to 24"),
        ({"use_low_resolution_subpixel_separable_concat_fusion": False}, "LSSCF"),
        ({"use_subpixel_phase_consistency_refiner": True}, "SPCR"),
        ({"use_haar_topology_preserving_rebase": True}, "HTPR"),
        ({"use_depthwise_separable_mask_head": True}, "DSMH"),
        ({"stage4_mamba_channels": 160}, "full-channel"),
        ({"stage4_active_mamba_directions": ("horizontal", "vertical")}, "four-direction"),
    )
    for override, message in invalid:
        try:
            _build(PNTD_CONFIG, **override)
        except ValueError as error:
            assert message.lower() in str(error).lower(), (override, error)
        else:
            raise AssertionError(f"PNTD guard accepted {override}")

    torch.manual_seed(1600)
    baseline = _build(RSDM_CONFIG).eval()
    torch.manual_seed(1600)
    disabled = _build(RSDM_CONFIG, use_partial_narrow_transport_decoder=False).eval()
    assert baseline.state_dict().keys() == disabled.state_dict().keys()
    assert all(torch.equal(value, disabled.state_dict()[key]) for key, value in baseline.state_dict().items())


def test_pntd_structure_parameter_replacement_and_shared_initialization() -> None:
    torch.manual_seed(1601)
    baseline = _build(RSDM_CONFIG).eval()
    torch.manual_seed(1601)
    model = _build(PNTD_CONFIG).eval()
    decoder, pntd = model.decoder, model.decoder.partial_narrow_transport_decoder
    assert isinstance(pntd, PartialNarrowTransportDecoder)
    assert decoder.up3 is None and decoder.up2 is None
    assert isinstance(decoder.up1, LowResolutionSubpixelSeparableConcatUpBlock)
    assert decoder.refine is not None and decoder.narrow_transport_decoder is None
    assert pntd.transport_dim == 24
    assert (pntd.f4_channels, pntd.f3_channels, pntd.f2_channels) == (320, 160, 80)
    assert isinstance(model.rsdm_stage1, RIFTStructuralDirectionalModelingBlock)
    assert sum(isinstance(module, RIFTStructuralDirectionalModelingBlock) for module in model.modules()) == 1
    assert sum(isinstance(module, PartialNarrowTransportDecoder) for module in model.modules()) == 1
    assert not any(isinstance(module, nn.PixelShuffle) for module in pntd.modules())

    dense_mask = decoder.mask_head[0].block[0]
    assert dense_mask.in_channels == dense_mask.out_channels == 40
    assert dense_mask.kernel_size == (3, 3) and dense_mask.groups == 1
    assert sum(parameter.numel() for parameter in pntd.parameters()) == 47_232
    baseline_params = sum(parameter.numel() for parameter in baseline.parameters())
    candidate_params = sum(parameter.numel() for parameter in model.parameters())
    assert candidate_params == baseline_params - 310_528
    if RUNTIME_DEVICE.type == "cuda":
        assert candidate_params == 5_436_983
        assert model.mamba_backend_info()["stage4"] == ["mamba_ssm"] * 4

    baseline_state, candidate_state = baseline.state_dict(), model.state_dict()
    replaced = ("decoder.up3.", "decoder.up2.")
    for key, value in baseline_state.items():
        if not key.startswith(replaced):
            assert key in candidate_state and torch.equal(value, candidate_state[key]), key
    assert not any(key.startswith(replaced) for key in candidate_state)
    for suffix in (
        "weight", "bias", "running_mean", "running_var", "num_batches_tracked"
    ):
        pairs = (
            (f"decoder.up2.skip_projection.{suffix}", f"decoder.partial_narrow_transport_decoder.f2_skip_projection.{suffix}"),
            (f"decoder.up2.depthwise.{suffix}", f"decoder.partial_narrow_transport_decoder.f2_depthwise.{suffix}"),
            (f"decoder.up2.depthwise_norm.{suffix}", f"decoder.partial_narrow_transport_decoder.f2_depthwise_norm.{suffix}"),
            (f"decoder.up2.pointwise.{suffix}", f"decoder.partial_narrow_transport_decoder.f2_pointwise.{suffix}"),
            (f"decoder.up2.pointwise_norm.{suffix}", f"decoder.partial_narrow_transport_decoder.f2_pointwise_norm.{suffix}"),
        )
        for old, new in pairs:
            if old in baseline_state:
                assert new in candidate_state and torch.equal(baseline_state[old], candidate_state[new]), (old, new)


def test_pntd_shapes_preserve_f2_f1_and_all_forward_contracts() -> None:
    torch.manual_seed(1602)
    model = _build(PNTD_CONFIG).eval()
    pntd = model.decoder.partial_narrow_transport_decoder
    assert isinstance(pntd, PartialNarrowTransportDecoder)
    observed: dict[str, tuple[int, ...]] = {}
    handles = [
        model.decoder.up1.register_forward_pre_hook(
            lambda _module, inputs: observed.update(up1_decoder=tuple(inputs[0].shape), f1_skip=tuple(inputs[1].shape))
        ),
        model.decoder.refine.register_forward_pre_hook(
            lambda _module, inputs: observed.__setitem__("refine_input", tuple(inputs[0].shape))
        ),
        model.decoder.mask_head.register_forward_pre_hook(
            lambda _module, inputs: observed.__setitem__("mask_head_input", tuple(inputs[0].shape))
        ),
    ]
    image = torch.randn(1, 3, 64, 80, device=RUNTIME_DEVICE)
    try:
        with torch.inference_mode():
            features = model.extract_encoder_features(image)
            full = model(image)
            mask_only = model.forward_mask_only(image)
    finally:
        for handle in handles:
            handle.remove()
    assert features["f1"].shape == (1, 40, 64, 80)
    assert features["f2"].shape == (1, 80, 32, 40)
    assert features["f3"].shape == (1, 160, 16, 20)
    assert features["f4"].shape == (1, 320, 8, 10)
    shapes = pntd.last_debug_shapes
    assert shapes["f4_narrow"] == (1, 24, 8, 10)
    assert shapes["f3_native_narrow"] == (1, 24, 16, 20)
    assert shapes["f3_concat"] == (1, 48, 16, 20)
    assert shapes["f3_semantic"] == (1, 160, 16, 20)
    assert shapes["f3_transport"] == (1, 24, 16, 20)
    assert shapes["f2_deep_semantic"] == shapes["f2_native"] == (1, 80, 32, 40)
    assert shapes["f2_concat"] == (1, 160, 32, 40)
    assert shapes["f2_output"] == observed["up1_decoder"] == (1, 80, 32, 40)
    assert observed["f1_skip"] == observed["refine_input"] == observed["mask_head_input"] == (1, 40, 64, 80)
    assert full["mask"].shape == full["edge"].shape == full["reliability"].shape == (1, 1, 64, 80)
    assert {"mask", "edge", "reliability", "route_s4"}.issubset(full)
    assert torch.equal(full["mask"], mask_only["mask"])

    native = torch.randn(1, 3, 384, 544, device=RUNTIME_DEVICE)
    with torch.inference_mode():
        assert model(native)["mask"].shape == (1, 1, 384, 544)


def test_pntd_backward_and_optimizer_cover_every_live_parameter() -> None:
    torch.manual_seed(1603)
    model = _build(PNTD_CONFIG).train()
    config = _load(PNTD_CONFIG)
    batch = _synthetic_batch()
    outputs = model(batch["image"])
    loss, _ = DEMambaLoss(**config["loss"]).to(RUNTIME_DEVICE)(outputs, batch, epoch=0)
    assert torch.isfinite(loss)
    loss.backward()
    pntd = model.decoder.partial_narrow_transport_decoder
    assert isinstance(pntd, PartialNarrowTransportDecoder)
    for name in (
        "f4_projection", "f3_projection", "f3_depthwise", "f3_pointwise",
        "f2_decoder_projection", "f2_transport_adapter", "f2_skip_projection",
        "f2_depthwise", "f2_pointwise",
    ):
        _assert_nonzero_finite_gradients(getattr(pntd, name), f"PNTD {name}")
    _assert_nonzero_finite_gradients(model.rsdm_stage1, "RSDM")
    _assert_nonzero_finite_gradients(model.decoder.up1, "preserved F1 fusion")
    _assert_nonzero_finite_gradients(model.decoder.refine, "preserved final refine")
    _assert_nonzero_finite_gradients(model.decoder.mask_head, "dense mask head")

    parameter_ids = [id(parameter) for parameter in model.parameters() if parameter.requires_grad]
    assert len(parameter_ids) == len(set(parameter_ids))
    optimizer = make_optimizer(model, config["train"], RUNTIME_DEVICE, logging.getLogger(__name__))
    optimizer_ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    assert len(optimizer_ids) == len(set(optimizer_ids))
    assert set(optimizer_ids) == set(parameter_ids)


def test() -> None:
    checks = (
        test_pntd_config_is_strictly_isolated,
        test_pntd_guards_and_rsdm_disabled_compatibility,
        test_pntd_structure_parameter_replacement_and_shared_initialization,
        test_pntd_shapes_preserve_f2_f1_and_all_forward_contracts,
        test_pntd_backward_and_optimizer_cover_every_live_parameter,
    )
    for check in checks:
        check()
        print(f"PASS: {check.__name__}")


if __name__ == "__main__":
    test()
