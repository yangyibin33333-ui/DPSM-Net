"""Strict regression and CUDA one-step checks for DPER-DSMH.

Run from the repository root with
``python tests/test_depthwise_separable_mask_head.py``.
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
from models.decoder import DepthwiseSeparableMaskHead, LowResolutionSubpixelSeparableConcatUpBlock
from train import make_optimizer


DPER_CONFIG = (
    ROOT / "configs" / "experiments" / "lightweight_c40_s4mamba_lsscf_dper"
    / "C3_C40_S4MAMBA_LSSCF_DPER_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
DSMH_CONFIG = (
    ROOT / "configs" / "experiments" / "lightweight_c40_s4mamba_lsscf_dper_dsmh"
    / "C3_C40_S4MAMBA_LSSCF_DPER_DSMH_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
FAILED_FLAGS = (
    "use_subpixel_phase_consistency_refiner",
    "use_haar_topology_preserving_rebase",
    "use_narrow_dual_resolution_stage1",
    "use_full_resolution_joint_exchange",
    "use_early_serpentine_connectivity_ssm",
    "use_partial_full_resolution_stage1",
    "use_multiscale_directional_reallocation",
)
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


def _assert_tensor_dict_equal(left: dict, right: dict) -> None:
    assert set(left) == set(right)
    for key, value in left.items():
        other = right[key]
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, other), key
        else:
            assert value == other, key


def _assert_finite_nonzero_gradient(module: nn.Module, label: str) -> None:
    gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
    assert gradients and all(gradient is not None for gradient in gradients), f"{label}: missing gradient"
    present = [gradient for gradient in gradients if gradient is not None]
    assert all(torch.isfinite(gradient).all() for gradient in present), f"{label}: non-finite gradient"
    assert all(torch.count_nonzero(gradient).item() > 0 for gradient in present), f"{label}: zero gradient"


def _synthetic_batch(batch_size: int, height: int, width: int) -> dict[str, torch.Tensor]:
    mask = (torch.rand(batch_size, 1, height, width, device=RUNTIME_DEVICE) < 0.08).float()
    edge = (torch.rand(batch_size, 1, height, width, device=RUNTIME_DEVICE) < 0.04).float()
    direction_index = torch.randint(0, 4, (batch_size, 1, height, width), device=RUNTIME_DEVICE)
    tangent = torch.zeros(batch_size, 4, height, width, device=RUNTIME_DEVICE)
    tangent.scatter_(1, direction_index, 1.0)
    return {
        "image": torch.randn(batch_size, 3, height, width, device=RUNTIME_DEVICE),
        "mask": mask,
        "edge_gt": edge,
        "tangent_gt": tangent,
        "tangent_valid": torch.ones(batch_size, 1, height, width, device=RUNTIME_DEVICE),
    }


def test_dsmh_config_is_a_strict_dper_single_variable_change() -> None:
    dper = _load_config(DPER_CONFIG)
    dsmh = _load_config(DSMH_CONFIG)
    assert {key: value for key, value in dsmh.items() if key not in {"model", "save"}} == {
        key: value for key, value in dper.items() if key not in {"model", "save"}
    }
    expected_model = copy.deepcopy(dper["model"])
    expected_model["use_depthwise_separable_mask_head"] = True
    assert dsmh["model"] == expected_model
    assert dsmh["model"]["use_detail_preserving_early_rebase"] is True
    assert dsmh["model"]["use_low_resolution_subpixel_separable_concat_fusion"] is True
    assert all(not bool(dsmh["model"].get(flag, False)) for flag in FAILED_FLAGS)
    assert dsmh["save"]["save_dir"] == (
        "./runs/pilot/lightweight_c40_s4mamba_lsscf_dper_dsmh/"
        "C3_C40_S4MAMBA_LSSCF_DPER_DSMH_LR150_PHOTO_NATIVE384x544_EMA_BPCM"
    )


def test_dsmh_dependency_and_mutual_exclusion_guards() -> None:
    invalid = [
        ({"use_detail_preserving_early_rebase": False}, "early rebase"),
        ({"use_low_resolution_subpixel_separable_concat_fusion": False}, "requires"),
        ({"base_channels": 32}, "40"),
        ({"use_subpixel_phase_consistency_refiner": True}, "SPCR"),
        ({"use_haar_topology_preserving_rebase": True}, "HTPR"),
        ({"use_narrow_dual_resolution_stage1": True}, "NDRS"),
        ({"use_full_resolution_joint_exchange": True}, "FJSE"),
        ({"use_early_serpentine_connectivity_ssm": True}, "ESCS"),
        ({"use_partial_full_resolution_stage1": True}, "PFRS"),
        (
            {
                "use_multiscale_directional_reallocation": True,
                "stage3_active_mamba_directions": ("horizontal", "vertical"),
                "stage4_active_mamba_directions": ("diagonal", "anti_diagonal"),
            },
            "MSDR",
        ),
        ({"use_low_resolution_subpixel_additive_fusion": True}, "optional decoder"),
        ({"use_frequency_gate": True}, "optional decoder"),
        ({"use_semantic_skip_gate": True}, "optional decoder"),
        ({"use_detail_residual_gate": True}, "optional decoder"),
        ({"use_structure_harmonic_fusion": True}, "optional decoder"),
        ({"use_boundary_logit_calibration": True}, "optional decoder"),
        ({"use_sdf_head": True}, "optional decoder"),
        ({"use_edge_mask_consistency_refiner": True}, "optional decoder"),
        ({"use_deep_supervision": True}, "optional decoder"),
        ({"use_route_guided_directional_refiner": True}, "optional decoder"),
        ({"use_route_guided_anisotropic_diffusion": True}, "optional decoder"),
        ({"use_local_semantic_continuity_calibrator": True}, "optional decoder"),
        ({"use_local_pixel_reassembly_residual": True}, "optional decoder"),
        ({"use_local_spectral_confidence_rejector": True}, "optional decoder"),
        ({"use_high_resolution_spectral_detail_gate": True}, "optional decoder"),
        ({"use_local_route_gate": True}, "encoder ablations"),
        ({"use_residual_branch_competition_fusion": True}, "encoder ablations"),
        ({"use_deep_identity_decay": True}, "encoder ablations"),
        ({"mamba_stages": (3, 4)}, "topology"),
        ({"transformer_stages": ()}, "topology"),
        ({"stage4_mamba_spatial_reduction": 2}, "full-channel"),
        ({"stage4_mamba_channels": 160}, "full-channel"),
        ({"stage4_active_mamba_directions": ("horizontal", "vertical")}, "four-direction"),
    ]
    for override, expected in invalid:
        try:
            _build(DSMH_CONFIG, **override)
        except ValueError as error:
            assert expected.lower() in str(error).lower(), (override, error)
        else:
            raise AssertionError(f"DSMH guard accepted invalid override: {override}")


def test_dsmh_false_flag_is_bitwise_dper_equivalent() -> None:
    torch.manual_seed(1301)
    dper = _build(DPER_CONFIG).eval()
    torch.manual_seed(1301)
    disabled = _build(DPER_CONFIG, use_depthwise_separable_mask_head=False).eval()
    dper_params = sum(parameter.numel() for parameter in dper.parameters())
    assert sum(parameter.numel() for parameter in disabled.parameters()) == dper_params
    if RUNTIME_DEVICE.type == "cuda":
        assert dper_params == 5_740_751
    _assert_tensor_dict_equal(dper.state_dict(), disabled.state_dict())
    image = torch.randn(1, 3, 64, 80, device=RUNTIME_DEVICE)
    with torch.inference_mode():
        _assert_tensor_dict_equal(dper(image), disabled(image))
        _assert_tensor_dict_equal(dper.forward_mask_only(image), disabled.forward_mask_only(image))
        _assert_tensor_dict_equal(dper.extract_encoder_features(image), disabled.extract_encoder_features(image))


def test_dsmh_structure_shared_initialization_and_no_dense_head() -> None:
    torch.manual_seed(1302)
    dper = _build(DPER_CONFIG).eval()
    torch.manual_seed(1302)
    model = _build(DSMH_CONFIG).eval()
    head = model.decoder.mask_head
    assert model.use_depthwise_separable_mask_head
    assert model.decoder.use_depthwise_separable_mask_head
    assert isinstance(head, DepthwiseSeparableMaskHead)
    assert isinstance(model.decoder.up1, LowResolutionSubpixelSeparableConcatUpBlock)
    depthwise = head.depthwise.block[0]
    pointwise = head.pointwise.block[0]
    assert isinstance(depthwise, nn.Conv2d)
    assert depthwise.in_channels == depthwise.out_channels == depthwise.groups == 40
    assert depthwise.kernel_size == (3, 3) and depthwise.dilation == (1, 1)
    assert isinstance(head.depthwise.block[1], nn.BatchNorm2d)
    assert isinstance(head.depthwise.block[2], nn.GELU)
    assert isinstance(pointwise, nn.Conv2d)
    assert pointwise.in_channels == pointwise.out_channels == 40
    assert pointwise.kernel_size == (1, 1) and pointwise.groups == 1
    assert isinstance(head.pointwise.block[1], nn.BatchNorm2d)
    assert isinstance(head.pointwise.block[2], nn.GELU)
    assert head.output.in_channels == 40 and head.output.out_channels == 1
    assert head.output.kernel_size == (1, 1)
    assert not hasattr(head, "alpha") and not hasattr(head, "gate")
    assert not any(
        isinstance(module, nn.Conv2d)
        and module.in_channels == module.out_channels == 40
        and module.kernel_size == (3, 3)
        and module.groups == 1
        for module in head.modules()
    )
    dper_params = sum(parameter.numel() for parameter in dper.parameters())
    candidate_params = sum(parameter.numel() for parameter in model.parameters())
    assert candidate_params == dper_params - 12_360
    if RUNTIME_DEVICE.type == "cuda":
        assert candidate_params == 5_728_391

    dper_state = dper.state_dict()
    candidate_state = model.state_dict()
    for key, value in dper_state.items():
        if not key.startswith("decoder.mask_head."):
            assert key in candidate_state and torch.equal(value, candidate_state[key]), key
    assert not any(key.startswith("decoder.mask_head.0.") for key in candidate_state)
    assert not any(key.startswith("original_mask_head.") for key in candidate_state)
    assert all(key.startswith("decoder.mask_head.") for key in set(candidate_state) - set(dper_state))


def test_dsmh_shapes_same_instance_and_mask_only_auxiliary_skip() -> None:
    torch.manual_seed(1303)
    model = _build(DSMH_CONFIG).eval()
    head = model.decoder.mask_head
    for height, width in ((64, 80), (384, 544), (512, 512), (128, 96)):
        image = torch.randn(1, 3, height, width, device=RUNTIME_DEVICE)
        head_inputs: list[tuple[int, ...]] = []
        calls = {"mask": 0, "edge": 0, "reliability": 0}

        def capture_head(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            calls["mask"] += 1
            head_inputs.append(tuple(inputs[0].shape))

        handles = [
            head.register_forward_pre_hook(capture_head),
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
            assert calls == {"mask": 1, "edge": 1, "reliability": 1}
            calls.update(mask=0, edge=0, reliability=0)
            with torch.inference_mode():
                mask_only = model.forward_mask_only(image)
        finally:
            for handle in handles:
                handle.remove()
        assert calls == {"mask": 1, "edge": 0, "reliability": 0}
        assert head_inputs == [(1, 40, height, width), (1, 40, height, width)]
        assert full["mask"].shape == mask_only["mask"].shape == (1, 1, height, width)
        assert full["edge"].shape == full["reliability"].shape == (1, 1, height, width)
        assert torch.equal(full["mask"], mask_only["mask"])


def test_dsmh_small_backward_and_optimizer_coverage() -> None:
    torch.manual_seed(1304)
    model = _build(DSMH_CONFIG).train()
    config = _load_config(DSMH_CONFIG)
    criterion = DEMambaLoss(**config["loss"]).to(RUNTIME_DEVICE)
    batch = _synthetic_batch(1, 64, 80)
    outputs = model(batch["image"])
    loss, _ = criterion(outputs, batch, epoch=0)
    loss.backward()
    head = model.decoder.mask_head
    assert isinstance(head, DepthwiseSeparableMaskHead)
    _assert_finite_nonzero_gradient(head.depthwise.block[0], "depthwise convolution")
    _assert_finite_nonzero_gradient(head.depthwise.block[1], "depthwise BN")
    _assert_finite_nonzero_gradient(head.pointwise.block[0], "pointwise convolution")
    _assert_finite_nonzero_gradient(head.pointwise.block[1], "pointwise BN")
    _assert_finite_nonzero_gradient(head.output, "output convolution")
    _assert_finite_nonzero_gradient(model.decoder.refine, "decoder refine")
    _assert_finite_nonzero_gradient(model.stem, "DPER encoder stem")

    dsmh_missing = [
        name for name, parameter in head.named_parameters() if parameter.requires_grad and parameter.grad is None
    ]
    assert not dsmh_missing
    parameter_ids = [id(parameter) for parameter in model.parameters() if parameter.requires_grad]
    assert len(parameter_ids) == len(set(parameter_ids))
    optimizer = make_optimizer(model, config["train"], RUNTIME_DEVICE, logging.getLogger(__name__))
    optimizer_ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    assert len(optimizer_ids) == len(set(optimizer_ids))
    assert set(optimizer_ids) == set(parameter_ids)
    if RUNTIME_DEVICE.type == "cuda":
        assert all(group["fused"] for group in optimizer.param_groups)


def test_dsmh_formal_batch8_amp_fused_adamw_step() -> None:
    if RUNTIME_DEVICE.type != "cuda":
        return
    torch.manual_seed(1305)
    model = _build(DSMH_CONFIG).train()
    config = _load_config(DSMH_CONFIG)
    assert model.mamba_backend_info()["stage4"] == ["mamba_ssm"] * 4
    criterion = DEMambaLoss(**config["loss"]).to(RUNTIME_DEVICE)
    optimizer = make_optimizer(model, config["train"], RUNTIME_DEVICE, logging.getLogger(__name__))
    assert all(group["fused"] for group in optimizer.param_groups)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    batch = _synthetic_batch(8, 384, 544)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model(batch["image"])
        assert outputs["mask"].shape == outputs["edge"].shape == outputs["reliability"].shape == (8, 1, 384, 544)
        loss, _ = criterion(outputs, batch, epoch=0)
    assert torch.isfinite(loss)
    scaler.scale(loss).backward()
    head = model.decoder.mask_head
    assert isinstance(head, DepthwiseSeparableMaskHead)
    _assert_finite_nonzero_gradient(head, "formal DSMH step")
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test() -> None:
    checks = (
        test_dsmh_config_is_a_strict_dper_single_variable_change,
        test_dsmh_dependency_and_mutual_exclusion_guards,
        test_dsmh_false_flag_is_bitwise_dper_equivalent,
        test_dsmh_structure_shared_initialization_and_no_dense_head,
        test_dsmh_shapes_same_instance_and_mask_only_auxiliary_skip,
        test_dsmh_small_backward_and_optimizer_coverage,
        test_dsmh_formal_batch8_amp_fused_adamw_step,
    )
    for check in checks:
        check()
        print(f"PASS: {check.__name__}")


if __name__ == "__main__":
    test()
