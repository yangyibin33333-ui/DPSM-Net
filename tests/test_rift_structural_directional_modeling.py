"""Regression tests for the isolated DPER-RSDM candidate."""

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
from models.blocks import DetailPreservingHalfStem, LocalCrackBlock, RIFTStructuralDirectionalModelingBlock
from models.decoder import LowResolutionSubpixelSeparableConcatUpBlock
from train import make_optimizer


DPER_CONFIG = (
    ROOT / "configs" / "experiments" / "lightweight_c40_s4mamba_lsscf_dper"
    / "C3_C40_S4MAMBA_LSSCF_DPER_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
RSDM_CONFIG = (
    ROOT / "configs" / "experiments" / "lightweight_c40_s4mamba_lsscf_dper_rsdm"
    / "C3_C40_S4MAMBA_LSSCF_DPER_RSDM_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
FAILED_FLAGS = (
    "use_subpixel_phase_consistency_refiner",
    "use_haar_topology_preserving_rebase",
    "use_narrow_dual_resolution_stage1",
    "use_full_resolution_joint_exchange",
    "use_early_serpentine_connectivity_ssm",
    "use_partial_full_resolution_stage1",
    "use_multiscale_directional_reallocation",
    "use_depthwise_separable_mask_head",
)
RUNTIME_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build(path: Path, **overrides: object) -> nn.Module:
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


def test_rsdm_config_is_a_strict_dper_single_variable_change() -> None:
    dper = _load_config(DPER_CONFIG)
    rsdm = _load_config(RSDM_CONFIG)
    assert {key: value for key, value in rsdm.items() if key not in {"model", "save"}} == {
        key: value for key, value in dper.items() if key not in {"model", "save"}
    }
    expected_model = copy.deepcopy(dper["model"])
    expected_model["use_rsdm_stage1"] = True
    assert rsdm["model"] == expected_model
    assert all(not bool(rsdm["model"].get(flag, False)) for flag in FAILED_FLAGS)
    assert rsdm["save"]["save_dir"] == (
        "./runs/pilot/lightweight_c40_s4mamba_lsscf_dper_rsdm/"
        "C3_C40_S4MAMBA_LSSCF_DPER_RSDM_LR150_PHOTO_NATIVE384x544_EMA_BPCM"
    )


def test_rsdm_dependency_and_mutual_exclusion_guards() -> None:
    invalid = [
        ({"use_detail_preserving_early_rebase": False}, "early rebase"),
        ({"base_channels": 32}, "C40"),
        ({"use_low_resolution_subpixel_separable_concat_fusion": False}, "LSSCF"),
        ({"use_subpixel_phase_consistency_refiner": True}, "SPCR"),
        ({"use_haar_topology_preserving_rebase": True}, "HTPR"),
        ({"use_narrow_dual_resolution_stage1": True}, "NDRS"),
        ({"use_full_resolution_joint_exchange": True}, "FJSE"),
        ({"use_early_serpentine_connectivity_ssm": True}, "ESCS"),
        ({"use_partial_full_resolution_stage1": True}, "PFRS"),
        ({"use_depthwise_separable_mask_head": True}, "DSMH"),
        (
            {
                "use_multiscale_directional_reallocation": True,
                "stage3_active_mamba_directions": ("horizontal", "vertical"),
                "stage4_active_mamba_directions": ("diagonal", "anti_diagonal"),
            },
            "MSDR",
        ),
        ({"use_low_resolution_subpixel_additive_fusion": True}, "original decoder"),
        ({"use_frequency_gate": True}, "original decoder"),
        ({"use_semantic_skip_gate": True}, "original decoder"),
        ({"use_detail_residual_gate": True}, "original decoder"),
        ({"use_structure_harmonic_fusion": True}, "original decoder"),
        ({"use_boundary_logit_calibration": True}, "original decoder"),
        ({"use_sdf_head": True}, "original decoder"),
        ({"use_edge_mask_consistency_refiner": True}, "original decoder"),
        ({"use_deep_supervision": True}, "original decoder"),
        ({"use_route_guided_directional_refiner": True}, "original decoder"),
        ({"use_route_guided_anisotropic_diffusion": True}, "original decoder"),
        ({"use_local_semantic_continuity_calibrator": True}, "original decoder"),
        ({"use_local_pixel_reassembly_residual": True}, "original decoder"),
        ({"use_local_spectral_confidence_rejector": True}, "original decoder"),
        ({"use_high_resolution_spectral_detail_gate": True}, "original decoder"),
        ({"use_local_route_gate": True}, "half-resolution"),
        ({"use_residual_branch_competition_fusion": True}, "half-resolution"),
        ({"mamba_stages": (3, 4)}, "topology"),
        ({"transformer_stages": ()}, "topology"),
        ({"stage4_mamba_spatial_reduction": 2}, "full-channel"),
        ({"stage4_mamba_channels": 160}, "full-channel"),
        ({"stage4_active_mamba_directions": ("horizontal", "vertical")}, "four-direction"),
    ]
    for override, expected in invalid:
        try:
            _build(RSDM_CONFIG, **override)
        except ValueError as error:
            assert expected.lower() in str(error).lower(), (override, error)
        else:
            raise AssertionError(f"RSDM guard accepted invalid override: {override}")


def test_rsdm_accepts_the_dependency_aware_local_stage4() -> None:
    """RSDM remains on DPER semantics when the paper ablation removes Stage4 Mamba."""

    model = _build(RSDM_CONFIG, mamba_stages=()).eval()
    assert isinstance(model.stage4, LocalCrackBlock)
    assert model.route_stage4 is False
    assert "stage4" not in model.mamba_backend_info()
    assert isinstance(model.rsdm_stage1, RIFTStructuralDirectionalModelingBlock)
    image = torch.randn(1, 3, 64, 80, device=RUNTIME_DEVICE)
    with torch.inference_mode():
        features = model.extract_encoder_features(image)
        outputs = model(image)
    assert features["route_s4"] is None
    assert "route_s4" not in outputs
    assert features["f1"].shape == (1, 40, 64, 80)


def test_rsdm_false_flag_is_bitwise_dper_equivalent() -> None:
    torch.manual_seed(1401)
    dper = _build(DPER_CONFIG).eval()
    torch.manual_seed(1401)
    disabled = _build(DPER_CONFIG, use_rsdm_stage1=False).eval()
    assert sum(parameter.numel() for parameter in dper.parameters()) == sum(
        parameter.numel() for parameter in disabled.parameters()
    )
    _assert_tensor_dict_equal(dper.state_dict(), disabled.state_dict())
    image = torch.randn(1, 3, 64, 80, device=RUNTIME_DEVICE)
    with torch.inference_mode():
        _assert_tensor_dict_equal(dper.extract_encoder_features(image), disabled.extract_encoder_features(image))
        _assert_tensor_dict_equal(dper(image), disabled(image))
        _assert_tensor_dict_equal(dper.forward_mask_only(image), disabled.forward_mask_only(image))


def test_rsdm_structure_shared_initialization_and_original_decoder() -> None:
    torch.manual_seed(1402)
    dper = _build(DPER_CONFIG).eval()
    torch.manual_seed(1402)
    model = _build(RSDM_CONFIG).eval()
    rsdm = model.rsdm_stage1
    assert model.use_rsdm_stage1 and isinstance(rsdm, RIFTStructuralDirectionalModelingBlock)
    assert sum(isinstance(module, RIFTStructuralDirectionalModelingBlock) for module in model.modules()) == 1
    assert rsdm.channels == 40 and rsdm.kernel_size == 13 and rsdm.dilation == 3
    assert rsdm.hidden_ratio == 1.0
    assert isinstance(rsdm.pre_norm, nn.GroupNorm)
    assert rsdm.input_projection.in_channels == rsdm.input_projection.out_channels == 40
    assert rsdm.local_structure.groups == 40 and rsdm.local_structure.kernel_size == (3, 3)
    assert rsdm.horizontal.groups == 40 and rsdm.horizontal.kernel_size == (1, 13)
    assert rsdm.vertical.groups == 40 and rsdm.vertical.kernel_size == (13, 1)
    assert rsdm.diagonal.groups == 40 and rsdm.diagonal.kernel_size == (3, 3)
    assert rsdm.diagonal.dilation == (3, 3) and rsdm.diagonal.padding == (3, 3)
    assert rsdm.structure_gate.kernel_size == (1, 1)
    assert not hasattr(rsdm, "alpha") and not any(isinstance(module, nn.BatchNorm2d) for module in rsdm.modules())
    assert sum(parameter.numel() for parameter in rsdm.parameters()) == 6_760
    assert isinstance(model.decoder.up1, LowResolutionSubpixelSeparableConcatUpBlock)
    assert isinstance(model.decoder.mask_head, nn.Sequential)
    dense_mask = model.decoder.mask_head[0].block[0]
    assert dense_mask.in_channels == dense_mask.out_channels == 40
    assert dense_mask.kernel_size == (3, 3) and dense_mask.groups == 1
    assert set(model.mamba_backend_info()) == {"stage4"}
    assert len(model.mamba_backend_info()["stage4"]) == 4
    if RUNTIME_DEVICE.type == "cuda":
        assert model.mamba_backend_info()["stage4"] == ["mamba_ssm"] * 4

    dper_state, rsdm_state = dper.state_dict(), model.state_dict()
    for key, value in dper_state.items():
        assert key in rsdm_state and torch.equal(value, rsdm_state[key]), key
    assert all(key.startswith("rsdm_stage1.") for key in set(rsdm_state) - set(dper_state))
    dper_params = sum(parameter.numel() for parameter in dper.parameters())
    assert sum(parameter.numel() for parameter in model.parameters()) == dper_params + 6_760


def test_rsdm_full_detail_integrity_insertion_and_shapes() -> None:
    torch.manual_seed(1403)
    dper = _build(DPER_CONFIG).eval()
    torch.manual_seed(1403)
    model = _build(RSDM_CONFIG).eval()
    assert isinstance(model.stem, DetailPreservingHalfStem)
    image = torch.randn(1, 3, 64, 80, device=RUNTIME_DEVICE)
    with torch.no_grad():
        baseline_detail, baseline_semantic = dper.stem(image)
        candidate_detail, candidate_semantic = model.stem(image)
    assert torch.equal(baseline_detail, candidate_detail)
    assert torch.equal(baseline_semantic, candidate_semantic)

    observed: dict[str, torch.Tensor | tuple[torch.Tensor, ...]] = {}
    handles = [
        model.stem.pixel_unshuffle.register_forward_hook(
            lambda _module, _inputs, output: observed.__setitem__("pixel_unshuffle", output.detach().clone())
        ),
        model.stem.register_forward_hook(
            lambda _module, _inputs, output: observed.update(
                full_detail=output[0].detach().clone(), semantic_half=output[1].detach().clone()
            )
        ),
        model.stage1.register_forward_hook(
            lambda _module, _inputs, output: observed.__setitem__("pre_rsdm", output.detach().clone())
        ),
        model.rsdm_stage1.register_forward_hook(
            lambda _module, inputs, output: observed.update(
                rsdm_input=inputs[0].detach().clone(), rsdm_output=output.detach().clone()
            )
        ),
        model.f1_reconstruction.register_forward_pre_hook(
            lambda _module, inputs: observed.update(
                reconstruction_detail=inputs[0].detach().clone(), reconstruction_semantic=inputs[1].detach().clone()
            )
        ),
        model.stage2.register_forward_pre_hook(
            lambda _module, inputs: observed.__setitem__("stage2_input", inputs[0].detach().clone())
        ),
    ]
    try:
        with torch.inference_mode():
            features = model.extract_encoder_features(image)
    finally:
        for handle in handles:
            handle.remove()
    assert observed["pixel_unshuffle"].shape == (1, 16, 32, 40)
    assert observed["full_detail"].shape == (1, 16, 64, 80)
    assert observed["semantic_half"].shape == (1, 40, 32, 40)
    assert observed["pre_rsdm"].shape == observed["rsdm_input"].shape == (1, 40, 32, 40)
    assert torch.equal(observed["pre_rsdm"], observed["rsdm_input"])
    assert observed["rsdm_output"].shape == (1, 40, 32, 40)
    assert torch.equal(observed["full_detail"], observed["reconstruction_detail"])
    assert torch.equal(observed["rsdm_output"], observed["reconstruction_semantic"])
    assert observed["stage2_input"].shape == (1, 80, 32, 40)
    assert features["f1"].shape == (1, 40, 64, 80)
    assert features["f2"].shape == (1, 80, 32, 40)
    assert features["f3"].shape == (1, 160, 16, 20)
    assert features["f4"].shape == (1, 320, 8, 10)

    for height, width in ((128, 96), (384, 544)):
        probe = torch.randn(1, 3, height, width, device=RUNTIME_DEVICE)
        with torch.inference_mode():
            full = model(probe)
            mask_only = model.forward_mask_only(probe)
        assert full["mask"].shape == mask_only["mask"].shape == (1, 1, height, width)
        assert full["edge"].shape == full["reliability"].shape == (1, 1, height, width)
        assert torch.equal(full["mask"], mask_only["mask"])


def test_rsdm_forward_backward_gradients_and_optimizer_coverage() -> None:
    torch.manual_seed(1404)
    model = _build(RSDM_CONFIG).train()
    config = _load_config(RSDM_CONFIG)
    criterion = DEMambaLoss(**config["loss"]).to(RUNTIME_DEVICE)
    batch = _synthetic_batch(1, 64, 80)
    outputs = model(batch["image"])
    loss, _ = criterion(outputs, batch, epoch=0)
    assert torch.isfinite(loss)
    loss.backward()
    rsdm = model.rsdm_stage1
    assert isinstance(rsdm, RIFTStructuralDirectionalModelingBlock)
    for name in (
        "pre_norm",
        "input_projection",
        "local_structure",
        "horizontal",
        "vertical",
        "diagonal",
        "structure_gate",
        "output_norm",
        "output_projection",
    ):
        _assert_finite_nonzero_gradient(getattr(rsdm, name), f"RSDM {name}")
    assert all(torch.isfinite(parameter.grad).all() for parameter in rsdm.parameters() if parameter.grad is not None)

    parameter_ids = [id(parameter) for parameter in model.parameters() if parameter.requires_grad]
    assert len(parameter_ids) == len(set(parameter_ids))
    optimizer = make_optimizer(model, config["train"], RUNTIME_DEVICE, logging.getLogger(__name__))
    optimizer_ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    assert len(optimizer_ids) == len(set(optimizer_ids))
    assert set(optimizer_ids) == set(parameter_ids)
    if RUNTIME_DEVICE.type == "cuda":
        assert all(group["fused"] for group in optimizer.param_groups)
        assert model.mamba_backend_info()["stage4"] == ["mamba_ssm"] * 4


def test() -> None:
    checks = (
        test_rsdm_config_is_a_strict_dper_single_variable_change,
        test_rsdm_dependency_and_mutual_exclusion_guards,
        test_rsdm_accepts_the_dependency_aware_local_stage4,
        test_rsdm_false_flag_is_bitwise_dper_equivalent,
        test_rsdm_structure_shared_initialization_and_original_decoder,
        test_rsdm_full_detail_integrity_insertion_and_shapes,
        test_rsdm_forward_backward_gradients_and_optimizer_coverage,
    )
    for check in checks:
        check()
        print(f"PASS: {check.__name__}")


if __name__ == "__main__":
    test()
