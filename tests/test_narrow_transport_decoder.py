"""Strict regression checks for DPER-RSDM-NTD24."""

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
from models.decoder import NarrowTransportDecoder
from train import make_optimizer


RSDM_CONFIG = (
    ROOT / "configs" / "experiments" / "lightweight_c40_s4mamba_lsscf_dper_rsdm"
    / "C3_C40_S4MAMBA_LSSCF_DPER_RSDM_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
NTD_CONFIG = (
    ROOT / "configs" / "experiments" / "lightweight_c40_s4mamba_lsscf_dper_rsdm_ntd"
    / "C3_C40_S4MAMBA_LSSCF_DPER_RSDM_NTD24_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
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


def _assert_finite_nonzero_gradient(module: nn.Module, label: str) -> None:
    gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
    assert gradients and all(gradient is not None for gradient in gradients), f"{label}: missing gradient"
    present = [gradient for gradient in gradients if gradient is not None]
    assert all(torch.isfinite(gradient).all() for gradient in present), f"{label}: non-finite gradient"
    assert all(torch.count_nonzero(gradient).item() > 0 for gradient in present), f"{label}: zero gradient"


def _synthetic_batch(batch_size: int, height: int, width: int) -> dict[str, torch.Tensor]:
    direction_index = torch.randint(0, 4, (batch_size, 1, height, width), device=RUNTIME_DEVICE)
    tangent = torch.zeros(batch_size, 4, height, width, device=RUNTIME_DEVICE)
    tangent.scatter_(1, direction_index, 1.0)
    return {
        "image": torch.randn(batch_size, 3, height, width, device=RUNTIME_DEVICE),
        "mask": (torch.rand(batch_size, 1, height, width, device=RUNTIME_DEVICE) < 0.08).float(),
        "edge_gt": (torch.rand(batch_size, 1, height, width, device=RUNTIME_DEVICE) < 0.04).float(),
        "tangent_gt": tangent,
        "tangent_valid": torch.ones(batch_size, 1, height, width, device=RUNTIME_DEVICE),
    }


def test_ntd_config_changes_only_model_transport_and_save() -> None:
    rsdm = _load_config(RSDM_CONFIG)
    ntd = _load_config(NTD_CONFIG)
    assert {key: value for key, value in ntd.items() if key not in {"model", "save"}} == {
        key: value for key, value in rsdm.items() if key not in {"model", "save"}
    }
    expected_model = copy.deepcopy(rsdm["model"])
    expected_model.update({"use_narrow_transport_decoder": True, "decoder_transport_dim": 24})
    assert ntd["model"] == expected_model
    assert ntd["save"]["save_dir"] == (
        "./runs/pilot/lightweight_c40_s4mamba_lsscf_dper_rsdm_ntd/"
        "C3_C40_S4MAMBA_LSSCF_DPER_RSDM_NTD24_LR150_PHOTO_NATIVE384x544_EMA_BPCM"
    )


def test_ntd_guards_require_isolated_rsdm_and_fixed_widths() -> None:
    invalid = [
        ({"use_rsdm_stage1": False}, "requires"),
        ({"use_detail_preserving_early_rebase": False}, "DPER"),
        ({"base_channels": 32}, "C40"),
        ({"use_low_resolution_subpixel_separable_concat_fusion": False}, "LSSCF"),
        ({"decoder_transport_dim": 20}, "24"),
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
        ({"use_frequency_gate": True}, "decoder"),
        ({"use_semantic_skip_gate": True}, "decoder"),
        ({"use_detail_residual_gate": True}, "decoder"),
        ({"use_structure_harmonic_fusion": True}, "decoder"),
        ({"use_boundary_logit_calibration": True}, "decoder"),
        ({"use_edge_mask_consistency_refiner": True}, "decoder"),
        ({"use_deep_supervision": True}, "decoder"),
        ({"use_local_route_gate": True}, "half-resolution"),
        ({"mamba_stages": (3, 4)}, "topology"),
        ({"transformer_stages": ()}, "topology"),
        ({"stage4_mamba_spatial_reduction": 2}, "full-channel"),
        ({"stage4_mamba_channels": 160}, "full-channel"),
    ]
    for override, expected in invalid:
        try:
            _build(NTD_CONFIG, **override)
        except ValueError as error:
            assert expected.lower() in str(error).lower(), (override, error)
        else:
            raise AssertionError(f"NTD guard accepted invalid override: {override}")


def test_ntd_structure_unregistration_shared_initialization_and_dense_head() -> None:
    torch.manual_seed(1501)
    rsdm_model = _build(RSDM_CONFIG).eval()
    torch.manual_seed(1501)
    model = _build(NTD_CONFIG).eval()
    decoder = model.decoder
    ntd = decoder.narrow_transport_decoder
    assert model.use_narrow_transport_decoder and decoder.use_narrow_transport_decoder
    assert isinstance(ntd, NarrowTransportDecoder)
    assert ntd.transport_dim == 24 and ntd.output_channels == 40
    assert ntd.feature_channels == (40, 80, 160, 320)
    assert all(getattr(decoder, name) is None for name in ("proj4", "up3", "up2", "up1", "refine"))
    assert isinstance(model.rsdm_stage1, RIFTStructuralDirectionalModelingBlock)
    assert sum(isinstance(module, RIFTStructuralDirectionalModelingBlock) for module in model.modules()) == 1
    assert isinstance(decoder.mask_head, nn.Sequential)
    dense_mask = decoder.mask_head[0].block[0]
    assert dense_mask.in_channels == dense_mask.out_channels == 40
    assert dense_mask.kernel_size == (3, 3) and dense_mask.groups == 1
    assert sum(parameter.numel() for parameter in decoder.mask_head.parameters()) == 14_521
    assert sum(parameter.numel() for parameter in rsdm_model.decoder.mask_head.parameters()) == 14_521
    for key, value in rsdm_model.decoder.mask_head.state_dict().items():
        assert torch.equal(value, decoder.mask_head.state_dict()[key]), key
    for head_name in ("edge_head", "reliability_head"):
        baseline_state = getattr(rsdm_model.decoder, head_name).state_dict()
        candidate_state = getattr(decoder, head_name).state_dict()
        for key, value in baseline_state.items():
            assert torch.equal(value, candidate_state[key]), f"{head_name}.{key}"

    baseline_state, candidate_state = rsdm_model.state_dict(), model.state_dict()
    replaced_prefixes = (
        "decoder.up3.", "decoder.up2.", "decoder.up1.", "decoder.refine."
    )
    for key, value in baseline_state.items():
        if not key.startswith(replaced_prefixes):
            assert key in candidate_state and torch.equal(value, candidate_state[key]), key
    assert not any(key.startswith(replaced_prefixes) for key in candidate_state)
    assert all(
        key.startswith("decoder.narrow_transport_decoder.")
        for key in set(candidate_state) - set(baseline_state)
    )
    assert sum(parameter.numel() for parameter in ntd.parameters()) == 36_520
    baseline_params = sum(parameter.numel() for parameter in rsdm_model.parameters())
    candidate_params = sum(parameter.numel() for parameter in model.parameters())
    assert candidate_params == baseline_params - 344_040
    if RUNTIME_DEVICE.type == "cuda":
        assert baseline_params == 5_747_511
        assert candidate_params == 5_403_471
        assert model.mamba_backend_info()["stage4"] == ["mamba_ssm"] * 4


def test_ntd_shapes_dataflow_full_detail_and_shared_entrypoints() -> None:
    torch.manual_seed(1502)
    rsdm_model = _build(RSDM_CONFIG).eval()
    torch.manual_seed(1502)
    model = _build(NTD_CONFIG).eval()
    image = torch.randn(1, 3, 64, 80, device=RUNTIME_DEVICE)
    with torch.no_grad():
        baseline_detail, baseline_semantic = rsdm_model.stem(image)
        candidate_detail, candidate_semantic = model.stem(image)
    assert torch.equal(baseline_detail, candidate_detail)
    assert torch.equal(baseline_semantic, candidate_semantic)

    ntd = model.decoder.narrow_transport_decoder
    assert isinstance(ntd, NarrowTransportDecoder)
    observed: dict[str, tuple[int, ...]] = {}
    handles = []
    for name in ("proj1", "proj2", "proj3", "proj4", "refine4", "refine3", "refine2", "refine1", "adapter"):
        module = getattr(ntd, name)
        handles.append(
            module.register_forward_hook(
                lambda _module, _inputs, output, key=name: observed.__setitem__(key, tuple(output.shape))
            )
        )
    handles.append(
        model.decoder.mask_head.register_forward_pre_hook(
            lambda _module, inputs: observed.__setitem__("mask_head_input", tuple(inputs[0].shape))
        )
    )
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
    assert observed["proj1"] == observed["refine1"] == (1, 24, 64, 80)
    assert observed["proj2"] == observed["refine2"] == (1, 24, 32, 40)
    assert observed["proj3"] == observed["refine3"] == (1, 24, 16, 20)
    assert observed["proj4"] == observed["refine4"] == (1, 24, 8, 10)
    assert observed["adapter"] == observed["mask_head_input"] == (1, 40, 64, 80)
    assert full["mask"].shape == mask_only["mask"].shape == (1, 1, 64, 80)
    assert full["edge"].shape == full["reliability"].shape == (1, 1, 64, 80)
    assert torch.equal(full["mask"], mask_only["mask"])

    probe = torch.randn(1, 3, 384, 544, device=RUNTIME_DEVICE)
    with torch.inference_mode():
        output = model.forward_mask_only(probe)
    assert output["mask"].shape == (1, 1, 384, 544)


def test_ntd_forward_backward_all_branches_and_optimizer_coverage() -> None:
    torch.manual_seed(1503)
    model = _build(NTD_CONFIG).train()
    config = _load_config(NTD_CONFIG)
    criterion = DEMambaLoss(**config["loss"]).to(RUNTIME_DEVICE)
    batch = _synthetic_batch(1, 64, 80)
    outputs = model(batch["image"])
    loss, _ = criterion(outputs, batch, epoch=0)
    assert torch.isfinite(loss)
    loss.backward()
    ntd = model.decoder.narrow_transport_decoder
    assert isinstance(ntd, NarrowTransportDecoder)
    for name in ("proj1", "proj2", "proj3", "proj4", "refine4", "refine3", "refine2", "refine1", "adapter"):
        _assert_finite_nonzero_gradient(getattr(ntd, name), f"NTD {name}")
    _assert_finite_nonzero_gradient(model.rsdm_stage1, "preserved RSDM")
    _assert_finite_nonzero_gradient(model.decoder.mask_head, "original dense mask head")

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
        test_ntd_config_changes_only_model_transport_and_save,
        test_ntd_guards_require_isolated_rsdm_and_fixed_widths,
        test_ntd_structure_unregistration_shared_initialization_and_dense_head,
        test_ntd_shapes_dataflow_full_detail_and_shared_entrypoints,
        test_ntd_forward_backward_all_branches_and_optimizer_coverage,
    )
    for check in checks:
        check()
        print(f"PASS: {check.__name__}")


if __name__ == "__main__":
    test()
