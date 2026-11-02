"""Regression and one-step checks for the strict DPER-MSDR candidate.

Run from the repository root with
``python tests/test_multiscale_directional_reallocation.py``.
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
from models.blocks import LocalCrackBlock, TopoRouteBlock, TransformerGlobalBlock
from models.mamba_blocks import DirectionalMambaPath, SequenceMambaCore
from train import make_optimizer


DPER_CONFIG = (
    ROOT / "configs" / "experiments" / "lightweight_c40_s4mamba_lsscf_dper"
    / "C3_C40_S4MAMBA_LSSCF_DPER_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
MSDR_CONFIG = (
    ROOT / "configs" / "experiments" / "lightweight_c40_s4mamba_lsscf_dper_msdr"
    / "C3_C40_S4MAMBA_LSSCF_DPER_MSDR_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
STAGE3_DIRECTIONS = ("horizontal", "vertical")
STAGE4_DIRECTIONS = ("diagonal", "anti_diagonal")
FAILED_FLAGS = (
    "use_subpixel_phase_consistency_refiner",
    "use_haar_topology_preserving_rebase",
    "use_narrow_dual_resolution_stage1",
    "use_full_resolution_joint_exchange",
    "use_early_serpentine_connectivity_ssm",
    "use_partial_full_resolution_stage1",
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


def _assert_module_has_finite_nonzero_gradient(module: nn.Module, label: str) -> None:
    gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
    present = [gradient for gradient in gradients if gradient is not None]
    assert present, f"{label}: no gradients"
    assert all(torch.isfinite(gradient).all() for gradient in present), f"{label}: non-finite gradient"
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in present), f"{label}: all-zero gradient"


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


def test_msdr_config_is_isolated_from_original_dper() -> None:
    dper = _load_config(DPER_CONFIG)
    msdr = _load_config(MSDR_CONFIG)
    assert {key: value for key, value in msdr.items() if key not in {"model", "save"}} == {
        key: value for key, value in dper.items() if key not in {"model", "save"}
    }
    expected_model = copy.deepcopy(dper["model"])
    expected_model.update(
        {
            "use_multiscale_directional_reallocation": True,
            "stage3_active_mamba_directions": ["horizontal", "vertical"],
            "stage4_active_mamba_directions": ["diagonal", "anti_diagonal"],
            "stage3_mamba_window_size": None,
            "stage3_bidirectional_mamba": False,
            "stage4_mamba_spatial_reduction": 1,
            "stage4_mamba_channels": None,
        }
    )
    assert msdr["model"] == expected_model
    assert all(not bool(msdr["model"].get(flag, False)) for flag in FAILED_FLAGS)
    assert msdr["save"]["save_dir"] == (
        "./runs/pilot/lightweight_c40_s4mamba_lsscf_dper_msdr/"
        "C3_C40_S4MAMBA_LSSCF_DPER_MSDR_LR150_PHOTO_NATIVE384x544_EMA_BPCM"
    )


def test_msdr_dependency_and_fixed_topology_guards() -> None:
    common = {
        "use_multiscale_directional_reallocation": True,
        "stage3_active_mamba_directions": STAGE3_DIRECTIONS,
        "stage4_active_mamba_directions": STAGE4_DIRECTIONS,
    }
    invalid = [
        ({"use_detail_preserving_early_rebase": False}, "requires"),
        ({"use_subpixel_phase_consistency_refiner": True}, "SPCR"),
        ({"use_haar_topology_preserving_rebase": True}, "HTPR"),
        ({"use_narrow_dual_resolution_stage1": True}, "NDRS"),
        ({"use_full_resolution_joint_exchange": True}, "FJSE"),
        ({"use_early_serpentine_connectivity_ssm": True}, "ESCS"),
        ({"use_partial_full_resolution_stage1": True}, "PFRS"),
        ({"stage3_mamba_window_size": 8}, "windowed"),
        ({"stage3_bidirectional_mamba": True}, "bidirectional"),
        ({"stage4_mamba_spatial_reduction": 2}, "reduction"),
        ({"stage4_mamba_channels": 160}, "channels"),
        ({"stage3_active_mamba_directions": ("diagonal", "anti_diagonal")}, "Stage3 directions"),
        ({"stage4_active_mamba_directions": ("horizontal", "vertical")}, "Stage4 directions"),
        ({"transformer_stages": (3, 4)}, "Transformer"),
        ({"transformer_stages": ()}, "Transformer"),
        ({"use_local_route_gate": True}, "local route"),
        ({"use_residual_branch_competition_fusion": True}, "competition"),
        ({"use_deep_identity_decay": True}, "identity"),
        ({"use_density_balanced_prototype_context": True}, "prototype"),
        ({"use_semantic_conditioned_local_kernel_residual": True}, "semantic-conditioned"),
    ]
    for override, expected in invalid:
        candidate = dict(common)
        candidate.update(override)
        try:
            _build(DPER_CONFIG, **candidate)
        except ValueError as error:
            assert expected.lower() in str(error).lower(), (override, error)
        else:
            raise AssertionError(f"MSDR guard accepted invalid override: {override}")


def test_msdr_false_flag_is_bitwise_dper_equivalent() -> None:
    torch.manual_seed(1201)
    dper = _build(DPER_CONFIG).eval()
    torch.manual_seed(1201)
    disabled = _build(DPER_CONFIG, use_multiscale_directional_reallocation=False).eval()
    assert sum(parameter.numel() for parameter in dper.parameters()) == sum(
        parameter.numel() for parameter in disabled.parameters()
    )
    _assert_tensor_dict_equal(dper.state_dict(), disabled.state_dict())
    image = torch.randn(1, 3, 64, 80, device=RUNTIME_DEVICE)
    with torch.inference_mode():
        dper_features = dper.extract_encoder_features(image)
        disabled_features = disabled.extract_encoder_features(image)
        dper_full = dper(image)
        disabled_full = disabled(image)
        dper_mask = dper.forward_mask_only(image)
        disabled_mask = disabled.forward_mask_only(image)
    _assert_tensor_dict_equal(dper_features, disabled_features)
    _assert_tensor_dict_equal(dper_full, disabled_full)
    _assert_tensor_dict_equal(dper_mask, disabled_mask)


def test_msdr_structure_core_count_and_shared_initialization() -> None:
    torch.manual_seed(1202)
    dper = _build(DPER_CONFIG).eval()
    torch.manual_seed(1202)
    model = _build(MSDR_CONFIG).eval()
    assert model.use_multiscale_directional_reallocation
    assert isinstance(dper.stage3, LocalCrackBlock)
    assert isinstance(model.stage3, TopoRouteBlock) and model.route_stage3
    assert isinstance(model.stage4, TopoRouteBlock) and model.route_stage4

    stage3_route = model.stage3.route_mamba
    stage4_route = model.stage4.route_mamba
    assert model.stage3.channels == 160 and model.stage4.channels == 320
    assert stage3_route.active_mamba_directions == STAGE3_DIRECTIONS
    assert stage4_route.active_mamba_directions == STAGE4_DIRECTIONS
    assert stage3_route.active_mamba_path_count == stage4_route.active_mamba_path_count == 2
    assert stage3_route.spatial_reduction == stage4_route.spatial_reduction == 1
    assert stage3_route.mamba_channels == 160 and stage4_route.mamba_channels == 320
    assert isinstance(model.stage3.global_branch, nn.Identity)
    assert isinstance(model.stage4.global_branch, TransformerGlobalBlock)
    assert model.stage3.residual_branch_competition_fusion is None
    assert model.stage4.residual_branch_competition_fusion is None
    assert model.local_route_gates is None
    assert model.deep_identity_scales == {"stage3": 1.0, "stage4": 1.0}
    assert all(type(path) is DirectionalMambaPath for path in [*stage3_route.paths, *stage4_route.paths])
    assert stage3_route.local_compensations is not None
    assert stage4_route.local_compensations is not None
    assert tuple(stage3_route.local_compensations.keys()) == STAGE4_DIRECTIONS
    assert tuple(stage4_route.local_compensations.keys()) == STAGE3_DIRECTIONS
    assert sum(isinstance(module, SequenceMambaCore) for module in model.modules()) == 4
    assert set(model.mamba_backend_info()) == {"stage3", "stage4"}
    assert sum(parameter.numel() for parameter in model.parameters()) != 5_740_751
    if RUNTIME_DEVICE.type == "cuda":
        assert model.mamba_backend_info()["stage3"] == ["mamba_ssm"] * 2
        assert model.mamba_backend_info()["stage4"] == ["mamba_ssm"] * 2

    shared_prefixes = ("stem.", "stage1.", "down1.", "stage2.", "down2.", "down3.", "decoder.", "f1_reconstruction.")
    dper_state, msdr_state = dper.state_dict(), model.state_dict()
    for key, value in dper_state.items():
        if key.startswith(shared_prefixes):
            assert key in msdr_state and torch.equal(value, msdr_state[key]), key
    assert not any(key.startswith("original_stage3.") or key.startswith("original_stage4.") for key in msdr_state)

    model_parameter_ids = [id(parameter) for parameter in model.parameters() if parameter.requires_grad]
    assert len(model_parameter_ids) == len(set(model_parameter_ids))
    optimizer = make_optimizer(model, _load_config(MSDR_CONFIG)["train"], RUNTIME_DEVICE, logging.getLogger(__name__))
    optimizer_ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    assert len(optimizer_ids) == len(set(optimizer_ids))
    assert set(optimizer_ids) == set(model_parameter_ids)
    if RUNTIME_DEVICE.type == "cuda":
        assert all(group["fused"] for group in optimizer.param_groups)


def test_msdr_calls_shapes_and_all_encoder_entrypoints() -> None:
    torch.manual_seed(1203)
    model = _build(MSDR_CONFIG).eval()
    path_calls = {
        "stage3.horizontal": 0,
        "stage3.vertical": 0,
        "stage4.diagonal": 0,
        "stage4.anti_diagonal": 0,
    }
    handles = []
    for stage_name in ("stage3", "stage4"):
        route = getattr(model, stage_name).route_mamba
        for direction, path in zip(route.active_mamba_directions, route.paths, strict=True):
            key = f"{stage_name}.{direction}"

            def count(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], *, call_key: str = key) -> None:
                path_calls[call_key] += 1

            handles.append(path.core.register_forward_pre_hook(count))
    image = torch.randn(1, 3, 64, 80, device=RUNTIME_DEVICE)
    try:
        with torch.inference_mode():
            mask_only = model.forward_mask_only(image)
    finally:
        for handle in handles:
            handle.remove()
    assert path_calls == {key: 1 for key in path_calls}
    assert mask_only["mask"].shape == (1, 1, 64, 80)
    assert model.stage3.route_mamba.last_executed_mamba_directions == STAGE3_DIRECTIONS
    assert model.stage4.route_mamba.last_executed_mamba_directions == STAGE4_DIRECTIONS
    assert model.stage3.route_mamba.last_executed_local_compensation_directions == STAGE4_DIRECTIONS
    assert model.stage4.route_mamba.last_executed_local_compensation_directions == STAGE3_DIRECTIONS

    calls = {"stage3": 0, "stage4": 0}
    stage_handles = [
        model.stage3.register_forward_pre_hook(lambda _module, _inputs: calls.__setitem__("stage3", calls["stage3"] + 1)),
        model.stage4.register_forward_pre_hook(lambda _module, _inputs: calls.__setitem__("stage4", calls["stage4"] + 1)),
    ]
    try:
        with torch.inference_mode():
            features = model.extract_encoder_features(image)
            full = model(image)
            mask_only = model.forward_mask_only(image)
    finally:
        for handle in stage_handles:
            handle.remove()
    assert calls == {"stage3": 3, "stage4": 3}
    assert torch.equal(full["mask"], mask_only["mask"])
    assert set(key for key in full if key.startswith("route_")) == {"route_s3", "route_s4"}
    assert features["route_s3"].shape == (1, 4, 16, 20)
    assert features["route_s4"].shape == (1, 4, 8, 10)

    for height, width in ((384, 544), (512, 512), (128, 96)):
        probe = torch.randn(1, 3, height, width, device=RUNTIME_DEVICE)
        with torch.inference_mode():
            features = model.extract_encoder_features(probe)
            output = model.forward_mask_only(probe)
        assert features["f1"].shape == (1, 40, height, width)
        assert features["f2"].shape == (1, 80, height // 2, width // 2)
        assert features["f3"].shape == (1, 160, height // 4, width // 4)
        assert features["f4"].shape == (1, 320, height // 8, width // 8)
        assert features["route_s3"].shape == (1, 4, height // 4, width // 4)
        assert features["route_s4"].shape == (1, 4, height // 8, width // 8)
        assert output["mask"].shape == (1, 1, height, width)


def test_msdr_small_backward_reaches_all_required_branches() -> None:
    torch.manual_seed(1204)
    model = _build(MSDR_CONFIG).train()
    criterion = DEMambaLoss(**_load_config(MSDR_CONFIG)["loss"]).to(RUNTIME_DEVICE)
    batch = _synthetic_batch(1, 64, 80)
    outputs = model(batch["image"])
    assert "route_s3" in outputs and "route_s4" in outputs
    loss, _ = criterion(outputs, batch, epoch=0)
    loss.backward()
    for direction, path in zip(STAGE3_DIRECTIONS, model.stage3.route_mamba.paths, strict=True):
        _assert_module_has_finite_nonzero_gradient(path, f"Stage3 {direction} Mamba")
    _assert_module_has_finite_nonzero_gradient(model.stage3.route_mamba.route_head, "Stage3 route head")
    _assert_module_has_finite_nonzero_gradient(model.stage3.local, "Stage3 local")
    _assert_module_has_finite_nonzero_gradient(model.stage3.fuse, "Stage3 fuse")
    for direction, path in zip(STAGE4_DIRECTIONS, model.stage4.route_mamba.paths, strict=True):
        _assert_module_has_finite_nonzero_gradient(path, f"Stage4 {direction} Mamba")
    _assert_module_has_finite_nonzero_gradient(model.stage4.route_mamba.route_head, "Stage4 route head")
    _assert_module_has_finite_nonzero_gradient(model.stage4.global_branch, "Stage4 Transformer")
    _assert_module_has_finite_nonzero_gradient(model.stage4.fuse, "Stage4 fuse")
    _assert_module_has_finite_nonzero_gradient(model.decoder, "decoder")
    missing_gradients = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    }
    # The unchanged DPER loss does not consume the reliability prediction, so
    # its output head is executed by the full forward but intentionally has no
    # loss gradient.  This is baseline behavior, not an orphaned MSDR/replaced
    # Stage parameter.  No other trainable parameter may be disconnected.
    assert missing_gradients == {
        "decoder.reliability_head.weight",
        "decoder.reliability_head.bias",
    }, f"unexpected trainable parameters outside the full-loss path: {sorted(missing_gradients)}"


def test_msdr_formal_batch8_amp_fused_adamw_step() -> None:
    if RUNTIME_DEVICE.type != "cuda":
        return
    torch.manual_seed(1205)
    model = _build(MSDR_CONFIG).train()
    config = _load_config(MSDR_CONFIG)
    criterion = DEMambaLoss(**config["loss"]).to(RUNTIME_DEVICE)
    optimizer = make_optimizer(model, config["train"], RUNTIME_DEVICE, logging.getLogger(__name__))
    assert all(group["fused"] for group in optimizer.param_groups)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    batch = _synthetic_batch(8, 384, 544)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model(batch["image"])
        loss, _ = criterion(outputs, batch, epoch=0)
    assert torch.isfinite(loss)
    scaler.scale(loss).backward()
    for stage_name, directions in (("stage3", STAGE3_DIRECTIONS), ("stage4", STAGE4_DIRECTIONS)):
        stage = getattr(model, stage_name)
        for direction, path in zip(directions, stage.route_mamba.paths, strict=True):
            _assert_module_has_finite_nonzero_gradient(path, f"formal step {stage_name} {direction}")
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test() -> None:
    checks = (
        test_msdr_config_is_isolated_from_original_dper,
        test_msdr_dependency_and_fixed_topology_guards,
        test_msdr_false_flag_is_bitwise_dper_equivalent,
        test_msdr_structure_core_count_and_shared_initialization,
        test_msdr_calls_shapes_and_all_encoder_entrypoints,
        test_msdr_small_backward_reaches_all_required_branches,
        test_msdr_formal_batch8_amp_fused_adamw_step,
    )
    for check in checks:
        check()
        print(f"PASS: {check.__name__}")


if __name__ == "__main__":
    test()
