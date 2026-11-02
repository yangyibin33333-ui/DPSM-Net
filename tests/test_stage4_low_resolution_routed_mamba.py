"""Regression checks for Stage4 low-resolution routed Mamba (S4-LRRM).

Run with: ``python tests/test_stage4_low_resolution_routed_mamba.py``.
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
from models.blocks import LocalCrackBlock, TopoRouteBlock, TransformerGlobalBlock
from models.mamba_blocks import TangentRoutedMambaBlock


C40_CONFIG = ROOT / "configs" / "experiments" / "C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
S4MAMBA_CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "lightweight_c40_s4mamba"
    / "C3_C40_S4MAMBA_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
LSAF_CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "lightweight_c40_s4mamba_lsaf"
    / "C3_C40_S4MAMBA_LSAF_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
LSSCF_CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "lightweight_c40_s4mamba_lsscf"
    / "C3_C40_S4MAMBA_LSSCF_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
LRRM_CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "lightweight_c40_s4mamba_lsscf_s4lrrm"
    / "C3_C40_S4MAMBA_LSSCF_S4LRRM2_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
ALL_DIRECTIONS = ("horizontal", "vertical", "diagonal", "anti_diagonal")
RUNTIME_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build_from_config(path: Path) -> torch.nn.Module:
    model_cfg = copy.deepcopy(_load_config(path)["model"])
    model_name = model_cfg.pop("name")
    # Selective scan is CUDA-only. The CPU fallback preserves the topology for
    # direct regression checks on hosts without a CUDA device.
    if RUNTIME_DEVICE.type != "cuda":
        model_cfg["use_mamba_ssm"] = False
    return build_model(model_name, **model_cfg).to(RUNTIME_DEVICE)


def _fallback_block(*, spatial_reduction: int = 1) -> TangentRoutedMambaBlock:
    return TangentRoutedMambaBlock(16, use_mamba_ssm=False, spatial_reduction=spatial_reduction)


def test_reduction_one_is_bitwise_compatible_and_state_dict_compatible() -> None:
    torch.manual_seed(13)
    legacy_default = _fallback_block().eval()
    torch.manual_seed(13)
    explicit_reduction_one = _fallback_block(spatial_reduction=1).eval()
    assert set(legacy_default.state_dict()) == set(explicit_reduction_one.state_dict())
    strict_result = explicit_reduction_one.load_state_dict(legacy_default.state_dict(), strict=True)
    assert not strict_result.missing_keys and not strict_result.unexpected_keys

    image = torch.randn(1, 16, 7, 9)
    with torch.inference_mode():
        legacy_routed, legacy_logits = legacy_default(image)
        reduced_routed, reduced_logits = explicit_reduction_one(image)
    assert torch.equal(legacy_routed, reduced_routed)
    assert torch.equal(legacy_logits, reduced_logits)


def test_reduction_two_preserves_routes_and_executes_all_paths_at_low_resolution() -> None:
    block = _fallback_block(spatial_reduction=2).eval()
    assert block.active_mamba_directions == ALL_DIRECTIONS
    assert block.active_mamba_path_count == 4
    assert block.local_compensations is None

    path_calls = [0, 0, 0, 0]
    path_sizes: list[tuple[int, int]] = []
    out_proj_sizes: list[tuple[int, int]] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_path_hook(index: int):
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            path_calls[index] += 1
            path_sizes.append(tuple(inputs[0].shape[-2:]))

        return hook

    for index, path in enumerate(block.paths):
        handles.append(path.register_forward_pre_hook(make_path_hook(index)))
    handles.append(
        block.out_proj.register_forward_pre_hook(
            lambda _module, inputs: out_proj_sizes.append(tuple(inputs[0].shape[-2:]))
        )
    )
    image = torch.randn(1, 16, 9, 11)
    try:
        with torch.inference_mode():
            routed, route_logits = block(image)
    finally:
        for handle in handles:
            handle.remove()
    assert route_logits.shape == (1, 4, 9, 11)
    assert routed.shape == image.shape
    assert path_calls == [1, 1, 1, 1]
    assert path_sizes == [(4, 5)] * 4
    assert out_proj_sizes == [(4, 5)]
    route_weights_low = block._pooled_route_weights(torch.softmax(route_logits, dim=1))
    assert route_weights_low.shape == (1, 4, 4, 5)
    assert torch.allclose(route_weights_low.sum(dim=1), torch.ones_like(route_weights_low.sum(dim=1)))


def test_reduction_two_handles_odd_shapes_and_rejects_too_small_inputs() -> None:
    block = _fallback_block(spatial_reduction=2).eval()
    with torch.inference_mode():
        routed, route_logits = block(torch.randn(1, 16, 5, 7))
    assert routed.shape == (1, 16, 5, 7)
    assert route_logits.shape == (1, 4, 5, 7)
    try:
        block(torch.randn(1, 16, 1, 4))
    except ValueError as error:
        assert "at least 2" in str(error)
    else:
        raise AssertionError("Reduction=2 must reject a spatial dimension smaller than two.")
    try:
        _fallback_block(spatial_reduction=0)
    except ValueError as error:
        assert "positive integer" in str(error)
    else:
        raise AssertionError("spatial_reduction=0 must be rejected.")


def test_lrrm_config_is_single_variable_change() -> None:
    baseline = _load_config(LSSCF_CONFIG)
    candidate = _load_config(LRRM_CONFIG)
    assert {key: value for key, value in candidate.items() if key not in {"model", "save"}} == {
        key: value for key, value in baseline.items() if key not in {"model", "save"}
    }
    expected_model = copy.deepcopy(baseline["model"])
    expected_model["stage4_mamba_spatial_reduction"] = 2
    assert candidate["model"] == expected_model
    assert candidate["save"]["save_dir"] == (
        "./runs/pilot/lightweight_c40_s4mamba_lsscf_s4lrrm/"
        "C3_C40_S4MAMBA_LSSCF_S4LRRM2_LR150_PHOTO_NATIVE384x544_EMA_BPCM"
    )


def test_lrrm_model_full_resolution_non_mamba_branches_and_contracts() -> None:
    torch.manual_seed(42)
    reduction_one_model = _build_from_config(LSSCF_CONFIG)
    reduction_two_model = _build_from_config(LRRM_CONFIG)
    assert set(reduction_one_model.state_dict()) == set(reduction_two_model.state_dict())
    strict_result = reduction_two_model.load_state_dict(reduction_one_model.state_dict(), strict=True)
    assert not strict_result.missing_keys and not strict_result.unexpected_keys

    model = reduction_two_model.train()
    assert isinstance(model.stage3, LocalCrackBlock)
    assert isinstance(model.stage4, TopoRouteBlock)
    assert model.stage4.route_mamba.spatial_reduction == 2
    assert model.stage4.route_mamba.active_mamba_directions == ALL_DIRECTIONS
    assert model.stage4.route_mamba.local_compensations is None
    assert isinstance(model.stage4.global_branch, TransformerGlobalBlock)

    local_sizes: list[tuple[int, int]] = []
    transformer_sizes: list[tuple[int, int]] = []
    mamba_sizes: list[tuple[int, int]] = []
    handles = [
        model.stage4.local.register_forward_pre_hook(
            lambda _module, inputs: local_sizes.append(tuple(inputs[0].shape[-2:]))
        ),
        model.stage4.global_branch.register_forward_pre_hook(
            lambda _module, inputs: transformer_sizes.append(tuple(inputs[0].shape[-2:]))
        ),
    ]
    for path in model.stage4.route_mamba.paths:
        handles.append(
            path.register_forward_pre_hook(
                lambda _module, inputs: mamba_sizes.append(tuple(inputs[0].shape[-2:]))
            )
        )
    try:
        image = torch.randn(1, 3, 64, 80, device=RUNTIME_DEVICE)
        outputs = model(image)
    finally:
        for handle in handles:
            handle.remove()
    stage4_size = tuple(outputs["route_s4"].shape[-2:])
    assert local_sizes == [stage4_size]
    assert transformer_sizes == [stage4_size]
    assert mamba_sizes == [(stage4_size[0] // 2, stage4_size[1] // 2)] * 4
    assert {"mask", "edge", "reliability", "route_s4"}.issubset(outputs)
    assert "route_s3" not in outputs
    assert outputs["mask"].shape == (1, 1, 64, 80)
    loss = outputs["mask"].mean() + outputs["edge"].mean() + outputs["reliability"].mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)

    model.eval()
    with torch.inference_mode():
        mask_only = model.forward_mask_only(image)
    assert set(mask_only) == {"mask"}
    assert mask_only["mask"].shape == (1, 1, 64, 80)


def test_existing_configurations_default_to_full_resolution_routed_mamba() -> None:
    for config_path in (C40_CONFIG, S4MAMBA_CONFIG, LSAF_CONFIG, LSSCF_CONFIG):
        model = _build_from_config(config_path)
        assert isinstance(model.stage4, TopoRouteBlock)
        assert model.stage4.route_mamba.spatial_reduction == 1


def test() -> None:
    tests = (
        test_reduction_one_is_bitwise_compatible_and_state_dict_compatible,
        test_reduction_two_preserves_routes_and_executes_all_paths_at_low_resolution,
        test_reduction_two_handles_odd_shapes_and_rejects_too_small_inputs,
        test_lrrm_config_is_single_variable_change,
        test_lrrm_model_full_resolution_non_mamba_branches_and_contracts,
        test_existing_configurations_default_to_full_resolution_routed_mamba,
    )
    for check in tests:
        check()
        print(f"PASS: {check.__name__}")


if __name__ == "__main__":
    test()
