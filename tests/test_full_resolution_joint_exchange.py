"""Regression checks for the independent DPER-FJSE candidate.

Run with either ``python tests/test_full_resolution_joint_exchange.py`` or
``python -m pytest tests/test_full_resolution_joint_exchange.py -q``.
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
from models.blocks import (
    DetailPreservingHalfStem,
    FullResolutionJointSpatialExchange,
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
FJSE_CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "lightweight_c40_s4mamba_lsscf_dper_fjse"
    / "C3_C40_S4MAMBA_LSSCF_DPER_FJSE_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
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
    # CPU regression runs retain the four-path topology through the existing
    # fallback. Formal profiling/training remains guarded by train.py.
    if RUNTIME_DEVICE.type != "cuda":
        model_cfg["use_mamba_ssm"] = False
    return build_model(model_name, **model_cfg).to(RUNTIME_DEVICE)


def _assert_equal_state(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> None:
    assert set(left) == set(right)
    for key, value in left.items():
        assert torch.equal(value, right[key]), key


def test_fjse_config_is_a_strict_dper_single_variable_change() -> None:
    dper = _load_config(DPER_CONFIG)
    fjse = _load_config(FJSE_CONFIG)
    assert {key: value for key, value in fjse.items() if key not in {"model", "save"}} == {
        key: value for key, value in dper.items() if key not in {"model", "save"}
    }
    expected_model = copy.deepcopy(dper["model"])
    expected_model["use_full_resolution_joint_exchange"] = True
    assert fjse["model"] == expected_model
    assert fjse["save"]["save_dir"] == (
        "./runs/pilot/lightweight_c40_s4mamba_lsscf_dper_fjse/"
        "C3_C40_S4MAMBA_LSSCF_DPER_FJSE_LR150_PHOTO_NATIVE384x544_EMA_BPCM"
    )


def test_fjse_guards_and_false_flag_are_bitwise_dper_compatible() -> None:
    invalid = (
        ({"use_detail_preserving_early_rebase": False, "use_full_resolution_joint_exchange": True}, "requires"),
        ({"use_narrow_dual_resolution_stage1": True, "use_full_resolution_joint_exchange": True}, "cannot"),
        ({"use_subpixel_phase_consistency_refiner": True, "use_full_resolution_joint_exchange": True}, "cannot"),
        ({"use_haar_topology_preserving_rebase": True, "use_full_resolution_joint_exchange": True}, "cannot"),
    )
    for overrides, text in invalid:
        try:
            _build(DPER_CONFIG, **overrides)
        except ValueError as error:
            assert text in str(error)
        else:
            raise AssertionError("FJSE dependency validation did not reject an invalid configuration.")

    torch.manual_seed(501)
    dper = _build(DPER_CONFIG).eval()
    torch.manual_seed(501)
    disabled = _build(DPER_CONFIG, use_full_resolution_joint_exchange=False).eval()
    _assert_equal_state(dper.state_dict(), disabled.state_dict())
    image = torch.randn(1, 3, 64, 80, device=RUNTIME_DEVICE)
    with torch.inference_mode():
        expected = dper(image)
        actual = disabled(image)
    assert set(expected) == set(actual)
    for key in expected:
        assert torch.equal(expected[key], actual[key]), key


def test_fjse_structure_dataflow_and_shared_initialization() -> None:
    torch.manual_seed(502)
    dper = _build(DPER_CONFIG).eval()
    torch.manual_seed(502)
    model = _build(FJSE_CONFIG).eval()
    assert model.use_detail_preserving_early_rebase
    assert model.use_full_resolution_joint_exchange
    assert not model.use_narrow_dual_resolution_stage1
    assert not model.use_subpixel_phase_consistency_refiner
    assert not model.use_haar_topology_preserving_rebase
    assert isinstance(model.stem, DetailPreservingHalfStem)
    assert isinstance(model.stage1, HalfResolutionStage1Block)
    assert isinstance(model.full_resolution_joint_exchange, FullResolutionJointSpatialExchange)
    assert model.narrow_dual_resolution_stage1 is None
    assert model.f1_reconstruction is not None and model.f1_reconstruction.phase_refiner is None
    assert not any(isinstance(module, (NarrowDualResolutionStage1, SubpixelPhaseConsistencyRefiner)) for module in model.modules())

    fjse = model.full_resolution_joint_exchange
    assert fjse.detail_channels == 16 and fjse.semantic_channels == 40
    assert fjse.local_dense.in_channels == fjse.local_dense.out_channels == 16
    assert fjse.local_dense.kernel_size == (3, 3)
    assert fjse.local_dense.stride == (1, 1)
    assert fjse.local_dense.padding == (1, 1)
    assert fjse.local_dense.groups == 1
    assert fjse.local_dense.bias is None
    assert fjse.exchange_projection.in_channels == 64 and fjse.exchange_projection.out_channels == 40
    assert fjse.exchange_projection.bias is None

    dper_state = dper.state_dict()
    fjse_state = model.state_dict()
    assert set(fjse_state) - set(dper_state) == {
        key for key in fjse_state if key.startswith("full_resolution_joint_exchange.")
    }
    for key, value in dper_state.items():
        assert key in fjse_state, key
        assert torch.equal(value, fjse_state[key]), key

    assert isinstance(model.decoder.up3, LowResolutionSubpixelSeparableConcatUpBlock)
    assert isinstance(model.decoder.up2, LowResolutionSubpixelSeparableConcatUpBlock)
    assert isinstance(model.decoder.up1, LowResolutionSubpixelSeparableConcatUpBlock)
    assert isinstance(model.stage4, TopoRouteBlock)
    assert isinstance(model.stage4.global_branch, TransformerGlobalBlock)
    assert model.stage4.route_mamba.active_mamba_directions == ALL_DIRECTIONS
    assert model.stage4.route_mamba.active_mamba_path_count == 4
    stage4_backends = model.mamba_backend_info()["stage4"]
    assert len(stage4_backends) == 4
    assert len(set(stage4_backends)) == 1
    assert stage4_backends[0] in {"mamba_ssm", "fallback"}

    observed: dict[str, torch.Tensor] = {}
    calls = {"fjse": 0, "stage1": 0, "reconstruction": 0}

    def fjse_hook(
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        calls["fjse"] += 1
        observed["full_detail"] = inputs[0].detach()
        observed["semantic_half"] = inputs[1].detach()
        observed["local_detail"] = output[0].detach()
        observed["enhanced_half"] = output[1].detach()

    def capture(name: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            observed[name] = output.detach()

        return hook

    def stage1_hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        calls["stage1"] += 1
        observed["stage1_input"] = inputs[0].detach()

    def reconstruction_hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        calls["reconstruction"] += 1
        observed["reconstruction_detail"] = inputs[0].detach()
        observed["reconstruction_half"] = inputs[1].detach()

    handles = [
        fjse.register_forward_hook(fjse_hook),
        fjse.local_activation.register_forward_hook(capture("local_context")),
        fjse.phase_unshuffle.register_forward_hook(capture("phase_context")),
        fjse.exchange_norm.register_forward_hook(capture("semantic_delta")),
        model.stage1.register_forward_pre_hook(stage1_hook),
        model.stage1.register_forward_hook(capture("stage1_output")),
        model.f1_reconstruction.register_forward_pre_hook(reconstruction_hook),
    ]
    image = torch.randn(1, 3, 64, 80, device=RUNTIME_DEVICE)
    try:
        with torch.inference_mode():
            features = model.extract_encoder_features(image)
    finally:
        for handle in handles:
            handle.remove()

    assert calls == {"fjse": 1, "stage1": 1, "reconstruction": 1}
    assert observed["full_detail"].shape == observed["local_context"].shape == observed["local_detail"].shape == (1, 16, 64, 80)
    assert observed["phase_context"].shape == (1, 64, 32, 40)
    assert torch.equal(nn.PixelShuffle(2)(observed["phase_context"]), observed["local_context"])
    assert observed["semantic_half"].shape == observed["semantic_delta"].shape == observed["enhanced_half"].shape == (1, 40, 32, 40)
    assert torch.equal(observed["stage1_input"], observed["enhanced_half"])
    assert not torch.equal(observed["semantic_half"], observed["enhanced_half"])
    assert torch.equal(observed["reconstruction_detail"], observed["local_detail"])
    assert torch.equal(observed["reconstruction_half"], observed["stage1_output"])
    assert features["f1"].shape == (1, 40, 64, 80)
    assert features["f2"].shape == (1, 80, 32, 40)
    assert features["f3"].shape == (1, 160, 16, 20)
    assert features["f4"].shape == (1, 320, 8, 10)


def test_fjse_encoder_entrypoints_mask_identity_native_forward_and_backward() -> None:
    torch.manual_seed(503)
    model = _build(FJSE_CONFIG).eval()
    assert isinstance(model.full_resolution_joint_exchange, FullResolutionJointSpatialExchange)
    fjse = model.full_resolution_joint_exchange
    calls = 0

    def count_call(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...]) -> None:
        nonlocal calls
        calls += 1

    handle = fjse.register_forward_pre_hook(count_call)
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
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    for name, parameter in fjse.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert torch.count_nonzero(fjse.local_dense.weight.grad).item() > 0
    assert torch.count_nonzero(fjse.exchange_projection.weight.grad).item() > 0


def test() -> None:
    tests = (
        test_fjse_config_is_a_strict_dper_single_variable_change,
        test_fjse_guards_and_false_flag_are_bitwise_dper_compatible,
        test_fjse_structure_dataflow_and_shared_initialization,
        test_fjse_encoder_entrypoints_mask_identity_native_forward_and_backward,
    )
    for check in tests:
        check()
        print(f"PASS: {check.__name__}")


if __name__ == "__main__":
    test()
