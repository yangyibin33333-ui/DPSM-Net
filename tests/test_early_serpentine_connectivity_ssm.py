"""Regression checks for the strict DPER-ESCS single-variable candidate.

Run with either ``python tests/test_early_serpentine_connectivity_ssm.py`` or
``python -m pytest tests/test_early_serpentine_connectivity_ssm.py -q``.
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

from models import build_model
from models.blocks import DetailPreservingHalfStem, HalfResolutionStage1Block, TopoRouteBlock, TransformerGlobalBlock
from models.decoder import LowResolutionSubpixelSeparableConcatUpBlock
from models.mamba_blocks import EarlySerpentineConnectivitySSM, SequenceMambaCore
from train import make_optimizer


DPER_CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "lightweight_c40_s4mamba_lsscf_dper"
    / "C3_C40_S4MAMBA_LSSCF_DPER_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
ESCS_CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "lightweight_c40_s4mamba_lsscf_dper_escs"
    / "C3_C40_S4MAMBA_LSSCF_DPER_ESCS_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
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
    # CPU runs validate tensor/index mechanics with the established fallback.
    # Formal profiling and training use --require_mamba_ssm/train.py guards.
    if RUNTIME_DEVICE.type != "cuda":
        model_cfg["use_mamba_ssm"] = False
    return build_model(model_name, **model_cfg).to(RUNTIME_DEVICE)


def _assert_equal_state(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> None:
    assert set(left) == set(right)
    for key, value in left.items():
        assert torch.equal(value, right[key]), key


def _assert_path_is_complete_and_continuous(order: torch.Tensor, height: int, width: int) -> None:
    length = height * width
    assert order.shape == (length,)
    assert int(order.min()) == 0 and int(order.max()) == length - 1
    assert torch.equal(torch.sort(order).values, torch.arange(length, device=order.device))
    rows = torch.div(order, width, rounding_mode="floor")
    columns = order.remainder(width)
    row_step = (rows[1:] - rows[:-1]).abs()
    column_step = (columns[1:] - columns[:-1]).abs()
    assert bool(((row_step <= 1) & (column_step <= 1)).all())
    assert bool(((row_step + column_step) > 0).all())


def test_escs_config_is_a_strict_dper_single_variable_change() -> None:
    dper = _load_config(DPER_CONFIG)
    escs = _load_config(ESCS_CONFIG)
    assert {key: value for key, value in escs.items() if key not in {"model", "save"}} == {
        key: value for key, value in dper.items() if key not in {"model", "save"}
    }
    expected_model = copy.deepcopy(dper["model"])
    expected_model["use_early_serpentine_connectivity_ssm"] = True
    assert escs["model"] == expected_model
    assert escs["save"]["save_dir"] == (
        "./runs/pilot/lightweight_c40_s4mamba_lsscf_dper_escs/"
        "C3_C40_S4MAMBA_LSSCF_DPER_ESCS_LR150_PHOTO_NATIVE384x544_EMA_BPCM"
    )


def test_escs_false_flag_guards_and_dper_bitwise_compatibility() -> None:
    invalid = (
        ({"use_detail_preserving_early_rebase": False, "use_early_serpentine_connectivity_ssm": True}, "requires"),
        ({"use_subpixel_phase_consistency_refiner": True, "use_early_serpentine_connectivity_ssm": True}, "cannot"),
        ({"use_haar_topology_preserving_rebase": True, "use_early_serpentine_connectivity_ssm": True}, "cannot"),
        ({"use_narrow_dual_resolution_stage1": True, "use_early_serpentine_connectivity_ssm": True}, "cannot"),
        ({"use_full_resolution_joint_exchange": True, "use_early_serpentine_connectivity_ssm": True}, "cannot"),
    )
    for overrides, text in invalid:
        try:
            _build(DPER_CONFIG, **overrides)
        except ValueError as error:
            assert text in str(error)
        else:
            raise AssertionError("ESCS dependency validation did not reject an invalid configuration.")

    torch.manual_seed(601)
    dper = _build(DPER_CONFIG).eval()
    torch.manual_seed(601)
    disabled = _build(DPER_CONFIG, use_early_serpentine_connectivity_ssm=False).eval()
    _assert_equal_state(dper.state_dict(), disabled.state_dict())
    assert disabled.early_serpentine_connectivity_ssm is None
    image = torch.randn(1, 3, 64, 80, device=RUNTIME_DEVICE)
    with torch.inference_mode():
        expected = dper(image)
        actual = disabled(image)
    assert set(expected) == set(actual)
    for key in expected:
        assert torch.equal(expected[key], actual[key]), key


def test_escs_scan_indices_are_unique_invertible_continuous_and_cached() -> None:
    scanner = EarlySerpentineConnectivitySSM(40, 16, use_mamba_ssm=False).to(RUNTIME_DEVICE)
    assert scanner._scan_index_cache == {}
    for height, width in ((32, 40), (192, 272), (256, 256)):
        cache = scanner.scan_indices(height, width, RUNTIME_DEVICE)
        order, inverse = cache["order"], cache["inverse"]
        assert order.shape == inverse.shape == (4, height * width)
        horizontal, horizontal_reverse, diagonal, diagonal_reverse = order
        assert torch.equal(horizontal_reverse, horizontal.flip(0))
        assert torch.equal(diagonal_reverse, diagonal.flip(0))
        _assert_path_is_complete_and_continuous(horizontal, height, width)
        _assert_path_is_complete_and_continuous(diagonal, height, width)
        identity = torch.arange(height * width, device=RUNTIME_DEVICE)
        for path, inverse_path in zip(order, inverse, strict=True):
            assert torch.equal(path.index_select(0, inverse_path), identity)
            assert torch.equal(inverse_path.index_select(0, path), identity)
        # Raster has a row-end jump; horizontal snake must not.
        raster = torch.arange(height * width, device=RUNTIME_DEVICE)
        raster_rows = torch.div(raster, width, rounding_mode="floor")
        raster_cols = raster.remainder(width)
        raster_jump = ((raster_rows[1:] - raster_rows[:-1]).abs() > 1) | ((raster_cols[1:] - raster_cols[:-1]).abs() > 1)
        assert bool(raster_jump.any())

    before_hits = scanner.scan_cache_hits
    first = scanner.scan_indices(192, 272, RUNTIME_DEVICE)
    second = scanner.scan_indices(192, 272, RUNTIME_DEVICE)
    assert first is second
    assert scanner.scan_cache_hits == before_hits + 2
    assert len(scanner._scan_index_cache) == 3
    assert not any(key.startswith("_scan_index_cache") for key in scanner.state_dict())


def test_escs_model_topology_single_core_dataflow_and_shared_initialization() -> None:
    torch.manual_seed(602)
    dper = _build(DPER_CONFIG).eval()
    torch.manual_seed(602)
    model = _build(ESCS_CONFIG).eval()
    assert model.use_detail_preserving_early_rebase
    assert model.use_early_serpentine_connectivity_ssm
    assert isinstance(model.stem, DetailPreservingHalfStem)
    assert isinstance(model.stage1, HalfResolutionStage1Block)
    assert isinstance(model.early_serpentine_connectivity_ssm, EarlySerpentineConnectivitySSM)
    assert isinstance(model.stage4, TopoRouteBlock)
    assert isinstance(model.stage4.global_branch, TransformerGlobalBlock)
    assert model.stage4.route_mamba.active_mamba_directions == ALL_DIRECTIONS
    assert model.stage4.route_mamba.active_mamba_path_count == 4
    assert isinstance(model.decoder.up3, LowResolutionSubpixelSeparableConcatUpBlock)
    assert isinstance(model.decoder.up2, LowResolutionSubpixelSeparableConcatUpBlock)
    assert isinstance(model.decoder.up1, LowResolutionSubpixelSeparableConcatUpBlock)

    escs = model.early_serpentine_connectivity_ssm
    assert escs.input_channels == 40 and escs.bottleneck_channels == 16
    assert escs.d_state == 8 and escs.d_conv == 3 and escs.expand == 1
    assert escs.shared_mamba.force_fp32_under_autocast
    assert escs.path_count == 4 and escs.instantiated_core_count == 1 and escs.traversal_count == 4
    assert sum(isinstance(module, SequenceMambaCore) for module in escs.modules()) == 1
    assert not any(isinstance(module, nn.ModuleList) for module in escs.modules())
    assert not any("route" in name for name, _ in escs.named_modules())

    dper_state, escs_state = dper.state_dict(), model.state_dict()
    assert set(escs_state) - set(dper_state) == {
        key for key in escs_state if key.startswith("early_serpentine_connectivity_ssm.")
    }
    for key, value in dper_state.items():
        assert torch.equal(value, escs_state[key]), key

    backend_info = model.mamba_backend_info()
    assert set(backend_info) == {"early_connectivity", "stage4"}
    assert len(backend_info["early_connectivity"]) == 1
    assert len(backend_info["stage4"]) == 4
    assert len(set(backend_info["stage4"])) == 1
    assert backend_info["early_connectivity"][0] in {"mamba_ssm", "fallback"}
    assert backend_info["stage4"][0] in {"mamba_ssm", "fallback"}

    observed: dict[str, torch.Tensor] = {}
    calls = {"stage1": 0, "escs": 0, "shared": 0, "reconstruction": 0, "down1": 0}

    def capture(name: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            observed[name] = output.detach()

        return hook

    def pre_capture(name: str, counter: str):
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            calls[counter] += 1
            observed[name] = inputs[0].detach()

        return hook

    def reconstruction_pre(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        calls["reconstruction"] += 1
        observed["reconstruction_detail"] = inputs[0].detach()
        observed["reconstruction_half"] = inputs[1].detach()

    handles = [
        model.stage1.register_forward_hook(capture("stage1_half")),
        escs.register_forward_pre_hook(pre_capture("escs_input", "escs")),
        escs.register_forward_hook(capture("enhanced_stage1_half")),
        escs.input_projection.register_forward_hook(capture("bottleneck")),
        escs.scan_reorder.register_forward_hook(capture("path_sequences")),
        escs.shared_mamba.register_forward_pre_hook(pre_capture("batched_sequences", "shared")),
        escs.inverse_reorder.register_forward_hook(capture("restored_paths")),
        escs.path_fusion.register_forward_hook(capture("fused_sequences")),
        escs.output_projection.register_forward_hook(capture("connectivity_delta")),
        model.f1_reconstruction.register_forward_pre_hook(reconstruction_pre),
        model.down1.register_forward_pre_hook(pre_capture("down1_input", "down1")),
    ]
    image = torch.randn(1, 3, 64, 80, device=RUNTIME_DEVICE)
    try:
        with torch.inference_mode():
            features = model.extract_encoder_features(image)
    finally:
        for handle in handles:
            handle.remove()

    assert calls == {"stage1": 0, "escs": 1, "shared": 1, "reconstruction": 1, "down1": 1}
    assert torch.equal(observed["escs_input"], observed["stage1_half"])
    assert observed["stage1_half"].shape == observed["enhanced_stage1_half"].shape == (1, 40, 32, 40)
    assert observed["bottleneck"].shape == (1, 16, 32, 40)
    assert observed["path_sequences"].shape == (1, 4, 1280, 16)
    assert observed["batched_sequences"].shape == (4, 1280, 16)
    assert observed["restored_paths"].shape == (1, 4, 1280, 16)
    assert observed["fused_sequences"].shape == (1, 1280, 16)
    assert observed["connectivity_delta"].shape == (1, 40, 32, 40)
    assert torch.equal(observed["reconstruction_half"], observed["enhanced_stage1_half"])
    assert torch.equal(observed["down1_input"], observed["enhanced_stage1_half"])
    assert observed["reconstruction_detail"].shape == (1, 16, 64, 80)
    assert escs.last_shared_mamba_call_count == 1
    assert escs.last_path_sequence_shape == (1, 1280, 16)
    assert escs.last_batched_sequence_shape == (4, 1280, 16)
    assert features["f1"].shape == (1, 40, 64, 80)
    assert features["f2"].shape == (1, 80, 32, 40)
    assert features["f3"].shape == (1, 160, 16, 20)
    assert features["f4"].shape == (1, 320, 8, 10)


def test_escs_encoder_entrypoints_mask_identity_native_forward_and_backward() -> None:
    torch.manual_seed(603)
    model = _build(ESCS_CONFIG).eval()
    assert isinstance(model.early_serpentine_connectivity_ssm, EarlySerpentineConnectivitySSM)
    escs = model.early_serpentine_connectivity_ssm
    calls = 0

    def shared_counter(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...]) -> None:
        nonlocal calls
        calls += 1

    handle = escs.shared_mamba.register_forward_pre_hook(shared_counter)
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
    for name, parameter in escs.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert torch.count_nonzero(escs.input_projection[0].weight.grad).item() > 0
    assert torch.count_nonzero(escs.shared_mamba.core.in_proj.weight.grad).item() > 0
    assert torch.count_nonzero(escs.output_projection[0].weight.grad).item() > 0


def test_escs_amp_adamw_compatibility() -> None:
    """The ESCS AMP path must complete a real AdamW update in this CUDA runtime."""

    if RUNTIME_DEVICE.type != "cuda":
        return
    torch.manual_seed(604)
    model = _build(ESCS_CONFIG).train()
    optimizer = make_optimizer(model, _load_config(ESCS_CONFIG)["train"], RUNTIME_DEVICE, logging.getLogger(__name__))
    assert all(not group["fused"] for group in optimizer.param_groups)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    image = torch.randn(1, 3, 64, 80, device=RUNTIME_DEVICE)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model(image)
        loss = outputs["mask"].mean() + outputs["edge"].mean() + outputs["reliability"].mean()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test() -> None:
    tests = (
        test_escs_config_is_a_strict_dper_single_variable_change,
        test_escs_false_flag_guards_and_dper_bitwise_compatibility,
        test_escs_scan_indices_are_unique_invertible_continuous_and_cached,
        test_escs_model_topology_single_core_dataflow_and_shared_initialization,
        test_escs_encoder_entrypoints_mask_identity_native_forward_and_backward,
        test_escs_amp_adamw_compatibility,
    )
    for check in tests:
        check()
        print(f"PASS: {check.__name__}")


if __name__ == "__main__":
    test()
