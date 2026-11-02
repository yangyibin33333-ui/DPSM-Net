"""Strict registry and default-compatibility audit for final-model OFAT scans.

Run from repository root with:
    python tests/test_paper_parameter_sensitivity_configs.py

The YAML/protocol checks are read-only.  CUDA model checks require the real
``mamba_ssm`` backend; the test skips them when CUDA is unavailable instead of
silently treating the repository CPU fallback as formal backend evidence.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import utils.auto_protocol as auto_protocol_module

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    TORCH_AVAILABLE = False
else:
    TORCH_AVAILABLE = True
    from losses import DEMambaLoss
    from models import build_model
    from models.blocks import RIFTStructuralDirectionalModelingBlock
    from utils import model_checksum


REGISTRY = ROOT / "configs" / "experiments" / "paper_parameter_sensitivity" / "crack500"
M5_PATH = ROOT / "configs" / "experiments" / "paper_dependency_aware_ablation" / "crack500" / "M5_FrameworkOn_DPSMOn_StructSupOn.yaml"
OLD_A5_PATH = ROOT / "configs" / "experiments" / "paper_main_table_a" / "crack500" / "A5_C40_S4_LSSCF_DPER_RSDM_DPSM80.yaml"
ROWS = {
    "PS_DW08": ("model", "early_detail_channels", 8),
    "PS_DW12": ("model", "early_detail_channels", 12),
    "PS_DW20": ("model", "early_detail_channels", 20),
    "PS_DW24": ("model", "early_detail_channels", 24),
    "PS_K09": ("model", "rsdm_kernel_size", 9),
    "PS_K17": ("model", "rsdm_kernel_size", 17),
    "PS_R01": ("model", "rsdm_dilation", 1),
    "PS_R05": ("model", "rsdm_dilation", 5),
    "PS_LT010": ("loss", "lambda_tangent", 0.1),
    "PS_LT030": ("loss", "lambda_tangent", 0.3),
    "PS_LB0125": ("loss", "lambda_bpcm", 0.0125),
    "PS_LB0500": ("loss", "lambda_bpcm", 0.05),
}
CUDA_DEVICE = torch.device("cuda") if TORCH_AVAILABLE and torch.cuda.is_available() else None


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    assert isinstance(config, dict), path
    return config


def _flatten(value: object, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], object]:
    if isinstance(value, dict):
        flattened: dict[tuple[str, ...], object] = {}
        for key, child in value.items():
            flattened.update(_flatten(child, prefix + (str(key),)))
        return flattened
    return {prefix: value}


def _diff(left: dict, right: dict) -> set[tuple[str, ...]]:
    left_flat, right_flat = _flatten(left), _flatten(right)
    return {key for key in left_flat.keys() | right_flat.keys() if left_flat.get(key) != right_flat.get(key)}


def _build(config: dict) -> "torch.nn.Module":
    assert TORCH_AVAILABLE and CUDA_DEVICE is not None
    model_config = copy.deepcopy(config["model"])
    name = model_config.pop("name")
    assert model_config["use_mamba_ssm"] is True
    model = build_model(name, **model_config).to(CUDA_DEVICE)
    assert model.mamba_backend_info()["stage4"] == ["mamba_ssm"], "real mamba_ssm ×1 is required"
    return model


def _construct_before_backend(config: dict) -> "torch.nn.Module":
    """Exercise constructor guards only; do not claim a CPU backend is formal."""

    assert TORCH_AVAILABLE
    model_config = copy.deepcopy(config["model"])
    name = model_config.pop("name")
    return build_model(name, **model_config)


def _resolved_protocol(config: dict) -> object:
    """Resolve the frozen Crack500 facts without scanning real training data."""

    candidate = copy.deepcopy(config)
    original_sizes = auto_protocol_module.scan_train_image_sizes
    original_audit = auto_protocol_module.audit_train_foreground
    auto_protocol_module.scan_train_image_sizes = lambda _root: {(352, 640): 2357}
    auto_protocol_module.audit_train_foreground = lambda *_args, **_kwargs: {
        "dataset_fingerprint": "test-only-crack500-fingerprint",
        "negative_positive_ratio": 1.0,
    }
    try:
        return auto_protocol_module.resolve_auto_protocol(candidate)
    finally:
        auto_protocol_module.scan_train_image_sizes = original_sizes
        auto_protocol_module.audit_train_foreground = original_audit


def _assert_final_model(config: dict) -> None:
    model = config["model"]
    assert model["base_channels"] == 40 and model["route_bins"] == 4
    assert model["mamba_stages"] == [4] and model["transformer_stages"] == [4]
    assert model["use_mamba_ssm"] is True and model["use_transformer"] is True
    assert model["use_low_resolution_subpixel_separable_concat_fusion"] is True
    assert model["use_detail_preserving_early_rebase"] is True
    assert model["use_rsdm_stage1"] is True
    assert model["use_direction_partitioned_shared_mamba"] is True
    assert model["stage4_direction_group_channels"] == 80
    assert model.get("dpsm_scan_mode", "four_direction") == "four_direction"
    assert model.get("dpsm_route_mode", "learned") == "learned"
    assert model["use_topology_matched_no_dpsm_stage4"] is False


def test_registry_diff_and_frozen_protocol() -> None:
    # The explicit mapping is the active preregistered registry.  Do not glob
    # every PS_*.yaml: repositories can retain obsolete or parallel historical
    # parameter-scan files, which must not silently become formal rows or make
    # this active audit fail merely by being preserved.
    registered_paths = {identifier: REGISTRY / f"{identifier}.yaml" for identifier in ROWS}
    missing = [identifier for identifier, path in registered_paths.items() if not path.is_file()]
    assert not missing, f"Missing active parameter-sensitivity YAMLs: {missing}"
    assert len(registered_paths) == 12
    m5 = _load(M5_PATH)
    for identifier, (section, field, expected) in ROWS.items():
        config = _load(REGISTRY / f"{identifier}.yaml")
        assert config[section][field] == expected
        assert config["save"]["save_dir"] == f"runs/paper/parameter_sensitivity/crack500/{identifier}"
        assert _diff(m5, config) == {(section, field), ("save", "save_dir")}, identifier
        _assert_final_model(config)
        assert config["data"]["data_root"] == "./Crack500" and config["data"]["img_size"] is None
        assert config["auto_protocol"]["version"] == 2 and config["auto_protocol"]["enabled"] is True
        plan = _resolved_protocol(config)
        assert plan is not None
        assert plan.selected_img_size == (352, 640)
        assert plan.effective_batch_size == 8 and plan.resolved_epochs == 64 and plan.target_updates == 18_816
        assert plan.scnp_warmup_updates == 941 and plan.bpcm_start_update == 5_645 and plan.bpcm_ramp_updates == 3_763
        assert plan.validation_interval_updates == 400
        assert plan.min_early_stop_update == 10_349 and plan.patience_updates == 2_823


def test_center_reuse_and_loss_protocol() -> None:
    m5 = _load(M5_PATH)
    assert m5["model"].get("rsdm_kernel_size", 13) == 13
    assert m5["model"].get("rsdm_dilation", 3) == 3
    assert m5["model"]["early_detail_channels"] == 16
    assert m5["loss"]["lambda_tangent"] == 0.2 and m5["loss"]["lambda_bpcm"] == 0.025
    d1 = _load(ROOT / "configs" / "experiments" / "paper_dependency_aware_ablation" / "crack500" / "loss_2x2" / "D1_TangentOn_BPCMOff.yaml")
    d2 = _load(ROOT / "configs" / "experiments" / "paper_dependency_aware_ablation" / "crack500" / "loss_2x2" / "D2_TangentOff_BPCMOn.yaml")
    assert d1["loss"]["lambda_bpcm"] == 0.0 and d2["loss"]["lambda_tangent"] == 0.0


def _batch() -> dict[str, "torch.Tensor"]:
    assert TORCH_AVAILABLE and CUDA_DEVICE is not None
    height, width = 64, 96
    route = torch.randint(0, 4, (1, 1, height, width), device=CUDA_DEVICE)
    tangent = torch.zeros(1, 4, height, width, device=CUDA_DEVICE)
    tangent.scatter_(1, route, 1.0)
    return {
        "image": torch.randn(1, 3, height, width, device=CUDA_DEVICE),
        "mask": (torch.rand(1, 1, height, width, device=CUDA_DEVICE) < 0.12).float(),
        "edge_gt": (torch.rand(1, 1, height, width, device=CUDA_DEVICE) < 0.06).float(),
        "tangent_gt": tangent,
        "tangent_valid": torch.ones(1, 1, height, width, device=CUDA_DEVICE),
    }


def _criterion(config: dict) -> "DEMambaLoss":
    assert TORCH_AVAILABLE and CUDA_DEVICE is not None
    loss_config = copy.deepcopy(config["loss"])
    loss_config["mask_pos_weight"] = 1.0
    return DEMambaLoss(**loss_config).to(CUDA_DEVICE)


def _assert_cuda_available() -> bool:
    if CUDA_DEVICE is None:
        print("SKIP: CUDA+mamba_ssm model checks unavailable; no CPU fallback is accepted as formal evidence.")
        return False
    return True


def test_default_m5_explicit_13_3_state_checkpoint_and_output_regression() -> None:
    if not _assert_cuda_available():
        return
    m5 = _load(M5_PATH)
    legacy = _load(OLD_A5_PATH)
    explicit = copy.deepcopy(m5)
    explicit["model"].update(rsdm_kernel_size=13, rsdm_dilation=3)
    torch.manual_seed(911)
    implicit_model = _build(m5).eval()
    torch.manual_seed(911)
    explicit_model = _build(explicit).eval()
    torch.manual_seed(911)
    legacy_model = _build(legacy).eval()
    implicit_state, explicit_state = implicit_model.state_dict(), explicit_model.state_dict()
    assert implicit_state.keys() == explicit_state.keys()
    assert implicit_state.keys() == legacy_model.state_dict().keys()
    assert sum(parameter.numel() for parameter in implicit_model.parameters()) == 3_193_751
    assert sum(parameter.numel() for parameter in explicit_model.parameters()) == 3_193_751
    for key, value in implicit_state.items():
        assert value.shape == explicit_state[key].shape and torch.equal(value, explicit_state[key]), key
        assert value.shape == legacy_model.state_dict()[key].shape and torch.equal(value, legacy_model.state_dict()[key]), key
    assert model_checksum(implicit_model) == model_checksum(explicit_model)
    explicit_model.load_state_dict(copy.deepcopy(implicit_state), strict=True)
    legacy_model.load_state_dict(copy.deepcopy(implicit_state), strict=True)
    image = torch.randn(1, 3, 64, 96, device=CUDA_DEVICE)
    with torch.inference_mode():
        implicit_output, explicit_output, legacy_output = implicit_model(image), explicit_model(image), legacy_model(image)
    assert implicit_output.keys() == explicit_output.keys()
    for key, value in implicit_output.items():
        assert torch.equal(value, explicit_output[key]), key
        assert torch.equal(value, legacy_output[key]), key
    assert implicit_output["mask"].shape == implicit_output["edge"].shape == (1, 1, 64, 96)
    assert implicit_output["route_s4"].shape == (1, 4, 8, 12)
    print(f"cuda M5 default regression | params=3193751 | checksum={model_checksum(implicit_model)}")


def test_nondefault_rsdm_shapes_loss_backward_and_shared_initialization() -> None:
    if not _assert_cuda_available():
        return
    m5 = _load(M5_PATH)
    batch = _batch()
    torch.manual_seed(912)
    baseline = _build(m5).train()
    baseline_state = baseline.state_dict()
    for identifier in ("PS_K09", "PS_K17", "PS_R01", "PS_R05"):
        config = _load(REGISTRY / f"{identifier}.yaml")
        torch.manual_seed(912)
        model = _build(config).train()
        rsdm = model.rsdm_stage1
        assert isinstance(rsdm, RIFTStructuralDirectionalModelingBlock)
        assert rsdm.kernel_size == config["model"].get("rsdm_kernel_size", 13)
        assert rsdm.dilation == config["model"].get("rsdm_dilation", 3)
        for key, value in model.state_dict().items():
            if not key.startswith("rsdm_stage1."):
                assert torch.equal(value, baseline_state[key]), f"{identifier}: shared initialization changed: {key}"
        outputs = model(batch["image"])
        assert outputs["mask"].shape == outputs["edge"].shape == (1, 1, 64, 96)
        assert outputs["route_s4"].shape == (1, 4, 8, 12)
        loss, _ = _criterion(config)(outputs, batch, epoch=0, global_update=10_000)
        assert torch.isfinite(loss), identifier
        loss.backward()
        gradients = [parameter.grad for parameter in rsdm.parameters() if parameter.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients), identifier
        assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients), identifier


def test_invalid_rsdm_values_fail_clearly() -> None:
    if not TORCH_AVAILABLE:
        print("SKIP: PyTorch unavailable; run constructor guard checks on demamba21.")
        return
    m5 = _load(M5_PATH)
    for value in (0, 2, -3, 14, True):
        invalid = copy.deepcopy(m5)
        invalid["model"]["rsdm_kernel_size"] = value
        try:
            _construct_before_backend(invalid)
        except ValueError as error:
            assert "rsdm_kernel_size" in str(error), (value, error)
        else:
            raise AssertionError(f"invalid rsdm_kernel_size accepted: {value!r}")
    for value in (0, -1, True):
        invalid = copy.deepcopy(m5)
        invalid["model"]["rsdm_dilation"] = value
        try:
            _construct_before_backend(invalid)
        except ValueError as error:
            assert "rsdm_dilation" in str(error), (value, error)
        else:
            raise AssertionError(f"invalid rsdm_dilation accepted: {value!r}")


def test() -> None:
    checks = (
        test_registry_diff_and_frozen_protocol,
        test_center_reuse_and_loss_protocol,
        test_default_m5_explicit_13_3_state_checkpoint_and_output_regression,
        test_nondefault_rsdm_shapes_loss_backward_and_shared_initialization,
        test_invalid_rsdm_values_fail_clearly,
    )
    for check in checks:
        check()
        print(f"PASS: {check.__name__}")


if __name__ == "__main__":
    test()
