"""Static audit for the seed-42 Main Table A configuration registry.

Run from the repository root with:
    python tests/test_paper_main_table_a_configs.py

This test intentionally does not construct a model, invoke mamba_ssm, inspect a
dataset, or execute an optimizer update.  It is safe on a CPU-only workstation.
"""

from __future__ import annotations

from copy import deepcopy
import inspect
from pathlib import Path, PurePosixPath
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.de_mambanet import DEMambaNet


REGISTRY = ROOT / "configs" / "experiments" / "paper_main_table_a"
FROZEN_DPSM = ROOT / "configs" / "experiments" / "auto_protocol" / "C40_DPSM80_AUTO_PROTOCOL_V2.yaml"
IDS = (
    ("A0", "A0_C40_S34"),
    ("A1", "A1_C40_S4"),
    ("A2", "A2_C40_S4_LSSCF"),
    ("A3", "A3_C40_S4_LSSCF_DPER"),
    ("A4", "A4_C40_S4_LSSCF_DPER_RSDM"),
    ("A5", "A5_C40_S4_LSSCF_DPER_RSDM_DPSM80"),
)
DATASETS = {"deepcrack": "./DeepCrack", "crack500": "./Crack500"}
FAILED_MODEL_FLAGS = (
    "use_common_private_direction_partitioned_shared_mamba",  # CP-DPSM
    "use_full_width_residual_mask_head",  # FWRMH
    "use_difference_aware_up1_fusion",  # DAFU1
    "use_depthwise_separable_mask_head",  # DSMH
    "use_narrow_transport_decoder",  # NTD
    "use_partial_narrow_transport_decoder",  # PNTD
    "use_subpixel_phase_consistency_refiner",  # SPCR
    "use_haar_topology_preserving_rebase",  # HTPR
    "use_full_resolution_joint_exchange",  # FJSE
    "use_early_serpentine_connectivity_ssm",  # ESCS
    "use_partial_full_resolution_stage1",  # PFRS
    "use_multiscale_directional_reallocation",  # MSDR
    "use_low_resolution_subpixel_additive_fusion",  # LSAF
    "use_narrow_dual_resolution_stage1",
    "use_sdf_head",
    "use_edge_mask_consistency_refiner",
    "use_retinex_directional_valley_prior",
    "use_route_guided_directional_refiner",
    "use_route_guided_anisotropic_diffusion",
    "use_density_balanced_prototype_context",
    "use_local_semantic_continuity_calibrator",
    "use_local_pixel_reassembly_residual",
    "use_local_spectral_confidence_rejector",
    "use_high_resolution_spectral_detail_gate",
    "return_training_mask_feature",
    "use_semantic_conditioned_local_kernel_residual",
    "use_residual_branch_competition_fusion",
)


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    assert isinstance(value, dict), f"{path} must contain a mapping"
    return value


def _config(dataset: str, stem: str) -> dict:
    return _load(REGISTRY / dataset / f"{stem}.yaml")


def _normalized_relative_path(value: object) -> str:
    """Compare repository-relative save paths without a cosmetic './' prefix."""

    return str(PurePosixPath(str(value).replace("\\", "/")))


def _model_defaults() -> dict:
    defaults: dict = {}
    for name, parameter in inspect.signature(DEMambaNet.__init__).parameters.items():
        if name != "self" and parameter.default is not inspect.Parameter.empty:
            defaults[name] = deepcopy(parameter.default)
    return defaults


def _effective_model(model: dict) -> dict:
    resolved = _model_defaults()
    resolved.update(deepcopy(model))
    # DEMambaNet treats None as "derive from transformer_stages".  The frozen
    # V2 source uses that default while the registry records the resulting True
    # state explicitly, so canonicalize the two equivalent forms before the
    # frozen-A5 comparison.
    if resolved["use_transformer"] is None:
        resolved["use_transformer"] = bool(resolved["transformer_stages"])
    return resolved


def _only_model_difference(before: dict, after: dict, expected: str) -> None:
    differing = {
        key
        for key in set(before) | set(after)
        if before.get(key) != after.get(key)
    }
    assert differing == {expected}, f"expected only {expected}, got {sorted(differing)}"


def _without_model_and_save(config: dict) -> dict:
    result = deepcopy(config)
    result.pop("model")
    result.pop("save")
    return result


def test_all_yaml_and_main_chain() -> None:
    for dataset, data_root in DATASETS.items():
        configs = {identifier: _config(dataset, stem) for identifier, stem in IDS}
        assert len(configs) == 6
        for identifier, config in configs.items():
            assert config["data"]["data_root"] == data_root
            expected_save_dir = f"runs/paper/main_table_a/{dataset}/{dict(IDS)[identifier]}"
            actual_save_dir = _normalized_relative_path(config["save"]["save_dir"])
            assert actual_save_dir == expected_save_dir, (
                f"{dataset}/{identifier}: save_dir must be unique and repository-relative; "
                f"expected {expected_save_dir!r}, got {actual_save_dir!r} "
                f"(raw={config['save']['save_dir']!r})"
            )
            assert config["model"]["base_channels"] == 40
            assert config["model"]["transformer_stages"] == [4]
            assert config["model"]["route_bins"] == 4
            assert config["model"]["early_detail_channels"] == 16
            assert config["model"]["stage4_direction_group_channels"] == 80
            assert config["loss"]["lambda_cfps"] == 0.0
            assert config["distillation"] == {"enabled": False}
            for option in FAILED_MODEL_FLAGS:
                assert option in config["model"], f"{dataset}/{identifier} must explicitly disable {option}"
                assert config["model"][option] is False

        reference = _without_model_and_save(configs["A0"])
        for identifier in ("A1", "A2", "A3", "A4", "A5"):
            assert _without_model_and_save(configs[identifier]) == reference, (
                f"{dataset}/{identifier} changed a non-model training/protocol setting"
            )

        _only_model_difference(configs["A0"]["model"], configs["A1"]["model"], "mamba_stages")
        _only_model_difference(
            configs["A1"]["model"],
            configs["A2"]["model"],
            "use_low_resolution_subpixel_separable_concat_fusion",
        )
        _only_model_difference(
            configs["A2"]["model"], configs["A3"]["model"], "use_detail_preserving_early_rebase"
        )
        _only_model_difference(configs["A3"]["model"], configs["A4"]["model"], "use_rsdm_stage1")
        _only_model_difference(
            configs["A4"]["model"],
            configs["A5"]["model"],
            "use_direction_partitioned_shared_mamba",
        )

        assert configs["A0"]["model"]["mamba_stages"] == [3, 4]
        for identifier in ("A1", "A2", "A3", "A4", "A5"):
            assert configs[identifier]["model"]["mamba_stages"] == [4]
        assert configs["A1"]["model"]["use_low_resolution_subpixel_separable_concat_fusion"] is False
        assert configs["A2"]["model"]["use_low_resolution_subpixel_separable_concat_fusion"] is True
        assert configs["A3"]["model"]["use_detail_preserving_early_rebase"] is True
        assert configs["A4"]["model"]["use_rsdm_stage1"] is True


def test_cross_dataset_and_frozen_a5_contract() -> None:
    frozen = _load(FROZEN_DPSM)
    for identifier, stem in IDS:
        deepcrack = _config("deepcrack", stem)
        crack500 = _config("crack500", stem)
        for section in ("model", "loss", "train", "auto_protocol"):
            assert deepcrack[section] == crack500[section], f"{identifier} changed {section} across datasets"
        assert deepcrack["data"] == {**crack500["data"], "data_root": "./DeepCrack"}
        assert _normalized_relative_path(deepcrack["save"]["save_dir"]).startswith(
            "runs/paper/main_table_a/deepcrack/"
        )
        assert _normalized_relative_path(crack500["save"]["save_dir"]).startswith(
            "runs/paper/main_table_a/crack500/"
        )

    for dataset in DATASETS:
        a4 = _config(dataset, dict(IDS)["A4"])["model"]
        a5 = _config(dataset, dict(IDS)["A5"])["model"]
        assert a4["use_direction_partitioned_shared_mamba"] is False
        assert a4["mamba_stages"] == [4]
        assert a4["stage4_mamba_spatial_reduction"] == 1
        assert a4["stage4_mamba_channels"] is None
        assert a4["stage4_active_mamba_directions"] is None
        assert a5["use_direction_partitioned_shared_mamba"] is True
        assert a5["stage4_direction_group_channels"] == 80
        # The registry makes default-false historical branches explicit; compare
        # the constructor-resolved model to the frozen final V2 configuration.
        assert _effective_model(a5) == _effective_model(frozen["model"])


def test() -> None:
    for check in (test_all_yaml_and_main_chain, test_cross_dataset_and_frozen_a5_contract):
        check()
        print(f"PASS: {check.__name__}")


if __name__ == "__main__":
    test()
