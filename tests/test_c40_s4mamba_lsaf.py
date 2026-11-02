"""Regression checks for C40-S4Mamba with the existing LSAF decoder.

Run with: ``python tests/test_c40_s4mamba_lsaf.py``.
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
from models.decoder import BoundaryAwareDecoder, LowResolutionSubpixelAdditiveUpBlock
from models.mamba_blocks import SequenceMambaCore


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
ALL_DIRECTIONS = ("horizontal", "vertical", "diagonal", "anti_diagonal")
RUNTIME_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build_from_config(path: Path) -> torch.nn.Module:
    model_cfg = copy.deepcopy(_load_config(path)["model"])
    model_name = model_cfg.pop("name")
    # The installed mamba_ssm selective scan is CUDA-only.  This preserves the
    # formal config while allowing the direct structural test on CPU hosts.
    if RUNTIME_DEVICE.type != "cuda":
        model_cfg["use_mamba_ssm"] = False
    return build_model(model_name, **model_cfg).to(RUNTIME_DEVICE)


def test_config_isolated_to_lsaf_and_save_dir() -> None:
    baseline = _load_config(S4MAMBA_CONFIG)
    candidate = _load_config(LSAF_CONFIG)
    assert {key: value for key, value in candidate.items() if key not in {"model", "save"}} == {
        key: value for key, value in baseline.items() if key not in {"model", "save"}
    }
    expected_model = copy.deepcopy(baseline["model"])
    expected_model["use_low_resolution_subpixel_additive_fusion"] = True
    assert candidate["model"] == expected_model
    assert candidate["save"]["save_dir"] == (
        "./runs/pilot/lightweight_c40_s4mamba_lsaf/"
        "C3_C40_S4MAMBA_LSAF_LR150_PHOTO_NATIVE384x544_EMA_BPCM"
    )


def test_lsaf_decoder_topology_and_training_contract() -> None:
    torch.manual_seed(42)
    model = _build_from_config(LSAF_CONFIG).train()
    decoder = model.decoder
    assert decoder.proj4 is None
    assert isinstance(decoder.up3, LowResolutionSubpixelAdditiveUpBlock)
    assert isinstance(decoder.up2, LowResolutionSubpixelAdditiveUpBlock)
    assert isinstance(decoder.up1, LowResolutionSubpixelAdditiveUpBlock)
    assert isinstance(model.stage3, LocalCrackBlock)
    assert not any(isinstance(module, SequenceMambaCore) for module in model.stage3.modules())
    assert isinstance(model.stage4, TopoRouteBlock)
    assert model.stage4.route_mamba.active_mamba_directions == ALL_DIRECTIONS
    assert model.stage4.route_mamba.active_mamba_path_count == len(ALL_DIRECTIONS)
    assert isinstance(model.stage4.global_branch, TransformerGlobalBlock)

    calls: dict[str, int] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for name, block in (("up3", decoder.up3), ("up2", decoder.up2), ("up1", decoder.up1)):
        for component in ("decoder_projection", "pixel_shuffle", "skip_projection", "refine"):
            key = f"{name}.{component}"
            calls[key] = 0
            module = getattr(block, component)
            handles.append(module.register_forward_hook(lambda *_args, key=key: calls.__setitem__(key, calls[key] + 1)))
    try:
        image = torch.randn(1, 3, 64, 80, device=RUNTIME_DEVICE)
        outputs = model(image)
    finally:
        for handle in handles:
            handle.remove()
    assert all(count == 1 for count in calls.values())
    assert {"mask", "edge", "reliability"}.issubset(outputs)
    assert outputs["mask"].shape == (1, 1, 64, 80)
    assert "route_s3" not in outputs and "route_s4" in outputs
    loss = outputs["mask"].mean() + outputs["edge"].mean() + outputs["reliability"].mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


def test_lsaf_mask_only_and_existing_configurations_remain_separate() -> None:
    model = _build_from_config(LSAF_CONFIG).eval()
    image = torch.randn(1, 3, 64, 80, device=RUNTIME_DEVICE)
    with torch.inference_mode():
        outputs = model.forward_mask_only(image)
    assert set(outputs) == {"mask"}
    assert outputs["mask"].shape == (1, 1, 64, 80)

    for config_path in (C40_CONFIG, S4MAMBA_CONFIG):
        reference = _build_from_config(config_path)
        assert not reference.decoder.use_low_resolution_subpixel_additive_fusion
        assert reference.decoder.proj4 is not None
        assert not isinstance(reference.decoder.up3, LowResolutionSubpixelAdditiveUpBlock)


def test_lsaf_rejects_all_declared_incompatible_decoder_options() -> None:
    incompatible_options = (
        "use_frequency_gate",
        "use_semantic_skip_gate",
        "use_detail_residual_gate",
        "use_structure_harmonic_fusion",
        "use_boundary_logit_calibration",
        "use_sdf_head",
        "use_edge_mask_consistency_refiner",
        "use_deep_supervision",
        "use_route_guided_directional_refiner",
        "use_route_guided_anisotropic_diffusion",
        "use_local_semantic_continuity_calibrator",
        "use_local_pixel_reassembly_residual",
        "use_local_spectral_confidence_rejector",
        "use_high_resolution_spectral_detail_gate",
    )
    for option in incompatible_options:
        try:
            BoundaryAwareDecoder(
                [40, 80, 160, 320],
                40,
                use_low_resolution_subpixel_additive_fusion=True,
                **{option: True},
            )
        except ValueError as error:
            assert "LSAF" in str(error)
        else:
            raise AssertionError(f"LSAF must reject {option}.")


def test() -> None:
    tests = (
        test_config_isolated_to_lsaf_and_save_dir,
        test_lsaf_decoder_topology_and_training_contract,
        test_lsaf_mask_only_and_existing_configurations_remain_separate,
        test_lsaf_rejects_all_declared_incompatible_decoder_options,
    )
    for check in tests:
        check()
        print(f"PASS: {check.__name__}")


if __name__ == "__main__":
    test()
