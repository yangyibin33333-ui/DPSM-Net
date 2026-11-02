"""Direct regression checks for C40 eval-only mask inference.

Run with: ``python tests/test_c40_mask_only_inference.py``.
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
from models.decoder import BoundaryAwareDecoder


CONFIG_PATH = ROOT / "configs" / "experiments" / "C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
EXPECTED_PARAMS = 7_026_611
FULL_OUTPUT_KEYS = {"mask", "edge", "reliability", "route_s3", "route_s4"}


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _c40_model(device: torch.device) -> torch.nn.Module:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        model_cfg = copy.deepcopy(yaml.safe_load(handle)["model"])
    model_name = model_cfg.pop("name")
    return build_model(model_name, **model_cfg).to(device)


def test_full_and_mask_only_masks_are_bitwise_identical() -> None:
    device = _device()
    torch.manual_seed(42)
    model = _c40_model(device).eval()
    assert sum(parameter.numel() for parameter in model.parameters()) == EXPECTED_PARAMS

    with torch.inference_mode():
        for height, width in ((384, 544), (512, 512)):
            image = torch.randn(1, 3, height, width, device=device)
            full_outputs = model(image)
            mask_only_outputs = model.forward_mask_only(image)
            assert set(full_outputs) == FULL_OUTPUT_KEYS
            assert set(mask_only_outputs) == {"mask"}
            assert full_outputs["mask"].shape == (1, 1, height, width)
            assert mask_only_outputs["mask"].shape == (1, 1, height, width)
            assert torch.equal(full_outputs["mask"], mask_only_outputs["mask"])
            absolute_error = (full_outputs["mask"] - mask_only_outputs["mask"]).abs()
            assert absolute_error.max().item() == 0.0
            assert absolute_error.mean().item() == 0.0


def test_mask_only_rejects_training_mode() -> None:
    model = _c40_model(_device()).train()
    image = torch.randn(1, 3, 64, 80, device=_device())
    try:
        model.forward_mask_only(image)
    except RuntimeError as error:
        assert "eval mode" in str(error)
    else:
        raise AssertionError("Mask-only inference must reject training mode.")


def test_mask_only_rejects_auxiliary_logit_dependencies() -> None:
    options = {
        "use_frequency_gate": "frequency_gate",
        "use_boundary_logit_calibration": "boundary_logit_calibrator",
        "use_edge_mask_consistency_refiner": "edge_mask_consistency_refiner",
        "use_local_semantic_continuity_calibrator": "local_semantic_continuity_calibrator",
        "use_local_spectral_confidence_rejector": "local_spectral_confidence_rejector",
    }
    for option, dependency_name in options.items():
        decoder = BoundaryAwareDecoder([40, 80, 160, 320], 40, **{option: True}).eval()
        try:
            decoder.forward_mask_only([])
        except RuntimeError as error:
            assert dependency_name in str(error)
        else:
            raise AssertionError(f"Mask-only inference must reject {option}.")


def test_c40_state_dict_and_checkpoint_contract_are_unchanged() -> None:
    device = _device()
    torch.manual_seed(7)
    original = _c40_model(device).eval()
    original_state = {name: tensor.detach().clone() for name, tensor in original.state_dict().items()}
    original_keys = set(original_state)
    assert sum(parameter.numel() for parameter in original.parameters()) == EXPECTED_PARAMS

    reloaded = _c40_model(device).eval()
    result = reloaded.load_state_dict(original_state, strict=True)
    assert not result.missing_keys
    assert not result.unexpected_keys
    assert set(reloaded.state_dict()) == original_keys
    assert sum(parameter.numel() for parameter in reloaded.parameters()) == EXPECTED_PARAMS


def test() -> None:
    tests = (
        test_full_and_mask_only_masks_are_bitwise_identical,
        test_mask_only_rejects_training_mode,
        test_mask_only_rejects_auxiliary_logit_dependencies,
        test_c40_state_dict_and_checkpoint_contract_are_unchanged,
    )
    for check in tests:
        check()
        print(f"PASS: {check.__name__}")


if __name__ == "__main__":
    test()
