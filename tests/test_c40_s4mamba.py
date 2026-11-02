"""Regression checks for the C40 Stage4-only Mamba ablation.

Run with: ``python tests/test_c40_s4mamba.py``.
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
from models.mamba_blocks import SequenceMambaCore
from losses import TangentRouteLoss


BASELINE_CONFIG = ROOT / "configs" / "experiments" / "C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
S4MAMBA_CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "lightweight_c40_s4mamba"
    / "C3_C40_S4MAMBA_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
ALL_DIRECTIONS = ("horizontal", "vertical", "diagonal", "anti_diagonal")
RUNTIME_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build_s4mamba() -> torch.nn.Module:
    model_cfg = copy.deepcopy(_load_config(S4MAMBA_CONFIG)["model"])
    model_name = model_cfg.pop("name")
    # mamba_ssm's selective-scan backend is CUDA-only.  Preserve the formal
    # YAML setting, but use the existing local fallback only when this direct
    # regression test is run on a CPU-only host.
    if RUNTIME_DEVICE.type != "cuda":
        model_cfg["use_mamba_ssm"] = False
    return build_model(model_name, **model_cfg).to(RUNTIME_DEVICE)


def test_config_changes_only_mamba_stages() -> None:
    baseline = _load_config(BASELINE_CONFIG)
    candidate = _load_config(S4MAMBA_CONFIG)
    assert {key: value for key, value in candidate.items() if key not in {"model", "save"}} == {
        key: value for key, value in baseline.items() if key not in {"model", "save"}
    }
    expected_model = copy.deepcopy(baseline["model"])
    expected_model["mamba_stages"] = [4]
    assert candidate["model"] == expected_model
    assert candidate["save"]["save_dir"] == (
        "./runs/pilot/lightweight_c40_s4mamba/"
        "C3_C40_S4MAMBA_LR150_PHOTO_NATIVE384x544_EMA_BPCM"
    )


def test_s4mamba_structure_and_forward_backward_contract() -> None:
    torch.manual_seed(42)
    model = _build_s4mamba().train()
    assert isinstance(model.stage3, LocalCrackBlock)
    assert not any(isinstance(module, SequenceMambaCore) for module in model.stage3.modules())
    assert isinstance(model.stage4, TopoRouteBlock)
    assert model.stage4.route_mamba.active_mamba_directions == ALL_DIRECTIONS
    assert model.stage4.route_mamba.active_mamba_path_count == len(ALL_DIRECTIONS)
    assert isinstance(model.stage4.global_branch, TransformerGlobalBlock)
    assert tuple(model.mamba_backend_info()) == ("stage4",)

    image = torch.randn(1, 3, 64, 80, device=RUNTIME_DEVICE)
    outputs = model(image)
    assert {"mask", "edge", "reliability"}.issubset(outputs)
    assert outputs["mask"].shape == (1, 1, 64, 80)
    assert "route_s3" not in outputs
    assert "route_s4" in outputs
    tangent_gt = torch.zeros(1, 4, 64, 80, device=RUNTIME_DEVICE)
    tangent_gt[:, 0] = 1.0
    tangent_valid = torch.ones(1, 1, 64, 80, device=RUNTIME_DEVICE)
    route_loss = TangentRouteLoss()(outputs, tangent_gt, tangent_valid)
    assert torch.isfinite(route_loss)
    loss = outputs["mask"].mean() + outputs["edge"].mean() + outputs["reliability"].mean() + route_loss
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


def test_s4mamba_mask_only_eval_path_and_baseline_topology() -> None:
    model = _build_s4mamba().eval()
    image = torch.randn(1, 3, 64, 80, device=RUNTIME_DEVICE)
    with torch.inference_mode():
        mask_only_outputs = model.forward_mask_only(image)
    assert set(mask_only_outputs) == {"mask"}
    assert mask_only_outputs["mask"].shape == (1, 1, 64, 80)

    baseline_cfg = copy.deepcopy(_load_config(BASELINE_CONFIG)["model"])
    baseline_name = baseline_cfg.pop("name")
    if RUNTIME_DEVICE.type != "cuda":
        baseline_cfg["use_mamba_ssm"] = False
    baseline = build_model(baseline_name, **baseline_cfg).to(RUNTIME_DEVICE)
    assert isinstance(baseline.stage3, TopoRouteBlock)
    assert baseline.stage3.route_mamba.active_mamba_directions == ALL_DIRECTIONS
    assert isinstance(baseline.stage4, TopoRouteBlock)
    assert isinstance(baseline.stage4.global_branch, TransformerGlobalBlock)


def test() -> None:
    tests = (
        test_config_changes_only_mamba_stages,
        test_s4mamba_structure_and_forward_backward_contract,
        test_s4mamba_mask_only_eval_path_and_baseline_topology,
    )
    for check in tests:
        check()
        print(f"PASS: {check.__name__}")


if __name__ == "__main__":
    test()
