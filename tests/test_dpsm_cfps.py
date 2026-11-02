"""Regression tests for train-only DPER-RSDM-DPSM80-CFPS."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from losses import DEMambaLoss
from losses.loss import CohesiveFalsePositiveSuppressionLoss
from models import build_model
from models.blocks import RIFTStructuralDirectionalModelingBlock
from models.mamba_blocks import DirectionPartitionedSharedMamba, SequenceMambaCore
from models.transformer_blocks import TransformerGlobalBlock
from train import build_distillation_teacher


BASE_CONFIG = (
    ROOT / "configs/experiments/lightweight_c40_s4mamba_lsscf_dper_rsdm_dpsm"
    / "C3_C40_S4MAMBA_LSSCF_DPER_RSDM_DPSM80_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
CFPS_CONFIG = (
    ROOT / "configs/experiments/lightweight_c40_s4mamba_lsscf_dper_rsdm_dpsm_cfps"
    / "C3_C40_S4MAMBA_LSSCF_DPER_RSDM_DPSM80_CFPS_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build(path: Path):
    cfg = copy.deepcopy(_load(path)["model"])
    name = cfg.pop("name")
    if DEVICE.type != "cuda":
        cfg["use_mamba_ssm"] = False
    return build_model(name, **cfg).to(DEVICE)


def test_cfps_config_is_a_strict_dpsm_training_only_change() -> None:
    baseline, candidate = _load(BASE_CONFIG), _load(CFPS_CONFIG)
    assert candidate["model"] == baseline["model"]
    for section in ("data", "train", "evaluation"):
        assert candidate[section] == baseline[section], section
    for name, value in baseline["loss"].items():
        assert candidate["loss"][name] == value, name
    assert {name: value for name, value in candidate["loss"].items() if name not in baseline["loss"]} == {
        "lambda_cfps": 0.010,
        "cfps_start_epoch": 90,
        "cfps_ramp_epochs": 30,
        "cfps_exclusion_kernel_size": 5,
        "cfps_cohesion_kernel_size": 3,
        "cfps_min_active_neighbors": 3,
        "cfps_margin": 0.25,
    }
    assert "distillation" not in candidate
    assert build_distillation_teacher(candidate.get("distillation"), device=DEVICE) is None
    assert candidate["save"] != baseline["save"]


def test_student_topology_parameters_and_deployment_are_unchanged() -> None:
    torch.manual_seed(2301)
    baseline = _build(BASE_CONFIG).eval()
    torch.manual_seed(2301)
    candidate = _build(CFPS_CONFIG).eval()
    assert baseline.state_dict().keys() == candidate.state_dict().keys()
    assert all(torch.equal(value, candidate.state_dict()[name]) for name, value in baseline.state_dict().items())
    assert sum(parameter.numel() for parameter in baseline.parameters()) == sum(
        parameter.numel() for parameter in candidate.parameters()
    )
    route = candidate.stage4.route_mamba
    assert isinstance(route, DirectionPartitionedSharedMamba)
    assert route.group_channels == 80 and route.instantiated_core_count == 1
    assert sum(isinstance(module, SequenceMambaCore) for module in route.modules()) == 1
    assert isinstance(candidate.stage4.global_branch, TransformerGlobalBlock)
    assert sum(isinstance(module, RIFTStructuralDirectionalModelingBlock) for module in candidate.modules()) == 1
    assert not any("cfps" in name.lower() or "teacher" in name.lower() for name in candidate.state_dict())
    if DEVICE.type == "cuda":
        assert candidate.mamba_backend_info()["stage4"] == ["mamba_ssm"]

    image = torch.randn(1, 3, 64, 80, device=DEVICE)
    with torch.no_grad():
        full = candidate(image)
        mask_only = candidate.forward_mask_only(image)
    assert torch.equal(full["mask"], mask_only["mask"])


def test_cfps_schedule_is_one_based_and_exact() -> None:
    criterion = DEMambaLoss(lambda_cfps=0.010, cfps_start_epoch=90, cfps_ramp_epochs=30)
    assert criterion._cfps_scale(90) == 0.0
    assert 0.0 < criterion._cfps_scale(91) < 0.010
    assert abs(criterion._cfps_scale(105) - 0.005) < 1e-12
    assert criterion._cfps_scale(120) == 0.010
    assert criterion._cfps_scale(200) == 0.010


def test_cfps_selection_protection_cohesion_detach_and_gradients() -> None:
    loss_fn = CohesiveFalsePositiveSuppressionLoss(
        exclusion_kernel_size=5,
        cohesion_kernel_size=3,
        min_active_neighbors=3,
        margin=0.25,
    ).to(DEVICE)
    logits = torch.full((1, 1, 11, 11), -10.0, device=DEVICE, requires_grad=True)
    target = torch.zeros_like(logits)
    target[0, 0, 5, 5] = 1.0
    with torch.no_grad():
        logits[0, 0, 5, 5] = 4.0       # GT foreground
        logits[0, 0, 5, 7] = 4.0       # protected band
        logits[0, 0, 1, 1:4] = 1.0     # cohesive far-background line
        logits[0, 0, 9, 9] = 1.0       # isolated far-background point
        logits[0, 0, 9, 1:4] = -0.01   # cohesive shape below threshold
    selected = loss_fn.selected_mask(logits, target)
    assert selected.dtype == torch.bool and not selected.requires_grad and selected.grad_fn is None
    assert not selected[0, 0, 5, 5] and not selected[0, 0, 5, 7]
    assert not selected[0, 0, 9, 9] and not selected[0, 0, 9, 1:4].any()
    assert selected[0, 0, 1, 2]
    assert not selected[0, 0, 1, 1] and not selected[0, 0, 1, 3]
    loss = loss_fn.loss_from_selected(logits, selected)
    loss.backward()
    assert torch.isfinite(loss) and logits.grad is not None
    assert logits.grad[0, 0, 1, 2] > 0
    assert torch.count_nonzero(logits.grad[~selected]).item() == 0

    empty_logits = torch.full((1, 1, 7, 7), -10.0, device=DEVICE, requires_grad=True)
    empty_target = torch.zeros_like(empty_logits)
    empty_loss = loss_fn(empty_logits, empty_target)
    assert empty_loss.item() == 0.0 and empty_loss.requires_grad
    empty_loss.backward()
    assert empty_logits.grad is not None and torch.count_nonzero(empty_logits.grad).item() == 0


def test_cfps_total_loss_statistics_and_fp32_amp_are_valid() -> None:
    baseline = DEMambaLoss(lambda_edge=0.0, lambda_tangent=0.0, scnp_weight=0.0).to(DEVICE)
    criterion = DEMambaLoss(
        lambda_edge=0.0,
        lambda_tangent=0.0,
        scnp_weight=0.0,
        lambda_cfps=0.010,
        cfps_start_epoch=90,
        cfps_ramp_epochs=30,
        cfps_exclusion_kernel_size=5,
        cfps_cohesion_kernel_size=3,
        cfps_min_active_neighbors=3,
        cfps_margin=0.25,
    ).to(DEVICE)
    logits = torch.full((2, 1, 11, 11), -4.0, device=DEVICE, requires_grad=True)
    with torch.no_grad():
        logits[0, 0, 1, 1:4] = 1.0
    outputs = {"mask": logits, "edge": torch.zeros_like(logits)}
    batch = {
        "mask": torch.zeros_like(logits),
        "edge_gt": torch.zeros_like(logits),
        "tangent_gt": torch.zeros(2, 4, 11, 11, device=DEVICE),
        "tangent_valid": torch.zeros_like(logits),
    }
    baseline_total, _ = baseline(outputs, batch, epoch=104)
    total, items = criterion(outputs, batch, epoch=104, cfps_epoch=105)
    assert abs(items["cfps_scale"] - 0.005) < 1e-12
    assert items["cfps_loss"] > 0.0
    assert abs(items["cfps_weighted_loss"] - items["cfps_scale"] * items["cfps_loss"]) < 1e-7
    assert items["cfps_selected_pixels_per_image"] == 0.5
    assert items["cfps_active_image_ratio"] == 0.5
    assert torch.allclose(total, baseline_total + logits.new_tensor(items["cfps_weighted_loss"]), atol=1e-6)
    total.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert not list(criterion.cfps_loss.parameters())

    if DEVICE.type == "cuda":
        amp_logits = logits.detach().half().requires_grad_(True)
        amp_target = torch.zeros_like(amp_logits)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            amp_loss = criterion.cfps_loss(amp_logits, amp_target)
        assert torch.isfinite(amp_loss)
        amp_loss.backward()
        assert amp_logits.grad is not None and torch.isfinite(amp_logits.grad).all()


def run_all() -> None:
    checks = (
        test_cfps_config_is_a_strict_dpsm_training_only_change,
        test_student_topology_parameters_and_deployment_are_unchanged,
        test_cfps_schedule_is_one_based_and_exact,
        test_cfps_selection_protection_cohesion_detach_and_gradients,
        test_cfps_total_loss_statistics_and_fp32_amp_are_valid,
    )
    for check in checks:
        check()
        print(f"PASS: {check.__name__}")


if __name__ == "__main__":
    run_all()
