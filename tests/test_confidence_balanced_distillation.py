from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from losses import ConfidenceBalancedDistillationLoss, distillation_weight_for_epoch
from models import DEMambaNetLiteProWideEncoder, build_model
from train import build_distillation_teacher
from utils import load_checkpoint, save_checkpoint


def _loss() -> ConfidenceBalancedDistillationLoss:
    return ConfidenceBalancedDistillationLoss(
        temperature=2.0,
        foreground_weight=0.45,
        background_weight=0.55,
        confidence_power=2.0,
    )


def test_matching_logits_are_better_than_opposed_logits() -> None:
    teacher = torch.tensor([[[[4.0, -4.0], [3.0, -3.0]]]])
    target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    matching = _loss()(teacher.clone().requires_grad_(), teacher, target)
    opposed = _loss()((-teacher).requires_grad_(), teacher, target)
    assert matching < opposed


def test_teacher_is_detached_and_student_receives_gradient() -> None:
    student = torch.randn(1, 1, 3, 5, requires_grad=True)
    teacher = torch.randn(1, 1, 3, 5, requires_grad=True)
    target = torch.tensor([[[[1, 0, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 0, 0]]]], dtype=torch.float32)
    loss = _loss()(student, teacher, target)
    loss.backward()
    assert student.grad is not None and student.grad.abs().sum().item() > 0.0
    assert teacher.grad is None
    teacher_module = torch.nn.Conv2d(1, 1, kernel_size=1)
    image = torch.randn(1, 1, 3, 5)
    with torch.no_grad():
        frozen_teacher_logits = teacher_module(image)
    module_student = torch.randn_like(frozen_teacher_logits, requires_grad=True)
    _loss()(module_student, frozen_teacher_logits, target).backward()
    assert module_student.grad is not None and module_student.grad.abs().sum().item() > 0.0
    assert all(parameter.grad is None for parameter in teacher_module.parameters())


def test_foreground_background_are_independently_normalized() -> None:
    criterion = _loss()
    student = torch.zeros(1, 1, 2, 5, requires_grad=True)
    teacher = torch.tensor([[[[4.0, -4.0, -4.0, -4.0, -4.0], [-4.0, -4.0, -4.0, -4.0, -4.0]]]])
    target = torch.tensor([[[[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]]])
    loss, stats = criterion(student, teacher, target, return_stats=True)
    expected = 0.45 * stats["foreground_loss"] + 0.55 * stats["background_loss"]
    assert torch.allclose(loss.detach(), expected)
    assert stats["foreground_loss"] > 0.0 and stats["background_loss"] > 0.0


def test_empty_foreground_and_fp16_inputs_are_finite() -> None:
    student = torch.zeros(1, 1, 2, 4, dtype=torch.float16, requires_grad=True)
    teacher = torch.full((1, 1, 2, 4), -4.0, dtype=torch.float16, requires_grad=True)
    target = torch.zeros(1, 1, 2, 4, dtype=torch.float16)
    loss, stats = _loss()(student, teacher, target, return_stats=True)
    loss.backward()
    assert torch.isfinite(loss)
    assert stats["foreground_loss"].item() == 0.0
    assert teacher.grad is None
    assert student.grad is not None and torch.isfinite(student.grad).all()


def test_weight_schedule_matches_frozen_protocol() -> None:
    assert distillation_weight_for_epoch(1) == 0.0
    assert distillation_weight_for_epoch(10) == 0.0
    assert 0.0 < distillation_weight_for_epoch(25) < 0.25
    assert distillation_weight_for_epoch(40) == 0.25
    assert distillation_weight_for_epoch(200) == 0.25


def test_disabled_distillation_does_not_construct_teacher() -> None:
    assert build_distillation_teacher({"enabled": False}, device=torch.device("cpu")) is None


@torch.no_grad()
def test_student_checkpoint_and_forward_stay_teacher_free() -> None:
    student = build_model("DEMambaNetLiteProWideEncoder", use_mamba_ssm=False, drop_path_rate=0.0).eval()
    parameter_count = sum(parameter.numel() for parameter in student.parameters())
    reference = student(torch.randn(1, 3, 64, 80))
    assert build_distillation_teacher({"enabled": False}, device=torch.device("cpu")) is None
    after = student(torch.randn(1, 3, 64, 80))
    assert sum(parameter.numel() for parameter in student.parameters()) == parameter_count
    assert set(reference) == set(after) == {"mask", "edge", "reliability", "route_s1", "route_s3", "route_s4"}

    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "student.pth"
        save_checkpoint(path, student, optimizer, None, epoch=0, best_score=0.0)
        payload = torch.load(path, map_location="cpu")
        assert not any("teacher" in name.lower() for name in payload["model"])
        restored = DEMambaNetLiteProWideEncoder(use_mamba_ssm=False, drop_path_rate=0.0).eval()
        load_checkpoint(path, restored, map_location="cpu")
        restored_output = restored(torch.randn(1, 3, 64, 80))
        assert set(restored_output) == set(reference)


def test_cbkd_config_preserves_wide_encoder_protocol() -> None:
    import yaml

    wide_path = (
        REPO_ROOT
        / "configs/experiments/lightweight_demambanet_lite_pro_wide_encoder"
        / "DE_MAMBANET_LITE_PRO_WIDE_ENCODER_LR150_DEEPCRACK_NATIVE384x544_EMA_BPCM.yaml"
    )
    cbkd_path = (
        REPO_ROOT
        / "configs/experiments/lightweight_demambanet_lite_pro_wide_encoder_cbkd"
        / "DE_MAMBANET_LITE_PRO_WIDE_ENCODER_CBKD_LR150_DEEPCRACK_NATIVE384x544_EMA_BPCM.yaml"
    )
    with wide_path.open(encoding="utf-8") as handle:
        wide = yaml.safe_load(handle)
    with cbkd_path.open(encoding="utf-8") as handle:
        cbkd = yaml.safe_load(handle)
    assert cbkd["model"] == wide["model"]
    for section in ("data", "train", "loss", "evaluation"):
        assert cbkd[section] == wide[section], section
    assert cbkd["distillation"]["enabled"] is True
    assert cbkd["distillation"]["teacher_checkpoint"] is None
    assert cbkd["save"] != wide["save"]


if __name__ == "__main__":
    tests = (
        test_matching_logits_are_better_than_opposed_logits,
        test_teacher_is_detached_and_student_receives_gradient,
        test_foreground_background_are_independently_normalized,
        test_empty_foreground_and_fp16_inputs_are_finite,
        test_weight_schedule_matches_frozen_protocol,
        test_disabled_distillation_does_not_construct_teacher,
        test_student_checkpoint_and_forward_stay_teacher_free,
        test_cbkd_config_preserves_wide_encoder_protocol,
    )
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
