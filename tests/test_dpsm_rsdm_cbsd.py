"""Regression tests for training-only RSDM -> DPSM80 mask-logit CBSD."""

from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train
from losses import ConfidenceBalancedDistillationLoss, distillation_weight_for_epoch
from models import build_model
from models.blocks import RIFTStructuralDirectionalModelingBlock
from models.mamba_blocks import DirectionPartitionedSharedMamba, SequenceMambaCore, TangentRoutedMambaBlock
from models.transformer_blocks import TransformerGlobalBlock
from utils import ModelEMA, save_checkpoint


DPSM_CONFIG = (
    ROOT / "configs/experiments/lightweight_c40_s4mamba_lsscf_dper_rsdm_dpsm"
    / "C3_C40_S4MAMBA_LSSCF_DPER_RSDM_DPSM80_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
CBSD_CONFIG = (
    ROOT / "configs/experiments/lightweight_c40_s4mamba_lsscf_dper_rsdm_dpsm_cbsd"
    / "C3_C40_S4MAMBA_LSSCF_DPER_RSDM_DPSM80_CBSD_RSDM_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
RSDM_CONFIG = (
    ROOT / "configs/experiments/lightweight_c40_s4mamba_lsscf_dper_rsdm"
    / "C3_C40_S4MAMBA_LSSCF_DPER_RSDM_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
C40_CONFIG = ROOT / "configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
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


def test_cbsd_config_is_strictly_isolated() -> None:
    baseline, candidate = _load(DPSM_CONFIG), _load(CBSD_CONFIG)
    assert candidate["model"] == baseline["model"]
    for section in ("data", "train", "loss", "evaluation"):
        assert candidate[section] == baseline[section], section
    assert candidate["save"] != baseline["save"]
    assert candidate["distillation"] == {
        "enabled": True,
        "teacher_config": "configs/experiments/lightweight_c40_s4mamba_lsscf_dper_rsdm/C3_C40_S4MAMBA_LSSCF_DPER_RSDM_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml",
        "teacher_checkpoint": None,
        "temperature": 2.0,
        "lambda_max": 0.20,
        "foreground_weight": 0.45,
        "background_weight": 0.55,
        "confidence_power": 2.0,
        "warmup_epochs": 10,
        "ramp_end_epoch": 40,
    }


def test_student_and_teacher_topologies_are_exact() -> None:
    student, teacher = _build(CBSD_CONFIG).eval(), _build(RSDM_CONFIG).eval()
    student_route = student.stage4.route_mamba
    assert isinstance(student_route, DirectionPartitionedSharedMamba)
    assert student_route.dim == 320 and student_route.group_channels == 80
    assert student_route.instantiated_core_count == 1
    assert sum(isinstance(module, SequenceMambaCore) for module in student_route.modules()) == 1
    assert isinstance(student.stage4.global_branch, TransformerGlobalBlock)
    assert sum(isinstance(module, RIFTStructuralDirectionalModelingBlock) for module in student.modules()) == 1

    assert isinstance(teacher.stage4.route_mamba, TangentRoutedMambaBlock)
    assert teacher.stage4.route_mamba.active_mamba_path_count == 4
    assert teacher.stage4.route_mamba.mamba_channels == 320
    assert teacher.stage4.route_mamba.spatial_reduction == 1
    assert isinstance(teacher.stage4.global_branch, TransformerGlobalBlock)
    assert sum(isinstance(module, RIFTStructuralDirectionalModelingBlock) for module in teacher.modules()) == 1
    assert train._validate_cbsd_teacher_variant(_load(RSDM_CONFIG)["model"], teacher) == "DPER-RSDM"
    if DEVICE.type == "cuda":
        assert student.mamba_backend_info()["stage4"] == ["mamba_ssm"]
        assert teacher.mamba_backend_info()["stage4"] == ["mamba_ssm"] * 4


def test_schedule_matches_fixed_protocol() -> None:
    kwargs = {"lambda_max": 0.20, "warmup_epochs": 10, "ramp_end_epoch": 40}
    assert distillation_weight_for_epoch(1, **kwargs) == 0.0
    assert distillation_weight_for_epoch(10, **kwargs) == 0.0
    assert 0.0 < distillation_weight_for_epoch(11, **kwargs) < 0.20
    assert distillation_weight_for_epoch(40, **kwargs) == 0.20
    assert distillation_weight_for_epoch(200, **kwargs) == 0.20


def test_existing_formal_c40_teacher_variant_remains_allowed() -> None:
    teacher = _build(C40_CONFIG).eval()
    assert train._validate_cbsd_teacher_variant(_load(C40_CONFIG)["model"], teacher) == "C40"


def test_teacher_strict_load_freeze_and_checkpoint_rejection() -> None:
    reference = _build(RSDM_CONFIG).eval()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint_path = root / "best_structural_score.pth"
        torch.save({"model": reference.state_dict()}, checkpoint_path)
        teacher_cfg = _load(RSDM_CONFIG)
        teacher_cfg["model"]["use_mamba_ssm"] = DEVICE.type == "cuda"
        teacher_config_path = root / "teacher.yaml"
        teacher_config_path.write_text(yaml.safe_dump(teacher_cfg, sort_keys=False), encoding="utf-8")
        cfg = copy.deepcopy(_load(CBSD_CONFIG)["distillation"])
        cfg.update(teacher_config=str(teacher_config_path), teacher_checkpoint=str(checkpoint_path))

        original_backend_check = train._require_real_mamba_backends
        if DEVICE.type != "cuda":
            train._require_real_mamba_backends = lambda model, role: model.mamba_backend_info()
        try:
            teacher = train.build_distillation_teacher(cfg, device=DEVICE)
        finally:
            train._require_real_mamba_backends = original_backend_check
        assert teacher is not None and not teacher.training
        assert all(not parameter.requires_grad for parameter in teacher.parameters())
        assert all(parameter.grad is None for parameter in teacher.parameters())
        assert teacher.state_dict().keys() == reference.state_dict().keys()

        broken = dict(reference.state_dict())
        broken.pop(next(iter(broken)))
        torch.save({"model": broken}, checkpoint_path)
        try:
            train._load_frozen_teacher_checkpoint(checkpoint_path, reference, map_location=DEVICE)
        except RuntimeError:
            pass
        else:
            raise AssertionError("strict teacher loading accepted a missing state_dict key")


def test_mask_only_cbsd_backward_optimizer_ema_and_checkpoint_isolation() -> None:
    torch.manual_seed(2201)
    student, teacher = _build(CBSD_CONFIG), _build(RSDM_CONFIG).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    image = torch.randn(1, 3, 64, 80, device=DEVICE)
    target = (torch.rand(1, 1, 64, 80, device=DEVICE) < 0.08).float()
    student_outputs = student(image)
    with torch.no_grad():
        teacher_outputs = teacher(image)
    assert student_outputs["mask"].shape == teacher_outputs["mask"].shape == target.shape
    criterion = ConfidenceBalancedDistillationLoss(
        temperature=2.0,
        foreground_weight=0.45,
        background_weight=0.55,
        confidence_power=2.0,
    ).to(DEVICE)
    teacher_logits = teacher_outputs["mask"].detach().requires_grad_(True)
    loss = criterion(student_outputs["mask"], teacher_logits, target)
    assert loss.dtype == torch.float32 and loss.ndim == 0 and torch.isfinite(loss) and loss.item() > 0.0
    loss.backward()
    gradients = [parameter.grad for parameter in student.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)
    assert teacher_logits.grad is None
    assert all(parameter.grad is None for parameter in teacher.parameters())

    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    student_ids = {id(parameter) for parameter in student.parameters()}
    teacher_ids = {id(parameter) for parameter in teacher.parameters()}
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert optimizer_ids == student_ids and optimizer_ids.isdisjoint(teacher_ids)
    ema = ModelEMA(student)
    assert ema.state_dict().keys() == student.state_dict().keys()
    assert not any("teacher" in key.lower() for key in ema.state_dict())

    student.eval()
    with torch.no_grad():
        full_mask = student(image)["mask"]
        mask_only = student.forward_mask_only(image)["mask"]
    assert torch.equal(full_mask, mask_only)
    assert not any("teacher" in key.lower() for key in student.state_dict())
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "student.pth"
        save_checkpoint(path, student, optimizer, None, epoch=0, best_score=0.0)
        payload = torch.load(path, map_location="cpu")
        assert payload["model"].keys() == student.state_dict().keys()
        assert not any("teacher" in key.lower() for key in payload["model"])


def run_all() -> None:
    checks = (
        test_cbsd_config_is_strictly_isolated,
        test_student_and_teacher_topologies_are_exact,
        test_schedule_matches_fixed_protocol,
        test_existing_formal_c40_teacher_variant_remains_allowed,
        test_teacher_strict_load_freeze_and_checkpoint_rejection,
        test_mask_only_cbsd_backward_optimizer_ema_and_checkpoint_isolation,
    )
    for check in checks:
        check()
        print(f"PASS: {check.__name__}")


if __name__ == "__main__":
    run_all()
