"""Focused regression coverage for opt-in Auto Training Protocol V1."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import random
import sys
import tempfile

import torch
import yaml
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from losses import DEMambaLoss
from train import train_one_epoch
from utils.checkpoint import load_checkpoint_metadata, save_checkpoint
from utils.auto_protocol import (
    AutoProtocolCheckpointState,
    UpdateEarlyStopping,
    align_native_size,
    accumulation_steps,
    capture_rng_state,
    is_new_best_checkpoint,
    model_checksum,
    resolve_auto_protocol,
    restore_rng_state,
    select_auto_protocol_final_checkpoint,
)


EXAMPLE = ROOT / "configs" / "experiments" / "auto_protocol" / "C40_DPSM80_TUT_AUTO_PROTOCOL_V1.yaml"


def _load_example() -> dict:
    with EXAMPLE.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _auto_cfg(root: Path) -> dict:
    return {
        "data": {"data_root": str(root), "img_size": None},
        "train": {"epochs": 9, "batch_size": 8},
        "auto_protocol": {
            "enabled": True,
            "native_input": {"enabled": True, "align_multiple": 32, "max_pixels": 409600, "size_statistic": "mode"},
            "batch": {"micro_batch": "auto", "effective_batch_size": 8, "candidates": [8, 4, 2, 1]},
            "updates": {"target_updates": "auto", "min_target_updates": 9200, "max_target_updates": 18800, "updates_per_train_image": 8.0},
            "loss_phases": {"use_update_clock": True, "scnp_warmup_ratio": 0.05, "bpcm_start_ratio": 0.30, "bpcm_ramp_ratio": 0.20},
            "validation": {"interval_updates": 400},
            "early_stopping": {"enabled": True, "min_training_ratio": 0.55, "patience_ratio": 0.15, "min_delta": 0.0005},
        },
    }


def _write_image(path: Path, size: tuple[int, int]) -> None:
    Image.new("RGB", (size[1], size[0]), color=(20, 30, 40)).save(path)


class _ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"mask": image[:, :1] * self.weight}


class _ToyCriterion(torch.nn.Module):
    def forward(self, outputs, batch, **_kwargs):
        loss = ((outputs["mask"] - batch["mask"]) ** 2).mean()
        return loss, {}


class _Counter:
    def __init__(self) -> None:
        self.calls = 0

    def step(self) -> None:
        self.calls += 1

    def update(self, *_args) -> None:
        self.calls += 1


def test_disabled_protocol_keeps_config_identical() -> None:
    cfg = {"data": {"data_root": "unused", "img_size": [384, 544]}, "train": {"epochs": 200, "batch_size": 8}}
    before = deepcopy(cfg)
    assert resolve_auto_protocol(cfg) is None
    assert cfg == before


def test_tut_plan_and_native_mode_use_train_images_only() -> None:
    cfg = _load_example()
    plan = resolve_auto_protocol(cfg)
    assert plan is not None
    assert plan.train_samples == 987
    assert plan.selected_img_size == (640, 640)
    assert cfg["data"]["img_size"] == [640, 640]
    assert plan.optimizer_steps_per_epoch == 123
    assert plan.target_updates == 9200
    assert plan.resolved_epochs == 75
    assert plan.scnp_warmup_updates == 460
    assert plan.bpcm_start_update == 2760
    assert plan.bpcm_ramp_updates == 1840
    assert plan.validation_interval_updates == 369


def test_rng_capture_restore_is_exact_for_python_numpy_and_torch() -> None:
    random.seed(18)
    np.random.seed(18)
    torch.manual_seed(18)
    state = capture_rng_state()
    expected = (random.random(), float(np.random.rand()), torch.rand(3))
    restore_rng_state(state)
    actual = (random.random(), float(np.random.rand()), torch.rand(3))
    assert expected[0] == actual[0]
    assert expected[1] == actual[1]
    assert torch.equal(expected[2], actual[2])


def test_probe_rng_isolation_preserves_formal_model_and_loader_order() -> None:
    """Independent probe activity must not affect formal initialization/order."""

    torch.manual_seed(42)
    baseline_model = torch.nn.Linear(3, 2)
    baseline_checksum = model_checksum(baseline_model)
    baseline_generator = torch.Generator().manual_seed(42)
    baseline_order = torch.randperm(16, generator=baseline_generator)

    torch.manual_seed(42)
    saved = capture_rng_state()
    _probe_model = torch.nn.Linear(3, 2)
    _probe_generator = torch.Generator().manual_seed(104729)
    _ = torch.rand(5)
    _ = torch.randperm(16, generator=_probe_generator)
    restore_rng_state(saved)
    formal_model = torch.nn.Linear(3, 2)
    formal_generator = torch.Generator().manual_seed(42)
    formal_order = torch.randperm(16, generator=formal_generator)
    assert model_checksum(formal_model) == baseline_checksum
    assert torch.equal(formal_order, baseline_order)


def test_cuda_rng_capture_restore_is_exact_when_available() -> None:
    if not torch.cuda.is_available():
        return
    torch.cuda.manual_seed_all(29)
    saved = capture_rng_state()
    expected = torch.rand(4, device="cuda")
    restore_rng_state(saved)
    actual = torch.rand(4, device="cuda")
    assert torch.equal(expected, actual)


def test_native_mode_alignment_pixel_cap_and_split_isolation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "train" / "images").mkdir(parents=True)
        (root / "val" / "images").mkdir(parents=True)
        (root / "test" / "images").mkdir(parents=True)
        for index in range(8):
            _write_image(root / "train" / "images" / f"train_{index}.png", (641, 639))
        _write_image(root / "val" / "images" / "ignored.png", (2048, 2048))
        _write_image(root / "test" / "images" / "ignored.png", (32, 32))
        cfg = _auto_cfg(root)
        plan = resolve_auto_protocol(cfg)
        assert plan is not None
        assert plan.train_samples == 8
        assert plan.native_size_distribution == {"641x639": 8}
        assert plan.selected_img_size == (640, 608)
        explicit_cfg = _auto_cfg(root)
        explicit_cfg["data"]["img_size"] = [320, 480]
        explicit_plan = resolve_auto_protocol(explicit_cfg)
        assert explicit_plan is not None and explicit_plan.selected_img_size == (320, 480)
        capped = align_native_size(1024, 2048, align_multiple=32, max_pixels=409600)
        assert capped[0] * capped[1] <= 409600
        assert capped[0] % 32 == 0 and capped[1] % 32 == 0


def test_update_clock_scales_use_optimizer_updates_not_epochs() -> None:
    loss = DEMambaLoss(
        scnp_weight=0.2,
        lambda_bpcm=0.025,
        use_update_clock=True,
        scnp_warmup_updates=10,
        bpcm_start_update=30,
        bpcm_ramp_updates=20,
    )
    assert loss._scnp_scale(epoch=999, global_update=0) == 0.0
    assert loss._scnp_scale(epoch=0, global_update=5) == 0.1
    assert loss._scnp_scale(epoch=0, global_update=10) == 0.2
    assert loss._bpcm_scale(epoch=999, global_update=29) == 0.0
    assert loss._bpcm_scale(epoch=0, global_update=40) == 0.0125
    assert loss._bpcm_scale(epoch=0, global_update=50) == 0.025


def test_validation_interval_resolves_for_small_and_large_datasets() -> None:
    for dataset, batch, expected in (("CFD", 2, 123), ("CrackMap", 2, 126), ("Crack500", 8, 400)):
        cfg = _load_example()
        cfg["data"]["data_root"] = str(ROOT / dataset)
        cfg["data"]["img_size"] = [320, 480]
        cfg["auto_protocol"]["batch"]["effective_batch_size"] = batch
        plan = resolve_auto_protocol(cfg)
        assert plan is not None and plan.validation_interval_updates == expected, dataset


def test_accumulation_steps_scheduler_ema_and_target_truncation() -> None:
    torch.manual_seed(7)
    model = _ToyModel()
    initial_weight = model.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    scheduler, ema = _Counter(), _Counter()
    one_micro_batch = [{"image": torch.ones(4, 1, 1, 1), "mask": torch.zeros(4, 1, 1, 1)}]
    _, partial_stats = train_one_epoch(
        model,
        one_micro_batch,
        _ToyCriterion(),
        optimizer,
        scaler,
        torch.device("cpu"),
        False,
        torch.float16,
        epoch=0,
        model_ema=ema,
        scheduler=scheduler,
        accumulation=accumulation_steps(8, 4),
        global_update=0,
        target_updates=2,
        use_update_clock=True,
    )
    assert partial_stats["optimizer_updates"] == 0
    assert scheduler.calls == 0 and ema.calls == 0
    assert torch.equal(model.weight.detach(), initial_weight)
    loader = [
        {"image": torch.ones(4, 1, 1, 1), "mask": torch.zeros(4, 1, 1, 1)}
        for _ in range(6)
    ]
    _, stats = train_one_epoch(
        model,
        loader,
        _ToyCriterion(),
        optimizer,
        scaler,
        torch.device("cpu"),
        False,
        torch.float16,
        epoch=0,
        model_ema=ema,
        scheduler=scheduler,
        accumulation=accumulation_steps(8, 4),
        global_update=0,
        target_updates=2,
        use_update_clock=True,
    )
    assert stats["global_update"] == 2
    assert stats["optimizer_updates"] == 2
    assert stats["reached_target"]
    assert scheduler.calls == 2
    assert ema.calls == 2


def test_dry_run_update_cap_respects_accumulation_boundary() -> None:
    model = _ToyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    scheduler, ema = _Counter(), _Counter()
    loader = [
        {"image": torch.ones(4, 1, 1, 1), "mask": torch.zeros(4, 1, 1, 1)},
        {"image": torch.ones(4, 1, 1, 1), "mask": torch.zeros(4, 1, 1, 1)},
        {"image": torch.ones(4, 1, 1, 1), "mask": torch.zeros(4, 1, 1, 1)},
    ]
    _, stats = train_one_epoch(
        model,
        loader,
        _ToyCriterion(),
        optimizer,
        scaler,
        torch.device("cpu"),
        False,
        torch.float16,
        epoch=0,
        model_ema=ema,
        scheduler=scheduler,
        accumulation=2,
        global_update=0,
        target_updates=1,
        use_update_clock=True,
    )
    assert stats["batches"] == 2
    assert stats["optimizer_updates"] == 1
    assert stats["global_update"] == 1
    assert scheduler.calls == 1 and ema.calls == 1


def test_early_stopping_uses_updates_and_respects_minimum() -> None:
    early = UpdateEarlyStopping(min_update=550, patience_updates=150, min_delta=0.0005)
    assert early.update(0.80, 400)
    assert not early.should_stop(549)
    assert early.should_stop(550)
    assert early.update(0.801, 700)
    assert not early.should_stop(849)
    assert not early.update(0.8012, 750)  # below min_delta; patience is not reset
    assert early.last_significant_update == 700
    assert is_new_best_checkpoint(0.8012, 0.801)
    assert not is_new_best_checkpoint(0.8012, 0.8012)


def test_significant_checkpoint_selector_keeps_micro_improvements_diagnostic_only() -> None:
    selector = AutoProtocolCheckpointState(min_delta=0.0005)
    observations = [
        (0.8255, 100, True),
        (0.8261, 200, True),
        (0.8263, 300, False),
        (0.826508, 400, False),
    ]
    for score, update, expect_significant in observations:
        absolute, significant, _ = selector.consider(score, update)
        assert absolute
        assert significant is expect_significant
    assert selector.absolute_best_score == 0.826508
    assert selector.absolute_best_update == 400
    assert selector.significant_best_score == 0.8261
    assert selector.significant_best_update == 200
    assert selector.last_significant_update == 200


def test_significant_checkpoint_state_round_trips_for_resume() -> None:
    original = AutoProtocolCheckpointState(min_delta=0.0005)
    original.consider(0.8255, 100)
    original.consider(0.8261, 200)
    original.consider(0.826508, 400)
    restored = AutoProtocolCheckpointState(min_delta=0.0005)
    restored.load_state_dict(original.state_dict())
    assert restored.state_dict() == original.state_dict()


def test_checkpoint_metadata_persists_auto_protocol_resume_state() -> None:
    selector = AutoProtocolCheckpointState(min_delta=0.0005)
    selector.consider(0.8255, 100)
    state = selector.state_dict() | {"global_update": 123, "early_stopping": {"patience_updates": 50}}
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "latest.pth"
        save_checkpoint(
            path,
            model,
            optimizer,
            scheduler=None,
            epoch=3,
            best_score=selector.significant_best_score,
            checkpoint_type="latest",
            auto_protocol_state=state,
        )
        metadata = load_checkpoint_metadata(path)
    assert metadata["checkpoint_type"] == "latest"
    assert metadata["auto_protocol_state"] == state


def test_final_checkpoint_fallback_is_explicit_and_never_uses_latest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        significant = root / "best_structural_score.pth"
        absolute = root / "best_absolute_score.pth"
        latest = root / "latest.pth"
        latest.touch()
        absolute.touch()
        selected, selection_type = select_auto_protocol_final_checkpoint(significant, absolute)
        assert selected == absolute
        assert selection_type == "absolute_best_fallback"
        absolute.unlink()
        try:
            select_auto_protocol_final_checkpoint(significant, absolute)
        except FileNotFoundError as error:
            assert "latest.pth" in str(error)
        else:  # pragma: no cover - keeps the intended failure mode explicit
            raise AssertionError("latest.pth must never become a silent final checkpoint fallback")


def test() -> None:
    checks = [
        test_disabled_protocol_keeps_config_identical,
        test_tut_plan_and_native_mode_use_train_images_only,
        test_rng_capture_restore_is_exact_for_python_numpy_and_torch,
        test_probe_rng_isolation_preserves_formal_model_and_loader_order,
        test_cuda_rng_capture_restore_is_exact_when_available,
        test_native_mode_alignment_pixel_cap_and_split_isolation,
        test_update_clock_scales_use_optimizer_updates_not_epochs,
        test_validation_interval_resolves_for_small_and_large_datasets,
        test_accumulation_steps_scheduler_ema_and_target_truncation,
        test_dry_run_update_cap_respects_accumulation_boundary,
        test_early_stopping_uses_updates_and_respects_minimum,
        test_significant_checkpoint_selector_keeps_micro_improvements_diagnostic_only,
        test_significant_checkpoint_state_round_trips_for_resume,
        test_checkpoint_metadata_persists_auto_protocol_resume_state,
        test_final_checkpoint_fallback_is_explicit_and_never_uses_latest,
    ]
    for check in checks:
        check()
        print(f"PASS: {check.__name__}")


if __name__ == "__main__":
    test()
