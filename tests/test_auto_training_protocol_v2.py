"""Regression coverage for the opt-in, regime-driven Auto Training Protocol V2."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.auto_protocol import (
    _v2_regime,
    audit_train_foreground,
    divisors_descending,
    resolve_auto_protocol,
    validate_auto_protocol_resume,
)


V1_CONFIG = ROOT / "configs" / "experiments" / "auto_protocol" / "C40_DPSM80_TUT_AUTO_PROTOCOL_V1.yaml"
V2_CONFIG = ROOT / "configs" / "experiments" / "auto_protocol" / "C40_DPSM80_AUTO_PROTOCOL_V2.yaml"


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_pair(root: Path, stem: str, mask: np.ndarray) -> None:
    image_dir, mask_dir = root / "train" / "images", root / "train" / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (mask.shape[1], mask.shape[0]), color=(30, 40, 50)).save(image_dir / f"{stem}.png")
    Image.fromarray(mask.astype(np.uint8)).save(mask_dir / f"{stem}.png")


def _v2_cfg(root: Path) -> dict:
    cfg = _load(V2_CONFIG)
    cfg["data"]["data_root"] = str(root)
    return cfg


def test_regime_boundaries_and_six_reference_sizes() -> None:
    protocol = _load(V2_CONFIG)["auto_protocol"]
    expected = {
        83: ("tiny", 2, 200, 8200),
        84: ("tiny", 2, 200, 8400),
        368: ("small", 8, 200, 9200),
        553: ("medium", 4, 100, 13800),
        987: ("large", 8, 75, 9225),
        2357: ("xlarge", 8, 64, 18816),
    }
    for samples, (name, batch, epochs, updates) in expected.items():
        regime, _ = _v2_regime(protocol, samples)
        assert regime["name"] == name
        assert regime["effective_batch_size"] == batch
        assert regime["target_epochs"] == epochs
        assert samples // batch * epochs == updates
    for samples, expected_name in ((127, "tiny"), (128, "small"), (511, "small"), (512, "medium"), (767, "medium"), (768, "large"), (1535, "large"), (1536, "xlarge")):
        assert _v2_regime(protocol, samples)[0]["name"] == expected_name


def test_foreground_audit_inverse_prevalence_and_limits() -> None:
    settings = _load(V2_CONFIG)["auto_protocol"]["foreground_balance"]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sparse = np.zeros((10, 10), dtype=np.uint8)
        sparse[:2, :2] = 255
        _write_pair(root, "a", sparse)
        audit = audit_train_foreground(root, 127, settings)
        assert audit["positive_pixels"] == 4
        assert audit["total_pixels"] == 100
        assert audit["foreground_fraction"] == 0.04
        assert audit["negative_positive_ratio"] == 24.0
        assert audit["dataset_fingerprint"]

        all_background = Path(tmp) / "background"
        _write_pair(all_background, "a", np.zeros((8, 8), dtype=np.uint8))
        try:
            audit_train_foreground(all_background, 127, settings)
        except RuntimeError as error:
            assert "zero" in str(error)
        else:
            raise AssertionError("all-background masks must fail foreground audit")

        inverted = Path(tmp) / "inverted"
        _write_pair(inverted, "a", np.full((8, 8), 255, dtype=np.uint8))
        try:
            audit_train_foreground(inverted, 127, settings)
        except RuntimeError as error:
            assert "polarity" in str(error)
        else:
            raise AssertionError("suspected inverted masks must fail foreground audit")


def test_foreground_audit_rejects_corruption_and_stem_mismatch() -> None:
    settings = _load(V2_CONFIG)["auto_protocol"]["foreground_balance"]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_pair(root, "ok", np.array([[0, 255], [0, 0]], dtype=np.uint8))
        (root / "train" / "masks" / "bad.png").write_bytes(b"not an image")
        Image.new("RGB", (2, 2)).save(root / "train" / "images" / "bad.png")
        try:
            audit_train_foreground(root, 127, settings)
        except RuntimeError as error:
            assert "Unreadable" in str(error)
        else:
            raise AssertionError("corrupt masks must fail foreground audit")

        mismatch = Path(tmp) / "mismatch"
        _write_pair(mismatch, "ok", np.array([[0, 255], [0, 0]], dtype=np.uint8))
        Image.new("RGB", (2, 2)).save(mismatch / "train" / "images" / "image_only.png")
        try:
            audit_train_foreground(mismatch, 127, settings)
        except RuntimeError as error:
            assert "stem mismatch" in str(error)
        else:
            raise AssertionError("stem mismatches must fail foreground audit")


def test_v2_resolution_overrides_batch_and_writes_float_weight() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for index in range(4):
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[index, index] = 255
            _write_pair(root, str(index), mask)
        cfg = _v2_cfg(root)
        plan = resolve_auto_protocol(cfg)
        assert plan.protocol_version == 2
        assert plan.selected_regime == "tiny"
        assert plan.effective_batch_size == 2
        assert plan.optimizer_steps_per_epoch == 2
        assert plan.target_updates == 400
        assert cfg["auto_protocol"]["batch"]["candidates"] == [2, 1]
        assert isinstance(cfg["loss"]["mask_pos_weight"], float)
        assert cfg["loss"]["mask_pos_weight"] == 50.0  # inverse-prevalence upper clip
        assert cfg["save"]["save_dir"].endswith("AUTO_PROTOCOL_V2")

        balanced_root = root / "balanced"
        balanced = np.zeros((8, 8), dtype=np.uint8)
        balanced[:, :4] = 255
        _write_pair(balanced_root, "a", balanced)
        _write_pair(balanced_root, "b", balanced)
        balanced_cfg = _v2_cfg(balanced_root)
        resolve_auto_protocol(balanced_cfg)
        assert balanced_cfg["loss"]["mask_pos_weight"] == 1.0  # lower clip at exactly 50% foreground

        override = _v2_cfg(root)
        override["auto_protocol"]["effective_batch_size_override"] = 1
        override["auto_protocol"]["target_epochs_override"] = 3
        overridden = resolve_auto_protocol(override)
        assert overridden.effective_batch_size == 1
        assert overridden.resolved_epochs == 3
        assert "yaml.effective_batch_size_override" in override["auto_protocol_resolved"]["overrides"]

        cli = _v2_cfg(root)
        cli["train"]["batch_size"] = 1
        cli["train"]["epochs"] = 7
        cli_plan = resolve_auto_protocol(cli, cli_batch_size=True, cli_epochs=True)
        assert cli_plan.effective_batch_size == 1 and cli_plan.resolved_epochs == 7
        assert "cli.batch_size" in cli["auto_protocol_resolved"]["overrides"]
        assert "cli.epochs" in cli["auto_protocol_resolved"]["overrides"]


def test_micro_batch_divisors_v1_compatibility_and_resume_identity() -> None:
    assert divisors_descending(2) == [2, 1]
    assert divisors_descending(4) == [4, 2, 1]
    assert divisors_descending(8) == [8, 4, 2, 1]
    v1 = _load(V1_CONFIG)
    before = deepcopy(v1)
    plan = resolve_auto_protocol(v1)
    assert plan is not None and plan.protocol_version == 1 and plan.target_updates == 9200
    assert "protocol_version" not in v1["auto_protocol_resolved"]
    assert "version" not in before["auto_protocol"]

    current = {"resume_identity": {"dataset_fingerprint": "same", "effective_batch_size": 2}}
    validate_auto_protocol_resume(current, {"resume_identity": dict(current["resume_identity"])})
    try:
        validate_auto_protocol_resume(current, {"resume_identity": {"dataset_fingerprint": "changed", "effective_batch_size": 2}})
    except ValueError as error:
        assert "identity differs" in str(error)
    else:
        raise AssertionError("V2 resume identity mismatch must fail without an unsafe override")
    validate_auto_protocol_resume(
        current,
        {"resume_identity": {"dataset_fingerprint": "changed", "effective_batch_size": 2}},
        unsafe=True,
    )


def test() -> None:
    checks = [
        test_regime_boundaries_and_six_reference_sizes,
        test_foreground_audit_inverse_prevalence_and_limits,
        test_foreground_audit_rejects_corruption_and_stem_mismatch,
        test_v2_resolution_overrides_batch_and_writes_float_weight,
        test_micro_batch_divisors_v1_compatibility_and_resume_identity,
    ]
    for check in checks:
        check()
        print(f"PASS: {check.__name__}")


if __name__ == "__main__":
    test()
