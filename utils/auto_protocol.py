"""Opt-in resolution helpers for Auto Training Protocol V1.

The module deliberately performs only train-split metadata inspection.  It
does not construct validation/test datasets or inspect their images.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
from PIL import Image
import torch


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _as_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"auto_protocol.{name} must be a mapping.")
    return value


def auto_protocol_enabled(cfg: dict[str, Any]) -> bool:
    protocol = cfg.get("auto_protocol")
    return isinstance(protocol, dict) and bool(protocol.get("enabled", False))


def scan_train_image_sizes(data_root: str | Path) -> Counter[tuple[int, int]]:
    """Return only the training-image HxW distribution.

    Reading this directory rather than instantiating a complete dataset is
    intentional: targets, validation images, and test images cannot influence
    any auto-protocol decision.
    """

    image_dir = Path(data_root) / "train" / "images"
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing training image directory: {image_dir}")
    distribution: Counter[tuple[int, int]] = Counter()
    for path in sorted(image_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        with Image.open(path) as image:
            width, height = image.size
        distribution[(height, width)] += 1
    if not distribution:
        raise RuntimeError(f"No training images found under {image_dir}")
    return distribution


def _mode_size(distribution: Counter[tuple[int, int]]) -> tuple[int, int]:
    # A deterministic tie break keeps runs reproducible on filesystems with
    # different directory iteration order.
    return min(distribution.items(), key=lambda item: (-item[1], item[0]))[0]


def align_native_size(
    height: int,
    width: int,
    *,
    align_multiple: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Scale (if needed) and align a native size without exceeding max_pixels."""

    if align_multiple <= 0:
        raise ValueError("auto_protocol.native_input.align_multiple must be positive.")
    if max_pixels <= 0:
        raise ValueError("auto_protocol.native_input.max_pixels must be positive.")
    if height <= 0 or width <= 0:
        raise ValueError("Training image dimensions must be positive.")

    scale = min(1.0, math.sqrt(max_pixels / float(height * width)))
    scaled_h, scaled_w = height * scale, width * scale
    # Floor alignment makes the pixel-limit guarantee exact.  A minimum one
    # multiple protects very small diagnostic images.
    aligned_h = max(align_multiple, int(scaled_h // align_multiple) * align_multiple)
    aligned_w = max(align_multiple, int(scaled_w // align_multiple) * align_multiple)
    while aligned_h * aligned_w > max_pixels:
        if aligned_h >= aligned_w and aligned_h > align_multiple:
            aligned_h -= align_multiple
        elif aligned_w > align_multiple:
            aligned_w -= align_multiple
        else:  # pragma: no cover - guarded by the positive checks above
            raise ValueError("max_pixels is smaller than one aligned input tile.")
    return aligned_h, aligned_w


def calculate_target_updates(
    train_samples: int,
    updates_cfg: dict[str, Any],
) -> int:
    requested = updates_cfg.get("target_updates", "auto")
    if requested != "auto":
        target = int(requested)
        if target <= 0:
            raise ValueError("auto_protocol.updates.target_updates must be positive or 'auto'.")
        return target
    updates_per_image = float(updates_cfg.get("updates_per_train_image", 8.0))
    minimum = int(updates_cfg.get("min_target_updates", 9200))
    maximum = int(updates_cfg.get("max_target_updates", 18800))
    if updates_per_image <= 0 or minimum <= 0 or maximum < minimum:
        raise ValueError("Invalid auto_protocol.updates bounds.")
    return int(round(min(max(updates_per_image * train_samples, minimum), maximum)))


def _image_and_mask_stems(data_root: str | Path) -> tuple[dict[str, Path], dict[str, Path]]:
    root = Path(data_root)
    image_dir, mask_dir = root / "train" / "images", root / "train" / "masks"
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"Auto Protocol V2 requires train/images and train/masks under {root}.")
    images = {path.stem: path for path in sorted(image_dir.iterdir()) if path.suffix.lower() in IMAGE_EXTENSIONS}
    masks = {path.stem: path for path in sorted(mask_dir.iterdir()) if path.suffix.lower() in IMAGE_EXTENSIONS}
    return images, masks


def audit_train_foreground(
    data_root: str | Path,
    threshold: int,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Audit train masks only, using the dataset's grayscale/threshold rule."""

    images, masks = _image_and_mask_stems(data_root)
    image_stems, mask_stems = set(images), set(masks)
    missing_masks = sorted(image_stems - mask_stems)
    orphan_masks = sorted(mask_stems - image_stems)
    matched = sorted(image_stems & mask_stems)
    fractions: list[float] = []
    positive_pixels = total_pixels = empty_masks = unreadable_masks = 0
    fingerprint = hashlib.sha256()
    for stem in matched:
        image_path, mask_path = images[stem], masks[stem]
        try:
            with Image.open(mask_path) as image:
                raw_mask = np.asarray(image.convert("L"), dtype=np.uint8)
        except (OSError, ValueError) as error:
            unreadable_masks += 1
            if bool(settings.get("fail_on_unreadable_mask", True)):
                raise RuntimeError(f"Unreadable training mask: {mask_path}") from error
            continue
        positive = int((raw_mask > threshold).sum())
        pixels = int(raw_mask.size)
        positive_pixels += positive
        total_pixels += pixels
        fraction = positive / max(pixels, 1)
        fractions.append(fraction)
        empty_masks += int(positive == 0)
        for path in (image_path, mask_path):
            stat = path.stat()
            fingerprint.update(f"{path.name}|{stat.st_size}|{stat.st_mtime_ns}|".encode("utf-8"))
    stem_mismatches = len(missing_masks) + len(orphan_masks)
    if stem_mismatches and bool(settings.get("fail_on_stem_mismatch", True)):
        raise RuntimeError(
            f"Training image/mask stem mismatch: missing_masks={len(missing_masks)}, orphan_masks={len(orphan_masks)}."
        )
    if not fractions and bool(settings.get("fail_on_empty_dataset", True)):
        raise RuntimeError("No readable matched training masks were found for foreground audit.")
    foreground_fraction = positive_pixels / max(total_pixels, 1)
    if foreground_fraction <= 0.0:
        raise RuntimeError("Training foreground fraction is zero; inverse-prevalence weighting is undefined.")
    maximum_fraction = float(settings.get("fail_if_foreground_fraction_above", 0.5))
    if foreground_fraction > maximum_fraction:
        raise RuntimeError(
            f"Training foreground fraction {foreground_fraction:.6f} exceeds {maximum_fraction:.6f}; possible mask polarity reversal."
        )
    negatives = total_pixels - positive_pixels
    return {
        "train_mask_count": len(matched),
        "positive_pixels": positive_pixels,
        "total_pixels": total_pixels,
        "foreground_fraction": foreground_fraction,
        "negative_positive_ratio": negatives / max(positive_pixels, 1),
        "per_image_foreground_fraction_mean": float(np.mean(fractions)),
        "per_image_foreground_fraction_median": float(np.median(fractions)),
        "per_image_foreground_fraction_min": float(np.min(fractions)),
        "per_image_foreground_fraction_max": float(np.max(fractions)),
        "empty_mask_count": empty_masks,
        "unreadable_mask_count": unreadable_masks,
        "stem_mismatch_count": stem_mismatches,
        "dataset_fingerprint": fingerprint.hexdigest(),
    }


def _v2_regime(protocol: dict[str, Any], train_samples: int) -> tuple[dict[str, Any], str]:
    regimes = protocol.get("regimes")
    if not isinstance(regimes, list) or not regimes:
        raise ValueError("Auto Protocol V2 requires a non-empty auto_protocol.regimes list.")
    previous_max = -1
    selected: dict[str, Any] | None = None
    names: set[str] = set()
    for index, item in enumerate(regimes):
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError("Every Auto Protocol V2 regime needs a string name.")
        name, maximum = item["name"], item.get("max_train_samples")
        if name in names:
            raise ValueError(f"Duplicate Auto Protocol V2 regime name: {name}.")
        names.add(name)
        if maximum is None:
            if index != len(regimes) - 1:
                raise ValueError("Only the final Auto Protocol V2 regime may use max_train_samples: null.")
            if selected is None:
                selected = item
            continue
        maximum = int(maximum)
        if maximum <= previous_max:
            raise ValueError("Auto Protocol V2 regime max_train_samples values must be strictly increasing.")
        previous_max = maximum
        if selected is None and train_samples <= maximum:
            selected = item
    if regimes[-1].get("max_train_samples") is not None:
        raise ValueError("The final Auto Protocol V2 regime must be an explicit null fallback.")
    override = protocol.get("regime_override")
    if override is not None:
        selected = next((item for item in regimes if item.get("name") == override), None)
        if selected is None:
            raise ValueError(f"Unknown Auto Protocol V2 regime_override: {override}.")
        return selected, "yaml.regime_override"
    assert selected is not None
    return selected, "train_samples"


def divisors_descending(value: int) -> list[int]:
    if value <= 0:
        raise ValueError("effective_batch_size must be positive.")
    return [candidate for candidate in range(value, 0, -1) if value % candidate == 0]


def auto_save_dir(data_root: str | Path) -> str:
    label = "".join(character if character.isalnum() or character in "-_" else "_" for character in Path(data_root).name)
    if not label:
        label = "dataset"
    return f"./runs/pilot/auto_protocol_v2/C40_DPSM80_{label}_AUTO_PROTOCOL_V2"


def accumulation_steps(effective_batch_size: int, micro_batch: int) -> int:
    if effective_batch_size <= 0 or micro_batch <= 0:
        raise ValueError("effective_batch_size and micro_batch must be positive.")
    if effective_batch_size % micro_batch:
        raise ValueError("effective_batch_size must be divisible by micro_batch.")
    return effective_batch_size // micro_batch


@dataclass(frozen=True)
class AutoProtocolPlan:
    train_samples: int
    native_size_distribution: dict[str, int]
    selected_img_size: tuple[int, int]
    effective_batch_size: int
    optimizer_steps_per_epoch: int
    target_updates: int
    resolved_epochs: int
    scnp_warmup_updates: int
    bpcm_start_update: int
    bpcm_ramp_updates: int
    validation_interval_updates: int
    min_early_stop_update: int
    patience_updates: int
    protocol_version: int = 1
    selected_regime: str | None = None
    planned_sample_exposures: int | None = None
    approximate_dataset_passes: float | None = None
    foreground_audit: dict[str, Any] | None = None
    overrides: list[str] | None = None

    def as_dict(self) -> dict[str, Any]:
        resolved = {
            "train_samples": self.train_samples,
            "native_size_distribution": self.native_size_distribution,
            "selected_img_size": list(self.selected_img_size),
            "effective_batch_size": self.effective_batch_size,
            "optimizer_steps_per_epoch": self.optimizer_steps_per_epoch,
            "target_updates": self.target_updates,
            "resolved_epochs": self.resolved_epochs,
            "scnp_warmup_updates": self.scnp_warmup_updates,
            "bpcm_start_update": self.bpcm_start_update,
            "bpcm_ramp_updates": self.bpcm_ramp_updates,
            "configured_validation_interval_updates": None,
            "resolved_validation_interval_updates": self.validation_interval_updates,
            "approximate_validation_interval_epochs": None,
            "min_early_stop_update": self.min_early_stop_update,
            "patience_updates": self.patience_updates,
            "stop_reason": None,
            "best_checkpoint_score": None,
            "best_checkpoint_update": None,
            "best_checkpoint_epoch": None,
            "absolute_best_score": None,
            "absolute_best_update": None,
            "absolute_best_checkpoint": None,
            "significant_best_checkpoint": None,
            "significant_best_score": None,
            "last_significant_improvement_update": None,
            "final_checkpoint_type": None,
            "final_checkpoint_path": None,
            "final_checkpoint_score": None,
            "final_checkpoint_update": None,
            "final_global_update": 0,
            "dry_run": False,
            "dry_run_optimizer_updates": 0,
            "validation_calls": 0,
            "test_calls": 0,
        }
        if self.protocol_version == 2:
            resolved.update(
                protocol_version=2,
                selected_regime=self.selected_regime,
                planned_sample_exposures=self.planned_sample_exposures,
                approximate_dataset_passes=self.approximate_dataset_passes,
                foreground_audit=self.foreground_audit,
                overrides=self.overrides or [],
            )
        return resolved


def resolve_auto_protocol(
    cfg: dict[str, Any],
    *,
    cli_img_size: bool = False,
    cli_batch_size: bool = False,
    cli_epochs: bool = False,
    cli_data_root: bool = False,
    cli_save_dir: bool = False,
) -> AutoProtocolPlan | None:
    """Resolve opt-in V1 metadata and mutate only the derived train fields.

    ``data.img_size: null`` is the unambiguous opt-in for native-size selection.
    A non-null YAML value and the explicit CLI ``--img_size`` both take
    precedence over automatic native-size selection.
    """

    if not auto_protocol_enabled(cfg):
        return None
    protocol = _as_mapping(cfg["auto_protocol"], "")
    version = int(protocol.get("version", 1))
    if version == 2:
        return resolve_auto_protocol_v2(
            cfg,
            cli_img_size=cli_img_size,
            cli_batch_size=cli_batch_size,
            cli_epochs=cli_epochs,
            cli_data_root=cli_data_root,
            cli_save_dir=cli_save_dir,
        )
    if version != 1:
        raise ValueError(f"Unsupported Auto Protocol version: {version}.")
    native_cfg = _as_mapping(protocol.get("native_input", {}), "native_input")
    batch_cfg = _as_mapping(protocol.get("batch", {}), "batch")
    updates_cfg = _as_mapping(protocol.get("updates", {}), "updates")
    phases_cfg = _as_mapping(protocol.get("loss_phases", {}), "loss_phases")
    validation_cfg = _as_mapping(protocol.get("validation", {}), "validation")
    early_cfg = _as_mapping(protocol.get("early_stopping", {}), "early_stopping")

    configured_selection = str(early_cfg.get("selection_metric", cfg.get("evaluation", {}).get("selection_metric", "structural_score")))
    evaluation_selection = str(cfg.get("evaluation", {}).get("selection_metric", "structural_score"))
    if configured_selection != evaluation_selection:
        raise ValueError("auto_protocol.early_stopping.selection_metric must match evaluation.selection_metric.")

    data_cfg, train_cfg = cfg["data"], cfg["train"]
    distribution = scan_train_image_sizes(data_cfg["data_root"])
    native_mode = _mode_size(distribution)
    selected_size = tuple(int(value) for value in data_cfg["img_size"]) if data_cfg.get("img_size") is not None else None
    if bool(native_cfg.get("enabled", False)) and selected_size is None and not cli_img_size:
        statistic = str(native_cfg.get("size_statistic", "mode")).lower()
        if statistic != "mode":
            raise ValueError("Auto Protocol V1 supports only native_input.size_statistic='mode'.")
        selected_size = align_native_size(
            *native_mode,
            align_multiple=int(native_cfg.get("align_multiple", 32)),
            max_pixels=int(native_cfg.get("max_pixels", 409600)),
        )
        data_cfg["img_size"] = list(selected_size)
    elif selected_size is None:
        raise ValueError("auto_protocol requires data.img_size or native_input.enabled=true.")
    else:
        selected_size = tuple(selected_size)

    effective_batch = int(batch_cfg.get("effective_batch_size", 8))
    if effective_batch <= 0:
        raise ValueError("auto_protocol.batch.effective_batch_size must be positive.")
    steps_per_epoch = sum(distribution.values()) // effective_batch
    if steps_per_epoch <= 0:
        raise ValueError("effective_batch_size exceeds the number of training samples.")
    target_updates = calculate_target_updates(sum(distribution.values()), updates_cfg)
    if cli_epochs:
        # CLI has the highest precedence.  Its epoch request therefore becomes
        # the update budget rather than silently being overwritten by auto.
        target_updates = int(train_cfg["epochs"]) * steps_per_epoch
    resolved_epochs = math.ceil(target_updates / steps_per_epoch)
    if not cli_epochs:
        train_cfg["epochs"] = resolved_epochs

    use_update_clock = bool(phases_cfg.get("use_update_clock", True))
    scnp_updates = int(round(float(phases_cfg.get("scnp_warmup_ratio", 0.05)) * target_updates))
    bpcm_start = int(round(float(phases_cfg.get("bpcm_start_ratio", 0.30)) * target_updates))
    bpcm_ramp = int(round(float(phases_cfg.get("bpcm_ramp_ratio", 0.20)) * target_updates))
    if use_update_clock:
        protocol["loss_phases_resolved"] = {
            "use_update_clock": True,
            "scnp_warmup_updates": scnp_updates,
            "bpcm_start_update": bpcm_start,
            "bpcm_ramp_updates": bpcm_ramp,
        }

    configured_interval = int(validation_cfg.get("interval_updates", 400))
    interval = min(configured_interval, max(1, 3 * steps_per_epoch))
    if interval <= 0:
        raise ValueError("auto_protocol.validation.interval_updates must be positive.")
    min_early = int(math.ceil(float(early_cfg.get("min_training_ratio", 0.55)) * target_updates))
    patience = int(math.ceil(float(early_cfg.get("patience_ratio", 0.15)) * target_updates))
    if min_early < 0 or patience <= 0:
        raise ValueError("Invalid auto_protocol.early_stopping ratios.")

    # A CLI micro-batch is an explicit user instruction and is retained.  The
    # actual auto probe is performed by train.py after device selection.
    if cli_batch_size:
        requested_micro = int(train_cfg["batch_size"])
        accumulation_steps(effective_batch, requested_micro)
        protocol["micro_batch_request"] = requested_micro
    else:
        protocol["micro_batch_request"] = batch_cfg.get("micro_batch", "auto")

    plan = AutoProtocolPlan(
        train_samples=sum(distribution.values()),
        native_size_distribution={f"{height}x{width}": count for (height, width), count in sorted(distribution.items())},
        selected_img_size=selected_size,
        effective_batch_size=effective_batch,
        optimizer_steps_per_epoch=steps_per_epoch,
        target_updates=target_updates,
        resolved_epochs=resolved_epochs,
        scnp_warmup_updates=scnp_updates,
        bpcm_start_update=bpcm_start,
        bpcm_ramp_updates=bpcm_ramp,
        validation_interval_updates=interval,
        min_early_stop_update=min_early,
        patience_updates=patience,
    )
    cfg["auto_protocol_resolved"] = plan.as_dict()
    cfg["auto_protocol_resolved"]["configured_validation_interval_updates"] = configured_interval
    cfg["auto_protocol_resolved"]["approximate_validation_interval_epochs"] = interval / steps_per_epoch
    return plan


def resolve_auto_protocol_v2(
    cfg: dict[str, Any],
    *,
    cli_img_size: bool,
    cli_batch_size: bool,
    cli_epochs: bool,
    cli_data_root: bool,
    cli_save_dir: bool,
) -> AutoProtocolPlan:
    """Resolve the train-split-only, regime-driven Auto Protocol V2."""

    protocol = _as_mapping(cfg["auto_protocol"], "")
    data_cfg, train_cfg, loss_cfg = cfg["data"], cfg["train"], cfg["loss"]
    native_cfg = _as_mapping(protocol.get("native_input", {}), "native_input")
    batch_cfg = _as_mapping(protocol.get("batch", {}), "batch")
    phases_cfg = _as_mapping(protocol.get("loss_phases", {}), "loss_phases")
    validation_cfg = _as_mapping(protocol.get("validation", {}), "validation")
    early_cfg = _as_mapping(protocol.get("early_stopping", {}), "early_stopping")
    balance_cfg = _as_mapping(protocol.get("foreground_balance", {}), "foreground_balance")
    configured_selection = str(early_cfg.get("selection_metric", cfg.get("evaluation", {}).get("selection_metric", "structural_score")))
    if configured_selection != str(cfg.get("evaluation", {}).get("selection_metric", "structural_score")):
        raise ValueError("auto_protocol.early_stopping.selection_metric must match evaluation.selection_metric.")

    distribution = scan_train_image_sizes(data_cfg["data_root"])
    train_samples = sum(distribution.values())
    native_mode = _mode_size(distribution)
    overrides: list[str] = []
    if cli_data_root:
        overrides.append("cli.data_root")
    selected_size = tuple(int(value) for value in data_cfg["img_size"]) if data_cfg.get("img_size") is not None else None
    if bool(native_cfg.get("enabled", False)) and selected_size is None and not cli_img_size:
        if str(native_cfg.get("size_statistic", "mode")).lower() != "mode":
            raise ValueError("Auto Protocol V2 supports only native_input.size_statistic='mode'.")
        selected_size = align_native_size(
            *native_mode,
            align_multiple=int(native_cfg.get("align_multiple", 32)),
            max_pixels=int(native_cfg.get("max_pixels", 409600)),
        )
        data_cfg["img_size"] = list(selected_size)
    elif selected_size is None:
        raise ValueError("Auto Protocol V2 requires data.img_size or native_input.enabled=true.")
    else:
        if cli_img_size:
            overrides.append("cli.img_size")
        else:
            overrides.append("yaml.data.img_size")

    regime, regime_source = _v2_regime(protocol, train_samples)
    if regime_source != "train_samples":
        overrides.append(regime_source)
    effective_batch = int(regime["effective_batch_size"])
    if protocol.get("effective_batch_size_override") is not None:
        effective_batch = int(protocol["effective_batch_size_override"])
        overrides.append("yaml.effective_batch_size_override")
    elif batch_cfg.get("effective_batch_size") is not None:
        effective_batch = int(batch_cfg["effective_batch_size"])
        overrides.append("yaml.batch.effective_batch_size")
    if cli_batch_size:
        effective_batch = int(train_cfg["batch_size"])
        overrides.append("cli.batch_size")
    if effective_batch <= 0:
        raise ValueError("Auto Protocol V2 effective_batch_size must be positive.")
    steps_per_epoch = train_samples // effective_batch
    if steps_per_epoch <= 0:
        raise ValueError("Auto Protocol V2 effective_batch_size exceeds the number of training samples.")
    target_epochs = int(regime["target_epochs"])
    if protocol.get("target_epochs_override") is not None:
        target_epochs = int(protocol["target_epochs_override"])
        overrides.append("yaml.target_epochs_override")
    if cli_epochs:
        target_epochs = int(train_cfg["epochs"])
        overrides.append("cli.epochs")
    if target_epochs <= 0:
        raise ValueError("Auto Protocol V2 target_epochs must be positive.")
    target_updates = steps_per_epoch * target_epochs
    train_cfg["epochs"] = target_epochs

    audit = audit_train_foreground(data_cfg["data_root"], int(data_cfg["threshold"]), balance_cfg)
    weight_mode = str(regime.get("mask_pos_weight_mode", "fixed"))
    if weight_mode == "inverse_prevalence":
        raw_weight = float(audit["negative_positive_ratio"])
        minimum, maximum = float(balance_cfg.get("min_weight", 1.0)), float(balance_cfg.get("max_weight", 50.0))
        if minimum <= 0 or maximum < minimum:
            raise ValueError("Auto Protocol V2 foreground_balance bounds are invalid.")
        resolved_weight = min(max(raw_weight, minimum), maximum)
        weight_source = "inverse_prevalence"
        audit.update(raw_mask_pos_weight=raw_weight, resolved_mask_pos_weight=resolved_weight, mask_pos_weight_clipped=resolved_weight != raw_weight)
    elif weight_mode == "fixed":
        resolved_weight = float(regime.get("mask_pos_weight", 1.0))
        if resolved_weight <= 0:
            raise ValueError("Auto Protocol V2 fixed mask_pos_weight must be positive.")
        weight_source = "fixed"
        audit.update(raw_mask_pos_weight=resolved_weight, resolved_mask_pos_weight=resolved_weight, mask_pos_weight_clipped=False)
    else:
        raise ValueError(f"Unsupported Auto Protocol V2 mask_pos_weight_mode: {weight_mode}.")
    loss_cfg["mask_pos_weight"] = float(resolved_weight)
    audit["mask_pos_weight_source"] = weight_source

    if bool(phases_cfg.get("use_update_clock", True)):
        protocol["loss_phases_resolved"] = {
            "use_update_clock": True,
            "scnp_warmup_updates": int(round(float(phases_cfg.get("scnp_warmup_ratio", 0.05)) * target_updates)),
            "bpcm_start_update": int(round(float(phases_cfg.get("bpcm_start_ratio", 0.30)) * target_updates)),
            "bpcm_ramp_updates": int(round(float(phases_cfg.get("bpcm_ramp_ratio", 0.20)) * target_updates)),
        }
    phases = protocol["loss_phases_resolved"]
    configured_interval = int(validation_cfg.get("interval_updates", 400))
    interval = min(configured_interval, max(1, 3 * steps_per_epoch))
    min_early = int(math.ceil(float(early_cfg.get("min_training_ratio", 0.55)) * target_updates))
    patience = int(math.ceil(float(early_cfg.get("patience_ratio", 0.15)) * target_updates))
    if configured_interval <= 0 or min_early < 0 or patience <= 0:
        raise ValueError("Auto Protocol V2 validation or early-stopping settings are invalid.")

    batch_cfg["candidates"] = divisors_descending(effective_batch)
    batch_cfg["effective_batch_size"] = effective_batch
    batch_cfg["micro_batch"] = batch_cfg.get("micro_batch", "auto")
    protocol["micro_batch_request"] = effective_batch if cli_batch_size else batch_cfg["micro_batch"]
    if cfg.get("save", {}).get("save_dir") == "auto":
        cfg["save"]["save_dir"] = auto_save_dir(data_cfg["data_root"])
    elif cli_save_dir:
        overrides.append("cli.save_dir")

    planned_exposures = steps_per_epoch * effective_batch * target_epochs
    plan = AutoProtocolPlan(
        train_samples=train_samples,
        native_size_distribution={f"{height}x{width}": count for (height, width), count in sorted(distribution.items())},
        selected_img_size=selected_size,
        effective_batch_size=effective_batch,
        optimizer_steps_per_epoch=steps_per_epoch,
        target_updates=target_updates,
        resolved_epochs=target_epochs,
        scnp_warmup_updates=int(phases["scnp_warmup_updates"]),
        bpcm_start_update=int(phases["bpcm_start_update"]),
        bpcm_ramp_updates=int(phases["bpcm_ramp_updates"]),
        validation_interval_updates=interval,
        min_early_stop_update=min_early,
        patience_updates=patience,
        protocol_version=2,
        selected_regime=str(regime["name"]),
        planned_sample_exposures=planned_exposures,
        approximate_dataset_passes=planned_exposures / train_samples,
        foreground_audit=audit,
        overrides=overrides,
    )
    resolved = plan.as_dict()
    resolved.update(
        configured_validation_interval_updates=configured_interval,
        approximate_validation_interval_epochs=interval / steps_per_epoch,
        selected_regime_condition={"max_train_samples": regime.get("max_train_samples"), "source": regime_source},
        target_epochs=target_epochs,
        minimum_early_stop_sample_exposures=min_early * effective_batch,
        patience_sample_exposures=patience * effective_batch,
        resolved_mask_pos_weight=float(resolved_weight),
        mask_pos_weight_source=weight_source,
        resume_identity={
            "protocol_version": 2,
            "selected_regime": str(regime["name"]),
            "dataset_fingerprint": audit["dataset_fingerprint"],
            "train_samples": train_samples,
            "selected_img_size": list(selected_size),
            "foreground_audit": audit,
            "mask_pos_weight": float(resolved_weight),
            "effective_batch_size": effective_batch,
            "target_epochs": target_epochs,
            "target_updates": target_updates,
            "loss_phases": phases,
            "validation_interval_updates": interval,
            "min_early_stop_update": min_early,
            "patience_updates": patience,
        },
    )
    cfg["auto_protocol_resolved"] = resolved
    return plan


@dataclass
class UpdateEarlyStopping:
    min_update: int
    patience_updates: int
    min_delta: float
    best_significant_score: float = float("-inf")
    last_significant_update: int = 0

    def update(self, score: float, global_update: int) -> bool:
        if score >= self.best_significant_score + self.min_delta:
            self.best_significant_score = score
            self.last_significant_update = global_update
            return True
        return False

    def should_stop(self, global_update: int) -> bool:
        return global_update >= self.min_update and (global_update - self.last_significant_update) >= self.patience_updates


def is_new_best_checkpoint(score: float, best_checkpoint_score: float) -> bool:
    """Return true only for a strict structural-score improvement.

    Keeping this decision as a small pure helper documents the distinction
    between a checkpoint best and an early-stopping significant best. In
    particular, ties intentionally do not rewrite best_structural_score.pth.
    """

    return score > best_checkpoint_score


def select_auto_protocol_final_checkpoint(
    significant_checkpoint: str | Path,
    absolute_checkpoint: str | Path,
) -> tuple[Path, str]:
    """Choose deployable significant-best, with an explicit-only fallback."""

    significant = Path(significant_checkpoint)
    absolute = Path(absolute_checkpoint)
    if significant.exists():
        return significant, "significant_best"
    if absolute.exists():
        return absolute, "absolute_best_fallback"
    raise FileNotFoundError(
        "Auto Protocol has neither a significant-best nor an absolute-best checkpoint; latest.pth is not a valid final-selection fallback."
    )


def validate_auto_protocol_resume(
    resolved_cfg: dict[str, Any],
    checkpoint_state: dict[str, Any],
    *,
    unsafe: bool = False,
) -> None:
    """Reject V2 resumes whose train-only protocol identity changed."""

    expected = checkpoint_state.get("resume_identity")
    current = resolved_cfg.get("resume_identity")
    if expected is None or current is None:
        return
    if expected != current and not unsafe:
        raise ValueError(
            "Auto Protocol V2 resume identity differs from the checkpoint; train split, regime, batch, budget, or foreground audit changed. "
            "Use --auto-protocol-unsafe-resume only when intentionally accepting a non-reproducible resume."
        )


@dataclass
class AutoProtocolCheckpointState:
    """Separate diagnostic absolute-best from deployable significant-best."""

    min_delta: float
    absolute_best_score: float = float("-inf")
    absolute_best_update: int | None = None
    significant_best_score: float = float("-inf")
    significant_best_update: int | None = None
    last_significant_update: int = 0

    def consider(self, score: float, global_update: int) -> tuple[bool, bool, float]:
        delta_from_significant = score - self.significant_best_score
        absolute_updated = is_new_best_checkpoint(score, self.absolute_best_score)
        if absolute_updated:
            self.absolute_best_score = score
            self.absolute_best_update = global_update
        significant_updated = score >= self.significant_best_score + self.min_delta
        if significant_updated:
            self.significant_best_score = score
            self.significant_best_update = global_update
            self.last_significant_update = global_update
        return absolute_updated, significant_updated, delta_from_significant

    def state_dict(self) -> dict[str, Any]:
        return {
            "absolute_best_score": self.absolute_best_score,
            "absolute_best_update": self.absolute_best_update,
            "significant_best_score": self.significant_best_score,
            "significant_best_update": self.significant_best_update,
            "last_significant_update": self.last_significant_update,
            "min_delta": self.min_delta,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.absolute_best_score = float(state.get("absolute_best_score", self.absolute_best_score))
        absolute_update = state.get("absolute_best_update")
        self.absolute_best_update = None if absolute_update is None else int(absolute_update)
        self.significant_best_score = float(state.get("significant_best_score", self.significant_best_score))
        significant_update = state.get("significant_best_update")
        self.significant_best_update = None if significant_update is None else int(significant_update)
        self.last_significant_update = int(state.get("last_significant_update", self.last_significant_update))


def update_resolved_run_state(
    cfg: dict[str, Any],
    *,
    stop_reason: str | None = None,
    best_update: int | None = None,
    best_epoch: int | None = None,
    micro_batch: int | None = None,
    accumulation: int | None = None,
    peak_allocated_bytes: int | None = None,
    peak_reserved_bytes: int | None = None,
) -> None:
    resolved = cfg.get("auto_protocol_resolved")
    if not isinstance(resolved, dict):
        return
    if stop_reason is not None:
        resolved["stop_reason"] = stop_reason
    if best_update is not None:
        resolved["best_checkpoint_update"] = best_update
    if best_epoch is not None:
        resolved["best_checkpoint_epoch"] = best_epoch
    if micro_batch is not None:
        resolved["micro_batch"] = micro_batch
    if accumulation is not None:
        resolved["accumulation_steps"] = accumulation
    if peak_allocated_bytes is not None:
        resolved["peak_allocated_memory_bytes"] = peak_allocated_bytes
    if peak_reserved_bytes is not None:
        resolved["peak_reserved_memory_bytes"] = peak_reserved_bytes


def clone_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Small public seam for probe code and tests without mutating cfg."""

    return deepcopy(cfg)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def model_checksum(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
