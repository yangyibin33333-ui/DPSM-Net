"""Equal-recall control: is cell B's +1.8pp recall just a threshold shift?

Cell A (the reported M5 protocol) and cell B (A plus a structural-recall term)
reach the same pooled F1 from opposite directions: on Crack500 test at the paper
threshold 0.5, B trades about 1.7pp of precision for about 1.8pp of recall.  That
trade is the signature of a decision threshold moving, not of a better segmenter.
This script tests that reading without training anything:

1. take B's pooled recall at the paper threshold 0.5 as the target recall R*;
2. sweep A's decision threshold and take the t* whose pooled recall is closest to
   R* -- A's "equal-recall" operating point;
3. report A's complete metric set at t* next to B's at 0.5.

If A at t* reproduces B's F1 / boundary F1 / clDice / component error, the gain is
a pure operating-point shift and must not be claimed as a model contribution.  If
A at t* keeps B's recall but loses F1, the recall term changed the ranking of
pixels -- a model-level effect.

本脚本只构造"等召回工作点"作为对照，不用于任何模型/阈值选择。

t* is derived from the test split, so every number printed here is a diagnostic
control and never a headline result: no model, checkpoint, seed, or reported
threshold may be selected with it.  The paper's operating point stays 0.5.

Zero training.  Reads only ``config.yaml``, ``best_structural_score.pth`` and
``train.log`` inside the seed directories plus the image/target split; writes only
under ``--output-root`` (never to ``paper/`` or ``configs/``).

Dependencies
------------
The forward/loader/model plumbing (``parse_run``, ``threshold_grid``, ``load_model``,
``make_loader``, ``collect``) is a *minimal local copy* of the same helpers in
``scripts/diagnose_accuracy_headroom.py``, so this file can run without that one.
The metric implementation is never copied: every number comes from
``evaluate_orientation_strata.per_image_metrics`` (``metrics/metrics.py``
semantics), which therefore has to exist in the repository.

Anchor check
------------
The complete metric set is recomputed through
``evaluate_orientation_strata.per_image_metrics`` and pooled exactly the way
``metrics.metrics.SegmentationMetric.compute`` pools it: tp/fp/fn summed over the
split for F1/IoU/Precision/Recall, boundary counts summed with the 2px tolerance
for BoundaryF1, and clDice / component error averaged per image.  The final test
block that ``train.py`` wrote into ``train.log`` is re-read and diffed key by key,
so a metric path that disagrees with the training pipeline surfaces as an explicit
number; any |delta| above ``ANCHOR_TOLERANCE`` is reported as a WARNING and means
the recomputed metrics are not comparable with the run log.

Examples
--------
    # the acceptance run: one A seed directory against one B seed directory
    python scripts/diagnose_matched_recall_control.py \\
        --reference A=runs/paper/dependency_aware_ablation_strict_v2/crack500/M5_FrameworkOn_DPSMOn_StructSupOn/20260830-211010 \\
        --matched B=runs/experiments/loss_2x2/Crack500/B_SKELRECALL/<stamp> \\
        --split test --mask-only --num-workers 8

    # all three seeds of each cell, paired by training_seed
    python scripts/diagnose_matched_recall_control.py \\
        --reference A=runs/paper/dependency_aware_ablation_strict_v2/crack500/M5_FrameworkOn_DPSMOn_StructSupOn \\
        --matched B=runs/experiments/loss_2x2/Crack500/B_SKELRECALL \\
        --split test --mask-only --num-workers 8
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from datasets.crack_dataset import build_dataloader  # noqa: E402
try:  # pragma: no cover - only fails when the repository is incomplete
    from evaluate_orientation_strata import (  # noqa: E402
        discover_seed_dirs,
        per_image_metrics,
        read_config,
    )
except ImportError as error:  # pragma: no cover
    raise SystemExit(
        f"error: cannot import scripts/evaluate_orientation_strata.py ({error}). "
        "It provides the reference per_image_metrics that every number in this report "
        "is computed with, plus read_config/discover_seed_dirs; sync that file into "
        "the repository before running this diagnostic."
    ) from error
from models import build_model  # noqa: E402
from utils.checkpoint import load_checkpoint  # noqa: E402

FIXED_THRESHOLD = 0.5

#: the script must never write here (paper text and experiment configs are inputs)
FORBIDDEN_OUTPUT_ROOTS = (ROOT / "paper", ROOT / "configs")

#: |recomputed - logged| above this means the metric path differs from the one
#: that produced the run log; the acceptance protocol allows 2e-4.
ANCHOR_TOLERANCE = 2e-4

LOG_METRIC_KEYS = ("F1", "Precision", "Recall", "BoundaryF1", "clDice", "ComponentError")
LOG_PATTERN = re.compile(
    r"\|\s*INFO\s*\|\s*(F1|Precision|Recall|BoundaryF1|clDice|ComponentError):\s*([0-9]+(?:\.[0-9]+)?)"
)

DELTA_KEYS = ("F1", "BoundaryF1", "clDice", "ComponentError", "Precision", "Recall", "IoU")


# --------------------------------------------------------------------------- #
# plumbing: minimal local copy of scripts/diagnose_accuracy_headroom.py helpers
# --------------------------------------------------------------------------- #
# These five helpers are duplicated instead of imported so the script has no
# dependency on the sibling diagnostic file; the metric implementation itself is
# still imported (evaluate_orientation_strata.per_image_metrics) and never copied.
def parse_run(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError(f"--reference/--matched expects LABEL=DIR, got {value!r}")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"run directory does not exist: {path}")
    return label.strip(), path


def threshold_grid(spec: str) -> list[float]:
    values: list[float] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            parts = [piece for piece in chunk.split(":") if piece != ""]
            if len(parts) not in (2, 3):
                raise argparse.ArgumentTypeError(f"bad threshold range {chunk!r}")
            start, stop = float(parts[0]), float(parts[1])
            step = float(parts[2]) if len(parts) == 3 else 0.05
            count = int(round((stop - start) / step))
            values.extend(round(start + index * step, 4) for index in range(count + 1))
        else:
            values.append(float(chunk))
    grid = sorted({value for value in values if 0.0 < value < 1.0})
    if not grid:
        raise argparse.ArgumentTypeError("empty threshold grid")
    return grid


def load_model(seed_dir: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Rebuild the trained model and load the checkpoint the paper evaluates."""
    config = read_config(seed_dir / "config.yaml")
    model_config = dict(config["model"])
    model_name = str(model_config.pop("name", "DEMambaNet"))
    model = build_model(model_name, **model_config).to(device)
    load_checkpoint(seed_dir / "best_structural_score.pth", model, map_location=device)
    model.eval()
    return model, config


def make_loader(config: dict, split: str, *, batch_size: int, num_workers: int):
    """Dataloader for the split at exactly the resolution the run was evaluated on."""
    data_config = dict(config.get("data", {}))
    root = Path(data_config.get("data_root", "./Crack500"))
    if not root.is_absolute():
        root = (ROOT / root).resolve()
    img_size = data_config.get("img_size")
    img_size = tuple(int(value) for value in img_size) if img_size else None
    loader = build_dataloader(
        root, split, img_size, int(data_config.get("threshold", 127)),
        batch_size, num_workers, shuffle=False,
        target_dir_name=str(data_config.get("target_dir_name", "targets")),
    )
    return loader, root, img_size


def collect(
    model: torch.nn.Module,
    loader,
    *,
    device: torch.device,
    mask_only: bool,
    max_images: int | None,
) -> dict[str, object]:
    """Forward one split once and keep per-image logits plus targets on the host."""
    stems: list[str] = []
    logits_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            outputs = model.forward_mask_only(images) if mask_only else model(images)
            logits = outputs["mask"].detach().float().cpu()
            for index, path in enumerate(batch["image_path"]):
                stems.append(Path(path).stem)
                logits_chunks.append(logits[index:index + 1].half())
                target_chunks.append(batch["mask"][index:index + 1].detach().float().cpu().half())
            if max_images is not None and len(stems) >= max_images:
                stems = stems[:max_images]
                logits_chunks = logits_chunks[:max_images]
                target_chunks = target_chunks[:max_images]
                break
    return {
        "stems": stems,
        "logits": torch.cat(logits_chunks, dim=0) if logits_chunks else torch.empty(0),
        "target": torch.cat(target_chunks, dim=0) if target_chunks else torch.empty(0),
    }


# --------------------------------------------------------------------------- #
# metrics, pooled exactly like metrics.metrics.SegmentationMetric
# --------------------------------------------------------------------------- #
def pooled_metrics(logits: torch.Tensor, target: torch.Tensor, threshold: float) -> dict[str, float]:
    """Complete metric set for one split at one threshold.

    Pooling follows ``SegmentationMetric.compute`` so the numbers are directly
    comparable with a run's ``train.log``: ratios from summed counts, clDice and
    component error averaged per image.
    """
    counts = {key: 0.0 for key in
              ("tp", "fp", "fn", "boundary_matched", "boundary_prediction", "boundary_target")}
    cldice_values: list[float] = []
    component_errors: list[float] = []
    for index in range(logits.shape[0]):
        metrics = per_image_metrics(logits[index:index + 1], target[index:index + 1],
                                    threshold=threshold)
        for key in counts:
            counts[key] += float(metrics[key])
        cldice_values.append(float(metrics["clDice"]))
        component_errors.append(float(metrics["ComponentError"]))
    eps = 1e-7
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {
        "threshold": float(threshold),
        "F1": 2.0 * tp / (2.0 * tp + fp + fn + eps),
        "IoU": tp / (tp + fp + fn + eps),
        "Precision": tp / (tp + fp + eps),
        "Recall": tp / (tp + fn + eps),
        "BoundaryF1": counts["boundary_matched"]
        / (counts["boundary_prediction"] + counts["boundary_target"] + eps),
        "clDice": statistics.fmean(cldice_values) if cldice_values else 0.0,
        "ComponentError": statistics.fmean(component_errors) if component_errors else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "boundary_matched": counts["boundary_matched"],
        "boundary_prediction": counts["boundary_prediction"],
        "boundary_target": counts["boundary_target"],
        "n_images": int(logits.shape[0]),
        "n_positive_pixels": tp + fn,
    }


def recall_sweep(logits: torch.Tensor, target: torch.Tensor,
                 thresholds: list[float]) -> list[dict[str, float]]:
    """Pooled recall for every threshold, computed on the positive pixels only.

    ``recall(t) = #{positive pixels with p > t} / #{positive pixels}``, so the
    sweep never has to threshold the full split: the same strict ``>`` comparison
    as ``per_image_metrics`` is applied to the positive subset alone.
    """
    probability = torch.sigmoid(logits.detach().float().cpu())
    truth = target.detach().float().cpu() > 0.5
    positive = probability[truth]
    total = float(positive.numel())
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        if total <= 0.0:
            rows.append({"threshold": float(threshold), "recall": 0.0, "tp": 0.0,
                         "n_positive_pixels": 0.0})
            continue
        tp = float((positive > threshold).sum())
        rows.append({"threshold": float(threshold), "recall": tp / total, "tp": tp,
                     "n_positive_pixels": total})
    return rows


def select_matched_threshold(rows: list[dict[str, float]], target_recall: float) -> dict[str, float]:
    """The grid point whose pooled recall is closest to ``target_recall``.

    Ties are broken towards the paper threshold, then towards the lower
    threshold, so the choice is deterministic and does not depend on grid order.
    """
    def rank(row: dict[str, float]) -> tuple[float, float, float]:
        return (round(abs(row["recall"] - target_recall), 12),
                round(abs(row["threshold"] - FIXED_THRESHOLD), 12),
                row["threshold"])

    return min(rows, key=rank)


# --------------------------------------------------------------------------- #
# run log anchor
# --------------------------------------------------------------------------- #
def read_logged_metrics(seed_dir: Path) -> dict[str, float]:
    """Final test block of ``train.log`` (last value wins per key)."""
    log_path = seed_dir / "train.log"
    if not log_path.is_file():
        return {}
    logged: dict[str, float] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = LOG_PATTERN.search(line)
        if match is not None:
            logged[match.group(1)] = float(match.group(2))
    return logged


def anchor_entry(computed: dict[str, float], logged: dict[str, float]) -> dict[str, object]:
    deltas: dict[str, float | None] = {}
    for key in LOG_METRIC_KEYS:
        if key in logged:
            deltas[key] = computed[key] - logged[key]
        else:
            deltas[key] = None
    available = [abs(value) for value in deltas.values() if value is not None]
    worst = max(available) if available else None
    return {
        "logged": {key: logged.get(key) for key in LOG_METRIC_KEYS},
        "delta": deltas,
        "max_abs_delta": worst,
        "within_tolerance": None if worst is None else worst <= ANCHOR_TOLERANCE,
    }


# --------------------------------------------------------------------------- #
# per-seed evaluation
# --------------------------------------------------------------------------- #
def evaluate_seed_pair(
    reference_seed: int,
    reference_dir: Path,
    matched_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    """B at 0.5 defines R*; A is probed at 0.5 and at its matched-recall t*.

    B is forwarded first and its logits released before A is loaded, so peak host
    memory holds one model's logits rather than two.
    """
    # ---- matched cell (B): its 0.5 operating point is the target recall ----
    matched_model, matched_config = load_model(matched_dir, device)
    matched_loader, matched_root, _ = make_loader(
        matched_config, args.split, batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    matched = collect(matched_model, matched_loader, device=device,
                      mask_only=args.mask_only, max_images=args.max_images)
    matched_at_half = pooled_metrics(matched["logits"], matched["target"], FIXED_THRESHOLD)
    target_recall = matched_at_half["Recall"]
    matched_stems = list(matched["stems"])
    del matched_model, matched, matched_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- reference cell (A): anchor, sweep, matched operating point ----
    reference_model, reference_config = load_model(reference_dir, device)
    reference_loader, reference_root, reference_img_size = make_loader(
        reference_config, args.split, batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    reference = collect(reference_model, reference_loader, device=device,
                        mask_only=args.mask_only, max_images=args.max_images)

    reference_at_half = pooled_metrics(reference["logits"], reference["target"], FIXED_THRESHOLD)
    sweep = recall_sweep(reference["logits"], reference["target"], args.grid)
    # Reuse the grid row for 0.5 when it exists; only pay for a second pass otherwise.
    fast_at_half = next(
        (row for row in sweep if abs(row["threshold"] - FIXED_THRESHOLD) < 1e-9), None
    )
    if fast_at_half is None:
        fast_at_half = recall_sweep(reference["logits"], reference["target"], [FIXED_THRESHOLD])[0]
    matched_row = select_matched_threshold(sweep, target_recall)
    reference_at_matched = pooled_metrics(reference["logits"], reference["target"],
                                          matched_row["threshold"])

    stems_agree = list(reference["stems"]) == matched_stems
    anchor = anchor_entry(reference_at_half, read_logged_metrics(reference_dir))

    deltas = {key: reference_at_matched[key] - matched_at_half[key] for key in DELTA_KEYS}

    entry: dict[str, object] = {
        "seed": reference_seed,
        "reference_dir": str(reference_dir),
        "matched_dir": str(matched_dir),
        "data_root": str(reference_root),
        "img_size": list(reference_img_size) if reference_img_size else None,
        "n_images": int(reference_at_half["n_images"]),
        "stems_agree": stems_agree,
        "anchor": {
            "threshold": FIXED_THRESHOLD,
            "reference_at_0.5": reference_at_half,
            "fast_vs_reference_recall_delta": fast_at_half["recall"] - reference_at_half["Recall"],
            **anchor,
            "tolerance": ANCHOR_TOLERANCE,
        },
        "matched_point": {
            "target_recall": target_recall,
            "target_recall_source": f"{matched_dir.name} at threshold {FIXED_THRESHOLD}",
            "selected_threshold": matched_row["threshold"],
            "swept_recall": matched_row["recall"],
            "recall_deviation": matched_row["recall"] - target_recall,
            "within_tolerance": abs(matched_row["recall"] - target_recall) <= args.tolerance,
            "tolerance": args.tolerance,
            "grid_points": len(sweep),
            "grid_first": sweep[0]["threshold"],
            "grid_last": sweep[-1]["threshold"],
            "sweep": sweep,
        },
        "reference_at_matched": reference_at_matched,
        "matched_at_0.5": matched_at_half,
        "deltas": deltas,
        "matched_data_root": str(matched_root),
    }
    del reference_model, reference, reference_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return entry


def summarise(values: list[float]) -> dict[str, object]:
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "positive_in": int(sum(1 for value in values if value > 0.0)),
        "negative_in": int(sum(1 for value in values if value < 0.0)),
        "n_seeds": len(values),
        "values": values,
    }


def aggregate_seeds(per_seed: dict[str, dict[str, object]]) -> dict[str, object]:
    entries = list(per_seed.values())
    aggregate: dict[str, object] = {
        "n_seeds": len(entries),
        "seeds": [int(entry["seed"]) for entry in entries],
        "matched_thresholds": [entry["matched_point"]["selected_threshold"] for entry in entries],
        "recall_deviations": [entry["matched_point"]["recall_deviation"] for entry in entries],
        "deltas": {},
    }
    for key in DELTA_KEYS:
        values = [float(entry["deltas"][key]) for entry in entries]
        aggregate["deltas"][key] = summarise(values)
    anchors = [entry["anchor"]["max_abs_delta"] for entry in entries]
    available = [value for value in anchors if value is not None]
    aggregate["anchor"] = {
        "max_abs_delta": max(available) if available else None,
        "within_tolerance": None if not available else max(available) <= ANCHOR_TOLERANCE,
        "tolerance": ANCHOR_TOLERANCE,
        "logs_found_in": sum(1 for value in anchors if value is not None),
    }
    return aggregate


# --------------------------------------------------------------------------- #
# markdown
# --------------------------------------------------------------------------- #
def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_delta(value: object, digits: int = 4) -> str:
    if value is None:
        return "--"
    return f"{float(value):+.{digits}f}"


def render_markdown(payload: dict[str, object]) -> list[str]:
    metadata = payload["metadata"]
    per_seed = payload["pairs"]
    aggregate = payload["aggregate"]
    lines = [
        "# Equal-recall control (matched operating point)",
        "",
        f"reference: `{metadata['reference']['label']}` = `{metadata['reference']['dir']}`  ",
        f"matched: `{metadata['matched']['label']}` = `{metadata['matched']['dir']}`",
        "",
        f"split `{metadata['split']}`, fixed paper threshold "
        f"`{metadata['fixed_threshold']}`, threshold grid {metadata['grid'][0]:.2f}.."
        f"{metadata['grid'][-1]:.2f} ({len(metadata['grid'])} points), match tolerance "
        f"{metadata['tolerance']}, mask-only forward: {metadata['mask_only']}, "
        f"max images: {metadata['max_images']}.",
        "",
        "B is read at the paper threshold; A is re-read at the threshold whose pooled "
        "recall matches B's. `dX` in every table is A(matched) minus B(0.5), so a "
        "negative value means B is better.",
        "",
        f"> 本脚本只构造\"等召回工作点\"作为对照，不用于任何模型/阈值选择。 t* comes from the "
        f"test split; nothing in this file may be used to select a model, checkpoint, "
        f"seed or reported threshold.",
        "",
        "## 1. Anchor: A at 0.5 must reproduce the run log",
        "",
        "| seed | metric | computed | logged | delta |",
        "|---:|---|---:|---:|---:|",
    ]
    for seed, entry in per_seed.items():
        anchor = entry["anchor"]
        for key in LOG_METRIC_KEYS:
            lines.append(
                f"| {seed} | {key} | {_fmt(anchor['reference_at_0.5'][key])} "
                f"| {_fmt(anchor['logged'][key])} | {_fmt_delta(anchor['delta'][key], 5)} |"
            )
    lines += [
        "",
        "| seed | n images | max abs delta | within 2e-4 | fast-vs-reference recall |",
        "|---:|---:|---:|---|---:|",
    ]
    for seed, entry in per_seed.items():
        anchor = entry["anchor"]
        status = "--" if anchor["within_tolerance"] is None else (
            "yes" if anchor["within_tolerance"] else "NO -> metric path differs")
        lines.append(
            f"| {seed} | {entry['n_images']} | {_fmt(anchor['max_abs_delta'], 6)} | {status} "
            f"| {_fmt_delta(anchor['fast_vs_reference_recall_delta'], 8)} |"
        )

    lines += [
        "",
        "## 2. Matched-recall comparison",
        "",
        "### 2a. Operating points",
        "",
        "| seed | B recall @0.5 = R* | A threshold t* | A recall @t* | recall deviation "
        "| within tolerance | grid points |",
        "|---:|---:|---:|---:|---:|---|---:|",
    ]
    for seed, entry in per_seed.items():
        point = entry["matched_point"]
        lines.append(
            f"| {seed} | {_fmt(point['target_recall'])} | {_fmt(point['selected_threshold'], 2)} "
            f"| {_fmt(point['swept_recall'])} | {_fmt_delta(point['recall_deviation'], 5)} "
            f"| {'yes' if point['within_tolerance'] else 'NO (WARNING)'} | {point['grid_points']} |"
        )

    lines += [
        "",
        "### 2b. Complete metrics at the matched recall",
        "",
        "| seed | F1 A@t* | F1 B@0.5 | dF1 | bF1 A | bF1 B | dbF1 | clDice A | clDice B "
        "| dclDice | CE A | CE B | dCE |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed, entry in per_seed.items():
        reference = entry["reference_at_matched"]
        matched = entry["matched_at_0.5"]
        delta = entry["deltas"]
        lines.append(
            f"| {seed} | {_fmt(reference['F1'])} | {_fmt(matched['F1'])} | {_fmt_delta(delta['F1'])} "
            f"| {_fmt(reference['BoundaryF1'])} | {_fmt(matched['BoundaryF1'])} "
            f"| {_fmt_delta(delta['BoundaryF1'])} "
            f"| {_fmt(reference['clDice'])} | {_fmt(matched['clDice'])} "
            f"| {_fmt_delta(delta['clDice'])} "
            f"| {_fmt(reference['ComponentError'])} | {_fmt(matched['ComponentError'])} "
            f"| {_fmt_delta(delta['ComponentError'])} |"
        )

    lines += [
        "",
        "| seed | Precision A | Precision B | dPrecision | Recall A | Recall B | dRecall "
        "| IoU A | IoU B | dIoU | stems agree |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed, entry in per_seed.items():
        reference = entry["reference_at_matched"]
        matched = entry["matched_at_0.5"]
        delta = entry["deltas"]
        lines.append(
            f"| {seed} | {_fmt(reference['Precision'])} | {_fmt(matched['Precision'])} "
            f"| {_fmt_delta(delta['Precision'])} "
            f"| {_fmt(reference['Recall'])} | {_fmt(matched['Recall'])} "
            f"| {_fmt_delta(delta['Recall'])} "
            f"| {_fmt(reference['IoU'])} | {_fmt(matched['IoU'])} | {_fmt_delta(delta['IoU'])} "
            f"| {'yes' if entry['stems_agree'] else 'NO'} |"
        )

    lines += [
        "",
        "## 3. Aggregate across seeds (A at matched recall minus B at 0.5)",
        "",
        "| quantity | delta mean | delta sd | seeds with delta > 0 | per-seed |",
        "|---|---:|---:|---|---|",
    ]
    for key in ("F1", "BoundaryF1", "clDice", "ComponentError", "Precision", "Recall", "IoU"):
        entry = aggregate["deltas"][key]
        per_seed_text = ", ".join(_fmt_delta(value) for value in entry["values"])
        scale = "" if key == "ComponentError" else " (pp)"
        lines.append(
            f"| d{key}{scale} | {_fmt_delta(entry['mean'])} | {_fmt(entry['sd'])} "
            f"| {entry['positive_in']}/{entry['n_seeds']} | {per_seed_text} |"
        )
    lines += [
        "",
        f"Matched thresholds: "
        f"{', '.join(_fmt(value, 2) for value in aggregate['matched_thresholds'])}  ",
        f"Recall deviations: "
        f"{', '.join(_fmt_delta(value, 5) for value in aggregate['recall_deviations'])}  ",
        f"Anchor check: max abs delta {_fmt(aggregate['anchor']['max_abs_delta'], 6)} over "
        f"{aggregate['anchor']['logs_found_in']} run log(s) "
        f"(tolerance {aggregate['anchor']['tolerance']}).",
        "",
        "## Reading",
        "",
        "`dF1 ~ 0` with `dPrecision ~ 0` and `dRecall ~ 0` at the matched recall means A "
        "reaches B's recall by moving its threshold and scoring the same F1: the recall "
        "term changed where the operating point sits, not how well the model ranks "
        "pixels, and the gain must not be reported as a model contribution.",
        "",
        "`dF1 < 0` (A at matched recall is worse) with the same recall means the recall "
        "term re-ranked pixels towards the crack centres, which is a model-level effect "
        "that survives the operating-point control.",
        "",
        "`dbF1`, `dclDice` and `dCE` decide whether the trade touched structure rather "
        "than raw overlap: a boundary or topology change at equal recall is not "
        "explained by the threshold alone.",
        "",
        "If section 1 shows any max abs delta above 2e-4, the recomputed metric path "
        "differs from the pipeline that wrote the run log; stop and reconcile the "
        "evaluation code before reading sections 2 and 3.",
    ]
    return lines


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--reference", required=True, type=parse_run, metavar="LABEL=DIR",
                        help="Control cell A: a run directory or a seed directory "
                             "(a run directory may hold several seeds)")
    parser.add_argument("--matched", required=True, type=parse_run, metavar="LABEL=DIR",
                        help="Experiment cell B, same conventions as --reference; seeds are "
                             "paired by auto_protocol_resolved.training_seed")
    parser.add_argument("--split", default="test")
    parser.add_argument("--thresholds", default="0.10:0.90:0.01",
                        help="Threshold grid for the recall sweep: 'start:stop:step' or a "
                             "comma list")
    parser.add_argument("--tolerance", type=float, default=0.0005,
                        help="Accepted |R_reference(t*) - R*| before a WARNING is printed")
    parser.add_argument("--mask-only", action="store_true",
                        help="Use the eval-only mask path (bitwise identical mask, faster)")
    parser.add_argument("--max-images", type=int, default=None, help="Debug subset size")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Inference batch size (does not affect the metrics)")
    parser.add_argument("--device", default=None,
                        help="torch device; defaults to cuda when available")
    parser.add_argument("--output-root", type=Path,
                        default=Path("runs/diagnostics/matched_recall"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.grid = threshold_grid(args.thresholds)

    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = (ROOT / output_root).resolve()
    else:
        output_root = output_root.resolve()
    for forbidden in FORBIDDEN_OUTPUT_ROOTS:
        if output_root == forbidden or forbidden in output_root.parents:
            print(f"error: --output-root {output_root} is inside {forbidden}; this script never "
                  f"writes into paper/ or configs/", file=sys.stderr)
            return 2

    reference_label, reference_dir = args.reference
    matched_label, matched_dir = args.matched
    reference_seeds = dict(discover_seed_dirs(reference_dir))
    matched_seeds = dict(discover_seed_dirs(matched_dir))

    if set(reference_seeds) != set(matched_seeds):
        print(
            f"error: seed sets differ; {reference_label} has {sorted(reference_seeds)}, "
            f"{matched_label} has {sorted(matched_seeds)}. Every seed must exist on both "
            "sides to form a pair.",
            file=sys.stderr,
        )
        return 2

    seeds = sorted(reference_seeds)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    metadata = {
        "created": datetime.now().astimezone().isoformat(),
        "script": "diagnose_matched_recall_control.py",
        "question": "Does cell B's gain survive when A is moved to the same pooled recall?",
        "intent": "diagnostic control only; t* must not be used for model or threshold selection",
        "reference": {"label": reference_label, "dir": str(reference_dir)},
        "matched": {"label": matched_label, "dir": str(matched_dir)},
        "split": args.split,
        "fixed_threshold": FIXED_THRESHOLD,
        "grid": args.grid,
        "tolerance": args.tolerance,
        "anchor_tolerance": ANCHOR_TOLERANCE,
        "mask_only": bool(args.mask_only),
        "max_images": args.max_images,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": str(device),
        "seeds": seeds,
    }

    print(f"reference {reference_label}: {reference_dir}  seeds={seeds}", flush=True)
    print(f"matched   {matched_label}: {matched_dir}  seeds={seeds}", flush=True)
    print(f"split={args.split}  device={device}  grid={args.grid[0]:.2f}..{args.grid[-1]:.2f} "
          f"({len(args.grid)} points)  tolerance={args.tolerance}", flush=True)

    pairs: dict[str, dict[str, object]] = {}
    for seed in seeds:
        entry = evaluate_seed_pair(seed, reference_seeds[seed], matched_seeds[seed], args, device)
        pairs[str(seed)] = entry

        anchor = entry["anchor"]
        point = entry["matched_point"]
        delta = entry["deltas"]
        print(
            f"  seed {seed}: anchor F1 {anchor['reference_at_0.5']['F1']:.4f} "
            f"(log {_fmt(anchor['logged']['F1'])}, max|delta| {_fmt(anchor['max_abs_delta'], 6)})",
            flush=True,
        )
        if anchor["within_tolerance"] is False:
            print(f"  WARNING seed {seed}: recomputed A@0.5 differs from train.log by up to "
                  f"{anchor['max_abs_delta']:.2e} (> {ANCHOR_TOLERANCE:g}); the metric path is "
                  f"not identical to the training pipeline", flush=True)
        print(
            f"  seed {seed}: R* {point['target_recall']:.4f} -> t* "
            f"{point['selected_threshold']:.2f} gives R {point['swept_recall']:.4f} "
            f"(deviation {point['recall_deviation']:+.5f}); "
            f"dF1 {delta['F1'] * 100:+.2f}pp dbF1 {delta['BoundaryF1'] * 100:+.2f}pp "
            f"dclDice {delta['clDice'] * 100:+.2f}pp dCE {delta['ComponentError']:+.4f}",
            flush=True,
        )
        if point["within_tolerance"] is False:
            print(f"  WARNING seed {seed}: no grid threshold matches R* within "
                  f"{args.tolerance:g} (best deviation {point['recall_deviation']:+.5f})",
                  flush=True)
        if not entry["stems_agree"]:
            print(f"  WARNING seed {seed}: reference and matched splits yield different "
                  f"image order/stems; the two cells may not share a split", flush=True)
        if abs(entry["anchor"]["fast_vs_reference_recall_delta"]) > 1e-6:
            print(f"  WARNING seed {seed}: recall sweep and per_image_metrics disagree by "
                  f"{entry['anchor']['fast_vs_reference_recall_delta']:.2e}", flush=True)

    aggregate = aggregate_seeds(pairs)
    payload: dict[str, object] = {"metadata": metadata, "pairs": pairs, "aggregate": aggregate}

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = output_root / f"MATCHED_RECALL_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    markdown_text = "\n".join(render_markdown(payload)) + "\n"
    for target_dir in (output_dir, output_root):
        (target_dir / "matched_recall.json").write_text(json_text, encoding="utf-8")
        (target_dir / "matched_recall.md").write_text(markdown_text, encoding="utf-8")

    print()
    for key in ("F1", "BoundaryF1", "clDice", "ComponentError"):
        entry = aggregate["deltas"][key]
        scale = 100.0 if key != "ComponentError" else 1.0
        print(f"{key:>14s}: dmean {entry['mean'] * scale:+.4f}  sd {entry['sd'] * scale:.4f}  "
              f"delta>0 in {entry['positive_in']}/{entry['n_seeds']}")
    print(f"anchor max abs delta: {_fmt(aggregate['anchor']['max_abs_delta'], 6)} "
          f"(tolerance {aggregate['anchor']['tolerance']})")
    print(f"results written: {output_dir}")
    print(f"latest copy:     {output_root / 'matched_recall.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
