"""E2: orientation-stratified test evaluation for the Stage-4 variants.

The Stage-4 ablation is neutral on average overlap (Table 6), but the only
mechanism that four-direction scanning can plausibly buy is coverage of oblique
structure.  This script stratifies the frozen test split by the dominant crack
orientation of every ground-truth mask and re-evaluates each checkpoint on the
strata separately.

Orientation is read from the cached structural targets that the training
pipeline already writes (``targets/<stem>.npz``: ``tangent_gt`` is a 4-bin soft
tangent distribution over horizontal, vertical, diagonal and anti-diagonal, and
``tangent_valid`` marks the skeleton cells where a tangent exists).  No new
labels, no retraining, and no change to the production code paths.

Per-image metrics reuse the exact definitions of ``metrics.metrics``:
``SegmentationMetric`` pools boundary F1 over the split, so this script keeps the
raw boundary counts per image and reports both the pooled value (comparable to
Table 6) and the per-image mean.

Example
-------
    python scripts/evaluate_orientation_strata.py \
        --run L1=runs/paper/dpsm_lightweight_ablation/crack500/L1_SingleRasterMamba80 \
        --run L2=runs/paper/dpsm_lightweight_ablation/crack500/L2_FourIndependentMamba80 \
        --run L3=runs/paper/dependency_aware_ablation_strict_v2/crack500/M3_FrameworkOn_DPSMOn_StructSupOff \
        --compare L3:L1 --compare L3:L2

Writes ``strata.json``, ``per_image_metrics.csv``, ``summary.json``,
``summary.md`` and ``comparisons.json`` into a timestamped child directory.
``--strata-only`` writes the stratum assignment without touching the GPU, which
is the cheap way to check the class balance before booking GPU time.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.crack_dataset import build_dataloader  # noqa: E402
from metrics.metrics import _boundary, _topology_scores  # noqa: E402
from models import build_model  # noqa: E402
from utils.checkpoint import load_checkpoint  # noqa: E402

BOUNDARY_TOLERANCE = 2
METRIC_KEYS = ("F1", "IoU", "Precision", "Recall", "BoundaryF1", "clDice", "ComponentError")
POOLED_KEYS = ("BoundaryF1",)
STRATUM_ORDER = ("axial", "mixed", "diagonal", "sparse")
SKELETON_BIN_NAMES = ("H", "V", "D", "AD")


# --------------------------------------------------------------------------- #
# run discovery
# --------------------------------------------------------------------------- #
def parse_run(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError(f"--run expects LABEL=DIR, got {value!r}")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"--run directory does not exist: {path}")
    return label.strip(), path


def parse_pair(value: str) -> tuple[str, str]:
    left, separator, right = value.partition(":")
    if not separator or not left.strip() or not right.strip():
        raise argparse.ArgumentTypeError(f"--compare expects A:B, got {value!r}")
    return left.strip(), right.strip()


def read_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def find_key(node: object, key: str, default: object = None) -> object:
    """Depth-first lookup, because the resolved run config nests ``training_seed``."""
    queue: list[object] = [node]
    while queue:
        current = queue.pop(0)
        if isinstance(current, dict):
            if key in current:
                return current[key]
            queue.extend(value for value in current.values() if isinstance(value, (dict, list)))
        elif isinstance(current, list):
            queue.extend(value for value in current if isinstance(value, (dict, list)))
    return default


def discover_seed_dirs(directory: Path) -> list[tuple[int, Path]]:
    """Return (training_seed, run_directory) pairs that hold a checkpoint."""
    candidates = [directory] if (directory / "config.yaml").is_file() else sorted(
        path for path in directory.iterdir() if path.is_dir()
    )
    found: list[tuple[int, Path]] = []
    for candidate in candidates:
        checkpoint = candidate / "best_structural_score.pth"
        config_path = candidate / "config.yaml"
        if not checkpoint.is_file() or not config_path.is_file():
            continue
        seed = find_key(read_config(config_path), "training_seed")
        if seed is None:
            raise ValueError(f"{config_path} has no training_seed entry; cannot label this run")
        found.append((int(seed), candidate))
    if not found:
        raise FileNotFoundError(
            f"No seed directories with config.yaml + best_structural_score.pth under {directory}"
        )
    return sorted(found)


# --------------------------------------------------------------------------- #
# orientation strata
# --------------------------------------------------------------------------- #
def image_stratum(
    target_dir: Path,
    stem: str,
    *,
    diagonal_cut: float,
    axial_cut: float,
    min_valid_mass: float,
) -> dict[str, object]:
    path = target_dir / f"{stem}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing cached structural target {path}. "
            "Run `python scripts/precompute_structural_targets.py` for this dataset first."
        )
    with np.load(path) as cached:
        tangent = cached["tangent_gt"].astype(np.float64)
        valid = cached["tangent_valid"].astype(np.float64)
    mass = (tangent * valid).sum(axis=(1, 2))
    total = float(mass.sum())
    oblique = float(mass[2] + mass[3])
    axial = float(mass[0] + mass[1])
    ratio = oblique / total if total > 0.0 else math.nan
    if total < min_valid_mass:
        stratum = "sparse"
    elif not math.isfinite(ratio):
        stratum = "sparse"
    elif ratio >= diagonal_cut:
        stratum = "diagonal"
    elif ratio <= axial_cut:
        stratum = "axial"
    else:
        stratum = "mixed"
    return {
        "stem": stem,
        "stratum": stratum,
        "fixed_stratum": stratum,
        "oblique_ratio": None if not math.isfinite(ratio) else round(ratio, 6),
        "tangent_mass": round(total, 4),
        "axial_mass": round(axial, 4),
        "oblique_mass": round(oblique, 4),
        "bins": [round(float(value), 4) for value in mass],
    }


def apply_scheme(
    strata: dict[str, dict[str, object]],
    scheme: str,
    *,
    diagonal_cut: float,
    axial_cut: float,
) -> dict[str, object]:
    """Relabel strata according to ``fixed`` absolute cuts or relative tertiles.

    The absolute cuts left only a handful of images in the oblique-dominant
    stratum on Crack500, so the tertile split is offered as the statistically
    usable alternative.  It is a relative split: the most oblique third of the
    split is compared with the least oblique third.  The chosen boundaries are
    written into the metadata so the split stays reproducible.
    """
    if scheme == "fixed":
        return {"scheme": "fixed", "diagonal_cut": diagonal_cut, "axial_cut": axial_cut}
    if scheme != "tertile":
        raise ValueError(f"Unknown strata scheme {scheme!r}")
    ranked = sorted(
        (float(entry["oblique_ratio"]), stem)
        for stem, entry in strata.items()
        if entry["stratum"] != "sparse" and entry["oblique_ratio"] is not None
    )
    if len(ranked) < 9:
        raise ValueError(f"Too few stratifiable images ({len(ranked)}) for a tertile split")
    lower_index = len(ranked) // 3
    upper_index = (2 * len(ranked)) // 3
    for position, (_ratio, stem) in enumerate(ranked):
        if position < lower_index:
            strata[stem]["stratum"] = "axial"
        elif position >= upper_index:
            strata[stem]["stratum"] = "diagonal"
        else:
            strata[stem]["stratum"] = "mixed"
    return {
        "scheme": "tertile",
        "n_stratified": len(ranked),
        "lower_cut": ranked[lower_index][0],
        "upper_cut": ranked[upper_index][0],
        "definition": "rank split: least oblique third = axial, most oblique third = diagonal",
    }


def build_strata(
    data_root: Path,
    split: str,
    *,
    target_dir_name: str,
    diagonal_cut: float,
    axial_cut: float,
    min_valid_mass: float,
    max_images: int | None,
    scheme: str = "fixed",
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    """Map every image stem of the split to its orientation stratum."""
    dataset = build_dataloader(
        data_root, split, None, 127, 1, 0, shuffle=False, target_dir_name=target_dir_name
    ).dataset
    pairs = list(dataset.pairs)
    if max_images is not None:
        pairs = pairs[:max_images]
    # The cache lives per split, exactly as CrackSegmentationDataset expects:
    # <data_root>/<split>/<target_dir_name>/<stem>.npz
    target_dir = Path(data_root) / split / target_dir_name
    strata: dict[str, dict[str, object]] = {}
    for image_path, _mask_path in pairs:
        stem = Path(image_path).stem
        strata[stem] = image_stratum(
            target_dir,
            stem,
            diagonal_cut=diagonal_cut,
            axial_cut=axial_cut,
            min_valid_mass=min_valid_mass,
        )
    scheme_info = apply_scheme(
        strata, scheme, diagonal_cut=diagonal_cut, axial_cut=axial_cut
    )
    return strata, scheme_info


# --------------------------------------------------------------------------- #
# per-image metrics
# --------------------------------------------------------------------------- #
def per_image_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: float,
    tangent_gt: torch.Tensor | None = None,
    tangent_valid: torch.Tensor | None = None,
    skeleton_tolerance: int = 1,
) -> dict[str, float]:
    """One image. Boundary F1 keeps raw counts so strata can also be pooled."""
    probability = torch.sigmoid(logits.detach().float().cpu())
    prediction = probability > threshold
    truth = target.detach().float().cpu() > 0.5
    tp = float((prediction & truth).sum())
    fp = float((prediction & ~truth).sum())
    fn = float((~prediction & truth).sum())
    eps = 1e-7
    dice = 2.0 * tp / (2.0 * tp + fp + fn + eps)

    prediction_edge = _boundary(prediction)
    target_edge = _boundary(truth)
    radius = 2 * BOUNDARY_TOLERANCE + 1
    prediction_match = prediction_edge & (
        F.max_pool2d(target_edge.float(), radius, 1, BOUNDARY_TOLERANCE) > 0
    )
    target_match = target_edge & (
        F.max_pool2d(prediction_edge.float(), radius, 1, BOUNDARY_TOLERANCE) > 0
    )
    matched = float(prediction_match.sum()) + float(target_match.sum())
    edge_prediction = float(prediction_edge.sum())
    edge_target = float(target_edge.sum())

    cldice, component_error = _topology_scores(
        prediction[0, 0].numpy(), truth[0, 0].numpy()
    )
    metrics = {
        "F1": dice,
        "IoU": tp / (tp + fp + fn + eps),
        "Precision": tp / (tp + fp + eps),
        "Recall": tp / (tp + fn + eps),
        "BoundaryF1": matched / (edge_prediction + edge_target + eps),
        "clDice": float(cldice),
        "ComponentError": float(component_error),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "boundary_matched": matched,
        "boundary_prediction": edge_prediction,
        "boundary_target": edge_target,
    }
    if tangent_gt is not None and tangent_valid is not None:
        # Per-skeleton-pixel direction bins. This is the pixel-level version of
        # the image-level stratum: it asks whether oblique centreline pixels are
        # covered at the same rate as axial ones, which image-level overlap
        # cannot separate.
        validity = tangent_valid.detach().float().cpu()[0] > 0.5          # [1, H, W]
        bins = tangent_gt.detach().float().cpu()[0].argmax(dim=0)[None]   # [1, H, W]
        covered = (
            F.max_pool2d(prediction.float(), 2 * skeleton_tolerance + 1, 1, skeleton_tolerance) > 0
        )[0]                                                              # [1, H, W]
        for index, name in enumerate(SKELETON_BIN_NAMES):
            selected = validity & (bins == index)
            metrics[f"skel_n_{name}"] = float(selected.sum())
            metrics[f"skel_hit_{name}"] = float((selected & covered).sum())
    return metrics


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def evaluate_run(
    label: str,
    seed: int,
    run_dir: Path,
    strata: dict[str, dict[str, object]],
    *,
    split: str,
    device: torch.device,
    threshold: float,
    batch_size: int,
    num_workers: int,
    data_root: Path | None,
    mask_only: bool,
    max_images: int | None,
    skeleton: bool = False,
    skeleton_tolerance: int = 1,
) -> list[dict[str, object]]:
    config = read_config(run_dir / "config.yaml")
    model_config = dict(config["model"])
    model_name = str(model_config.pop("name", "DEMambaNet"))
    data_config = dict(config.get("data", {}))
    root = data_root if data_root is not None else Path(data_config.get("data_root", "./Crack500"))
    if not Path(root).is_absolute():
        root = (ROOT / root).resolve()
    img_size = data_config.get("img_size")
    img_size = tuple(int(value) for value in img_size) if img_size else None
    target_dir_name = str(data_config.get("target_dir_name", "targets"))

    loader = build_dataloader(
        root,
        split,
        img_size,
        int(data_config.get("threshold", 127)),
        batch_size,
        num_workers,
        shuffle=False,
        target_dir_name=target_dir_name,
    )
    model = build_model(model_name, **model_config).to(device)
    load_checkpoint(run_dir / "best_structural_score.pth", model, map_location=device)
    model.eval()

    rows: list[dict[str, object]] = []
    seen = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            outputs = model.forward_mask_only(images) if mask_only else model(images)
            logits = outputs["mask"]
            for index, path in enumerate(batch["image_path"]):
                stem = Path(path).stem
                stratum = strata.get(stem)
                if stratum is None:
                    continue
                metrics = per_image_metrics(
                    logits[index : index + 1],
                    batch["mask"][index : index + 1],
                    threshold=threshold,
                    tangent_gt=batch["tangent_gt"][index : index + 1] if skeleton else None,
                    tangent_valid=batch["tangent_valid"][index : index + 1] if skeleton else None,
                    skeleton_tolerance=skeleton_tolerance,
                )
                row = {"run": label, "seed": seed, "stem": stem, "stratum": stratum["stratum"],
                       "oblique_ratio": stratum["oblique_ratio"], **metrics}
                rows.append(row)
                seen += 1
                if max_images is not None and seen >= max_images:
                    break
            if max_images is not None and seen >= max_images:
                break
    return rows


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def _stratum_entry(rows: list[dict[str, object]], seeds: list[int]) -> dict[str, object]:
    """Seed-mean of per-image metrics plus split-pooled ratios for one stratum."""
    per_seed: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        per_seed.setdefault(int(row["seed"]), []).append(row)
    entry: dict[str, object] = {
        "n_images": float(round(len(rows) / max(len(per_seed), 1))),
        "seeds": seeds,
    }
    for key in METRIC_KEYS:
        seed_means = [
            statistics.fmean(float(row[key]) for row in per_seed[seed])
            for seed in seeds if seed in per_seed
        ]
        entry[key] = statistics.fmean(seed_means)
        entry[f"{key}_sd"] = statistics.stdev(seed_means) if len(seed_means) > 1 else 0.0
        entry[f"{key}_per_seed"] = seed_means
    # Pooled ratios reproduce metrics.metrics.SegmentationMetric, whose F1 is a
    # true-positive pooled ratio over the split rather than a mean per image.
    # Reporting both keeps the stratified table reconcilable with Table 6.
    sums = {
        key: sum(float(row[key]) for row in rows)
        for key in ("tp", "fp", "fn", "boundary_matched", "boundary_prediction", "boundary_target")
    }
    eps = 1e-7
    entry["F1_pooled"] = 2.0 * sums["tp"] / (2.0 * sums["tp"] + sums["fp"] + sums["fn"] + eps)
    entry["IoU_pooled"] = sums["tp"] / (sums["tp"] + sums["fp"] + sums["fn"] + eps)
    entry["Precision_pooled"] = sums["tp"] / (sums["tp"] + sums["fp"] + eps)
    entry["Recall_pooled"] = sums["tp"] / (sums["tp"] + sums["fn"] + eps)
    entry["BoundaryF1_pooled"] = sums["boundary_matched"] / (
        sums["boundary_prediction"] + sums["boundary_target"] + eps
    )
    return entry


def aggregate(rows: list[dict[str, object]]) -> dict[str, dict[str, dict[str, object]]]:
    """run -> stratum (and ``all``) -> aggregated metrics."""
    summary: dict[str, dict[str, dict[str, object]]] = {}
    by_run: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_run.setdefault(str(row["run"]), []).append(row)
    for label, run_rows in by_run.items():
        seeds = sorted({int(row["seed"]) for row in run_rows})
        summary[label] = {}
        for stratum in (*STRATUM_ORDER, "all"):
            subset = (
                run_rows if stratum == "all"
                else [row for row in run_rows if row["stratum"] == stratum]
            )
            if not subset:
                continue
            summary[label][stratum] = _stratum_entry(subset, seeds)
    return summary


def compare(
    summary: dict[str, dict[str, dict[str, object]]],
    pairs: list[tuple[str, str]],
) -> list[dict[str, object]]:
    comparisons: list[dict[str, object]] = []
    for left, right in pairs:
        if left not in summary or right not in summary:
            raise KeyError(f"--compare needs both labels; missing {left if left not in summary else right}")
        for stratum in (*STRATUM_ORDER, "all"):
            if stratum not in summary[left] or stratum not in summary[right]:
                continue
            entry: dict[str, object] = {"left": left, "right": right, "stratum": stratum,
                                        "n_images": summary[left][stratum]["n_images"]}
            for key in ("F1", "BoundaryF1", "clDice", "ComponentError"):
                left_values = summary[left][stratum][f"{key}_per_seed"]
                right_values = summary[right][stratum][f"{key}_per_seed"]
                if len(left_values) != len(right_values):
                    entry[key] = None
                    continue
                deltas = [l - r for l, r in zip(left_values, right_values)]
                scale = 100.0 if key != "ComponentError" else 1.0
                entry[key] = {
                    "delta_mean": statistics.fmean(deltas) * scale,
                    "delta_per_seed": [value * scale for value in deltas],
                    "positive_seeds": sum(1 for value in deltas if value > 0),
                    "n_seeds": len(deltas),
                }
            comparisons.append(entry)
    return comparisons


# --------------------------------------------------------------------------- #
# orientation-binned skeleton recall
# --------------------------------------------------------------------------- #
SKELETON_GROUPS: dict[str, tuple[str, ...]] = {
    **{name: (name,) for name in SKELETON_BIN_NAMES},
    "axial": ("H", "V"),
    "oblique": ("D", "AD"),
}


def aggregate_skeleton(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    """run -> per-bin centreline recall, pooled over images and averaged over seeds."""
    summary: dict[str, dict[str, object]] = {}
    by_run: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        if "skel_n_H" in row:
            by_run.setdefault(str(row["run"]), []).append(row)
    for label, run_rows in by_run.items():
        seeds = sorted({int(row["seed"]) for row in run_rows})
        entry: dict[str, object] = {"seeds": seeds}
        for group, parts in SKELETON_GROUPS.items():
            recall_per_seed: list[float] = []
            pixels_per_seed: list[float] = []
            for seed in seeds:
                seed_rows = [row for row in run_rows if int(row["seed"]) == seed]
                hits = 0.0
                total = 0.0
                for part in parts:
                    hits += sum(float(row[f"skel_hit_{part}"]) for row in seed_rows)
                    total += sum(float(row[f"skel_n_{part}"]) for row in seed_rows)
                recall_per_seed.append(hits / total if total > 0 else float("nan"))
                pixels_per_seed.append(total)
            entry[group] = {
                "recall": statistics.fmean(recall_per_seed),
                "recall_sd": (
                    statistics.stdev(recall_per_seed) if len(recall_per_seed) > 1 else 0.0
                ),
                "recall_per_seed": recall_per_seed,
                "n_skeleton_pixels": statistics.fmean(pixels_per_seed),
            }
        summary[label] = entry
    return summary


def compare_skeleton(
    summary: dict[str, dict[str, object]],
    pairs: list[tuple[str, str]],
) -> list[dict[str, object]]:
    """Per-bin skeleton-recall difference in percentage points, left minus right."""
    comparisons: list[dict[str, object]] = []
    for left, right in pairs:
        if left not in summary or right not in summary:
            continue
        for group in SKELETON_GROUPS:
            left_values = summary[left][group]["recall_per_seed"]
            right_values = summary[right][group]["recall_per_seed"]
            if len(left_values) != len(right_values):
                continue
            deltas = [(a - b) * 100.0 for a, b in zip(left_values, right_values)]
            comparisons.append({
                "left": left,
                "right": right,
                "bin": group,
                "delta_mean": statistics.fmean(deltas),
                "delta_per_seed": deltas,
                "positive_seeds": sum(1 for value in deltas if value > 0),
                "n_seeds": len(deltas),
                "n_skeleton_pixels": summary[left][group]["n_skeleton_pixels"],
            })
    return comparisons


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def write_outputs(
    output_dir: Path,
    strata: dict[str, dict[str, object]],
    rows: list[dict[str, object]],
    summary: dict[str, dict[str, dict[str, float]]],
    comparisons: list[dict[str, object]],
    metadata: dict[str, object],
    skeleton_summary: dict[str, dict[str, object]] | None = None,
    skeleton_comparisons: list[dict[str, object]] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "strata.json").write_text(
        json.dumps({"metadata": metadata, "images": strata}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    columns = ["run", "seed", "stem", "stratum", "oblique_ratio", *METRIC_KEYS,
               "tp", "fp", "fn", "boundary_matched", "boundary_prediction", "boundary_target"]
    with (output_dir / "per_image_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})
    (output_dir / "summary.json").write_text(
        json.dumps({"metadata": metadata, "summary": summary, "comparisons": comparisons},
                   indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = ["# Orientation-stratified Stage-4 evaluation", ""]
    counts: dict[str, int] = {}
    for entry in strata.values():
        counts[str(entry["stratum"])] = counts.get(str(entry["stratum"]), 0) + 1
    lines.append("Stratum sizes: " + ", ".join(
        f"{name}={counts.get(name, 0)}" for name in STRATUM_ORDER))
    lines.append("")
    lines.append("| run | stratum | n | F1 per-image | F1 pooled | BoundaryF1 pooled | clDice | ComponentError |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for label in summary:
        for stratum in (*STRATUM_ORDER, "all"):
            entry = summary[label].get(stratum)
            if entry is None:
                continue
            lines.append(
                f"| {label} | {stratum} | {int(entry['n_images'])} "
                f"| {entry['F1']:.4f} ± {entry['F1_sd']:.4f} "
                f"| {entry['F1_pooled']:.4f} "
                f"| {entry['BoundaryF1_pooled']:.4f} "
                f"| {entry['clDice']:.4f} ± {entry['clDice_sd']:.4f} "
                f"| {entry['ComponentError']:.4f} |"
            )
    if comparisons:
        lines += ["", "## Paired comparisons (left minus right, percentage points)", "",
                  "| comparison | stratum | ΔF1 | per-seed | sign | ΔBoundaryF1 | ΔclDice | ΔComponentError |",
                  "|---|---|---:|---|---:|---:|---:|---:|"]
        for entry in comparisons:
            f1 = entry["F1"]
            if f1 is None:
                continue
            lines.append(
                f"| {entry['left']} - {entry['right']} | {entry['stratum']} "
                f"| {f1['delta_mean']:+.2f} "
                f"| {'/'.join(f'{value:+.2f}' for value in f1['delta_per_seed'])} "
                f"| {f1['positive_seeds']}/{f1['n_seeds']} "
                f"| {entry['BoundaryF1']['delta_mean']:+.2f} "
                f"| {entry['clDice']['delta_mean']:+.2f} "
                f"| {entry['ComponentError']['delta_mean']:+.3f} |"
            )
    lines += ["", "Decision rule (pre-registered): four-direction coverage is treated as a",
              "conditional accuracy mechanism only if the diagonal stratum improves by at",
              "least 1.0 point of F1 with the same sign in every seed.", ""]
    if skeleton_summary:
        tolerance = metadata.get("skeleton_tolerance", 1)
        lines += ["", f"## Orientation-binned centreline recall (%, {tolerance} px tolerance)", "",
                  "| run | H | V | D | AD | axial (H+V) | oblique (D+AD) |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for label, entry in skeleton_summary.items():
            cells = " | ".join(
                f"{entry[name]['recall'] * 100:.2f} ± {entry[name]['recall_sd'] * 100:.2f}"
                for name in (*SKELETON_BIN_NAMES, "axial", "oblique")
            )
            lines.append(f"| {label} | {cells} |")
        pixels = skeleton_summary[next(iter(skeleton_summary))]["axial"]["n_skeleton_pixels"]
        lines += ["", f"Centreline pixel counts per seed: axial≈{pixels:,.0f}", ""]
        if skeleton_comparisons:
            lines += ["", "### Paired centreline-recall differences (percentage points)", "",
                      "| comparison | bin | Δrecall | per-seed | sign |",
                      "|---|---|---:|---|---:|"]
            for entry in skeleton_comparisons:
                lines.append(
                    f"| {entry['left']} - {entry['right']} | {entry['bin']} "
                    f"| {entry['delta_mean']:+.2f} "
                    f"| {'/'.join(f'{value:+.2f}' for value in entry['delta_per_seed'])} "
                    f"| {entry['positive_seeds']}/{entry['n_seeds']} |"
                )
        (output_dir / "skeleton_recall.json").write_text(
            json.dumps({"summary": skeleton_summary, "comparisons": skeleton_comparisons},
                       indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "comparisons.json").write_text(
        json.dumps(comparisons, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", type=parse_run, required=True,
                        metavar="LABEL=DIR",
                        help="Run directory containing seed subdirectories (repeatable)")
    parser.add_argument("--compare", action="append", type=parse_pair, default=[],
                        metavar="A:B", help="Label pair to compare, left minus right (repeatable)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--diagonal-cut", type=float, default=0.60,
                        help="Oblique mass fraction at or above which an image is diagonal-dominant")
    parser.add_argument("--axial-cut", type=float, default=0.40,
                        help="Oblique mass fraction at or below which an image is axial-dominant")
    parser.add_argument("--min-valid-mass", type=float, default=50.0,
                        help="Minimum total tangent mass before an image is stratified")
    parser.add_argument("--strata-scheme", choices=("fixed", "tertile"), default="fixed",
                        help="fixed: absolute oblique-ratio cuts; tertile: relative third-of-split split")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Override the data root recorded in each run config")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mask-only", action="store_true",
                        help="Use the eval-only mask path (bitwise identical mask, faster)")
    parser.add_argument("--max-images", type=int, default=None, help="Debug subset size")
    parser.add_argument("--strata-only", action="store_true",
                        help="Write the stratum assignment and exit without loading a model")
    parser.add_argument("--skeleton-recall", action="store_true",
                        help="Also report orientation-binned centreline recall (pixel level)")
    parser.add_argument("--skeleton-tolerance", type=int, default=1,
                        help="Pixel tolerance when checking whether a centreline pixel is covered")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "runs/diagnostics/orientation_strata")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_root = args.data_root
    if data_root is not None and not data_root.is_absolute():
        data_root = (ROOT / data_root).resolve()

    # Every run in one invocation must share the split geometry; read it from the
    # first seed directory of the first run.
    first_label, first_dir = args.run[0]
    first_seed_dirs = discover_seed_dirs(first_dir)
    first_config = read_config(first_seed_dirs[0][1] / "config.yaml")
    first_data = dict(first_config.get("data", {}))
    root = data_root if data_root is not None else Path(first_data.get("data_root", "./Crack500"))
    if not Path(root).is_absolute():
        root = (ROOT / root).resolve()
    target_dir_name = str(first_data.get("target_dir_name", "targets"))

    strata, scheme_info = build_strata(
        root, args.split,
        target_dir_name=target_dir_name,
        diagonal_cut=args.diagonal_cut,
        axial_cut=args.axial_cut,
        min_valid_mass=args.min_valid_mass,
        max_images=args.max_images,
        scheme=args.strata_scheme,
    )
    counts: dict[str, int] = {}
    for entry in strata.values():
        counts[str(entry["stratum"])] = counts.get(str(entry["stratum"]), 0) + 1
    print("stratum sizes: " + ", ".join(f"{name}={counts.get(name, 0)}" for name in STRATUM_ORDER))
    print("strata scheme: " + json.dumps(scheme_info, ensure_ascii=False))

    metadata = {
        "created": datetime.now().astimezone().isoformat(),
        "split": args.split,
        "data_root": str(root),
        "threshold": args.threshold,
        "diagonal_cut": args.diagonal_cut,
        "axial_cut": args.axial_cut,
        "min_valid_mass": args.min_valid_mass,
        "strata_scheme": scheme_info,
        "skeleton_tolerance": args.skeleton_tolerance if args.skeleton_recall else None,
        "runs": [{"label": label, "directory": str(directory)} for label, directory in args.run],
        "stratum_sizes": counts,
        "metric_definitions": "metrics.metrics (SegmentationMetric boundary tolerance 2)",
    }

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_root / timestamp
    if args.strata_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "strata.json").write_text(
            json.dumps({"metadata": metadata, "images": strata}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"strata written: {output_dir / 'strata.json'}")
        return 0

    rows: list[dict[str, object]] = []
    for label, directory in args.run:
        for seed, run_dir in discover_seed_dirs(directory):
            print(f"evaluating {label} seed {seed}: {run_dir}", flush=True)
            rows.extend(
                evaluate_run(
                    label, seed, run_dir, strata,
                    split=args.split, device=device, threshold=args.threshold,
                    batch_size=args.batch_size, num_workers=args.num_workers,
                    data_root=root, mask_only=args.mask_only, max_images=args.max_images,
                    skeleton=args.skeleton_recall, skeleton_tolerance=args.skeleton_tolerance,
                )
            )
    summary = aggregate(rows)
    comparisons = compare(summary, args.compare) if args.compare else []
    skeleton_summary = aggregate_skeleton(rows) if args.skeleton_recall else None
    skeleton_comparisons = (
        compare_skeleton(skeleton_summary, args.compare)
        if (args.skeleton_recall and args.compare and skeleton_summary) else None
    )
    write_outputs(output_dir, strata, rows, summary, comparisons, metadata,
                  skeleton_summary, skeleton_comparisons)

    for label in summary:
        for stratum in STRATUM_ORDER:
            entry = summary[label].get(stratum)
            if entry is None:
                continue
            print(f"{label:>4s} {stratum:>9s} n={int(entry['n_images']):3d} "
                  f"F1={entry['F1']:.4f}±{entry['F1_sd']:.4f} "
                  f"bF1={entry['BoundaryF1']:.4f} clDice={entry['clDice']:.4f}")
    for entry in comparisons:
        f1 = entry["F1"]
        if f1 is None:
            continue
        print(f"Δ {entry['left']}-{entry['right']} {entry['stratum']:>9s} "
              f"ΔF1={f1['delta_mean']:+.2f}pp "
              f"seeds={'/'.join(f'{value:+.2f}' for value in f1['delta_per_seed'])} "
              f"({f1['positive_seeds']}/{f1['n_seeds']})")
    print(f"results written: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
