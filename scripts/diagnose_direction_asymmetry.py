"""D1: is the horizontal/vertical centreline asymmetry universal across datasets?

E2/P1 measured, on Crack500, a large and stable gap between horizontal and
vertical centreline recall (H ~= 89.3%, V ~= 83.4%, five Stage-4 designs, seed
SD ~ 0.2 pp).  That is an order of magnitude larger than any architecture
difference in this study, so it is the only accuracy target with enough
magnitude to justify a redesign.

Before redesigning anything, this script checks whether the asymmetry is a
property of Crack500 or of the model family, by repeating the orientation-binned
centreline recall on the other datasets that already have trained checkpoints.

It trains nothing and writes nothing outside its own output directory.  The
metric definitions and aggregation are imported from
``evaluate_orientation_strata.py``, so the numbers stay comparable with P1.

Examples
--------
    # every dataset that has a seed-42 model in the main comparison table
    python scripts/diagnose_direction_asymmetry.py \
        --run DeepCrack=runs/paper/main_table_a/deepcrack/A5_C40_S4_LSSCF_DPER_RSDM_DPSM80 \
        --run CrackMap=runs/paper/main_table_a/crackmap/A5_C40_S4_LSSCF_DPER_RSDM_DPSM80 \
        --run TUT=runs/paper/main_table_a/tut/A5_C40_S4_LSSCF_DPER_RSDM_DPSM80

    # a single custom run directory
    python scripts/diagnose_direction_asymmetry.py \
        --run Crack500=runs/paper/dependency_aware_ablation_strict_v2/crack500/M3_FrameworkOn_DPSMOn_StructSupOff
"""

from __future__ import annotations

import argparse
import csv
import json
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
from evaluate_orientation_strata import (  # noqa: E402
    SKELETON_BIN_NAMES,
    aggregate_skeleton,
    discover_seed_dirs,
    per_image_metrics,
    read_config,
)
from models import build_model  # noqa: E402
from utils.checkpoint import load_checkpoint  # noqa: E402


def parse_dataset(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError(f"--run expects DATASET=DIR, got {value!r}")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"run directory does not exist: {path}")
    return label.strip(), path


def resolve_run(run_dir: Path) -> tuple[int, Path, dict]:
    """Return (training_seed, seed_directory, resolved_config).

    ``--run`` accepts either a seed directory or a parent directory that holds
    seed subdirectories; the lowest training seed is used when several exist.
    """
    seed, seed_dir = discover_seed_dirs(run_dir)[0]
    return seed, seed_dir, read_config(seed_dir / "config.yaml")


def evaluate_dataset(
    label: str,
    run_dir: Path,
    *,
    split: str,
    device: torch.device,
    threshold: float,
    batch_size: int,
    num_workers: int,
    tolerance: int,
    mask_only: bool,
    max_images: int | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    seed, seed_dir, config = resolve_run(run_dir)
    print(f"  using seed {seed}: {seed_dir.name}", flush=True)
    model_config = dict(config["model"])
    model_name = str(model_config.pop("name", "DEMambaNet"))
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
    model = build_model(model_name, **model_config).to(device)
    load_checkpoint(seed_dir / "best_structural_score.pth", model, map_location=device)
    model.eval()

    rows: list[dict[str, object]] = []
    seen = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            outputs = model.forward_mask_only(images) if mask_only else model(images)
            logits = outputs["mask"]
            for index, path in enumerate(batch["image_path"]):
                metrics = per_image_metrics(
                    logits[index : index + 1],
                    batch["mask"][index : index + 1],
                    threshold=threshold,
                    tangent_gt=batch["tangent_gt"][index : index + 1],
                    tangent_valid=batch["tangent_valid"][index : index + 1],
                    skeleton_tolerance=tolerance,
                )
                rows.append({"run": label, "seed": seed, "stem": Path(path).stem, **metrics})
                seen += 1
                if max_images is not None and seen >= max_images:
                    break
            if max_images is not None and seen >= max_images:
                break
    context = {
        "run_dir": str(seed_dir),
        "requested_dir": str(run_dir),
        "seed": seed,
        "data_root": str(root),
        "img_size": list(img_size) if img_size else None,
        "n_images": seen,
        "f1_pooled": _pooled_f1(rows),
    }
    return rows, context


def _pooled_f1(rows: list[dict[str, object]]) -> float:
    tp = sum(float(row["tp"]) for row in rows)
    fp = sum(float(row["fp"]) for row in rows)
    fn = sum(float(row["fn"]) for row in rows)
    return 2.0 * tp / (2.0 * tp + fp + fn + 1e-7)


def skeleton_concentration(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    """How concentrated are the misses of each direction bin?

    A bin whose misses come from a handful of images is an outlier story; a bin
    whose misses are spread over the split is a systematic failure.  This is the
    difference between "CFD has one broken image" and "CFD cannot see vertical
    centrelines".
    """
    result: dict[str, dict[str, float]] = {}
    for name in SKELETON_BIN_NAMES:
        recalled: list[float] = []
        missed: list[float] = []
        for row in rows:
            total = float(row.get(f"skel_n_{name}", 0.0))
            if total <= 0.0:
                continue
            hits = float(row.get(f"skel_hit_{name}", 0.0))
            recalled.append(hits / total)
            missed.append(total - hits)
        images = len(recalled)
        if images == 0:
            result[name] = {"images": 0.0}
            continue
        ranked = sorted(missed, reverse=True)
        top_count = max(1, round(0.1 * images))
        total_missed = sum(missed)
        percentile = sorted(recalled)
        result[name] = {
            "images": float(images),
            "median_image_recall": statistics.median(recalled),
            "p10_image_recall": percentile[max(0, int(0.1 * (images - 1)))],
            "share_images_below_50pct": sum(1 for value in recalled if value < 0.5) / images,
            "top_decile_share_of_misses": (
                sum(ranked[:top_count]) / total_missed if total_missed > 0 else 0.0
            ),
        }
    return result


def write_per_image_csv(path: Path, rows: list[dict[str, object]]) -> None:
    columns = ["run", "seed", "stem", "tp", "fp", "fn"]
    for name in SKELETON_BIN_NAMES:
        columns += [f"skel_n_{name}", f"skel_hit_{name}", f"recall_{name}"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            record = {key: row.get(key) for key in columns}
            for name in SKELETON_BIN_NAMES:
                total = float(row.get(f"skel_n_{name}", 0.0))
                hits = float(row.get(f"skel_hit_{name}", 0.0))
                record[f"recall_{name}"] = round(hits / total, 6) if total > 0 else ""
            writer.writerow(record)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", action="append", type=parse_dataset, required=True,
                        metavar="DATASET=DIR",
                        help="Dataset label and a run directory holding a checkpoint (repeatable)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--skeleton-tolerance", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mask-only", action="store_true",
                        help="Use the eval-only mask path (bitwise identical mask, faster)")
    parser.add_argument("--max-images", type=int, default=None, help="Debug subset size")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "runs/diagnostics/direction_asymmetry")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    metadata = {
        "created": datetime.now().astimezone().isoformat(),
        "script": "diagnose_direction_asymmetry.py (D1)",
        "split": args.split,
        "threshold": args.threshold,
        "skeleton_tolerance": args.skeleton_tolerance,
        "question": "Is the horizontal/vertical centreline recall gap present on every dataset?",
    }

    rows: list[dict[str, object]] = []
    contexts: dict[str, dict[str, object]] = {}
    per_dataset_rows: dict[str, list[dict[str, object]]] = {}
    for label, run_dir in args.run:
        print(f"evaluating {label}: {run_dir}", flush=True)
        dataset_rows, context = evaluate_dataset(
            label, run_dir, split=args.split, device=device, threshold=args.threshold,
            batch_size=args.batch_size, num_workers=args.num_workers,
            tolerance=args.skeleton_tolerance, mask_only=args.mask_only,
            max_images=args.max_images,
        )
        rows.extend(dataset_rows)
        contexts[label] = context
        per_dataset_rows[label] = dataset_rows

    skeleton = aggregate_skeleton(rows)
    table: dict[str, dict[str, object]] = {}
    for label, entry in skeleton.items():
        table[label] = {
            **{name: entry[name]["recall"] for name in (*SKELETON_BIN_NAMES, "axial", "oblique")},
            "n_skeleton_pixels": entry["axial"]["n_skeleton_pixels"],
            "h_minus_v": entry["H"]["recall"] - entry["V"]["recall"],
            "concentration": skeleton_concentration(per_dataset_rows.get(label, [])),
            **contexts.get(label, {}),
        }

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_root / f"D1_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "d1_orientation_asymmetry.json").write_text(
        json.dumps({"metadata": metadata, "datasets": table},
                   indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for label, rows in per_dataset_rows.items():
        write_per_image_csv(output_dir / f"{label}_per_image.csv", rows)

    lines = ["# D1: is the H/V centreline asymmetry universal?", "",
             f"Protocol: {args.split} split, threshold {args.threshold}, "
             f"{args.skeleton_tolerance} px centreline tolerance.", "",
             "| dataset | n px | H | V | D | AD | H−V (pp) | F1 pooled |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, entry in table.items():
        lines.append(
            f"| {label} | {entry['n_skeleton_pixels']:,.0f} "
            f"| {entry['H'] * 100:.2f} | {entry['V'] * 100:.2f} "
            f"| {entry['D'] * 100:.2f} | {entry['AD'] * 100:.2f} "
            f"| {entry['h_minus_v'] * 100:+.2f} "
            f"| {entry.get('f1_pooled', float('nan')):.4f} |"
        )
    gaps = [entry["h_minus_v"] for entry in table.values()]
    concentration_entries = {
        label: entry["concentration"] for label, entry in table.items()
        if entry.get("concentration")
    }
    if concentration_entries:
        lines += [
            "",
            "## Miss concentration (is the deficit driven by a few images?)",
            "",
            "Per-image recall over skeleton pixels whose tangent is closest to vertical "
            "(V) or horizontal (H). `top-decile share` is the fraction of a bin's missed "
            "skeleton pixels contributed by its worst 10% of images; a uniform failure "
            "gives ~10%, an outlier story gives a value approaching 100%.",
            "",
            "| dataset | V median | V p10 | V imgs <50% | V top-decile share "
            "| H median | H top-decile share | n imgs |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for label, entry in concentration_entries.items():
            vertical = entry.get("V", {})
            horizontal = entry.get("H", {})
            images = vertical.get("images", 0.0)
            if images <= 0:
                continue
            lines.append(
                f"| {label} "
                f"| {vertical['median_image_recall'] * 100:.2f} "
                f"| {vertical['p10_image_recall'] * 100:.2f} "
                f"| {vertical['share_images_below_50pct'] * 100:.2f} "
                f"| {vertical['top_decile_share_of_misses'] * 100:.2f} "
                f"| {horizontal.get('median_image_recall', float('nan')) * 100:.2f} "
                f"| {horizontal.get('top_decile_share_of_misses', float('nan')) * 100:.2f} "
                f"| {int(images)} |"
            )
        lines += [
            "",
            "Reading: if the V top-decile share sits near 10%, the V deficit is spread "
            "evenly over the split and describes the model; if it is far above 10% with "
            "a depressed V p10, the headline gap is dominated by a few pathological "
            "fields and must be reported as a regime change rather than a handicap.",
        ]
    lines += ["", "## Reading", ""]
    if gaps:
        lines.append(
            f"H−V gap: min {min(gaps) * 100:+.2f} pp, max {max(gaps) * 100:+.2f} pp, "
            f"mean {sum(gaps) / len(gaps) * 100:+.2f} pp over {len(gaps)} dataset(s)."
        )
        consistent = all(gap > 0.01 for gap in gaps)
        lines.append("")
        lines.append(
            "VERDICT: the gap has the same sign on every dataset."
            if consistent else
            "VERDICT: the gap does not have the same sign everywhere, so it is not a "
            "universal property of the model family; do not build a redesign on it."
        )
    (output_dir / "d1_orientation_asymmetry.md").write_text("\n".join(lines), encoding="utf-8")

    for label, entry in table.items():
        print(f"{label:>12s} H={entry['H'] * 100:6.2f} V={entry['V'] * 100:6.2f} "
              f"D={entry['D'] * 100:6.2f} AD={entry['AD'] * 100:6.2f} "
              f"H-V={entry['h_minus_v'] * 100:+6.2f}pp n={entry['n_skeleton_pixels']:,.0f}")
    print(f"results written: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
