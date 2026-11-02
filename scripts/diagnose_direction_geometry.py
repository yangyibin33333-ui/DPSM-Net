"""D2: why are vertical centrelines covered worse?  Geometry of each direction bin.

P1 measured the recall of ground-truth centreline pixels by local tangent bin and
found H far above V.  This script asks whether the ground truth itself explains
that gap: are vertically oriented centreline pixels thinner, shorter, or of lower
local contrast than horizontal ones?

Pure NumPy/SciPy/scikit-image, no model, no GPU, no training.  It reads the masks,
the images, and the cached ``tangent_gt`` / ``tangent_valid`` targets, and writes
one table per dataset.

Interpretation guide
--------------------
* V width / H width clearly below 1  -> the gap is a resolution problem: vertical
  structures are thinner in this dataset, so the fix is extra effective
  resolution in that direction, not a scan or kernel redesign.
* widths comparable but recall differs -> the gap is a property of the model's
  anisotropic processing (Stage-4 grid is not square and the 1x13 / 13x1 strips
  span different fractions of the image), so the fix is aspect-aware operators.
* No gap in geometry and no gap in recall -> nothing to fix.

Example
-------
    python scripts/diagnose_direction_geometry.py --data Crack500=./Crack500
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.measure import label as connected_label
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.crack_dataset import _find_pairs  # noqa: E402

BIN_NAMES = ("H", "V", "D", "AD")


def analyse_arrays(
    mask: np.ndarray,
    tangent: np.ndarray,
    valid: np.ndarray,
    gray: np.ndarray,
    *,
    threshold: int = 127,
    window: int = 5,
) -> dict[str, np.ndarray | float]:
    """Per-bin geometry for one image.

    ``mask`` is uint8 (H, W), ``tangent`` float (4, H, W), ``valid`` float (1, H, W)
    or (H, W), and ``gray`` float (H, W) in [0, 255].
    Returns per-bin pixel counts and sums so datasets can be pooled exactly.
    """
    binary = mask > threshold
    validity = np.asarray(valid).reshape(binary.shape) > 0.5
    bins = np.asarray(tangent).argmax(axis=0)

    # Centreline pixels and their local half-width from the distance transform.
    skeleton = skeletonize(binary)
    distance = ndimage.distance_transform_edt(binary) if binary.any() else np.zeros_like(gray, dtype=float)
    width = 2.0 * distance

    # Length proxy: size of the connected skeleton component each pixel belongs to.
    components = connected_label(skeleton, connectivity=2)
    sizes = np.bincount(components.ravel())
    component_size = sizes[components] if components.max() > 0 else np.zeros_like(components, dtype=float)

    # Local contrast: intensity range inside a small window, normalised to [0, 1].
    if binary.any() and validity.any():
        hi = ndimage.maximum_filter(gray, size=window)
        lo = ndimage.minimum_filter(gray, size=window)
        contrast = (hi - lo) / 255.0
    else:
        contrast = np.zeros_like(gray, dtype=float)

    selected = validity & skeleton
    result: dict[str, np.ndarray | float] = {}
    for index, name in enumerate(BIN_NAMES):
        mask_bin = selected & (bins == index)
        result[f"n_{name}"] = float(mask_bin.sum())
        result[f"width_{name}"] = float(width[mask_bin].sum())
        result[f"contrast_{name}"] = float(contrast[mask_bin].sum())
        result[f"length_{name}"] = float(component_size[mask_bin].sum())
    result["n_valid"] = float(selected.sum())
    result["n_mask"] = float(binary.sum())
    return result


def parse_dataset(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError(f"--data expects DATASET=ROOT, got {value!r}")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"dataset root does not exist: {path}")
    return label.strip(), path


def analyse_dataset(
    label: str,
    root: Path,
    *,
    split: str,
    target_dir_name: str,
    threshold: int,
    window: int,
    max_images: int | None,
) -> dict[str, object]:
    pairs = _find_pairs(root, split)
    if max_images is not None:
        pairs = pairs[:max_images]
    target_dir = root / split / target_dir_name
    totals = {f"{measure}_{name}": 0.0 for measure in ("n", "width", "contrast", "length") for name in BIN_NAMES}
    totals["n_valid"] = 0.0
    totals["n_mask"] = 0.0
    per_image_ratio: list[float] = []
    missing = 0
    for image_path, mask_path in pairs:
        target_path = target_dir / f"{Path(image_path).stem}.npz"
        if not target_path.is_file():
            missing += 1
            continue
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        with np.load(target_path) as cached:
            tangent = cached["tangent_gt"].astype(np.float32)
            valid = cached["tangent_valid"].astype(np.float32)
        if tangent.shape[-2:] != mask.shape:
            raise ValueError(f"target/mask shape mismatch for {target_path}")
        stats = analyse_arrays(mask, tangent, valid, gray, threshold=threshold, window=window)
        for key, value in stats.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        if stats["n_H"] > 0 and stats["n_V"] > 0:
            per_image_ratio.append(
                (stats["width_V"] / stats["n_V"]) / max(stats["width_H"] / stats["n_H"], 1e-6)
            )
    if missing:
        raise FileNotFoundError(
            f"{missing} images in {root}/{split} have no cached target under {target_dir}. "
            "Run `python scripts/precompute_structural_targets.py --data_root "
            f"{root}` first."
        )
    rows: dict[str, object] = {}
    total_pixels = totals["n_valid"] or 1.0
    for name in BIN_NAMES:
        count = totals[f"n_{name}"]
        rows[name] = {
            "share": count / total_pixels,
            "n_pixels": count,
            "mean_width_px": totals[f"width_{name}"] / count if count else float("nan"),
            "mean_contrast": totals[f"contrast_{name}"] / count if count else float("nan"),
            "mean_component_px": totals[f"length_{name}"] / count if count else float("nan"),
        }
    rows["v_over_h_width"] = (
        rows["V"]["mean_width_px"] / rows["H"]["mean_width_px"]
        if rows["H"]["mean_width_px"] else float("nan")
    )
    rows["v_over_h_contrast"] = (
        rows["V"]["mean_contrast"] / rows["H"]["mean_contrast"]
        if rows["H"]["mean_contrast"] else float("nan")
    )
    rows["per_image_v_over_h_width_median"] = (
        statistics.median(per_image_ratio) if per_image_ratio else float("nan")
    )
    rows["n_images"] = len(pairs)
    rows["dataset_root"] = str(root)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", action="append", type=parse_dataset, required=True,
                        metavar="DATASET=ROOT", help="Dataset label and root (repeatable)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--target-dir", default="targets")
    parser.add_argument("--threshold", type=int, default=127)
    parser.add_argument("--contrast-window", type=int, default=5)
    parser.add_argument("--max-images", type=int, default=None, help="Debug subset size")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "runs/diagnostics/direction_asymmetry")
    args = parser.parse_args()

    metadata = {
        "created": datetime.now().astimezone().isoformat(),
        "script": "diagnose_direction_geometry.py (D2)",
        "split": args.split,
        "threshold": args.threshold,
        "contrast_window": args.contrast_window,
        "definitions": {
            "width_px": "2 x Euclidean distance transform at the centreline pixel",
            "contrast": "intensity range inside a window, divided by 255",
            "component_px": "pixel count of the connected centreline component",
        },
    }
    results = {}
    for label, root in args.data:
        print(f"analysing {label}: {root}", flush=True)
        results[label] = analyse_dataset(
            label, root, split=args.split, target_dir_name=args.target_dir,
            threshold=args.threshold, window=args.contrast_window,
            max_images=args.max_images,
        )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_root / f"D2_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "d2_direction_geometry.json").write_text(
        json.dumps({"metadata": metadata, "datasets": results}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = ["# D2: geometry of each direction bin", "",
             "| dataset | bin | share | width (px) | contrast | component (px) |",
             "|---|---|---:|---:|---:|---:|"]
    for label, entry in results.items():
        for name in BIN_NAMES:
            row = entry[name]
            lines.append(
                f"| {label} | {name} | {row['share'] * 100:.1f}% "
                f"| {row['mean_width_px']:.2f} | {row['mean_contrast']:.3f} "
                f"| {row['mean_component_px']:.1f} |"
            )
        lines.append(
            f"| {label} | **V/H** | | **{entry['v_over_h_width']:.3f}** "
            f"| **{entry['v_over_h_contrast']:.3f}** | |"
        )
    lines += ["", "## Reading", "",
              "`V/H` close to 1 means the ground truth has no systematic width or contrast "
              "difference between vertical and horizontal centrelines, so a recall gap must "
              "come from the model.  `V/H` well below 1 means vertical structures are "
              "genuinely thinner or fainter, and the gap is a resolution problem."]
    (output_dir / "d2_direction_geometry.md").write_text("\n".join(lines), encoding="utf-8")

    for label, entry in results.items():
        print(f"{label}: share H/V/D/AD = "
              + "/".join(f"{entry[name]['share'] * 100:.1f}%" for name in BIN_NAMES)
              + f"  width H/V = {entry['H']['mean_width_px']:.2f}/{entry['V']['mean_width_px']:.2f} px"
              + f"  V/H width = {entry['v_over_h_width']:.3f}"
              + f"  V/H contrast = {entry['v_over_h_contrast']:.3f}")
    print(f"results written: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
