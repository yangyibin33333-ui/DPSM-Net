"""Audit fixed-threshold false positives on DeepCrack validation data only."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import build_dataloader
from models import DEMambaNet
from utils import load_checkpoint
from utils.fp_geometry import bridge_false_positive_mask


FIXED_THRESHOLD = 0.5
BOUNDARY_EXCLUSION_KERNEL_SIZE = 5
COHESION_KERNEL_SIZE = 3
MIN_ACTIVE_NEIGHBORS = 3
TEXTURE_FP_MIN_SHARE = 0.50
TEXTURE_FP_MIN_IMAGES = 3
TEXTURE_FP_MAX_TOP2_SHARE = 0.50
CATEGORY_NAMES = ("boundary_overextension", "far_isolated", "far_texture", "bridge")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Audit validation false-positive geometry at the fixed threshold 0.5")
    parser.add_argument("--config", default="configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml")
    parser.add_argument("--checkpoint", required=True, help="EMA+BPCM best_structural_score checkpoint")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--img_size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def classify_false_positives(probability: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """Create a disjoint fixed-threshold partition of validation false positives."""
    if probability.shape != target.shape or probability.ndim != 4 or probability.size(1) != 1:
        raise ValueError("probability and target must have matching shape [B, 1, H, W].")
    if probability.size(0) != 1:
        raise ValueError("classify_false_positives expects exactly one image at a time.")
    probability = probability.detach().float().cpu()
    target = target.detach().float().cpu()
    prediction = probability > FIXED_THRESHOLD
    foreground = target > 0.5
    false_positive = prediction & (~foreground)
    target_np = foreground[0, 0].numpy()
    prediction_np = prediction[0, 0].numpy()
    fp_np = false_positive[0, 0].numpy()
    bridge = torch.from_numpy(bridge_false_positive_mask(prediction_np, target_np, fp_np)).view_as(false_positive)

    dilated_gt = F.max_pool2d(
        foreground.float(),
        kernel_size=BOUNDARY_EXCLUSION_KERNEL_SIZE,
        stride=1,
        padding=BOUNDARY_EXCLUSION_KERNEL_SIZE // 2,
    ) > 0.5
    boundary_overextension = false_positive & dilated_gt & (~bridge)
    far_background = false_positive & (~dilated_gt) & (~bridge)
    cohesion_kernel = torch.ones((1, 1, COHESION_KERNEL_SIZE, COHESION_KERNEL_SIZE), dtype=torch.float32)
    cohesion_count = F.conv2d(far_background.float(), cohesion_kernel, padding=COHESION_KERNEL_SIZE // 2)
    far_texture = far_background & (cohesion_count >= MIN_ACTIVE_NEIGHBORS)
    far_isolated = far_background & (~far_texture)
    categories = {
        "boundary_overextension": boundary_overextension,
        "far_isolated": far_isolated,
        "far_texture": far_texture,
        "bridge": bridge,
        "all_false_positives": false_positive,
    }
    partition = sum(value.to(torch.int64) for name, value in categories.items() if name != "all_false_positives")
    if not torch.equal(partition, false_positive.to(torch.int64)):
        raise RuntimeError("False-positive categories must form a disjoint complete partition.")
    return categories


def _ratio(count: int, total: int) -> float:
    return 0.0 if total == 0 else count / total


def summarize_records(records: list[dict[str, object]]) -> dict[str, object]:
    total_false_positives = sum(int(record["false_positive_pixels"]) for record in records)
    category_pixels = {
        name: sum(int(record[f"{name}_pixels"]) for record in records)
        for name in CATEGORY_NAMES
    }
    categories: dict[str, dict[str, object]] = {}
    for name in CATEGORY_NAMES:
        ratios = [float(record[f"{name}_ratio"]) for record in records]
        ranked = sorted(records, key=lambda record: int(record[f"{name}_pixels"]), reverse=True)
        categories[name] = {
            "pixels": category_pixels[name],
            "share_of_all_false_positives": _ratio(category_pixels[name], total_false_positives),
            "per_image_ratio_mean": float(np.mean(ratios)) if ratios else 0.0,
            "per_image_ratio_median": float(np.median(ratios)) if ratios else 0.0,
            "positive_image_count": sum(int(record[f"{name}_pixels"]) > 0 for record in records),
            "top_10_images": [
                {
                    "stem": str(record["stem"]),
                    "pixels": int(record[f"{name}_pixels"]),
                    "ratio_within_image_false_positives": float(record[f"{name}_ratio"]),
                }
                for record in ranked[:10]
                if int(record[f"{name}_pixels"]) > 0
            ],
        }

    texture_per_image = sorted((int(record["far_texture_pixels"]) for record in records), reverse=True)
    texture_pixels = category_pixels["far_texture"]
    top2_texture_share = _ratio(sum(texture_per_image[:2]), texture_pixels)
    texture_dominates_geometry = texture_pixels > category_pixels["boundary_overextension"] and texture_pixels > category_pixels["bridge"]
    gate_conditions = {
        "far_texture_share_at_least_50_percent": categories["far_texture"]["share_of_all_false_positives"] >= TEXTURE_FP_MIN_SHARE,
        "far_texture_exceeds_boundary_and_bridge": texture_dominates_geometry,
        "far_texture_present_in_at_least_3_images": categories["far_texture"]["positive_image_count"] >= TEXTURE_FP_MIN_IMAGES,
        "top2_images_contribute_at_most_50_percent_of_far_texture": top2_texture_share <= TEXTURE_FP_MAX_TOP2_SHARE,
    }
    allowed = total_false_positives > 0 and all(gate_conditions.values())
    return {
        "fixed_threshold": FIXED_THRESHOLD,
        "total_images": len(records),
        "total_false_positive_pixels": total_false_positives,
        "categories": categories,
        "gate": {
            "decision": "PASS" if allowed else "BLOCK",
            "conditions": gate_conditions,
            "far_texture_top2_image_share": top2_texture_share,
            "criteria": {
                "minimum_far_texture_share": TEXTURE_FP_MIN_SHARE,
                "minimum_far_texture_positive_images": TEXTURE_FP_MIN_IMAGES,
                "maximum_top2_far_texture_share": TEXTURE_FP_MAX_TOP2_SHARE,
            },
        },
    }


def write_report(output_dir: Path, records: list[dict[str, object]], summary: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = ["stem", "image_path", "false_positive_pixels"]
        columns.extend(f"{name}_pixels" for name in CATEGORY_NAMES)
        columns.extend(f"{name}_ratio" for name in CATEGORY_NAMES)
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def _build_record(image_path: str, categories: dict[str, torch.Tensor]) -> dict[str, object]:
    false_positive_pixels = int(categories["all_false_positives"].sum())
    record: dict[str, object] = {
        "stem": Path(image_path).stem,
        "image_path": image_path,
        "false_positive_pixels": false_positive_pixels,
    }
    for name in CATEGORY_NAMES:
        pixels = int(categories[name].sum())
        record[f"{name}_pixels"] = pixels
        record[f"{name}_ratio"] = _ratio(pixels, false_positive_pixels)
    return record


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.data_root is not None:
        cfg["data"]["data_root"] = args.data_root
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.img_size is not None:
        cfg["data"]["img_size"] = args.img_size
    if args.num_workers is not None:
        cfg["train"]["num_workers"] = args.num_workers

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing validation-audit checkpoint: {checkpoint}")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_cfg = dict(cfg["model"])
    model_cfg.pop("name", None)
    model = DEMambaNet(**model_cfg).to(device)
    backend_info = model.mamba_backend_info()
    if not backend_info or any(backend != "mamba_ssm" for backends in backend_info.values() for backend in backends):
        raise RuntimeError(f"Validation FP audit requires official mamba_ssm paths, got: {backend_info}")
    load_checkpoint(checkpoint, model, map_location=device)
    model.eval()

    loader = build_dataloader(
        cfg["data"]["data_root"],
        "val",
        cfg["data"]["img_size"],
        int(cfg["data"]["threshold"]),
        int(cfg["train"]["batch_size"]),
        int(cfg["train"]["num_workers"]),
        shuffle=False,
        target_dir_name=cfg["data"].get("target_dir_name", "targets"),
        persistent_workers=bool(cfg["train"].get("persistent_workers", False)),
        prefetch_factor=int(cfg["train"].get("prefetch_factor", 2)),
    )
    records: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device))["mask"]
            probability = torch.sigmoid(logits).cpu()
            for index, image_path in enumerate(batch["image_path"]):
                categories = classify_false_positives(probability[index : index + 1], batch["mask"][index : index + 1])
                records.append(_build_record(image_path, categories))

    summary = summarize_records(records)
    summary["checkpoint"] = str(checkpoint)
    summary["config"] = str(Path(args.config))
    summary["split"] = "val"
    output_dir = Path(args.output_dir) if args.output_dir is not None else checkpoint.parent / "val_fp_audit"
    write_report(output_dir, records, summary)
    print(f"Validation FP audit | threshold: {FIXED_THRESHOLD:.1f} | images: {summary['total_images']} | FP pixels: {summary['total_false_positive_pixels']}")
    for name in CATEGORY_NAMES:
        category = summary["categories"][name]
        print(
            f"{name}: {category['pixels']} pixels | share {category['share_of_all_false_positives']:.4f} | "
            f"mean {category['per_image_ratio_mean']:.4f} | median {category['per_image_ratio_median']:.4f}"
        )
    gate = summary["gate"]
    print(f"MFMPT100_PRETRAIN_GATE: {gate['decision']} | far_texture_top2_image_share: {gate['far_texture_top2_image_share']:.4f}")
    print(f"Report: {output_dir / 'summary.json'}")
    return 0 if gate["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
