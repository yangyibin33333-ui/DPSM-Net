from __future__ import annotations

import argparse
from decimal import Decimal
from pathlib import Path

import torch
import yaml

from datasets import build_dataloader
from metrics import SegmentationMetric
from models import build_model
from utils import load_checkpoint
from utils.visualize import save_diagnostic_bundle, save_prediction_triplet, write_diagnostics_readme


def parse_args():
    parser = argparse.ArgumentParser("Evaluate DE-MambaNet")
    parser.add_argument("--config", default="configs/demambanet.yaml")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--img_size", type=int, nargs=2, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save_pred", action="store_true")
    parser.add_argument("--pred_dir", default=None)
    parser.add_argument("--save_diagnostics", action="store_true")
    parser.add_argument("--diagnostic_dir", default=None)
    parser.add_argument("--diagnostic_stems", nargs="+", default=None, metavar="STEM")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threshold_sweep", type=float, nargs=3, metavar=("START", "END", "STEP"), default=None)
    args = parser.parse_args()
    if args.threshold_sweep is not None and args.split != "val":
        parser.error("--threshold_sweep is only allowed with --split val")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be in [0, 1]")
    return args


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_threshold_sweep(start: float, end: float, step: float) -> list[float]:
    """Build an inclusive threshold range without accumulating float error."""
    start_decimal, end_decimal, step_decimal = (Decimal(str(value)) for value in (start, end, step))
    if step_decimal <= 0:
        raise ValueError("threshold_sweep STEP must be greater than 0")
    if start_decimal < 0 or end_decimal > 1 or start_decimal > end_decimal:
        raise ValueError("threshold_sweep must satisfy 0 <= START <= END <= 1")

    thresholds: list[float] = []
    threshold = start_decimal
    while threshold <= end_decimal:
        thresholds.append(float(threshold))
        threshold += step_decimal
    if thresholds[-1] != float(end_decimal):
        thresholds.append(float(end_decimal))
    return thresholds


def update_threshold_sweep(
    sweep: dict[float, list[torch.Tensor]], logits: torch.Tensor, target: torch.Tensor
) -> None:
    probability = torch.sigmoid(logits.detach()).cpu()
    binary_target = target.detach().cpu() > 0.5
    for threshold, counts in sweep.items():
        prediction = probability > threshold
        counts[0] += (prediction & binary_target).sum().float()
        counts[1] += (prediction & ~binary_target).sum().float()
        counts[2] += (~prediction & binary_target).sum().float()


def summarize_threshold_sweep(sweep: dict[float, list[torch.Tensor]]) -> list[tuple[float, dict[str, float]]]:
    results: list[tuple[float, dict[str, float]]] = []
    eps = 1e-7
    for threshold, (tp, fp, fn) in sweep.items():
        iou = tp / (tp + fp + fn + eps)
        f1 = 2.0 * tp / (2.0 * tp + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        results.append(
            (
                threshold,
                {
                    "IoU": float(iou.item()),
                    "F1": float(f1.item()),
                    "Precision": float(precision.item()),
                    "Recall": float(recall.item()),
                },
            )
        )
    return results


def main():
    args = parse_args()
    sweep_thresholds = build_threshold_sweep(*args.threshold_sweep) if args.threshold_sweep is not None else None
    cfg = load_config(args.config)
    if args.data_root is not None:
        cfg["data"]["data_root"] = args.data_root
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.img_size is not None:
        cfg["data"]["img_size"] = args.img_size
    if args.num_workers is not None:
        cfg["train"]["num_workers"] = args.num_workers

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = Path(args.checkpoint or Path(cfg["save"]["save_dir"]) / "best_structural_score.pth")
    pred_dir = Path(args.pred_dir or Path(cfg["save"]["save_dir"]) / "predictions")
    diagnostic_dir = Path(args.diagnostic_dir) if args.diagnostic_dir is not None else pred_dir / "diagnostics"
    diagnostic_stems = set(args.diagnostic_stems or [])
    if args.save_diagnostics:
        write_diagnostics_readme(diagnostic_dir)

    model_cfg = dict(cfg["model"])
    model_name = str(model_cfg.pop("name", "DEMambaNet"))
    model = build_model(model_name, **model_cfg).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    model.eval()

    loader = build_dataloader(
        cfg["data"]["data_root"],
        args.split,
        cfg["data"]["img_size"],
        int(cfg["data"]["threshold"]),
        int(cfg["train"]["batch_size"]),
        int(cfg["train"]["num_workers"]),
        shuffle=False,
        target_dir_name=cfg["data"].get("target_dir_name", "targets"),
        persistent_workers=bool(cfg["train"].get("persistent_workers", False)),
        prefetch_factor=int(cfg["train"].get("prefetch_factor", 2)),
    )

    metric = SegmentationMetric(primary_threshold=args.threshold)
    threshold_sweep = (
        {threshold: [torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)] for threshold in sweep_thresholds}
        if sweep_thresholds is not None
        else None
    )
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            outputs = model(images)
            metric.update(outputs["mask"], masks)
            if threshold_sweep is not None:
                update_threshold_sweep(threshold_sweep, outputs["mask"], masks)
            if args.save_pred:
                for i, path in enumerate(batch["image_path"]):
                    stem = Path(path).stem
                    save_prediction_triplet(batch["image"][i], outputs["mask"][i], stem, pred_dir, threshold=args.threshold)
            if args.save_diagnostics:
                for i, path in enumerate(batch["image_path"]):
                    stem = Path(path).stem
                    if diagnostic_stems and stem not in diagnostic_stems:
                        continue
                    save_diagnostic_bundle(
                        batch["image"][i],
                        batch["mask"][i],
                        outputs["mask"][i],
                        outputs["edge"][i],
                        stem,
                        diagnostic_dir,
                        threshold=args.threshold,
                        route_s3=outputs.get("route_s3", None)[i] if outputs.get("route_s3", None) is not None else None,
                        route_s4=outputs.get("route_s4", None)[i] if outputs.get("route_s4", None) is not None else None,
                    )

    results = metric.compute()
    for key in ["IoU", "mIoU", "Dice", "F1", "Precision", "Recall", "ODS", "OIS", "BoundaryF1", "clDice", "ComponentError"]:
        print(f"{key}: {results[key]:.4f}")
    weights = cfg.get("evaluation", {}).get("structural_score_weights", {})
    score = (
        float(weights.get("f1", 0.5)) * results["F1"]
        + float(weights.get("cldice", 0.3)) * results["clDice"]
        + float(weights.get("boundary_f1", 0.2)) * results["BoundaryF1"]
    )
    print(f"structural_score: {score:.4f}")
    if threshold_sweep is not None:
        best_threshold = 0.0
        best_f1 = float("-inf")
        for threshold, metrics in summarize_threshold_sweep(threshold_sweep):
            print(
                f"Threshold: {threshold:.4f} | IoU: {metrics['IoU']:.4f} | F1: {metrics['F1']:.4f} "
                f"| Precision: {metrics['Precision']:.4f} | Recall: {metrics['Recall']:.4f}"
            )
            if metrics["F1"] >= best_f1:
                best_threshold, best_f1 = threshold, metrics["F1"]
        print(f"BEST_THRESHOLD: {best_threshold:.4f}")
        print(f"BEST_VAL_F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
