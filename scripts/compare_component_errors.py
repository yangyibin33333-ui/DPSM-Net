"""Compare fixed-threshold Native and EMA predictions for component regressions.

This is a read-only test diagnostic.  It deliberately mirrors ``test.py`` for
configuration, checkpoint, dataset, and official metric loading, while fixing
the binary threshold at 0.5 to keep the comparison on the experiment protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

# Running ``python scripts/compare_component_errors.py`` makes ``scripts`` the
# import root.  Add the repository root so this uses the same project modules
# as ``test.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import build_dataloader
from datasets.crack_dataset import _letterbox_geometry
from metrics import SegmentationMetric, binary_metrics_from_logits
from metrics.metrics import _topology_scores
from models import DEMambaNet
from utils import load_checkpoint


FIXED_IMAGE_SIZE = (384, 544)
FIXED_THRESHOLD = 0.5
METRIC_KEYS = (
    "mIoU",
    "F1",
    "Precision",
    "Recall",
    "ODS",
    "OIS",
    "BoundaryF1",
    "clDice",
    "ComponentError",
)
CSV_FIELDS = (
    "image_name",
    "image_path",
    "native_f1",
    "ema_f1",
    "delta_f1",
    "native_cldice",
    "ema_cldice",
    "delta_cldice",
    "native_component_error",
    "ema_component_error",
    "delta_component_error",
    "native_foreground_pixels",
    "ema_foreground_pixels",
    "ema_only_foreground_pixels",
    "native_only_foreground_pixels",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Compare Native and EMA ComponentError on DeepCrack test")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--native_config", required=True)
    parser.add_argument("--native_checkpoint", required=True)
    parser.add_argument("--ema_config", required=True)
    parser.add_argument("--ema_checkpoint", required=True)
    parser.add_argument("--img_size", type=int, nargs=2, default=list(FIXED_IMAGE_SIZE), metavar=("H", "W"))
    parser.add_argument("--threshold", type=float, default=FIXED_THRESHOLD)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    validate_fixed_protocol(args.img_size, args.threshold, args.top_k)
    return args


def validate_fixed_protocol(img_size: list[int] | tuple[int, int], threshold: float, top_k: int) -> None:
    if tuple(img_size) != FIXED_IMAGE_SIZE:
        raise ValueError(f"This diagnostic is fixed to --img_size {FIXED_IMAGE_SIZE[0]} {FIXED_IMAGE_SIZE[1]}.")
    if float(threshold) != FIXED_THRESHOLD:
        raise ValueError("This diagnostic is fixed to --threshold 0.5; threshold tuning is not allowed.")
    if top_k < 1:
        raise ValueError("--top_k must be at least 1.")


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_config(cfg: dict[str, Any], label: str) -> None:
    if not bool(cfg.get("model", {}).get("use_mamba_ssm", False)):
        raise ValueError(f"{label} config must set model.use_mamba_ssm: true for this formal diagnostic.")
    if tuple(cfg.get("data", {}).get("img_size", ())) != FIXED_IMAGE_SIZE:
        raise ValueError(f"{label} config must set data.img_size: [{FIXED_IMAGE_SIZE[0]}, {FIXED_IMAGE_SIZE[1]}].")


def build_model(cfg: dict[str, Any], checkpoint: str | Path, device: torch.device) -> torch.nn.Module:
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    model_cfg = dict(cfg["model"])
    model_cfg.pop("name", None)
    model = DEMambaNet(**model_cfg).to(device)
    # The standard ``model`` state is intentionally loaded, exactly as test.py
    # does.  EMA runs save their shadow weights in that standard field.
    load_checkpoint(checkpoint, model, map_location=device)
    return model.eval()


def validate_mamba_backends(model: torch.nn.Module, label: str) -> None:
    backend_info = model.mamba_backend_info()  # type: ignore[attr-defined]
    invalid: dict[str, list[str]] = {}
    for stage in ("stage3", "stage4"):
        backends = backend_info.get(stage)
        if backends is None or len(backends) != 4 or any(backend != "mamba_ssm" for backend in backends):
            invalid[stage] = [] if backends is None else backends
    if invalid:
        detail = "; ".join(f"{stage}: {', '.join(backends)}" for stage, backends in invalid.items())
        raise RuntimeError(f"{label} model is not using mamba_ssm for every stage3/4 route: {detail}")


def build_test_loader(cfg: dict[str, Any], data_root: str | Path):
    return build_dataloader(
        data_root,
        "test",
        cfg["data"]["img_size"],
        int(cfg["data"]["threshold"]),
        int(cfg["train"]["batch_size"]),
        int(cfg["train"]["num_workers"]),
        shuffle=False,
        target_dir_name=cfg["data"].get("target_dir_name", "targets"),
        persistent_workers=bool(cfg["train"].get("persistent_workers", False)),
        prefetch_factor=int(cfg["train"].get("prefetch_factor", 2)),
    )


def binary_prediction(probability: torch.Tensor) -> torch.Tensor:
    """Use the exact strict comparison used by SegmentationMetric."""
    return probability > FIXED_THRESHOLD


def comparison_masks(native_prediction: np.ndarray, ema_prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    native_prediction = native_prediction.astype(bool, copy=False)
    ema_prediction = ema_prediction.astype(bool, copy=False)
    return ema_prediction & ~native_prediction, native_prediction & ~ema_prediction


def make_image_record(
    image_name: str,
    image_path: str,
    native_f1: float,
    ema_f1: float,
    native_topology: tuple[float, float],
    ema_topology: tuple[float, float],
    native_prediction: np.ndarray,
    ema_prediction: np.ndarray,
) -> dict[str, Any]:
    """Create a serializable per-image record from official topology results."""
    ema_only, native_only = comparison_masks(native_prediction, ema_prediction)
    native_cldice, native_component_error = native_topology
    ema_cldice, ema_component_error = ema_topology
    return {
        "image_name": image_name,
        "image_path": image_path,
        "native_f1": float(native_f1),
        "ema_f1": float(ema_f1),
        "delta_f1": float(ema_f1 - native_f1),
        "native_cldice": float(native_cldice),
        "ema_cldice": float(ema_cldice),
        "delta_cldice": float(ema_cldice - native_cldice),
        "native_component_error": float(native_component_error),
        "ema_component_error": float(ema_component_error),
        "delta_component_error": float(ema_component_error - native_component_error),
        "native_foreground_pixels": int(native_prediction.sum()),
        "ema_foreground_pixels": int(ema_prediction.sum()),
        "ema_only_foreground_pixels": int(ema_only.sum()),
        "native_only_foreground_pixels": int(native_only.sum()),
    }


def rank_component_regressions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda record: (-float(record["delta_component_error"]), str(record["image_name"])))


def select_positive_regressions(records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    return [record for record in rank_component_regressions(records) if float(record["delta_component_error"]) > 0.0][:top_k]


def write_records(path: str | Path, records: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def restore_model_map(
    value: np.ndarray,
    image_path: str | Path,
    is_probability: bool,
) -> np.ndarray:
    """Remove letterbox padding and resize a model-space map to the raw image."""
    with Image.open(image_path) as raw_image:
        raw_width, raw_height = raw_image.size
    resized_height, resized_width, top, left = _letterbox_geometry(raw_height, raw_width, FIXED_IMAGE_SIZE)
    crop = np.ascontiguousarray(value[top : top + resized_height, left : left + resized_width])
    if is_probability:
        return np.asarray(
            Image.fromarray(crop.astype(np.float32), mode="F").resize((raw_width, raw_height), Image.BILINEAR),
            dtype=np.float32,
        )
    restored = Image.fromarray((crop.astype(bool) * 255).astype(np.uint8)).resize((raw_width, raw_height), Image.NEAREST)
    return np.asarray(restored, dtype=np.uint8) > 127


def _mask_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask.astype(bool) * 255).astype(np.uint8)).convert("RGB")


def _change_image(mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    value = np.zeros((*mask.shape, 3), dtype=np.uint8)
    value[mask.astype(bool)] = color
    return Image.fromarray(value, mode="RGB")


def _probability_difference_image(difference: np.ndarray) -> Image.Image:
    """A fixed [-1, 1] blue-white-red map shared by every exported image."""
    difference = np.clip(difference.astype(np.float32), -1.0, 1.0)
    negative = np.clip(-difference, 0.0, 1.0)
    positive = np.clip(difference, 0.0, 1.0)
    rgb = np.empty((*difference.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.round(255.0 * (1.0 - negative)).astype(np.uint8)
    rgb[..., 1] = np.round(255.0 * (1.0 - negative - positive)).astype(np.uint8)
    rgb[..., 2] = np.round(255.0 * (1.0 - positive)).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _panel(title: str, image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    title_height = 24
    panel = Image.new("RGB", (image.width, image.height + title_height), "white")
    ImageDraw.Draw(panel).text((4, 4), title, fill="black")
    panel.paste(image, (0, title_height))
    return panel


def _text_panel(size: tuple[int, int], record: dict[str, Any]) -> Image.Image:
    width, height = size
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)
    lines = (
        "Fixed threshold: probability > 0.5",
        f"Native | CE {record['native_component_error']:.4f} | clDice {record['native_cldice']:.4f} | F1 {record['native_f1']:.4f}",
        f"EMA    | CE {record['ema_component_error']:.4f} | clDice {record['ema_cldice']:.4f} | F1 {record['ema_f1']:.4f}",
        f"Delta  | CE {record['delta_component_error']:+.4f} | clDice {record['delta_cldice']:+.4f} | F1 {record['delta_f1']:+.4f}",
        "EMA-only foreground: red",
        "Native-only foreground: blue",
        "Probability difference: EMA - Native, fixed [-1, 1]",
    )
    y = 8
    for line in lines:
        draw.text((8, y), line, fill="black")
        y += 22
    return panel


def save_visualization(
    output_path: str | Path,
    record: dict[str, Any],
    native_probability: np.ndarray,
    ema_probability: np.ndarray,
    target: np.ndarray,
) -> None:
    """Save a raw-image-coordinate 4x2 panel without letterbox padding."""
    image_path = record["image_path"]
    with Image.open(image_path) as source:
        raw_image = source.convert("RGB")
    native_prediction = restore_model_map(
        binary_prediction(torch.from_numpy(native_probability)).numpy(), image_path, is_probability=False
    )
    ema_prediction = restore_model_map(
        binary_prediction(torch.from_numpy(ema_probability)).numpy(), image_path, is_probability=False
    )
    native_probability = restore_model_map(native_probability, image_path, is_probability=True)
    ema_probability = restore_model_map(ema_probability, image_path, is_probability=True)
    target = restore_model_map(target, image_path, is_probability=False)
    ema_only, native_only = comparison_masks(native_prediction, ema_prediction)

    panels = (
        _panel("Original RGB", raw_image),
        _panel("Ground Truth", _mask_image(target)),
        _panel("Native prediction", _mask_image(native_prediction)),
        _panel("EMA prediction", _mask_image(ema_prediction)),
        _panel("EMA-only foreground (red)", _change_image(ema_only, (255, 0, 0))),
        _panel("Native-only foreground (blue)", _change_image(native_only, (0, 0, 255))),
        _panel("EMA probability - Native probability [-1, 1]", _probability_difference_image(ema_probability - native_probability)),
        _panel("Metrics summary", _text_panel(raw_image.size, record)),
    )
    width, height = panels[0].size
    canvas = Image.new("RGB", (width * 4, height * 2), "white")
    for index, panel in enumerate(panels):
        canvas.paste(panel, ((index % 4) * width, (index // 4) * height))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _summary_metrics(metric: SegmentationMetric) -> dict[str, float]:
    result = metric.compute()
    return {key: float(result[key]) for key in METRIC_KEYS}


def _cache_top_visualization(
    cache: dict[str, dict[str, Any]],
    record: dict[str, Any],
    visual_data: dict[str, np.ndarray],
    top_k: int,
) -> None:
    """Retain only current top positive records, avoiding a full-test tensor cache."""
    if float(record["delta_component_error"]) <= 0.0:
        return
    cache[record["image_name"]] = {"record": record, **visual_data}
    retained = select_positive_regressions([item["record"] for item in cache.values()], top_k)
    retained_names = {item["image_name"] for item in retained}
    for image_name in list(cache):
        if image_name not in retained_names:
            del cache[image_name]


def run_comparison(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    native_cfg = load_config(args.native_config)
    ema_cfg = load_config(args.ema_config)
    validate_config(native_cfg, "Native")
    validate_config(ema_cfg, "EMA")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    native_model = build_model(native_cfg, args.native_checkpoint, device)
    ema_model = build_model(ema_cfg, args.ema_checkpoint, device)
    validate_mamba_backends(native_model, "Native")
    validate_mamba_backends(ema_model, "EMA")
    loader = build_test_loader(native_cfg, args.data_root)
    native_metric = SegmentationMetric(primary_threshold=FIXED_THRESHOLD)
    ema_metric = SegmentationMetric(primary_threshold=FIXED_THRESHOLD)
    records: list[dict[str, Any]] = []
    visualization_cache: dict[str, dict[str, Any]] = {}

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            target = batch["mask"].to(device)
            native_logits = native_model(images)["mask"]
            ema_logits = ema_model(images)["mask"]
            native_metric.update(native_logits, target)
            ema_metric.update(ema_logits, target)
            native_probability = torch.sigmoid(native_logits).cpu()
            ema_probability = torch.sigmoid(ema_logits).cpu()
            target_cpu = target.detach().cpu()
            for index, image_path in enumerate(batch["image_path"]):
                native_prediction = binary_prediction(native_probability[index, 0]).numpy()
                ema_prediction = binary_prediction(ema_probability[index, 0]).numpy()
                target_mask = (target_cpu[index, 0] > 0.5).numpy()
                native_f1 = binary_metrics_from_logits(
                    native_logits[index : index + 1].detach().cpu(), target_cpu[index : index + 1], threshold=FIXED_THRESHOLD
                )["F1"]
                ema_f1 = binary_metrics_from_logits(
                    ema_logits[index : index + 1].detach().cpu(), target_cpu[index : index + 1], threshold=FIXED_THRESHOLD
                )["F1"]
                record = make_image_record(
                    Path(image_path).name,
                    str(image_path),
                    native_f1,
                    ema_f1,
                    _topology_scores(native_prediction, target_mask),
                    _topology_scores(ema_prediction, target_mask),
                    native_prediction,
                    ema_prediction,
                )
                records.append(record)
                _cache_top_visualization(
                    visualization_cache,
                    record,
                    {
                        "native_probability": native_probability[index, 0].numpy(),
                        "ema_probability": ema_probability[index, 0].numpy(),
                        "target": target_cpu[index, 0].numpy(),
                    },
                    args.top_k,
                )

    native_summary = _summary_metrics(native_metric)
    ema_summary = _summary_metrics(ema_metric)
    summary = {
        "threshold": FIXED_THRESHOLD,
        "image_size": list(FIXED_IMAGE_SIZE),
        "native": native_summary,
        "ema": ema_summary,
        "component_error_delta": ema_summary["ComponentError"] - native_summary["ComponentError"],
        "positive_component_error_images": sum(float(record["delta_component_error"]) > 0.0 for record in records),
    }
    selected = select_positive_regressions(records, args.top_k)
    visualizations = [visualization_cache[record["image_name"]] for record in selected]
    return records, summary, visualizations


def write_outputs(
    output_dir: str | Path,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    visualizations: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "visualizations").mkdir(exist_ok=True)
    ranked_records = rank_component_regressions(records)
    selected = select_positive_regressions(ranked_records, top_k)
    write_records(output_dir / "all_images.csv", ranked_records)
    write_records(output_dir / "top_component_regressions.csv", selected)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    visualization_by_name = {item["record"]["image_name"]: item for item in visualizations}
    for rank, record in enumerate(selected, start=1):
        if record["image_name"] not in visualization_by_name:
            continue
        item = visualization_by_name[record["image_name"]]
        original_name = Path(record["image_name"]).name.replace("/", "_").replace("\\", "_")
        filename = (
            f"{rank:02d}_{original_name}_native_ce_{record['native_component_error']:.4f}"
            f"_ema_ce_{record['ema_component_error']:.4f}.png"
        )
        save_visualization(
            output_dir / "visualizations" / filename,
            record,
            item["native_probability"],
            item["ema_probability"],
            item["target"],
        )
    return selected


def print_summary(summary: dict[str, Any], selected: list[dict[str, Any]], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    print(f"Native global ComponentError: {summary['native']['ComponentError']:.4f}")
    print(f"EMA global ComponentError: {summary['ema']['ComponentError']:.4f}")
    print(f"Delta ComponentError: {summary['component_error_delta']:+.4f}")
    print(f"Positive-delta images: {summary['positive_component_error_images']}")
    print(f"Saved CSV: {output_dir / 'all_images.csv'}")
    print(f"Saved visualizations: {output_dir / 'visualizations'}")
    print("rank | image_name | native_component_error | ema_component_error | delta")
    for rank, record in enumerate(selected[:10], start=1):
        print(
            f"{rank:>4d} | {record['image_name']} | {record['native_component_error']:.4f} | "
            f"{record['ema_component_error']:.4f} | {record['delta_component_error']:+.4f}"
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    records, summary, visualizations = run_comparison(args)
    selected = write_outputs(args.output_dir, records, summary, visualizations, args.top_k)
    print_summary(summary, selected, args.output_dir)


if __name__ == "__main__":
    main()
