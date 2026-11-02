from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from models import DEMambaNet
from utils import load_checkpoint
from utils.visualize import save_mask, save_overlay


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args():
    parser = argparse.ArgumentParser("Infer with DE-MambaNet")
    parser.add_argument("--config", default="configs/demambanet.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default="runs/demambanet/infer")
    parser.add_argument("--img_size", type=int, nargs=2, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def preprocess(path: Path, img_size: list[int] | tuple[int, int] | None):
    original = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    height, width = original.shape[:2]
    if img_size is None:
        model_image, metadata = original, (height, width, height, width, 0, 0)
    else:
        target_h, target_w = img_size
        scale = min(target_h / height, target_w / width)
        resized_h, resized_w = max(1, round(height * scale)), max(1, round(width * scale))
        top, left = (target_h - resized_h) // 2, (target_w - resized_w) // 2
        resized = np.asarray(Image.fromarray(original).resize((resized_w, resized_h), Image.BILINEAR))
        model_image = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        model_image[top : top + resized_h, left : left + resized_w] = resized
        metadata = (height, width, resized_h, resized_w, top, left)
    image_np = model_image.astype(np.float32) / 255.0
    tensor = torch.from_numpy(image_np).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0), original, metadata


def restore_mask(probability: np.ndarray, metadata: tuple[int, int, int, int, int, int]) -> np.ndarray:
    original_h, original_w, resized_h, resized_w, top, left = metadata
    cropped = probability[top : top + resized_h, left : left + resized_w]
    if cropped.shape == (original_h, original_w):
        return cropped
    return np.asarray(
        Image.fromarray(cropped.astype(np.float32), mode="F").resize((original_w, original_h), Image.BILINEAR),
        dtype=np.float32,
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    img_size = args.img_size or cfg["data"].get("img_size")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = dict(cfg["model"])
    model_cfg.pop("name", None)
    model = DEMambaNet(**model_cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    with torch.no_grad():
        for image_path in list_images(Path(args.input)):
            tensor, image_np, metadata = preprocess(image_path, img_size)
            outputs = model(tensor.to(device))
            prob = restore_mask(torch.sigmoid(outputs["mask"])[0, 0].cpu().numpy(), metadata)
            mask = prob > args.threshold
            save_mask(mask, output_dir / f"{image_path.stem}_mask.png")
            save_overlay(image_np, mask, output_dir / f"{image_path.stem}_overlay.png")
            print(f"Saved inference results for {image_path}")


if __name__ == "__main__":
    main()
