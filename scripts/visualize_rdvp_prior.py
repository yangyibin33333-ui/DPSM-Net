"""Render R-DVP priors using the same preprocessed DeepCrack samples as training."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

# Support direct ``python scripts/visualize_rdvp_prior.py`` execution from any
# working directory without requiring users to set PYTHONPATH manually.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import CrackSegmentationDataset
from models.blocks import retinex_directional_valley_maps
from utils.visualize import denormalize_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Visualize Retinex-Directional Dark-Valley priors")
    parser.add_argument("--data_root", default="DeepCrack")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--img_size", type=int, nargs=2, default=(384, 544), metavar=("H", "W"))
    parser.add_argument("--output_dir", default="runs/diagnostics/rdvp_native")
    parser.add_argument("--num_samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stems", nargs="*", default=None)
    return parser.parse_args()


def heatmap(value: np.ndarray) -> np.ndarray:
    """A small dependency-free blue-cyan-yellow heatmap for nonnegative priors."""
    high = float(value.max())
    normalized = np.zeros_like(value, dtype=np.float32) if high <= 0.0 else np.clip(value / high, 0.0, 1.0)
    return np.stack(
        [
            np.clip(255.0 * (2.0 * normalized - 0.5), 0.0, 255.0),
            np.clip(255.0 * (1.5 - np.abs(2.0 * normalized - 1.0)), 0.0, 255.0),
            np.clip(255.0 * (1.0 - 1.5 * normalized), 0.0, 255.0),
        ],
        axis=-1,
    ).astype(np.uint8)


def confidence_heatmap(value: np.ndarray) -> np.ndarray:
    """Render confidence with a fixed [0, 1] range; zero stays dark."""
    normalized = np.clip(value.astype(np.float32, copy=False), 0.0, 1.0)
    return np.stack(
        [
            255.0 * normalized,
            255.0 * np.sqrt(normalized),
            48.0 * (1.0 - normalized),
        ],
        axis=-1,
    ).astype(np.uint8)


def panel(image: np.ndarray, label: str) -> Image.Image:
    rendered = Image.fromarray(image)
    canvas = Image.new("RGB", (rendered.width, rendered.height + 24), "white")
    canvas.paste(rendered, (0, 24))
    ImageDraw.Draw(canvas).text((6, 5), label, fill="black")
    return canvas


def save_four_panel(
    image: torch.Tensor,
    mask: torch.Tensor,
    valley_strength: torch.Tensor,
    directional_coherence: torch.Tensor,
    path: Path,
) -> None:
    rgb = denormalize_image(image)
    target = (mask.detach().cpu().squeeze(0).numpy() > 0.5).astype(np.uint8) * 255
    target_rgb = np.repeat(target[..., None], 3, axis=2)
    valley = heatmap(valley_strength.detach().cpu().squeeze().numpy())
    coherence = confidence_heatmap(directional_coherence.detach().cpu().squeeze().numpy())
    panels = [
        panel(rgb, "Input RGB"),
        panel(target_rgb, "GT mask"),
        panel(valley, "Valley strength"),
        panel(coherence, "Directional confidence (gated)"),
    ]
    width = sum(item.width for item in panels)
    height = max(item.height for item in panels)
    grid = Image.new("RGB", (width, height), "white")
    offset = 0
    for item in panels:
        grid.paste(item, (offset, 0))
        offset += item.width
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def resolve_requested_stems(available_stems: set[str], requested_stems: list[str]) -> set[str]:
    """Accept exact stems and common split-prefixed short forms such as ``11204``."""
    resolved: set[str] = set()
    missing: list[str] = []
    ambiguous: list[str] = []
    for requested in requested_stems:
        if requested in available_stems:
            resolved.add(requested)
            continue
        matches = sorted(stem for stem in available_stems if stem.endswith(f"_{requested}"))
        if len(matches) == 1:
            resolved.add(matches[0])
        elif not matches:
            missing.append(requested)
        else:
            ambiguous.append(f"{requested} -> {', '.join(matches)}")
    if missing:
        preview = ", ".join(sorted(available_stems)[:12])
        raise ValueError(f"Requested stems are not available: {', '.join(missing)}. Available examples: {preview}")
    if ambiguous:
        raise ValueError(f"Requested stems are ambiguous: {'; '.join(ambiguous)}")
    return resolved


def main() -> None:
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("--num_samples must be at least 1.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = CrackSegmentationDataset(
        args.data_root,
        split=args.split,
        img_size=tuple(args.img_size),
        augment=False,
    )
    available_stems = {image_path.stem for image_path, _ in dataset.pairs}
    wanted_stems = resolve_requested_stems(available_stems, args.stems) if args.stems else set()
    saved = 0
    for index in range(len(dataset)):
        stem = dataset.pairs[index][0].stem
        if wanted_stems and stem not in wanted_stems:
            continue
        sample = dataset[index]
        image = sample["image"]
        mask = sample["mask"]
        if not isinstance(image, torch.Tensor) or not isinstance(mask, torch.Tensor):
            raise TypeError("CrackSegmentationDataset must provide tensor image and mask values.")
        with torch.no_grad():
            valley_strength, directional_coherence = retinex_directional_valley_maps(image.unsqueeze(0))
        save_four_panel(image, mask, valley_strength[0], directional_coherence[0], Path(args.output_dir) / f"{stem}_rdvp_prior.png")
        saved += 1
        if saved >= args.num_samples and not wanted_stems:
            break
    if wanted_stems:
        missing = sorted(wanted_stems - available_stems)
        if missing:
            raise ValueError(f"Requested stems are not available in split '{args.split}': {', '.join(missing)}")
    if saved == 0:
        raise RuntimeError("No samples were selected for R-DVP visualization.")
    print(f"Saved {saved} R-DVP diagnostic panels to {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
