from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.structural_targets import build_structural_targets, require_scikit_image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Precompute cached edge and tangent targets")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--threshold", type=int, default=127)
    parser.add_argument("--target_dir_name", default="targets")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_scikit_image()
    root = Path(args.data_root)
    total = 0
    for split in ("train", "val", "test"):
        mask_dir = root / split / "masks"
        target_dir = root / split / args.target_dir_name
        if not mask_dir.exists():
            raise FileNotFoundError(f"Missing mask directory: {mask_dir}")
        target_dir.mkdir(parents=True, exist_ok=True)
        masks = sorted(path for path in mask_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)
        for mask_path in masks:
            output_path = target_dir / f"{mask_path.stem}.npz"
            if output_path.exists() and not args.overwrite:
                continue
            mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
            np.savez_compressed(output_path, **build_structural_targets(mask, args.threshold))
            total += 1
        print(f"{split}: {len(masks)} masks cached under {target_dir}")
    print(f"Generated {total} structural target files.")


if __name__ == "__main__":
    main()
