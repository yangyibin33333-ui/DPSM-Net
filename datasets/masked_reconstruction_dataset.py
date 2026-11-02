from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .crack_dataset import IMAGE_EXTS, IMAGENET_MEAN, IMAGENET_STD, _letterbox_image


class TrainImageReconstructionDataset(Dataset):
    """Unlabeled train-image view for the train-only reconstruction pretraining."""

    def __init__(self, data_root: str | Path, img_size: tuple[int, int] | list[int] = (384, 544)):
        self.root = Path(data_root)
        self.img_size = tuple(int(value) for value in img_size)
        if self.img_size != (384, 544):
            raise ValueError("Masked reconstruction pretraining requires img_size=(384, 544).")
        self.image_dir = self.root / "train" / "images"
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Missing train image directory: {self.image_dir}")
        self.image_paths = sorted(
            path for path in self.image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        )
        if not self.image_paths:
            raise RuntimeError(f"No train images found under {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path = self.image_paths[index]
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        image = _letterbox_image(image, self.img_size)
        target_rgb = torch.from_numpy(np.ascontiguousarray(image).astype(np.float32) / 255.0).permute(2, 0, 1)
        normalized_image = (target_rgb - IMAGENET_MEAN) / IMAGENET_STD
        return {
            "image": normalized_image.float(),
            "target_rgb": target_rgb.float(),
            "image_path": str(image_path),
        }


def build_train_image_reconstruction_dataloader(
    data_root: str | Path,
    batch_size: int,
    num_workers: int,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
) -> DataLoader:
    """Build the only allowed pretraining loader: DeepCrack train/images."""

    dataset = TrainImageReconstructionDataset(data_root, img_size=(384, 544))
    kwargs: dict[str, object] = {}
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=True,
        drop_last=True,
        **kwargs,
    )
