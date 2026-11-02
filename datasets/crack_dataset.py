from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from utils.structural_targets import build_truncated_sdf_targets


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
MASK_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def build_component_id_map(mask: np.ndarray) -> np.ndarray:
    """Label final binary masks with 8-connectivity without persisting a new target."""
    if mask.ndim != 2:
        raise ValueError("component target masks must have shape [H, W].")
    binary = np.ascontiguousarray(mask > 0.5)
    try:
        from scipy import ndimage

        labels, _ = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    except ImportError:
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError(
                "Component-query supervision requires scipy.ndimage or OpenCV for 8-connected components."
            ) from error
        _, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    return np.ascontiguousarray(labels.astype(np.int64, copy=False))


def scan_train_component_counts(
    data_root: str | Path,
    img_size: tuple[int, int] | list[int] | None,
    threshold: int,
) -> list[tuple[str, int]]:
    """Audit component counts in final letterbox coordinates before component-query training."""
    root = Path(data_root)
    counts: list[tuple[str, int]] = []
    for _, mask_path in _find_pairs(root, "train"):
        raw_mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        mask = (raw_mask > threshold).astype(np.float32)
        final_mask = _letterbox_target(mask[None], img_size, "nearest")[0]
        counts.append((mask_path.stem, int(build_component_id_map(final_mask).max())))
    return counts


def _find_pairs(root: Path, split: str) -> list[tuple[Path, Path]]:
    image_dir, mask_dir = root / split / "images", root / split / "masks"
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Missing mask directory: {mask_dir}")
    masks = {path.stem: path for path in mask_dir.iterdir() if path.suffix.lower() in MASK_EXTS}
    pairs = [(image_path, masks[image_path.stem]) for image_path in sorted(image_dir.iterdir()) if image_path.suffix.lower() in IMAGE_EXTS and image_path.stem in masks]
    if not pairs:
        raise RuntimeError(f"No image/mask pairs found under {root / split}")
    return pairs


def _letterbox_geometry(height: int, width: int, size: tuple[int, int] | None) -> tuple[int, int, int, int]:
    if size is None:
        return height, width, 0, 0
    target_h, target_w = size
    scale = min(target_h / height, target_w / width)
    resized_h, resized_w = max(1, round(height * scale)), max(1, round(width * scale))
    return resized_h, resized_w, (target_h - resized_h) // 2, (target_w - resized_w) // 2


def _letterbox_image(image: np.ndarray, size: tuple[int, int] | None) -> np.ndarray:
    height, width = image.shape[:2]
    resized_h, resized_w, top, left = _letterbox_geometry(height, width, size)
    if size is None:
        return np.ascontiguousarray(image)
    resized = np.asarray(Image.fromarray(image).resize((resized_w, resized_h), Image.BILINEAR))
    canvas = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    canvas[top : top + resized_h, left : left + resized_w] = resized
    return canvas


def _letterbox_target(target: np.ndarray, size: tuple[int, int] | None, mode: str) -> np.ndarray:
    if size is None:
        return np.ascontiguousarray(target)
    height, width = target.shape[-2:]
    resized_h, resized_w, top, left = _letterbox_geometry(height, width, size)
    value = torch.from_numpy(np.ascontiguousarray(target)).float().unsqueeze(0)
    kwargs = {"size": (resized_h, resized_w), "mode": mode}
    if mode != "nearest":
        kwargs["align_corners"] = False
    value = F.interpolate(value, **kwargs)
    canvas = value.new_zeros((1, value.size(1), size[0], size[1]))
    canvas[:, :, top : top + resized_h, left : left + resized_w] = value
    return canvas.squeeze(0).numpy()


def _swap_diagonals(tangent: np.ndarray) -> np.ndarray:
    return tangent[[0, 1, 3, 2]]


def _rotate_route_bins(tangent: np.ndarray, turns: int) -> np.ndarray:
    if turns % 2:
        return tangent[[1, 0, 3, 2]]
    return tangent


def _augment(
    image: np.ndarray,
    mask: np.ndarray,
    edge: np.ndarray,
    tangent: np.ndarray,
    valid: np.ndarray,
    allow_quarter_turns: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        image, mask, edge, tangent, valid = image[:, ::-1], mask[:, ::-1], edge[:, ::-1], tangent[:, :, ::-1], valid[:, :, ::-1]
        tangent = _swap_diagonals(tangent)
    if random.random() < 0.5:
        image, mask, edge, tangent, valid = image[::-1], mask[::-1], edge[::-1], tangent[:, ::-1], valid[:, ::-1]
        tangent = _swap_diagonals(tangent)
    turns = random.randint(0, 3) if allow_quarter_turns else random.choice((0, 2))
    if turns:
        image = np.rot90(image, turns).copy()
        mask = np.rot90(mask, turns).copy()
        edge = np.rot90(edge, turns).copy()
        tangent = np.rot90(tangent, turns, axes=(1, 2)).copy()
        valid = np.rot90(valid, turns, axes=(1, 2)).copy()
        tangent = _rotate_route_bins(tangent, turns)
    return tuple(np.ascontiguousarray(value) for value in (image, mask, edge, tangent, valid))


def _native_scale_jitter(
    image: np.ndarray,
    mask: np.ndarray,
    edge: np.ndarray,
    tangent: np.ndarray,
    valid: np.ndarray,
    cfg: dict | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply centered isotropic scaling while preserving target semantics."""
    if cfg is None:
        return image, mask, edge, tangent, valid
    if not isinstance(cfg, dict):
        raise ValueError("scale_jitter must be a mapping or None.")
    probability = float(cfg.get("probability", 0.0))
    scale_range = cfg.get("scale_range", (1.0, 1.0))
    if not 0.0 <= probability <= 1.0:
        raise ValueError("scale_jitter.probability must be in [0, 1].")
    if not isinstance(scale_range, (list, tuple)) or len(scale_range) != 2:
        raise ValueError("scale_jitter.scale_range must contain exactly two values.")
    min_scale, max_scale = float(scale_range[0]), float(scale_range[1])
    if min_scale <= 0.0 or max_scale <= 0.0 or min_scale > max_scale:
        raise ValueError("scale_jitter.scale_range must contain positive values ordered min <= max.")
    if not bool(cfg.get("enabled", False)) or probability == 0.0 or random.random() >= probability:
        return image, mask, edge, tangent, valid

    scale = random.uniform(min_scale, max_scale)
    if scale == 1.0:
        return image, mask, edge, tangent, valid
    height, width = image.shape[:2]
    scaled_h, scaled_w = max(1, round(height * scale)), max(1, round(width * scale))

    image_scaled = np.asarray(Image.fromarray(image).resize((scaled_w, scaled_h), Image.BILINEAR))

    def resize_target(target: np.ndarray, mode: str) -> np.ndarray:
        value = torch.from_numpy(np.ascontiguousarray(target)).float().unsqueeze(0)
        kwargs = {"size": (scaled_h, scaled_w), "mode": mode}
        if mode != "nearest":
            kwargs["align_corners"] = False
        return F.interpolate(value, **kwargs).squeeze(0).numpy().astype(np.float32, copy=False)

    mask_scaled = resize_target(mask[None], "nearest")
    edge_scaled = resize_target(edge[None], "nearest")
    tangent_scaled = resize_target(tangent, "bilinear")
    valid_scaled = resize_target(valid, "nearest")
    if scale > 1.0:
        top, left = (scaled_h - height) // 2, (scaled_w - width) // 2
        image = image_scaled[top : top + height, left : left + width]
        mask = mask_scaled[:, top : top + height, left : left + width][0]
        edge = edge_scaled[:, top : top + height, left : left + width][0]
        tangent = tangent_scaled[:, top : top + height, left : left + width]
        valid = valid_scaled[:, top : top + height, left : left + width]
    else:
        top, left = (height - scaled_h) // 2, (width - scaled_w) // 2
        bottom, right = height - scaled_h - top, width - scaled_w - left
        image = np.pad(image_scaled, ((top, bottom), (left, right), (0, 0)), mode="reflect")

        def center_target(target: np.ndarray) -> np.ndarray:
            canvas = np.zeros((target.shape[0], height, width), dtype=np.float32)
            canvas[:, top : top + scaled_h, left : left + scaled_w] = target
            return canvas

        mask = center_target(mask_scaled)[0]
        edge = center_target(edge_scaled)[0]
        tangent = center_target(tangent_scaled)
        valid = center_target(valid_scaled)
    return tuple(
        np.ascontiguousarray(value.astype(dtype, copy=False))
        for value, dtype in zip(
            (image, mask, edge, tangent, valid),
            (np.uint8, np.float32, np.float32, np.float32, np.float32),
        )
    )


def _photometric_augment(image: np.ndarray, cfg: dict | None) -> np.ndarray:
    """Apply configured RGB-only photometric changes without touching labels."""
    if not cfg or not bool(cfg.get("enabled", False)):
        return image

    probability = float(cfg.get("probability", 0.0))
    channel_gain_probability = float(cfg.get("channel_gain_probability", 0.0))
    grayscale_probability = float(cfg.get("grayscale_probability", 0.0))
    if probability <= 0.0 and channel_gain_probability <= 0.0 and grayscale_probability <= 0.0:
        return image

    augmented: np.ndarray | None = None
    if probability > 0.0 and random.random() < probability:
        brightness = float(cfg.get("brightness", 0.0))
        contrast = float(cfg.get("contrast", 0.0))
        gamma_range = cfg.get("gamma_range", (1.0, 1.0))
        augmented = image.astype(np.float32)

        brightness_factor = random.uniform(1.0 - brightness, 1.0 + brightness)
        augmented = np.clip(augmented * brightness_factor, 0.0, 255.0)

        contrast_factor = random.uniform(1.0 - contrast, 1.0 + contrast)
        channel_mean = augmented.mean(axis=(0, 1), keepdims=True)
        augmented = np.clip((augmented - channel_mean) * contrast_factor + channel_mean, 0.0, 255.0)

        gamma = random.uniform(float(gamma_range[0]), float(gamma_range[1]))
        augmented = np.clip(np.power(augmented / 255.0, gamma) * 255.0, 0.0, 255.0)

    if channel_gain_probability > 0.0 and random.random() < channel_gain_probability:
        channel_gain_range = cfg.get("channel_gain_range", (1.0, 1.0))
        gains = np.array(
            [random.uniform(float(channel_gain_range[0]), float(channel_gain_range[1])) for _ in range(3)], dtype=np.float32
        )
        if augmented is None:
            augmented = image.astype(np.float32)
        augmented = np.clip(augmented * gains.reshape(1, 1, 3), 0.0, 255.0)

    if grayscale_probability > 0.0 and random.random() < grayscale_probability:
        if augmented is None:
            augmented = image.astype(np.float32)
        gray = 0.299 * augmented[..., 0] + 0.587 * augmented[..., 1] + 0.114 * augmented[..., 2]
        augmented = np.repeat(gray[..., None], 3, axis=2)

    return image if augmented is None else np.clip(augmented, 0.0, 255.0).astype(np.uint8)


class CrackSegmentationDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        img_size: tuple[int, int] | list[int] | None = (512, 512),
        threshold: int = 127,
        augment: bool | None = None,
        target_dir_name: str = "targets",
        require_cached_targets: bool = True,
        photometric_augment: dict | None = None,
        allow_quarter_turns: bool = True,
        scale_jitter: dict | None = None,
        return_sdf_target: bool = False,
        sdf_truncation: float = 8.0,
        return_component_id_map: bool = False,
    ):
        self.root = Path(data_root)
        self.split = split
        self.img_size = tuple(img_size) if img_size is not None else None
        self.threshold = threshold
        self.augment = split == "train" if augment is None else augment
        self.photometric_augment = photometric_augment if split == "train" else None
        self.allow_quarter_turns = bool(allow_quarter_turns)
        self.scale_jitter = scale_jitter if split == "train" else None
        self.return_sdf_target = bool(return_sdf_target)
        self.sdf_truncation = float(sdf_truncation)
        self.return_component_id_map = bool(return_component_id_map) and split == "train"
        self.target_dir = self.root / split / target_dir_name
        self.require_cached_targets = require_cached_targets
        self.pairs = _find_pairs(self.root, split)

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_targets(self, stem: str, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        path = self.target_dir / f"{stem}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing cached targets: {path}. Run `python scripts/precompute_structural_targets.py --data_root {self.root}` first."
            )
        with np.load(path) as cached:
            edge = cached["edge_gt"].astype(np.float32)
            tangent = cached["tangent_gt"].astype(np.float32)
            valid = cached["tangent_valid"].astype(np.float32)
        if edge.shape[-2:] != shape or tangent.shape != (4, *shape) or valid.shape != (1, *shape):
            raise ValueError(f"Cached target shape mismatch for {path}")
        return edge, tangent, valid

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path, mask_path = self.pairs[index]
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        raw_mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        mask = (raw_mask > self.threshold).astype(np.float32)
        edge, tangent, valid = self._load_targets(image_path.stem, mask.shape)
        if self.augment and self.split == "train":
            image, mask, edge, tangent, valid = _augment(
                image,
                mask,
                edge[0],
                tangent,
                valid,
                allow_quarter_turns=self.allow_quarter_turns,
            )
            edge = edge[None]
            image, mask, edge, tangent, valid = _native_scale_jitter(
                image,
                mask,
                edge[0],
                tangent,
                valid,
                self.scale_jitter,
            )
            edge = edge[None]
        image = _photometric_augment(image, self.photometric_augment)
        mask = _letterbox_target(mask[None], self.img_size, "nearest")
        edge = _letterbox_target(edge, self.img_size, "nearest")
        tangent = _letterbox_target(tangent, self.img_size, "bilinear")
        valid = _letterbox_target(valid, self.img_size, "nearest")
        tangent *= valid
        tangent /= tangent.sum(axis=0, keepdims=True).clip(min=1e-6)
        image = _letterbox_image(image, self.img_size)
        image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
        image_tensor = (image_tensor - IMAGENET_MEAN) / IMAGENET_STD
        sample = {
            "image": image_tensor.float(),
            "mask": torch.from_numpy(mask).float(),
            "edge_gt": torch.from_numpy(edge).float(),
            "tangent_gt": torch.from_numpy(tangent).float(),
            "tangent_valid": torch.from_numpy(valid).float(),
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }
        if self.return_sdf_target:
            # Build this only after all geometry and letterboxing, so it is in
            # exactly the same coordinate system as the network outputs.
            sdf_target, sdf_valid = build_truncated_sdf_targets(mask[0], truncation=self.sdf_truncation)
            sample["sdf_target"] = torch.from_numpy(sdf_target).float()
            sample["sdf_valid"] = torch.from_numpy(sdf_valid).float()
        if self.return_component_id_map:
            component_id_map = build_component_id_map(mask[0])
            component_count = int(component_id_map.max())
            if component_count > 64:
                raise ValueError(
                    f"Component-query supervision supports at most 64 components; "
                    f"sample '{image_path.stem}' has {component_count}."
                )
            sample["component_id_map"] = torch.from_numpy(component_id_map[None]).long()
        return sample


def build_dataloader(
    data_root: str | Path,
    split: str,
    img_size: tuple[int, int] | list[int] | None,
    threshold: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    target_dir_name: str = "targets",
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    photometric_augment: dict | None = None,
    allow_quarter_turns: bool = True,
    scale_jitter: dict | None = None,
    return_sdf_target: bool = False,
    sdf_truncation: float = 8.0,
    return_component_id_map: bool = False,
    generator: torch.Generator | None = None,
) -> DataLoader:
    dataset = CrackSegmentationDataset(
        data_root,
        split,
        img_size,
        threshold,
        target_dir_name=target_dir_name,
        photometric_augment=photometric_augment,
        allow_quarter_turns=allow_quarter_turns,
        scale_jitter=scale_jitter,
        return_sdf_target=return_sdf_target,
        sdf_truncation=sdf_truncation,
        return_component_id_map=return_component_id_map,
    )
    kwargs: dict[str, object] = {}
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=split == "train",
        generator=generator,
        **kwargs,
    )
