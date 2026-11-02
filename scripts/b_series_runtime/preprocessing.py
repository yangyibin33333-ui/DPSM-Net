from __future__ import annotations
import random
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
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
