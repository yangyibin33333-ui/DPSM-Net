from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ROUTE_DIRECTIONS = ("horizontal", "vertical", "diagonal", "anti_diagonal")
DIAGNOSTICS_README = """DE-MambaNet diagnostic exports

Error-map colors:
  TP: green  [0, 255, 0]
  FP: red    [255, 0, 0]
  FN: blue   [0, 0, 255]
  TN: black  [0, 0, 0]

Route channel order:
  0: horizontal
  1: vertical
  2: diagonal
  3: anti_diagonal

All saved images use the model-input coordinate system after letterboxing; they
are not restored to the original image size. These exports are diagnostic only
and do not affect metrics or evaluation.
"""


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().cpu().float()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = (image * std + mean).clamp(0, 1)
    return (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def save_mask(mask: np.ndarray, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def save_overlay(image: np.ndarray, mask: np.ndarray, path: str | Path, alpha: float = 0.45):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay = image.copy()
    red = np.zeros_like(image)
    red[..., 0] = 255
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = (
        image[mask_bool].astype(np.float32) * (1.0 - alpha)
        + red[mask_bool].astype(np.float32) * alpha
    ).astype(np.uint8)
    Image.fromarray(overlay).save(path)


def save_prediction_triplet(
    image_tensor: torch.Tensor,
    logits: torch.Tensor,
    stem: str,
    save_dir: str | Path,
    threshold: float = 0.5,
):
    save_dir = Path(save_dir)
    prob = torch.sigmoid(logits.detach().cpu()).squeeze().numpy()
    mask = prob > threshold
    image = denormalize_image(image_tensor)
    save_mask(mask, save_dir / f"{stem}_mask.png")
    save_overlay(image, mask, save_dir / f"{stem}_overlay.png")


def _save_probability(probability: np.ndarray, path_base: Path) -> None:
    probability = probability.astype(np.float32, copy=False)
    np.save(path_base.with_suffix(".npy"), probability)
    Image.fromarray(np.clip(probability * 255.0, 0.0, 255.0).astype(np.uint8)).save(path_base.with_suffix(".png"))


def write_diagnostics_readme(diagnostic_dir: str | Path) -> None:
    diagnostic_dir = Path(diagnostic_dir)
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    (diagnostic_dir / "README.txt").write_text(DIAGNOSTICS_README, encoding="utf-8")


def _save_route_probabilities(route_logits: torch.Tensor | None, stem: str, stage: str, save_dir: Path, size: tuple[int, int]) -> None:
    if route_logits is None:
        return
    route_probability = torch.softmax(route_logits.detach().cpu().float(), dim=0)
    route_probability = F.interpolate(route_probability.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)
    for index, direction in enumerate(ROUTE_DIRECTIONS):
        _save_probability(route_probability[index].numpy(), save_dir / f"{stem}_{stage}_{direction}")


def save_diagnostic_bundle(
    image_tensor: torch.Tensor,
    target: torch.Tensor,
    mask_logits: torch.Tensor,
    edge_logits: torch.Tensor,
    stem: str,
    save_dir: str | Path,
    threshold: float,
    route_s3: torch.Tensor | None = None,
    route_s4: torch.Tensor | None = None,
) -> None:
    """Save input-coordinate diagnostic artifacts for a single evaluated sample."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    image = denormalize_image(image_tensor)
    target_mask = target.detach().cpu().squeeze(0).numpy() > 0.5
    mask_probability = torch.sigmoid(mask_logits.detach().cpu().float()).squeeze(0).numpy()
    edge_probability = torch.sigmoid(edge_logits.detach().cpu().float()).squeeze(0).numpy()
    prediction = mask_probability > threshold

    Image.fromarray(image).save(save_dir / f"{stem}_input.png")
    save_mask(target_mask, save_dir / f"{stem}_target.png")
    _save_probability(mask_probability, save_dir / f"{stem}_mask_probability")
    _save_probability(edge_probability, save_dir / f"{stem}_edge_probability")
    save_mask(prediction, save_dir / f"{stem}_prediction_mask.png")
    save_overlay(image, prediction, save_dir / f"{stem}_overlay.png")

    error_map = np.zeros((*target_mask.shape, 3), dtype=np.uint8)
    error_map[prediction & target_mask] = [0, 255, 0]
    error_map[prediction & ~target_mask] = [255, 0, 0]
    error_map[~prediction & target_mask] = [0, 0, 255]
    Image.fromarray(error_map).save(save_dir / f"{stem}_error_map.png")

    size = tuple(mask_probability.shape)
    _save_route_probabilities(route_s3, stem, "route_s3", save_dir, size)
    _save_route_probabilities(route_s4, stem, "route_s4", save_dir, size)
