from __future__ import annotations

import math

import numpy as np

try:
    from skimage.morphology import binary_dilation, binary_erosion, skeletonize
except ImportError:  # pragma: no cover - exercised by the CLI error path
    binary_dilation = binary_erosion = skeletonize = None

try:
    from scipy.ndimage import distance_transform_edt
except ImportError:  # pragma: no cover - only required by the optional SDF ablation
    distance_transform_edt = None


ROUTE_ANGLES = np.array([0.0, math.pi / 2.0, math.pi / 4.0, 3.0 * math.pi / 4.0])


def require_scikit_image() -> None:
    if skeletonize is None:
        raise RuntimeError(
            "scikit-image is required for structural target preprocessing. "
            "Install it with `pip install scikit-image`."
        )


def require_scipy_distance_transform() -> None:
    if distance_transform_edt is None:
        raise RuntimeError(
            "SciPy is required when SDF supervision is enabled. "
            "Install it with `pip install scipy`."
        )


def build_truncated_sdf_targets(mask: np.ndarray, truncation: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    """Build normalized signed-distance targets and a narrow valid band.

    Foreground distances are positive and background distances are negative.
    The valid mask excludes locations farther than ``truncation`` from a
    foreground/background boundary so far-field background cannot dominate the
    auxiliary regression loss.
    """
    require_scipy_distance_transform()
    if truncation <= 0:
        raise ValueError("SDF truncation must be greater than 0.")
    binary = np.asarray(mask) > 0.5
    if binary.ndim != 2:
        raise ValueError(f"SDF mask must have shape [H, W], got {binary.shape}.")

    if not binary.any():
        sdf_raw = np.full(binary.shape, -np.inf, dtype=np.float32)
    elif binary.all():
        sdf_raw = np.full(binary.shape, np.inf, dtype=np.float32)
    else:
        distance_inside = distance_transform_edt(binary)
        distance_from_background = distance_transform_edt(~binary)
        sdf_raw = distance_inside - distance_from_background

    sdf_valid = (np.abs(sdf_raw) <= truncation).astype(np.float32)[None]
    sdf_target = (np.clip(sdf_raw, -truncation, truncation) / truncation).astype(np.float32)[None]
    return sdf_target, sdf_valid


def edge_from_mask(mask: np.ndarray) -> np.ndarray:
    """Return a one-pixel morphological boundary for a binary mask."""
    require_scikit_image()
    binary = mask.astype(bool)
    return np.logical_xor(binary_dilation(binary), binary_erosion(binary)).astype(np.float32)


def _angular_weights(angle: float, sigma: float = math.pi / 10.0) -> np.ndarray:
    distance = np.abs(angle - ROUTE_ANGLES)
    distance = np.minimum(distance, math.pi - distance)
    weights = np.exp(-0.5 * (distance / sigma) ** 2)
    return weights / max(float(weights.sum()), 1e-8)


def tangent_from_skeleton(mask: np.ndarray, radius: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Estimate soft four-bin tangent distributions from a crack centerline.

    Each skeleton pixel gathers all local centerline vectors. At junctions this naturally
    yields a multi-modal distribution instead of forcing a single direction class.
    """
    require_scikit_image()
    skeleton = skeletonize(mask.astype(bool))
    height, width = skeleton.shape
    tangent = np.zeros((4, height, width), dtype=np.float32)
    valid = np.zeros((1, height, width), dtype=np.float32)
    points = np.argwhere(skeleton)

    for row, col in points:
        r0, r1 = max(0, row - radius), min(height, row + radius + 1)
        c0, c1 = max(0, col - radius), min(width, col + radius + 1)
        neighbors = np.argwhere(skeleton[r0:r1, c0:c1])
        if not len(neighbors):
            continue
        weights = np.zeros(4, dtype=np.float32)
        for nr, nc in neighbors:
            dr, dc = int(nr + r0 - row), int(nc + c0 - col)
            if dr == 0 and dc == 0:
                continue
            angle = math.atan2(float(dr), float(dc)) % math.pi
            weights += _angular_weights(angle)
        total = float(weights.sum())
        if total > 0:
            tangent[:, row, col] = weights / total
            valid[0, row, col] = 1.0
    return tangent, valid


def build_structural_targets(mask: np.ndarray, threshold: int = 127) -> dict[str, np.ndarray]:
    binary = np.asarray(mask, dtype=np.uint8) > threshold
    tangent_gt, tangent_valid = tangent_from_skeleton(binary)
    return {
        "edge_gt": edge_from_mask(binary)[None].astype(np.float32),
        "tangent_gt": tangent_gt.astype(np.float32),
        "tangent_valid": tangent_valid.astype(np.float32),
    }
