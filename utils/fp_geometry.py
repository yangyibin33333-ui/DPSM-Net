"""Shared fixed-threshold false-positive geometry helpers."""

from __future__ import annotations

import numpy as np


CONNECTED_8 = np.ones((3, 3), dtype=np.uint8)


def bridge_false_positive_mask(
    prediction: np.ndarray,
    target: np.ndarray,
    false_positive: np.ndarray | None = None,
) -> np.ndarray:
    """Mark FP pixels in predicted components that merge two GT components.

    Inputs are two-dimensional boolean arrays for one image. Every false-positive
    pixel in a predicted 8-connected component touching at least two separate
    GT 8-connected components is marked as a bridge pixel.
    """
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.ndim != 2 or target.shape != prediction.shape:
        raise ValueError("prediction and target must be matching 2D arrays.")
    if false_positive is None:
        false_positive = prediction & (~target)
    else:
        false_positive = np.asarray(false_positive, dtype=bool)
        if false_positive.shape != prediction.shape:
            raise ValueError("false_positive must match prediction shape.")

    try:
        from scipy import ndimage
    except ImportError as error:  # pragma: no cover - depends on optional runtime dependency
        raise RuntimeError("scipy is required for bridge false-positive geometry.") from error

    predicted_components, _ = ndimage.label(prediction, structure=CONNECTED_8)
    target_components, _ = ndimage.label(target, structure=CONNECTED_8)
    overlap = target_components > 0
    component_ids = predicted_components[overlap]
    target_ids = target_components[overlap]
    component_to_targets: dict[int, set[int]] = {}
    for component_id, target_id in zip(component_ids.tolist(), target_ids.tolist()):
        if component_id > 0:
            component_to_targets.setdefault(component_id, set()).add(target_id)
    bridge_component_ids = [
        component_id
        for component_id, target_ids in component_to_targets.items()
        if len(target_ids) >= 2
    ]
    if not bridge_component_ids:
        return np.zeros_like(false_positive, dtype=bool)
    return false_positive & np.isin(predicted_components, bridge_component_ids)
