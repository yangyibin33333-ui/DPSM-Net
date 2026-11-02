from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

try:
    from skimage.measure import label
    from skimage.morphology import skeletonize
except ImportError:  # pragma: no cover - dependency is validated at evaluation time
    label = skeletonize = None


def _binary_counts(pred: torch.Tensor, target: torch.Tensor):
    pred, target = pred.bool(), target.bool()
    return (pred & target).sum().float(), (pred & ~target).sum().float(), (~pred & target).sum().float(), (~pred & ~target).sum().float()


def _boundary(mask: torch.Tensor) -> torch.Tensor:
    value = mask.float()
    dilated = F.max_pool2d(value, 3, stride=1, padding=1)
    eroded = -F.max_pool2d(-value, 3, stride=1, padding=1)
    return (dilated - eroded) > 0


def _topology_scores(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    if skeletonize is None or label is None:
        raise RuntimeError("scikit-image is required for clDice and component-error evaluation.")
    pred_skeleton = skeletonize(pred)
    target_skeleton = skeletonize(target)
    tprec = float((pred_skeleton & target).sum()) / max(float(pred_skeleton.sum()), 1.0)
    tsens = float((target_skeleton & pred).sum()) / max(float(target_skeleton.sum()), 1.0)
    cldice = 2.0 * tprec * tsens / max(tprec + tsens, 1e-7)
    component_error = abs(int(label(pred, connectivity=2).max()) - int(label(target, connectivity=2).max()))
    return cldice, float(component_error)


class SegmentationMetric:
    def __init__(
        self,
        thresholds: list[float] | None = None,
        boundary_tolerance: int = 2,
        primary_threshold: float = 0.5,
    ):
        if not 0.0 <= primary_threshold <= 1.0:
            raise ValueError(f"primary_threshold must be in [0, 1], got {primary_threshold}")
        self.thresholds = thresholds or [i / 10.0 for i in range(1, 10)]
        self.boundary_tolerance = boundary_tolerance
        self.primary_threshold = float(primary_threshold)
        self.reset()

    def reset(self) -> None:
        self.tp = self.fp = self.fn = self.tn = torch.tensor(0.0)
        self.sweep = {threshold: [torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)] for threshold in self.thresholds}
        self.ois_values: list[float] = []
        self.boundary_tp = torch.tensor(0.0)
        self.boundary_pred = torch.tensor(0.0)
        self.boundary_target = torch.tensor(0.0)
        self.cldice_values: list[float] = []
        self.component_errors: list[float] = []

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        probability = torch.sigmoid(logits.detach()).cpu()
        target = target.detach().cpu() > 0.5
        prediction = probability > self.primary_threshold
        tp, fp, fn, tn = _binary_counts(prediction, target)
        self.tp, self.fp, self.fn, self.tn = self.tp + tp, self.fp + fp, self.fn + fn, self.tn + tn
        for threshold in self.thresholds:
            stp, sfp, sfn, _ = _binary_counts(probability > threshold, target)
            self.sweep[threshold][0] += stp
            self.sweep[threshold][1] += sfp
            self.sweep[threshold][2] += sfn
        prediction_edge, target_edge = _boundary(prediction), _boundary(target)
        radius = 2 * self.boundary_tolerance + 1
        pred_match = prediction_edge & (F.max_pool2d(target_edge.float(), radius, 1, self.boundary_tolerance) > 0)
        target_match = target_edge & (F.max_pool2d(prediction_edge.float(), radius, 1, self.boundary_tolerance) > 0)
        self.boundary_tp += pred_match.sum() + target_match.sum()
        self.boundary_pred += prediction_edge.sum()
        self.boundary_target += target_edge.sum()
        for index in range(probability.size(0)):
            pred_np = prediction[index, 0].numpy()
            target_np = target[index, 0].numpy()
            cldice, component_error = _topology_scores(pred_np, target_np)
            self.cldice_values.append(cldice)
            self.component_errors.append(component_error)
            best = 0.0
            for threshold in self.thresholds:
                stp, sfp, sfn, _ = _binary_counts(probability[index : index + 1] > threshold, target[index : index + 1])
                best = max(best, float((2.0 * stp / (2.0 * stp + sfp + sfn + 1e-7)).item()))
            self.ois_values.append(best)

    def compute(self) -> dict[str, float]:
        eps = 1e-7
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        bg_iou = self.tn / (self.tn + self.fp + self.fn + eps)
        dice = 2.0 * self.tp / (2.0 * self.tp + self.fp + self.fn + eps)
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        ods = max(float((2.0 * stp / (2.0 * stp + sfp + sfn + eps)).item()) for stp, sfp, sfn in self.sweep.values())
        boundary_f1 = self.boundary_tp / (self.boundary_pred + self.boundary_target + eps)
        return {
            "IoU": float(iou.item()),
            "mIoU": float(((iou + bg_iou) * 0.5).item()),
            "Dice": float(dice.item()),
            "F1": float(dice.item()),
            "Precision": float(precision.item()),
            "Recall": float(recall.item()),
            "ODS": ods,
            "OIS": sum(self.ois_values) / max(len(self.ois_values), 1),
            "BoundaryF1": float(boundary_f1.item()),
            "clDice": sum(self.cldice_values) / max(len(self.cldice_values), 1),
            "ComponentError": sum(self.component_errors) / max(len(self.component_errors), 1),
        }


def binary_metrics_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    prediction, target = torch.sigmoid(logits) > threshold, target > 0.5
    tp, fp, fn, _ = _binary_counts(prediction, target)
    eps = 1e-7
    dice = 2.0 * tp / (2.0 * tp + fp + fn + eps)
    return {
        "IoU": float((tp / (tp + fp + fn + eps)).item()),
        "Dice": float(dice.item()),
        "F1": float(dice.item()),
        "Precision": float((tp / (tp + fp + eps)).item()),
        "Recall": float((tp / (tp + fn + eps)).item()),
    }
