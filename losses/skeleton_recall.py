"""Skeleton-recall loss for thin structures (2D crack centrelines).

Motivation
----------
Overlap losses (BCE/Dice) optimise volumetric agreement, so a prediction can lose
a whole thin branch, or break it into pieces, while keeping the same Dice value.
Kirchhoff et al., "Skeleton Recall Loss" (ECCV 2024), make the opposite trade:
they evaluate a recall term directly on the *ground-truth* skeleton, using the
soft prediction at those pixels, which needs no differentiable skeletonisation
and no GPU topology machinery.

Implementation notes specific to this repository
------------------------------------------------
* The skeleton is taken from the cached ``tangent_valid`` target instead of being
  recomputed.  ``utils.structural_targets.tangent_from_skeleton`` marks exactly
  the centreline pixels that carry a tangent estimate, and the reported
  centreline-recall diagnostic (``scripts/evaluate_orientation_strata.py``,
  ``scripts/diagnose_direction_asymmetry.py``) measures recall on that same
  support.  Training and reporting therefore look at the same pixel set, and the
  loss costs one masked mean per step.
* Reductions run in float32 even under AMP, so the sum over thousands of
  centreline pixels does not lose the smaller hits in fp16.
* ``band`` widens the support with a max-pool before the recall is taken.  With
  ``band=0`` the term is the strict ECCV formulation; with ``band=1`` a hit
  within one pixel of the centreline counts, which matches the one-pixel
  tolerance used when the centreline recall is reported and is less able to pull
  probability mass off the centreline to satisfy the term.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SkeletonRecallLoss(nn.Module):
    """``1 - mean_image recall(probabilities on the GT skeleton)``.

    Args:
        band: non-negative integer.  ``0`` uses the exact skeleton, ``k > 0``
            unions a ``(2k+1) x (2k+1)`` window around every skeleton pixel.
        eps: numerical floor for the recall ratio.
        background_weight: optional weight for the complementary "skeleton
            should stay quiet when it does not exist" direction.  Kept at 0 by
            default because the primary mask loss already covers background.
    """

    def __init__(self, band: int = 0, eps: float = 1e-6) -> None:
        super().__init__()
        if band < 0:
            raise ValueError("SkeletonRecallLoss band must be non-negative.")
        if eps <= 0:
            raise ValueError("SkeletonRecallLoss eps must be positive.")
        self.band = int(band)
        self.eps = float(eps)

    def _support(self, skeleton: torch.Tensor) -> torch.Tensor:
        support = skeleton.float()
        if self.band > 0:
            kernel = 2 * self.band + 1
            support = F.max_pool2d(support, kernel, stride=1, padding=self.band)
        return support

    def forward(
        self,
        logits: torch.Tensor,
        skeleton: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(loss, mean_recall)``.

        ``logits`` is ``[B, 1, H, W]``; ``skeleton`` accepts ``[B, 1, H, W]`` or
        ``[B, H, W]``.  Images without any skeleton support are dropped from the
        mean instead of contributing a constant zero, and a batch with no
        support at all returns a zero loss that still carries the graph.
        """
        if skeleton.dim() == logits.dim() - 1:
            skeleton = skeleton.unsqueeze(1)
        if skeleton.shape[-2:] != logits.shape[-2:]:
            skeleton = F.interpolate(
                skeleton.float(), size=logits.shape[-2:], mode="nearest"
            )
        if skeleton.shape[0] != logits.shape[0]:
            raise ValueError(
                "SkeletonRecallLoss expects one skeleton map per logit map, got "
                f"{skeleton.shape[0]} and {logits.shape[0]}."
            )

        probabilities = torch.sigmoid(logits.float())
        support = self._support(skeleton)
        numerator = (probabilities * support).flatten(1).sum(dim=1)
        denominator = support.flatten(1).sum(dim=1)
        present = denominator > 0
        if not bool(present.any()):
            zero = logits.sum() * 0.0
            return zero, zero.detach()
        recall = (numerator[present] + self.eps) / (denominator[present] + self.eps)
        return 1.0 - recall.mean(), recall.mean().detach()
