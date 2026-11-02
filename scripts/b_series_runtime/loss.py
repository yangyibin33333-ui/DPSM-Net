from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = torch.sigmoid(logits)
        target = target.float()
        intersection = (prediction * target).sum(dim=(1, 2, 3))
        union = prediction.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        return 1.0 - ((2.0 * intersection + self.eps) / (union + self.eps)).mean()

class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        if pos_weight <= 0:
            raise ValueError("mask_pos_weight must be greater than 0.")
        self.pos_weight = float(pos_weight)
        self.bce = (
            nn.BCEWithLogitsLoss()
            if self.pos_weight == 1.0
            else nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.pos_weight]))
        )
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, target.float()) + self.dice(logits, target)
