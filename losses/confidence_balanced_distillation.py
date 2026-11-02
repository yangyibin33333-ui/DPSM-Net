"""Confidence-balanced, class-normalized mask-logit distillation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def distillation_weight_for_epoch(
    epoch: int,
    *,
    lambda_max: float = 0.25,
    warmup_epochs: int = 10,
    ramp_end_epoch: int = 40,
) -> float:
    """Return the one-based epoch CBSD weight with a linear post-warmup ramp."""

    if lambda_max < 0.0:
        raise ValueError("lambda_max must be non-negative.")
    if warmup_epochs < 0 or ramp_end_epoch <= warmup_epochs:
        raise ValueError("ramp_end_epoch must be greater than warmup_epochs, both non-negative.")
    if epoch <= warmup_epochs:
        return 0.0
    if epoch >= ramp_end_epoch:
        return float(lambda_max)
    return float(lambda_max) * (epoch - warmup_epochs) / (ramp_end_epoch - warmup_epochs)


class ConfidenceBalancedDistillationLoss(nn.Module):
    """FP32 confidence-weighted mask distillation with separate class averages."""

    def __init__(
        self,
        temperature: float = 2.0,
        foreground_weight: float = 0.45,
        background_weight: float = 0.55,
        confidence_power: float = 2.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("CBSD temperature must be positive.")
        if foreground_weight < 0.0 or background_weight < 0.0:
            raise ValueError("CBSD foreground/background weights must be non-negative.")
        if foreground_weight + background_weight <= 0.0:
            raise ValueError("CBSD foreground/background weights must have positive total weight.")
        if confidence_power <= 0.0:
            raise ValueError("CBSD confidence_power must be positive.")
        if eps <= 0.0:
            raise ValueError("CBSD eps must be positive.")
        self.temperature = float(temperature)
        self.foreground_weight = float(foreground_weight)
        self.background_weight = float(background_weight)
        self.confidence_power = float(confidence_power)
        self.eps = float(eps)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        target: torch.Tensor,
        *,
        return_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if student_logits.shape != teacher_logits.shape or student_logits.shape != target.shape:
            raise ValueError("CBSD student logits, teacher logits, and target must have the same shape.")
        student_fp32 = student_logits.float()
        teacher_fp32 = teacher_logits.detach().float()
        target_fp32 = target.float()
        teacher_probability = torch.sigmoid(teacher_fp32)
        teacher_soft_probability = torch.sigmoid(teacher_fp32 / self.temperature)
        confidence = (2.0 * teacher_probability - 1.0).abs().pow(self.confidence_power)
        kd_pixel = F.binary_cross_entropy_with_logits(
            student_fp32 / self.temperature,
            teacher_soft_probability,
            reduction="none",
        ) * (self.temperature * self.temperature)
        foreground = target_fp32 >= 0.5
        background = ~foreground

        def class_average(mask: torch.Tensor) -> torch.Tensor:
            mask_fp32 = mask.to(dtype=confidence.dtype)
            denominator = (confidence * mask_fp32).sum()
            numerator = (kd_pixel * confidence * mask_fp32).sum()
            # Preserve a valid zero-gradient scalar when the class is absent.
            return torch.where(mask.any(), numerator / denominator.clamp_min(self.eps), numerator * 0.0)

        foreground_loss = class_average(foreground)
        background_loss = class_average(background)
        loss = self.foreground_weight * foreground_loss + self.background_weight * background_loss
        if not return_stats:
            return loss

        def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            mask_fp32 = mask.to(dtype=value.dtype)
            denominator = mask_fp32.sum()
            numerator = (value * mask_fp32).sum()
            return torch.where(mask.any(), numerator / denominator.clamp_min(1.0), numerator * 0.0)

        stats = {
            "foreground_loss": foreground_loss.detach(),
            "background_loss": background_loss.detach(),
            "teacher_foreground_probability": masked_mean(teacher_probability, foreground).detach(),
            "teacher_background_probability": masked_mean(teacher_probability, background).detach(),
            "foreground_confidence": masked_mean(confidence, foreground).detach(),
            "background_confidence": masked_mean(confidence, background).detach(),
        }
        return loss, stats
