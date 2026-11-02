from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .de_mambanet import DEMambaNet


ENCODER_PREFIXES = ("stem", "stage1", "down1", "stage2", "down2", "stage3", "down3", "stage4")


def is_encoder_state_key(name: str) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in ENCODER_PREFIXES)


def sample_patch_mask(
    batch_size: int,
    height: int,
    width: int,
    *,
    patch_size: int = 16,
    mask_ratio: float = 0.60,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample an independent pixel-aligned patch mask for every image."""

    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("Image dimensions must be divisible by the reconstruction patch size.")
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must be in (0, 1).")
    patch_height, patch_width = height // patch_size, width // patch_size
    patch_count = patch_height * patch_width
    # 816 patches cannot represent 60% exactly.  Round gives the closest
    # feasible per-image mask count (490 / 816) without varying it by sample.
    masked_count = int(round(patch_count * mask_ratio))
    noise = torch.rand((batch_size, patch_count), generator=generator)
    selected = noise.argsort(dim=1) < masked_count
    patch_mask = selected.reshape(batch_size, 1, patch_height, patch_width)
    return patch_mask.repeat_interleave(patch_size, dim=2).repeat_interleave(patch_size, dim=3)


def masked_smooth_l1_rgb_loss(
    reconstruction: torch.Tensor,
    target_rgb: torch.Tensor,
    patch_mask: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """Smooth L1 reconstruction loss averaged only over masked RGB pixels."""

    if reconstruction.shape != target_rgb.shape:
        raise ValueError("Reconstruction and RGB target must have identical shapes.")
    if patch_mask.shape != (reconstruction.shape[0], 1, *reconstruction.shape[-2:]):
        raise ValueError("patch_mask must have shape [B, 1, H, W] aligned to the reconstruction.")
    error = F.smooth_l1_loss(reconstruction, target_rgb, beta=beta, reduction="none")
    mask = patch_mask.to(dtype=error.dtype)
    return (error * mask).sum() / (mask.sum().clamp_min(1.0) * reconstruction.shape[1])


class _ConvGNAct(nn.Sequential):
    def __init__(self, channels: int):
        super().__init__(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
            nn.GELU(),
        )


class ReconstructionDecoder(nn.Module):
    """Standalone, pretraining-only top-down RGB reconstruction decoder."""

    def __init__(self, encoder_channels: tuple[int, int, int, int] = (40, 80, 160, 320), channels: int = 40):
        super().__init__()
        c1, c2, c3, c4 = encoder_channels
        self.lateral1 = nn.Conv2d(c1, channels, kernel_size=1, bias=False)
        self.lateral2 = nn.Conv2d(c2, channels, kernel_size=1, bias=False)
        self.lateral3 = nn.Conv2d(c3, channels, kernel_size=1, bias=False)
        self.lateral4 = nn.Conv2d(c4, channels, kernel_size=1, bias=False)
        self.refine3 = _ConvGNAct(channels)
        self.refine2 = _ConvGNAct(channels)
        self.refine1 = _ConvGNAct(channels)
        self.final_refine = _ConvGNAct(channels)
        self.rgb_head = nn.Conv2d(channels, 3, kernel_size=1)

    @staticmethod
    def _upsample(feature: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.interpolate(feature, size=target.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, features: Mapping[str, torch.Tensor | None], output_size: tuple[int, int]) -> torch.Tensor:
        f1, f2, f3, f4 = (features[name] for name in ("f1", "f2", "f3", "f4"))
        if not all(torch.is_tensor(feature) for feature in (f1, f2, f3, f4)):
            raise ValueError("Reconstruction decoder requires f1 through f4 encoder features.")
        p4 = self.lateral4(f4)  # type: ignore[arg-type]
        p3 = self.refine3(self.lateral3(f3) + self._upsample(p4, f3))  # type: ignore[arg-type]
        p2 = self.refine2(self.lateral2(f2) + self._upsample(p3, f2))  # type: ignore[arg-type]
        p1 = self.refine1(self.lateral1(f1) + self._upsample(p2, f1))  # type: ignore[arg-type]
        p1 = self.final_refine(p1)
        reconstruction = self.rgb_head(p1)
        return F.interpolate(reconstruction, size=output_size, mode="bilinear", align_corners=False)


class MaskedC40Reconstruction(nn.Module):
    """C40 encoder plus a disposable masked RGB reconstruction decoder."""

    def __init__(self, encoder_kwargs: dict | None = None):
        super().__init__()
        kwargs = dict(encoder_kwargs or {})
        kwargs.pop("name", None)
        if int(kwargs.get("base_channels", 40)) != 40:
            raise ValueError("Masked reconstruction pretraining is fixed to the C40 encoder.")
        self.encoder = DEMambaNet(**kwargs)
        # The standard segmentation decoder exists only because DEMambaNet
        # preserves C40 construction and checkpoint compatibility. It is never
        # called or optimized during pretraining.
        self.encoder.decoder.requires_grad_(False)
        self.mask_token = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.reconstruction_decoder = ReconstructionDecoder()

    def forward(self, images: torch.Tensor, patch_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if images.ndim != 4 or images.size(1) != 3:
            raise ValueError("Masked reconstruction expects normalized RGB images with shape [B, 3, H, W].")
        if patch_mask.shape != (images.size(0), 1, *images.shape[-2:]):
            raise ValueError("patch_mask must be [B, 1, H, W] and match images.")
        masked_images = torch.where(
            patch_mask.to(dtype=torch.bool).expand_as(images),
            self.mask_token.to(dtype=images.dtype).expand_as(images),
            images,
        )
        features = self.encoder.extract_encoder_features(masked_images)
        reconstruction = self.reconstruction_decoder(features, output_size=images.shape[-2:])
        return {"reconstruction": reconstruction, "patch_mask": patch_mask}

    def encoder_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: tensor.detach().clone()
            for name, tensor in self.encoder.state_dict().items()
            if is_encoder_state_key(name)
        }
