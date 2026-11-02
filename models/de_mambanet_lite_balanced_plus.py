from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .de_mambanet_lite import ConvGNAct
from .de_mambanet_lite_balanced import BalancedStructuralDecoder, DEMambaNetLiteBalanced


class PartialDirectionalContinuityBlock(nn.Module):
    """Apply cooperative directional filtering to half of the channels only."""

    def __init__(self, channels: int = 48) -> None:
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("PartialDirectionalContinuityBlock requires an even channel count.")
        self.channels = int(channels)
        self.active_channels = self.channels // 2
        self.bypass_channels = self.channels - self.active_channels
        active = self.active_channels
        self.local = nn.Conv2d(active, active, kernel_size=3, padding=1, groups=active, bias=False)
        self.horizontal = nn.Conv2d(
            active,
            active,
            kernel_size=(1, 9),
            padding=(0, 4),
            groups=active,
            bias=False,
        )
        self.vertical = nn.Conv2d(
            active,
            active,
            kernel_size=(9, 1),
            padding=(4, 0),
            groups=active,
            bias=False,
        )
        self.dilated = nn.Conv2d(
            active,
            active,
            kernel_size=3,
            padding=3,
            dilation=3,
            groups=active,
            bias=False,
        )
        self.residual_projection = nn.Conv2d(self.channels, self.channels, kernel_size=1)
        nn.init.zeros_(self.residual_projection.weight)
        nn.init.zeros_(self.residual_projection.bias)

    def split_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.split(x, (self.active_channels, self.bypass_channels), dim=1)

    def branch_features(self, x: torch.Tensor) -> torch.Tensor:
        active, bypass = self.split_input(x)
        active_result = (
            self.local(active)
            + self.horizontal(active)
            + self.vertical(active)
            + self.dilated(active)
        )
        return torch.cat([active_result, bypass], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual_projection(self.branch_features(x))


class PhaseAwareMaskResidual(nn.Module):
    """Learn four sub-pixel phases without a full-resolution spatial convolution."""

    def __init__(self, in_channels: int = 48) -> None:
        super().__init__()
        self.projection = nn.Conv2d(in_channels, 4, kernel_size=1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.pixel_shuffle(self.projection(feature))


class MaskContinuityHead(nn.Module):
    """Half-resolution continuity reasoning plus a zero-start phase residual."""

    def __init__(self, in_channels: int = 32, hidden_channels: int = 48) -> None:
        super().__init__()
        if hidden_channels != 48:
            raise ValueError("MaskContinuityHead hidden_channels must remain 48.")
        self.input_projection = ConvGNAct(in_channels, hidden_channels, kernel_size=1, padding=0)
        self.continuity = nn.Sequential(
            PartialDirectionalContinuityBlock(hidden_channels),
            PartialDirectionalContinuityBlock(hidden_channels),
        )
        self.half_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.phase_residual = PhaseAwareMaskResidual(hidden_channels)

    def forward(
        self,
        half_feature: torch.Tensor,
        output_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        continuity_feature = self.continuity(self.input_projection(half_feature))
        mask_half_logits = self.half_head(continuity_feature)
        base_logits = F.interpolate(mask_half_logits, size=output_size, mode="bilinear", align_corners=False)
        phase_residual = self.phase_residual(continuity_feature)
        if phase_residual.shape[-2:] != output_size:
            phase_residual = F.interpolate(
                phase_residual,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        return base_logits + phase_residual, mask_half_logits


class BalancedPlusStructuralDecoder(BalancedStructuralDecoder):
    """Balanced decoder with only its private mask branch replaced."""

    def __init__(self, encoder_channels: Sequence[int], decoder_channels: int) -> None:
        super().__init__(encoder_channels, decoder_channels)
        c1 = int(encoder_channels[0])
        del self.mask_refine
        del self.mask_head
        self.mask_continuity_head = MaskContinuityHead(c1, hidden_channels=48)

    @property
    def mask_input_projection(self) -> nn.Module:
        return self.mask_continuity_head.input_projection

    @property
    def mask_continuity(self) -> nn.Module:
        return self.mask_continuity_head.continuity

    @property
    def mask_half_head(self) -> nn.Module:
        return self.mask_continuity_head.half_head

    @property
    def phase_residual_head(self) -> nn.Module:
        return self.mask_continuity_head.phase_residual

    def forward(
        self,
        features: Sequence[torch.Tensor],
        detail_half: torch.Tensor,
        output_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        x1, x2, x3, x4 = features
        p4 = self.proj4(x4)
        p3 = self.fuse3(p4, self.proj3(x3))
        p2 = self.fuse2(p3, self.proj2(x2))
        quarter_feature = self.final_refine(self.fuse1(p2, self.proj1(x1)))

        route_s1 = self.route_head(quarter_feature)
        half_feature = self.half_fusion(self.half_upsample(quarter_feature), detail_half)
        edge = self.edge_head(half_feature)
        reliability = self.reliability_head(half_feature)
        mask, _ = self.mask_continuity_head(half_feature, output_size)
        return {
            "mask": mask,
            "edge": self._resize_output(edge, output_size),
            "reliability": self._resize_output(reliability, output_size),
            "route_s1": route_s1,
        }


class DEMambaNetLiteBalancedPlus(DEMambaNetLiteBalanced):
    """Balanced model with mask-only continuity and sub-pixel phase recovery."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        encoder_widths: Sequence[int] = (32, 64, 128, 192),
        encoder_depths: Sequence[int] = (2, 2, 3, 2),
        decoder_channels: int = 64,
        directional_kernel: int = 13,
        expansion_ratio: float = 2.0,
        mamba_channels: int = 96,
        use_mamba_ssm: bool = True,
        drop_path_rate: float = 0.05,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            encoder_widths=encoder_widths,
            encoder_depths=encoder_depths,
            decoder_channels=decoder_channels,
            directional_kernel=directional_kernel,
            expansion_ratio=expansion_ratio,
            mamba_channels=mamba_channels,
            use_mamba_ssm=use_mamba_ssm,
            drop_path_rate=drop_path_rate,
        )
        self.decoder = BalancedPlusStructuralDecoder(encoder_widths, decoder_channels)
