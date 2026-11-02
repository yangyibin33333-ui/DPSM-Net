from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .de_mambanet_lite import (
    ConvGNAct,
    DepthwiseSeparableRefine,
    SharedAxialMambaContext,
    StructureDirectionalBlock,
)


class SobelDetailBranch(nn.Module):
    """Parameter-free RGB Sobel evidence followed by a low-cost 1x1 projection."""

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        ).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-1, -2).contiguous()
        self.register_buffer("sobel_x", sobel_x.repeat(3, 1, 1, 1), persistent=False)
        self.register_buffer("sobel_y", sobel_y.repeat(3, 1, 1, 1), persistent=False)
        self.projection = nn.Conv2d(3, out_channels, kernel_size=1, bias=False)

    def forward(self, image: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        horizontal = F.conv2d(image, self.sobel_x, padding=1, groups=3)
        vertical = F.conv2d(image, self.sobel_y, padding=1, groups=3)
        magnitude = torch.sqrt(horizontal.square() + vertical.square() + 1e-6)
        magnitude = F.interpolate(magnitude, size=output_size, mode="bilinear", align_corners=False)
        return self.projection(magnitude)


class ResidualBaselineGatedScaleFusion(nn.Module):
    """Preserve both inputs and learn only a bounded lateral correction."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(8, channels // 4)
        self.gate = nn.Sequential(
            ConvGNAct(channels * 2, hidden, kernel_size=1, padding=0),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.refine = DepthwiseSeparableRefine(channels)

    def residual_fusion(self, high: torch.Tensor, lateral: torch.Tensor) -> torch.Tensor:
        correction = 0.5 * torch.tanh(self.gate(torch.cat([high, lateral], dim=1)))
        return high + lateral + correction * lateral

    def forward(self, high: torch.Tensor, lateral: torch.Tensor) -> torch.Tensor:
        high = F.interpolate(high, size=lateral.shape[-2:], mode="bilinear", align_corners=False)
        return self.refine(self.residual_fusion(high, lateral))


class BalancedStructuralDecoder(nn.Module):
    """Quarter-resolution top-down decoder with half-resolution detail recovery."""

    def __init__(self, encoder_channels: Sequence[int], decoder_channels: int) -> None:
        super().__init__()
        if len(encoder_channels) != 4:
            raise ValueError("BalancedStructuralDecoder requires four encoder channel widths.")
        c1, c2, c3, c4 = (int(value) for value in encoder_channels)
        self.proj1 = ConvGNAct(c1, decoder_channels, kernel_size=1, padding=0)
        self.proj2 = ConvGNAct(c2, decoder_channels, kernel_size=1, padding=0)
        self.proj3 = ConvGNAct(c3, decoder_channels, kernel_size=1, padding=0)
        self.proj4 = ConvGNAct(c4, decoder_channels, kernel_size=1, padding=0)
        self.fuse3 = ResidualBaselineGatedScaleFusion(decoder_channels)
        self.fuse2 = ResidualBaselineGatedScaleFusion(decoder_channels)
        self.fuse1 = ResidualBaselineGatedScaleFusion(decoder_channels)
        self.final_refine = DepthwiseSeparableRefine(decoder_channels)

        self.route_head = nn.Conv2d(decoder_channels, 4, kernel_size=1)
        self.half_upsample = nn.Sequential(
            nn.Conv2d(decoder_channels, c1 * 4, kernel_size=1),
            nn.PixelShuffle(2),
        )
        self.half_fusion = ResidualBaselineGatedScaleFusion(c1)
        self.edge_head = nn.Conv2d(c1, 1, kernel_size=1)
        self.reliability_head = nn.Conv2d(c1, 1, kernel_size=1)
        self.mask_refine = nn.Sequential(
            DepthwiseSeparableRefine(c1),
            DepthwiseSeparableRefine(c1),
        )
        self.mask_head = nn.Conv2d(c1, 1, kernel_size=1)

    @staticmethod
    def _resize_output(logits: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)

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
        mask = self.mask_head(self.mask_refine(half_feature))
        return {
            "mask": self._resize_output(mask, output_size),
            "edge": self._resize_output(edge, output_size),
            "reliability": self._resize_output(reliability, output_size),
            "route_s1": route_s1,
        }


class DEMambaNetLiteBalanced(nn.Module):
    """DE-MambaNet-Lite with a bounded-cost half-resolution detail recovery path."""

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
        super().__init__()
        widths = tuple(int(value) for value in encoder_widths)
        depths = tuple(int(value) for value in encoder_depths)
        if int(in_channels) != 3 or int(num_classes) != 1:
            raise ValueError("DEMambaNetLiteBalanced supports three-channel RGB binary segmentation only.")
        if len(widths) != 4 or any(value < 8 for value in widths):
            raise ValueError("encoder_widths must contain four values of at least 8 channels.")
        if len(depths) != 4 or any(value < 1 for value in depths):
            raise ValueError("encoder_depths must contain four positive integers.")
        if any(next_width <= width for width, next_width in zip(widths, widths[1:])):
            raise ValueError("encoder_widths must increase strictly between stages.")
        if decoder_channels < 8 or mamba_channels < 8:
            raise ValueError("decoder_channels and mamba_channels must be at least 8.")
        if not 0.0 <= drop_path_rate < 1.0:
            raise ValueError("drop_path_rate must be in [0, 1).")

        c1, c2, c3, c4 = widths
        self.encoder_widths = widths
        self.encoder_depths = depths
        self.stem_half = ConvGNAct(3, c1, kernel_size=3, stride=2)
        self.detail_branch = SobelDetailBranch(c1)
        self.detail_refine = DepthwiseSeparableRefine(c1)
        self.stem_quarter = ConvGNAct(c1, c1, kernel_size=3, stride=2)

        total_blocks = sum(depths)
        drop_rates = torch.linspace(0.0, drop_path_rate, total_blocks).tolist()
        offset = 0

        def make_stage(channels: int, depth: int) -> nn.Sequential:
            nonlocal offset
            stage = nn.Sequential(
                *(
                    StructureDirectionalBlock(
                        channels,
                        expansion_ratio=expansion_ratio,
                        directional_kernel=directional_kernel,
                        drop_path=float(drop_rates[offset + index]),
                    )
                    for index in range(depth)
                )
            )
            offset += depth
            return stage

        self.stage1 = make_stage(c1, depths[0])
        self.down1 = ConvGNAct(c1, c2, kernel_size=3, stride=2)
        self.stage2 = make_stage(c2, depths[1])
        self.down2 = ConvGNAct(c2, c3, kernel_size=3, stride=2)
        self.stage3 = make_stage(c3, depths[2])
        self.down3 = ConvGNAct(c3, c4, kernel_size=3, stride=2)
        self.stage4 = make_stage(c4, depths[3])
        self.global_context = SharedAxialMambaContext(c4, mamba_channels, use_mamba_ssm=use_mamba_ssm)
        self.decoder = BalancedStructuralDecoder(widths, decoder_channels)

    def extract_encoder_features(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        stem_half = self.stem_half(image)
        detail_half = self.detail_refine(
            stem_half + self.detail_branch(image, stem_half.shape[-2:]),
        )
        f1 = self.stage1(self.stem_quarter(detail_half))
        f2 = self.stage2(self.down1(f1))
        f3 = self.stage3(self.down2(f2))
        f4_local = self.stage4(self.down3(f3))
        f4, route_s4 = self.global_context(f4_local)
        return {
            "detail_half": detail_half,
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "f4": f4,
            "route_s4": route_s4,
        }

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        output_size = image.shape[-2:]
        features = self.extract_encoder_features(image)
        outputs = self.decoder(
            [features["f1"], features["f2"], features["f3"], features["f4"]],
            features["detail_half"],
            output_size,
        )
        outputs["route_s4"] = features["route_s4"]
        return outputs

    def mamba_backend_info(self) -> dict[str, list[str]]:
        return {"global_context": [self.global_context.backend]}
