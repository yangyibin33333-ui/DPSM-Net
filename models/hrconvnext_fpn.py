from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for NCHW feature maps."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.channels = int(channels)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.channels))
        self.bias = nn.Parameter(torch.zeros(self.channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (self.channels,), self.weight, self.bias, self.eps)
        # Conv2d parameters use contiguous NCHW storage. Returning a contiguous
        # feature map keeps their gradients compatible with fused AdamW state.
        return x.permute(0, 3, 1, 2).contiguous()


class ConvNeXtBlock(nn.Module):
    """Minimal ConvNeXt block without global response normalization or attention."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        expanded_channels = 4 * channels
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = LayerNorm2d(channels)
        self.expand = nn.Conv2d(channels, expanded_channels, kernel_size=1)
        self.activation = nn.GELU()
        self.project = nn.Conv2d(expanded_channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.norm(x)
        x = self.expand(x)
        x = self.activation(x)
        x = self.project(x)
        return residual + x


class FPNRefine(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(1, channels)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(x)))


class HRConvNeXtFPN(nn.Module):
    """Standalone high-resolution ConvNeXt encoder with a lightweight FPN decoder."""

    _EXPECTED_STAGE_DIMS = (56, 112, 224, 448)
    _EXPECTED_STAGE_DEPTHS = (1, 2, 6, 2)
    _EXPECTED_FPN_CHANNELS = 104

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        stage_dims: Sequence[int] = _EXPECTED_STAGE_DIMS,
        stage_depths: Sequence[int] = _EXPECTED_STAGE_DEPTHS,
        fpn_channels: int = _EXPECTED_FPN_CHANNELS,
    ) -> None:
        super().__init__()
        stage_dims = tuple(int(channel) for channel in stage_dims)
        stage_depths = tuple(int(depth) for depth in stage_depths)
        if int(in_channels) != 3:
            raise ValueError("HRConvNeXtFPN is fixed to three-channel RGB input.")
        if int(num_classes) != 1:
            raise ValueError("HRConvNeXtFPN is fixed to one binary segmentation class.")
        if stage_dims != self._EXPECTED_STAGE_DIMS or stage_depths != self._EXPECTED_STAGE_DEPTHS:
            raise ValueError(
                "HRConvNeXtFPN uses fixed stage_dims=(56, 112, 224, 448) "
                "and stage_depths=(1, 2, 6, 2)."
            )
        if int(fpn_channels) != self._EXPECTED_FPN_CHANNELS:
            raise ValueError("HRConvNeXtFPN uses a fixed FPN width of 104 channels.")

        self.is_mamba_free = True
        self.stem = nn.Conv2d(3, stage_dims[0], kernel_size=3, stride=2, padding=1)
        self.stage1 = self._make_stage(stage_dims[0], stage_depths[0])
        self.down1 = self._make_downsample(stage_dims[0], stage_dims[1])
        self.stage2 = self._make_stage(stage_dims[1], stage_depths[1])
        self.down2 = self._make_downsample(stage_dims[1], stage_dims[2])
        self.stage3 = self._make_stage(stage_dims[2], stage_depths[2])
        self.down3 = self._make_downsample(stage_dims[2], stage_dims[3])
        self.stage4 = self._make_stage(stage_dims[3], stage_depths[3])

        self.lateral1 = nn.Conv2d(stage_dims[0], fpn_channels, kernel_size=1)
        self.lateral2 = nn.Conv2d(stage_dims[1], fpn_channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(stage_dims[2], fpn_channels, kernel_size=1)
        self.lateral4 = nn.Conv2d(stage_dims[3], fpn_channels, kernel_size=1)
        self.refine3 = FPNRefine(fpn_channels)
        self.refine2 = FPNRefine(fpn_channels)
        self.refine1 = FPNRefine(fpn_channels)
        self.final_refine = FPNRefine(fpn_channels)

        self.mask_head = nn.Conv2d(fpn_channels, 1, kernel_size=1)
        self.edge_head = nn.Conv2d(fpn_channels, 1, kernel_size=1)
        self.reliability_head = nn.Conv2d(fpn_channels, 1, kernel_size=1)
        self.route_s1_head = nn.Conv2d(fpn_channels, 4, kernel_size=1)

    @staticmethod
    def _make_stage(channels: int, depth: int) -> nn.Sequential:
        return nn.Sequential(*(ConvNeXtBlock(channels) for _ in range(depth)))

    @staticmethod
    def _make_downsample(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            LayerNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
        )

    @staticmethod
    def _upsample_to(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        c1 = self.stage1(self.stem(x))
        c2 = self.stage2(self.down1(c1))
        c3 = self.stage3(self.down2(c2))
        c4 = self.stage4(self.down3(c3))

        p4 = self.lateral4(c4)
        p3 = self.refine3(self.lateral3(c3) + self._upsample_to(p4, c3))
        p2 = self.refine2(self.lateral2(c2) + self._upsample_to(p3, c2))
        p1 = self.refine1(self.lateral1(c1) + self._upsample_to(p2, c1))
        feature = self.final_refine(p1)

        return {
            "mask": F.interpolate(self.mask_head(feature), size=input_size, mode="bilinear", align_corners=False),
            "edge": F.interpolate(self.edge_head(feature), size=input_size, mode="bilinear", align_corners=False),
            "reliability": F.interpolate(
                self.reliability_head(feature), size=input_size, mode="bilinear", align_corners=False
            ),
            "route_s1": self.route_s1_head(feature),
        }

    @staticmethod
    def mamba_backend_info() -> dict[str, list[str]]:
        return {}
