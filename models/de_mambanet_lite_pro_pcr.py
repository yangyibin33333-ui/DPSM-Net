"""Partial-compute encoder capacity reallocation for the LitePro decoder."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .de_mambanet_lite import (
    DropPath,
    SharedAxialMambaContext,
    StructureDirectionalBlock,
    _group_count,
)
from .de_mambanet_lite_balanced import DEMambaNetLiteBalanced
from .de_mambanet_lite_pro import ProStructuralDecoder


class LiteDepthwiseDownsample(nn.Module):
    """Stride-two depthwise sampling followed by a lightweight channel change."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.depthwise_norm = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.depthwise_activation = nn.SiLU()
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pointwise_norm = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.pointwise_activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_activation(self.depthwise_norm(self.depthwise(x)))
        return self.pointwise_activation(self.pointwise_norm(self.pointwise(x)))


class PartialDirectionalBranch(nn.Module):
    """Directional processing applied only to a block's active channel subset."""

    def __init__(
        self,
        channels: int,
        expansion_ratio: float,
        directional_kernel: int,
    ) -> None:
        super().__init__()
        if expansion_ratio < 1.0:
            raise ValueError("PartialStructureDirectionalBlock expansion_ratio must be at least 1.0.")
        if directional_kernel < 3 or directional_kernel % 2 == 0:
            raise ValueError("directional_kernel must be an odd integer of at least 3.")
        hidden_channels = max(channels, int(round(channels * expansion_ratio)))
        dilation = max(1, directional_kernel // 4)
        self.input_norm = nn.GroupNorm(_group_count(channels), channels)
        self.input_projection = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False)
        self.local = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels, bias=False
        )
        self.horizontal = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=(1, directional_kernel),
            padding=(0, directional_kernel // 2),
            groups=hidden_channels,
            bias=False,
        )
        self.vertical = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=(directional_kernel, 1),
            padding=(directional_kernel // 2, 0),
            groups=hidden_channels,
            bias=False,
        )
        self.oblique = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=hidden_channels,
            bias=False,
        )
        self.direction_gate = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)
        self.output_norm = nn.GroupNorm(_group_count(hidden_channels), hidden_channels)
        self.activation = nn.SiLU()
        self.output_projection = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, active: torch.Tensor) -> torch.Tensor:
        expanded = self.input_projection(self.input_norm(active))
        local = self.local(expanded)
        directional = self.horizontal(expanded) + self.vertical(expanded) + self.oblique(expanded)
        fused = local + torch.sigmoid(self.direction_gate(local + directional)) * directional
        return self.output_projection(self.activation(self.output_norm(fused)))


class PartialStructureDirectionalBlock(nn.Module):
    """Preserve bypass channels while directionally processing an aligned subset."""

    def __init__(
        self,
        channels: int,
        active_ratio: float,
        expansion_ratio: float = 1.5,
        directional_kernel: int = 13,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        if int(channels) < 8 or int(channels) % 8 != 0:
            raise ValueError("PartialStructureDirectionalBlock channels must be an integer multiple of 8.")
        if not 0.0 < float(active_ratio) <= 1.0:
            raise ValueError("PartialStructureDirectionalBlock active_ratio must be in (0, 1].")
        self.channels = int(channels)
        self.active_ratio = float(active_ratio)
        self.active_channels = max(8, min(
            self.channels,
            int(round((self.channels * self.active_ratio) / 8.0)) * 8,
        ))
        self.bypass_channels = self.channels - self.active_channels
        self.active_branch = PartialDirectionalBranch(
            self.active_channels,
            expansion_ratio=expansion_ratio,
            directional_kernel=directional_kernel,
        )
        self.channel_mixer = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(self.channels), self.channels),
            nn.SiLU(),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        active = x[:, : self.active_channels]
        bypass = x[:, self.active_channels :]
        active = self.active_branch(active)
        mixed = torch.cat((active, bypass), dim=1)
        return x + self.drop_path(self.channel_mixer(mixed))


class DEMambaNetLiteProPCR(DEMambaNetLiteBalanced):
    """LitePro decoder with capacity reallocated into partial Stage2--4 blocks."""

    EXPECTED_WIDTHS = (40, 96, 192, 256)
    EXPECTED_DEPTHS = (2, 2, 4, 2)

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        encoder_widths: Sequence[int] = EXPECTED_WIDTHS,
        encoder_depths: Sequence[int] = EXPECTED_DEPTHS,
        decoder_channels: int = 80,
        detail_channels: int = 40,
        mask_channels: int = 64,
        directional_kernel: int = 13,
        partial_expansion_ratio: float = 1.5,
        stage2_active_ratio: float = 0.75,
        stage3_active_ratio: float = 0.50,
        stage4_active_ratio: float = 0.50,
        stage3_mamba_channels: int = 112,
        stage4_mamba_channels: int = 144,
        use_mamba_ssm: bool = True,
        drop_path_rate: float = 0.05,
    ) -> None:
        widths = tuple(int(value) for value in encoder_widths)
        depths = tuple(int(value) for value in encoder_depths)
        if widths != self.EXPECTED_WIDTHS:
            raise ValueError(f"DEMambaNetLiteProPCR encoder_widths must be {self.EXPECTED_WIDTHS}.")
        if depths != self.EXPECTED_DEPTHS:
            raise ValueError(f"DEMambaNetLiteProPCR encoder_depths must be {self.EXPECTED_DEPTHS}.")
        if (int(decoder_channels), int(detail_channels), int(mask_channels)) != (80, 40, 64):
            raise ValueError("DEMambaNetLiteProPCR requires decoder/detail/mask channels of 80/40/64.")
        if (int(stage3_mamba_channels), int(stage4_mamba_channels)) != (112, 144):
            raise ValueError("DEMambaNetLiteProPCR requires Stage3/Stage4 Mamba channels of 112/144.")

        # This builds the unchanged stem and full Stage1 exactly as LitePro does.
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            encoder_widths=widths,
            encoder_depths=depths,
            decoder_channels=decoder_channels,
            directional_kernel=directional_kernel,
            expansion_ratio=2.0,
            mamba_channels=stage4_mamba_channels,
            use_mamba_ssm=use_mamba_ssm,
            drop_path_rate=drop_path_rate,
        )
        total_blocks = sum(depths)
        drop_rates = torch.linspace(0.0, drop_path_rate, total_blocks).tolist()

        def make_partial_stage(channels: int, depth: int, active_ratio: float, offset: int) -> nn.Sequential:
            return nn.Sequential(
                *(
                    PartialStructureDirectionalBlock(
                        channels,
                        active_ratio=active_ratio,
                        expansion_ratio=partial_expansion_ratio,
                        directional_kernel=directional_kernel,
                        drop_path=float(drop_rates[offset + index]),
                    )
                    for index in range(depth)
                )
            )

        c1, c2, c3, c4 = widths
        self.down1 = LiteDepthwiseDownsample(c1, c2)
        self.stage2 = make_partial_stage(c2, depths[1], stage2_active_ratio, depths[0])
        self.down2 = LiteDepthwiseDownsample(c2, c3)
        self.stage3 = make_partial_stage(c3, depths[2], stage3_active_ratio, depths[0] + depths[1])
        self.down3 = LiteDepthwiseDownsample(c3, c4)
        self.stage4 = make_partial_stage(c4, depths[3], stage4_active_ratio, depths[0] + depths[1] + depths[2])
        self.stage4_context = self.global_context
        del self.global_context
        self.stage3_context = SharedAxialMambaContext(
            c3,
            stage3_mamba_channels,
            use_mamba_ssm=use_mamba_ssm,
        )
        self.decoder = ProStructuralDecoder(widths, decoder_channels, mask_channels)
        self.detail_channels = int(detail_channels)
        self.mask_channels = int(mask_channels)

    def extract_encoder_features(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        stem_half = self.stem_half(image)
        detail_half = self.detail_refine(
            stem_half + self.detail_branch(image, stem_half.shape[-2:]),
        )
        f1 = self.stage1(self.stem_quarter(detail_half))
        f2 = self.stage2(self.down1(f1))
        f3_local = self.stage3(self.down2(f2))
        f3, route_s3 = self.stage3_context(f3_local)
        f4_local = self.stage4(self.down3(f3))
        f4, route_s4 = self.stage4_context(f4_local)
        return {
            "detail_half": detail_half,
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "f4": f4,
            "route_s3": route_s3,
            "route_s4": route_s4,
        }

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.extract_encoder_features(image)
        outputs = self.decoder(
            [features["f1"], features["f2"], features["f3"], features["f4"]],
            features["detail_half"],
            image.shape[-2:],
        )
        outputs["route_s3"] = features["route_s3"]
        outputs["route_s4"] = features["route_s4"]
        return outputs

    def mamba_backend_info(self) -> dict[str, list[str]]:
        return {
            "stage3_context": [self.stage3_context.backend],
            "stage4_context": [self.stage4_context.backend],
        }
