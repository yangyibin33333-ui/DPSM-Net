from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .de_mambanet_lite import ConvGNAct, DepthwiseSeparableRefine, SharedAxialMambaContext
from .de_mambanet_lite_balanced import BalancedStructuralDecoder, DEMambaNetLiteBalanced


class ProStructuralDecoder(BalancedStructuralDecoder):
    """Expanded Balanced decoder with a private 64-channel mask refinement."""

    def __init__(
        self,
        encoder_channels: Sequence[int],
        decoder_channels: int,
        mask_channels: int,
    ) -> None:
        super().__init__(encoder_channels, decoder_channels)
        detail_channels = int(encoder_channels[0])
        del self.mask_refine
        del self.mask_head
        self.mask_projection = ConvGNAct(
            detail_channels,
            mask_channels,
            kernel_size=1,
            padding=0,
        )
        self.mask_refine = nn.Sequential(
            DepthwiseSeparableRefine(mask_channels),
            DepthwiseSeparableRefine(mask_channels),
        )
        self.mask_head = nn.Conv2d(mask_channels, 1, kernel_size=1)

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
        mask_feature = self.mask_refine(self.mask_projection(half_feature))
        mask = self.mask_head(mask_feature)
        return {
            "mask": self._resize_output(mask, output_size),
            "edge": self._resize_output(edge, output_size),
            "reliability": self._resize_output(reliability, output_size),
            "route_s1": route_s1,
        }


class DEMambaNetLitePro(DEMambaNetLiteBalanced):
    """Expanded Balanced model with shared axial contexts at stages 3 and 4."""

    EXPECTED_WIDTHS = (40, 80, 160, 240)
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
        expansion_ratio: float = 2.0,
        stage3_mamba_channels: int = 96,
        stage4_mamba_channels: int = 128,
        use_mamba_ssm: bool = True,
        drop_path_rate: float = 0.05,
    ) -> None:
        widths = tuple(int(value) for value in encoder_widths)
        depths = tuple(int(value) for value in encoder_depths)
        if widths != self.EXPECTED_WIDTHS:
            raise ValueError(f"DEMambaNetLitePro encoder_widths must be {self.EXPECTED_WIDTHS}.")
        if depths != self.EXPECTED_DEPTHS:
            raise ValueError(f"DEMambaNetLitePro encoder_depths must be {self.EXPECTED_DEPTHS}.")
        if int(decoder_channels) != 80 or int(detail_channels) != 40 or int(mask_channels) != 64:
            raise ValueError("DEMambaNetLitePro requires decoder/detail/mask channels of 80/40/64.")

        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            encoder_widths=widths,
            encoder_depths=depths,
            decoder_channels=decoder_channels,
            directional_kernel=directional_kernel,
            expansion_ratio=expansion_ratio,
            mamba_channels=stage4_mamba_channels,
            use_mamba_ssm=use_mamba_ssm,
            drop_path_rate=drop_path_rate,
        )
        self.stage4_context = self.global_context
        del self.global_context
        self.stage3_context = SharedAxialMambaContext(
            widths[2],
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
        output_size = image.shape[-2:]
        features = self.extract_encoder_features(image)
        outputs = self.decoder(
            [features["f1"], features["f2"], features["f3"], features["f4"]],
            features["detail_half"],
            output_size,
        )
        outputs["route_s3"] = features["route_s3"]
        outputs["route_s4"] = features["route_s4"]
        return outputs

    def mamba_backend_info(self) -> dict[str, list[str]]:
        return {
            "stage3_context": [self.stage3_context.backend],
            "stage4_context": [self.stage4_context.backend],
        }
