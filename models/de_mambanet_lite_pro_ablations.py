from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .de_mambanet_lite import SharedAxialMambaContext
from .de_mambanet_lite_balanced import DEMambaNetLiteBalanced
from .de_mambanet_lite_pro import DEMambaNetLitePro, ProStructuralDecoder


class RouteOnlyAuxHead(nn.Module):
    """Predict route supervision without modifying the main feature path."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(in_channels, 4, kernel_size=1)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.projection(feature)


class DEMambaNetLiteProNoS3(DEMambaNetLitePro):
    """Strict Pro ablation with Stage3 context removed but route_s3 retained."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        encoder_widths: Sequence[int] = (40, 80, 160, 240),
        encoder_depths: Sequence[int] = (2, 2, 4, 2),
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
        if widths != (40, 80, 160, 240) or depths != (2, 2, 4, 2):
            raise ValueError("DEMambaNetLiteProNoS3 must retain Pro widths and depths.")
        if (decoder_channels, detail_channels, mask_channels) != (80, 40, 64):
            raise ValueError("DEMambaNetLiteProNoS3 must retain Pro decoder/detail/mask widths.")
        if int(stage3_mamba_channels) != 96 or int(stage4_mamba_channels) != 128:
            raise ValueError("DEMambaNetLiteProNoS3 must retain Pro Mamba channel configuration.")

        # Call the Balanced initializer directly so a Stage3 Mamba core is
        # never instantiated, even transiently during construction.
        DEMambaNetLiteBalanced.__init__(
            self,
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
        self.route_s3_head = RouteOnlyAuxHead(widths[2])
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
        f3 = self.stage3(self.down2(f2))
        route_s3 = self.route_s3_head(f3)
        # The unmodified Stage3 tensor, not route logits or a fused tensor, is
        # consumed by both the decoder skip and the Stage4 downsampling path.
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

    def mamba_backend_info(self) -> dict[str, list[str]]:
        return {
            "stage3_context": ["disabled"],
            "stage4_context": [self.stage4_context.backend],
        }


class DEMambaNetLiteProNarrow(DEMambaNetLitePro):
    """Strict Pro ablation retaining both contexts with uniformly reduced widths."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        encoder_widths: Sequence[int] = (32, 64, 128, 192),
        encoder_depths: Sequence[int] = (2, 2, 4, 2),
        decoder_channels: int = 64,
        detail_channels: int = 32,
        mask_channels: int = 48,
        directional_kernel: int = 13,
        expansion_ratio: float = 2.0,
        stage3_mamba_channels: int = 80,
        stage4_mamba_channels: int = 96,
        use_mamba_ssm: bool = True,
        drop_path_rate: float = 0.05,
    ) -> None:
        widths = tuple(int(value) for value in encoder_widths)
        depths = tuple(int(value) for value in encoder_depths)
        if widths != (32, 64, 128, 192) or depths != (2, 2, 4, 2):
            raise ValueError("DEMambaNetLiteProNarrow requires widths/depths of 32,64,128,192 / 2,2,4,2.")
        if (decoder_channels, detail_channels, mask_channels) != (64, 32, 48):
            raise ValueError("DEMambaNetLiteProNarrow requires decoder/detail/mask widths of 64/32/48.")
        if (stage3_mamba_channels, stage4_mamba_channels) != (80, 96):
            raise ValueError("DEMambaNetLiteProNarrow requires Stage3/Stage4 Mamba widths of 80/96.")

        DEMambaNetLiteBalanced.__init__(
            self,
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

    # The dual-context topology and output contract are exactly Pro's.
    extract_encoder_features = DEMambaNetLitePro.extract_encoder_features
    forward = DEMambaNetLitePro.forward
    mamba_backend_info = DEMambaNetLitePro.mamba_backend_info
