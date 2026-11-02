"""Full-convolution wide-encoder capacity control for LitePro."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .de_mambanet_lite import SharedAxialMambaContext
from .de_mambanet_lite_balanced import DEMambaNetLiteBalanced
from .de_mambanet_lite_pro import ProStructuralDecoder


class DEMambaNetLiteProWideEncoder(DEMambaNetLiteBalanced):
    """CSDR-width encoder and Mamba contexts with the unmodified Pro decoder."""

    EXPECTED_WIDTHS = (48, 96, 192, 256)
    EXPECTED_DEPTHS = (2, 2, 4, 2)

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        encoder_widths: Sequence[int] = EXPECTED_WIDTHS,
        encoder_depths: Sequence[int] = EXPECTED_DEPTHS,
        decoder_channels: int = 80,
        detail_channels: int = 48,
        mask_channels: int = 64,
        directional_kernel: int = 13,
        expansion_ratio: float = 2.0,
        stage3_mamba_channels: int = 112,
        stage4_mamba_channels: int = 144,
        use_mamba_ssm: bool = True,
        drop_path_rate: float = 0.05,
    ) -> None:
        widths = tuple(int(value) for value in encoder_widths)
        depths = tuple(int(value) for value in encoder_depths)
        if widths != self.EXPECTED_WIDTHS:
            raise ValueError(f"DEMambaNetLiteProWideEncoder encoder_widths must be {self.EXPECTED_WIDTHS}.")
        if depths != self.EXPECTED_DEPTHS:
            raise ValueError(f"DEMambaNetLiteProWideEncoder encoder_depths must be {self.EXPECTED_DEPTHS}.")
        if (int(decoder_channels), int(detail_channels), int(mask_channels)) != (80, 48, 64):
            raise ValueError("DEMambaNetLiteProWideEncoder requires decoder/detail/mask channels of 80/48/64.")
        if (int(stage3_mamba_channels), int(stage4_mamba_channels)) != (112, 144):
            raise ValueError("DEMambaNetLiteProWideEncoder requires Stage3/Stage4 Mamba channels of 112/144.")

        # Balanced supplies the complete, regular ConvGNAct downsampling and
        # full StructureDirectionalBlocks required for this capacity control.
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
