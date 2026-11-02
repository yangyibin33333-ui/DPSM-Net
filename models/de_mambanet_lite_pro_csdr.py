from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .de_mambanet_lite import ConvGNAct, DropPath, SharedAxialMambaContext, _group_count
from .de_mambanet_lite_balanced import DEMambaNetLiteBalanced
from .de_mambanet_lite_pro import DEMambaNetLitePro


class CSDRFusion(nn.Module):
    """Additive cross-scale fusion with deep-semantic branch selection."""

    def __init__(self, channels: int = 96, drop_path: float = 0.0) -> None:
        super().__init__()
        hidden = max(8, channels // 4)
        self.local = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.horizontal = nn.Conv2d(
            channels, channels, (1, 7), padding=(0, 3), groups=channels, bias=False
        )
        self.vertical = nn.Conv2d(
            channels, channels, (7, 1), padding=(3, 0), groups=channels, bias=False
        )
        self.weight_selector = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden, 3, kernel_size=1),
        )
        self.mix_norm = nn.GroupNorm(_group_count(channels), channels)
        self.activation = nn.SiLU()
        self.projection = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.drop_path = DropPath(drop_path)

    def branch_weights(self, deep_semantic: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(deep_semantic, output_size=1)
        return torch.softmax(self.weight_selector(pooled), dim=1)

    def refine(self, z: torch.Tensor, deep_semantic: torch.Tensor) -> torch.Tensor:
        weights = self.branch_weights(deep_semantic)
        mixed = (
            weights[:, 0:1] * self.local(z)
            + weights[:, 1:2] * self.horizontal(z)
            + weights[:, 2:3] * self.vertical(z)
        )
        return z + self.drop_path(self.projection(self.activation(self.mix_norm(mixed))))

    def forward(self, deep: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        deep = F.interpolate(deep, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.refine(deep + skip, deep)


class SemanticDirectionalDetailGate(nn.Module):
    """Feature-only semantic and detached route-confidence detail selector."""

    def __init__(self, channels: int = 48) -> None:
        super().__init__()
        self.semantic_dw = nn.Conv2d(channels, channels, 7, padding=3, groups=channels, bias=False)
        self.semantic_projection = nn.Conv2d(channels, channels, kernel_size=1)
        self.activation = nn.SiLU()

    def forward(self, semantic_half: torch.Tensor, route_s1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        semantic_gate = torch.sigmoid(self.semantic_projection(self.activation(self.semantic_dw(semantic_half))))
        semantic_gate = 0.25 + 0.75 * semantic_gate
        route_probabilities = torch.softmax(route_s1.detach(), dim=1)
        entropy = -(route_probabilities * torch.log(route_probabilities.clamp_min(1e-6))).sum(dim=1, keepdim=True)
        confidence = 1.0 - entropy / torch.log(route_probabilities.new_tensor(4.0))
        confidence = F.interpolate(confidence, size=semantic_half.shape[-2:], mode="bilinear", align_corners=False)
        direction_gate = 0.5 + 0.5 * confidence
        return semantic_gate * direction_gate, confidence


class BottomUpSemanticVerification(nn.Module):
    """Verify selected half-resolution detail once in the quarter-scale semantic space."""

    def __init__(self, detail_channels: int = 48, semantic_channels: int = 96) -> None:
        super().__init__()
        self.downsample = nn.Conv2d(
            detail_channels,
            detail_channels,
            3,
            stride=2,
            padding=1,
            groups=detail_channels,
            bias=False,
        )
        self.projection = nn.Conv2d(detail_channels, semantic_channels, kernel_size=1, bias=False)
        self.refinement = CSDRFusion(semantic_channels)
        self.upsample = nn.Sequential(
            nn.Conv2d(semantic_channels, detail_channels * 4, kernel_size=1),
            nn.PixelShuffle(2),
        )

    def forward(self, fused_half: torch.Tensor, p1: torch.Tensor) -> torch.Tensor:
        verify_quarter = self.projection(self.downsample(fused_half))
        if verify_quarter.shape[-2:] != p1.shape[-2:]:
            verify_quarter = F.interpolate(
                verify_quarter,
                size=p1.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        verified_quarter = self.refinement.refine(p1 + verify_quarter, p1)
        verified_half = self.upsample(verified_quarter)
        if verified_half.shape[-2:] != fused_half.shape[-2:]:
            verified_half = F.interpolate(
                verified_half,
                size=fused_half.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return verified_half


class ResidualFinalRefineBlock(nn.Module):
    def __init__(self, channels: int, drop_path: float = 0.0) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.activation = nn.SiLU()
        self.projection = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop_path(self.projection(self.activation(self.norm(self.depthwise(x)))))


class CSDRDecoder(nn.Module):
    """Cross-scale semantic-detail reassembly replacing LitePro's decoder."""

    def __init__(
        self,
        encoder_channels: Sequence[int],
        decoder_channels: int = 96,
        detail_channels: int = 48,
        final_refine_channels: int = 64,
        drop_path_rate: float = 0.05,
    ) -> None:
        super().__init__()
        if tuple(int(value) for value in encoder_channels) != (48, 96, 192, 256):
            raise ValueError("CSDRDecoder requires encoder widths (48, 96, 192, 256).")
        if (decoder_channels, detail_channels, final_refine_channels) != (96, 48, 64):
            raise ValueError("CSDRDecoder requires decoder/detail/final widths of 96/48/64.")
        c1, c2, c3, c4 = (int(value) for value in encoder_channels)
        self.proj1 = ConvGNAct(c1, decoder_channels, kernel_size=1, padding=0)
        self.proj2 = ConvGNAct(c2, decoder_channels, kernel_size=1, padding=0)
        self.proj3 = ConvGNAct(c3, decoder_channels, kernel_size=1, padding=0)
        self.proj4 = ConvGNAct(c4, decoder_channels, kernel_size=1, padding=0)
        self.fuse3 = CSDRFusion(decoder_channels, drop_path=drop_path_rate * 0.5)
        self.fuse2 = CSDRFusion(decoder_channels, drop_path=drop_path_rate * 0.75)
        self.fuse1 = CSDRFusion(decoder_channels, drop_path=drop_path_rate)
        self.route_head = nn.Conv2d(decoder_channels, 4, kernel_size=1)
        self.semantic_upsample = nn.Sequential(
            nn.Conv2d(decoder_channels, detail_channels * 4, kernel_size=1),
            nn.PixelShuffle(2),
        )
        self.detail_preprocess = ConvGNAct(detail_channels, detail_channels, kernel_size=1, padding=0)
        self.detail_depthwise = nn.Conv2d(
            detail_channels, detail_channels, 3, padding=1, groups=detail_channels, bias=False
        )
        self.detail_gate = SemanticDirectionalDetailGate(detail_channels)
        self.bottom_up_verification = BottomUpSemanticVerification(detail_channels, decoder_channels)
        self.final_projection = ConvGNAct(detail_channels, final_refine_channels, kernel_size=1, padding=0)
        self.final_refine = nn.Sequential(
            ResidualFinalRefineBlock(final_refine_channels, drop_path=drop_path_rate * 0.5),
            ResidualFinalRefineBlock(final_refine_channels, drop_path=drop_path_rate),
        )
        self.mask_head = nn.Conv2d(final_refine_channels, 1, kernel_size=1)
        self.edge_head = nn.Conv2d(final_refine_channels, 1, kernel_size=1)
        self.reliability_head = nn.Conv2d(final_refine_channels, 1, kernel_size=1)

    @staticmethod
    def _resize_output(logits: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)

    def forward(
        self,
        features: Sequence[torch.Tensor],
        detail_half: torch.Tensor,
        output_size: tuple[int, int],
        *,
        return_debug: bool = False,
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        x1, x2, x3, x4 = features
        p4 = self.proj4(x4)
        p3 = self.fuse3(p4, self.proj3(x3))
        p2 = self.fuse2(p3, self.proj2(x2))
        p1 = self.fuse1(p2, self.proj1(x1))
        route_s1 = self.route_head(p1)
        semantic_half = self.semantic_upsample(p1)
        if semantic_half.shape[-2:] != detail_half.shape[-2:]:
            semantic_half = F.interpolate(
                semantic_half,
                size=detail_half.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        detail = self.detail_preprocess(detail_half)
        detail = detail + self.detail_depthwise(detail)
        detail_gate, route_confidence = self.detail_gate(semantic_half, route_s1)
        fused_half = semantic_half + detail_gate * detail
        verified_half = self.bottom_up_verification(fused_half, p1)
        joint_feature = self.final_refine(self.final_projection(fused_half + verified_half))
        outputs = {
            "mask": self._resize_output(self.mask_head(joint_feature), output_size),
            "edge": self._resize_output(self.edge_head(joint_feature), output_size),
            "reliability": self._resize_output(self.reliability_head(joint_feature), output_size),
            "route_s1": route_s1,
        }
        debug = {"detail_gate": detail_gate, "route_confidence": route_confidence}
        return (outputs, debug) if return_debug else outputs


class DEMambaNetLiteProCSDR(DEMambaNetLitePro):
    """LitePro backbone with a feature-level cross-scale semantic-detail decoder."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        encoder_widths: Sequence[int] = (48, 96, 192, 256),
        encoder_depths: Sequence[int] = (2, 2, 4, 2),
        decoder_channels: int = 96,
        detail_channels: int = 48,
        final_refine_channels: int = 64,
        directional_kernel: int = 13,
        expansion_ratio: float = 2.0,
        stage3_mamba_channels: int = 112,
        stage4_mamba_channels: int = 144,
        use_mamba_ssm: bool = True,
        drop_path_rate: float = 0.05,
    ) -> None:
        widths = tuple(int(value) for value in encoder_widths)
        depths = tuple(int(value) for value in encoder_depths)
        if widths != (48, 96, 192, 256) or depths != (2, 2, 4, 2):
            raise ValueError("DEMambaNetLiteProCSDR requires widths/depths 48,96,192,256 / 2,2,4,2.")
        if (decoder_channels, detail_channels, final_refine_channels) != (96, 48, 64):
            raise ValueError("DEMambaNetLiteProCSDR requires decoder/detail/final widths 96/48/64.")
        if (stage3_mamba_channels, stage4_mamba_channels) != (112, 144):
            raise ValueError("DEMambaNetLiteProCSDR requires Stage3/Stage4 Mamba widths 112/144.")
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
        self.stage3_context = SharedAxialMambaContext(widths[2], stage3_mamba_channels, use_mamba_ssm=use_mamba_ssm)
        del self.decoder
        self.csdr_decoder = CSDRDecoder(widths, decoder_channels, detail_channels, final_refine_channels, drop_path_rate)
        self.detail_channels = int(detail_channels)
        self.final_refine_channels = int(final_refine_channels)

    def extract_encoder_features(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        stem_half = self.stem_half(image)
        detail_half = self.detail_refine(stem_half + self.detail_branch(image, stem_half.shape[-2:]))
        f1 = self.stage1(self.stem_quarter(detail_half))
        f2 = self.stage2(self.down1(f1))
        f3, route_s3 = self.stage3_context(self.stage3(self.down2(f2)))
        f4, route_s4 = self.stage4_context(self.stage4(self.down3(f3)))
        return {"detail_half": detail_half, "f1": f1, "f2": f2, "f3": f3, "f4": f4, "route_s3": route_s3, "route_s4": route_s4}

    def forward(
        self, image: torch.Tensor, return_debug: bool = False
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        features = self.extract_encoder_features(image)
        result = self.csdr_decoder(
            [features["f1"], features["f2"], features["f3"], features["f4"]],
            features["detail_half"],
            image.shape[-2:],
            return_debug=return_debug,
        )
        if return_debug:
            outputs, debug = result
        else:
            outputs = result
        outputs["route_s3"] = features["route_s3"]
        outputs["route_s4"] = features["route_s4"]
        return (outputs, debug) if return_debug else outputs

    def mamba_backend_info(self) -> dict[str, list[str]]:
        return {"stage3_context": [self.stage3_context.backend], "stage4_context": [self.stage4_context.backend]}
