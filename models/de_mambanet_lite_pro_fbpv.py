from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .de_mambanet_lite import ConvGNAct, DropPath, SharedAxialMambaContext, _group_count
from .de_mambanet_lite_balanced import DEMambaNetLiteBalanced
from .de_mambanet_lite_pro import DEMambaNetLitePro


class DepthwiseSeparableResidualBlock(nn.Module):
    def __init__(self, channels: int, drop_path: float = 0.0) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.activation = nn.SiLU()
        self.projection = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop_path(self.projection(self.activation(self.norm(self.depthwise(x)))))


class PerImageDualPrototype(nn.Module):
    """FP32 per-image foreground/background prototypes with stable learned anchors."""

    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.learned_fg_anchor = nn.Parameter(torch.randn(1, channels, 1, 1) * 0.02)
        self.learned_bg_anchor = nn.Parameter(torch.randn(1, channels, 1, 1) * 0.02)
        self.anchor_mix_parameter = nn.Parameter(torch.tensor(0.0))

    def anchor_mix(self) -> torch.Tensor:
        return 0.1 + 0.4 * torch.sigmoid(self.anchor_mix_parameter)

    def forward(self, proto_feature: torch.Tensor, coarse_logit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=proto_feature.device.type, enabled=False):
            feature = proto_feature.float()
            probability = torch.sigmoid(coarse_logit.detach().float())
            fg_weight = probability.square()
            bg_weight = (1.0 - probability).square()
            fg_image = (feature * fg_weight).sum(dim=(-2, -1), keepdim=True) / fg_weight.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            bg_image = (feature * bg_weight).sum(dim=(-2, -1), keepdim=True) / bg_weight.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            mix = self.anchor_mix().float()
            fg = (1.0 - mix) * fg_image + mix * self.learned_fg_anchor.float()
            bg = (1.0 - mix) * bg_image + mix * self.learned_bg_anchor.float()
            return F.normalize(fg, dim=1, eps=1e-6), F.normalize(bg, dim=1, eps=1e-6)


class MultiHeadPrototypeVerifier(nn.Module):
    def __init__(self, channels: int = 64, heads: int = 4) -> None:
        super().__init__()
        if channels % heads != 0:
            raise ValueError("verification channels must divide evenly across heads.")
        self.channels, self.heads, self.head_channels = channels, heads, channels // heads
        self.temperature_parameter = nn.Parameter(torch.tensor(0.0))

    def temperature(self) -> torch.Tensor:
        return 0.10 + 0.40 * torch.sigmoid(self.temperature_parameter)

    def forward(self, query: torch.Tensor, fg_proto: torch.Tensor, bg_proto: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = query.shape
        query = F.normalize(query.reshape(batch, self.heads, self.head_channels, height, width), dim=2, eps=1e-6)
        fg = F.normalize(fg_proto.reshape(batch, self.heads, self.head_channels, 1, 1), dim=2, eps=1e-6)
        bg = F.normalize(bg_proto.reshape(batch, self.heads, self.head_channels, 1, 1), dim=2, eps=1e-6)
        margin = ((query * fg).sum(dim=2) - (query * bg).sum(dim=2)).mean(dim=1, keepdim=True)
        return torch.sigmoid(margin / self.temperature())


class PrototypeVerifiedMaskHead(nn.Module):
    def __init__(self, channels: int = 64, drop_path_rate: float = 0.05) -> None:
        super().__init__()
        self.context_projection = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.refine = nn.Sequential(
            DepthwiseSeparableResidualBlock(channels, drop_path=drop_path_rate * 0.5),
            DepthwiseSeparableResidualBlock(channels, drop_path=drop_path_rate),
        )
        self.detail_delta_head = nn.Conv2d(channels, 1, kernel_size=1)
        nn.init.zeros_(self.detail_delta_head.weight)
        nn.init.zeros_(self.detail_delta_head.bias)
        self.correction_scale_parameter = nn.Parameter(torch.tensor(0.0))

    def correction_scale(self) -> torch.Tensor:
        return 1.5 * torch.tanh(self.correction_scale_parameter)

    def forward(self, query: torch.Tensor, verification: torch.Tensor, fg: torch.Tensor, bg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_scale = 0.75 + 0.50 * verification
        context = verification * fg + (1.0 - verification) * bg
        verified = self.refine(query * feature_scale + 0.25 * self.context_projection(context))
        return self.detail_delta_head(verified), feature_scale


class FBPVDecoder(nn.Module):
    def __init__(self, encoder_channels: Sequence[int], drop_path_rate: float = 0.05) -> None:
        super().__init__()
        if tuple(int(v) for v in encoder_channels) != (48, 96, 192, 256):
            raise ValueError("FBPVDecoder requires widths (48, 96, 192, 256).")
        c1, c2, c3, c4 = (int(v) for v in encoder_channels)
        self.proj1, self.proj2 = ConvGNAct(c1, 96, 1, padding=0), ConvGNAct(c2, 96, 1, padding=0)
        self.proj3, self.proj4 = ConvGNAct(c3, 96, 1, padding=0), ConvGNAct(c4, 96, 1, padding=0)
        self.refine3 = DepthwiseSeparableResidualBlock(96, drop_path=drop_path_rate * 0.5)
        self.refine2 = DepthwiseSeparableResidualBlock(96, drop_path=drop_path_rate * 0.75)
        self.refine1 = nn.Sequential(DepthwiseSeparableResidualBlock(96, drop_path=drop_path_rate * 0.5), DepthwiseSeparableResidualBlock(96, drop_path=drop_path_rate))
        self.route_head = nn.Conv2d(96, 4, kernel_size=1)
        self.proto_feature = nn.Sequential(ConvGNAct(96, 64, 1, padding=0), DepthwiseSeparableResidualBlock(64))
        self.coarse_mask_head = nn.Conv2d(64, 1, kernel_size=1)
        self.per_image_prototypes = PerImageDualPrototype(64)
        self.semantic_upsample = nn.Sequential(nn.Conv2d(96, 48 * 4, 1), nn.PixelShuffle(2))
        self.detail_preprocess = ConvGNAct(48, 48, 1, padding=0)
        self.detail_depthwise = nn.Conv2d(48, 48, 3, padding=1, groups=48, bias=False)
        self.query_projection = nn.Sequential(ConvGNAct(48, 64, 1, padding=0), DepthwiseSeparableResidualBlock(64))
        self.prototype_verifier = MultiHeadPrototypeVerifier(64, 4)
        self.prototype_mask_head = PrototypeVerifiedMaskHead(64, drop_path_rate)
        self.edge_head, self.reliability_head = nn.Conv2d(64, 1, 1), nn.Conv2d(64, 1, 1)

    @staticmethod
    def resize(logit: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(logit, size=size, mode="bilinear", align_corners=False)

    def _add_refine(self, deep: torch.Tensor, skip: torch.Tensor, refine: nn.Module) -> torch.Tensor:
        return refine(F.interpolate(deep, size=skip.shape[-2:], mode="bilinear", align_corners=False) + skip)

    def forward(self, features: Sequence[torch.Tensor], detail_half: torch.Tensor, output_size: tuple[int, int], *, return_debug: bool = False):
        x1, x2, x3, x4 = features
        d4 = self.proj4(x4)
        d3 = self._add_refine(d4, self.proj3(x3), self.refine3)
        d2 = self._add_refine(d3, self.proj2(x2), self.refine2)
        d1 = self._add_refine(d2, self.proj1(x1), self.refine1)
        route_s1 = self.route_head(d1)
        proto = self.proto_feature(d1)
        coarse = self.coarse_mask_head(proto)
        fg, bg = self.per_image_prototypes(proto, coarse)
        semantic_half = self.semantic_upsample(d1)
        if semantic_half.shape[-2:] != detail_half.shape[-2:]:
            semantic_half = F.interpolate(semantic_half, size=detail_half.shape[-2:], mode="bilinear", align_corners=False)
        detail = self.detail_preprocess(detail_half)
        detail = detail + self.detail_depthwise(detail)
        query = self.query_projection(semantic_half + detail)
        verification = self.prototype_verifier(query, fg, bg)
        delta, feature_scale = self.prototype_mask_head(query, verification, fg, bg)
        base = F.interpolate(coarse, size=query.shape[-2:], mode="bilinear", align_corners=False)
        final = base + delta + self.prototype_mask_head.correction_scale() * (2.0 * verification - 1.0)
        outputs = {"mask": self.resize(final, output_size), "edge": self.resize(self.edge_head(query), output_size), "reliability": self.resize(self.reliability_head(query), output_size), "route_s1": route_s1}
        debug = {"coarse_mask_logit_quarter": coarse, "fg_proto": fg, "bg_proto": bg, "verification_probability": verification, "feature_scale": feature_scale}
        return (outputs, debug) if return_debug else outputs


class DEMambaNetLiteProFBPV(DEMambaNetLitePro):
    """LitePro backbone with per-image foreground/background prototype verification."""
    def __init__(self, in_channels: int = 3, num_classes: int = 1, encoder_widths: Sequence[int] = (48, 96, 192, 256), encoder_depths: Sequence[int] = (2, 2, 4, 2), decoder_channels: int = 96, detail_channels: int = 48, verification_channels: int = 64, directional_kernel: int = 13, expansion_ratio: float = 2.0, stage3_mamba_channels: int = 112, stage4_mamba_channels: int = 144, use_mamba_ssm: bool = True, drop_path_rate: float = 0.05) -> None:
        widths, depths = tuple(int(v) for v in encoder_widths), tuple(int(v) for v in encoder_depths)
        if widths != (48, 96, 192, 256) or depths != (2, 2, 4, 2) or (decoder_channels, detail_channels, verification_channels) != (96, 48, 64) or (stage3_mamba_channels, stage4_mamba_channels) != (112, 144):
            raise ValueError("DEMambaNetLiteProFBPV requires the configured FBPV structural widths.")
        DEMambaNetLiteBalanced.__init__(self, in_channels=in_channels, num_classes=num_classes, encoder_widths=widths, encoder_depths=depths, decoder_channels=decoder_channels, directional_kernel=directional_kernel, expansion_ratio=expansion_ratio, mamba_channels=stage4_mamba_channels, use_mamba_ssm=use_mamba_ssm, drop_path_rate=drop_path_rate)
        self.stage4_context = self.global_context
        del self.global_context
        self.stage3_context = SharedAxialMambaContext(widths[2], stage3_mamba_channels, use_mamba_ssm=use_mamba_ssm)
        del self.decoder
        self.decoder = FBPVDecoder(widths, drop_path_rate)

    def extract_encoder_features(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        stem_half = self.stem_half(image)
        detail = self.detail_refine(stem_half + self.detail_branch(image, stem_half.shape[-2:]))
        f1 = self.stage1(self.stem_quarter(detail)); f2 = self.stage2(self.down1(f1)); f3, r3 = self.stage3_context(self.stage3(self.down2(f2))); f4, r4 = self.stage4_context(self.stage4(self.down3(f3)))
        return {"detail_half": detail, "f1": f1, "f2": f2, "f3": f3, "f4": f4, "route_s3": r3, "route_s4": r4}

    def forward(self, image: torch.Tensor, return_debug: bool = False):
        features = self.extract_encoder_features(image)
        result = self.decoder([features["f1"], features["f2"], features["f3"], features["f4"]], features["detail_half"], image.shape[-2:], return_debug=return_debug)
        if return_debug: outputs, debug = result
        else: outputs = result
        outputs["route_s3"], outputs["route_s4"] = features["route_s3"], features["route_s4"]
        return (outputs, debug) if return_debug else outputs

    def mamba_backend_info(self) -> dict[str, list[str]]:
        return {"stage3_context": [self.stage3_context.backend], "stage4_context": [self.stage4_context.backend]}
