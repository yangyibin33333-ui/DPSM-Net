"""Local semantic-guided detail reassembly decoder for LitePro."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .de_mambanet_lite_pro import DEMambaNetLitePro, ProStructuralDecoder


class DetailDownsample(nn.Module):
    """Bring half-resolution detail into the quarter-resolution LSRA space."""

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
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.depthwise(x))


class LocalSemanticAttention(nn.Module):
    """Use only a fixed local neighborhood for semantic-detail reassembly."""

    def __init__(self, channels: int, heads: int, kernel_size: int) -> None:
        super().__init__()
        if int(channels) < 1 or int(heads) < 1:
            raise ValueError("LSRA channels and heads must be positive.")
        if int(channels) % int(heads) != 0:
            raise ValueError("LSRA channels must be divisible by heads.")
        if int(kernel_size) < 1 or int(kernel_size) % 2 == 0:
            raise ValueError("LSRA kernel_size must be a positive odd number.")
        self.channels = int(channels)
        self.heads = int(heads)
        self.head_dim = self.channels // self.heads
        self.kernel_size = int(kernel_size)
        self.neighborhood = self.kernel_size * self.kernel_size
        self.query_projection = nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False)
        self.key_projection = nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False)
        self.value_projection = nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False)
        self.relative_bias = nn.Parameter(torch.zeros(self.heads, self.neighborhood))
        center_index = self.neighborhood // 2
        with torch.no_grad():
            self.relative_bias[:, center_index] = 2.0

    def forward(
        self,
        semantic_q: torch.Tensor,
        detail_q: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, height, width = semantic_q.shape
        if detail_q.shape[-2:] != (height, width):
            raise ValueError("LSRA semantic and detail features must share the quarter-resolution size.")
        spatial_size = height * width
        query = self.query_projection(semantic_q).reshape(
            batch_size, self.heads, self.head_dim, spatial_size
        )
        key = F.unfold(
            self.key_projection(detail_q),
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        ).reshape(batch_size, self.heads, self.head_dim, self.neighborhood, spatial_size)
        value = F.unfold(
            self.value_projection(detail_q),
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        ).reshape(batch_size, self.heads, self.head_dim, self.neighborhood, spatial_size)
        query = F.normalize(query, p=2.0, dim=2)
        key = F.normalize(key, p=2.0, dim=2)
        score = (query.unsqueeze(3) * key).sum(dim=2) / math.sqrt(self.head_dim)
        score = score + self.relative_bias.unsqueeze(0).unsqueeze(-1)
        attention = torch.softmax(score.float(), dim=2)
        reassembled = (attention.to(dtype=value.dtype).unsqueeze(2) * value).sum(dim=3)
        reassembled = reassembled.reshape(batch_size, self.channels, height, width)
        if return_attention:
            return reassembled, attention
        return reassembled


class LocalSemanticReassemblyAttention(nn.Module):
    """Quarter-scale local semantic-detail reassembly followed by 2x recovery."""

    def __init__(
        self,
        semantic_channels: int = 80,
        detail_channels: int = 40,
        local_channels: int = 24,
        heads: int = 4,
        kernel_size: int = 3,
        initial_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if not 0.0 < float(initial_scale) < 0.5:
            raise ValueError("LSRA initial_scale must be strictly between 0 and 0.5.")
        self.semantic_channels = int(semantic_channels)
        self.detail_channels = int(detail_channels)
        self.local_channels = int(local_channels)
        self.semantic_projection = nn.Conv2d(
            self.semantic_channels, self.local_channels, kernel_size=1, bias=False
        )
        self.detail_downsample = DetailDownsample(self.detail_channels, self.local_channels)
        self.local_attention = LocalSemanticAttention(self.local_channels, heads, kernel_size)
        self.upsample = nn.Sequential(
            nn.Conv2d(self.local_channels, self.detail_channels * 4, kernel_size=1, bias=False),
            nn.PixelShuffle(2),
        )
        self.refine = nn.Conv2d(
            self.detail_channels,
            self.detail_channels,
            kernel_size=3,
            padding=1,
            groups=self.detail_channels,
            bias=False,
        )
        self.output_projection = nn.Conv2d(
            self.detail_channels, self.detail_channels, kernel_size=1, bias=False
        )
        # 0.5 * sigmoid(logit(2 * 0.1)) == 0.1 at initialization.
        self.scale_parameter = nn.Parameter(
            torch.tensor(math.log((2.0 * float(initial_scale)) / (1.0 - 2.0 * float(initial_scale))))
        )

    def residual_scale(self) -> torch.Tensor:
        return 0.5 * torch.sigmoid(self.scale_parameter)

    def forward(
        self,
        quarter_feature: torch.Tensor,
        detail_half: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        semantic_q = self.semantic_projection(quarter_feature)
        detail_q = self.detail_downsample(detail_half)
        if detail_q.shape[-2:] != semantic_q.shape[-2:]:
            detail_q = F.interpolate(detail_q, size=semantic_q.shape[-2:], mode="bilinear", align_corners=False)
        reassembled = self.local_attention(
            semantic_q,
            detail_q,
            return_attention=return_attention,
        )
        if return_attention:
            reassembled, attention = reassembled
        local_detail_half = self.output_projection(self.refine(self.upsample(reassembled)))
        if local_detail_half.shape[-2:] != detail_half.shape[-2:]:
            local_detail_half = F.interpolate(
                local_detail_half,
                size=detail_half.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return (local_detail_half, attention) if return_attention else local_detail_half


class LSRAStructuralDecoder(ProStructuralDecoder):
    """LitePro decoder with only its half-resolution fusion augmented by LSRA."""

    def __init__(
        self,
        encoder_channels: Sequence[int],
        decoder_channels: int,
        mask_channels: int,
        lsra_channels: int,
        lsra_heads: int,
        lsra_kernel_size: int,
        lsra_initial_scale: float,
    ) -> None:
        super().__init__(encoder_channels, decoder_channels, mask_channels)
        self.lsra = LocalSemanticReassemblyAttention(
            semantic_channels=decoder_channels,
            detail_channels=int(encoder_channels[0]),
            local_channels=lsra_channels,
            heads=lsra_heads,
            kernel_size=lsra_kernel_size,
            initial_scale=lsra_initial_scale,
        )

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
        quarter_feature = self.final_refine(self.fuse1(p2, self.proj1(x1)))
        route_s1 = self.route_head(quarter_feature)
        semantic_half = self.half_upsample(quarter_feature)
        base_half = self.half_fusion(semantic_half, detail_half)
        local_result = self.lsra(quarter_feature, detail_half, return_attention=return_debug)
        if return_debug:
            local_detail_half, attention = local_result
        else:
            local_detail_half = local_result
        correction = self.lsra.residual_scale() * local_detail_half
        half_feature = base_half + correction
        edge = self.edge_head(half_feature)
        reliability = self.reliability_head(half_feature)
        mask_feature = self.mask_refine(self.mask_projection(half_feature))
        outputs = {
            "mask": self._resize_output(self.mask_head(mask_feature), output_size),
            "edge": self._resize_output(edge, output_size),
            "reliability": self._resize_output(reliability, output_size),
            "route_s1": route_s1,
        }
        if not return_debug:
            return outputs

        center_index = self.lsra.local_attention.neighborhood // 2
        entropy = -(attention * attention.clamp_min(1e-8).log()).sum(dim=2).mean()
        debug = {
            "residual_scale": self.lsra.residual_scale().detach(),
            "attention_center_mean": attention[:, :, center_index, :].mean().detach(),
            "attention_entropy": entropy.detach(),
            "correction_to_base_norm_ratio": (
                correction.norm() / base_half.norm().clamp_min(1e-6)
            ).detach(),
        }
        return outputs, debug


class DEMambaNetLiteProLSRA(DEMambaNetLitePro):
    """LitePro backbone with a local-only semantic-guided decoder reassembly."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        encoder_widths: Sequence[int] = DEMambaNetLitePro.EXPECTED_WIDTHS,
        encoder_depths: Sequence[int] = DEMambaNetLitePro.EXPECTED_DEPTHS,
        decoder_channels: int = 80,
        detail_channels: int = 40,
        mask_channels: int = 64,
        directional_kernel: int = 13,
        expansion_ratio: float = 2.0,
        stage3_mamba_channels: int = 96,
        stage4_mamba_channels: int = 128,
        lsra_channels: int = 24,
        lsra_heads: int = 4,
        lsra_kernel_size: int = 3,
        lsra_initial_scale: float = 0.1,
        use_mamba_ssm: bool = True,
        drop_path_rate: float = 0.05,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            encoder_widths=encoder_widths,
            encoder_depths=encoder_depths,
            decoder_channels=decoder_channels,
            detail_channels=detail_channels,
            mask_channels=mask_channels,
            directional_kernel=directional_kernel,
            expansion_ratio=expansion_ratio,
            stage3_mamba_channels=stage3_mamba_channels,
            stage4_mamba_channels=stage4_mamba_channels,
            use_mamba_ssm=use_mamba_ssm,
            drop_path_rate=drop_path_rate,
        )
        self.decoder = LSRAStructuralDecoder(
            self.encoder_widths,
            decoder_channels,
            mask_channels,
            lsra_channels,
            lsra_heads,
            lsra_kernel_size,
            lsra_initial_scale,
        )

    def forward(
        self,
        image: torch.Tensor,
        *,
        return_debug: bool = False,
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        features = self.extract_encoder_features(image)
        result = self.decoder(
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
