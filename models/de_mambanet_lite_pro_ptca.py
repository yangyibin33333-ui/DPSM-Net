"""Isolated Stage-3 transposed-channel-attention ablation for LitePro."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .de_mambanet_lite import _group_count
from .de_mambanet_lite_pro import DEMambaNetLitePro


class ParallelTransposedChannelAttention(nn.Module):
    """A compact channel supplement with no spatial HW-by-HW attention.

    Queries and keys are normalized across spatial positions, then their
    transposed product produces a per-head ``head_dim x head_dim`` matrix.
    This module deliberately has no residual connection: the parent model
    controls its only contribution through a zero-initialized scalar gate.
    """

    def __init__(
        self,
        in_channels: int = 160,
        attention_channels: int = 40,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        if int(in_channels) < 1:
            raise ValueError("PTCA in_channels must be positive.")
        if int(attention_channels) < 1:
            raise ValueError("PTCA attention_channels must be positive.")
        if int(num_heads) < 1:
            raise ValueError("PTCA num_heads must be positive.")
        if int(attention_channels) % int(num_heads) != 0:
            raise ValueError("PTCA attention_channels must be divisible by num_heads.")

        self.in_channels = int(in_channels)
        self.attention_channels = int(attention_channels)
        self.num_heads = int(num_heads)
        self.head_dim = self.attention_channels // self.num_heads
        self.input_norm = nn.GroupNorm(_group_count(self.in_channels), self.in_channels)
        self.qkv_projection = nn.Conv2d(
            self.in_channels,
            3 * self.attention_channels,
            kernel_size=1,
            bias=False,
        )
        self.qkv_depthwise = nn.Conv2d(
            3 * self.attention_channels,
            3 * self.attention_channels,
            kernel_size=3,
            padding=1,
            groups=3 * self.attention_channels,
            bias=False,
        )
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.output_projection = nn.Conv2d(
            self.attention_channels,
            self.in_channels,
            kernel_size=1,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, height, width = x.shape
        spatial_size = height * width
        qkv = self.qkv_depthwise(self.qkv_projection(self.input_norm(x)))
        qkv = qkv.reshape(
            batch_size,
            3,
            self.num_heads,
            self.head_dim,
            spatial_size,
        )
        query, key, value = qkv.unbind(dim=1)
        query = F.normalize(query, p=2.0, dim=-1)
        key = F.normalize(key, p=2.0, dim=-1)
        attention = (query @ key.transpose(-2, -1)) * self.temperature.unsqueeze(0)
        attention = attention.softmax(dim=-1)
        channel_feature = attention @ value
        channel_feature = channel_feature.reshape(
            batch_size,
            self.attention_channels,
            height,
            width,
        )
        output = self.output_projection(channel_feature)
        if return_attention:
            return output, attention
        return output


class DEMambaNetLiteProPTCA(DEMambaNetLitePro):
    """LitePro with one zero-gated PTCA supplement at Stage 3 only."""

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
        ptca_channels: int = 40,
        ptca_heads: int = 4,
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
        self.stage3_ptca = ParallelTransposedChannelAttention(
            in_channels=self.encoder_widths[2],
            attention_channels=ptca_channels,
            num_heads=ptca_heads,
        )
        # At initialization this makes Stage 3 exactly the LitePro Mamba path.
        self.alpha = nn.Parameter(torch.zeros(()))

    def extract_encoder_features(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        stem_half = self.stem_half(image)
        detail_half = self.detail_refine(
            stem_half + self.detail_branch(image, stem_half.shape[-2:]),
        )
        f1 = self.stage1(self.stem_quarter(detail_half))
        f2 = self.stage2(self.down1(f1))
        f3_local = self.stage3(self.down2(f2))
        f3_mamba, route_s3 = self.stage3_context(f3_local)
        f3_channel = self.stage3_ptca(f3_local)
        f3 = f3_mamba + (0.5 * torch.tanh(self.alpha)) * f3_channel
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
