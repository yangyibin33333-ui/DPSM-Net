from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .de_mambanet_lite import ConvGNAct, DropPath, _group_count
from .de_mambanet_lite_pro import DEMambaNetLitePro, ProStructuralDecoder


class LightweightDirectionalRecoveryBlock(nn.Module):
    """Cooperative half-resolution directional recovery with a stable residual."""

    def __init__(self, channels: int, kernel_size: int = 9, drop_path: float = 0.0) -> None:
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("recovery kernel_size must be an odd integer of at least 3.")
        drop_path = float(drop_path)
        if not 0.0 <= drop_path <= 0.05 + 1e-8:
            raise ValueError("recovery drop_path must be in [0, 0.05].")
        drop_path = min(drop_path, 0.05)
        padding = kernel_size // 2
        self.horizontal = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, kernel_size),
            padding=(0, padding),
            groups=channels,
            bias=False,
        )
        self.vertical = nn.Conv2d(
            channels,
            channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            groups=channels,
            bias=False,
        )
        self.local = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.mix_norm = nn.GroupNorm(_group_count(channels), channels)
        self.activation = nn.SiLU()
        self.projection = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.output_norm = nn.GroupNorm(_group_count(channels), channels)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        directional = self.horizontal(x) + self.vertical(x) + self.local(x)
        residual = self.output_norm(self.projection(self.activation(self.mix_norm(directional))))
        return x + self.drop_path(residual)


class SemanticEdgeGuidedResidualRecovery(nn.Module):
    """Constrained semantic-edge guided logit recovery at half resolution."""

    def __init__(
        self,
        semantic_channels: int = 64,
        detail_channels: int = 40,
        recovery_channels: int = 48,
        recovery_depth: int = 2,
        recovery_kernel_size: int = 9,
        recovery_drop_path: float = 0.05,
        residual_scale_max: float = 3.0,
        detach_recovery_gate_inputs: bool = True,
    ) -> None:
        super().__init__()
        if recovery_channels < 8 or recovery_depth < 1:
            raise ValueError("recovery_channels must be at least 8 and recovery_depth must be positive.")
        if not 0.0 <= recovery_drop_path <= 0.05:
            raise ValueError("recovery_drop_path must be in [0, 0.05].")
        if residual_scale_max <= 1.0:
            raise ValueError("residual_scale_max must be greater than 1.0.")
        if not detach_recovery_gate_inputs:
            raise ValueError("SEGR requires detached base, edge, and route guidance inputs.")

        self.semantic_projection = ConvGNAct(
            semantic_channels,
            recovery_channels,
            kernel_size=1,
            padding=0,
        )
        self.detail_projection = ConvGNAct(
            detail_channels,
            recovery_channels,
            kernel_size=1,
            padding=0,
        )
        self.route_projection = nn.Conv2d(4, recovery_channels, kernel_size=1, bias=False)
        self.fusion_norm = nn.GroupNorm(_group_count(recovery_channels), recovery_channels)
        self.fusion_activation = nn.SiLU()
        drop_rates = (
            [float(recovery_drop_path)]
            if recovery_depth == 1
            else [
                recovery_drop_path * index / (recovery_depth - 1)
                for index in range(recovery_depth)
            ]
        )
        self.recovery_blocks = nn.Sequential(
            *(
                LightweightDirectionalRecoveryBlock(
                    recovery_channels,
                    kernel_size=recovery_kernel_size,
                    drop_path=float(drop_rates[index]),
                )
                for index in range(recovery_depth)
            )
        )
        self.delta_head = nn.Conv2d(recovery_channels, 1, kernel_size=1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

        self.residual_scale_max = float(residual_scale_max)
        initial_ratio = 1.0 / self.residual_scale_max
        initial_logit = math.log(initial_ratio / (1.0 - initial_ratio))
        self.scale_parameter = nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))
        self.detach_recovery_gate_inputs = True

    def residual_scale(self) -> torch.Tensor:
        return self.residual_scale_max * torch.sigmoid(self.scale_parameter)

    @staticmethod
    def recovery_gate(
        base_mask_logit_half: torch.Tensor,
        edge_logit_half: torch.Tensor,
    ) -> torch.Tensor:
        p_base = torch.sigmoid(base_mask_logit_half.detach())
        uncertainty = 4.0 * p_base * (1.0 - p_base)
        p_edge = torch.sigmoid(edge_logit_half.detach())
        missing_gate = p_edge * (1.0 - p_base)
        return torch.clamp(uncertainty + missing_gate, min=0.0, max=1.0)

    def forward(
        self,
        mask_feature_half: torch.Tensor,
        detail_feature_half: torch.Tensor,
        route_s1: torch.Tensor,
        base_mask_logit_half: torch.Tensor,
        edge_logit_half: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        route_probabilities = torch.softmax(route_s1.detach(), dim=1)
        route_probabilities = F.interpolate(
            route_probabilities,
            size=mask_feature_half.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        recovery_feature = (
            self.semantic_projection(mask_feature_half)
            + self.detail_projection(detail_feature_half)
            + self.route_projection(route_probabilities)
        )
        recovery_feature = self.fusion_activation(self.fusion_norm(recovery_feature))
        recovery_feature = self.recovery_blocks(recovery_feature)

        raw_delta = self.delta_head(recovery_feature)
        delta_logit = self.residual_scale() * torch.tanh(raw_delta)
        gate = self.recovery_gate(base_mask_logit_half, edge_logit_half)
        final_mask_logit_half = base_mask_logit_half + gate * delta_logit
        return final_mask_logit_half, {
            "base_mask_logit_half": base_mask_logit_half,
            "delta_logit": delta_logit,
            "recovery_gate": gate,
        }


class SEGRStructuralDecoder(ProStructuralDecoder):
    """Expose Pro's half-resolution features without changing its modules."""

    def forward_features(
        self,
        features: Sequence[torch.Tensor],
        detail_half: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x1, x2, x3, x4 = features
        p4 = self.proj4(x4)
        p3 = self.fuse3(p4, self.proj3(x3))
        p2 = self.fuse2(p3, self.proj2(x2))
        quarter_feature = self.final_refine(self.fuse1(p2, self.proj1(x1)))
        route_s1 = self.route_head(quarter_feature)
        half_feature = self.half_fusion(self.half_upsample(quarter_feature), detail_half)
        edge_logit_half = self.edge_head(half_feature)
        reliability_logit_half = self.reliability_head(half_feature)
        mask_feature_half = self.mask_refine(self.mask_projection(half_feature))
        base_mask_logit_half = self.mask_head(mask_feature_half)
        return {
            "route_s1": route_s1,
            "edge_logit_half": edge_logit_half,
            "reliability_logit_half": reliability_logit_half,
            "mask_feature_half": mask_feature_half,
            "base_mask_logit_half": base_mask_logit_half,
        }


class DEMambaNetLiteProSEGR(DEMambaNetLitePro):
    """LitePro with mask-only semantic-edge guided residual recovery."""

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
        recovery_channels: int = 48,
        recovery_depth: int = 2,
        recovery_kernel_size: int = 9,
        recovery_drop_path: float = 0.05,
        residual_scale_max: float = 3.0,
        detach_recovery_gate_inputs: bool = True,
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
        self.decoder = SEGRStructuralDecoder(encoder_widths, decoder_channels, mask_channels)
        self.segr = SemanticEdgeGuidedResidualRecovery(
            semantic_channels=mask_channels,
            detail_channels=detail_channels,
            recovery_channels=recovery_channels,
            recovery_depth=recovery_depth,
            recovery_kernel_size=recovery_kernel_size,
            recovery_drop_path=recovery_drop_path,
            residual_scale_max=residual_scale_max,
            detach_recovery_gate_inputs=detach_recovery_gate_inputs,
        )

    def forward(
        self,
        image: torch.Tensor,
        return_debug: bool = False,
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        output_size = image.shape[-2:]
        features = self.extract_encoder_features(image)
        decoded = self.decoder.forward_features(
            [features["f1"], features["f2"], features["f3"], features["f4"]],
            features["detail_half"],
        )
        final_mask_logit_half, debug = self.segr(
            decoded["mask_feature_half"],
            features["detail_half"],
            decoded["route_s1"],
            decoded["base_mask_logit_half"],
            decoded["edge_logit_half"],
        )
        outputs = {
            "mask": self.decoder._resize_output(final_mask_logit_half, output_size),
            "edge": self.decoder._resize_output(decoded["edge_logit_half"], output_size),
            "reliability": self.decoder._resize_output(decoded["reliability_logit_half"], output_size),
            "route_s1": decoded["route_s1"],
            "route_s3": features["route_s3"],
            "route_s4": features["route_s4"],
        }
        return (outputs, debug) if return_debug else outputs
