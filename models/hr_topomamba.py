"""High-resolution detail/semantic exchange variant of the Native encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import CNNEdgeStem, ConvBNAct, LocalCrackBlock, TopoRouteBlock
from .decoder import BoundaryAwareDecoder


class DetailToSemanticExchange(nn.Module):
    """Feed only local half-resolution detail into the stage-3 semantic stream."""

    def __init__(self, detail_channels: int, semantic_channels: int):
        super().__init__()
        self.downsample = nn.Conv2d(
            detail_channels,
            detail_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=detail_channels,
            bias=False,
        )
        self.project = nn.Conv2d(detail_channels, semantic_channels, kernel_size=1, bias=False)
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, detail: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
        delta = self.project(self.downsample(detail))
        if delta.shape != semantic.shape:
            raise ValueError(
                "DetailToSemanticExchange requires a stride-two detail feature "
                "that projects exactly to the semantic feature shape."
            )
        return semantic + torch.tanh(self.alpha) * delta


class SemanticToDetailExchange(nn.Module):
    """Inject one semantic source with a local, target-conditioned gate."""

    def __init__(self, target_channels: int, source_channels: int):
        super().__init__()
        hidden_channels = max(8, target_channels // 4)
        self.source_projection = nn.Conv2d(source_channels, target_channels, kernel_size=1, bias=False)
        self.gate_context = nn.Conv2d(target_channels * 2, hidden_channels, kernel_size=1, bias=False)
        self.gate_norm = nn.GroupNorm(1, hidden_channels)
        self.gate_activation = nn.GELU()
        self.gate_out = nn.Conv2d(hidden_channels, target_channels, kernel_size=1)
        self.alpha = nn.Parameter(torch.zeros(()))

    def projected_source(self, target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        if source.shape[-2:] != target.shape[-2:]:
            source = F.interpolate(source, size=target.shape[-2:], mode="bilinear", align_corners=False)
        return self.source_projection(source)

    def residual(self, target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        projected = self.projected_source(target, source)
        gate = torch.sigmoid(self.gate_out(self.gate_activation(self.gate_norm(self.gate_context(torch.cat([target, projected], dim=1))))))
        return torch.tanh(self.alpha) * gate * projected

    def forward(self, target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        return target + self.residual(target, source)


class HRTopoMambaNet(nn.Module):
    """Native C40 encoder with zero-initialized H/2 detail/semantic exchanges."""

    _HR_TOPO_KEYS = {
        "detail_to_semantic",
        "semantic_to_detail_from_stage3",
        "semantic_to_detail_from_stage4",
        "zero_initialized_residual_scales",
    }

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 40,
        use_mamba_ssm: bool = True,
        use_transformer: bool | None = None,
        mamba_stages: list[int] | tuple[int, ...] = (3, 4),
        transformer_stages: list[int] | tuple[int, ...] = (4,),
        route_bins: int = 4,
        use_frequency_gate: bool = False,
        use_semantic_skip_gate: bool = False,
        use_detail_residual_gate: bool = False,
        use_structure_harmonic_fusion: bool = False,
        use_boundary_logit_calibration: bool = False,
        use_local_route_gate: bool = False,
        use_deep_identity_decay: bool = False,
        deep_identity_scales: dict[str, float] | None = None,
        hr_topo: dict[str, bool] | None = None,
    ):
        super().__init__()
        if num_classes != 1:
            raise ValueError("HRTopoMambaNet currently implements binary crack segmentation.")
        disabled_features = {
            "use_frequency_gate": use_frequency_gate,
            "use_semantic_skip_gate": use_semantic_skip_gate,
            "use_detail_residual_gate": use_detail_residual_gate,
            "use_structure_harmonic_fusion": use_structure_harmonic_fusion,
            "use_boundary_logit_calibration": use_boundary_logit_calibration,
            "use_local_route_gate": use_local_route_gate,
            "use_deep_identity_decay": use_deep_identity_decay,
        }
        enabled_features = [name for name, enabled in disabled_features.items() if enabled]
        if enabled_features:
            raise ValueError(
                "HRTopoMambaNet is an independent backbone and does not support "
                f"stacking existing optional features: {', '.join(enabled_features)}."
            )
        if deep_identity_scales not in (None, {"stage3": 1.0, "stage4": 1.0}):
            raise ValueError("HRTopoMambaNet requires the Native identity scales of 1.0.")
        configured_hr_topo = {
            "detail_to_semantic": True,
            "semantic_to_detail_from_stage3": True,
            "semantic_to_detail_from_stage4": True,
            "zero_initialized_residual_scales": True,
        }
        if hr_topo is not None:
            unexpected = set(hr_topo) - self._HR_TOPO_KEYS
            if unexpected:
                raise ValueError(f"Unsupported hr_topo keys: {', '.join(sorted(unexpected))}.")
            configured_hr_topo.update(hr_topo)
        disabled_exchanges = [name for name, enabled in configured_hr_topo.items() if not enabled]
        if disabled_exchanges:
            raise ValueError(
                "HRTopoMambaNet requires all three exchanges and zero-initialized "
                f"residual scales: {', '.join(disabled_exchanges)}."
            )
        self.hr_topo = configured_hr_topo
        self.mamba_stages = set(mamba_stages)
        self.transformer_stages = set(transformer_stages)
        if use_transformer is False:
            self.transformer_stages.clear()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8

        # Preserve the Native component creation order before adding exchanges.
        self.stem = CNNEdgeStem(in_channels, c1)
        self.stage1, self.route_stage1 = self._make_stage(c1, 1, route_bins, use_mamba_ssm)
        self.down1 = ConvBNAct(c1, c2, stride=2)
        self.stage2, self.route_stage2 = self._make_stage(c2, 2, route_bins, use_mamba_ssm)
        self.down2 = ConvBNAct(c2, c3, stride=2)
        self.stage3, self.route_stage3 = self._make_stage(c3, 3, route_bins, use_mamba_ssm)
        self.down3 = ConvBNAct(c3, c4, stride=2)
        self.stage4, self.route_stage4 = self._make_stage(c4, 4, route_bins, use_mamba_ssm)
        self.decoder = BoundaryAwareDecoder([c1, c2, c3, c4], c1)

        # These exchanges are intentionally created after the full Native core.
        self.detail_to_semantic = DetailToSemanticExchange(c2, c3)
        self.semantic3_to_detail = SemanticToDetailExchange(c2, c3)
        self.semantic4_to_detail = SemanticToDetailExchange(c2, c4)

    def _make_stage(
        self,
        channels: int,
        stage: int,
        route_bins: int,
        use_mamba_ssm: bool,
    ) -> tuple[nn.Module, bool]:
        use_transformer = stage in self.transformer_stages
        if stage in self.mamba_stages:
            return TopoRouteBlock(channels, route_bins, use_transformer, use_mamba_ssm), True
        return LocalCrackBlock(channels, channels, use_transformer), False

    @staticmethod
    def _run_stage(stage: nn.Module, routed: bool, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if routed:
            return stage(x)  # type: ignore[return-value]
        return stage(x), None

    def mamba_backend_info(self) -> dict[str, list[str]]:
        info: dict[str, list[str]] = {}
        for index in range(1, 5):
            stage = getattr(self, f"stage{index}")
            if getattr(self, f"route_stage{index}"):
                info[f"stage{index}"] = stage.mamba_backends  # type: ignore[attr-defined]
        return info

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        output_size = x.shape[-2:]
        f1, route_s1 = self._run_stage(self.stage1, self.route_stage1, self.stem(x))
        f2, route_s2 = self._run_stage(self.stage2, self.route_stage2, self.down1(f1))
        f3, route_s3 = self._run_stage(self.stage3, self.route_stage3, self.down2(f2))
        f3_refined = self.detail_to_semantic(f2, f3)
        f4, route_s4 = self._run_stage(self.stage4, self.route_stage4, self.down3(f3_refined))

        # Both semantic-to-detail exchanges condition on the original f2 target.
        f2_refined = f2 + self.semantic3_to_detail.residual(f2, f3_refined) + self.semantic4_to_detail.residual(f2, f4)
        outputs = self.decoder([f1, f2_refined, f3_refined, f4], output_size=output_size)
        for name, route_logits in (("route_s1", route_s1), ("route_s2", route_s2), ("route_s3", route_s3), ("route_s4", route_s4)):
            if route_logits is not None:
                outputs[name] = route_logits
        return outputs
