from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .mamba_blocks import SequenceMambaCore


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvGNAct(nn.Module):
    """Small-batch-stable convolution used throughout the lightweight model."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        activate: bool = True,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        ]
        if activate:
            layers.append(nn.SiLU())
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DropPath(nn.Module):
    """Per-sample stochastic depth with zero inference overhead."""

    def __init__(self, probability: float) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError(f"drop-path probability must be in [0, 1), got {probability}.")
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0.0:
            return x
        keep_probability = 1.0 - self.probability
        shape = (x.size(0),) + (1,) * (x.ndim - 1)
        keep = x.new_empty(shape).bernoulli_(keep_probability)
        return x * keep / keep_probability


class StructureDirectionalBlock(nn.Module):
    """Cooperatively model local evidence and moderate directional continuity.

    The block deliberately uses additive directions instead of competitive
    routing. A local 3x3 response supports horizontal, vertical, and dilated
    oblique depth-wise responses before a single learned gate suppresses
    background coupling.
    """

    def __init__(
        self,
        channels: int,
        expansion_ratio: float = 2.0,
        directional_kernel: int = 13,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        if expansion_ratio < 1.0:
            raise ValueError("StructureDirectionalBlock expansion_ratio must be at least 1.0.")
        if directional_kernel < 3 or directional_kernel % 2 == 0:
            raise ValueError("directional_kernel must be an odd integer of at least 3.")
        hidden_channels = max(channels, int(round(channels * expansion_ratio)))
        dilation = max(1, directional_kernel // 4)

        self.input_norm = nn.GroupNorm(_group_count(channels), channels)
        self.input_projection = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False)
        self.local = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
            bias=False,
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
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expanded = self.input_projection(self.input_norm(x))
        local = self.local(expanded)
        directional = self.horizontal(expanded) + self.vertical(expanded) + self.oblique(expanded)
        gate = torch.sigmoid(self.direction_gate(local + directional))
        fused = local + gate * directional
        residual = self.output_projection(self.activation(self.output_norm(fused)))
        return x + self.drop_path(residual)


class SharedAxialMambaContext(nn.Module):
    """Two geometry-aligned traversals sharing one low-resolution Mamba core.

    Four supervised route bins are retained. Horizontal and vertical bins
    select their matching traversal, while each diagonal bin contributes half
    to both traversals. This avoids diagonal serialization and duplicate SSM
    parameters without discarding the original direction semantics.
    """

    active_mamba_directions = ("horizontal", "vertical")
    shared_core = True
    scan_count = 2

    def __init__(self, channels: int, mamba_channels: int, use_mamba_ssm: bool = True) -> None:
        super().__init__()
        hidden = max(16, channels // 4)
        self.route_head = nn.Sequential(
            ConvGNAct(channels, hidden, kernel_size=3),
            nn.Conv2d(hidden, 4, kernel_size=1),
        )
        self.input_projection = ConvGNAct(channels, mamba_channels, kernel_size=1, padding=0)
        self.core = SequenceMambaCore(mamba_channels, use_mamba_ssm=use_mamba_ssm)
        self.output_projection = ConvGNAct(mamba_channels, channels, kernel_size=1, padding=0, activate=False)

    @property
    def backend(self) -> str:
        return self.core.backend

    @staticmethod
    def axial_weights(route_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probabilities = torch.softmax(route_logits, dim=1)
        diagonal_support = 0.5 * (probabilities[:, 2:3] + probabilities[:, 3:4])
        return probabilities[:, 0:1] + diagonal_support, probabilities[:, 1:2] + diagonal_support

    def _scan(self, x: torch.Tensor, direction: str) -> torch.Tensor:
        batch, channels, height, width = x.shape
        if direction == "horizontal":
            sequence = x.permute(0, 2, 3, 1).reshape(batch * height, width, channels)
            output = self.core(sequence).reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        elif direction == "vertical":
            sequence = x.permute(0, 3, 2, 1).reshape(batch * width, height, channels)
            output = self.core(sequence).reshape(batch, width, height, channels).permute(0, 3, 2, 1)
        else:  # pragma: no cover - private method is called only with fixed directions.
            raise ValueError(f"unsupported axial direction: {direction!r}")
        return output.contiguous()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        route_logits = self.route_head(x)
        horizontal_weight, vertical_weight = self.axial_weights(route_logits)
        projected = self.input_projection(x)
        horizontal = self._scan(projected, "horizontal")
        vertical = self._scan(projected, "vertical")
        fused = horizontal_weight * horizontal + vertical_weight * vertical
        return x + self.output_projection(fused), route_logits


class DepthwiseSeparableRefine(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = ConvGNAct(channels, channels, kernel_size=3, groups=channels)
        self.pointwise = ConvGNAct(channels, channels, kernel_size=1, padding=0, activate=False)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.pointwise(self.depthwise(x)))


class GatedScaleFusion(nn.Module):
    """Inject a lateral structure only when supported by deeper context."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(8, channels // 4)
        self.gate = nn.Sequential(
            ConvGNAct(channels * 2, hidden, kernel_size=1, padding=0),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.refine = DepthwiseSeparableRefine(channels)

    def forward(self, high: torch.Tensor, lateral: torch.Tensor) -> torch.Tensor:
        high = F.interpolate(high, size=lateral.shape[-2:], mode="bilinear", align_corners=False)
        gate = self.gate(torch.cat([high, lateral], dim=1))
        return self.refine(high + gate * lateral)


class LightweightStructuralDecoder(nn.Module):
    """Quarter-resolution gated additive decoder with shared prediction trunk."""

    def __init__(
        self,
        encoder_channels: Sequence[int],
        decoder_channels: int,
        use_decoder_route_supervision: bool = True,
    ) -> None:
        super().__init__()
        if len(encoder_channels) != 4:
            raise ValueError("LightweightStructuralDecoder requires four encoder channel widths.")
        c1, c2, c3, c4 = (int(value) for value in encoder_channels)
        self.proj1 = ConvGNAct(c1, decoder_channels, kernel_size=1, padding=0)
        self.proj2 = ConvGNAct(c2, decoder_channels, kernel_size=1, padding=0)
        self.proj3 = ConvGNAct(c3, decoder_channels, kernel_size=1, padding=0)
        self.proj4 = ConvGNAct(c4, decoder_channels, kernel_size=1, padding=0)
        self.fuse3 = GatedScaleFusion(decoder_channels)
        self.fuse2 = GatedScaleFusion(decoder_channels)
        self.fuse1 = GatedScaleFusion(decoder_channels)
        self.final_refine = DepthwiseSeparableRefine(decoder_channels)
        self.mask_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        self.edge_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        self.reliability_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        self.use_decoder_route_supervision = bool(use_decoder_route_supervision)
        if self.use_decoder_route_supervision:
            self.route_head = nn.Conv2d(decoder_channels, 4, kernel_size=1)

    @staticmethod
    def _resize_output(logits: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)

    def forward(
        self,
        features: Sequence[torch.Tensor],
        output_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        x1, x2, x3, x4 = features
        p4 = self.proj4(x4)
        p3 = self.fuse3(p4, self.proj3(x3))
        p2 = self.fuse2(p3, self.proj2(x2))
        feature = self.final_refine(self.fuse1(p2, self.proj1(x1)))
        outputs = {
            "mask": self._resize_output(self.mask_head(feature), output_size),
            "edge": self._resize_output(self.edge_head(feature), output_size),
            "reliability": self._resize_output(self.reliability_head(feature), output_size),
        }
        if self.use_decoder_route_supervision:
            outputs["route_s1"] = self.route_head(feature)
        return outputs


class DEMambaNetLite(nn.Module):
    """Task-aligned DE-MambaNet rebuilt for accuracy-efficiency balance."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        encoder_widths: Sequence[int] = (32, 64, 128, 192),
        encoder_depths: Sequence[int] = (2, 2, 3, 2),
        decoder_channels: int = 64,
        directional_kernel: int = 13,
        expansion_ratio: float = 2.0,
        mamba_channels: int = 96,
        use_mamba_ssm: bool = True,
        drop_path_rate: float = 0.05,
        use_decoder_route_supervision: bool = True,
    ) -> None:
        super().__init__()
        widths = tuple(int(value) for value in encoder_widths)
        depths = tuple(int(value) for value in encoder_depths)
        if int(in_channels) != 3 or int(num_classes) != 1:
            raise ValueError("DEMambaNetLite supports three-channel RGB binary segmentation only.")
        if len(widths) != 4 or any(value < 8 for value in widths):
            raise ValueError("encoder_widths must contain four values of at least 8 channels.")
        if len(depths) != 4 or any(value < 1 for value in depths):
            raise ValueError("encoder_depths must contain four positive integers.")
        if any(next_width <= width for width, next_width in zip(widths, widths[1:])):
            raise ValueError("encoder_widths must increase strictly between stages.")
        if decoder_channels < 8 or mamba_channels < 8:
            raise ValueError("decoder_channels and mamba_channels must be at least 8.")
        if not 0.0 <= drop_path_rate < 1.0:
            raise ValueError("drop_path_rate must be in [0, 1).")

        c1, c2, c3, c4 = widths
        self.encoder_widths = widths
        self.encoder_depths = depths
        self.use_decoder_route_supervision = bool(use_decoder_route_supervision)
        self.stem = nn.Sequential(
            ConvGNAct(3, max(8, c1 // 2), kernel_size=3, stride=2),
            ConvGNAct(max(8, c1 // 2), c1, kernel_size=3, stride=2),
        )

        total_blocks = sum(depths)
        drop_rates = torch.linspace(0.0, drop_path_rate, total_blocks).tolist()
        offset = 0

        def make_stage(channels: int, depth: int) -> nn.Sequential:
            nonlocal offset
            stage = nn.Sequential(
                *(
                    StructureDirectionalBlock(
                        channels,
                        expansion_ratio=expansion_ratio,
                        directional_kernel=directional_kernel,
                        drop_path=float(drop_rates[offset + index]),
                    )
                    for index in range(depth)
                )
            )
            offset += depth
            return stage

        self.stage1 = make_stage(c1, depths[0])
        self.down1 = ConvGNAct(c1, c2, kernel_size=3, stride=2)
        self.stage2 = make_stage(c2, depths[1])
        self.down2 = ConvGNAct(c2, c3, kernel_size=3, stride=2)
        self.stage3 = make_stage(c3, depths[2])
        self.down3 = ConvGNAct(c3, c4, kernel_size=3, stride=2)
        self.stage4 = make_stage(c4, depths[3])
        self.global_context = SharedAxialMambaContext(c4, mamba_channels, use_mamba_ssm=use_mamba_ssm)
        self.decoder = LightweightStructuralDecoder(
            widths,
            decoder_channels,
            use_decoder_route_supervision=self.use_decoder_route_supervision,
        )

    def extract_encoder_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        f1 = self.stage1(self.stem(x))
        f2 = self.stage2(self.down1(f1))
        f3 = self.stage3(self.down2(f2))
        f4_local = self.stage4(self.down3(f3))
        f4, route_s4 = self.global_context(f4_local)
        return {"f1": f1, "f2": f2, "f3": f3, "f4": f4, "route_s4": route_s4}

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        output_size = x.shape[-2:]
        features = self.extract_encoder_features(x)
        outputs = self.decoder(
            [features["f1"], features["f2"], features["f3"], features["f4"]],
            output_size,
        )
        outputs["route_s4"] = features["route_s4"]
        return outputs

    def mamba_backend_info(self) -> dict[str, list[str]]:
        return {"global_context": [self.global_context.backend]}
