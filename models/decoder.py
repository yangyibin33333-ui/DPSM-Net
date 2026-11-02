from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBNAct, DWConvBlock


class SemanticSkipGate(nn.Module):
    """Use decoder semantics to softly recalibrate a skip feature map."""

    def __init__(self, in_channels: int, skip_channels: int):
        super().__init__()
        self.context_proj = nn.Conv2d(in_channels, skip_channels, kernel_size=1)
        self.skip_proj = nn.Conv2d(skip_channels, skip_channels, kernel_size=1)
        self.norm = nn.GroupNorm(1, skip_channels)
        self.act = nn.GELU()
        self.gate_out = nn.Conv2d(skip_channels, skip_channels, kernel_size=1)

        # A zero logit makes gate_scale exactly one at initialization.
        nn.init.zeros_(self.gate_out.weight)
        nn.init.zeros_(self.gate_out.bias)

    def forward(self, decoder_context: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        gate_feature = self.act(self.norm(self.context_proj(decoder_context) + self.skip_proj(skip)))
        gate_scale = 1.0 + 0.5 * torch.tanh(self.gate_out(gate_feature))
        return skip * gate_scale


class DetailResidualSkipGate(nn.Module):
    """Apply a bounded semantic residual only to local skip details."""

    def __init__(self, decoder_channels: int, skip_channels: int):
        super().__init__()
        hidden_channels = max(8, min(32, decoder_channels // 4))
        self.context_proj = nn.Conv2d(decoder_channels, hidden_channels, kernel_size=1)
        self.norm = nn.GroupNorm(1, hidden_channels)
        self.act = nn.GELU()
        self.gate_logits = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.skip_channels = skip_channels

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        smooth = F.avg_pool2d(skip, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        detail = skip - smooth
        context = self.act(self.norm(self.context_proj(x)))
        spatial_gate = torch.tanh(self.gate_logits(context))
        strength = 0.5 * torch.tanh(self.alpha)
        return skip + strength * spatial_gate * detail


class BoundaryLogitCalibrator(nn.Module):
    """Use confident edge evidence to calibrate uncertain mask logits only."""

    def __init__(self, decoder_channels: int):
        super().__init__()
        hidden_channels = max(8, decoder_channels // 2)
        self.context_proj = nn.Conv2d(decoder_channels + 3, hidden_channels, kernel_size=1, bias=False)
        self.norm1 = nn.GroupNorm(1, hidden_channels)
        self.act1 = nn.GELU()
        self.depthwise = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(1, hidden_channels)
        self.act2 = nn.GELU()
        self.delta_out = nn.Conv2d(hidden_channels, 1, kernel_size=1)

        # Preserve the baseline mask logits at initialization.
        nn.init.zeros_(self.delta_out.weight)
        nn.init.zeros_(self.delta_out.bias)

    def forward(
        self,
        decoder_feature: torch.Tensor,
        mask_logits: torch.Tensor,
        edge_logits: torch.Tensor,
    ) -> torch.Tensor:
        mask_prob = torch.sigmoid(mask_logits)
        edge_prob = torch.sigmoid(edge_logits)
        uncertainty = 4.0 * mask_prob * (1.0 - mask_prob)
        calibration_input = torch.cat([decoder_feature, mask_prob, edge_prob, uncertainty], dim=1)
        context = self.act1(self.norm1(self.context_proj(calibration_input)))
        context = self.act2(self.norm2(self.depthwise(context)))
        raw_delta = torch.tanh(self.delta_out(context))
        correction = 0.5 * uncertainty * edge_prob * raw_delta
        return mask_logits + correction


class EdgeMaskConsistencyRefiner(nn.Module):
    """Refine mask logits only where mask and edge evidence disagree."""

    def __init__(self, decoder_channels: int):
        super().__init__()
        hidden_channels = max(8, decoder_channels // 2)
        self.context_proj = nn.Conv2d(decoder_channels + 5, hidden_channels, kernel_size=1, bias=False)
        self.norm1 = nn.GroupNorm(1, hidden_channels)
        self.act1 = nn.GELU()
        self.depthwise = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(1, hidden_channels)
        self.act2 = nn.GELU()
        self.expand_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.shrink_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        # The gates can initialize normally: this zero scale makes the full
        # module an exact identity until optimization learns a residual.
        self.residual_scale = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        decoder_feature: torch.Tensor,
        mask_logits: torch.Tensor,
        edge_logits: torch.Tensor,
    ) -> torch.Tensor:
        mask_prob = torch.sigmoid(mask_logits)
        edge_prob = torch.sigmoid(edge_logits)
        dilated = F.max_pool2d(mask_prob, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-mask_prob, kernel_size=3, stride=1, padding=1)
        soft_boundary = (dilated - eroded).clamp(0.0, 1.0)
        expand_evidence = edge_prob * (1.0 - mask_prob)
        shrink_evidence = soft_boundary * mask_prob * (1.0 - edge_prob)
        context_input = torch.cat(
            [decoder_feature, mask_prob, edge_prob, soft_boundary, expand_evidence, shrink_evidence], dim=1
        )
        context = self.act1(self.norm1(self.context_proj(context_input)))
        context = self.act2(self.norm2(self.depthwise(context)))
        expand_gate = torch.sigmoid(self.expand_head(context))
        shrink_gate = torch.sigmoid(self.shrink_head(context))
        signal = expand_evidence * expand_gate - shrink_evidence * shrink_gate
        correction = 0.5 * torch.tanh(self.residual_scale) * signal
        return mask_logits + correction


class LocalSemanticContinuityCalibrator(nn.Module):
    """Correct uncertain logits from semantically compatible local evidence.

    This module deliberately has no global pooling or direction-specific
    propagation.  It uses the stage-3 feature only to score compatibility
    within an eight-neighbor window; consequently a high-probability neighbor
    cannot cross a local semantic boundary merely because it is nearby.
    """

    _NEIGHBOR_OFFSETS = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )

    def __init__(self, decoder_channels: int, semantic_channels: int, semantic_dim: int = 8):
        super().__init__()
        hidden_channels = max(8, min(16, decoder_channels // 2))
        self.semantic_projection = nn.Conv2d(semantic_channels, semantic_dim, kernel_size=1, bias=False)
        self.gate_projection = nn.Conv2d(
            decoder_channels + semantic_dim + 3,
            hidden_channels,
            kernel_size=1,
            bias=False,
        )
        self.gate_norm = nn.GroupNorm(1, hidden_channels)
        self.gate_activation = nn.GELU()
        self.gate_out = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        # This makes the complete logit-calibration branch an exact identity
        # at initialization, while retaining gradients for the scale itself.
        self.residual_scale = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _replicate_shift(feature: torch.Tensor, row_offset: int, column_offset: int) -> torch.Tensor:
        if (row_offset, column_offset) == (0, 0) or row_offset not in (-1, 0, 1) or column_offset not in (-1, 0, 1):
            raise ValueError("LSCC supports non-central one-pixel neighbor shifts only.")
        height, width = feature.shape[-2:]
        padded = F.pad(feature, (1, 1, 1, 1), mode="replicate")
        top, left = 1 + row_offset, 1 + column_offset
        return padded[..., top : top + height, left : left + width]

    def semantic_descriptor(self, deep_feature: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        semantic = self.semantic_projection(deep_feature)
        if semantic.shape[-2:] != output_size:
            semantic = F.interpolate(semantic, size=output_size, mode="bilinear", align_corners=False)
        return F.normalize(semantic, p=2, dim=1, eps=1e-6)

    def local_support_and_agreement(
        self,
        semantic_descriptor: torch.Tensor,
        evidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return local evidence and its absolute semantic compatibility.

        Unlike a softmax over neighbors, the agreement stays near zero if all
        nearby features differ from the center.  This is the guard that blocks
        evidence transfer across a background bridge.
        """
        weighted_evidence = torch.zeros_like(evidence)
        affinity_sum = torch.zeros_like(evidence)
        for row_offset, column_offset in self._NEIGHBOR_OFFSETS:
            neighbor_semantic = self._replicate_shift(semantic_descriptor, row_offset, column_offset)
            similarity = (semantic_descriptor * neighbor_semantic).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
            # Unit-norm equality maps to one; orthogonal semantics maps to
            # exp(-5), so unrelated local content cannot gain support merely
            # by being the least dissimilar neighbor.
            affinity = torch.exp((similarity - 1.0) / 0.2)
            neighbor_evidence = self._replicate_shift(evidence, row_offset, column_offset)
            weighted_evidence = weighted_evidence + affinity * neighbor_evidence
            affinity_sum = affinity_sum + affinity
        support = weighted_evidence / affinity_sum.clamp_min(1e-6)
        agreement = affinity_sum / float(len(self._NEIGHBOR_OFFSETS))
        return support, agreement

    def forward(
        self,
        decoder_feature: torch.Tensor,
        deep_feature: torch.Tensor,
        mask_logits: torch.Tensor,
        edge_logits: torch.Tensor,
    ) -> torch.Tensor:
        # Limiting the calibrator to half resolution bounds its additional
        # activations at native 384x544 / batch 8.  It is a residual branch,
        # not an auxiliary prediction or a change to the evaluation grid.
        local_size = tuple(max(1, dimension // 2) for dimension in mask_logits.shape[-2:])
        local_decoder = F.interpolate(decoder_feature, size=local_size, mode="bilinear", align_corners=False)
        local_mask_logits = F.interpolate(mask_logits, size=local_size, mode="bilinear", align_corners=False)
        local_edge_logits = F.interpolate(edge_logits, size=local_size, mode="bilinear", align_corners=False)
        mask_probability = torch.sigmoid(local_mask_logits)
        edge_probability = torch.sigmoid(local_edge_logits)
        uncertainty = 4.0 * mask_probability * (1.0 - mask_probability)
        semantic = self.semantic_descriptor(deep_feature, local_size)
        # Edge confidence asks the branch to reinforce only crack-like local
        # candidates; semantic agreement still rejects a nearby background
        # texture or a separate crack with incompatible deep features.
        support, agreement = self.local_support_and_agreement(semantic, mask_probability * edge_probability)
        gate_input = torch.cat([local_decoder, semantic, mask_probability, edge_probability, uncertainty], dim=1)
        gate = torch.sigmoid(self.gate_out(self.gate_activation(self.gate_norm(self.gate_projection(gate_input)))))
        correction = (
            0.75
            * torch.tanh(self.residual_scale)
            * uncertainty
            * edge_probability
            * agreement
            * gate
            * (support - mask_probability)
        )
        correction = F.interpolate(correction, size=mask_logits.shape[-2:], mode="bilinear", align_corners=False)
        return mask_logits + correction


class SDFHead(nn.Module):
    """Training-only auxiliary regressor for normalized signed distances."""

    def __init__(self, decoder_channels: int):
        super().__init__()
        self.block = ConvBNAct(decoder_channels, decoder_channels)
        self.out = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.block(x))


class StructureConsistentHarmonicFusion(nn.Module):
    """Fuse all encoder scales under the final decoder semantics.

    The decoder state is kept as an anchor.  A spatial softmax selects which
    aligned encoder scale should contribute at every pixel, and a zero-initialized
    residual projection makes the module an exact identity at initialization.
    """

    def __init__(self, feature_channels: list[int], decoder_channels: int):
        super().__init__()
        hidden_channels = max(8, decoder_channels // 2)
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, decoder_channels, kernel_size=1, bias=False),
                    nn.GroupNorm(1, decoder_channels),
                    nn.GELU(),
                )
                for channels in feature_channels
            ]
        )
        self.gate = nn.Sequential(
            nn.Conv2d(decoder_channels * (len(feature_channels) + 1), hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, len(feature_channels), kernel_size=1),
        )
        self.residual = nn.Conv2d(decoder_channels, decoder_channels, kernel_size=1)

        # Preserve the C3 forward path at initialization while allowing the
        # residual projection to activate and train the scale-fusion branch.
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, decoder_feature: torch.Tensor, features: list[torch.Tensor]) -> torch.Tensor:
        if len(features) != len(self.projections):
            raise ValueError("The number of encoder features must match the harmonic-fusion projections.")
        target_size = decoder_feature.shape[-2:]
        aligned = [
            F.interpolate(projection(feature), size=target_size, mode="bilinear", align_corners=False)
            if feature.shape[-2:] != target_size
            else projection(feature)
            for feature, projection in zip(features, self.projections)
        ]
        weights = torch.softmax(self.gate(torch.cat([decoder_feature, *aligned], dim=1)), dim=1)
        fused = sum(weights[:, index : index + 1] * feature for index, feature in enumerate(aligned))
        return decoder_feature + self.residual(fused)


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_semantic_skip_gate: bool = False,
    ):
        super().__init__()
        self.semantic_skip_gate = (
            SemanticSkipGate(in_channels, skip_channels) if use_semantic_skip_gate else None
        )
        self.block = nn.Sequential(
            ConvBNAct(in_channels + skip_channels, out_channels, kernel_size=3),
            DWConvBlock(out_channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        detail_residual_gate: DetailResidualSkipGate | None = None,
    ) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        if self.semantic_skip_gate is not None:
            skip = self.semantic_skip_gate(x, skip)
        if detail_residual_gate is not None:
            skip = detail_residual_gate(x, skip)
        return self.block(torch.cat([x, skip], dim=1))


class LowResolutionSubpixelAdditiveUpBlock(nn.Module):
    """Fuse a decoder feature and skip feature without a high-resolution concat conv.

    The decoder projection is evaluated before the two-times PixelShuffle so
    cross-channel mixing stays in the lower-resolution domain.  PixelShuffle
    preserves separate sub-pixel outputs, while the skip path is only aligned
    with a one-by-one projection before additive fusion.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.decoder_projection = nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.skip_projection = nn.Conv2d(skip_channels, out_channels, kernel_size=1, bias=False)
        self.refine = DWConvBlock(out_channels)
        self._initialize_subpixel_projection()

    def _initialize_subpixel_projection(self) -> None:
        """Start all four sub-pixel phases from the same learned LR projection."""

        with torch.no_grad():
            base = torch.empty(
                self.out_channels,
                self.in_channels,
                1,
                1,
                device=self.decoder_projection.weight.device,
                dtype=self.decoder_projection.weight.dtype,
            )
            nn.init.kaiming_uniform_(base, a=math.sqrt(5))
            self.decoder_projection.weight.copy_(base.repeat_interleave(4, dim=0))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if x.size(1) != self.in_channels:
            raise ValueError(f"LSAF expected {self.in_channels} decoder channels, received {x.size(1)}.")
        if skip.size(1) != self.skip_channels:
            raise ValueError(f"LSAF expected {self.skip_channels} skip channels, received {skip.size(1)}.")

        upsampled = self.pixel_shuffle(self.decoder_projection(x))
        if upsampled.shape[-2:] != skip.shape[-2:]:
            raise ValueError(
                "LSAF requires the skip spatial size to be exactly two times the decoder feature size, "
                f"got decoder {tuple(x.shape[-2:])} and skip {tuple(skip.shape[-2:])}."
            )
        return self.refine(upsampled + self.skip_projection(skip))


class LowResolutionSubpixelSeparableConcatUpBlock(nn.Module):
    """Fuse sub-pixel decoder features with skips through separable concat fusion.

    Both channel projections happen before or alongside the sub-pixel resize.
    The only high-resolution spatial convolution is depthwise; the following
    one-by-one projection restores joint decoder/skip channel mixing.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.decoder_projection = nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.skip_projection = nn.Conv2d(skip_channels, out_channels, kernel_size=1, bias=False)
        fused_channels = out_channels * 2
        self.depthwise = nn.Conv2d(
            fused_channels,
            fused_channels,
            kernel_size=3,
            padding=1,
            groups=fused_channels,
            bias=False,
        )
        self.depthwise_norm = nn.BatchNorm2d(fused_channels)
        self.depthwise_act = nn.GELU()
        self.pointwise = nn.Conv2d(fused_channels, out_channels, kernel_size=1, bias=False)
        self.pointwise_norm = nn.BatchNorm2d(out_channels)
        self.pointwise_act = nn.GELU()
        self._initialize_subpixel_projection()

    def _initialize_subpixel_projection(self) -> None:
        """Initialize all PixelShuffle phases from one low-resolution projection."""

        with torch.no_grad():
            base = torch.empty(
                self.out_channels,
                self.in_channels,
                1,
                1,
                device=self.decoder_projection.weight.device,
                dtype=self.decoder_projection.weight.dtype,
            )
            nn.init.kaiming_uniform_(base, a=math.sqrt(5))
            self.decoder_projection.weight.copy_(base.repeat_interleave(4, dim=0))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if x.size(1) != self.in_channels:
            raise ValueError(f"LSSCF expected {self.in_channels} decoder channels, received {x.size(1)}.")
        if skip.size(1) != self.skip_channels:
            raise ValueError(f"LSSCF expected {self.skip_channels} skip channels, received {skip.size(1)}.")

        upsampled = self.pixel_shuffle(self.decoder_projection(x))
        if upsampled.shape[-2:] != skip.shape[-2:]:
            raise ValueError(
                "LSSCF requires the skip spatial size to be exactly two times the decoder feature size, "
                f"got decoder {tuple(x.shape[-2:])} and skip {tuple(skip.shape[-2:])}."
            )
        fused = torch.cat([upsampled, self.skip_projection(skip)], dim=1)
        fused = self.depthwise_act(self.depthwise_norm(self.depthwise(fused)))
        return self.pointwise_act(self.pointwise_norm(self.pointwise(fused)))


class DifferenceAwareSubpixelUpBlock(nn.Module):
    """Full-width sum transport plus full-width difference correction for Up1.

    The pre-existing low-resolution decoder projection and full-resolution skip
    projection are reused verbatim.  This preserves both source feature paths
    while removing only the high-resolution concat 2C→C pointwise fusion.
    """

    def __init__(self, baseline_up_block: LowResolutionSubpixelSeparableConcatUpBlock):
        super().__init__()
        if not isinstance(baseline_up_block, LowResolutionSubpixelSeparableConcatUpBlock):
            raise TypeError("DAFU1 requires the original LSSCF Up1 block.")
        if baseline_up_block.out_channels != 40:
            raise ValueError("DAFU1 is fixed to the original 40-channel Up1 output.")
        self.in_channels = baseline_up_block.in_channels
        self.skip_channels = baseline_up_block.skip_channels
        self.out_channels = baseline_up_block.out_channels
        # Reuse the native projections and PixelShuffle exactly; assignment
        # moves them into this module when BoundaryAwareDecoder replaces up1.
        self.decoder_projection = baseline_up_block.decoder_projection
        self.pixel_shuffle = baseline_up_block.pixel_shuffle
        self.skip_projection = baseline_up_block.skip_projection
        self.delta_depthwise = nn.Conv2d(
            self.out_channels,
            self.out_channels,
            kernel_size=3,
            padding=1,
            groups=self.out_channels,
            bias=False,
        )
        self.delta_depthwise_norm = nn.BatchNorm2d(self.out_channels)
        self.delta_depthwise_act = nn.GELU()
        self.delta_pointwise = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, bias=False)
        self.delta_pointwise_norm = nn.BatchNorm2d(self.out_channels)
        self.delta_pointwise_act = nn.GELU()
        self.debug_shapes = False
        self.last_debug_shapes: dict[str, tuple[int, ...]] = {}

    def set_debug_shapes(self, enabled: bool) -> None:
        self.debug_shapes = bool(enabled)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if x.size(1) != self.in_channels:
            raise ValueError(f"DAFU1 expected {self.in_channels} decoder channels, received {x.size(1)}.")
        if skip.size(1) != self.skip_channels:
            raise ValueError(f"DAFU1 expected {self.skip_channels} skip channels, received {skip.size(1)}.")
        upsampled = self.pixel_shuffle(self.decoder_projection(x))
        projected_skip = self.skip_projection(skip)
        if upsampled.shape[-2:] != projected_skip.shape[-2:]:
            raise ValueError(
                "DAFU1 requires the skip spatial size to be exactly two times the decoder feature size, "
                f"got decoder {tuple(x.shape[-2:])} and skip {tuple(skip.shape[-2:])}."
            )
        base = upsampled + projected_skip
        delta = upsampled - projected_skip
        correction = self.delta_depthwise_act(self.delta_depthwise_norm(self.delta_depthwise(delta)))
        correction = self.delta_pointwise_act(self.delta_pointwise_norm(self.delta_pointwise(correction)))
        output = base + correction
        self.last_debug_shapes = {
            "input": tuple(x.shape),
            "skip": tuple(skip.shape),
            "u": tuple(upsampled.shape),
            "s": tuple(projected_skip.shape),
            "base": tuple(base.shape),
            "delta": tuple(delta.shape),
            "out": tuple(output.shape),
        }
        if self.debug_shapes:
            print("DAFU1 shapes | " + " | ".join(f"{name}={shape}" for name, shape in self.last_debug_shapes.items()))
        return output


class HaarFrequencyGate(nn.Module):
    """Optional fixed-Haar detail injection modulated by predicted reliability."""

    def __init__(self, channels: int):
        super().__init__()
        self.register_buffer("horizontal", torch.tensor([[1.0, -1.0]]) / math.sqrt(2.0), persistent=False)
        self.register_buffer("vertical", torch.tensor([[1.0], [-1.0]]) / math.sqrt(2.0), persistent=False)
        self.project = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, reliability: torch.Tensor) -> torch.Tensor:
        channels = x.size(1)
        kh = self.horizontal.view(1, 1, 1, 2).expand(channels, 1, 1, 2)
        kv = self.vertical.view(1, 1, 2, 1).expand(channels, 1, 2, 1)
        high_h = F.pad(F.conv2d(x, kh, groups=channels), (0, 1, 0, 0)).abs()
        high_v = F.pad(F.conv2d(x, kv, groups=channels), (0, 0, 0, 1)).abs()
        return x + reliability * self.project(high_h + high_v)


class RouteGuidedDirectionalRefiner(nn.Module):
    """Propagate decoder details only along confident stage-3 route directions."""

    def __init__(self, decoder_channels: int):
        super().__init__()
        hidden_channels = max(8, decoder_channels // 4)
        self.register_buffer(
            "horizontal_kernel",
            torch.tensor([[[[0.2, 0.2, 0.2, 0.2, 0.2]]]]),
            persistent=False,
        )
        self.register_buffer(
            "vertical_kernel",
            torch.tensor([[[[0.2], [0.2], [0.2], [0.2], [0.2]]]]),
            persistent=False,
        )
        self.register_buffer(
            "diagonal_kernel",
            torch.eye(5).view(1, 1, 5, 5) / 5.0,
            persistent=False,
        )
        self.register_buffer(
            "anti_diagonal_kernel",
            torch.fliplr(torch.eye(5)).view(1, 1, 5, 5) / 5.0,
            persistent=False,
        )
        self.context = nn.Sequential(
            nn.Conv2d(decoder_channels * 2 + 1, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
        )
        self.residual = nn.Conv2d(hidden_channels, decoder_channels, kernel_size=1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    @staticmethod
    def _depthwise_context(
        feature: torch.Tensor,
        kernel: torch.Tensor,
        padding: tuple[int, int, int, int],
    ) -> torch.Tensor:
        channels = feature.size(1)
        padded = F.pad(feature, padding, mode="replicate")
        weight = kernel.to(dtype=feature.dtype).expand(channels, -1, -1, -1)
        return F.conv2d(padded, weight, groups=channels)

    def route_probabilities_and_confidence(
        self,
        route_logits: torch.Tensor,
        output_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if route_logits.ndim != 4 or route_logits.size(1) != 4:
            raise ValueError("RouteGuidedDirectionalRefiner requires route_s3 logits with shape [B, 4, H, W].")
        route_logits_up = F.interpolate(route_logits, size=output_size, mode="bilinear", align_corners=False)
        route_probs = torch.softmax(route_logits_up, dim=1)
        entropy = -(route_probs * torch.log(route_probs.clamp_min(1e-6))).sum(dim=1, keepdim=True)
        route_confidence = (1.0 - entropy / math.log(4.0)).clamp(0.0, 1.0)
        return route_probs, route_confidence

    def forward(self, decoder_feature: torch.Tensor, route_logits: torch.Tensor) -> torch.Tensor:
        if decoder_feature.size(0) != route_logits.size(0):
            raise ValueError("RouteGuidedDirectionalRefiner decoder feature and route logits must share a batch size.")
        route_probs, route_confidence = self.route_probabilities_and_confidence(route_logits, decoder_feature.shape[-2:])
        horizontal_context = self._depthwise_context(decoder_feature, self.horizontal_kernel, (2, 2, 0, 0))
        vertical_context = self._depthwise_context(decoder_feature, self.vertical_kernel, (0, 0, 2, 2))
        diagonal_context = self._depthwise_context(decoder_feature, self.diagonal_kernel, (2, 2, 2, 2))
        anti_diagonal_context = self._depthwise_context(decoder_feature, self.anti_diagonal_kernel, (2, 2, 2, 2))
        routed_context = (
            route_probs[:, 0:1] * horizontal_context
            + route_probs[:, 1:2] * vertical_context
            + route_probs[:, 2:3] * diagonal_context
            + route_probs[:, 3:4] * anti_diagonal_context
        )
        directional_delta = (routed_context - decoder_feature) * route_confidence
        residual = self.residual(self.context(torch.cat([decoder_feature, directional_delta, route_confidence], dim=1)))
        return decoder_feature + residual


class RouteGuidedAnisotropicDiffusionRefiner(nn.Module):
    """Diffuse only across feature-similar neighbors selected by stage-3 routes."""

    def __init__(self, decoder_channels: int):
        super().__init__()
        hidden_channels = max(8, decoder_channels // 4)
        self.context = nn.Sequential(
            nn.Conv2d(decoder_channels * 2 + 1, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
        )
        self.residual = nn.Conv2d(hidden_channels, decoder_channels, kernel_size=1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    @staticmethod
    def _replicate_shift(feature: torch.Tensor, row_offset: int, column_offset: int) -> torch.Tensor:
        """Return a one-pixel neighbor with replicate, rather than zero, padding."""
        if row_offset not in (-1, 0, 1) or column_offset not in (-1, 0, 1):
            raise ValueError("RG-AFD supports one-pixel shifts only.")
        height, width = feature.shape[-2:]
        padded = F.pad(feature, (1, 1, 1, 1), mode="replicate")
        top, left = 1 + row_offset, 1 + column_offset
        return padded[..., top : top + height, left : left + width]

    @staticmethod
    def _affinity(center_normalized: torch.Tensor, neighbor: torch.Tensor) -> torch.Tensor:
        neighbor_normalized = F.normalize(neighbor, p=2, dim=1, eps=1e-6)
        cosine_similarity = (center_normalized * neighbor_normalized).sum(dim=1, keepdim=True)
        return ((cosine_similarity + 1.0) * 0.5).clamp(0.0, 1.0)

    def directional_deltas_and_affinities(
        self,
        feature: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]:
        center_normalized = F.normalize(feature, p=2, dim=1, eps=1e-6)
        affinities: dict[str, torch.Tensor] = {}

        def bilateral_delta(
            first: str,
            first_offset: tuple[int, int],
            second: str,
            second_offset: tuple[int, int],
        ) -> torch.Tensor:
            first_neighbor = self._replicate_shift(feature, *first_offset)
            second_neighbor = self._replicate_shift(feature, *second_offset)
            first_affinity = self._affinity(center_normalized, first_neighbor)
            second_affinity = self._affinity(center_normalized, second_neighbor)
            affinities[first], affinities[second] = first_affinity, second_affinity
            return (
                first_affinity * (first_neighbor - feature) + second_affinity * (second_neighbor - feature)
            ) / (first_affinity + second_affinity + 1e-6)

        return (
            bilateral_delta("left", (0, -1), "right", (0, 1)),
            bilateral_delta("up", (-1, 0), "down", (1, 0)),
            bilateral_delta("up_left", (-1, -1), "down_right", (1, 1)),
            bilateral_delta("up_right", (-1, 1), "down_left", (1, -1)),
        ), affinities

    def route_probabilities_and_confidence(
        self,
        route_logits: torch.Tensor,
        output_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if route_logits.ndim != 4 or route_logits.size(1) != 4:
            raise ValueError("RouteGuidedAnisotropicDiffusionRefiner requires route_s3 logits with shape [B, 4, H, W].")
        route_logits_up = F.interpolate(route_logits, size=output_size, mode="bilinear", align_corners=False)
        route_probs = torch.softmax(route_logits_up, dim=1)
        entropy = -(route_probs * torch.log(route_probs.clamp_min(1e-6))).sum(dim=1, keepdim=True)
        route_confidence = (1.0 - entropy / math.log(4.0)).clamp(0.0, 1.0)
        return route_probs, route_confidence

    def _detached_half_resolution_context(
        self,
        decoder_feature: torch.Tensor,
        route_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the affinity prior cheaply, without a decoder or route gradient graph."""
        if route_logits.ndim != 4 or route_logits.size(1) != 4:
            raise ValueError("RouteGuidedAnisotropicDiffusionRefiner requires route_s3 logits with shape [B, 4, H, W].")
        with torch.no_grad():
            feature_half = F.avg_pool2d(decoder_feature.detach(), kernel_size=2, stride=2)
            route_logits_half = F.interpolate(
                route_logits.detach(),
                size=feature_half.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            route_probs_half = torch.softmax(route_logits_half, dim=1)
            entropy_half = -(route_probs_half * torch.log(route_probs_half.clamp_min(1e-6))).sum(dim=1, keepdim=True)
            route_confidence_half = (1.0 - entropy_half / math.log(4.0)).clamp(0.0, 1.0)
            center_normalized = F.normalize(feature_half, p=2, dim=1, eps=1e-6)
            directional_delta_half = torch.zeros_like(feature_half)
            directions = (
                (0, (0, -1), (0, 1)),
                (1, (-1, 0), (1, 0)),
                (2, (-1, -1), (1, 1)),
                (3, (-1, 1), (1, -1)),
            )
            for direction_index, first_offset, second_offset in directions:
                first_neighbor = self._replicate_shift(feature_half, *first_offset)
                second_neighbor = self._replicate_shift(feature_half, *second_offset)
                first_affinity = self._affinity(center_normalized, first_neighbor)
                second_affinity = self._affinity(center_normalized, second_neighbor)
                delta = (
                    first_affinity * (first_neighbor - feature_half)
                    + second_affinity * (second_neighbor - feature_half)
                ) / (first_affinity + second_affinity + 1e-6)
                directional_delta_half += route_probs_half[:, direction_index : direction_index + 1] * delta
                del first_neighbor, second_neighbor, first_affinity, second_affinity, delta

            directional_delta_half = directional_delta_half * route_confidence_half
            directional_delta_high = F.interpolate(
                directional_delta_half,
                size=decoder_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            route_confidence_high = F.interpolate(
                route_confidence_half,
                size=decoder_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)
            del (
                feature_half,
                center_normalized,
                route_logits_half,
                route_probs_half,
                route_confidence_half,
                directional_delta_half,
            )
        return directional_delta_high, route_confidence_high

    def forward(self, decoder_feature: torch.Tensor, route_logits: torch.Tensor) -> torch.Tensor:
        if decoder_feature.size(0) != route_logits.size(0):
            raise ValueError("RG-AFD decoder feature and route logits must share a batch size.")
        directional_delta, route_confidence = self._detached_half_resolution_context(decoder_feature, route_logits)
        residual = self.residual(self.context(torch.cat([decoder_feature, directional_delta, route_confidence], dim=1)))
        return decoder_feature + residual


class LocalPixelReassemblyResidual(nn.Module):
    """Mix only non-overlapping 2-by-2 local feature windows through channels."""

    def __init__(self, channels: int):
        super().__init__()
        hidden_channels = max(1, channels * 2 // 5)
        folded_channels = channels * 4
        self.channels = channels
        self.unshuffle = nn.PixelUnshuffle(2)
        self.reduce = nn.Conv2d(folded_channels, hidden_channels, kernel_size=1, bias=False)
        self.expand = nn.Conv2d(hidden_channels, folded_channels, kernel_size=1, bias=False)
        self.shuffle = nn.PixelShuffle(2)
        self.gamma = nn.Parameter(torch.zeros(()))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.size(1) != self.channels:
            raise ValueError(f"LPR2 expected {self.channels} channels, received {feature.size(1)}.")
        folded = self.unshuffle(feature)
        mixed = F.gelu(self.reduce(folded))
        detail = self.shuffle(self.expand(mixed))
        return feature + 0.5 * torch.tanh(self.gamma) * detail


class LocalSpectralConfidenceRejector(nn.Module):
    """Calibrate confident texture-like mask logits from local learned features."""

    _DCT_SIZE = 4
    _ANALYSIS_CHANNELS = 8
    _AC_BANDS = _DCT_SIZE * _DCT_SIZE - 1

    def __init__(self, decoder_channels: int, stage3_channels: int):
        super().__init__()
        self.analysis_proj = nn.Conv2d(decoder_channels, self._ANALYSIS_CHANNELS, kernel_size=1, bias=False)
        self.semantic_proj = nn.Conv2d(stage3_channels, 8, kernel_size=1, bias=False)
        self.context_proj = nn.Conv2d(self._AC_BANDS + 8 + 3, 16, kernel_size=1, bias=False)
        self.norm1 = nn.GroupNorm(1, 16)
        self.act1 = nn.GELU()
        self.depthwise = nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16, bias=False)
        self.norm2 = nn.GroupNorm(1, 16)
        self.act2 = nn.GELU()
        self.delta_out = nn.Conv2d(16, 1, kernel_size=1)
        self.register_buffer(
            "dct_basis",
            self._dct_basis().repeat(self._ANALYSIS_CHANNELS, 1, 1, 1),
            persistent=False,
        )
        nn.init.zeros_(self.delta_out.weight)
        nn.init.zeros_(self.delta_out.bias)

    @classmethod
    def _dct_basis(cls) -> torch.Tensor:
        basis: list[torch.Tensor] = []
        coordinates = torch.arange(cls._DCT_SIZE, dtype=torch.float32)
        for vertical_frequency in range(cls._DCT_SIZE):
            for horizontal_frequency in range(cls._DCT_SIZE):
                if vertical_frequency == 0 and horizontal_frequency == 0:
                    continue
                vertical_scale = math.sqrt(1.0 / cls._DCT_SIZE) if vertical_frequency == 0 else math.sqrt(2.0 / cls._DCT_SIZE)
                horizontal_scale = (
                    math.sqrt(1.0 / cls._DCT_SIZE) if horizontal_frequency == 0 else math.sqrt(2.0 / cls._DCT_SIZE)
                )
                vertical = vertical_scale * torch.cos(math.pi * (2.0 * coordinates + 1.0) * vertical_frequency / (2.0 * cls._DCT_SIZE))
                horizontal = horizontal_scale * torch.cos(
                    math.pi * (2.0 * coordinates + 1.0) * horizontal_frequency / (2.0 * cls._DCT_SIZE)
                )
                basis.append(torch.outer(vertical, horizontal))
        return torch.stack(basis, dim=0).unsqueeze(1)

    def _spectral_energy(self, analysis_feature: torch.Tensor) -> torch.Tensor:
        if analysis_feature.shape[-2] % self._DCT_SIZE or analysis_feature.shape[-1] % self._DCT_SIZE:
            raise ValueError("LSCR decoder feature spatial dimensions must be divisible by 4.")
        coefficients = F.conv2d(
            analysis_feature,
            self.dct_basis,
            stride=self._DCT_SIZE,
            groups=self._ANALYSIS_CHANNELS,
        )
        batch, _, height, width = coefficients.shape
        coefficients = coefficients.reshape(batch, self._ANALYSIS_CHANNELS, self._AC_BANDS, height, width)
        return torch.log1p(coefficients.square().mean(dim=1))

    @staticmethod
    def _resize_probability(logits: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(torch.sigmoid(logits), size=size, mode="bilinear", align_corners=False)

    def forward(
        self,
        decoder_feature: torch.Tensor,
        stage3_feature: torch.Tensor,
        mask_logits: torch.Tensor,
        edge_logits: torch.Tensor,
        reliability_logits: torch.Tensor,
    ) -> torch.Tensor:
        spectral_energy = self._spectral_energy(self.analysis_proj(decoder_feature))
        target_size = spectral_energy.shape[-2:]
        semantic_feature = self.semantic_proj(stage3_feature)
        if semantic_feature.shape[-2:] != target_size:
            semantic_feature = F.interpolate(semantic_feature, size=target_size, mode="bilinear", align_corners=False)
        context_input = torch.cat(
            [
                spectral_energy,
                semantic_feature,
                self._resize_probability(mask_logits, target_size),
                self._resize_probability(edge_logits, target_size),
                self._resize_probability(reliability_logits, target_size),
            ],
            dim=1,
        )
        context = self.act1(self.norm1(self.context_proj(context_input)))
        context = self.act2(self.norm2(self.depthwise(context)))
        delta = F.interpolate(
            torch.tanh(self.delta_out(context)),
            size=mask_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return mask_logits + 0.75 * torch.sigmoid(mask_logits).detach() * delta


class HighResolutionSpectralDetailGate(nn.Module):
    """Use local learned-feature spectra to gate existing high-resolution detail."""

    _DCT_SIZE = 3
    _ANALYSIS_CHANNELS = 4
    _AC_BANDS = _DCT_SIZE * _DCT_SIZE - 1

    def __init__(self, decoder_channels: int, stage3_channels: int):
        super().__init__()
        self.analysis_proj = nn.Conv2d(decoder_channels, self._ANALYSIS_CHANNELS, kernel_size=1, bias=False)
        self.semantic_proj = nn.Conv2d(stage3_channels, 4, kernel_size=1, bias=False)
        self.context_proj = nn.Conv2d(self._AC_BANDS + 4, 8, kernel_size=1, bias=False)
        self.norm1 = nn.GroupNorm(1, 8)
        self.act1 = nn.GELU()
        self.depthwise = nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8, bias=False)
        self.norm2 = nn.GroupNorm(1, 8)
        self.act2 = nn.GELU()
        self.gate_out = nn.Conv2d(8, decoder_channels, kernel_size=1)
        self.register_buffer("dct_basis", self._dct_basis(), persistent=False)
        nn.init.zeros_(self.gate_out.weight)
        nn.init.zeros_(self.gate_out.bias)

    @classmethod
    def _dct_basis(cls) -> torch.Tensor:
        basis: list[torch.Tensor] = []
        coordinates = torch.arange(cls._DCT_SIZE, dtype=torch.float32)
        for vertical_frequency in range(cls._DCT_SIZE):
            for horizontal_frequency in range(cls._DCT_SIZE):
                if vertical_frequency == 0 and horizontal_frequency == 0:
                    continue
                vertical_scale = math.sqrt(1.0 / cls._DCT_SIZE) if vertical_frequency == 0 else math.sqrt(2.0 / cls._DCT_SIZE)
                horizontal_scale = (
                    math.sqrt(1.0 / cls._DCT_SIZE) if horizontal_frequency == 0 else math.sqrt(2.0 / cls._DCT_SIZE)
                )
                vertical = vertical_scale * torch.cos(math.pi * (2.0 * coordinates + 1.0) * vertical_frequency / (2.0 * cls._DCT_SIZE))
                horizontal = horizontal_scale * torch.cos(
                    math.pi * (2.0 * coordinates + 1.0) * horizontal_frequency / (2.0 * cls._DCT_SIZE)
                )
                basis.append(torch.outer(vertical, horizontal))
        return torch.stack(basis, dim=0).unsqueeze(1)

    def _spectral_energy(self, analysis_feature: torch.Tensor) -> torch.Tensor:
        energies: list[torch.Tensor] = []
        for basis in self.dct_basis:
            coefficients = F.conv2d(
                analysis_feature,
                basis.unsqueeze(0).expand(self._ANALYSIS_CHANNELS, -1, -1, -1),
                padding=1,
                groups=self._ANALYSIS_CHANNELS,
            )
            energies.append(torch.log1p(coefficients.square().mean(dim=1, keepdim=True)))
        return torch.cat(energies, dim=1)

    def forward(self, decoder_feature: torch.Tensor, stage3_feature: torch.Tensor) -> torch.Tensor:
        spectral_energy = self._spectral_energy(self.analysis_proj(decoder_feature))
        semantic_feature = self.semantic_proj(stage3_feature)
        if semantic_feature.shape[-2:] != decoder_feature.shape[-2:]:
            semantic_feature = F.interpolate(
                semantic_feature,
                size=decoder_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        context = self.act1(self.norm1(self.context_proj(torch.cat([spectral_energy, semantic_feature], dim=1))))
        context = self.act2(self.norm2(self.depthwise(context)))
        local_smooth = F.avg_pool2d(
            decoder_feature,
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=False,
        )
        local_detail = decoder_feature - local_smooth
        return decoder_feature + 0.35 * torch.tanh(self.gate_out(context)) * local_detail


class DepthwiseSeparableMaskHead(nn.Module):
    """Fixed-width depthwise-separable replacement for the dense mask head."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = int(channels)
        self.depthwise = ConvBNAct(
            self.channels,
            self.channels,
            kernel_size=3,
            stride=1,
            groups=self.channels,
        )
        self.pointwise = ConvBNAct(
            self.channels,
            self.channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.output = nn.Conv2d(self.channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.pointwise(self.depthwise(x)))


class FullWidthResidualMaskHead(nn.Module):
    """Keep the full decoder feature while learning a narrow residual correction.

    This is intentionally *not* a depthwise-separable replacement for the
    dense prediction path: the 40-channel input reaches the classifier through
    an identity branch.  The 16-channel depthwise spatial operation only
    supplies an additive low-rank correction before the original-width 1x1
    classifier.
    """

    def __init__(self, channels: int, hidden_channels: int = 16):
        super().__init__()
        if channels != 40 or hidden_channels != 16:
            raise ValueError("FWRMH is fixed to a 40-channel input and a 16-channel residual bottleneck.")
        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)
        self.reduce = ConvBNAct(self.channels, self.hidden_channels, kernel_size=1, padding=0)
        self.spatial = ConvBNAct(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size=3,
            groups=self.hidden_channels,
        )
        self.expand = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.channels),
        )
        self.activation = nn.GELU()
        self.classifier = nn.Conv2d(self.channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.expand(self.spatial(self.reduce(x)))
        return self.classifier(self.activation(x + residual))


class ConvGroupNormSiLU(nn.Module):
    """Fixed lightweight Conv-GN-SiLU used only by the NTD transport core."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        groups = 8 if out_channels % 8 == 0 else 1
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(x)))


class NarrowTransportDecoder(nn.Module):
    """Project every native encoder scale, then transport only narrow features."""

    def __init__(
        self,
        feature_channels: list[int],
        transport_dim: int,
        output_channels: int,
    ):
        super().__init__()
        if len(feature_channels) != 4:
            raise ValueError("NarrowTransportDecoder requires exactly four encoder feature widths.")
        if transport_dim not in (16, 24):
            raise ValueError("decoder_transport_dim must be 24, or 16 for the single fallback complexity check.")
        self.feature_channels = tuple(int(channels) for channels in feature_channels)
        self.transport_dim = int(transport_dim)
        self.output_channels = int(output_channels)
        c1, c2, c3, c4 = self.feature_channels
        self.proj1 = ConvGroupNormSiLU(c1, self.transport_dim, kernel_size=1)
        self.proj2 = ConvGroupNormSiLU(c2, self.transport_dim, kernel_size=1)
        self.proj3 = ConvGroupNormSiLU(c3, self.transport_dim, kernel_size=1)
        self.proj4 = ConvGroupNormSiLU(c4, self.transport_dim, kernel_size=1)
        self.refine4 = ConvGroupNormSiLU(self.transport_dim, self.transport_dim, kernel_size=3)
        self.refine3 = ConvGroupNormSiLU(self.transport_dim, self.transport_dim, kernel_size=3)
        self.refine2 = ConvGroupNormSiLU(self.transport_dim, self.transport_dim, kernel_size=3)
        self.refine1 = ConvGroupNormSiLU(self.transport_dim, self.transport_dim, kernel_size=3)
        self.adapter = nn.Conv2d(self.transport_dim, self.output_channels, kernel_size=1)

    @staticmethod
    def _upsample_add(high: torch.Tensor, lateral: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            high,
            size=lateral.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ) + lateral

    def forward(
        self,
        features: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(features) != 4:
            raise ValueError("NarrowTransportDecoder forward requires F1, F2, F3, and F4.")
        f1, f2, f3, f4 = features
        for index, (feature, channels) in enumerate(zip(features, self.feature_channels, strict=True), start=1):
            if feature.size(1) != channels:
                raise ValueError(f"NTD expected F{index} with {channels} channels, received {feature.size(1)}.")
        p1, p2, p3, p4 = self.proj1(f1), self.proj2(f2), self.proj3(f3), self.proj4(f4)
        t4 = self.refine4(p4)
        t3 = self.refine3(self._upsample_add(t4, p3))
        t2 = self.refine2(self._upsample_add(t3, p2))
        t1 = self.refine1(self._upsample_add(t2, p1))
        aligned = (
            t1
            if t1.shape[-2:] == f1.shape[-2:]
            else F.interpolate(t1, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        )
        return self.adapter(aligned), t3, t2


class PartialNarrowTransportDecoder(nn.Module):
    """Narrow deep semantics while preserving native F2/F1 scale identity.

    F4 and F3 are independently projected before concat fusion. The resulting
    160-channel F3 semantic feature uses a narrow decoder-side transport into
    F2, while the original LSSCF F2 skip projection and separable concat
    fusion are reused verbatim. F2 skip and all F1 processing stay wide.
    """

    def __init__(
        self,
        baseline_up3: LowResolutionSubpixelSeparableConcatUpBlock,
        baseline_up2: LowResolutionSubpixelSeparableConcatUpBlock,
        transport_dim: int,
    ):
        super().__init__()
        if not isinstance(baseline_up3, LowResolutionSubpixelSeparableConcatUpBlock):
            raise TypeError("PNTD requires the original LSSCF up3 block.")
        if not isinstance(baseline_up2, LowResolutionSubpixelSeparableConcatUpBlock):
            raise TypeError("PNTD requires the original LSSCF up2 block.")
        if transport_dim != 24:
            raise ValueError("partial_transport_dim is fixed to 24 for this single-variable candidate.")
        self.f4_channels = int(baseline_up3.in_channels)
        self.f3_channels = int(baseline_up3.skip_channels)
        self.f2_channels = int(baseline_up2.skip_channels)
        self.transport_dim = int(transport_dim)
        self.f4_projection = ConvGroupNormSiLU(self.f4_channels, self.transport_dim, kernel_size=1)
        self.f3_projection = ConvGroupNormSiLU(self.f3_channels, self.transport_dim, kernel_size=1)
        fused_deep_channels = self.transport_dim * 2
        deep_groups = 8 if fused_deep_channels % 8 == 0 else 1
        self.f3_depthwise = nn.Conv2d(
            fused_deep_channels,
            fused_deep_channels,
            kernel_size=3,
            padding=1,
            groups=fused_deep_channels,
            bias=False,
        )
        self.f3_depthwise_norm = nn.GroupNorm(deep_groups, fused_deep_channels)
        self.f3_depthwise_act = nn.SiLU()
        self.f3_pointwise = ConvGroupNormSiLU(fused_deep_channels, self.f3_channels, kernel_size=1)
        self.f2_decoder_projection = ConvGroupNormSiLU(
            self.f3_channels, self.transport_dim, kernel_size=1
        )
        self.f2_transport_adapter = ConvGroupNormSiLU(
            self.transport_dim, self.f2_channels, kernel_size=1
        )

        # Preserve the exact native F2 skip and concat-fusion modules. Only
        # the old decoder_projection/PixelShuffle transport is discarded.
        self.f2_skip_projection = baseline_up2.skip_projection
        self.f2_depthwise = baseline_up2.depthwise
        self.f2_depthwise_norm = baseline_up2.depthwise_norm
        self.f2_depthwise_act = baseline_up2.depthwise_act
        self.f2_pointwise = baseline_up2.pointwise
        self.f2_pointwise_norm = baseline_up2.pointwise_norm
        self.f2_pointwise_act = baseline_up2.pointwise_act
        self.debug_shapes = False
        self.last_debug_shapes: dict[str, tuple[int, ...]] = {}

    def set_debug_shapes(self, enabled: bool) -> None:
        self.debug_shapes = bool(enabled)

    def forward(
        self,
        f4: torch.Tensor,
        f3: torch.Tensor,
        f2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (self.f4_channels, self.f3_channels, self.f2_channels)
        for name, feature, channels in zip(("F4", "F3", "F2"), (f4, f3, f2), expected, strict=True):
            if feature.size(1) != channels:
                raise ValueError(f"PNTD expected {name} with {channels} channels, received {feature.size(1)}.")

        f4_narrow = self.f4_projection(f4)
        f4_at_f3 = F.interpolate(f4_narrow, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        f3_native_narrow = self.f3_projection(f3)
        f3_concat = torch.cat([f4_at_f3, f3_native_narrow], dim=1)
        f3_local = self.f3_depthwise_act(self.f3_depthwise_norm(self.f3_depthwise(f3_concat)))
        f3_semantic = self.f3_pointwise(f3_local)

        f3_transport = self.f2_decoder_projection(f3_semantic)
        f3_at_f2 = F.interpolate(f3_transport, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        f2_deep_semantic = self.f2_transport_adapter(f3_at_f2)
        f2_native = self.f2_skip_projection(f2)
        f2_concat = torch.cat([f2_deep_semantic, f2_native], dim=1)
        f2_local = self.f2_depthwise_act(self.f2_depthwise_norm(self.f2_depthwise(f2_concat)))
        f2_output = self.f2_pointwise_act(self.f2_pointwise_norm(self.f2_pointwise(f2_local)))
        self.last_debug_shapes = {
            "f4": tuple(f4.shape),
            "f3": tuple(f3.shape),
            "f2": tuple(f2.shape),
            "f4_narrow": tuple(f4_narrow.shape),
            "f3_native_narrow": tuple(f3_native_narrow.shape),
            "f3_concat": tuple(f3_concat.shape),
            "f3_semantic": tuple(f3_semantic.shape),
            "f3_transport": tuple(f3_transport.shape),
            "f2_deep_semantic": tuple(f2_deep_semantic.shape),
            "f2_native": tuple(f2_native.shape),
            "f2_concat": tuple(f2_concat.shape),
            "f2_output": tuple(f2_output.shape),
        }
        if self.debug_shapes:
            print("PNTD shapes | " + " | ".join(f"{name}={shape}" for name, shape in self.last_debug_shapes.items()))
        return f3_semantic, f2_output


class BoundaryAwareDecoder(nn.Module):
    def __init__(
        self,
        channels: list[int],
        decoder_channels: int,
        use_frequency_gate: bool = False,
        use_semantic_skip_gate: bool = False,
        use_detail_residual_gate: bool = False,
        use_structure_harmonic_fusion: bool = False,
        use_boundary_logit_calibration: bool = False,
        use_sdf_head: bool = False,
        use_edge_mask_consistency_refiner: bool = False,
        use_deep_supervision: bool = False,
        use_route_guided_directional_refiner: bool = False,
        use_route_guided_anisotropic_diffusion: bool = False,
        use_local_semantic_continuity_calibrator: bool = False,
        use_local_pixel_reassembly_residual: bool = False,
        use_local_spectral_confidence_rejector: bool = False,
        use_high_resolution_spectral_detail_gate: bool = False,
        use_low_resolution_subpixel_additive_fusion: bool = False,
        return_training_mask_feature: bool = False,
        use_low_resolution_subpixel_separable_concat_fusion: bool = False,
    ):
        super().__init__()
        if use_boundary_logit_calibration and use_edge_mask_consistency_refiner:
            raise ValueError("BoundaryLogitCalibrator and EdgeMaskConsistencyRefiner cannot be enabled together.")
        if use_route_guided_directional_refiner and use_route_guided_anisotropic_diffusion:
            raise ValueError("Route-guided directional refiner and anisotropic diffusion cannot be enabled together.")
        if (
            use_low_resolution_subpixel_additive_fusion
            and use_low_resolution_subpixel_separable_concat_fusion
        ):
            raise ValueError("LSAF and LSSCF are mutually exclusive decoder fusion modes.")
        if use_low_resolution_subpixel_additive_fusion and any(
            (
                use_frequency_gate,
                use_semantic_skip_gate,
                use_detail_residual_gate,
                use_structure_harmonic_fusion,
                use_boundary_logit_calibration,
                use_sdf_head,
                use_edge_mask_consistency_refiner,
                use_deep_supervision,
                use_route_guided_directional_refiner,
                use_route_guided_anisotropic_diffusion,
                use_local_semantic_continuity_calibrator,
                use_local_pixel_reassembly_residual,
                use_local_spectral_confidence_rejector,
                use_high_resolution_spectral_detail_gate,
            )
        ):
            raise ValueError("LSAF is an independent decoder ablation and cannot be combined with optional decoder modules.")
        if use_low_resolution_subpixel_separable_concat_fusion and any(
            (
                use_frequency_gate,
                use_semantic_skip_gate,
                use_detail_residual_gate,
                use_structure_harmonic_fusion,
                use_boundary_logit_calibration,
                use_sdf_head,
                use_edge_mask_consistency_refiner,
                # Deep supervision is deliberately composed with LSSCF/DPER-RSDM for the
                # experiment in runs/experiments/deep_supervision; the isolation rule above
                # still applies to every other optional decoder module.
                use_route_guided_directional_refiner,
                use_route_guided_anisotropic_diffusion,
                use_local_semantic_continuity_calibrator,
                use_local_pixel_reassembly_residual,
                use_local_spectral_confidence_rejector,
                use_high_resolution_spectral_detail_gate,
            )
        ):
            raise ValueError("LSSCF is an independent decoder ablation and cannot be combined with optional decoder modules.")
        c1, c2, c3, c4 = channels
        self.narrow_transport_feature_channels = (c1, c2, c3, c4)
        self.use_low_resolution_subpixel_additive_fusion = bool(use_low_resolution_subpixel_additive_fusion)
        self.use_low_resolution_subpixel_separable_concat_fusion = bool(
            use_low_resolution_subpixel_separable_concat_fusion
        )
        self.uses_low_resolution_subpixel_fusion = (
            self.use_low_resolution_subpixel_additive_fusion
            or self.use_low_resolution_subpixel_separable_concat_fusion
        )
        self.decoder_channels = int(decoder_channels)
        self.use_depthwise_separable_mask_head = False
        self.use_full_width_residual_mask_head = False
        self.use_difference_aware_up1_fusion = False
        self.use_narrow_transport_decoder = False
        self.narrow_transport_decoder: NarrowTransportDecoder | None = None
        self.use_partial_narrow_transport_decoder = False
        self.partial_narrow_transport_decoder: PartialNarrowTransportDecoder | None = None
        if self.use_low_resolution_subpixel_additive_fusion:
            self.proj4: nn.Module | None = None
            self.up3: nn.Module = LowResolutionSubpixelAdditiveUpBlock(c4, c3, c3)
            self.up2: nn.Module = LowResolutionSubpixelAdditiveUpBlock(c3, c2, c2)
            self.up1: nn.Module = LowResolutionSubpixelAdditiveUpBlock(c2, c1, decoder_channels)
        elif self.use_low_resolution_subpixel_separable_concat_fusion:
            self.proj4 = None
            self.up3 = LowResolutionSubpixelSeparableConcatUpBlock(c4, c3, c3)
            self.up2 = LowResolutionSubpixelSeparableConcatUpBlock(c3, c2, c2)
            self.up1 = LowResolutionSubpixelSeparableConcatUpBlock(c2, c1, decoder_channels)
        else:
            self.proj4 = ConvBNAct(c4, c3, kernel_size=1, padding=0)
            self.up3 = UpBlock(c3, c3, c3, use_semantic_skip_gate=use_semantic_skip_gate)
            self.up2 = UpBlock(c3, c2, c2, use_semantic_skip_gate=use_semantic_skip_gate)
            self.up1 = UpBlock(c2, c1, decoder_channels, use_semantic_skip_gate=use_semantic_skip_gate)
        self.refine = nn.Sequential(DWConvBlock(decoder_channels), DWConvBlock(decoder_channels))
        self.mask_head = nn.Sequential(ConvBNAct(decoder_channels, decoder_channels), nn.Conv2d(decoder_channels, 1, 1))
        self.edge_head = nn.Sequential(ConvBNAct(decoder_channels, decoder_channels), nn.Conv2d(decoder_channels, 1, 1))
        self.reliability_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        self.frequency_gate = HaarFrequencyGate(decoder_channels) if use_frequency_gate else None
        self.detail_gates: nn.ModuleList | None = None
        if use_detail_residual_gate:
            # Construct after all baseline decoder layers to preserve shared
            # C3 parameter initialization for an identical random seed.
            self.detail_gates = nn.ModuleList(
                [
                    DetailResidualSkipGate(c3, c3),
                    DetailResidualSkipGate(c3, c2),
                    DetailResidualSkipGate(c2, c1),
                ]
            )
        # Construct after all baseline decoder layers so an enabled ablation
        # keeps every shared C3 parameter initialized identically for a seed.
        self.structure_harmonic_fusion = (
            StructureConsistentHarmonicFusion([c1, c2, c3, c4], decoder_channels)
            if use_structure_harmonic_fusion
            else None
        )
        # Keep all shared decoder parameters initialized before this ablation.
        self.boundary_logit_calibrator = (
            BoundaryLogitCalibrator(decoder_channels) if use_boundary_logit_calibration else None
        )
        # Construct last so enabling SDF supervision does not perturb the
        # initialization sequence of any baseline or existing ablation layer.
        self.sdf_head = SDFHead(decoder_channels) if use_sdf_head else None
        # Create after all existing decoder components to preserve their shared
        # initialization sequence when this independent ablation is enabled.
        self.edge_mask_consistency_refiner = (
            EdgeMaskConsistencyRefiner(decoder_channels) if use_edge_mask_consistency_refiner else None
        )
        self.deep_supervision_enabled = bool(use_deep_supervision)
        self.aux_mask_quarter: nn.Module | None = None
        self.aux_edge_quarter: nn.Module | None = None
        self.aux_mask_half: nn.Module | None = None
        self.aux_edge_half: nn.Module | None = None
        if self.deep_supervision_enabled:
            # Create all training-only heads last so their initialization does
            # not perturb baseline or existing optional decoder parameters.
            # Training-only 1x1 heads: negligible parameter delta, inference path unchanged.
            self.aux_mask_quarter = nn.Conv2d(c3, 1, kernel_size=1)
            self.aux_edge_quarter = nn.Conv2d(c3, 1, kernel_size=1)
            self.aux_mask_half = nn.Conv2d(c2, 1, kernel_size=1)
            self.aux_edge_half = nn.Conv2d(c2, 1, kernel_size=1)
        # Keep this last so enabling RG-DDP leaves every existing decoder
        # parameter at its original seeded initialization.
        self.route_guided_directional_refiner = (
            RouteGuidedDirectionalRefiner(decoder_channels) if use_route_guided_directional_refiner else None
        )
        # Construct RG-AFD after every existing module, including RG-DDP, so
        # it cannot perturb any pre-existing seeded parameter initialization.
        self.route_guided_anisotropic_diffusion = (
            RouteGuidedAnisotropicDiffusionRefiner(decoder_channels)
            if use_route_guided_anisotropic_diffusion
            else None
        )
        # Construct after every existing decoder component so enabling LSCC
        # preserves all shared seeded parameter initializations.
        self.local_semantic_continuity_calibrator = (
            LocalSemanticContinuityCalibrator(decoder_channels, c3)
            if use_local_semantic_continuity_calibrator
            else None
        )
        # Construct last so LPR2 cannot alter the seeded initialization of any
        # baseline layer or previously evaluated optional component.
        self.local_pixel_reassembly_residual = (
            LocalPixelReassemblyResidual(decoder_channels) if use_local_pixel_reassembly_residual else None
        )
        # Construct after every prior decoder component so LSCR cannot alter
        # their seeded initialization under an otherwise identical experiment.
        self.local_spectral_confidence_rejector = (
            LocalSpectralConfidenceRejector(decoder_channels, c3)
            if use_local_spectral_confidence_rejector
            else None
        )
        # Construct after every prior decoder component so HRSDG cannot alter
        # shared seeded parameter initialization under an identical experiment.
        self.high_resolution_spectral_detail_gate = (
            HighResolutionSpectralDetailGate(decoder_channels, c3)
            if use_high_resolution_spectral_detail_gate
            else None
        )
        self.return_training_mask_feature = bool(return_training_mask_feature)

    def enable_depthwise_separable_mask_head(self) -> None:
        """Replace the registered dense mask head after full model construction."""

        if self.use_depthwise_separable_mask_head:
            raise RuntimeError("DepthwiseSeparableMaskHead is already enabled.")
        if not self.use_low_resolution_subpixel_separable_concat_fusion:
            raise ValueError("DPER-DSMH requires the unchanged LSSCF decoder fusion.")
        if self.decoder_channels != 40:
            raise ValueError("DPER-DSMH is fixed to decoder_channels=40.")
        if not isinstance(self.mask_head, nn.Sequential):
            raise RuntimeError("DPER-DSMH expected the original dense Sequential mask head before replacement.")
        self.mask_head = DepthwiseSeparableMaskHead(self.decoder_channels)
        self.use_depthwise_separable_mask_head = True

    def enable_full_width_residual_mask_head(self) -> None:
        """Replace only the dense mask head with the isolated FWRMH candidate."""

        if self.use_full_width_residual_mask_head:
            raise RuntimeError("FullWidthResidualMaskHead is already enabled.")
        if self.use_depthwise_separable_mask_head:
            raise ValueError("FWRMH and DPER-DSMH are mutually exclusive mask-head replacements.")
        if not self.use_low_resolution_subpixel_separable_concat_fusion:
            raise ValueError("DPSM80-FWRMH requires the unchanged LSSCF decoder fusion.")
        if self.decoder_channels != 40:
            raise ValueError("DPSM80-FWRMH is fixed to decoder_channels=40.")
        if not isinstance(self.mask_head, nn.Sequential):
            raise RuntimeError("DPSM80-FWRMH expected the original dense Sequential mask head before replacement.")
        self.mask_head = FullWidthResidualMaskHead(self.decoder_channels, hidden_channels=16)
        self.use_full_width_residual_mask_head = True

    def enable_difference_aware_up1_fusion(self) -> None:
        """Replace only native LSSCF Up1 after the full decoder is constructed."""

        if self.use_difference_aware_up1_fusion:
            raise RuntimeError("DifferenceAwareSubpixelUpBlock is already enabled.")
        if not self.use_low_resolution_subpixel_separable_concat_fusion:
            raise ValueError("DAFU1 requires the unchanged LSSCF decoder fusion.")
        if self.decoder_channels != 40:
            raise ValueError("DAFU1 is fixed to decoder_channels=40.")
        if not isinstance(self.up1, LowResolutionSubpixelSeparableConcatUpBlock):
            raise RuntimeError("DAFU1 expected the original LowResolutionSubpixelSeparableConcatUpBlock at up1.")
        self.up1 = DifferenceAwareSubpixelUpBlock(self.up1)
        self.use_difference_aware_up1_fusion = True

    def enable_narrow_transport_decoder(self, transport_dim: int) -> None:
        """Replace only LSSCF transport/refine after complete model construction."""

        if self.use_narrow_transport_decoder or self.narrow_transport_decoder is not None:
            raise RuntimeError("NarrowTransportDecoder is already enabled.")
        if not self.use_low_resolution_subpixel_separable_concat_fusion:
            raise ValueError("DPER-RSDM-NTD requires the original LSSCF decoder baseline.")
        if self.decoder_channels != 40:
            raise ValueError("DPER-RSDM-NTD is fixed to the original 40-channel dense prediction heads.")
        self.narrow_transport_decoder = NarrowTransportDecoder(
            list(self.narrow_transport_feature_channels),
            transport_dim,
            self.decoder_channels,
        )
        # Reassignment to None unregisters the old wide transport modules, so
        # they cannot remain in state_dict, Params, FLOPs, or optimizer groups.
        self.proj4 = None
        self.up3 = None
        self.up2 = None
        self.up1 = None
        self.refine = None
        self.use_narrow_transport_decoder = True

    def enable_partial_narrow_transport_decoder(self, transport_dim: int) -> None:
        """Replace only the deep F4 transport while preserving F3/F2/F1 fusion."""

        if self.use_narrow_transport_decoder or self.narrow_transport_decoder is not None:
            raise ValueError("PNTD and the all-scale NTD are mutually exclusive.")
        if self.use_partial_narrow_transport_decoder or self.partial_narrow_transport_decoder is not None:
            raise RuntimeError("PartialNarrowTransportUpBlock is already enabled.")
        if not self.use_low_resolution_subpixel_separable_concat_fusion:
            raise ValueError("DPER-RSDM-PNTD requires the original LSSCF decoder baseline.")
        if self.decoder_channels != 40:
            raise ValueError("DPER-RSDM-PNTD is fixed to the original 40-channel dense prediction heads.")
        if not isinstance(self.up3, LowResolutionSubpixelSeparableConcatUpBlock) or not isinstance(
            self.up2, LowResolutionSubpixelSeparableConcatUpBlock
        ):
            raise RuntimeError("DPER-RSDM-PNTD expected the original registered LSSCF up3/up2 blocks.")
        replacement = PartialNarrowTransportDecoder(self.up3, self.up2, transport_dim)
        # Keep one registered path only. The old LSSCF up3 (including its
        # decoder_projection/PixelShuffle) is released after its preserved
        # F3 fusion children have been moved into the replacement.
        self.up3 = None
        self.up2 = None
        self.partial_narrow_transport_decoder = replacement
        self.use_partial_narrow_transport_decoder = True

    def _mask_only_dependencies(self) -> tuple[str, ...]:
        """Return enabled modules whose mask prediction needs auxiliary logits."""

        dependencies = (
            ("frequency_gate", self.frequency_gate),
            ("boundary_logit_calibrator", self.boundary_logit_calibrator),
            ("edge_mask_consistency_refiner", self.edge_mask_consistency_refiner),
            ("local_semantic_continuity_calibrator", self.local_semantic_continuity_calibrator),
            ("local_spectral_confidence_rejector", self.local_spectral_confidence_rejector),
        )
        return tuple(name for name, module in dependencies if module is not None)

    def _decode_to_mask_feature(
        self,
        features: list[torch.Tensor],
        route_s3: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared decoder trunk for the full and the mask-only entry points.

        This body was previously duplicated verbatim in :meth:`forward` and
        :meth:`forward_mask_only`.  It is extracted without reordering a single
        operation, so both public paths keep producing the same mask logits.
        """

        x1, x2, x3, x4 = features
        detail_gates = self.detail_gates
        if self.narrow_transport_decoder is not None:
            x, x_quarter, x_half = self.narrow_transport_decoder(features)
        elif self.partial_narrow_transport_decoder is not None:
            assert self.up1 is not None
            x_quarter, x_half = self.partial_narrow_transport_decoder(x4, x3, x2)
            x = self.up1(x_half, x1)
        elif self.uses_low_resolution_subpixel_fusion:
            assert self.up3 is not None and self.up2 is not None and self.up1 is not None
            x_quarter = self.up3(x4, x3)
            x_half = self.up2(x_quarter, x2)
            x = self.up1(x_half, x1)
        else:
            assert self.proj4 is not None
            x_quarter = self.up3(self.proj4(x4), x3, None if detail_gates is None else detail_gates[0])
            x_half = self.up2(x_quarter, x2, None if detail_gates is None else detail_gates[1])
            x = self.up1(x_half, x1, None if detail_gates is None else detail_gates[2])
        if self.narrow_transport_decoder is None:
            assert self.refine is not None
            x = self.refine(x)
        if self.structure_harmonic_fusion is not None:
            x = self.structure_harmonic_fusion(x, [x1, x2, x3, x4])
        if self.route_guided_directional_refiner is not None:
            if route_s3 is None:
                raise ValueError("RouteGuidedDirectionalRefiner requires stage3 route logits.")
            x = self.route_guided_directional_refiner(x, route_s3)
        if self.route_guided_anisotropic_diffusion is not None:
            if route_s3 is None:
                raise ValueError("RouteGuidedAnisotropicDiffusionRefiner requires stage3 route logits.")
            x = self.route_guided_anisotropic_diffusion(x, route_s3)
        if self.local_pixel_reassembly_residual is not None:
            x = self.local_pixel_reassembly_residual(x)
        if self.high_resolution_spectral_detail_gate is not None:
            x = self.high_resolution_spectral_detail_gate(x, x3)
        return x, x_quarter, x_half

    def forward_mask_only(
        self,
        features: list[torch.Tensor],
        output_size: tuple[int, int] | None = None,
        route_s3: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the eval-only mask path without evaluating auxiliary heads.

        This is deliberately separate from :meth:`forward` so the established
        training/full-inference execution order remains byte-for-byte intact.
        """

        if self.training:
            raise RuntimeError("BoundaryAwareDecoder.forward_mask_only is only available in eval mode.")
        dependencies = self._mask_only_dependencies()
        if dependencies:
            raise RuntimeError(
                "Mask-only inference is unavailable because mask prediction depends on auxiliary outputs: "
                + ", ".join(dependencies)
            )

        x, _, _ = self._decode_to_mask_feature(features, route_s3)
        mask_logits = self.mask_head(x)
        if output_size is not None:
            mask_logits = F.interpolate(mask_logits, size=output_size, mode="bilinear", align_corners=False)
        return {"mask": mask_logits}

    def forward(
        self,
        features: list[torch.Tensor],
        output_size: tuple[int, int] | None = None,
        route_s3: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        x, x_quarter, x_half = self._decode_to_mask_feature(features, route_s3)
        x3 = features[2]
        reliability = self.reliability_head(x)
        if self.frequency_gate is not None:
            x = self.frequency_gate(x, torch.sigmoid(reliability))
            reliability = self.reliability_head(x)
        mask_logits = self.mask_head(x)
        edge_logits = self.edge_head(x)
        if self.boundary_logit_calibrator is not None:
            mask_logits = self.boundary_logit_calibrator(x, mask_logits, edge_logits)
        if self.edge_mask_consistency_refiner is not None:
            mask_logits = self.edge_mask_consistency_refiner(x, mask_logits, edge_logits)
        if self.local_semantic_continuity_calibrator is not None:
            mask_logits = self.local_semantic_continuity_calibrator(x, x3, mask_logits, edge_logits)
        if self.local_spectral_confidence_rejector is not None:
            mask_logits = self.local_spectral_confidence_rejector(x, x3, mask_logits, edge_logits, reliability)
        outputs = {
            "mask": mask_logits,
            "edge": edge_logits,
            "reliability": reliability,
        }
        if self.training and self.return_training_mask_feature:
            outputs["mask_feature"] = x
        if self.sdf_head is not None and self.training:
            outputs["sdf"] = self.sdf_head(x)
        if self.training and self.deep_supervision_enabled:
            assert self.aux_mask_quarter is not None and self.aux_edge_quarter is not None
            assert self.aux_mask_half is not None and self.aux_edge_half is not None
            outputs.update(
                {
                    "aux_mask_quarter": self.aux_mask_quarter(x_quarter),
                    "aux_edge_quarter": self.aux_edge_quarter(x_quarter),
                    "aux_mask_half": self.aux_mask_half(x_half),
                    "aux_edge_half": self.aux_edge_half(x_half),
                }
            )
        if output_size is not None:
            outputs = {
                name: value
                if name.startswith("aux_") or name == "mask_feature"
                else F.interpolate(value, size=output_size, mode="bilinear", align_corners=False)
                for name, value in outputs.items()
            }
        return outputs
