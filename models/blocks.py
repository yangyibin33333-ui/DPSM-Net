from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .edge_direction import sobel_edge
from .mamba_blocks import (
    CommonPrivateDirectionPartitionedSharedMamba,
    DirectionPartitionedSharedMamba,
    SingleRasterMamba80,
    TangentRoutedMambaBlock,
)
from .transformer_blocks import TransformerGlobalBlock


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        act: bool = True,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if act:
            layers.append(nn.GELU())
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DWConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=3, groups=channels),
            ConvBNAct(channels, channels, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class RIFTStructuralDirectionalModelingBlock(nn.Module):
    """One half-resolution cooperative structural-directional residual block."""

    kernel_size = 13
    dilation = 3
    hidden_ratio = 1.0

    def __init__(self, channels: int, kernel_size: int = 13, dilation: int = 3):
        super().__init__()
        if channels < 1:
            raise ValueError("RSDM channels must be positive.")
        if (
            not isinstance(kernel_size, int)
            or isinstance(kernel_size, bool)
            or kernel_size < 3
            or kernel_size % 2 == 0
        ):
            raise ValueError(
                "RSDM kernel_size must be an odd integer greater than or equal to 3, "
                f"got {kernel_size!r}."
            )
        if (
            not isinstance(dilation, int)
            or isinstance(dilation, bool)
            or dilation < 1
        ):
            raise ValueError(
                "RSDM dilation must be an integer greater than or equal to 1, "
                f"got {dilation!r}."
            )
        groups = 8 if channels % 8 == 0 else 1
        self.channels = int(channels)
        # Plain metadata only.  It consumes no RNG or state registration, so
        # the frozen 13/3 path retains its module order and initialization.
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.pre_norm = nn.GroupNorm(groups, self.channels)
        self.input_projection = nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False)
        self.local_structure = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
            groups=self.channels,
            bias=False,
        )
        self.horizontal = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=(1, self.kernel_size),
            padding=(0, self.kernel_size // 2),
            groups=self.channels,
            bias=False,
        )
        self.vertical = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=(self.kernel_size, 1),
            padding=(self.kernel_size // 2, 0),
            groups=self.channels,
            bias=False,
        )
        self.diagonal = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=3,
            padding=self.dilation,
            dilation=self.dilation,
            groups=self.channels,
            bias=False,
        )
        self.structure_gate = nn.Conv2d(self.channels, self.channels, kernel_size=1)
        self.output_norm = nn.GroupNorm(groups, self.channels)
        self.activation = nn.GELU()
        self.output_projection = nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_projection(self.pre_norm(x))
        structure = self.local_structure(z)
        directional = self.horizontal(z) + self.vertical(z) + self.diagonal(z)
        gate = torch.sigmoid(self.structure_gate(structure + directional))
        fused = structure + gate * directional
        residual = self.output_projection(self.activation(self.output_norm(fused)))
        return x + residual


class StripConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.horizontal = nn.Conv2d(channels, channels, (1, kernel_size), padding=(0, padding), groups=channels)
        self.vertical = nn.Conv2d(channels, channels, (kernel_size, 1), padding=(padding, 0), groups=channels)
        self.mix = ConvBNAct(channels * 2, channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mix(torch.cat([self.horizontal(x), self.vertical(x)], dim=1))


class CNNEdgeStem(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(in_channels + 1, out_channels, kernel_size=3),
            DWConvBlock(out_channels),
            StripConvBlock(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(torch.cat([x, sobel_edge(x)], dim=1))


class DetailPreservingHalfStem(nn.Module):
    """Retain pixel evidence while moving early semantics to half resolution.

    The full-resolution branch intentionally contains one 1-by-1 projection
    only.  All spatial semantic processing starts after PixelUnshuffle, so the
    four RGB/Sobel sub-pixel phases are carried losslessly to half resolution.
    """

    def __init__(self, in_channels: int, semantic_channels: int, detail_channels: int):
        super().__init__()
        if detail_channels < 1 or detail_channels >= semantic_channels:
            raise ValueError(
                "DPER detail_channels must be positive and smaller than semantic_channels, "
                f"got {detail_channels} for {semantic_channels} semantic channels."
            )
        evidence_channels = in_channels + 1
        self.in_channels = in_channels
        self.semantic_channels = semantic_channels
        self.detail_channels = detail_channels
        self.evidence_channels = evidence_channels
        self.full_detail_projection = nn.Conv2d(evidence_channels, detail_channels, kernel_size=1, bias=False)
        self.pixel_unshuffle = nn.PixelUnshuffle(2)
        self.semantic_projection = ConvBNAct(
            evidence_channels * 4,
            semantic_channels,
            kernel_size=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or x.size(1) != self.in_channels:
            raise ValueError(
                f"DPER expected [B, {self.in_channels}, H, W] input, received {tuple(x.shape)}."
            )
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError("DPER requires even input height and width for PixelUnshuffle(2).")
        evidence = torch.cat([x, sobel_edge(x)], dim=1)
        full_detail = self.full_detail_projection(evidence)
        semantic_half = self.semantic_projection(self.pixel_unshuffle(evidence))
        return full_detail, semantic_half


class NarrowDualResolutionStage1(nn.Module):
    """One-way local-detail exchange inserted before DPER's half-resolution Stage1.

    The block intentionally keeps the original DPER stem, PixelUnshuffle
    semantics, Stage1, and reconstruction intact.  It first adds one narrow
    full-resolution depthwise neighbourhood residual to the 16-channel detail
    stream, then sends that learned local evidence once to the 40-channel
    half-resolution semantic stream.  There is deliberately no reverse
    exchange, gate, learnable scale, or post-PixelShuffle processing.
    """

    def __init__(self, detail_channels: int, semantic_channels: int):
        super().__init__()
        if detail_channels < 1 or semantic_channels < 1:
            raise ValueError("NDRS detail_channels and semantic_channels must be positive.")
        self.detail_channels = detail_channels
        self.semantic_channels = semantic_channels

        # The only full-resolution spatial operation added by NDRS.
        self.local_depthwise = nn.Conv2d(
            detail_channels,
            detail_channels,
            kernel_size=3,
            padding=1,
            groups=detail_channels,
            bias=False,
        )
        self.local_norm = nn.BatchNorm2d(detail_channels)
        self.local_activation = nn.GELU()
        self.local_pointwise = nn.Conv2d(detail_channels, detail_channels, kernel_size=1, bias=False)

        # The sole high-to-low exchange.  It is spatially reduced before the
        # 16-to-40 pointwise projection, avoiding a high-resolution expansion.
        self.exchange_depthwise = nn.Conv2d(
            detail_channels,
            detail_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=detail_channels,
            bias=False,
        )
        self.exchange_norm = nn.BatchNorm2d(detail_channels)
        self.exchange_activation = nn.GELU()
        self.exchange_projection = nn.Conv2d(detail_channels, semantic_channels, kernel_size=1, bias=False)

    def forward(
        self,
        full_detail: torch.Tensor,
        semantic_half: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if full_detail.ndim != 4 or full_detail.size(1) != self.detail_channels:
            raise ValueError(
                f"NDRS expected full_detail with {self.detail_channels} channels, "
                f"received {tuple(full_detail.shape)}."
            )
        if semantic_half.ndim != 4 or semantic_half.size(1) != self.semantic_channels:
            raise ValueError(
                f"NDRS expected semantic_half with {self.semantic_channels} channels, "
                f"received {tuple(semantic_half.shape)}."
            )
        expected_half_size = (full_detail.shape[-2] // 2, full_detail.shape[-1] // 2)
        if full_detail.shape[-2] % 2 or full_detail.shape[-1] % 2 or semantic_half.shape[-2:] != expected_half_size:
            raise ValueError(
                "NDRS requires semantic_half to be exactly half the full_detail spatial size, "
                f"got detail {tuple(full_detail.shape[-2:])} and semantic {tuple(semantic_half.shape[-2:])}."
            )

        local_residual = self.local_pointwise(self.local_activation(self.local_norm(self.local_depthwise(full_detail))))
        local_detail = full_detail + local_residual
        exchange = self.exchange_projection(
            self.exchange_activation(self.exchange_norm(self.exchange_depthwise(local_detail)))
        )
        return local_detail, semantic_half + exchange


class FullResolutionJointSpatialExchange(nn.Module):
    """DPER-FJSE's single dense local mapping and one-way phase exchange.

    A dense 16-to-16 3-by-3 is the sole new full-resolution spatial operator.
    Its local context is both added to DPER's full-detail tensor and carried
    losslessly through PixelUnshuffle before one 64-to-40 semantic residual.
    """

    def __init__(self, detail_channels: int, semantic_channels: int):
        super().__init__()
        if detail_channels < 1 or semantic_channels < 1:
            raise ValueError("FJSE detail_channels and semantic_channels must be positive.")
        self.detail_channels = detail_channels
        self.semantic_channels = semantic_channels
        self.local_dense = nn.Conv2d(
            detail_channels,
            detail_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        )
        self.local_norm = nn.BatchNorm2d(detail_channels)
        self.local_activation = nn.GELU()
        self.phase_unshuffle = nn.PixelUnshuffle(2)
        self.exchange_projection = nn.Conv2d(
            detail_channels * 4,
            semantic_channels,
            kernel_size=1,
            bias=False,
        )
        self.exchange_norm = nn.BatchNorm2d(semantic_channels)

    def forward(
        self,
        full_detail: torch.Tensor,
        semantic_half: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if full_detail.ndim != 4 or full_detail.size(1) != self.detail_channels:
            raise ValueError(
                f"FJSE expected full_detail with {self.detail_channels} channels, "
                f"received {tuple(full_detail.shape)}."
            )
        if semantic_half.ndim != 4 or semantic_half.size(1) != self.semantic_channels:
            raise ValueError(
                f"FJSE expected semantic_half with {self.semantic_channels} channels, "
                f"received {tuple(semantic_half.shape)}."
            )
        expected_half_size = (full_detail.shape[-2] // 2, full_detail.shape[-1] // 2)
        if full_detail.shape[-2] % 2 or full_detail.shape[-1] % 2 or semantic_half.shape[-2:] != expected_half_size:
            raise ValueError(
                "FJSE requires semantic_half to be exactly half the full_detail spatial size, "
                f"got detail {tuple(full_detail.shape[-2:])} and semantic {tuple(semantic_half.shape[-2:])}."
            )

        local_context = self.local_activation(self.local_norm(self.local_dense(full_detail)))
        local_detail = full_detail + local_context
        phase_context = self.phase_unshuffle(local_context)
        semantic_delta = self.exchange_norm(self.exchange_projection(phase_context))
        return local_detail, semantic_half + semantic_delta


class FixedHaarPhaseDecomposition(nn.Module):
    """Fixed orthogonal 2-by-2 Haar decomposition implemented with slices."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Haar rebase expected BCHW input, received {tuple(x.shape)}.")
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError("Haar rebase requires even input height and width.")
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]
        return (
            (x00 + x01 + x10 + x11) * 0.5,
            (x00 - x01 + x10 - x11) * 0.5,
            (x00 + x01 - x10 - x11) * 0.5,
            (x00 - x01 - x10 + x11) * 0.5,
        )


class HaarTopologyPreservingRebase(nn.Module):
    """DPER half-resolution semantic entry with gated Haar high frequencies."""

    def __init__(
        self,
        in_channels: int,
        semantic_channels: int,
        detail_channels: int,
        *,
        full_detail_projection: nn.Conv2d | None = None,
    ):
        super().__init__()
        if detail_channels < 1 or detail_channels >= semantic_channels:
            raise ValueError(
                "HTPR detail_channels must be positive and smaller than semantic_channels, "
                f"got {detail_channels} for {semantic_channels} semantic channels."
            )
        evidence_channels = in_channels + 1
        if full_detail_projection is not None and (
            full_detail_projection.in_channels != evidence_channels
            or full_detail_projection.out_channels != detail_channels
            or full_detail_projection.kernel_size != (1, 1)
        ):
            raise ValueError("HTPR full_detail_projection must match DPER's 1-by-1 evidence projection.")
        self.in_channels = in_channels
        self.semantic_channels = semantic_channels
        self.detail_channels = detail_channels
        self.haar_rebase = FixedHaarPhaseDecomposition()
        # Reusing this module retains DPER's lightweight full-resolution
        # evidence path and its seeded initialization exactly.
        self.full_detail_projection = (
            full_detail_projection
            if full_detail_projection is not None
            else nn.Conv2d(evidence_channels, detail_channels, kernel_size=1, bias=False)
        )
        self.semantic_projection = ConvBNAct(in_channels * 4, semantic_channels, kernel_size=1, padding=0)
        self.hf_projection = nn.Conv2d(in_channels * 3, semantic_channels, kernel_size=1, bias=False)
        self.hf_norm1 = nn.BatchNorm2d(semantic_channels)
        self.hf_activation = nn.GELU()
        self.hf_depthwise = nn.Conv2d(
            semantic_channels,
            semantic_channels,
            kernel_size=3,
            padding=1,
            groups=semantic_channels,
            bias=False,
        )
        self.hf_norm2 = nn.BatchNorm2d(semantic_channels)
        self.low_frequency_gate = nn.Conv2d(in_channels, semantic_channels, kernel_size=1)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def inverse_haar(
        ll: torch.Tensor,
        lh: torch.Tensor,
        hl: torch.Tensor,
        hh: torch.Tensor,
    ) -> torch.Tensor:
        if not (ll.shape == lh.shape == hl.shape == hh.shape):
            raise ValueError("Haar inverse requires LL, LH, HL, and HH tensors with identical shapes.")
        batch, channels, height, width = ll.shape
        restored = ll.new_empty(batch, channels, height * 2, width * 2)
        restored[:, :, 0::2, 0::2] = (ll + lh + hl + hh) * 0.5
        restored[:, :, 0::2, 1::2] = (ll - lh + hl - hh) * 0.5
        restored[:, :, 1::2, 0::2] = (ll + lh - hl - hh) * 0.5
        restored[:, :, 1::2, 1::2] = (ll - lh - hl + hh) * 0.5
        return restored

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or x.size(1) != self.in_channels:
            raise ValueError(
                f"HTPR expected [B, {self.in_channels}, H, W] input, received {tuple(x.shape)}."
            )
        ll, lh, hl, hh = self.haar_rebase(x)
        evidence = torch.cat([x, sobel_edge(x)], dim=1)
        full_detail = self.full_detail_projection(evidence)
        base = self.semantic_projection(torch.cat([ll, lh, hl, hh], dim=1))
        high_frequency = torch.cat([lh, hl, hh], dim=1)
        high_frequency = self.hf_projection(high_frequency)
        high_frequency = self.hf_norm1(high_frequency)
        high_frequency = self.hf_activation(high_frequency)
        high_frequency = self.hf_norm2(self.hf_depthwise(high_frequency))
        gate = torch.sigmoid(self.low_frequency_gate(ll))
        return full_detail, base + self.alpha * gate * high_frequency


class HalfResolutionStage1Block(nn.Module):
    """Single dense residual 3-by-3 Stage1 used by DPER."""

    def __init__(self, channels: int):
        super().__init__()
        self.residual = ConvBNAct(channels, channels, kernel_size=3, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual(x)


class SubpixelPhaseConsistencyRefiner(nn.Module):
    """Learn local agreement across PixelShuffle phases through a zero residual."""

    def __init__(self, channels: int):
        super().__init__()
        if channels < 1:
            raise ValueError(f"SPCR channels must be positive, got {channels}.")
        self.channels = channels
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.norm1 = nn.BatchNorm2d(channels)
        self.activation = nn.GELU()
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)
        # This makes delta exactly zero in both train and eval modes at
        # initialization, preserving the original DPER forward bit-for-bit.
        nn.init.zeros_(self.norm2.weight)
        nn.init.zeros_(self.norm2.bias)

    def forward(self, semantic_detail: torch.Tensor) -> torch.Tensor:
        if semantic_detail.ndim != 4 or semantic_detail.size(1) != self.channels:
            raise ValueError(
                f"SPCR expected [B, {self.channels}, H, W], received {tuple(semantic_detail.shape)}."
            )
        delta = self.norm2(self.pointwise(self.activation(self.norm1(self.depthwise(semantic_detail)))))
        return semantic_detail + delta


class SubpixelDetailReconstruction(nn.Module):
    """Rebuild the full-resolution Stage1 skip without a spatial fusion layer."""

    def __init__(self, semantic_channels: int, detail_channels: int):
        super().__init__()
        if detail_channels < 1 or detail_channels >= semantic_channels:
            raise ValueError(
                "DPER detail_channels must be positive and smaller than semantic_channels, "
                f"got {detail_channels} for {semantic_channels} semantic channels."
            )
        self.semantic_channels = semantic_channels
        self.detail_channels = detail_channels
        self.reconstructed_channels = semantic_channels - detail_channels
        self.semantic_projection = nn.Conv2d(
            semantic_channels,
            self.reconstructed_channels * 4,
            kernel_size=1,
            bias=False,
        )
        self.pixel_shuffle = nn.PixelShuffle(2)
        # Assigned only after the complete baseline DPER model has been
        # initialized, so an enabled SPCR cannot perturb shared parameters.
        self.phase_refiner: SubpixelPhaseConsistencyRefiner | None = None
        self._initialize_phase_consistent_projection()

    def _initialize_phase_consistent_projection(self) -> None:
        """Copy one LR projection into all four PixelShuffle phases (ICNR style)."""

        with torch.no_grad():
            base = self.semantic_projection.weight[: self.reconstructed_channels].detach().clone()
            self.semantic_projection.weight.copy_(base.repeat_interleave(4, dim=0))

    def forward(self, full_detail: torch.Tensor, semantic_half: torch.Tensor) -> torch.Tensor:
        if full_detail.size(1) != self.detail_channels:
            raise ValueError(
                f"DPER expected {self.detail_channels} full-detail channels, received {full_detail.size(1)}."
            )
        if semantic_half.size(1) != self.semantic_channels:
            raise ValueError(
                f"DPER expected {self.semantic_channels} half-resolution channels, received {semantic_half.size(1)}."
            )
        semantic_detail = self.pixel_shuffle(self.semantic_projection(semantic_half))
        if self.phase_refiner is not None:
            semantic_detail = self.phase_refiner(semantic_detail)
        if semantic_detail.shape[-2:] != full_detail.shape[-2:]:
            raise ValueError(
                "DPER reconstruction requires full_detail to be exactly two times semantic_half, "
                f"got detail {tuple(full_detail.shape[-2:])} and semantic {tuple(semantic_half.shape[-2:])}."
            )
        return torch.cat([full_detail, semantic_detail], dim=1)


class DensityBalancedPrototypeContext(nn.Module):
    """Full-resolution cross-attention against density-balanced stage-3 prototypes."""

    def __init__(self, channels: int, attention_dim: int = 64, num_heads: int = 4):
        super().__init__()
        if attention_dim % num_heads != 0:
            raise ValueError("DB-PCA attention_dim must be divisible by num_heads.")
        self.channels = channels
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.query = nn.Linear(channels, attention_dim, bias=False)
        self.key = nn.Linear(channels, attention_dim, bias=False)
        self.value = nn.Linear(channels, attention_dim, bias=False)
        self.output_projection = nn.Linear(attention_dim, channels)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    @staticmethod
    def _replicate_shift(feature: torch.Tensor, row_offset: int, column_offset: int) -> torch.Tensor:
        if row_offset not in (-1, 0, 1) or column_offset not in (-1, 0, 1):
            raise ValueError("DB-PCA supports one-pixel neighbor shifts only.")
        height, width = feature.shape[-2:]
        padded = F.pad(feature, (1, 1, 1, 1), mode="replicate")
        top, left = 1 + row_offset, 1 + column_offset
        return padded[..., top : top + height, left : left + width]

    def _local_density(self, feature: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            detached = feature.detach()
            normalized = F.normalize(detached, p=2, dim=1, eps=1e-6)
            similarities = []
            for row_offset, column_offset in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                neighbor = self._replicate_shift(detached, row_offset, column_offset)
                neighbor_normalized = F.normalize(neighbor, p=2, dim=1, eps=1e-6)
                similarities.append(((normalized * neighbor_normalized).sum(dim=1, keepdim=True) + 1.0) * 0.5)
            return torch.stack(similarities, dim=0).mean(dim=0).clamp(0.0, 1.0)

    @staticmethod
    def _restore_prototype_norm(
        source_tokens: torch.Tensor,
        density: torch.Tensor,
        sign: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(sign * density.flatten(1) / 0.1, dim=1).detach()
        prototype = (weights.unsqueeze(-1) * source_tokens).sum(dim=1)
        target_norm = (weights * source_tokens.norm(dim=-1)).sum(dim=1)
        return F.normalize(prototype, p=2, dim=-1, eps=1e-6) * target_norm.unsqueeze(-1), target_norm

    def prototypes_and_target_norms(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return two deterministic prototypes per 2-by-4 spatial cell."""
        batch, channels, height, width = feature.shape
        if channels != self.channels:
            raise ValueError(f"DB-PCA expected {self.channels} channels, received {channels}.")
        density = self._local_density(feature)
        prototypes: list[torch.Tensor] = []
        target_norms: list[torch.Tensor] = []
        for row in range(2):
            top, bottom = row * height // 2, (row + 1) * height // 2
            for column in range(4):
                left, right = column * width // 4, (column + 1) * width // 4
                source_tokens = feature[:, :, top:bottom, left:right].flatten(2).transpose(1, 2)
                cell_density = density[:, :, top:bottom, left:right]
                for sign in (1.0, -1.0):
                    prototype, target_norm = self._restore_prototype_norm(source_tokens, cell_density, sign)
                    prototypes.append(prototype)
                    target_norms.append(target_norm)
        return torch.stack(prototypes, dim=1), torch.stack(target_norms, dim=1)

    def _split_heads(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tokens.shape
        return tokens.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = feature.shape
        tokens = feature.flatten(2).transpose(1, 2)
        prototypes, _ = self.prototypes_and_target_norms(feature)
        query = self._split_heads(self.query(tokens))
        key = self._split_heads(self.key(prototypes))
        value = self._split_heads(self.value(prototypes))
        attended = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        attended = attended.transpose(1, 2).reshape(batch, height * width, self.attention_dim)
        residual = self.output_projection(attended).transpose(1, 2).reshape(batch, channels, height, width)
        return feature + residual


class SemanticConditionedLocalKernelResidual(nn.Module):
    """Use stage-4 semantics to select, but never provide, local stage-3 values."""

    _OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1))
    _SEMANTIC_DIM = 16
    _NUM_HEADS = 4
    _CONTEXT_GRID = (4, 4)
    _COMPATIBILITY_TEMPERATURE = 0.25

    def __init__(self, detail_channels: int, context_channels: int):
        super().__init__()
        if detail_channels % self._NUM_HEADS != 0:
            raise ValueError("SCLKR detail_channels must be divisible by four heads.")
        self.detail_channels = detail_channels
        self.context_channels = context_channels
        self.head_dim = self._SEMANTIC_DIM // self._NUM_HEADS
        self.q_proj = nn.Conv2d(detail_channels, self._SEMANTIC_DIM, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(context_channels, self._SEMANTIC_DIM, kernel_size=1, bias=False)
        self.affinity_to_kernel = nn.Parameter(torch.empty(self._NUM_HEADS, 16, 9))
        self.kernel_bias = nn.Parameter(torch.zeros(self._NUM_HEADS, 9))
        self.out_proj = nn.Conv2d(detail_channels, detail_channels, kernel_size=1, bias=False)
        self.residual_scale = nn.Parameter(torch.zeros(()))
        nn.init.kaiming_uniform_(self.affinity_to_kernel, a=5**0.5)

    @staticmethod
    def _replicate_shift(feature: torch.Tensor, row_offset: int, column_offset: int) -> torch.Tensor:
        height, width = feature.shape[-2:]
        padded = F.pad(feature, (1, 1, 1, 1), mode="replicate")
        top, left = 1 + row_offset, 1 + column_offset
        return padded[..., top : top + height, left : left + width]

    def kernel_logits(self, detail: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized local queries and semantic-conditioned 3-by-3 logits."""
        batch, channels, height, width = detail.shape
        if channels != self.detail_channels:
            raise ValueError(f"SCLKR expected {self.detail_channels} detail channels, received {channels}.")
        if context.ndim != 4 or context.size(0) != batch or context.size(1) != self.context_channels:
            raise ValueError("SCLKR context must match the detail batch and configured context channels.")

        query = F.normalize(
            self.q_proj(detail).view(batch, self._NUM_HEADS, self.head_dim, height, width),
            p=2,
            dim=2,
            eps=1e-6,
        )
        keys = F.adaptive_avg_pool2d(self.k_proj(context), self._CONTEXT_GRID)
        keys = keys.view(batch, self._NUM_HEADS, self.head_dim, 16).permute(0, 1, 3, 2)
        keys = F.normalize(keys, p=2, dim=-1, eps=1e-6)
        affinity = torch.einsum("bhdxy,bhkd->bhxyk", query, keys)
        logits = torch.einsum("bhxyk,hkn->bhxyn", affinity, self.affinity_to_kernel)
        return query, logits + self.kernel_bias.view(1, self._NUM_HEADS, 1, 1, 9)

    def _local_score(
        self,
        query: torch.Tensor,
        kernel_logits: torch.Tensor,
        offset_index: int,
        row_offset: int,
        column_offset: int,
    ) -> torch.Tensor:
        batch, heads, head_dim, height, width = query.shape
        neighbor_query = self._replicate_shift(query.reshape(batch, heads * head_dim, height, width), row_offset, column_offset)
        neighbor_query = neighbor_query.view_as(query)
        similarity = (query * neighbor_query).sum(dim=2).clamp(-1.0, 1.0)
        compatibility = torch.exp((similarity - 1.0) / self._COMPATIBILITY_TEMPERATURE)
        return kernel_logits[..., offset_index] + torch.log(compatibility.clamp_min(1e-6))

    def forward(self, detail: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = detail.shape
        query, kernel_logits = self.kernel_logits(detail, context)

        max_score: torch.Tensor | None = None
        for index, (row_offset, column_offset) in enumerate(self._OFFSETS):
            score = self._local_score(query, kernel_logits, index, row_offset, column_offset)
            max_score = score if max_score is None else torch.maximum(max_score, score)
        assert max_score is not None

        normalizer = torch.zeros_like(max_score)
        for index, (row_offset, column_offset) in enumerate(self._OFFSETS):
            score = self._local_score(query, kernel_logits, index, row_offset, column_offset)
            normalizer = normalizer + torch.exp(score - max_score)

        channels_per_head = channels // self._NUM_HEADS
        local_mix = torch.zeros_like(detail)
        for index, (row_offset, column_offset) in enumerate(self._OFFSETS):
            score = self._local_score(query, kernel_logits, index, row_offset, column_offset)
            local_weight = (torch.exp(score - max_score) / normalizer).unsqueeze(2)
            neighbor_value = self._replicate_shift(detail, row_offset, column_offset)
            local_mix = local_mix + (
                local_weight * neighbor_value.view(batch, self._NUM_HEADS, channels_per_head, height, width)
            ).reshape_as(detail)

        delta = self.out_proj(local_mix - detail)
        return detail + torch.tanh(self.residual_scale) * delta


def retinex_directional_valley_maps(
    image: torch.Tensor,
    eps: float = 1e-6,
    blur_kernel_sizes: tuple[int, ...] = (5, 9, 17),
    directional_confidence_tau: float = 0.03,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Retinex dark-valley and directional-coherence priors.

    DE-MambaNet receives ImageNet-normalized RGB tensors. The conversion back
    to the physical [0, 1] RGB domain is deliberately local to this prior; it
    never alters the tensor passed through the regular backbone.
    """
    if image.ndim != 4 or image.size(1) != 3:
        raise ValueError("R-DVP expects ImageNet-normalized RGB input with shape [B, 3, H, W].")
    if directional_confidence_tau <= 0:
        raise ValueError("R-DVP directional_confidence_tau must be positive.")

    mean = image.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = image.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    rgb = (image * std + mean).clamp(0.0, 1.0)
    luminance = (rgb * image.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
    log_luminance = torch.log(luminance.clamp_min(eps))

    retinex_dark = [
        torch.relu(
            torch.nn.functional.avg_pool2d(
                log_luminance,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                count_include_pad=False,
            )
            - log_luminance
        )
        for kernel_size in blur_kernel_sizes
    ]
    dark_response = torch.stack(retinex_dark, dim=0).mean(dim=0)

    # Each kernel measures whether the center pixel is darker than the two
    # opposing neighbors along one of the four routed directions.
    directional_kernels = image.new_tensor(
        [
            [[0.0, 0.0, 0.0], [0.5, -1.0, 0.5], [0.0, 0.0, 0.0]],
            [[0.0, 0.5, 0.0], [0.0, -1.0, 0.0], [0.0, 0.5, 0.0]],
            [[0.5, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.5]],
            [[0.0, 0.0, 0.5], [0.0, -1.0, 0.0], [0.5, 0.0, 0.0]],
        ]
    ).unsqueeze(1)
    padded_log_luminance = torch.nn.functional.pad(log_luminance, (1, 1, 1, 1), mode="replicate")
    directional_response = torch.relu(torch.nn.functional.conv2d(padded_log_luminance, directional_kernels))
    directional_max = directional_response.max(dim=1, keepdim=True).values
    directional_sum = directional_response.sum(dim=1, keepdim=True)
    selectivity = torch.where(
        directional_sum > eps,
        directional_max / directional_sum.clamp_min(eps),
        torch.zeros_like(directional_max),
    )
    tau = float(directional_confidence_tau)
    directional_energy_gate = 1.0 - torch.exp(-directional_max / tau)
    dark_evidence_gate = 1.0 - torch.exp(-dark_response / tau)
    directional_coherence = selectivity * directional_energy_gate * dark_evidence_gate
    valley_strength = 0.5 * (dark_response + directional_max)
    return valley_strength, directional_coherence


class RetinexDirectionalValleyPrior(nn.Module):
    """Inject an initially neutral dark-valley prior into stem features."""

    def __init__(self, feature_channels: int):
        super().__init__()
        hidden_channels = max(8, feature_channels // 4)
        self.context = nn.Sequential(
            nn.Conv2d(feature_channels + 2, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
        )
        self.residual = nn.Conv2d(hidden_channels, feature_channels, kernel_size=1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def prior_maps(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return retinex_directional_valley_maps(image)

    def forward(self, image: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        if image.shape[0] != feature.shape[0] or image.shape[-2:] != feature.shape[-2:]:
            raise ValueError("R-DVP image and stem feature must have matching batch and spatial dimensions.")
        valley_strength, directional_coherence = self.prior_maps(image)
        residual = self.residual(self.context(torch.cat([feature, valley_strength, directional_coherence], dim=1)))
        return feature + residual


class LocalCrackBlock(nn.Module):
    """Cheap local directional stage used before Mamba is enabled."""

    def __init__(self, in_channels: int, out_channels: int, use_transformer: bool = False):
        super().__init__()
        self.project = ConvBNAct(in_channels, out_channels, kernel_size=1, padding=0) if in_channels != out_channels else nn.Identity()
        self.local = nn.Sequential(DWConvBlock(out_channels), StripConvBlock(out_channels))
        self.global_branch = TransformerGlobalBlock(out_channels) if use_transformer else nn.Identity()
        self.refine = DWConvBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project(x)
        return self.refine(x + self.local(x) + self.global_branch(x))


class NoDPSMTopoBlock(nn.Module):
    """Topology-matched Stage4 without a route/Mamba branch.

    This is deliberately *not* ``LocalCrackBlock``.  It retains the local and
    Transformer branches surrounding DPSM's routed Stage4, then fuses exactly
    those two branches.  ``from_topo_route`` is used by the paper ablation to
    retain same-seed initialization of every shared Stage4 component while
    unregistering the original route/Mamba parameters completely.
    """

    def __init__(
        self,
        channels: int,
        use_transformer: bool = True,
        *,
        local: nn.Module | None = None,
        global_branch: nn.Module | None = None,
        refine: nn.Module | None = None,
    ):
        super().__init__()
        self.channels = int(channels)
        self.local = local if local is not None else nn.Sequential(DWConvBlock(channels), StripConvBlock(channels))
        self.global_branch = (
            global_branch
            if global_branch is not None
            else (TransformerGlobalBlock(channels) if use_transformer else nn.Identity())
        )
        self.fuse = ConvBNAct(channels * 2, channels, kernel_size=1, padding=0)
        self.refine = refine if refine is not None else DWConvBlock(channels)

    @classmethod
    def from_topo_route(cls, original: "TopoRouteBlock") -> "NoDPSMTopoBlock":
        """Reuse initialized shared branches and discard all routed modules.

        The caller creates a normal ``TopoRouteBlock`` first, so the local,
        Transformer, and refine modules consume the identical random sequence
        to DPSM-on.  This constructor creates only the required 2-branch
        fusion after shared initialization is complete.
        """

        if not isinstance(original.global_branch, TransformerGlobalBlock):
            raise ValueError("NoDPSMTopoBlock requires the frozen Stage4 TransformerGlobalBlock.")
        if original.residual_branch_competition_fusion is not None:
            raise ValueError("NoDPSMTopoBlock cannot retain routed branch-competition fusion.")
        return cls(
            original.channels,
            use_transformer=True,
            local=original.local,
            global_branch=original.global_branch,
            refine=original.refine,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local(x)
        global_feature = self.global_branch(x)
        fused = self.fuse(torch.cat([local, global_feature], dim=1))
        return self.refine(x + fused)


class Stage4LocalOnly(nn.Module):
    """Strict T0: original local branch, one-input fusion, residual and refine."""

    def __init__(self, original: NoDPSMTopoBlock):
        super().__init__()
        self.channels = original.channels
        self.local = original.local
        self.fuse = ConvBNAct(self.channels, self.channels, kernel_size=1, padding=0)
        self.refine = original.refine

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(x + self.fuse(self.local(x)))


class Stage4LocalDPSM80(nn.Module):
    """Strict T2 reuses formal DPSM80 unchanged and physically omits Transformer."""

    def __init__(self, original: "TopoRouteBlock"):
        super().__init__()
        if not isinstance(original.route_mamba, DirectionPartitionedSharedMamba):
            raise ValueError("T2 requires the formal DirectionPartitionedSharedMamba.")
        self.channels = original.channels
        self.local = original.local
        self.route_mamba = original.route_mamba
        self.fuse = ConvBNAct(self.channels * 2, self.channels, kernel_size=1, padding=0)
        self.refine = original.refine

    @property
    def mamba_backends(self) -> list[str]:
        return self.route_mamba.backends

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        local = self.local(x)
        routed, route_logits = self.route_mamba(x)
        return self.refine(x + self.fuse(torch.cat([local, routed], dim=1))), route_logits


class PartialFullResolutionStage1(nn.Module):
    """DPER-PFRS: reuse the native full-resolution local Stage1 once.

    Only the spatial residual learned by ``LocalCrackBlock`` is exchanged with
    the half-resolution semantic path.  PixelUnshuffle preserves all four
    phases; the linear 64-to-40 projection intentionally has no activation so
    signed residual evidence is retained.
    """

    def __init__(self, detail_channels: int, semantic_channels: int):
        super().__init__()
        if detail_channels < 1 or semantic_channels < 1:
            raise ValueError("PFRS detail_channels and semantic_channels must be positive.")
        self.detail_channels = int(detail_channels)
        self.semantic_channels = int(semantic_channels)
        self.full_stage1 = LocalCrackBlock(
            self.detail_channels,
            self.detail_channels,
            use_transformer=False,
        )
        self.phase_unshuffle = nn.PixelUnshuffle(2)
        self.exchange_projection = nn.Sequential(
            nn.Conv2d(self.detail_channels * 4, self.semantic_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.semantic_channels),
        )

    def forward(
        self,
        full_detail: torch.Tensor,
        semantic_stage1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if full_detail.ndim != 4 or full_detail.size(1) != self.detail_channels:
            raise ValueError(
                f"PFRS expected full_detail [B, {self.detail_channels}, H, W], got {tuple(full_detail.shape)}."
            )
        if semantic_stage1.ndim != 4 or semantic_stage1.size(1) != self.semantic_channels:
            raise ValueError(
                "PFRS expected semantic_stage1 "
                f"[B, {self.semantic_channels}, H/2, W/2], got {tuple(semantic_stage1.shape)}."
            )
        if full_detail.shape[0] != semantic_stage1.shape[0] or (
            full_detail.shape[-2] != semantic_stage1.shape[-2] * 2
            or full_detail.shape[-1] != semantic_stage1.shape[-1] * 2
        ):
            raise ValueError("PFRS requires full_detail to be exactly twice the semantic Stage1 resolution.")
        detail_stage1 = self.full_stage1(full_detail)
        detail_delta = detail_stage1 - full_detail
        exchange = self.exchange_projection(self.phase_unshuffle(detail_delta))
        return detail_stage1, semantic_stage1 + exchange


class LocalRouteConsistencyGate(nn.Module):
    """Calibrate only the routed residual using local structural evidence."""

    def __init__(self, channels: int):
        super().__init__()
        hidden_channels = max(8, channels // 8)
        self.local_proj = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False)
        self.route_proj = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(1, hidden_channels)
        self.act = nn.GELU()
        self.gate_out = nn.Conv2d(hidden_channels, 1, kernel_size=1)

        # Preserve the original routed feature before this gate is trained.
        nn.init.zeros_(self.gate_out.weight)
        nn.init.zeros_(self.gate_out.bias)

    def gate_scale(self, x: torch.Tensor, local: torch.Tensor, routed: torch.Tensor) -> torch.Tensor:
        local_delta = local - x
        route_delta = routed - x
        gate_feature = self.act(self.norm(self.local_proj(local_delta) + self.route_proj(route_delta)))
        return 1.0 + 0.5 * torch.tanh(self.gate_out(gate_feature))

    def forward(self, x: torch.Tensor, local: torch.Tensor, routed: torch.Tensor) -> torch.Tensor:
        route_delta = routed - x
        return x + self.gate_scale(x, local, routed) * route_delta


class ResidualBranchCompetitionFusion(nn.Module):
    """Independently calibrate existing branch residuals at each pixel."""

    def __init__(self, channels: int):
        super().__init__()
        hidden_channels = max(8, channels // 8)
        self.local_proj = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False)
        self.mamba_proj = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False)
        self.transformer_proj = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(hidden_channels)
        self.act = nn.GELU()
        self.gate_out = nn.Conv2d(hidden_channels, 3, kernel_size=1)
        nn.init.zeros_(self.gate_out.weight)
        nn.init.zeros_(self.gate_out.bias)

    def _channel_layer_norm(self, x: torch.Tensor) -> torch.Tensor:
        # LayerNorm operates on channels only. Restore a standard contiguous
        # NCHW layout before the 1x1 gate so fused AdamW sees matching weight
        # and gradient layouts for the zero-initialized gate parameters.
        normalized = self.norm(x.permute(0, 2, 3, 1).contiguous())
        return normalized.permute(0, 3, 1, 2).contiguous()

    def gate_delta(
        self,
        x: torch.Tensor,
        local: torch.Tensor,
        routed: torch.Tensor,
        global_feature: torch.Tensor,
    ) -> torch.Tensor:
        local_delta = local - x
        mamba_delta = routed - x
        transformer_delta = global_feature - x
        context = (
            self.local_proj(local_delta)
            + self.mamba_proj(mamba_delta)
            + self.transformer_proj(transformer_delta)
        )
        return 0.5 * torch.tanh(self.gate_out(self.act(self._channel_layer_norm(context))))

    def forward(
        self,
        x: torch.Tensor,
        local: torch.Tensor,
        routed: torch.Tensor,
        global_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate_delta = self.gate_delta(x, local, routed, global_feature)
        local_delta = local - x
        mamba_delta = routed - x
        transformer_delta = global_feature - x
        return (
            local + gate_delta[:, 0:1] * local_delta,
            routed + gate_delta[:, 1:2] * mamba_delta,
            global_feature + gate_delta[:, 2:3] * transformer_delta,
        )


class TopoRouteBlock(nn.Module):
    """Local features plus tangent-routed four-path Mamba fusion."""

    valid_hybrid_orders = ("parallel", "dpsm_then_transformer", "transformer_then_dpsm")

    def __init__(
        self,
        channels: int,
        route_bins: int = 4,
        use_transformer: bool = False,
        use_mamba_ssm: bool = True,
        mamba_window_size: int | None = None,
        mamba_bidirectional: bool = False,
        active_mamba_directions: tuple[str, ...] | list[str] | None = None,
        mamba_spatial_reduction: int = 1,
        mamba_channels: int | None = None,
    ):
        super().__init__()
        self.channels = channels
        self.local = nn.Sequential(DWConvBlock(channels), StripConvBlock(channels))
        self.route_mamba = TangentRoutedMambaBlock(
            channels,
            route_bins,
            use_mamba_ssm,
            window_size=mamba_window_size,
            bidirectional=mamba_bidirectional,
            active_mamba_directions=active_mamba_directions,
            spatial_reduction=mamba_spatial_reduction,
            mamba_channels=mamba_channels,
        )
        self.global_branch = TransformerGlobalBlock(channels) if use_transformer else nn.Identity()
        self.fuse = ConvBNAct(channels * 3, channels, kernel_size=1, padding=0)
        self.refine = DWConvBlock(channels)
        self.residual_branch_competition_fusion: ResidualBranchCompetitionFusion | None = None
        # Plain metadata only: this must not alter module registration, RNG
        # consumption, checkpoints, or the frozen parallel default.
        self.hybrid_order = "parallel"

    def set_hybrid_order(self, order: str) -> None:
        if order not in self.valid_hybrid_orders:
            raise ValueError(
                f"Invalid Stage4 hybrid order {order!r}; expected one of {self.valid_hybrid_orders}."
            )
        self.hybrid_order = order

    def enable_direction_partitioned_shared_mamba(
        self,
        group_channels: int,
        use_mamba_ssm: bool,
        scan_mode: str = "four_direction",
        route_mode: str = "learned",
    ) -> None:
        """Replace only route_mamba, retaining route supervision and outer Stage4 branches."""

        if not isinstance(self.route_mamba, TangentRoutedMambaBlock):
            raise RuntimeError("DPSM requires the original TangentRoutedMambaBlock before replacement.")
        original = self.route_mamba
        if original.spatial_reduction != 1 or original.mamba_channels != self.channels:
            raise ValueError("DPSM requires full-resolution, full-width original Stage4 routing.")
        if tuple(original.active_mamba_directions) != DirectionPartitionedSharedMamba.directions:
            raise ValueError("DPSM requires all four original Stage4 directions.")
        if original.local_compensations is not None:
            raise ValueError("DPSM cannot inherit LocalDirectionalCompensation paths.")
        self.route_mamba = DirectionPartitionedSharedMamba(
            self.channels,
            route_head=original.route_head,
            group_channels=group_channels,
            use_mamba_ssm=use_mamba_ssm,
            scan_mode=scan_mode,
            route_mode=route_mode,
        )

    def enable_single_raster_mamba80(self, use_mamba_ssm: bool) -> None:
        """Replace only Stage4 routing with the strict single-raster L1 path."""
        if not isinstance(self.route_mamba, TangentRoutedMambaBlock):
            raise RuntimeError("SingleRasterMamba80 requires the original routed Stage4 scaffold.")
        original = self.route_mamba
        if original.spatial_reduction != 1 or original.mamba_channels != self.channels:
            raise ValueError("SingleRasterMamba80 requires the frozen full-width Stage4 scaffold.")
        if tuple(original.active_mamba_directions) != DirectionPartitionedSharedMamba.directions:
            raise ValueError("SingleRasterMamba80 requires all four scaffold directions before replacement.")
        self.route_mamba = SingleRasterMamba80(use_mamba_ssm)

    def enable_four_independent_mamba80(self, use_mamba_ssm: bool) -> None:
        """Reuse the frozen route head while replacing all four cores with 80ch paths."""
        if not isinstance(self.route_mamba, TangentRoutedMambaBlock):
            raise RuntimeError("FourIndependentMamba80 requires the original routed Stage4 scaffold.")
        original = self.route_mamba
        if original.spatial_reduction != 1 or original.mamba_channels != self.channels:
            raise ValueError("FourIndependentMamba80 requires the frozen full-width Stage4 scaffold.")
        if tuple(original.active_mamba_directions) != DirectionPartitionedSharedMamba.directions:
            raise ValueError("FourIndependentMamba80 requires all four scaffold directions before replacement.")
        self.route_mamba = TangentRoutedMambaBlock(
            self.channels,
            route_bins=4,
            use_mamba_ssm=use_mamba_ssm,
            gate_inputs=original.gate_inputs,
            spatial_reduction=1,
            mamba_channels=80,
            route_head=original.route_head,
        )

    def enable_common_private_direction_partitioned_shared_mamba(
        self,
        shared_channels: int,
        private_channels: int,
        use_mamba_ssm: bool,
    ) -> None:
        """Replace the original four full-width paths with CP-DPSM64 only."""

        if not isinstance(self.route_mamba, TangentRoutedMambaBlock):
            raise RuntimeError("CP-DPSM requires the original TangentRoutedMambaBlock before replacement.")
        original = self.route_mamba
        if original.spatial_reduction != 1 or original.mamba_channels != self.channels:
            raise ValueError("CP-DPSM requires full-resolution, full-width original Stage4 routing.")
        if tuple(original.active_mamba_directions) != CommonPrivateDirectionPartitionedSharedMamba.directions:
            raise ValueError("CP-DPSM requires all four original Stage4 directions.")
        if original.local_compensations is not None:
            raise ValueError("CP-DPSM cannot inherit LocalDirectionalCompensation paths.")
        self.route_mamba = CommonPrivateDirectionPartitionedSharedMamba(
            self.channels,
            route_head=original.route_head,
            shared_channels=shared_channels,
            private_channels=private_channels,
            use_mamba_ssm=use_mamba_ssm,
        )

    def enable_residual_branch_competition_fusion(self) -> None:
        if self.residual_branch_competition_fusion is None:
            self.residual_branch_competition_fusion = ResidualBranchCompetitionFusion(self.channels)

    @property
    def mamba_backends(self) -> list[str]:
        return self.route_mamba.backends

    def forward(
        self,
        x: torch.Tensor,
        local_route_gate: LocalRouteConsistencyGate | None = None,
        identity_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0.0 < identity_scale <= 1.0:
            raise ValueError(f"identity_scale must be in (0, 1], got {identity_scale}.")
        local = self.local(x)
        # Keep the frozen parallel execution order byte-for-byte equivalent:
        # local -> DPSM(x) -> Transformer(x) -> fusion -> residual -> refine.
        if self.hybrid_order == "parallel":
            routed, route_logits = self.route_mamba(x)
            global_feature = self.global_branch(x)
        elif self.hybrid_order == "dpsm_then_transformer":
            routed, route_logits = self.route_mamba(x)
            global_feature = self.global_branch(routed)
        elif self.hybrid_order == "transformer_then_dpsm":
            global_feature = self.global_branch(x)
            routed, route_logits = self.route_mamba(global_feature)
        else:  # Defensive: setters/constructors validate, but never silently fall back.
            raise RuntimeError(f"Unsupported Stage4 hybrid order at forward: {self.hybrid_order!r}.")
        if local_route_gate is not None:
            routed = local_route_gate(x, local, routed)
        if self.residual_branch_competition_fusion is not None:
            local, routed, global_feature = self.residual_branch_competition_fusion(
                x,
                local,
                routed,
                global_feature,
            )
        fused = self.fuse(torch.cat([local, routed, global_feature], dim=1))
        pre_refine = x + fused if identity_scale == 1.0 else identity_scale * x + fused
        return self.refine(pre_refine), route_logits
