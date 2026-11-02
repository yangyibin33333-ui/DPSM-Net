from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
def _to_gray(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(1)
    return x.float() if x.size(1) == 1 else x.float().mean(dim=1, keepdim=True)

def sobel_edge(x: torch.Tensor, normalize: bool = True, eps: float = 1e-6) -> torch.Tensor:
    """Lightweight image edge prior used only by the CNN stem."""
    gray = _to_gray(x)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    dx, dy = F.conv2d(gray, kx, padding=1), F.conv2d(gray, ky, padding=1)
    edge = torch.sqrt(dx.square() + dy.square() + eps)
    return edge / edge.amax(dim=(2, 3), keepdim=True).clamp_min(eps) if normalize else edge
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
