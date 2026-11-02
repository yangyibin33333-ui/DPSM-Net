from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except Exception:  # pragma: no cover - optional dependency
    Mamba = None


class _FallbackSequenceBlock(nn.Module):
    def __init__(self, dim: int, expansion: int = 2, kernel_size: int = 7):
        super().__init__()
        hidden = dim * expansion
        self.in_proj = nn.Linear(dim, hidden * 2)
        self.dwconv = nn.Conv1d(hidden, hidden, kernel_size, padding=kernel_size // 2, groups=hidden)
        self.out_proj = nn.Linear(hidden, dim)
        self.act = nn.SiLU()

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        value, gate = self.in_proj(sequence).chunk(2, dim=-1)
        value = self.dwconv(value.transpose(1, 2)).transpose(1, 2)
        return self.out_proj(self.act(value) * torch.sigmoid(gate))


class SequenceMambaCore(nn.Module):
    """A sequence backend shared by one directional scan path."""

    def __init__(
        self,
        dim: int,
        use_mamba_ssm: bool = True,
        *,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        fallback_kernel_size: int | None = None,
        force_fp32_under_autocast: bool = False,
    ):
        super().__init__()
        if d_state < 1 or d_conv < 1 or expand < 1:
            raise ValueError("SequenceMambaCore d_state, d_conv, and expand must be positive.")
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.force_fp32_under_autocast = bool(force_fp32_under_autocast)
        self.norm = nn.LayerNorm(dim)
        self.uses_mamba_ssm = bool(use_mamba_ssm and Mamba is not None)
        self.backend = "mamba_ssm" if self.uses_mamba_ssm else "fallback"
        # Keep the historical fallback's 7-wide temporal convolution unless
        # a new caller explicitly asks to mirror a different Mamba d_conv.
        resolved_fallback_kernel = 7 if fallback_kernel_size is None else fallback_kernel_size
        if resolved_fallback_kernel < 1:
            raise ValueError("SequenceMambaCore fallback_kernel_size must be positive when provided.")
        self.core = (
            Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
            if self.uses_mamba_ssm
            else _FallbackSequenceBlock(dim, expansion=expand, kernel_size=resolved_fallback_kernel)
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        # The narrow ESCS Mamba is shared by four batched long sequences.  Some
        # mamba_ssm/CUDA combinations return reduced-precision parameter
        # gradients for this particular AMP path, which is incompatible with
        # AdamW's FP32 fused state tensors.  Keep that optional path's internal
        # Mamba computation in FP32, then restore the surrounding AMP dtype.
        # Existing routed Mamba callers retain their historical AMP behavior.
        if self.force_fp32_under_autocast and sequence.is_cuda and torch.is_autocast_enabled():
            input_dtype = sequence.dtype
            with torch.autocast(device_type="cuda", enabled=False):
                output = self.out_norm(self.core(self.norm(sequence.float())))
            return output.to(dtype=input_dtype)
        return self.out_norm(self.core(self.norm(sequence)))


class _SerpentineScanReorder(nn.Module):
    def forward(self, flat: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        return flat[:, order, :]


class _SerpentineInverseReorder(nn.Module):
    def forward(self, scanned: torch.Tensor, inverse: torch.Tensor) -> torch.Tensor:
        index = inverse.view(1, inverse.size(0), inverse.size(1), 1).expand_as(scanned)
        return scanned.gather(2, index)


class _ArithmeticPathFusion(nn.Module):
    def forward(self, restored: torch.Tensor) -> torch.Tensor:
        return restored.mean(dim=1)


class EarlySerpentineConnectivitySSM(nn.Module):
    """One shared narrow Mamba over four cached connectivity-preserving scans.

    Scan-order motivation: SASS in Liu et al., *SCSegamba* (CVPR 2025),
    Apache-2.0 repository https://github.com/Karl1109/SCSegamba.  This is an
    independent tensor-index implementation; no source code or model modules
    from that repository are copied.  In particular, it deliberately excludes
    GBC, gating, routing, and their decoder.
    """

    path_count = 4
    traversal_names = (
        "horizontal_serpentine_forward",
        "horizontal_serpentine_reverse",
        "diagonal_serpentine_forward",
        "diagonal_serpentine_reverse",
    )

    def __init__(
        self,
        input_channels: int,
        bottleneck_channels: int = 16,
        use_mamba_ssm: bool = True,
    ):
        super().__init__()
        if input_channels < 1 or bottleneck_channels < 1:
            raise ValueError("ESCS input_channels and bottleneck_channels must be positive.")
        self.input_channels = input_channels
        self.bottleneck_channels = bottleneck_channels
        self.d_state = 8
        self.d_conv = 3
        self.expand = 1
        self.input_projection = nn.Sequential(
            nn.Conv2d(input_channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.GELU(),
        )
        self.shared_mamba = SequenceMambaCore(
            bottleneck_channels,
            use_mamba_ssm,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            fallback_kernel_size=self.d_conv,
            force_fp32_under_autocast=True,
        )
        self.output_projection = nn.Sequential(
            nn.Conv2d(bottleneck_channels, input_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(input_channels),
        )
        self.scan_reorder = _SerpentineScanReorder()
        self.inverse_reorder = _SerpentineInverseReorder()
        self.path_fusion = _ArithmeticPathFusion()
        # Cache values are device-local Long tensors, intentionally excluded
        # from state_dict and created only once per (H, W, device) key.
        self._scan_index_cache: dict[tuple[int, int, str, int | None], dict[str, torch.Tensor]] = {}
        self.scan_cache_hits = 0
        self.scan_cache_misses = 0
        self.last_shared_mamba_call_count = 0
        self.last_batched_sequence_shape: tuple[int, int, int] | None = None
        self.last_path_sequence_shape: tuple[int, int, int] | None = None

    @property
    def backend(self) -> str:
        return self.shared_mamba.backend

    @property
    def instantiated_core_count(self) -> int:
        return 1

    @property
    def traversal_count(self) -> int:
        return self.path_count

    @staticmethod
    def _horizontal_order(height: int, width: int) -> list[int]:
        return [
            row * width + column
            for row in range(height)
            for column in (range(width) if row % 2 == 0 else range(width - 1, -1, -1))
        ]

    @staticmethod
    def _diagonal_order(height: int, width: int) -> list[int]:
        """Return an 8-neighbour-continuous anti-diagonal serpentine path."""

        order: list[int] = []
        for diagonal_sum in range(height + width - 1):
            row_min = max(0, diagonal_sum - (width - 1))
            row_max = min(height - 1, diagonal_sum)
            # Alternating the in-band direction makes the transition to the
            # next anti-diagonal band an 8-neighbour step, including non-square
            # grids and the tapered bands at both ends.
            rows = range(row_max, row_min - 1, -1) if diagonal_sum % 2 == 0 else range(row_min, row_max + 1)
            for row in rows:
                column = diagonal_sum - row
                order.append(row * width + column)
        return order

    @staticmethod
    def _cache_key(height: int, width: int, device: torch.device) -> tuple[int, int, str, int | None]:
        return height, width, device.type, device.index

    def scan_indices(self, height: int, width: int, device: torch.device) -> dict[str, torch.Tensor]:
        """Return cached path and inverse path indices for the requested grid."""

        if height < 1 or width < 1:
            raise ValueError(f"ESCS spatial dimensions must be positive, got {(height, width)}.")
        key = self._cache_key(height, width, device)
        cached = self._scan_index_cache.get(key)
        if cached is not None:
            # A cache entry may have been made by an older process/version while
            # ``torch.inference_mode`` was active.  Such indices are illegal in
            # later autograd-tracked advanced indexing.  Normalise the entry
            # once on cache hit; normal cache entries take this branch with no
            # allocation or runtime effect.
            is_inference = getattr(cached["order"], "is_inference", lambda: False)
            if is_inference():
                with torch.inference_mode(False):
                    cached = {name: tensor.clone() for name, tensor in cached.items()}
                self._scan_index_cache[key] = cached
            self.scan_cache_hits += 1
            return cached

        # ``forward_mask_only`` is normally called inside ``torch.inference_mode``.
        # Index tensors created in that mode cannot later participate in an
        # autograd-tracked advanced-indexing operation, yet this cache is shared
        # by evaluation and training forwards.  Construct cache entries with
        # inference mode explicitly disabled so one cached order is valid for
        # both paths.  These are non-trainable Long indices and remain excluded
        # from the state dict.
        with torch.inference_mode(False):
            horizontal = torch.tensor(self._horizontal_order(height, width), device=device, dtype=torch.long)
            diagonal = torch.tensor(self._diagonal_order(height, width), device=device, dtype=torch.long)
            orders = torch.stack((horizontal, horizontal.flip(0), diagonal, diagonal.flip(0)), dim=0)
            length = height * width
            inverse = torch.empty_like(orders)
            inverse.scatter_(
                1,
                orders,
                torch.arange(length, device=device, dtype=torch.long).expand(self.path_count, -1),
            )
        cached = {"order": orders, "inverse": inverse}
        self._scan_index_cache[key] = cached
        self.scan_cache_misses += 1
        return cached

    def analytical_mamba_estimate_flops(self, batch_size: int, height: int, width: int) -> int:
        """Transparent lower-order estimate for the early Mamba, excluded by fvcore.

        It counts the Mamba input/output projections, depthwise temporal
        convolution, state projections, and two state updates; it intentionally
        does not claim to include fused-kernel implementation overhead.
        """

        inner = self.bottleneck_channels * self.expand
        dt_rank = max(1, (self.bottleneck_channels + 15) // 16)
        per_token = (
            self.bottleneck_channels * (2 * inner)
            + inner * self.d_conv
            + inner * (dt_rank + 2 * self.d_state)
            + dt_rank * inner
            + 2 * inner * self.d_state
            + inner * self.bottleneck_channels
        )
        return int(batch_size * self.path_count * height * width * per_token)

    def forward(self, stage1_half: torch.Tensor) -> torch.Tensor:
        if stage1_half.ndim != 4 or stage1_half.size(1) != self.input_channels:
            raise ValueError(
                f"ESCS expected [B, {self.input_channels}, H, W], received {tuple(stage1_half.shape)}."
            )
        batch, _, height, width = stage1_half.shape
        bottleneck = self.input_projection(stage1_half)
        length = height * width
        cache = self.scan_indices(height, width, bottleneck.device)
        order = cache["order"]
        inverse = cache["inverse"]
        flat = bottleneck.permute(0, 2, 3, 1).reshape(batch, length, self.bottleneck_channels)
        paths = self.scan_reorder(flat, order)
        self.last_path_sequence_shape = (batch, length, self.bottleneck_channels)
        batched_paths = paths.reshape(batch * self.path_count, length, self.bottleneck_channels)
        self.last_batched_sequence_shape = tuple(batched_paths.shape)
        self.last_shared_mamba_call_count = 0
        scanned = self.shared_mamba(batched_paths)
        self.last_shared_mamba_call_count += 1
        scanned = scanned.reshape(batch, self.path_count, length, self.bottleneck_channels)
        restored = self.inverse_reorder(scanned, inverse)
        fused = self.path_fusion(restored)
        fused_map = fused.transpose(1, 2).reshape(batch, self.bottleneck_channels, height, width)
        return stage1_half + self.output_projection(fused_map)


class DirectionalMambaPath(nn.Module):
    """Apply one Mamba core to rows, columns, or non-wrapping diagonals."""

    def __init__(self, dim: int, direction: str, use_mamba_ssm: bool = True):
        super().__init__()
        self.direction = direction
        self.core = SequenceMambaCore(dim, use_mamba_ssm)

    @property
    def backend(self) -> str:
        return self.core.backend

    def _diagonal_sequences(self, x: torch.Tensor, anti: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        device = x.device
        rows = torch.arange(h, device=device).view(h, 1).expand(h, w)
        cols = torch.arange(w, device=device).view(1, w).expand(h, w)
        diagonal = rows + cols if anti else rows - cols + (w - 1)
        start_row = (diagonal - (w - 1)).clamp_min(0)
        position = rows - start_row
        count = h + w - 1
        max_length = min(h, w)
        pixels = x.permute(0, 2, 3, 1)
        packed = x.new_zeros((b, count, max_length, c))
        packed[:, diagonal.reshape(-1), position.reshape(-1)] = pixels.reshape(b, -1, c)
        return packed, diagonal, position

    def _restore_diagonals(
        self,
        sequences: torch.Tensor,
        diagonal: torch.Tensor,
        position: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        b, _, _, c = sequences.shape
        pixels = sequences[:, diagonal.reshape(-1), position.reshape(-1)]
        return pixels.reshape(b, height, width, c).permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if self.direction == "horizontal":
            sequence = x.permute(0, 2, 3, 1).reshape(b * h, w, c)
            output = self.core(sequence).reshape(b, h, w, c).permute(0, 3, 1, 2)
        elif self.direction == "vertical":
            sequence = x.permute(0, 3, 2, 1).reshape(b * w, h, c)
            output = self.core(sequence).reshape(b, w, h, c).permute(0, 3, 2, 1)
        else:
            packed, diagonal, position = self._diagonal_sequences(x, anti=self.direction == "anti_diagonal")
            output = self.core(packed.flatten(0, 1)).reshape_as(packed)
            output = self._restore_diagonals(output, diagonal, position, h, w)
        return output.contiguous()


class BidirectionalDirectionalMambaPath(nn.Module):
    """Reuse one directional Mamba path for forward and reversed scans."""

    def __init__(self, dim: int, direction: str, use_mamba_ssm: bool = True):
        super().__init__()
        self.direction = direction
        self.path = DirectionalMambaPath(dim, direction, use_mamba_ssm)

    @property
    def backend(self) -> str:
        return self.path.backend

    def _reverse_dims(self) -> tuple[int, ...]:
        if self.direction == "horizontal":
            return (-1,)
        if self.direction == "vertical":
            return (-2,)
        if self.direction in ("diagonal", "anti_diagonal"):
            return (-2, -1)
        raise ValueError(f"Unsupported directional Mamba path: {self.direction!r}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reverse_dims = self._reverse_dims()
        forward_output = self.path(x)
        reversed_input = torch.flip(x, dims=reverse_dims)
        backward_output_reversed = self.path(reversed_input)
        backward_output = torch.flip(backward_output_reversed, dims=reverse_dims)
        return 0.5 * (forward_output + backward_output)


class WindowedDirectionalMambaPath(nn.Module):
    """Run one existing directional path independently inside fixed windows."""

    def __init__(self, dim: int, direction: str, window_size: int, use_mamba_ssm: bool = True):
        super().__init__()
        if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size <= 0:
            raise ValueError(f"window_size must be a positive integer, got {window_size!r}.")
        self.window_size = window_size
        self.path = DirectionalMambaPath(dim, direction, use_mamba_ssm)

    @property
    def backend(self) -> str:
        return self.path.backend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        if height % self.window_size != 0 or width % self.window_size != 0:
            raise ValueError(
                f"WindowedDirectionalMambaPath requires H/W divisible by {self.window_size}, got {(height, width)}."
            )
        num_h = height // self.window_size
        num_w = width // self.window_size
        windows = (
            x.contiguous()
            .view(batch, channels, num_h, self.window_size, num_w, self.window_size)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch * num_h * num_w, channels, self.window_size, self.window_size)
        )
        window_outputs = self.path(windows)
        return (
            window_outputs.view(batch, num_h, num_w, channels, self.window_size, self.window_size)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(batch, channels, height, width)
            .contiguous()
        )


class LocalDirectionalCompensation(nn.Module):
    """Cheap local replacement for a disabled directional Mamba scan."""

    def __init__(self, dim: int, direction: str):
        super().__init__()
        if direction not in ("horizontal", "vertical", "diagonal", "anti_diagonal"):
            raise ValueError(f"Unsupported local compensation direction: {direction!r}.")
        self.direction = direction
        dilation = 1 if direction in ("horizontal", "vertical", "diagonal") else 2
        self.branch = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch(x)


class TangentRoutedMambaBlock(nn.Module):
    """Four route bins with configurable Mamba scans and local replacements."""

    def __init__(
        self,
        dim: int,
        route_bins: int = 4,
        use_mamba_ssm: bool = True,
        gate_inputs: bool = True,
        window_size: int | None = None,
        bidirectional: bool = False,
        active_mamba_directions: tuple[str, ...] | list[str] | None = None,
        spatial_reduction: int = 1,
        mamba_channels: int | None = None,
        route_head: nn.Module | None = None,
    ):
        super().__init__()
        if route_bins != 4:
            raise ValueError("Lite-TopoRoute-Mamba uses four directional route bins.")
        if window_size is not None and (
            not isinstance(window_size, int) or isinstance(window_size, bool) or window_size <= 0
        ):
            raise ValueError(f"window_size must be a positive integer or None, got {window_size!r}.")
        if window_size is not None and bidirectional:
            raise ValueError("Windowed and bidirectional Mamba paths cannot be enabled together.")
        if (
            not isinstance(spatial_reduction, int)
            or isinstance(spatial_reduction, bool)
            or spatial_reduction < 1
        ):
            raise ValueError(f"spatial_reduction must be a positive integer, got {spatial_reduction!r}.")
        if mamba_channels is not None and (
            not isinstance(mamba_channels, int)
            or isinstance(mamba_channels, bool)
            or mamba_channels < 1
            or mamba_channels > dim
        ):
            raise ValueError(f"mamba_channels must be an integer in [1, {dim}], got {mamba_channels!r}.")
        resolved_mamba_channels = dim if mamba_channels is None else mamba_channels
        if resolved_mamba_channels < dim and spatial_reduction > 1:
            raise ValueError(
                "Mamba channel bottleneck and spatial reduction are independent ablations and cannot be combined."
            )
        directions = ("horizontal", "vertical", "diagonal", "anti_diagonal")
        if active_mamba_directions is None:
            active_directions = directions
        else:
            if isinstance(active_mamba_directions, str):
                raise ValueError("active_mamba_directions must be a sequence of direction names, not a string.")
            requested = tuple(active_mamba_directions)
            if not requested or len(set(requested)) != len(requested) or any(direction not in directions for direction in requested):
                raise ValueError(
                    "active_mamba_directions must be a non-empty, duplicate-free subset of "
                    "horizontal, vertical, diagonal, anti_diagonal."
                )
            active_directions = tuple(direction for direction in directions if direction in requested)
        # Reusing the scaffold route head permits a strict route-branch-only
        # replacement without consuming another random-initialization stream.
        # The default remains byte-for-byte equivalent to historical blocks.
        if route_head is None:
            hidden = max(dim // 4, 8)
            route_head = nn.Sequential(
                nn.Conv2d(dim, hidden, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.GELU(),
                nn.Conv2d(hidden, route_bins, kernel_size=1),
            )
        self.route_head = route_head
        self.gate_inputs = gate_inputs
        self.spatial_reduction = spatial_reduction
        self.mamba_channels = resolved_mamba_channels
        self.input_projection: nn.Module
        if self.mamba_channels == dim:
            self.input_projection = nn.Identity()
        else:
            self.input_projection = nn.Sequential(
                nn.Conv2d(dim, self.mamba_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.mamba_channels),
                nn.GELU(),
            )
        self.active_mamba_directions = active_directions
        self._uses_local_compensation = active_mamba_directions is not None
        if window_size is not None:
            self.paths = nn.ModuleList(
                [
                    WindowedDirectionalMambaPath(
                        self.mamba_channels, direction, window_size, use_mamba_ssm
                    )
                    for direction in active_directions
                ]
            )
        elif bidirectional:
            self.paths = nn.ModuleList(
                [BidirectionalDirectionalMambaPath(self.mamba_channels, direction, use_mamba_ssm) for direction in active_directions]
            )
        else:
            self.paths = nn.ModuleList(
                [DirectionalMambaPath(self.mamba_channels, direction, use_mamba_ssm) for direction in active_directions]
            )
        self.local_compensations: nn.ModuleDict | None = None
        if self._uses_local_compensation:
            self.local_compensations = nn.ModuleDict(
                {
                    direction: LocalDirectionalCompensation(self.mamba_channels, direction)
                    for direction in directions
                    if direction not in active_directions
                }
            )
        self.last_executed_local_compensation_directions: tuple[str, ...] = ()
        self.last_executed_mamba_directions: tuple[str, ...] = ()
        self.out_proj = nn.Sequential(
            nn.Conv2d(self.mamba_channels, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

    @property
    def backends(self) -> list[str]:
        return [path.backend for path in self.paths]

    @property
    def active_mamba_path_count(self) -> int:
        return len(self.paths)

    @property
    def instantiated_core_count(self) -> int:
        return len(self.paths)

    def _pooled_route_weights(self, route_weights: torch.Tensor) -> torch.Tensor:
        """Average and renormalize directional weights for reduced Mamba scans."""

        if self.spatial_reduction == 1:
            return route_weights
        height, width = route_weights.shape[-2:]
        if height < self.spatial_reduction or width < self.spatial_reduction:
            raise ValueError(
                "TangentRoutedMambaBlock spatial reduction requires both spatial dimensions to be at least "
                f"{self.spatial_reduction}, got {(height, width)}."
            )
        route_weights_low = F.avg_pool2d(
            route_weights,
            kernel_size=self.spatial_reduction,
            stride=self.spatial_reduction,
        )
        return route_weights_low / (route_weights_low.sum(dim=1, keepdim=True) + 1e-8)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.last_executed_mamba_directions = self.active_mamba_directions
        if self.spatial_reduction > 1 and (
            x.shape[-2] < self.spatial_reduction or x.shape[-1] < self.spatial_reduction
        ):
            raise ValueError(
                "TangentRoutedMambaBlock spatial reduction requires both spatial dimensions to be at least "
                f"{self.spatial_reduction}, got {tuple(x.shape[-2:])}."
            )
        route_logits = self.route_head(x)
        route_weights = torch.softmax(route_logits, dim=1)
        if self.spatial_reduction == 1:
            if self.mamba_channels != x.size(1):
                compact = self.input_projection(x)
                if not self._uses_local_compensation:
                    routed_paths = []
                    for index, path in enumerate(self.paths):
                        weight = route_weights[:, index : index + 1]
                        path_input = compact * (1.0 + weight) if self.gate_inputs else compact
                        routed_paths.append(path(path_input) * weight)
                    fused = torch.stack(routed_paths, dim=1).sum(dim=1)
                    self.last_executed_local_compensation_directions = ()
                    return x + self.out_proj(fused), route_logits

                assert self.local_compensations is not None
                directions = ("horizontal", "vertical", "diagonal", "anti_diagonal")
                path_by_direction = dict(zip(self.active_mamba_directions, self.paths, strict=True))
                routed_paths = []
                executed_compensations: list[str] = []
                for index, direction in enumerate(directions):
                    weight = route_weights[:, index : index + 1]
                    if direction in path_by_direction:
                        path_input = compact * (1.0 + weight) if self.gate_inputs else compact
                        directional_feature = path_by_direction[direction](path_input)
                    else:
                        directional_feature = self.local_compensations[direction](compact)
                        executed_compensations.append(direction)
                    routed_paths.append(directional_feature * weight)
                fused = torch.stack(routed_paths, dim=1).sum(dim=1)
                self.last_executed_local_compensation_directions = tuple(executed_compensations)
                return x + self.out_proj(fused), route_logits

            if not self._uses_local_compensation:
                # Preserve the legacy four-Mamba execution and arithmetic exactly.
                routed_paths = []
                for index, path in enumerate(self.paths):
                    weight = route_weights[:, index : index + 1]
                    path_input = x * (1.0 + weight) if self.gate_inputs else x
                    routed_paths.append(path(path_input) * weight)
                fused = torch.stack(routed_paths, dim=1).sum(dim=1)
                self.last_executed_local_compensation_directions = ()
                return x + self.out_proj(fused), route_logits

            assert self.local_compensations is not None
            directions = ("horizontal", "vertical", "diagonal", "anti_diagonal")
            path_by_direction = dict(zip(self.active_mamba_directions, self.paths, strict=True))
            routed_paths = []
            executed_compensations: list[str] = []
            for index, direction in enumerate(directions):
                weight = route_weights[:, index : index + 1]
                if direction in path_by_direction:
                    path_input = x * (1.0 + weight) if self.gate_inputs else x
                    directional_feature = path_by_direction[direction](path_input)
                else:
                    directional_feature = self.local_compensations[direction](x)
                    executed_compensations.append(direction)
                routed_paths.append(directional_feature * weight)
            fused = torch.stack(routed_paths, dim=1).sum(dim=1)
            self.last_executed_local_compensation_directions = tuple(executed_compensations)
            return x + self.out_proj(fused), route_logits

        height, width = x.shape[-2:]
        x_low = F.avg_pool2d(x, kernel_size=self.spatial_reduction, stride=self.spatial_reduction)
        route_weights_low = self._pooled_route_weights(route_weights)
        if not self._uses_local_compensation:
            routed_paths = []
            for index, path in enumerate(self.paths):
                weight = route_weights_low[:, index : index + 1]
                path_input = x_low * (1.0 + weight) if self.gate_inputs else x_low
                routed_paths.append(path(path_input) * weight)
            fused = torch.stack(routed_paths, dim=1).sum(dim=1)
            self.last_executed_local_compensation_directions = ()
            delta_low = self.out_proj(fused)
            delta = F.interpolate(delta_low, size=(height, width), mode="bilinear", align_corners=False)
            return x + delta, route_logits

        assert self.local_compensations is not None
        directions = ("horizontal", "vertical", "diagonal", "anti_diagonal")
        path_by_direction = dict(zip(self.active_mamba_directions, self.paths, strict=True))
        routed_paths = []
        executed_compensations: list[str] = []
        for index, direction in enumerate(directions):
            weight = route_weights_low[:, index : index + 1]
            if direction in path_by_direction:
                path_input = x_low * (1.0 + weight) if self.gate_inputs else x_low
                directional_feature = path_by_direction[direction](path_input)
            else:
                directional_feature = self.local_compensations[direction](x_low)
                executed_compensations.append(direction)
            routed_paths.append(directional_feature * weight)
        fused = torch.stack(routed_paths, dim=1).sum(dim=1)
        self.last_executed_local_compensation_directions = tuple(executed_compensations)
        delta_low = self.out_proj(fused)
        delta = F.interpolate(delta_low, size=(height, width), mode="bilinear", align_corners=False)
        return x + delta, route_logits


class SingleRasterMamba80(nn.Module):
    """One non-directional 80-channel raster Mamba route-branch replacement.

    This intentionally has no route head, scan object, pixel-wise weighting,
    or directional/local compensation.  Zero logits exist solely because
    ``TopoRouteBlock`` returns the historical ``(feature, route_logits)`` API;
    they never participate in feature computation.
    """

    dim = 320
    mamba_channels = 80
    route_bins = 4
    active_mamba_directions: tuple[str, ...] = ()
    spatial_reduction = 1
    route_disabled = True

    def __init__(self, use_mamba_ssm: bool = True):
        super().__init__()
        self.input_projection = nn.Conv2d(self.dim, self.mamba_channels, kernel_size=1, bias=False)
        self.raster_mamba = SequenceMambaCore(self.mamba_channels, use_mamba_ssm)
        self.output_projection = nn.Conv2d(self.mamba_channels, self.dim, kernel_size=1, bias=False)
        self.last_core_call_count = 0
        self.last_raster_sequence_shape: tuple[int, int, int] | None = None

    @property
    def backends(self) -> list[str]:
        return [self.raster_mamba.backend]

    @property
    def instantiated_core_count(self) -> int:
        return 1

    @property
    def active_mamba_path_count(self) -> int:
        # One executed raster path, not four directional traversals.
        return 1

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or x.size(1) != self.dim:
            raise ValueError(f"SingleRasterMamba80 expected [B, {self.dim}, H, W], received {tuple(x.shape)}.")
        batch, _, height, width = x.shape
        compact = self.input_projection(x)
        sequence = compact.permute(0, 2, 3, 1).reshape(batch, height * width, self.mamba_channels)
        self.last_raster_sequence_shape = tuple(sequence.shape)
        self.last_core_call_count = 0
        sequence = self.raster_mamba(sequence)
        self.last_core_call_count = 1
        restored = sequence.reshape(batch, height, width, self.mamba_channels).permute(0, 3, 1, 2).contiguous()
        routed = x + self.output_projection(restored)
        # Unparameterized compatibility output only; do not feed it back.
        return routed, x.new_zeros((batch, self.route_bins, height, width))


class DirectionalGridPermutation(nn.Module):
    """One cached, invertible full-grid ordering for a DPSM direction."""

    directions = ("horizontal", "vertical", "diagonal", "anti_diagonal")

    def __init__(self, direction: str):
        super().__init__()
        if direction not in self.directions:
            raise ValueError(f"Unsupported DPSM direction: {direction!r}.")
        self.direction = direction
        self._cache: dict[tuple[int, int, str, int | None], tuple[torch.Tensor, torch.Tensor]] = {}
        self.reorder_call_count = 0
        self.inverse_call_count = 0

    @staticmethod
    def _horizontal(height: int, width: int) -> list[int]:
        return [
            row * width + column
            for row in range(height)
            for column in (range(width) if row % 2 == 0 else range(width - 1, -1, -1))
        ]

    @staticmethod
    def _vertical(height: int, width: int) -> list[int]:
        return [
            row * width + column
            for column in range(width)
            for row in (range(height) if column % 2 == 0 else range(height - 1, -1, -1))
        ]

    @staticmethod
    def _anti_diagonal(height: int, width: int) -> list[int]:
        order: list[int] = []
        for diagonal_sum in range(height + width - 1):
            row_min = max(0, diagonal_sum - (width - 1))
            row_max = min(height - 1, diagonal_sum)
            rows = range(row_max, row_min - 1, -1) if diagonal_sum % 2 == 0 else range(row_min, row_max + 1)
            for row in rows:
                order.append(row * width + diagonal_sum - row)
        return order

    @classmethod
    def _order_list(cls, direction: str, height: int, width: int) -> list[int]:
        if direction == "horizontal":
            return cls._horizontal(height, width)
        if direction == "vertical":
            return cls._vertical(height, width)
        anti = cls._anti_diagonal(height, width)
        if direction == "anti_diagonal":
            return anti
        # Horizontal reflection converts anti-diagonal bands to the other
        # diagonal family without changing full-grid coverage.
        return [index // width * width + (width - 1 - index % width) for index in anti]

    def indices(self, height: int, width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if height < 1 or width < 1:
            raise ValueError(f"DPSM spatial dimensions must be positive, got {(height, width)}.")
        key = (height, width, device.type, device.index)
        cached = self._cache.get(key)
        if cached is not None:
            is_inference = getattr(cached[0], "is_inference", lambda: False)
            if is_inference():
                with torch.inference_mode(False):
                    cached = (cached[0].clone(), cached[1].clone())
                self._cache[key] = cached
            return cached
        with torch.inference_mode(False):
            order = torch.tensor(self._order_list(self.direction, height, width), device=device, dtype=torch.long)
            inverse = torch.empty_like(order)
            inverse.scatter_(0, order, torch.arange(height * width, device=device, dtype=torch.long))
        self._cache[key] = (order, inverse)
        return order, inverse

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = feature.shape
        order, _ = self.indices(height, width, feature.device)
        flat = feature.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        self.reorder_call_count += 1
        return flat[:, order, :]

    def inverse(self, sequence: torch.Tensor, height: int, width: int) -> torch.Tensor:
        if sequence.ndim != 3 or sequence.size(1) != height * width:
            raise ValueError("DPSM inverse permutation received an incompatible sequence shape.")
        _, inverse = self.indices(height, width, sequence.device)
        self.inverse_call_count += 1
        restored = sequence[:, inverse, :]
        return restored.reshape(sequence.size(0), height, width, sequence.size(2)).permute(0, 3, 1, 2).contiguous()


class DirectionPartitionedSharedMamba(nn.Module):
    """Four 80-channel directional subspaces packed through one Mamba core."""

    directions = ("horizontal", "vertical", "diagonal", "anti_diagonal")
    valid_scan_modes = ("four_direction", "axial_repeated")
    valid_route_modes = ("learned", "uniform")

    def __init__(
        self,
        dim: int,
        route_head: nn.Module,
        group_channels: int = 80,
        use_mamba_ssm: bool = True,
        scan_mode: str = "four_direction",
        route_mode: str = "learned",
    ):
        super().__init__()
        if dim != 320:
            raise ValueError("DPSM80 is fixed to the full 320-channel Stage4 representation.")
        if group_channels != 80 or len(self.directions) * group_channels != dim:
            raise ValueError("DPSM80 requires exactly four 80-channel groups whose total width is 320.")
        if scan_mode not in self.valid_scan_modes:
            raise ValueError(
                f"Invalid DPSM80 scan_mode {scan_mode!r}; expected one of {self.valid_scan_modes}."
            )
        if route_mode not in self.valid_route_modes:
            raise ValueError(
                f"Invalid DPSM80 route_mode {route_mode!r}; expected one of {self.valid_route_modes}."
            )
        self.dim = int(dim)
        self.group_channels = int(group_channels)
        self.route_bins = len(self.directions)
        self.shared_core = True
        self.spatial_reduction = 1
        self.mamba_channels = self.dim
        self.active_mamba_directions = self.directions
        # This is deliberately instance metadata rather than a replacement for
        # ``directions``: several frozen paths use the class-level four-way
        # contract.  The repeated axial experiment still retains four 80ch
        # groups and executes each group independently.
        self.scan_mode = scan_mode
        self.route_mode = route_mode
        self.group_scan_directions = (
            self.directions
            if scan_mode == "four_direction"
            else ("horizontal", "vertical", "horizontal", "vertical")
        )
        self.route_head = route_head
        self.pre_mix = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.scans = nn.ModuleDict({direction: DirectionalGridPermutation(direction) for direction in self.directions})
        self.shared_mamba = SequenceMambaCore(group_channels, use_mamba_ssm)
        self.output_mix = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.last_shared_mamba_call_count = 0
        self.last_packed_shape: tuple[int, int, int] | None = None
        self.last_direction_sequence_shapes: dict[str, tuple[int, ...]] = {}
        self.last_restored_shapes: dict[str, tuple[int, ...]] = {}
        # ``g0`` ... ``g3`` never collide when axial scans are repeated.  They
        # are diagnostic metadata only, so no scan cache or experiment setting
        # becomes a checkpoint tensor/buffer.
        self.last_group_sequence_shapes: dict[str, tuple[int, ...]] = {}
        self.last_group_restored_shapes: dict[str, tuple[int, ...]] = {}

    @property
    def group_scan_mapping(self) -> dict[str, str]:
        """Return the actual per-80-channel-group scan assignment."""

        return {f"g{index}": direction for index, direction in enumerate(self.group_scan_directions)}

    @property
    def backends(self) -> list[str]:
        return [self.shared_mamba.backend]

    @property
    def active_mamba_path_count(self) -> int:
        return self.route_bins

    @property
    def instantiated_core_count(self) -> int:
        return 1

    @staticmethod
    def apply_route_weights(
        restored: torch.Tensor,
        route_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Modulate only the matching direction group with fixed scale four."""

        if restored.ndim != 5 or restored.size(0) != 4:
            raise ValueError("DPSM restored directions must have shape [4, B, C, H, W].")
        if route_weights.ndim != 4 or route_weights.size(1) != 4:
            raise ValueError("DPSM route weights must have shape [B, 4, H, W].")
        weights = route_weights.permute(1, 0, 2, 3).unsqueeze(2)
        return restored * (4.0 * weights)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or x.size(1) != self.dim:
            raise ValueError(f"DPSM80 expected [B, {self.dim}, H, W], received {tuple(x.shape)}.")
        batch, _, height, width = x.shape
        route_logits = self.route_head(x)
        if self.route_mode == "learned":
            route_weights = torch.softmax(route_logits, dim=1)
        else:
            # Keep route_head and raw logits in the graph/API for strict
            # state_dict compatibility, but make mask modulation exactly
            # independent of them.  The existing fixed factor four therefore
            # gives every group unit gain (4 * 1/4 = 1).
            route_weights = torch.full_like(route_logits, 1.0 / self.route_bins)
        mixed = self.pre_mix(x)
        groups = mixed.split(self.group_channels, dim=1)
        if len(groups) != self.route_bins:
            raise RuntimeError("DPSM channel split did not produce four direction groups.")

        sequences = [
            self.scans[direction](group)
            for direction, group in zip(self.group_scan_directions, groups, strict=True)
        ]
        self.last_group_sequence_shapes = {
            f"g{index}": tuple(sequence.shape) for index, sequence in enumerate(sequences)
        }
        # Preserve the old default diagnostics exactly.  Repeated H/V scans
        # use group keys so neither duplicate silently overwrites the other.
        self.last_direction_sequence_shapes = (
            {
                direction: tuple(sequence.shape)
                for direction, sequence in zip(self.directions, sequences, strict=True)
            }
            if self.scan_mode == "four_direction"
            else dict(self.last_group_sequence_shapes)
        )
        packed = torch.stack(sequences, dim=0).reshape(
            self.route_bins * batch,
            height * width,
            self.group_channels,
        )
        self.last_packed_shape = tuple(packed.shape)
        self.last_shared_mamba_call_count = 0
        packed_output = self.shared_mamba(packed)
        self.last_shared_mamba_call_count += 1
        directional_outputs = packed_output.reshape(
            self.route_bins,
            batch,
            height * width,
            self.group_channels,
        )
        restored_list = [
            self.scans[direction].inverse(directional_outputs[index], height, width)
            for index, direction in enumerate(self.group_scan_directions)
        ]
        self.last_group_restored_shapes = {
            f"g{index}": tuple(feature.shape) for index, feature in enumerate(restored_list)
        }
        self.last_restored_shapes = (
            {
                direction: tuple(feature.shape)
                for direction, feature in zip(self.directions, restored_list, strict=True)
            }
            if self.scan_mode == "four_direction"
            else dict(self.last_group_restored_shapes)
        )
        restored = torch.stack(restored_list, dim=0)
        weighted = self.apply_route_weights(restored, route_weights)
        concatenated = torch.cat(tuple(weighted.unbind(dim=0)), dim=1)
        return x + self.output_mix(concatenated), route_logits


class CommonPrivateDirectionPartitionedSharedMamba(nn.Module):
    """One 128-channel Mamba over four common-private directional sequences.

    A 64-channel common subspace is reused in every direction while four
    disjoint 64-channel private subspaces retain directional specificity. The
    only sequential backend is one batch-packed 128-channel Mamba core.
    """

    directions = DirectionPartitionedSharedMamba.directions

    def __init__(
        self,
        dim: int,
        route_head: nn.Module,
        shared_channels: int = 64,
        private_channels: int = 64,
        use_mamba_ssm: bool = True,
    ):
        super().__init__()
        if dim != 320:
            raise ValueError("CP-DPSM64 is fixed to the full 320-channel Stage4 representation.")
        if shared_channels != 64 or private_channels != 64:
            raise ValueError("CP-DPSM64 requires exactly 64 common and 64 private channels per direction.")
        if shared_channels + len(self.directions) * private_channels != dim:
            raise ValueError("CP-DPSM64 common/private channels must sum exactly to the Stage4 width.")
        self.dim = int(dim)
        self.shared_channels = int(shared_channels)
        self.private_channels = int(private_channels)
        self.route_bins = len(self.directions)
        self.per_direction_channels = self.shared_channels + self.private_channels
        self.shared_core = True
        self.spatial_reduction = 1
        self.mamba_channels = self.per_direction_channels
        self.active_mamba_directions = self.directions
        self.route_head = route_head
        self.pre_mix = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.scans = nn.ModuleDict({direction: DirectionalGridPermutation(direction) for direction in self.directions})
        self.shared_mamba = SequenceMambaCore(self.per_direction_channels, use_mamba_ssm)
        self.output_mix = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.last_shared_mamba_call_count = 0
        self.last_packed_shape: tuple[int, int, int] | None = None
        self.last_direction_input_shapes: dict[str, tuple[int, ...]] = {}
        self.last_direction_sequence_shapes: dict[str, tuple[int, ...]] = {}
        self.last_restored_shapes: dict[str, tuple[int, ...]] = {}
        self.last_common_output_shape: tuple[int, ...] | None = None
        self.last_private_output_shape: tuple[int, ...] | None = None
        self.last_combined_shape: tuple[int, ...] | None = None

    @property
    def backends(self) -> list[str]:
        return [self.shared_mamba.backend]

    @property
    def active_mamba_path_count(self) -> int:
        return self.route_bins

    @property
    def instantiated_core_count(self) -> int:
        return 1

    @staticmethod
    def apply_route_weights(
        common: torch.Tensor,
        private: torch.Tensor,
        route_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse common paths normally and preserve private-path magnitude by four."""

        if common.ndim != 5 or private.ndim != 5 or common.shape[:2] != private.shape[:2] or common.size(0) != 4:
            raise ValueError("CP-DPSM common/private directions must both have shape [4, B, C, H, W].")
        if route_weights.ndim != 4 or route_weights.size(1) != 4:
            raise ValueError("CP-DPSM route weights must have shape [B, 4, H, W].")
        weights = route_weights.permute(1, 0, 2, 3).unsqueeze(2)
        fused_common = (common * weights).sum(dim=0)
        weighted_private = private * (4.0 * weights)
        return fused_common, weighted_private

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or x.size(1) != self.dim:
            raise ValueError(f"CP-DPSM64 expected [B, {self.dim}, H, W], received {tuple(x.shape)}.")
        batch, _, height, width = x.shape
        route_logits = self.route_head(x)
        route_weights = torch.softmax(route_logits, dim=1)
        mixed = self.pre_mix(x)
        common, *private_groups = mixed.split((self.shared_channels, *([self.private_channels] * self.route_bins)), dim=1)
        if len(private_groups) != self.route_bins:
            raise RuntimeError("CP-DPSM channel split did not produce four private direction groups.")
        direction_inputs = [torch.cat((common, private), dim=1) for private in private_groups]
        self.last_direction_input_shapes = {
            direction: tuple(feature.shape)
            for direction, feature in zip(self.directions, direction_inputs, strict=True)
        }
        sequences = [self.scans[direction](feature) for direction, feature in zip(self.directions, direction_inputs, strict=True)]
        self.last_direction_sequence_shapes = {
            direction: tuple(sequence.shape) for direction, sequence in zip(self.directions, sequences, strict=True)
        }
        packed = torch.stack(sequences, dim=0).reshape(
            self.route_bins * batch,
            height * width,
            self.per_direction_channels,
        )
        self.last_packed_shape = tuple(packed.shape)
        self.last_shared_mamba_call_count = 0
        packed_output = self.shared_mamba(packed)
        self.last_shared_mamba_call_count += 1
        directional_outputs = packed_output.reshape(
            self.route_bins,
            batch,
            height * width,
            self.per_direction_channels,
        )
        restored_list = [
            self.scans[direction].inverse(directional_outputs[index], height, width)
            for index, direction in enumerate(self.directions)
        ]
        self.last_restored_shapes = {
            direction: tuple(feature.shape)
            for direction, feature in zip(self.directions, restored_list, strict=True)
        }
        restored = torch.stack(restored_list, dim=0)
        common_outputs, private_outputs = restored.split((self.shared_channels, self.private_channels), dim=2)
        fused_common, weighted_private = self.apply_route_weights(common_outputs, private_outputs, route_weights)
        concatenated_private = torch.cat(tuple(weighted_private.unbind(dim=0)), dim=1)
        combined = torch.cat((fused_common, concatenated_private), dim=1)
        self.last_common_output_shape = tuple(fused_common.shape)
        self.last_private_output_shape = tuple(concatenated_private.shape)
        self.last_combined_shape = tuple(combined.shape)
        return x + self.output_mix(combined), route_logits


# Kept as a public compatibility alias for older imports.
DirectionGatedMambaBlock = TangentRoutedMambaBlock
