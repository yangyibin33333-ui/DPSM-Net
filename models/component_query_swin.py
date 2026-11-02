from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for NCHW feature maps."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.channels = int(channels)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.channels))
        self.bias = nn.Parameter(torch.zeros(self.channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (self.channels,), self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2).contiguous()


def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition NHWC tensors into non-overlapping windows."""
    batch, height, width, channels = x.shape
    x = x.view(batch, height // window_size, window_size, width // window_size, window_size, channels)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size, channels)


def _window_reverse(
    windows: torch.Tensor,
    batch: int,
    height: int,
    width: int,
    window_size: int,
) -> torch.Tensor:
    channels = windows.size(-1)
    x = windows.view(batch, height // window_size, width // window_size, window_size, window_size, channels)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(batch, height, width, channels)


class WindowSelfAttention(nn.Module):
    """Shifted local-window attention with relative bias and padded-token masking."""

    def __init__(self, channels: int, num_heads: int, window_size: int, shift_size: int) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("WindowSelfAttention channels must be divisible by num_heads.")
        if shift_size not in (0, window_size // 2):
            raise ValueError("WindowSelfAttention shift_size must be 0 or half the window size.")
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.head_dim = self.channels // self.num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(self.channels, 3 * self.channels)
        self.proj = nn.Linear(self.channels, self.channels)

        relative_size = 2 * self.window_size - 1
        self.relative_position_bias_table = nn.Parameter(torch.zeros(relative_size * relative_size, self.num_heads))
        coordinates = torch.stack(
            torch.meshgrid(torch.arange(self.window_size), torch.arange(self.window_size), indexing="ij")
        ).flatten(1)
        relative_coordinates = coordinates[:, :, None] - coordinates[:, None, :]
        relative_coordinates[0] += self.window_size - 1
        relative_coordinates[1] += self.window_size - 1
        relative_index = relative_coordinates[0] * relative_size + relative_coordinates[1]
        self.register_buffer("relative_position_index", relative_index, persistent=False)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def _attention_bias(
        self,
        batch: int,
        height: int,
        width: int,
        padded_height: int,
        padded_width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        window = self.window_size
        valid = torch.ones((1, height, width, 1), dtype=torch.bool, device=device)
        valid = F.pad(valid, (0, 0, 0, padded_width - width, 0, padded_height - height))

        region = torch.zeros((1, padded_height, padded_width, 1), device=device)
        if self.shift_size > 0:
            counter = 0
            height_slices = (slice(0, -window), slice(-window, -self.shift_size), slice(-self.shift_size, None))
            width_slices = (slice(0, -window), slice(-window, -self.shift_size), slice(-self.shift_size, None))
            for height_slice in height_slices:
                for width_slice in width_slices:
                    region[:, height_slice, width_slice] = counter
                    counter += 1
            valid = torch.roll(valid, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            region = torch.roll(region, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        valid_windows = _window_partition(valid.to(dtype=torch.float32), window).squeeze(-1) > 0.5
        region_windows = _window_partition(region, window).squeeze(-1)
        region_difference = region_windows.unsqueeze(1) != region_windows.unsqueeze(2)
        invalid_key = ~valid_windows.unsqueeze(1).expand(-1, window * window, -1)
        local_mask = region_difference | invalid_key
        local_mask = local_mask.unsqueeze(0).expand(batch, -1, -1, -1).reshape(
            batch * local_mask.size(0), window * window, window * window
        )
        attention_bias = torch.zeros(
            (local_mask.size(0), 1, window * window, window * window), dtype=dtype, device=device
        )
        return attention_bias.masked_fill(local_mask.unsqueeze(1), torch.finfo(dtype).min)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        if channels != self.channels:
            raise ValueError(f"WindowSelfAttention expected {self.channels} channels, got {channels}.")
        window = self.window_size
        padded_height = ((height + window - 1) // window) * window
        padded_width = ((width + window - 1) // window) * window
        x = x.permute(0, 2, 3, 1)
        x = F.pad(x, (0, 0, 0, padded_width - width, 0, padded_height - height))
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        windows = _window_partition(x, window)
        query, key, value = self.qkv(windows).reshape(
            windows.size(0), windows.size(1), 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        relative_bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)].view(
            window * window, window * window, self.num_heads
        ).permute(2, 0, 1).to(dtype=query.dtype)
        attention_bias = self._attention_bias(
            batch, height, width, padded_height, padded_width, query.dtype, query.device
        ) + relative_bias.unsqueeze(0)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_bias,
            dropout_p=0.0,
            scale=self.scale,
        )
        attended = attended.transpose(1, 2).reshape(windows.size(0), window * window, self.channels)
        x = self.proj(attended)
        x = _window_reverse(x, batch, padded_height, padded_width, window)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        return x[:, :height, :width].permute(0, 3, 1, 2).contiguous()


class SwinBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int, window_size: int, shift_size: int) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.attention = WindowSelfAttention(channels, num_heads, window_size, shift_size)
        self.norm2 = LayerNorm2d(channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, 4 * channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(4 * channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PatchMerging(nn.Module):
    """Swin-style 2x2 merge implemented independently of existing encoders."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.norm = LayerNorm2d(4 * in_channels)
        self.reduction = nn.Conv2d(4 * in_channels, 2 * in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        if height % 2 or width % 2:
            x = F.pad(x, (0, width % 2, 0, height % 2))
        x = torch.cat((x[..., 0::2, 0::2], x[..., 1::2, 0::2], x[..., 0::2, 1::2], x[..., 1::2, 1::2]), dim=1)
        return self.reduction(self.norm(x))


class QueryDecoderLayer(nn.Module):
    def __init__(self, channels: int, num_heads: int) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(channels)
        self.self_attention = nn.MultiheadAttention(channels, num_heads, dropout=0.0, batch_first=True)
        self.cross_norm = nn.LayerNorm(channels)
        self.cross_attention = nn.MultiheadAttention(channels, num_heads, dropout=0.0, batch_first=True)
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.GELU(),
            nn.Linear(4 * channels, channels),
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        normalized = self.self_norm(queries)
        queries = queries + self.self_attention(normalized, normalized, normalized, need_weights=False)[0]
        queries = queries + self.cross_attention(self.cross_norm(queries), memory, memory, need_weights=False)[0]
        return queries + self.ffn(self.ffn_norm(queries))


class ComponentQuerySwin(nn.Module):
    """Mamba-free shifted-window encoder with differentiable crack-component queries."""

    _EXPECTED_STAGE_DIMS = (44, 88, 176, 352)
    _EXPECTED_STAGE_DEPTHS = (2, 2, 8, 2)
    _EXPECTED_NUM_HEADS = (4, 4, 8, 16)
    _WINDOW_SIZE = 8
    _NUM_QUERIES = 64
    _DECODER_DIM = 128
    _DECODER_LAYERS = 2
    _DECODER_HEADS = 8

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        stage_dims: Sequence[int] = _EXPECTED_STAGE_DIMS,
        stage_depths: Sequence[int] = _EXPECTED_STAGE_DEPTHS,
        num_heads: Sequence[int] = _EXPECTED_NUM_HEADS,
        window_size: int = _WINDOW_SIZE,
        num_queries: int = _NUM_QUERIES,
        decoder_dim: int = _DECODER_DIM,
        decoder_layers: int = _DECODER_LAYERS,
        decoder_heads: int = _DECODER_HEADS,
    ) -> None:
        super().__init__()
        stage_dims = tuple(int(value) for value in stage_dims)
        stage_depths = tuple(int(value) for value in stage_depths)
        num_heads = tuple(int(value) for value in num_heads)
        if int(in_channels) != 3 or int(num_classes) != 1:
            raise ValueError("ComponentQuerySwin is fixed to RGB binary segmentation.")
        if stage_dims != self._EXPECTED_STAGE_DIMS or stage_depths != self._EXPECTED_STAGE_DEPTHS:
            raise ValueError("ComponentQuerySwin uses fixed dimensions (44, 88, 176, 352) and depths (2, 2, 8, 2).")
        if num_heads != self._EXPECTED_NUM_HEADS or int(window_size) != self._WINDOW_SIZE:
            raise ValueError("ComponentQuerySwin uses fixed heads (4, 4, 8, 16) and window_size=8.")
        if (int(num_queries), int(decoder_dim), int(decoder_layers), int(decoder_heads)) != (
            self._NUM_QUERIES,
            self._DECODER_DIM,
            self._DECODER_LAYERS,
            self._DECODER_HEADS,
        ):
            raise ValueError("ComponentQuerySwin uses fixed query decoder settings: 64 queries, 128 dim, 2 layers, 8 heads.")

        self.is_mamba_free = True
        self.patch_embed = nn.Conv2d(3, stage_dims[0], kernel_size=3, stride=2, padding=1)
        self.stage1 = self._make_stage(stage_dims[0], stage_depths[0], num_heads[0])
        self.merge1 = PatchMerging(stage_dims[0])
        self.stage2 = self._make_stage(stage_dims[1], stage_depths[1], num_heads[1])
        self.merge2 = PatchMerging(stage_dims[1])
        self.stage3 = self._make_stage(stage_dims[2], stage_depths[2], num_heads[2])
        self.merge3 = PatchMerging(stage_dims[2])
        self.stage4 = self._make_stage(stage_dims[3], stage_depths[3], num_heads[3])

        self.memory_projections = nn.ModuleList(
            nn.Conv2d(channels, self._DECODER_DIM, kernel_size=1) for channels in stage_dims[1:]
        )
        self.scale_embeddings = nn.Parameter(torch.zeros(3, self._DECODER_DIM))
        self.query_embeddings = nn.Parameter(torch.zeros(self._NUM_QUERIES, self._DECODER_DIM))
        self.query_decoder = nn.ModuleList(
            QueryDecoderLayer(self._DECODER_DIM, self._DECODER_HEADS) for _ in range(self._DECODER_LAYERS)
        )
        self.pixel_projection = nn.Conv2d(stage_dims[0], self._DECODER_DIM, kernel_size=1)
        self.mask_embedding = nn.Linear(self._DECODER_DIM, self._DECODER_DIM)
        self.component_class_head = nn.Linear(self._DECODER_DIM, 2)
        self.edge_head = nn.Conv2d(self._DECODER_DIM, 1, kernel_size=1)
        self.reliability_head = nn.Conv2d(self._DECODER_DIM, 1, kernel_size=1)
        self.route_s1_head = nn.Conv2d(self._DECODER_DIM, 4, kernel_size=1)
        nn.init.trunc_normal_(self.scale_embeddings, std=0.02)
        nn.init.trunc_normal_(self.query_embeddings, std=0.02)

    @staticmethod
    def _make_stage(channels: int, depth: int, heads: int) -> nn.Sequential:
        return nn.Sequential(*(SwinBlock(channels, heads, 8, 0 if index % 2 == 0 else 4) for index in range(depth)))

    def _decode_queries(self, c2: torch.Tensor, c3: torch.Tensor, c4: torch.Tensor) -> torch.Tensor:
        memory_tokens = []
        for index, (projection, feature) in enumerate(zip(self.memory_projections, (c2, c3, c4))):
            projected = projection(feature).flatten(2).transpose(1, 2).contiguous()
            memory_tokens.append(projected + self.scale_embeddings[index].view(1, 1, -1))
        memory = torch.cat(memory_tokens, dim=1).contiguous()
        queries = self.query_embeddings.unsqueeze(0).expand(c2.size(0), -1, -1).contiguous()
        for layer in self.query_decoder:
            queries = layer(queries, memory)
        return queries

    @staticmethod
    def _component_union(component_masks: torch.Tensor, component_classes: torch.Tensor) -> torch.Tensor:
        class_probability = component_classes.softmax(dim=-1)[..., 1].unsqueeze(-1).unsqueeze(-1)
        mask_probability = component_masks.sigmoid()
        union_probability = 1.0 - torch.prod(1.0 - class_probability * mask_probability, dim=1, keepdim=True)
        return torch.logit(union_probability.clamp(min=1e-6, max=1.0 - 1e-6))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        c1 = self.stage1(self.patch_embed(x))
        c2 = self.stage2(self.merge1(c1))
        c3 = self.stage3(self.merge2(c2))
        c4 = self.stage4(self.merge3(c3))

        pixel_feature = self.pixel_projection(c1)
        queries = self._decode_queries(c2, c3, c4)
        component_masks = torch.einsum("bqc,bchw->bqhw", self.mask_embedding(queries), pixel_feature)
        component_classes = self.component_class_head(queries)
        mask_logits = self._component_union(component_masks, component_classes)
        return {
            "mask": F.interpolate(mask_logits, size=input_size, mode="bilinear", align_corners=False),
            "edge": F.interpolate(self.edge_head(pixel_feature), size=input_size, mode="bilinear", align_corners=False),
            "reliability": F.interpolate(
                self.reliability_head(pixel_feature), size=input_size, mode="bilinear", align_corners=False
            ),
            "route_s1": self.route_s1_head(pixel_feature),
            "component_mask_logits": component_masks,
            "component_class_logits": component_classes,
        }

    @staticmethod
    def mamba_backend_info() -> dict[str, list[str]]:
        return {}
