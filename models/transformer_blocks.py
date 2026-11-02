import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, max_tokens: int = 1024):
        super().__init__()
        self.heads = heads
        self.max_tokens = max_tokens
        self.head_dim = dim // heads
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.norm = nn.GroupNorm(1, dim)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.heads, self.head_dim, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = q.transpose(-2, -1)
        attn = (q @ k) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
        out = out.reshape(b, c, h, w)
        return self.proj(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        if h * w <= self.max_tokens:
            return self._attention(x)
        side = max(1, int(math.sqrt(self.max_tokens)))
        pooled = F.adaptive_avg_pool2d(x, output_size=(side, side))
        out = self._attention(pooled)
        return F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)


class TransformerGlobalBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, mlp_ratio: float = 2.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.attn = LightweightSelfAttention(dim, heads=heads)
        self.norm = nn.GroupNorm(1, dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(self.norm(x))
        return x
