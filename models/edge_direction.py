from __future__ import annotations

import torch
import torch.nn.functional as F

# Fixed Sobel responses.  The two 3x3 kernels are constants, so they are built
# once per (device, dtype) instead of being reallocated and re-copied on every
# forward pass.  They are created outside any ``torch.inference_mode`` context
# so that the cached tensors stay usable by grad-enabled training steps.
_SOBEL_KX = ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))
_SOBEL_KY = ((-1.0, -2.0, -1.0), (0.0, 0.0, 0.0), (1.0, 2.0, 1.0))
_SOBEL_CACHE: dict[tuple[str, int | None, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}


def _sobel_kernels(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    key = (device.type, device.index, dtype)
    cached = _SOBEL_CACHE.get(key)
    if cached is not None:
        is_inference = getattr(cached[0], "is_inference", lambda: False)
        if not is_inference():
            return cached
        # An inference-mode cache cannot be reused by a grad-enabled step.
        _SOBEL_CACHE.pop(key, None)
    with torch.inference_mode(False):
        kx = torch.tensor(_SOBEL_KX, dtype=dtype, device=device).view(1, 1, 3, 3)
        ky = torch.tensor(_SOBEL_KY, dtype=dtype, device=device).view(1, 1, 3, 3)
    _SOBEL_CACHE[key] = (kx, ky)
    return kx, ky


def _to_gray(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(1)
    return x.float() if x.size(1) == 1 else x.float().mean(dim=1, keepdim=True)


def sobel_edge(x: torch.Tensor, normalize: bool = True, eps: float = 1e-6) -> torch.Tensor:
    """Lightweight image edge prior used only by the CNN stem."""
    gray = _to_gray(x)
    kx, ky = _sobel_kernels(gray.device, gray.dtype)
    dx, dy = F.conv2d(gray, kx, padding=1), F.conv2d(gray, ky, padding=1)
    edge = torch.sqrt(dx.square() + dy.square() + eps)
    return edge / edge.amax(dim=(2, 3), keepdim=True).clamp_min(eps) if normalize else edge
