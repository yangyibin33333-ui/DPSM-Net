from __future__ import annotations

from copy import deepcopy

import torch


class ModelEMA:
    """A non-trainable exponential moving average of a model state dict."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        if not 0.0 <= decay < 1.0:
            raise ValueError("Model EMA decay must be in [0, 1).")
        self.decay = float(decay)
        self.module = deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        ema_state = self.module.state_dict(keep_vars=True)
        raw_state = model.state_dict(keep_vars=True)
        if ema_state.keys() != raw_state.keys():
            raise RuntimeError("Model EMA and raw model state dictionaries must have identical keys.")
        for name, ema_value in ema_state.items():
            raw_value = raw_state[name].detach().to(device=ema_value.device, dtype=ema_value.dtype)
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(raw_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(raw_value)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.module.state_dict()

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.module.load_state_dict(state_dict, strict=False)
