from __future__ import annotations

import math
import unittest
from pathlib import Path

import torch
import yaml

from models import DEMambaNet
from train import make_scheduler


BASE_CONFIG = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml")
WARMUP_CONFIG = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM_WARMUP5.yaml")
BASE_LR = 1.5e-4


class _Logger:
    def info(self, *_args, **_kwargs) -> None:
        pass


def _optimizer() -> torch.optim.Optimizer:
    return torch.optim.SGD([torch.nn.Parameter(torch.zeros(()))], lr=BASE_LR)


def _epoch_lrs(optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler, epochs: int) -> list[float]:
    learning_rates: list[float] = []
    for _ in range(epochs):
        learning_rates.append(float(optimizer.param_groups[0]["lr"]))
        optimizer.step()
        scheduler.step()
    return learning_rates


class WarmupCosineScheduleTests(unittest.TestCase):
    @staticmethod
    def _load_config(path: Path) -> dict:
        with path.open(encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def test_warmup5_lr_trace_and_plain_cosine_regression(self) -> None:
        warmup_cfg = self._load_config(WARMUP_CONFIG)
        optimizer = _optimizer()
        scheduler = make_scheduler(optimizer, warmup_cfg, _Logger())
        lrs = _epoch_lrs(optimizer, scheduler, epochs=200)

        self.assertAlmostEqual(lrs[0], 3.0e-5, places=12)
        self.assertAlmostEqual(lrs[4], BASE_LR, places=12)
        self.assertAlmostEqual(lrs[5], BASE_LR, places=12)
        self.assertTrue(all(next_lr <= current_lr for current_lr, next_lr in zip(lrs[5:], lrs[6:])))
        self.assertAlmostEqual(lrs[-1], 0.0, places=12)

        expected_epoch_100 = BASE_LR * 0.5 * (1.0 + math.cos(math.pi * 94 / 194))
        self.assertAlmostEqual(lrs[99], expected_epoch_100, places=12)
        print(
            "WarmupCosine LR trace | "
            f"epoch1={lrs[0]:.12f} | epoch5={lrs[4]:.12f} | epoch6={lrs[5]:.12f} | "
            f"epoch100={lrs[99]:.12f} | epoch200={lrs[-1]:.12f}"
        )

        cosine_cfg = self._load_config(BASE_CONFIG)["train"]
        factory_optimizer, reference_optimizer = _optimizer(), _optimizer()
        factory_scheduler = make_scheduler(factory_optimizer, cosine_cfg, _Logger())
        reference_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            reference_optimizer,
            T_max=int(cosine_cfg["epochs"]),
            eta_min=float(cosine_cfg["min_lr"]),
        )
        for epoch in range(int(cosine_cfg["epochs"])):
            self.assertEqual(factory_optimizer.param_groups[0]["lr"], reference_optimizer.param_groups[0]["lr"], epoch)
            factory_optimizer.step()
            reference_optimizer.step()
            factory_scheduler.step()
            reference_scheduler.step()
        self.assertEqual(factory_optimizer.param_groups[0]["lr"], reference_optimizer.param_groups[0]["lr"])

    def test_warmup_config_only_adds_scheduler_strategy_and_preserves_model_audit(self) -> None:
        base = self._load_config(BASE_CONFIG)
        warmup = self._load_config(WARMUP_CONFIG)
        expected = dict(base)
        expected["scheduler"] = {"name": "warmup_cosine", "warmup_epochs": 5, "warmup_start_factor": 0.2}
        self.assertEqual(warmup, expected)

        torch.manual_seed(42)
        model = DEMambaNet(base_channels=40, use_mamba_ssm=True, mamba_stages=(3, 4), transformer_stages=(4,))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 7_026_611)
        self.assertTrue(
            all(backend == "mamba_ssm" for backends in model.mamba_backend_info().values() for backend in backends)
        )


if __name__ == "__main__":
    unittest.main()
