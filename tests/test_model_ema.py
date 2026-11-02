from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from train import make_model_ema
from utils import ModelEMA, load_checkpoint, save_checkpoint


class TinyBatchNormModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)
        self.norm = nn.BatchNorm1d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(x))


class ModelEMATests(unittest.TestCase):
    @staticmethod
    def _fill_float_state(model: TinyBatchNormModel, value: float, batches: int) -> None:
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(value)
            model.norm.running_mean.fill_(value + 1.0)
            model.norm.running_var.fill_(value + 2.0)
            model.norm.num_batches_tracked.fill_(batches)

    def test_ema_update_matches_formula_and_does_not_mutate_raw_model(self) -> None:
        raw = TinyBatchNormModel()
        self._fill_float_state(raw, value=1.0, batches=2)
        ema = ModelEMA(raw, decay=0.75)
        raw_before = {name: value.detach().clone() for name, value in raw.state_dict().items()}

        self._fill_float_state(raw, value=5.0, batches=9)
        raw_after_change = {name: value.detach().clone() for name, value in raw.state_dict().items()}
        ema.update(raw)

        for name, parameter in ema.module.named_parameters():
            self.assertTrue(torch.equal(parameter, torch.full_like(parameter, 2.0)), name)
        self.assertTrue(torch.equal(ema.module.norm.running_mean, torch.full_like(ema.module.norm.running_mean, 3.0)))
        self.assertTrue(torch.equal(ema.module.norm.running_var, torch.full_like(ema.module.norm.running_var, 4.0)))
        self.assertEqual(int(ema.module.norm.num_batches_tracked), 9)
        for name, value in raw.state_dict().items():
            self.assertTrue(torch.equal(value, raw_after_change[name]), name)
            self.assertFalse(torch.equal(value, raw_before[name]), name)
        self.assertTrue(all(not parameter.requires_grad for parameter in ema.module.parameters()))

    def test_ema_checkpoint_uses_shadow_weights(self) -> None:
        raw = TinyBatchNormModel()
        self._fill_float_state(raw, value=1.0, batches=1)
        ema = ModelEMA(raw, decay=0.5)
        self._fill_float_state(raw, value=5.0, batches=7)
        ema.update(raw)
        optimizer = torch.optim.SGD(raw.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ema.pth"
            save_checkpoint(path, ema.module, optimizer, scheduler, epoch=3, best_score=0.8, raw_model=raw)
            checkpoint = torch.load(path, map_location="cpu")
            self.assertTrue(torch.equal(checkpoint["model"]["linear.weight"], ema.module.linear.weight))
            self.assertFalse(torch.equal(checkpoint["model"]["linear.weight"], raw.linear.weight))
            self.assertTrue(torch.equal(checkpoint["raw_model"]["linear.weight"], raw.linear.weight))

            loaded_for_inference = TinyBatchNormModel()
            start_epoch, best_score = load_checkpoint(path, loaded_for_inference, map_location="cpu")
            self.assertEqual(start_epoch, 4)
            self.assertEqual(best_score, 0.8)
            self.assertTrue(torch.equal(loaded_for_inference.linear.weight, ema.module.linear.weight))

            loaded_for_resume = TinyBatchNormModel()
            load_checkpoint(path, loaded_for_resume, map_location="cpu", prefer_raw_model=True)
            self.assertTrue(torch.equal(loaded_for_resume.linear.weight, raw.linear.weight))

    def test_ema_disabled_config_and_experiment_config_are_native_compatible(self) -> None:
        raw = TinyBatchNormModel()
        self.assertIsNone(make_model_ema(raw, {"use_model_ema": False, "model_ema_decay": 0.999}))

        default_path = Path("configs/demambanet.yaml")
        native_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544.yaml")
        ema_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA.yaml")
        with default_path.open(encoding="utf-8") as handle:
            defaults = yaml.safe_load(handle)
        with native_path.open(encoding="utf-8") as handle:
            native = yaml.safe_load(handle)
        with ema_path.open(encoding="utf-8") as handle:
            experiment = yaml.safe_load(handle)

        self.assertFalse(defaults["train"]["use_model_ema"])
        self.assertEqual(defaults["train"]["model_ema_decay"], 0.999)
        for section in ("model", "data", "evaluation"):
            self.assertEqual(experiment[section], native[section])
        self.assertEqual(
            {name: experiment["train"][name] for name in experiment["train"] if name not in native["train"]},
            {"use_model_ema": True, "model_ema_decay": 0.999},
        )
        for name, value in native["train"].items():
            self.assertEqual(experiment["train"][name], value)
        for name, value in native["loss"].items():
            self.assertEqual(experiment["loss"][name], value)
        self.assertEqual(experiment["loss"]["lambda_tnhnc"], 0.0)
        self.assertEqual(experiment["save"]["save_dir"], "./runs/pilot/C3_C40_LR150_PHOTO_NATIVE384x544_EMA")


if __name__ == "__main__":
    unittest.main()
