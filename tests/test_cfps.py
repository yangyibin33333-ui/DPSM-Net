from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from losses import DEMambaLoss
from losses.loss import CohesiveFalsePositiveSuppressionLoss
from models import DEMambaNet


class CohesiveFalsePositiveSuppressionTests(unittest.TestCase):
    @staticmethod
    def _batch_and_outputs() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        torch.manual_seed(9)
        outputs = {
            "mask": torch.randn(1, 1, 8, 9),
            "edge": torch.randn(1, 1, 8, 9),
            "route_s3": torch.randn(1, 4, 4, 5),
        }
        batch = {
            "mask": torch.randint(0, 2, (1, 1, 8, 9), dtype=torch.float32),
            "edge_gt": torch.randint(0, 2, (1, 1, 8, 9), dtype=torch.float32),
            "tangent_gt": torch.softmax(torch.randn(1, 4, 8, 9), dim=1),
            "tangent_valid": torch.ones(1, 1, 8, 9),
        }
        return outputs, batch

    def test_cfps_is_safe_detached_cohesive_and_disabled_is_legacy_exact(self) -> None:
        outputs, batch = self._batch_and_outputs()
        baseline = DEMambaLoss(lambda_bpcm=0.025, bpcm_start_epoch=60, bpcm_ramp_epochs=40)
        disabled = DEMambaLoss(
            lambda_bpcm=0.025,
            bpcm_start_epoch=60,
            bpcm_ramp_epochs=40,
            lambda_cfps=0.0,
        )
        baseline_total, _ = baseline(outputs, batch, epoch=150)
        disabled_total, disabled_items = disabled(outputs, batch, epoch=150)
        self.assertTrue(torch.equal(disabled_total, baseline_total))
        self.assertEqual(disabled_items["cfps_loss"], 0.0)
        self.assertEqual(disabled_items["cfps_scale"], 0.0)
        self.assertEqual(disabled_items["lambda_cfps"], 0.0)
        self.assertEqual(disabled_items["cfps_selected_pixels_per_image"], 0.0)
        self.assertEqual(disabled_items["cfps_weighted_loss"], 0.0)
        self.assertEqual(disabled_items["cfps_active_image_ratio"], 0.0)

        loss_fn = CohesiveFalsePositiveSuppressionLoss(
            exclusion_kernel_size=5,
            cohesion_kernel_size=3,
            min_active_neighbors=3,
            margin=0.25,
        )
        logits = torch.full((1, 1, 11, 11), -10.0, requires_grad=True)
        mask = torch.zeros_like(logits)
        mask[0, 0, 5, 5] = 1.0
        with torch.no_grad():
            logits[0, 0, 5, 5] = 4.0
            logits[0, 0, 5, 7] = 4.0
            logits[0, 0, 1, 1:4] = 1.0
            logits[0, 0, 9, 9] = 1.0
        selected = loss_fn.selected_mask(logits, mask)
        self.assertFalse(selected.requires_grad)
        self.assertIsNone(selected.grad_fn)
        self.assertFalse(bool(selected[0, 0, 5, 5]))
        self.assertFalse(bool(selected[0, 0, 5, 7]))
        self.assertFalse(bool(selected[0, 0, 9, 9]))
        # A three-pixel line gives its center three active cells in the 3x3
        # neighborhood. The two endpoints correctly see only two cells.
        self.assertTrue(bool(selected[0, 0, 1, 2]))
        self.assertFalse(bool(selected[0, 0, 1, 1]))
        self.assertFalse(bool(selected[0, 0, 1, 3]))

        cfps_loss = loss_fn(logits, mask)
        cfps_loss.backward()
        self.assertGreater(float(logits.grad[0, 0, 1, 2]), 0.0)
        self.assertEqual(float(logits.grad[0, 0, 5, 5]), 0.0)
        self.assertEqual(float(logits.grad[0, 0, 5, 7]), 0.0)
        self.assertEqual(float(logits.grad[0, 0, 9, 9]), 0.0)

        ramp = DEMambaLoss(lambda_cfps=0.010, cfps_start_epoch=120, cfps_ramp_epochs=30)
        self.assertEqual(ramp._cfps_scale(119), 0.0)
        self.assertEqual(ramp._cfps_scale(120), 0.0)
        self.assertEqual(ramp._cfps_scale(135), 0.005)
        self.assertEqual(ramp._cfps_scale(150), 0.010)

    def test_cfps_config_only_adds_loss_fields_and_keeps_model_audit(self) -> None:
        base_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml")
        experiment_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM_CFPS.yaml")
        with base_path.open(encoding="utf-8") as handle:
            base = yaml.safe_load(handle)
        with experiment_path.open(encoding="utf-8") as handle:
            experiment = yaml.safe_load(handle)
        for section in ("model", "data", "train", "evaluation"):
            self.assertEqual(experiment[section], base[section])
        for name, value in base["loss"].items():
            self.assertEqual(experiment["loss"][name], value)
        self.assertEqual(
            {name: experiment["loss"][name] for name in experiment["loss"] if name not in base["loss"]},
            {
                "lambda_cfps": 0.010,
                "cfps_start_epoch": 120,
                "cfps_ramp_epochs": 30,
                "cfps_exclusion_kernel_size": 5,
                "cfps_cohesion_kernel_size": 3,
                "cfps_min_active_neighbors": 3,
                "cfps_margin": 0.25,
            },
        )
        self.assertEqual(experiment["save"]["save_dir"], "./runs/pilot/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM_CFPS")
        model_defaults = {
            name: parameter.default
            for name, parameter in inspect.signature(DEMambaNet.__init__).parameters.items()
            if name != "self" and parameter.default is not inspect.Parameter.empty
        }
        for name, default in model_defaults.items():
            if name.startswith("use_") and name not in {"use_mamba_ssm", "use_transformer"}:
                self.assertFalse(experiment["model"].get(name, default), name)
        self.assertEqual(experiment["train"]["lr_scheduler"], "cosine")

        torch.manual_seed(42)
        model = DEMambaNet(base_channels=40, use_mamba_ssm=True, mamba_stages=(3, 4), transformer_stages=(4,))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 7_026_611)
        self.assertTrue(
            all(backend == "mamba_ssm" for backends in model.mamba_backend_info().values() for backend in backends)
        )
        self.assertTrue(experiment["train"]["use_model_ema"])
        self.assertEqual(experiment["train"]["model_ema_decay"], 0.999)
        self.assertTrue(experiment["train"]["amp"])
        self.assertEqual(experiment["train"]["amp_dtype"], "fp16")
        self.assertEqual(experiment["train"]["batch_size"], 8)


if __name__ == "__main__":
    unittest.main()
