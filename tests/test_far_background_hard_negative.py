from __future__ import annotations

import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from losses import DEMambaLoss, FarBackgroundHardNegativeLoss


class FarBackgroundHardNegativeTests(unittest.TestCase):
    def test_fbhn_excludes_crack_and_boundary_band_then_selects_per_image_topk(self) -> None:
        loss_fn = FarBackgroundHardNegativeLoss(exclusion_kernel_size=5, topk_per_image=1, margin=0.25)
        logits = torch.full((2, 1, 7, 7), -10.0)
        mask = torch.zeros_like(logits)
        mask[0, 0, 3, 3] = 1.0
        logits[0, 0, 3, 5] = 20.0  # Within the 5x5 exclusion band.
        logits[0, 0, 3, 0] = 2.0   # Exactly three pixels from the foreground.
        logits[1, 0, 0, 0] = 5.0
        logits[1, 0, 6, 6] = 1.0

        far_background = loss_fn._far_background_mask(mask)
        self.assertFalse(bool(far_background[0, 0, 3, 3]))
        self.assertFalse(bool(far_background[0, 0, 3, 5]))
        self.assertTrue(bool(far_background[0, 0, 3, 0]))
        # Each image contributes its own top-1; batch-global top-k would drop
        # the first image's logit 2.0 in favor of the second image's 5.0.
        expected = (F.softplus(torch.tensor(2.25)) + F.softplus(torch.tensor(5.25))) / 2.0
        self.assertTrue(torch.allclose(loss_fn(logits, mask), expected))

    def test_fbhn_all_background_selects_highest_logits_and_is_monotonic(self) -> None:
        loss_fn = FarBackgroundHardNegativeLoss(exclusion_kernel_size=5, topk_per_image=2, margin=0.25)
        logits = torch.tensor([[[[-3.0, 2.0], [1.0, 4.0]]]])
        mask = torch.zeros_like(logits)
        self.assertTrue(bool(loss_fn._far_background_mask(mask).all()))
        expected = F.softplus(torch.tensor([4.25, 2.25])).mean()
        self.assertTrue(torch.allclose(loss_fn(logits, mask), expected))
        lowered = logits.clone()
        lowered[0, 0, 1, 1] = 0.0
        self.assertLess(loss_fn(lowered, mask), loss_fn(logits, mask))

    def test_fbhn_gradients_only_reach_selected_far_background(self) -> None:
        loss_fn = FarBackgroundHardNegativeLoss(exclusion_kernel_size=5, topk_per_image=1)
        logits = torch.full((1, 1, 7, 7), -10.0, requires_grad=True)
        mask = torch.zeros_like(logits)
        mask[0, 0, 3, 3] = 1.0
        with torch.no_grad():
            logits[0, 0, 3, 5] = 20.0  # Excluded boundary-band background.
            logits[0, 0, 3, 0] = 2.0   # Selected far-background hard negative.
        loss_fn(logits, mask).backward()
        self.assertNotEqual(float(logits.grad[0, 0, 3, 0]), 0.0)
        self.assertEqual(float(logits.grad[0, 0, 3, 5]), 0.0)
        self.assertEqual(float(logits.grad[0, 0, 3, 3]), 0.0)

    def test_fbhn_returns_zero_without_candidates_and_validates_hyperparameters(self) -> None:
        loss_fn = FarBackgroundHardNegativeLoss()
        logits = torch.zeros(1, 1, 1, 1)
        mask = torch.ones_like(logits)
        self.assertEqual(float(loss_fn(logits, mask)), 0.0)
        with self.assertRaises(ValueError):
            FarBackgroundHardNegativeLoss(exclusion_kernel_size=4)
        with self.assertRaises(ValueError):
            FarBackgroundHardNegativeLoss(exclusion_kernel_size=0)
        with self.assertRaises(ValueError):
            FarBackgroundHardNegativeLoss(topk_per_image=0)

    def test_disabled_fbhn_matches_ema_bpcm_loss_and_warmup_is_linear(self) -> None:
        torch.manual_seed(11)
        outputs = {
            "mask": torch.randn(1, 1, 3, 4),
            "edge": torch.randn(1, 1, 3, 4),
            "route_s3": torch.randn(1, 4, 2, 2),
        }
        batch = {
            "mask": torch.randint(0, 2, (1, 1, 3, 4), dtype=torch.float32),
            "edge_gt": torch.randint(0, 2, (1, 1, 3, 4), dtype=torch.float32),
            "tangent_gt": torch.softmax(torch.randn(1, 4, 3, 4), dim=1),
            "tangent_valid": torch.ones(1, 1, 3, 4),
        }
        ema_bpcm = DEMambaLoss(lambda_bpcm=0.025, bpcm_start_epoch=60, bpcm_ramp_epochs=40)
        disabled = DEMambaLoss(lambda_bpcm=0.025, bpcm_start_epoch=60, bpcm_ramp_epochs=40, lambda_fbhn=0.0)
        baseline_total, _ = ema_bpcm(outputs, batch, epoch=140)
        disabled_total, items = disabled(outputs, batch, epoch=140)
        self.assertTrue(torch.equal(disabled_total, baseline_total))
        self.assertEqual(items["fbhn_loss"], 0.0)
        self.assertEqual(items["fbhn_scale"], 0.0)
        self.assertEqual(items["lambda_fbhn"], 0.0)

        warmup = DEMambaLoss(lambda_fbhn=0.02, fbhn_start_epoch=100, fbhn_ramp_epochs=40)
        self.assertEqual(warmup._fbhn_scale(99), 0.0)
        self.assertEqual(warmup._fbhn_scale(100), 0.0)
        self.assertAlmostEqual(warmup._fbhn_scale(120), 0.01)
        self.assertEqual(warmup._fbhn_scale(140), 0.02)

    def test_ema_bpcm_fbhn_config_only_adds_fbhn_fields_and_model_parameters_are_zero(self) -> None:
        base_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml")
        experiment_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM_FBHNN.yaml")
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
                "lambda_fbhn": 0.02,
                "fbhn_start_epoch": 100,
                "fbhn_ramp_epochs": 40,
                "fbhn_exclusion_kernel_size": 5,
                "fbhn_topk_per_image": 256,
                "fbhn_margin": 0.25,
            },
        )
        self.assertEqual(experiment["save"]["save_dir"], "./runs/pilot/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM_FBHNN")
        self.assertEqual(sum(parameter.numel() for parameter in FarBackgroundHardNegativeLoss().parameters()), 0)
