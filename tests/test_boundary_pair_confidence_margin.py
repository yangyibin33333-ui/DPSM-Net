from __future__ import annotations

import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from losses import BoundaryPairConfidenceMarginLoss, DEMambaLoss
from models import DEMambaNet


class BoundaryPairConfidenceMarginTests(unittest.TestCase):
    @staticmethod
    def _pair_logits(z_fg: float, z_bg: float) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor([[[[z_fg, z_bg]]]], dtype=torch.float32),
            torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32),
        )

    def test_bpcm_only_uses_unique_gt_cross_class_pairs(self) -> None:
        loss_fn = BoundaryPairConfidenceMarginLoss(margin=0.5, neg_temperature=0.5)
        logits = torch.tensor([[[[1.0, -1.0], [2.0, -2.0]]]])
        checkerboard = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        self.assertEqual(int(loss_fn._cross_class_pair_count(checkerboard)), 4)

        # With a 2x2 checkerboard, exactly two right and two down pairs are
        # present; no padding or reverse traversal contributes a fifth pair.
        expected_terms = []
        for z_fg, z_bg in ((1.0, -1.0), (-2.0, 2.0), (1.0, 2.0), (-2.0, -1.0)):
            weight = torch.sigmoid(torch.tensor(z_bg) / 0.5)
            expected_terms.append(F.softplus(torch.tensor(0.5 - (z_fg - z_bg))) + 0.5 * weight * F.softplus(torch.tensor(0.5 + z_bg)))
        self.assertTrue(torch.allclose(loss_fn(logits, checkerboard), torch.stack(expected_terms).mean()))

    def test_bpcm_is_zero_without_cross_class_pairs_and_small_when_separated(self) -> None:
        loss_fn = BoundaryPairConfidenceMarginLoss()
        logits = torch.randn(2, 1, 3, 4)
        self.assertEqual(float(loss_fn(logits, torch.zeros_like(logits))), 0.0)
        separated_logits, mask = self._pair_logits(10.0, -10.0)
        self.assertLess(float(loss_fn(separated_logits, mask)), 1e-7)

    def test_bpcm_improves_when_foreground_rises_or_background_falls(self) -> None:
        loss_fn = BoundaryPairConfidenceMarginLoss()
        baseline_logits, mask = self._pair_logits(0.0, 0.0)
        raised_foreground, _ = self._pair_logits(1.0, 0.0)
        lowered_background, _ = self._pair_logits(0.0, -1.0)
        baseline = loss_fn(baseline_logits, mask)
        self.assertLess(loss_fn(raised_foreground, mask), baseline)
        self.assertLess(loss_fn(lowered_background, mask), baseline)

    def test_bpcm_includes_horizontal_and_vertical_pairs(self) -> None:
        loss_fn = BoundaryPairConfidenceMarginLoss()
        horizontal = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        vertical = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
        logits = torch.zeros(1, 1, 2, 2)
        self.assertEqual(int(loss_fn._cross_class_pair_count(horizontal)), 2)
        self.assertEqual(int(loss_fn._cross_class_pair_count(vertical)), 2)
        self.assertGreater(float(loss_fn(logits, horizontal)), 0.0)
        self.assertGreater(float(loss_fn(logits, vertical)), 0.0)

    def test_bpcm_detaches_only_background_confidence_weight(self) -> None:
        loss_fn = BoundaryPairConfidenceMarginLoss(margin=0.5, neg_temperature=0.5)
        logits = torch.tensor([[[[0.2, 0.7]]]], requires_grad=True)
        mask = torch.tensor([[[[1.0, 0.0]]]])
        loss_fn(logits, mask).backward()
        actual_gradient = logits.grad.detach().clone()

        expected_logits = logits.detach().clone().requires_grad_(True)
        z_fg, z_bg = expected_logits[..., 0], expected_logits[..., 1]
        detached_weight = torch.sigmoid(z_bg.detach() / 0.5)
        expected = F.softplus(0.5 - (z_fg - z_bg)) + 0.5 * detached_weight * F.softplus(0.5 + z_bg)
        expected.mean().backward()
        self.assertTrue(torch.allclose(actual_gradient, expected_logits.grad))
        self.assertTrue(torch.isfinite(actual_gradient).all())
        self.assertNotEqual(float(actual_gradient[..., 0]), 0.0)
        self.assertNotEqual(float(actual_gradient[..., 1]), 0.0)

    def test_disabled_bpcm_is_exactly_legacy_loss_and_warmup_is_linear(self) -> None:
        torch.manual_seed(7)
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
        baseline = DEMambaLoss()
        disabled = DEMambaLoss(lambda_bpcm=0.0)
        baseline_total, _ = baseline(outputs, batch, epoch=100)
        disabled_total, items = disabled(outputs, batch, epoch=100)
        self.assertTrue(torch.equal(disabled_total, baseline_total))
        self.assertEqual(items["bpcm_loss"], 0.0)
        self.assertEqual(items["bpcm_scale"], 0.0)
        self.assertEqual(items["lambda_bpcm"], 0.0)

        warmup = DEMambaLoss(lambda_bpcm=0.025, bpcm_start_epoch=60, bpcm_ramp_epochs=40)
        self.assertEqual(warmup._bpcm_scale(59), 0.0)
        self.assertEqual(warmup._bpcm_scale(60), 0.0)
        self.assertAlmostEqual(warmup._bpcm_scale(80), 0.0125)
        self.assertEqual(warmup._bpcm_scale(100), 0.025)

    def test_ema_bpcm_config_only_adds_bpcm_loss_fields_and_model_parameter_count_is_unchanged(self) -> None:
        ema_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA.yaml")
        experiment_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml")
        with ema_path.open(encoding="utf-8") as handle:
            ema = yaml.safe_load(handle)
        with experiment_path.open(encoding="utf-8") as handle:
            experiment = yaml.safe_load(handle)
        for section in ("model", "data", "train", "evaluation"):
            self.assertEqual(experiment[section], ema[section])
        for name, value in ema["loss"].items():
            self.assertEqual(experiment["loss"][name], value)
        self.assertEqual(
            {name: experiment["loss"][name] for name in experiment["loss"] if name not in ema["loss"]},
            {
                "lambda_bpcm": 0.025,
                "bpcm_start_epoch": 60,
                "bpcm_ramp_epochs": 40,
                "bpcm_margin": 0.5,
                "bpcm_neg_temperature": 0.5,
            },
        )
        self.assertEqual(experiment["save"]["save_dir"], "./runs/pilot/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM")
        self.assertEqual(sum(parameter.numel() for parameter in BoundaryPairConfidenceMarginLoss().parameters()), 0)
        model_cfg = dict(ema["model"])
        model_cfg.pop("name")
        self.assertEqual(sum(parameter.numel() for parameter in DEMambaNet(**model_cfg).parameters()), 7026611)
