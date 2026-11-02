from __future__ import annotations

import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from losses import DEMambaLoss, TangentNormalHardNegativeCalibrationLoss


class TangentNormalContrastTests(unittest.TestCase):
    @staticmethod
    def _horizontal_anchor_case(
        z_pos: float = 0.0,
        up_logits: tuple[float, float] = (-10.0, -10.0),
        down_logits: tuple[float, float] = (-10.0, -10.0),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = torch.full((1, 1, 9, 9), -10.0)
        mask = torch.zeros_like(logits)
        tangent = torch.zeros(1, 4, 9, 9)
        valid = torch.zeros_like(logits)
        row, column = 4, 4
        logits[0, 0, row, column] = z_pos
        logits[0, 0, row - 1, column], logits[0, 0, row - 2, column] = up_logits
        logits[0, 0, row + 1, column], logits[0, 0, row + 2, column] = down_logits
        mask[0, 0, row, column] = 1.0
        tangent[0, 0, row, column] = 1.0
        valid[0, 0, row, column] = 1.0
        return logits, mask, tangent, valid

    def test_tnhnc_uses_only_valid_exterior_background_and_hard_negatives(self) -> None:
        loss_fn = TangentNormalHardNegativeCalibrationLoss(margin=0.5, temperature=0.25, max_normal_offset=2)
        logits, mask, tangent, valid = self._horizontal_anchor_case(
            z_pos=0.0,
            up_logits=(1.0, 3.0),
            down_logits=(-1.0, 2.0),
        )
        loss = loss_fn(logits, mask, tangent, valid)
        positive_term = F.softplus(torch.tensor(0.5))
        expected = 0.5 * (
            positive_term
            + torch.sigmoid(torch.tensor(3.0 / 0.25)) * F.softplus(torch.tensor(0.5 + 3.0))
            + positive_term
            + torch.sigmoid(torch.tensor(2.0 / 0.25)) * F.softplus(torch.tensor(0.5 + 2.0))
        )
        self.assertTrue(torch.allclose(loss, expected, atol=1e-6, rtol=1e-6))

        mask_with_foreground_candidate = mask.clone()
        mask_with_foreground_candidate[0, 0, 2, 4] = 1.0
        loss_without_foreground_candidate = loss_fn(logits, mask_with_foreground_candidate, tangent, valid)
        self.assertLess(loss_without_foreground_candidate, loss)

    def test_tnhnc_is_small_for_separated_logits_and_monotonic_for_better_logits(self) -> None:
        loss_fn = TangentNormalHardNegativeCalibrationLoss(margin=0.5, temperature=0.25, max_normal_offset=1)
        separated = self._horizontal_anchor_case(z_pos=10.0, up_logits=(-10.0, -10.0), down_logits=(-10.0, -10.0))
        self.assertLess(float(loss_fn(*separated)), 1e-4)

        baseline = self._horizontal_anchor_case(z_pos=0.0, up_logits=(1.0, -10.0), down_logits=(1.0, -10.0))
        higher_positive = self._horizontal_anchor_case(z_pos=1.0, up_logits=(1.0, -10.0), down_logits=(1.0, -10.0))
        lower_negative = self._horizontal_anchor_case(z_pos=0.0, up_logits=(-1.0, -10.0), down_logits=(-1.0, -10.0))
        baseline_loss = loss_fn(*baseline)
        self.assertLess(loss_fn(*higher_positive), baseline_loss)
        self.assertLess(loss_fn(*lower_negative), baseline_loss)

    def test_tnhnc_ignores_invalid_anchors_foreground_candidates_and_out_of_bounds(self) -> None:
        loss_fn = TangentNormalHardNegativeCalibrationLoss(max_normal_offset=1)
        logits, mask, tangent, valid = self._horizontal_anchor_case(z_pos=0.0, up_logits=(5.0, -10.0), down_logits=(5.0, -10.0))
        valid.zero_()
        self.assertEqual(float(loss_fn(logits, mask, tangent, valid)), 0.0)

        logits, mask, tangent, valid = self._horizontal_anchor_case(z_pos=0.0, up_logits=(5.0, -10.0), down_logits=(5.0, -10.0))
        mask[0, 0, 3, 4] = 1.0
        mask[0, 0, 5, 4] = 1.0
        self.assertEqual(float(loss_fn(logits, mask, tangent, valid)), 0.0)

        boundary_logits = torch.full((1, 1, 3, 3), 5.0)
        boundary_mask = torch.zeros_like(boundary_logits)
        boundary_tangent = torch.zeros(1, 4, 3, 3)
        boundary_valid = torch.zeros_like(boundary_logits)
        boundary_mask[0, 0, 0, 1] = 1.0
        boundary_mask[0, 0, 1, 1] = 1.0
        boundary_tangent[0, 0, 0, 1] = 1.0
        boundary_valid[0, 0, 0, 1] = 1.0
        self.assertEqual(float(loss_fn(boundary_logits, boundary_mask, boundary_tangent, boundary_valid)), 0.0)

    def test_tnhnc_normal_mapping_and_disabled_demamba_loss(self) -> None:
        self.assertEqual(
            TangentNormalHardNegativeCalibrationLoss.NORMAL_OFFSETS,
            {
                0: ((-1, 0), (1, 0)),
                1: ((0, -1), (0, 1)),
                2: ((-1, 1), (1, -1)),
                3: ((-1, -1), (1, 1)),
            },
        )

        torch.manual_seed(7)
        outputs = {
            "mask": torch.randn(1, 1, 7, 9),
            "edge": torch.randn(1, 1, 7, 9),
        }
        batch = {
            "mask": torch.randint(0, 2, (1, 1, 7, 9), dtype=torch.float32),
            "edge_gt": torch.randint(0, 2, (1, 1, 7, 9), dtype=torch.float32),
            "tangent_gt": torch.zeros(1, 4, 7, 9),
            "tangent_valid": torch.zeros(1, 1, 7, 9),
        }
        baseline = DEMambaLoss()
        disabled = DEMambaLoss(lambda_tnhnc=0.0)
        baseline_total, _ = baseline(outputs, batch, epoch=80)
        disabled_total, items = disabled(outputs, batch, epoch=80)
        self.assertTrue(torch.equal(disabled_total, baseline_total))
        self.assertEqual(items["tnhnc_loss"], 0.0)
        self.assertEqual(items["tnhnc_scale"], 0.0)
        self.assertEqual(items["lambda_tnhnc"], 0.0)

        warmup = DEMambaLoss(lambda_tnhnc=0.08, tnhnc_start_epoch=30, tnhnc_ramp_epochs=50)
        self.assertEqual(warmup._tnhnc_scale(29), 0.0)
        self.assertEqual(warmup._tnhnc_scale(30), 0.0)
        self.assertAlmostEqual(warmup._tnhnc_scale(55), 0.04)
        self.assertEqual(warmup._tnhnc_scale(80), 0.08)

    def test_tnhnc_experiment_config_only_adds_loss_fields(self) -> None:
        default_path = Path("configs/demambanet.yaml")
        baseline_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544.yaml")
        experiment_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_TNHNC.yaml")
        with default_path.open(encoding="utf-8") as handle:
            defaults = yaml.safe_load(handle)
        with baseline_path.open(encoding="utf-8") as handle:
            baseline = yaml.safe_load(handle)
        with experiment_path.open(encoding="utf-8") as handle:
            experiment = yaml.safe_load(handle)

        self.assertEqual(
            {name: defaults["loss"][name] for name in (
                "lambda_tnhnc",
                "tnhnc_start_epoch",
                "tnhnc_ramp_epochs",
                "tnhnc_margin",
                "tnhnc_temperature",
                "tnhnc_max_normal_offset",
            )},
            {
                "lambda_tnhnc": 0.0,
                "tnhnc_start_epoch": 30,
                "tnhnc_ramp_epochs": 50,
                "tnhnc_margin": 0.5,
                "tnhnc_temperature": 0.25,
                "tnhnc_max_normal_offset": 6,
            },
        )

        for section in ("model", "data", "train", "evaluation"):
            self.assertEqual(experiment[section], baseline[section])
        for name, value in baseline["loss"].items():
            self.assertEqual(experiment["loss"][name], value)
        self.assertEqual(
            {name: experiment["loss"][name] for name in experiment["loss"] if name not in baseline["loss"]},
            {
                "lambda_tnhnc": 0.08,
                "tnhnc_start_epoch": 30,
                "tnhnc_ramp_epochs": 50,
                "tnhnc_margin": 0.5,
                "tnhnc_temperature": 0.25,
                "tnhnc_max_normal_offset": 6,
            },
        )


if __name__ == "__main__":
    unittest.main()
