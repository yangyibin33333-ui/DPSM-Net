from __future__ import annotations

import unittest
from pathlib import Path

import torch
import yaml

from losses import DEMambaLoss
from losses.loss import EndpointProtectedHardConfuserFeatureContrastLoss
from models import DEMambaNet


class EndpointProtectedHardConfuserFeatureContrastTests(unittest.TestCase):
    @staticmethod
    def _contrast_case(hard_negative_feature: tuple[float, float]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feature = torch.zeros(1, 2, 11, 11)
        mask = torch.zeros(1, 1, 11, 11)
        valid = torch.zeros_like(mask)
        logits = torch.full_like(mask, -5.0)
        anchor = (5, 5)
        support = (5, 6)
        excluded_background = (5, 8)
        far_background = (0, 0)
        anchor_row, anchor_column = anchor
        support_row, support_column = support
        excluded_row, excluded_column = excluded_background
        far_row, far_column = far_background
        mask[0, 0, anchor_row, anchor_column] = 1.0
        mask[0, 0, support_row, support_column] = 1.0
        valid.copy_(mask)
        logits[0, 0, anchor_row, anchor_column] = -2.0
        logits[0, 0, support_row, support_column] = 3.0
        logits[0, 0, excluded_row, excluded_column] = 20.0
        logits[0, 0, far_row, far_column] = 4.0
        feature[0, :, anchor_row, anchor_column] = torch.tensor([1.0, 0.0])
        feature[0, :, support_row, support_column] = torch.tensor([1.0, 0.0])
        feature[0, :, far_row, far_column] = torch.tensor(hard_negative_feature)
        return feature.requires_grad_(), logits.requires_grad_(), mask, valid

    def test_feature_contrast_excludes_gt_band_and_backpropagates_only_to_features(self) -> None:
        loss_fn = EndpointProtectedHardConfuserFeatureContrastLoss(
            temperature=0.20,
            exclusion_kernel_size=7,
            endpoint_anchors=1,
            foreground_anchors=1,
            positive_supports=1,
            hard_negatives=1,
        )
        same_feature, same_logits, mask, valid = self._contrast_case((1.0, 0.0))
        hard_negative_mask = loss_fn._hard_negative_mask(mask, same_logits)
        self.assertFalse(bool(hard_negative_mask[0, 0, 5, 8]))
        self.assertTrue(bool(hard_negative_mask[0, 0, 0, 0]))

        same_direction_loss = loss_fn(same_feature, same_logits, mask, valid)
        self.assertTrue(torch.isfinite(same_direction_loss))

        opposite_feature, opposite_logits, _, _ = self._contrast_case((-1.0, 0.0))
        opposite_direction_loss = loss_fn(opposite_feature, opposite_logits, mask, valid)
        self.assertLess(opposite_direction_loss, same_direction_loss.detach())

        gradient_feature, gradient_logits, _, _ = self._contrast_case((0.0, 1.0))
        gradient_loss = loss_fn(gradient_feature, gradient_logits, mask, valid)
        gradient_loss.backward()
        self.assertIsNotNone(gradient_feature.grad)
        self.assertTrue(torch.isfinite(gradient_feature.grad).all())
        self.assertNotEqual(float(gradient_feature.grad.abs().sum()), 0.0)
        self.assertIsNone(gradient_logits.grad)

    def test_zero_cases_endpoint_selection_and_linear_ramp(self) -> None:
        loss_fn = EndpointProtectedHardConfuserFeatureContrastLoss()
        feature = torch.zeros(1, 4, 9, 11)
        logits = torch.full((1, 1, 9, 11), -1.0)
        mask = torch.zeros_like(logits)
        valid = torch.zeros_like(logits)
        self.assertEqual(float(loss_fn(feature, logits, mask, valid)), 0.0)
        self.assertTrue(torch.isfinite(loss_fn(feature, logits, mask, valid)))

        mask[0, 0, 4, 4] = 1.0
        valid[0, 0, 4, 4] = 1.0
        self.assertEqual(float(loss_fn(feature, logits, mask, valid)), 0.0)

        skeleton = torch.zeros(1, 1, 7, 9)
        skeleton[0, 0, 1, 1] = 1.0
        skeleton[0, 0, 4, 2:5] = 1.0
        endpoint_mask = loss_fn._endpoint_mask(skeleton > 0.5, skeleton)
        self.assertTrue(bool(endpoint_mask[0, 0, 1, 1]))
        self.assertTrue(bool(endpoint_mask[0, 0, 4, 2]))
        self.assertFalse(bool(endpoint_mask[0, 0, 4, 3]))
        self.assertTrue(bool(endpoint_mask[0, 0, 4, 4]))

        endpoint_logits = torch.zeros_like(skeleton)
        endpoint_logits[0, 0, 4, 2] = -3.0
        endpoint_logits[0, 0, 4, 4] = -1.0
        selected = EndpointProtectedHardConfuserFeatureContrastLoss(
            endpoint_anchors=1,
            foreground_anchors=1,
        )._anchor_indices(endpoint_logits[0, 0], skeleton[0, 0].bool(), skeleton[0, 0])
        self.assertEqual(selected.tolist(), [4 * 9 + 2])

        outputs = {
            "mask": torch.randn(1, 1, 3, 4),
            "edge": torch.randn(1, 1, 3, 4),
        }
        batch = {
            "mask": torch.randint(0, 2, (1, 1, 3, 4), dtype=torch.float32),
            "edge_gt": torch.randint(0, 2, (1, 1, 3, 4), dtype=torch.float32),
            "tangent_gt": torch.zeros(1, 4, 3, 4),
            "tangent_valid": torch.zeros(1, 1, 3, 4),
        }
        baseline = DEMambaLoss()
        disabled = DEMambaLoss(lambda_ep_hcfc=0.0)
        baseline_total, _ = baseline(outputs, batch, epoch=150)
        disabled_total, disabled_items = disabled(outputs, batch, epoch=150)
        self.assertTrue(torch.equal(disabled_total, baseline_total))
        self.assertEqual(disabled_items["ep_hcfc_loss"], 0.0)
        self.assertEqual(disabled_items["ep_hcfc_scale"], 0.0)
        self.assertEqual(disabled_items["lambda_ep_hcfc"], 0.0)

        criterion = DEMambaLoss(lambda_ep_hcfc=0.02, ep_hcfc_start_epoch=120, ep_hcfc_ramp_epochs=30)
        self.assertEqual(criterion._ep_hcfc_scale(119), 0.0)
        self.assertEqual(criterion._ep_hcfc_scale(120), 0.0)
        self.assertAlmostEqual(criterion._ep_hcfc_scale(135), 0.01)
        self.assertEqual(criterion._ep_hcfc_scale(150), 0.02)
        with self.assertRaisesRegex(RuntimeError, "return_training_mask_feature"):
            criterion(outputs, batch, epoch=121)

    def test_training_only_feature_output_preserves_model_and_experiment_config(self) -> None:
        base_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml")
        experiment_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM_EPHCFC.yaml")
        with base_path.open(encoding="utf-8") as handle:
            base_cfg = yaml.safe_load(handle)
        with experiment_path.open(encoding="utf-8") as handle:
            experiment_cfg = yaml.safe_load(handle)
        self.assertEqual(experiment_cfg["data"], base_cfg["data"])
        self.assertEqual(experiment_cfg["train"], base_cfg["train"])
        self.assertEqual(experiment_cfg["evaluation"], base_cfg["evaluation"])
        for name, value in base_cfg["model"].items():
            self.assertEqual(experiment_cfg["model"][name], value)
        for name, value in base_cfg["loss"].items():
            self.assertEqual(experiment_cfg["loss"][name], value)
        self.assertEqual(
            {name: value for name, value in experiment_cfg["model"].items() if name not in base_cfg["model"]},
            {
                "stage3_mamba_window_size": None,
                "stage3_bidirectional_mamba": False,
                "use_sdf_head": False,
                "use_edge_mask_consistency_refiner": False,
                "use_deep_supervision": False,
                "use_retinex_directional_valley_prior": False,
                "use_route_guided_directional_refiner": False,
                "use_route_guided_anisotropic_diffusion": False,
                "use_density_balanced_prototype_context": False,
                "use_local_semantic_continuity_calibrator": False,
                "use_local_pixel_reassembly_residual": False,
                "use_local_spectral_confidence_rejector": False,
                "use_high_resolution_spectral_detail_gate": False,
                "use_semantic_conditioned_local_kernel_residual": False,
                "use_residual_branch_competition_fusion": False,
                "return_training_mask_feature": True,
            },
        )
        self.assertEqual(
            {name: value for name, value in experiment_cfg["loss"].items() if name not in base_cfg["loss"]},
            {
                "lambda_fbhn": 0.0,
                "lambda_sdf": 0.0,
                "lambda_aux_mask": 0.0,
                "lambda_aux_edge": 0.0,
                "lambda_ep_hcfc": 0.02,
                "ep_hcfc_start_epoch": 120,
                "ep_hcfc_ramp_epochs": 30,
                "ep_hcfc_temperature": 0.20,
                "ep_hcfc_exclusion_kernel_size": 7,
                "ep_hcfc_endpoint_anchors": 48,
                "ep_hcfc_foreground_anchors": 80,
                "ep_hcfc_positive_supports": 128,
                "ep_hcfc_hard_negatives": 128,
            },
        )
        self.assertEqual(
            experiment_cfg["save"]["save_dir"],
            "./runs/pilot/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM_EPHCFC",
        )
        enabled_optional_modules = {
            name
            for name, value in experiment_cfg["model"].items()
            if name.startswith("use_") and value and name != "use_mamba_ssm"
        }
        self.assertEqual(enabled_optional_modules, set())
        self.assertFalse(experiment_cfg["model"]["stage3_bidirectional_mamba"])

        base_model_cfg = dict(base_cfg["model"])
        base_model_cfg.pop("name")
        experiment_model_cfg = dict(experiment_cfg["model"])
        experiment_model_cfg.pop("name")
        torch.manual_seed(42)
        native_c40 = DEMambaNet(**base_model_cfg).eval()
        torch.manual_seed(42)
        feature_c40 = DEMambaNet(**experiment_model_cfg).eval()
        self.assertEqual(sum(parameter.numel() for parameter in native_c40.parameters()), 7026611)
        self.assertEqual(sum(parameter.numel() for parameter in feature_c40.parameters()), 7026611)
        for index, (native_parameter, feature_parameter) in enumerate(
            zip(native_c40.parameters(), feature_c40.parameters(), strict=True)
        ):
            self.assertTrue(torch.equal(native_parameter, feature_parameter), index)

        cpu_kwargs = {
            "base_channels": 8,
            "use_mamba_ssm": False,
            "transformer_stages": [],
        }
        torch.manual_seed(42)
        cpu_native = DEMambaNet(**cpu_kwargs).eval()
        torch.manual_seed(42)
        cpu_feature = DEMambaNet(**cpu_kwargs, return_training_mask_feature=True).eval()
        image = torch.randn(1, 3, 64, 80)
        with torch.no_grad():
            native_outputs = cpu_native(image)
            feature_outputs = cpu_feature(image)
        self.assertEqual(set(native_outputs), set(feature_outputs))
        self.assertNotIn("mask_feature", feature_outputs)
        for name, native_output in native_outputs.items():
            self.assertTrue(torch.equal(native_output, feature_outputs[name]), name)

        cpu_feature.train()
        training_outputs = cpu_feature(image)
        self.assertIn("mask_feature", training_outputs)
        self.assertEqual(tuple(training_outputs["mask_feature"].shape[-2:]), tuple(training_outputs["mask"].shape[-2:]))


if __name__ == "__main__":
    unittest.main()
