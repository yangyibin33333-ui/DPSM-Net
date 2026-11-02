from __future__ import annotations

import copy
import inspect
import unittest
from pathlib import Path

import torch
import yaml

from losses import DEMambaLoss
from losses.loss import AuditAlignedBoundaryBridgeFalsePositiveLoss
from models import DEMambaNet
from scripts.audit_validation_false_positives import classify_false_positives
from utils.fp_geometry import bridge_false_positive_mask


class AuditAlignedBoundaryBridgeFalsePositiveTests(unittest.TestCase):
    @staticmethod
    def _selection_case() -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.full((1, 1, 13, 13), -10.0, requires_grad=True)
        mask = torch.zeros_like(logits)
        mask[0, 0, 5, 2] = 1.0
        mask[0, 0, 5, 8] = 1.0
        mask[0, 0, 9, 9] = 1.0
        with torch.no_grad():
            logits[0, 0, 5, 2:9] = 2.0  # A single predicted component bridges two GT components.
            logits[0, 0, 9, 7] = 2.0  # Within the 5x5 dilation of the isolated GT component.
            logits[0, 0, 1, 1:4] = 2.0  # Far-background cohesive texture, intentionally excluded.
            logits[0, 0, 1, 8] = 2.0  # Far isolated false positive, intentionally excluded.
        return logits, mask

    @staticmethod
    def _batch_and_outputs() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        torch.manual_seed(31)
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

    def test_aabfp_matches_audit_geometry_is_detached_and_disabled_is_legacy_exact(self) -> None:
        logits, mask = self._selection_case()
        aabfp = AuditAlignedBoundaryBridgeFalsePositiveLoss(boundary_kernel_size=5, margin=0.25)
        selected, boundary, bridge = aabfp.selected_masks(logits, mask)
        self.assertFalse(selected.requires_grad)
        self.assertIsNone(selected.grad_fn)
        self.assertEqual(int(bridge.sum()), 5)
        self.assertEqual(int(boundary.sum()), 1)
        self.assertTrue(torch.equal(boundary & bridge, torch.zeros_like(boundary)))
        self.assertFalse(bool(selected[0, 0, 5, 2]))  # GT foreground is never selected.
        self.assertFalse(bool(selected[0, 0, 3, 3]))  # Correct background is never selected.
        self.assertFalse(bool(selected[0, 0, 1, 2]))  # Far texture is not an AABFP target.
        self.assertFalse(bool(selected[0, 0, 1, 8]))  # Far isolated is not an AABFP target.

        prediction = (torch.sigmoid(logits.detach()) > 0.5)[0, 0].cpu().numpy()
        target = (mask > 0.5)[0, 0].cpu().numpy()
        false_positive = prediction & (~target)
        helper_bridge = torch.from_numpy(bridge_false_positive_mask(prediction, target, false_positive))
        audit_bridge = classify_false_positives(torch.sigmoid(logits.detach()), mask)["bridge"][0, 0]
        self.assertTrue(torch.equal(helper_bridge, audit_bridge))
        self.assertTrue(torch.equal(helper_bridge, bridge[0, 0].cpu()))

        aabfp_loss = aabfp.loss_from_selected(logits, selected)
        aabfp_loss.backward()
        self.assertTrue(bool((logits.grad[selected] > 0).all()))
        self.assertTrue(torch.equal(logits.grad[~selected], torch.zeros_like(logits.grad[~selected])))
        empty_logits = torch.full((1, 1, 5, 5), -10.0)
        self.assertTrue(torch.equal(aabfp(empty_logits, torch.zeros_like(empty_logits)), empty_logits.new_zeros(())))

        outputs, batch = self._batch_and_outputs()
        baseline = DEMambaLoss(lambda_bpcm=0.025, bpcm_start_epoch=60, bpcm_ramp_epochs=40)
        disabled = DEMambaLoss(
            lambda_bpcm=0.025,
            bpcm_start_epoch=60,
            bpcm_ramp_epochs=40,
            lambda_aabfp=0.0,
        )
        baseline_total, _ = baseline(outputs, batch, epoch=150)
        disabled_total, disabled_items = disabled(outputs, batch, epoch=150)
        self.assertTrue(torch.equal(disabled_total, baseline_total))
        self.assertEqual(disabled_items["aabfp_loss"], 0.0)
        self.assertEqual(disabled_items["aabfp_scale"], 0.0)
        self.assertEqual(disabled_items["lambda_aabfp"], 0.0)
        self.assertEqual(disabled_items["aabfp_boundary_selected_pixels_per_image"], 0.0)
        self.assertEqual(disabled_items["aabfp_bridge_selected_pixels_per_image"], 0.0)

        ramp = DEMambaLoss(lambda_aabfp=0.010, aabfp_start_epoch=120, aabfp_ramp_epochs=30)
        self.assertEqual(ramp._aabfp_scale(119), 0.0)
        self.assertEqual(ramp._aabfp_scale(120), 0.0)
        self.assertEqual(ramp._aabfp_scale(135), 0.005)
        self.assertEqual(ramp._aabfp_scale(150), 0.010)

        base_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml")
        experiment_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM_AABFP.yaml")
        with base_path.open(encoding="utf-8") as handle:
            base_config = yaml.safe_load(handle)
        with experiment_path.open(encoding="utf-8") as handle:
            experiment_config = yaml.safe_load(handle)
        expected_config = copy.deepcopy(base_config)
        expected_config["loss"].update(
            {
                "lambda_aabfp": 0.010,
                "aabfp_start_epoch": 120,
                "aabfp_ramp_epochs": 30,
                "aabfp_boundary_kernel_size": 5,
                "aabfp_margin": 0.25,
            }
        )
        expected_config["save"]["save_dir"] = "./runs/pilot/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM_AABFP"
        self.assertEqual(experiment_config, expected_config)

        defaults = {
            name: parameter.default
            for name, parameter in inspect.signature(DEMambaNet.__init__).parameters.items()
            if name != "self" and parameter.default is not inspect.Parameter.empty
        }
        for name, default in defaults.items():
            if name.startswith("use_") and name not in {"use_mamba_ssm", "use_transformer"}:
                self.assertFalse(experiment_config["model"].get(name, default), name)
        self.assertTrue(experiment_config["train"]["use_model_ema"])
        self.assertEqual(experiment_config["train"]["model_ema_decay"], 0.999)
        self.assertTrue(experiment_config["train"]["amp"])
        self.assertEqual(experiment_config["train"]["amp_dtype"], "fp16")
        self.assertEqual(experiment_config["train"]["batch_size"], 8)

        torch.manual_seed(42)
        model = DEMambaNet(base_channels=40, use_mamba_ssm=True, mamba_stages=(3, 4), transformer_stages=(4,))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 7_026_611)
        self.assertTrue(
            all(backend == "mamba_ssm" for backends in model.mamba_backend_info().values() for backend in backends)
        )


if __name__ == "__main__":
    unittest.main()
