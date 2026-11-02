from __future__ import annotations

import copy
import inspect
import unittest
from pathlib import Path

import torch
import yaml

from losses import DEMambaLoss
from models import DEMambaNet


class NoTangentAblationTests(unittest.TestCase):
    @staticmethod
    def _batch() -> dict[str, torch.Tensor]:
        mask = torch.zeros(1, 1, 6, 7)
        mask[0, 0, 2:4, 2:5] = 1.0
        tangent_gt = torch.zeros(1, 4, 6, 7)
        tangent_gt[:, 2] = 1.0
        return {
            "mask": mask,
            "edge_gt": mask.clone(),
            "tangent_gt": tangent_gt,
            "tangent_valid": torch.ones(1, 1, 6, 7),
        }

    @staticmethod
    def _outputs(route_logits: torch.Tensor) -> dict[str, torch.Tensor]:
        torch.manual_seed(17)
        return {
            "mask": torch.randn(1, 1, 6, 7, requires_grad=True),
            "edge": torch.randn(1, 1, 6, 7, requires_grad=True),
            "route_s3": route_logits,
        }

    def test_config_is_single_variable_and_tangent_loss_is_gradient_free_when_disabled(self) -> None:
        base_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml")
        experiment_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM_NOTANGENT.yaml")
        with base_path.open(encoding="utf-8") as handle:
            base = yaml.safe_load(handle)
        with experiment_path.open(encoding="utf-8") as handle:
            experiment = yaml.safe_load(handle)
        expected = copy.deepcopy(base)
        expected["loss"]["lambda_tangent"] = 0.0
        expected["save"]["save_dir"] = "./runs/pilot/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM_NOTANGENT"
        self.assertEqual(experiment, expected)

        route_a = torch.zeros(1, 4, 3, 4, requires_grad=True)
        route_b = torch.full((1, 4, 3, 4), 5.0, requires_grad=True)
        batch = self._batch()
        disabled = DEMambaLoss(
            lambda_edge=0.3,
            lambda_tangent=0.0,
            scnp_weight=0.2,
            scnp_warmup_epochs=10,
            lambda_bpcm=0.025,
            bpcm_start_epoch=60,
            bpcm_ramp_epochs=40,
            bpcm_margin=0.5,
            bpcm_neg_temperature=0.5,
        )
        total_a, items_a = disabled(self._outputs(route_a), batch, epoch=150)
        total_b, _ = disabled(self._outputs(route_b), batch, epoch=150)
        self.assertTrue(torch.equal(total_a, total_b))
        self.assertEqual(items_a["tangent_loss"], 0.0)
        self.assertGreater(items_a["mask_loss"], 0.0)
        self.assertGreater(items_a["edge_loss"], 0.0)
        self.assertGreater(items_a["scnp_loss"], 0.0)
        self.assertGreater(items_a["bpcm_loss"], 0.0)
        total_a.backward()
        self.assertIsNone(route_a.grad)

        enabled_route_a = torch.zeros(1, 4, 3, 4, requires_grad=True)
        enabled_route_b = torch.zeros(1, 4, 3, 4)
        enabled_route_b[:, 0] = 5.0
        enabled_route_b.requires_grad_()
        enabled = DEMambaLoss(lambda_tangent=0.2, scnp_weight=0.2, scnp_warmup_epochs=10)
        enabled_total_a, _ = enabled(self._outputs(enabled_route_a), batch, epoch=150)
        enabled_total_b, _ = enabled(self._outputs(enabled_route_b), batch, epoch=150)
        self.assertFalse(torch.equal(enabled_total_a, enabled_total_b))

    def test_no_tangent_config_preserves_model_capacity_and_off_switches(self) -> None:
        path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM_NOTANGENT.yaml")
        with path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        defaults = {
            name: parameter.default
            for name, parameter in inspect.signature(DEMambaNet.__init__).parameters.items()
            if name != "self" and parameter.default is not inspect.Parameter.empty
        }
        for name, default in defaults.items():
            if name.startswith("use_") and name not in {"use_mamba_ssm", "use_transformer"}:
                self.assertFalse(config["model"].get(name, default), name)
        self.assertTrue(config["model"]["use_mamba_ssm"])
        self.assertTrue(config["train"]["use_model_ema"])
        self.assertEqual(config["train"]["model_ema_decay"], 0.999)
        self.assertTrue(config["train"]["amp"])
        self.assertEqual(config["train"]["amp_dtype"], "fp16")
        self.assertEqual(config["train"]["batch_size"], 8)
        self.assertEqual(config["train"]["lr_scheduler"], "cosine")

        torch.manual_seed(42)
        model = DEMambaNet(base_channels=40, use_mamba_ssm=True, mamba_stages=(3, 4), transformer_stages=(4,))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 7_026_611)
        self.assertTrue(
            all(backend == "mamba_ssm" for backends in model.mamba_backend_info().values() for backend in backends)
        )


if __name__ == "__main__":
    unittest.main()
