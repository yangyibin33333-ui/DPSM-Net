from __future__ import annotations

import copy
import unittest
from pathlib import Path

import torch
import yaml

from losses import DEMambaLoss
from models import DEMambaNet, HRConvNeXtFPN, HRTopoMambaNet, build_model
from models.hrconvnext_fpn import LayerNorm2d


BASELINE_CONFIG = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml")
EXPERIMENT_CONFIG = Path("configs/experiments/HRCONVNEXTFPN_NATIVE384x544_EMA_BPCM.yaml")
HRTOPO_CONFIG = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM_HRTOPO.yaml")


class HRConvNeXtFPNTests(unittest.TestCase):
    @staticmethod
    def _load_config(path: Path) -> dict:
        with path.open(encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    @classmethod
    def _model_from_config(cls) -> HRConvNeXtFPN:
        config = cls._load_config(EXPERIMENT_CONFIG)
        model_config = dict(config["model"])
        model_name = str(model_config.pop("name"))
        model = build_model(model_name, **model_config)
        if not isinstance(model, HRConvNeXtFPN):
            raise AssertionError("HRConvNeXtFPN experiment did not build HRConvNeXtFPN.")
        return model

    def test_model_factory_and_small_cpu_forward_contract(self) -> None:
        model = self._model_from_config().eval()
        with torch.no_grad():
            outputs = model(torch.randn(1, 3, 64, 80))
        self.assertEqual(set(outputs), {"mask", "edge", "reliability", "route_s1"})
        self.assertEqual(outputs["mask"].shape, (1, 1, 64, 80))
        self.assertEqual(outputs["edge"].shape, (1, 1, 64, 80))
        self.assertEqual(outputs["reliability"].shape, (1, 1, 64, 80))
        self.assertEqual(outputs["route_s1"].shape, (1, 4, 32, 40))
        self.assertTrue(all(torch.isfinite(output).all() for output in outputs.values()))
        self.assertEqual(model.mamba_backend_info(), {})
        self.assertTrue(model.is_mamba_free)

    def test_unknown_model_name_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown model.name 'not-a-model'"):
            build_model("not-a-model")

    def test_native_resolution_shape_contract(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("Native-resolution HRConvNeXtFPN contract is checked on CUDA when available.")
        model = self._model_from_config().to("cuda").eval()
        with torch.no_grad():
            outputs = model(torch.randn(1, 3, 384, 544, device="cuda"))
        self.assertEqual(outputs["mask"].shape, (1, 1, 384, 544))
        self.assertEqual(outputs["edge"].shape, (1, 1, 384, 544))
        self.assertEqual(outputs["reliability"].shape, (1, 1, 384, 544))
        self.assertEqual(outputs["route_s1"].shape, (1, 4, 192, 272))
        self.assertTrue(all(torch.isfinite(output).all() for output in outputs.values()))

    def test_layer_norm_and_fused_adamw_use_compatible_layouts(self) -> None:
        layer_norm = LayerNorm2d(56)
        self.assertTrue(layer_norm(torch.randn(1, 56, 16, 20)).is_contiguous())
        if not torch.cuda.is_available():
            self.skipTest("Fused AdamW layout regression is checked on CUDA when available.")
        model = self._model_from_config().to("cuda").train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-4, fused=True)
        outputs = model(torch.randn(1, 3, 64, 80, device="cuda"))
        loss = sum(output.float().square().mean() for output in outputs.values())
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertEqual(parameter.stride(), parameter.grad.stride())
        optimizer.step()

    def test_existing_loss_contract_supports_backward(self) -> None:
        torch.manual_seed(5)
        config = self._load_config(EXPERIMENT_CONFIG)
        model = self._model_from_config().train()
        image = torch.randn(1, 3, 64, 80)
        mask = torch.zeros(1, 1, 64, 80)
        mask[..., 20:44, 30:50] = 1.0
        edge = torch.zeros_like(mask)
        edge[..., 20:44, 30] = 1.0
        tangent_gt = torch.zeros(1, 4, 64, 80)
        tangent_gt[:, 0] = 1.0
        batch = {
            "mask": mask,
            "edge_gt": edge,
            "tangent_gt": tangent_gt,
            "tangent_valid": torch.ones_like(mask),
        }
        criterion = DEMambaLoss(**config["loss"])
        total_loss, items = criterion(model(image), batch, epoch=100)
        self.assertTrue(torch.isfinite(total_loss))
        self.assertTrue(torch.isfinite(torch.tensor(items["mask_loss"])))
        self.assertTrue(torch.isfinite(torch.tensor(items["edge_loss"])))
        self.assertTrue(torch.isfinite(torch.tensor(items["tangent_loss"])))
        self.assertTrue(torch.isfinite(torch.tensor(items["scnp_loss"])))
        self.assertTrue(torch.isfinite(torch.tensor(items["bpcm_loss"])))
        total_loss.backward()
        self.assertIsNotNone(model.stem.weight.grad)
        self.assertTrue(torch.isfinite(model.stem.weight.grad).all())
        self.assertNotEqual(float(model.stem.weight.grad.abs().sum()), 0.0)

    def test_parameter_budget_and_standalone_implementation(self) -> None:
        model = self._model_from_config()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        self.assertEqual(parameter_count, 6_998_263)
        self.assertGreaterEqual(parameter_count, 6_675_280)
        self.assertLessEqual(parameter_count, 7_377_942)
        source = Path("models/hrconvnext_fpn.py").read_text(encoding="utf-8")
        self.assertNotIn("from .blocks", source)
        self.assertNotIn("from .decoder", source)

    def test_config_preserves_all_non_model_baseline_settings(self) -> None:
        baseline = self._load_config(BASELINE_CONFIG)
        experiment = self._load_config(EXPERIMENT_CONFIG)
        self.assertEqual(baseline["model"]["name"], "DEMambaNet")
        self.assertEqual(experiment["model"], {
            "name": "HRConvNeXtFPN",
            "in_channels": 3,
            "num_classes": 1,
            "stage_dims": [56, 112, 224, 448],
            "stage_depths": [1, 2, 6, 2],
            "fpn_channels": 104,
        })
        for section in ("data", "train", "loss", "evaluation"):
            self.assertEqual(experiment[section], baseline[section], section)
        expected = copy.deepcopy(baseline)
        expected["model"] = experiment["model"]
        expected["save"]["save_dir"] = "./runs/pilot/HRCONVNEXTFPN_NATIVE384x544_EMA_BPCM"
        self.assertEqual(experiment, expected)

    def test_test_py_uses_the_shared_factory_for_each_registered_experiment(self) -> None:
        test_source = Path("test.py").read_text(encoding="utf-8")
        self.assertIn("from models import build_model", test_source)
        self.assertIn('model_name = str(model_cfg.pop("name", "DEMambaNet"))', test_source)
        self.assertIn("model = build_model(model_name, **model_cfg).to(device)", test_source)
        self.assertNotIn("DEMambaNet(**model_cfg)", test_source)

        cases = (
            (BASELINE_CONFIG, DEMambaNet),
            (HRTOPO_CONFIG, HRTopoMambaNet),
            (EXPERIMENT_CONFIG, HRConvNeXtFPN),
        )
        for config_path, expected_type in cases:
            config = self._load_config(config_path)
            model_config = dict(config["model"])
            model_name = str(model_config.pop("name", "DEMambaNet"))
            self.assertIsInstance(build_model(model_name, **model_config), expected_type)


if __name__ == "__main__":
    unittest.main()
