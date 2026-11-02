from __future__ import annotations

import copy
import unittest
from pathlib import Path

import torch
import yaml

from models import DEMambaNet, HRTopoMambaNet, build_model
from models.hr_topomamba import DetailToSemanticExchange, SemanticToDetailExchange


class HRTopoMambaTests(unittest.TestCase):
    @staticmethod
    def _load_experiment_config() -> dict:
        with Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM_HRTOPO.yaml").open(
            encoding="utf-8"
        ) as handle:
            return yaml.safe_load(handle)

    @classmethod
    def _model_from_config(cls) -> HRTopoMambaNet:
        config = cls._load_experiment_config()
        model_cfg = dict(config["model"])
        model_name = model_cfg.pop("name")
        model = build_model(model_name, **model_cfg)
        if not isinstance(model, HRTopoMambaNet):
            raise AssertionError("HRTopo configuration did not select HRTopoMambaNet.")
        return model

    def test_zero_initialized_exchanges_are_exact_identity(self) -> None:
        torch.manual_seed(42)
        detail = torch.randn(1, 80, 16, 20)
        semantic3 = torch.randn(1, 160, 8, 10)
        semantic4 = torch.randn(1, 320, 4, 5)
        detail_to_semantic = DetailToSemanticExchange(80, 160).eval()
        semantic3_to_detail = SemanticToDetailExchange(80, 160).eval()
        semantic4_to_detail = SemanticToDetailExchange(80, 320).eval()
        self.assertEqual(detail_to_semantic.alpha.item(), 0.0)
        self.assertEqual(semantic3_to_detail.alpha.item(), 0.0)
        self.assertEqual(semantic4_to_detail.alpha.item(), 0.0)
        with torch.no_grad():
            self.assertTrue(torch.equal(detail_to_semantic(detail, semantic3), semantic3))
            self.assertTrue(torch.equal(semantic3_to_detail(detail, semantic3), detail))
            self.assertTrue(torch.equal(semantic4_to_detail(detail, semantic4), detail))
        self.assertEqual(sum(parameter.numel() for parameter in detail_to_semantic.parameters()), 13_521)
        self.assertEqual(sum(parameter.numel() for parameter in semantic3_to_detail.parameters()), 17_721)
        self.assertEqual(sum(parameter.numel() for parameter in semantic4_to_detail.parameters()), 30_521)

        # Exchange creation follows the Native core, so zero alpha also leaves
        # all shared seeded parameters and forward outputs exactly unchanged.
        kwargs = {"base_channels": 8, "use_mamba_ssm": False, "transformer_stages": []}
        torch.manual_seed(77)
        native = DEMambaNet(**kwargs).eval()
        torch.manual_seed(77)
        hr_topo = HRTopoMambaNet(**kwargs).eval()
        native_state = native.state_dict()
        hr_topo_state = hr_topo.state_dict()
        for name, tensor in native_state.items():
            self.assertTrue(torch.equal(tensor, hr_topo_state[name]), name)
        image = torch.randn(1, 3, 64, 80)
        with torch.no_grad():
            native_outputs = native(image)
            hr_topo_outputs = hr_topo(image)
        self.assertEqual(native_outputs.keys(), hr_topo_outputs.keys())
        for name in native_outputs:
            self.assertTrue(torch.equal(native_outputs[name], hr_topo_outputs[name]), name)

    def test_forward_shape_contract_at_native_resolution(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("Native-resolution mamba_ssm forward requires CUDA.")
        device = torch.device("cuda")
        model = self._model_from_config().to(device).eval()
        with torch.no_grad():
            outputs = model(torch.randn(1, 3, 384, 544, device=device))
        self.assertEqual(outputs["mask"].shape, (1, 1, 384, 544))
        self.assertEqual(outputs["edge"].shape, (1, 1, 384, 544))
        self.assertEqual(outputs["reliability"].shape, (1, 1, 384, 544))
        self.assertEqual(outputs["route_s3"].shape, (1, 4, 96, 136))
        self.assertEqual(outputs["route_s4"].shape, (1, 4, 48, 68))
        self.assertTrue(all(torch.isfinite(value).all() for value in outputs.values()))

    def test_uses_mamba_ssm_for_stage3_and_stage4(self) -> None:
        model = self._model_from_config()
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 7_088_374)
        backend_info = model.mamba_backend_info()
        self.assertEqual(backend_info["stage3"], ["mamba_ssm"] * 4)
        self.assertEqual(backend_info["stage4"], ["mamba_ssm"] * 4)

    def test_native_model_and_baseline_config_are_unchanged(self) -> None:
        base_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml")
        experiment_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM_HRTOPO.yaml")
        with base_path.open(encoding="utf-8") as handle:
            baseline = yaml.safe_load(handle)
        with experiment_path.open(encoding="utf-8") as handle:
            experiment = yaml.safe_load(handle)
        self.assertEqual(baseline["model"]["name"], "DEMambaNet")
        self.assertEqual(baseline["loss"]["lambda_bpcm"], 0.025)
        self.assertEqual(baseline["train"]["use_model_ema"], True)
        self.assertEqual(baseline["train"]["model_ema_decay"], 0.999)
        self.assertIsInstance(DEMambaNet(base_channels=8, use_mamba_ssm=False, transformer_stages=[]), DEMambaNet)
        baseline_model_cfg = dict(baseline["model"])
        baseline_model_name = baseline_model_cfg.pop("name")
        self.assertIsInstance(build_model(baseline_model_name, **baseline_model_cfg), DEMambaNet)

        expected = copy.deepcopy(baseline)
        expected["model"]["name"] = "HRTopoMambaNet"
        expected["model"]["hr_topo"] = {
            "detail_to_semantic": True,
            "semantic_to_detail_from_stage3": True,
            "semantic_to_detail_from_stage4": True,
            "zero_initialized_residual_scales": True,
        }
        expected["save"]["save_dir"] = "./runs/pilot/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM_HRTOPO"
        self.assertEqual(experiment, expected)
        self.assertTrue(experiment["model"]["use_mamba_ssm"])
        self.assertEqual(experiment["model"]["mamba_stages"], [3, 4])
        self.assertEqual(experiment["model"]["transformer_stages"], [4])
        self.assertEqual(experiment["data"]["img_size"], [384, 544])
        self.assertEqual(experiment["train"]["batch_size"], 8)
        self.assertEqual(experiment["train"]["amp_dtype"], "fp16")
        self.assertEqual(experiment["train"]["lr_scheduler"], "cosine")


if __name__ == "__main__":
    unittest.main()
