from __future__ import annotations

import inspect
import unittest
from pathlib import Path

import torch
import yaml

from losses import DEMambaLoss
from models import DEMambaNet


BASE_CONFIG = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml")
C48_CONFIG = Path("configs/experiments/C3_C48_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml")
C40_PARAMETER_COUNT = 7_026_611


class C48CapacityAblationTests(unittest.TestCase):
    @staticmethod
    def _load_config(path: Path) -> dict:
        with path.open(encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    @staticmethod
    def _model_defaults() -> dict[str, object]:
        return {
            name: parameter.default
            for name, parameter in inspect.signature(DEMambaNet.__init__).parameters.items()
            if name != "self" and parameter.default is not inspect.Parameter.empty
        }

    @staticmethod
    def _loss_defaults() -> dict[str, object]:
        return {
            name: parameter.default
            for name, parameter in inspect.signature(DEMambaLoss.__init__).parameters.items()
            if name != "self" and parameter.default is not inspect.Parameter.empty
        }

    @staticmethod
    def _assert_all_mamba_ssm(model: DEMambaNet) -> None:
        backend_info = model.mamba_backend_info()
        assert backend_info, "Expected routed Mamba stages in the C48 experiment."
        for stage, backends in backend_info.items():
            assert len(backends) == 4, f"Expected four Mamba paths at {stage}, got {backends}."
            assert all(backend == "mamba_ssm" for backend in backends), f"Non-official backend at {stage}: {backends}"

    def test_c48_config_has_only_capacity_and_run_directory_differences(self) -> None:
        base = self._load_config(BASE_CONFIG)
        c48 = self._load_config(C48_CONFIG)
        expected = {**base, "model": dict(base["model"]), "save": dict(base["save"])}
        expected["model"]["base_channels"] = 48
        expected["save"]["save_dir"] = "./runs/pilot/C3_C48_LR150_PHOTO_NATIVE384x544_EMA_BPCM"
        self.assertEqual(c48, expected)

        self.assertEqual(base["data"], c48["data"])
        self.assertEqual(base["train"], c48["train"])
        self.assertEqual(base["loss"], c48["loss"])
        self.assertEqual(base["evaluation"], c48["evaluation"])
        for name in ("lambda_bpcm", "bpcm_start_epoch", "bpcm_ramp_epochs", "bpcm_margin", "bpcm_neg_temperature"):
            self.assertEqual(c48["loss"][name], base["loss"][name], name)

        model_defaults = self._model_defaults()
        self.assertTrue(c48["model"]["use_mamba_ssm"])
        self.assertIsNone(c48["model"].get("stage3_mamba_window_size", model_defaults["stage3_mamba_window_size"]))
        self.assertFalse(c48["model"].get("stage3_bidirectional_mamba", model_defaults["stage3_bidirectional_mamba"]))
        self.assertFalse(c48["model"].get("return_training_mask_feature", model_defaults["return_training_mask_feature"]))
        for name, default in model_defaults.items():
            if name.startswith("use_") and name not in {"use_mamba_ssm", "use_transformer"}:
                self.assertFalse(c48["model"].get(name, default), name)

        self.assertFalse((c48["data"].get("scale_jitter") or {}).get("enabled", False))
        loss_defaults = self._loss_defaults()
        for name in ("lambda_tnhnc", "lambda_fbhn", "lambda_ep_hcfc", "lambda_sdf", "lambda_aux_mask", "lambda_aux_edge"):
            self.assertEqual(c48["loss"].get(name, loss_defaults[name]), 0.0, name)

    def test_c48_model_uses_official_mamba_and_preserves_native_output_shapes(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("Official mamba_ssm output-shape audit requires a CUDA host.")
        device = torch.device("cuda")
        torch.manual_seed(42)
        model = DEMambaNet(
            base_channels=48,
            use_mamba_ssm=True,
            mamba_stages=(3, 4),
            transformer_stages=(4,),
        ).to(device).eval()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        print(f"C48 parameters: {parameter_count}")
        self.assertGreater(parameter_count, C40_PARAMETER_COUNT)
        self._assert_all_mamba_ssm(model)

        with torch.no_grad():
            outputs = model(torch.zeros(1, 3, 384, 544, device=device))
        for name in ("mask", "edge", "reliability"):
            self.assertEqual(tuple(outputs[name].shape), (1, 1, 384, 544), name)
            self.assertTrue(torch.isfinite(outputs[name]).all(), name)

    def test_c48_seeded_initialization_is_reproducible_and_differs_in_capacity_from_c40(self) -> None:
        kwargs = {
            "base_channels": 48,
            "use_mamba_ssm": True,
            "mamba_stages": (3, 4),
            "transformer_stages": (4,),
        }
        torch.manual_seed(42)
        first = DEMambaNet(**kwargs)
        torch.manual_seed(42)
        second = DEMambaNet(**kwargs)
        first_parameter_count = sum(parameter.numel() for parameter in first.parameters())
        second_parameter_count = sum(parameter.numel() for parameter in second.parameters())
        print(f"C48 reproducible parameter count: {first_parameter_count}")
        self.assertEqual(first_parameter_count, second_parameter_count)
        for name, parameter in first.state_dict().items():
            self.assertTrue(torch.equal(parameter, second.state_dict()[name]), name)
        self._assert_all_mamba_ssm(first)

        torch.manual_seed(42)
        c40 = DEMambaNet(base_channels=40, use_mamba_ssm=True, mamba_stages=(3, 4), transformer_stages=(4,))
        self.assertEqual(sum(parameter.numel() for parameter in c40.parameters()), C40_PARAMETER_COUNT)
        self.assertNotEqual(first_parameter_count, C40_PARAMETER_COUNT)
        self.assertGreater(first_parameter_count, C40_PARAMETER_COUNT)


if __name__ == "__main__":
    unittest.main()
