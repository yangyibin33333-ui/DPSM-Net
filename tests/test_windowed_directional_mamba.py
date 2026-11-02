from __future__ import annotations

import unittest
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from models import DEMambaNet
from models.mamba_blocks import DirectionalMambaPath, WindowedDirectionalMambaPath


class _WindowMeanMockPath(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + x.mean(dim=(-2, -1), keepdim=True)


class WindowedDirectionalMambaTests(unittest.TestCase):
    def test_stage3_windowed_path_round_trips_and_cannot_mix_across_windows(self) -> None:
        path = WindowedDirectionalMambaPath(dim=3, direction="horizontal", window_size=8, use_mamba_ssm=False)
        path.path = nn.Identity()
        feature = torch.randn(2, 3, 16, 24)
        self.assertTrue(torch.equal(path(feature), feature))

        path.path = _WindowMeanMockPath()
        reference = torch.zeros(1, 3, 16, 16)
        changed = reference.clone()
        changed[:, :, :8, :8] = 2.0
        reference_output = path(reference)
        changed_output = path(changed)
        self.assertFalse(torch.equal(changed_output[:, :, :8, :8], reference_output[:, :, :8, :8]))
        outside_changed_window = changed_output - reference_output
        outside_changed_window[:, :, :8, :8] = 0.0
        self.assertTrue(torch.equal(outside_changed_window, torch.zeros_like(outside_changed_window)))
        with self.assertRaisesRegex(ValueError, "divisible"):
            path(torch.randn(1, 3, 15, 16))

        baseline = DEMambaNet(base_channels=8, use_mamba_ssm=False, transformer_stages=[]).eval()
        windowed = DEMambaNet(
            base_channels=8,
            use_mamba_ssm=False,
            transformer_stages=[],
            stage3_mamba_window_size=8,
        ).eval()
        self.assertTrue(all(isinstance(path, DirectionalMambaPath) for path in baseline.stage3.route_mamba.paths))
        self.assertTrue(all(isinstance(path, DirectionalMambaPath) for path in baseline.stage4.route_mamba.paths))
        self.assertTrue(all(isinstance(path, WindowedDirectionalMambaPath) for path in windowed.stage3.route_mamba.paths))
        self.assertTrue(all(isinstance(path, DirectionalMambaPath) for path in windowed.stage4.route_mamba.paths))
        self.assertEqual(windowed.mamba_backend_info(), {"stage3": ["fallback"] * 4, "stage4": ["fallback"] * 4})

        base_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml")
        experiment_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM_S3WDM8.yaml")
        with base_path.open(encoding="utf-8") as handle:
            base_cfg = yaml.safe_load(handle)
        with experiment_path.open(encoding="utf-8") as handle:
            experiment_cfg = yaml.safe_load(handle)
        self.assertEqual(experiment_cfg["data"], base_cfg["data"])
        self.assertEqual(experiment_cfg["train"], base_cfg["train"])
        self.assertEqual(experiment_cfg["evaluation"], base_cfg["evaluation"])
        for name, value in base_cfg["loss"].items():
            self.assertEqual(experiment_cfg["loss"][name], value)
        self.assertEqual(experiment_cfg["loss"]["lambda_fbhn"], 0.0)
        self.assertEqual(experiment_cfg["model"]["stage3_mamba_window_size"], 8)
        self.assertEqual(experiment_cfg["save"]["save_dir"], "./runs/pilot/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM_S3WDM8")
        enabled_optional_modules = {
            name
            for name, value in experiment_cfg["model"].items()
            if name.startswith("use_") and value and name != "use_mamba_ssm"
        }
        self.assertEqual(enabled_optional_modules, set())

        base_model_cfg = dict(base_cfg["model"])
        base_model_cfg.pop("name")
        experiment_model_cfg = dict(experiment_cfg["model"])
        experiment_model_cfg.pop("name")
        baseline_c40 = DEMambaNet(**base_model_cfg)
        windowed_c40 = DEMambaNet(**experiment_model_cfg)
        self.assertEqual(sum(parameter.numel() for parameter in baseline_c40.parameters()), 7026611)
        self.assertEqual(sum(parameter.numel() for parameter in windowed_c40.parameters()), 7026611)
        with self.assertRaisesRegex(ValueError, "stage3_mamba_window_size"):
            DEMambaNet(base_channels=8, use_mamba_ssm=False, stage3_mamba_window_size=0)


if __name__ == "__main__":
    unittest.main()
