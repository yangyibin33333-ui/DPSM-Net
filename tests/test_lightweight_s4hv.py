from __future__ import annotations

import copy
import unittest
from pathlib import Path

import torch
import yaml

from models import DEMambaNet
from models.mamba_blocks import TangentRoutedMambaBlock


BASE_CONFIG = Path("configs/experiments/lightweight_s4mamba/C3_C32_S4MAMBA_LR150_DEEPCRACK_NATIVE384x544_EMA_BPCM.yaml")
HV_CONFIG = Path("configs/experiments/lightweight_s4hv/C3_C32_S4HV_LR150_DEEPCRACK_NATIVE384x544_EMA_BPCM.yaml")


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class LightweightS4HVTests(unittest.TestCase):
    def test_s4hv_instantiates_only_hv_mamba_and_preserves_four_route_bins(self) -> None:
        block = TangentRoutedMambaBlock(
            16,
            use_mamba_ssm=False,
            active_mamba_directions=["horizontal", "vertical"],
        ).eval()
        self.assertEqual(block.active_mamba_directions, ("horizontal", "vertical"))
        self.assertEqual(block.active_mamba_path_count, 2)
        self.assertEqual(len(block.paths), 2)
        self.assertEqual(tuple(path.direction for path in block.paths), ("horizontal", "vertical"))
        self.assertEqual(set(block.local_compensations or {}), {"diagonal", "anti_diagonal"})
        with torch.no_grad():
            routed, route_logits = block(torch.randn(1, 16, 12, 18))
        self.assertEqual(tuple(routed.shape), (1, 16, 12, 18))
        self.assertEqual(tuple(route_logits.shape), (1, 4, 12, 18))
        self.assertEqual(block.last_executed_local_compensation_directions, ("diagonal", "anti_diagonal"))

        with self.assertRaisesRegex(ValueError, "active_mamba_directions"):
            TangentRoutedMambaBlock(16, use_mamba_ssm=False, active_mamba_directions=[])
        with self.assertRaisesRegex(ValueError, "active_mamba_directions"):
            TangentRoutedMambaBlock(16, use_mamba_ssm=False, active_mamba_directions=["horizontal", "invalid"])

        legacy = TangentRoutedMambaBlock(16, use_mamba_ssm=False).eval()
        self.assertEqual(legacy.active_mamba_path_count, 4)
        self.assertEqual(legacy.active_mamba_directions, ("horizontal", "vertical", "diagonal", "anti_diagonal"))
        self.assertIsNone(legacy.local_compensations)

    def test_s4hv_deepcrack_shape_contract_and_config_is_single_variable(self) -> None:
        baseline = _load_config(BASE_CONFIG)
        candidate = _load_config(HV_CONFIG)
        expected = copy.deepcopy(baseline)
        expected["model"]["stage4_active_mamba_directions"] = ["horizontal", "vertical"]
        expected["save"]["save_dir"] = "./runs/pilot/lightweight_s4hv/C3_C32_S4HV_LR150_DEEPCRACK_NATIVE384x544_EMA_BPCM"
        self.assertEqual(candidate, expected)

        model_cfg = dict(candidate["model"])
        model_cfg.pop("name")
        model_cfg["use_mamba_ssm"] = False
        model = DEMambaNet(**model_cfg).eval()
        self.assertEqual(model.mamba_backend_info(), {"stage4": ["fallback", "fallback"]})
        self.assertEqual(model.stage4.route_mamba.active_mamba_path_count, 2)
        with torch.no_grad():
            outputs = model(torch.randn(1, 3, 384, 544))
        for name in ("mask", "edge", "reliability"):
            self.assertIn(name, outputs)
            self.assertEqual(tuple(outputs[name].shape), (1, 1, 384, 544))
            self.assertTrue(torch.isfinite(outputs[name]).all())
        self.assertIn("route_s4", outputs)
        self.assertEqual(tuple(outputs["route_s4"].shape), (1, 4, 48, 68))
        self.assertTrue(torch.isfinite(outputs["route_s4"]).all())


if __name__ == "__main__":
    unittest.main()
