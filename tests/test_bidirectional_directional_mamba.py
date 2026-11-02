from __future__ import annotations

import unittest
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from models import DEMambaNet
from models.mamba_blocks import (
    BidirectionalDirectionalMambaPath,
    DirectionalMambaPath,
    TangentRoutedMambaBlock,
)


class _CausalCumsumCore(nn.Module):
    """Deterministic left-to-right sequence core for directionality checks."""

    backend = "mock"

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return torch.cumsum(sequence, dim=1)


class BidirectionalDirectionalMambaTests(unittest.TestCase):
    def test_bidirectional_path_reuses_one_core_and_recovers_opposite_direction_context(self) -> None:
        feature = torch.zeros(1, 1, 1, 5)
        feature[..., -1] = 1.0

        causal_path = DirectionalMambaPath(dim=1, direction="horizontal", use_mamba_ssm=False)
        causal_path.core = _CausalCumsumCore()
        causal_output = causal_path(feature)
        self.assertEqual(float(causal_output[..., 0]), 0.0)

        bidirectional_path = BidirectionalDirectionalMambaPath(dim=1, direction="horizontal", use_mamba_ssm=False)
        mock_core = _CausalCumsumCore()
        bidirectional_path.path.core = mock_core
        bidirectional_output = bidirectional_path(feature)
        self.assertEqual(
            len([module for module in bidirectional_path.modules() if isinstance(module, DirectionalMambaPath)]),
            1,
        )
        self.assertEqual(mock_core.calls, 2)
        self.assertEqual(bidirectional_path.backend, bidirectional_path.path.backend)
        self.assertEqual(tuple(bidirectional_output.shape), tuple(feature.shape))
        self.assertGreater(float(bidirectional_output[..., 0]), 0.0)

        for direction, expected_dims in {
            "horizontal": (-1,),
            "vertical": (-2,),
            "diagonal": (-2, -1),
            "anti_diagonal": (-2, -1),
        }.items():
            self.assertEqual(
                BidirectionalDirectionalMambaPath(4, direction, use_mamba_ssm=False)._reverse_dims(),
                expected_dims,
            )
        invalid = BidirectionalDirectionalMambaPath(4, "invalid", use_mamba_ssm=False)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            invalid._reverse_dims()

        fallback_model = DEMambaNet(
            base_channels=8,
            use_mamba_ssm=False,
            transformer_stages=[],
            stage3_bidirectional_mamba=True,
        ).eval()
        self.assertTrue(
            all(isinstance(path, BidirectionalDirectionalMambaPath) for path in fallback_model.stage3.route_mamba.paths)
        )
        self.assertTrue(
            all(isinstance(path, DirectionalMambaPath) for path in fallback_model.stage4.route_mamba.paths)
        )
        self.assertEqual(fallback_model.mamba_backend_info(), {"stage3": ["fallback"] * 4, "stage4": ["fallback"] * 4})
        with self.assertRaisesRegex(ValueError, "Windowed and bidirectional"):
            TangentRoutedMambaBlock(8, use_mamba_ssm=False, window_size=8, bidirectional=True)
        with self.assertRaisesRegex(ValueError, "stage3_bidirectional_mamba"):
            DEMambaNet(
                base_channels=8,
                use_mamba_ssm=False,
                mamba_stages=(4,),
                transformer_stages=[],
                stage3_bidirectional_mamba=True,
            )

        base_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml")
        experiment_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM_S3BIMAMBA.yaml")
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
        self.assertIsNone(experiment_cfg["model"]["stage3_mamba_window_size"])
        self.assertTrue(experiment_cfg["model"]["stage3_bidirectional_mamba"])
        self.assertEqual(
            experiment_cfg["save"]["save_dir"],
            "./runs/pilot/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM_S3BIMAMBA",
        )
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
        torch.manual_seed(42)
        native_c40 = DEMambaNet(**base_model_cfg)
        torch.manual_seed(42)
        bidirectional_c40 = DEMambaNet(**experiment_model_cfg)
        self.assertEqual(sum(parameter.numel() for parameter in native_c40.parameters()), 7026611)
        self.assertEqual(sum(parameter.numel() for parameter in bidirectional_c40.parameters()), 7026611)
        native_parameters = list(native_c40.parameters())
        bidirectional_parameters = list(bidirectional_c40.parameters())
        self.assertEqual(len(native_parameters), len(bidirectional_parameters))
        for index, (native_parameter, bidirectional_parameter) in enumerate(
            zip(native_parameters, bidirectional_parameters, strict=True)
        ):
            self.assertTrue(torch.equal(native_parameter, bidirectional_parameter), index)
        self.assertTrue(
            all(isinstance(path, BidirectionalDirectionalMambaPath) for path in bidirectional_c40.stage3.route_mamba.paths)
        )
        self.assertTrue(
            all(isinstance(path, DirectionalMambaPath) for path in bidirectional_c40.stage4.route_mamba.paths)
        )


if __name__ == "__main__":
    unittest.main()
