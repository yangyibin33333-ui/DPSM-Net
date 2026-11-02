from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import DEMambaNetLite, build_model
from models.de_mambanet_lite import SharedAxialMambaContext, StructureDirectionalBlock
from models.mamba_blocks import SequenceMambaCore


BASE_CONFIG = REPO_ROOT / (
    "configs/experiments/lightweight_s4hv/"
    "C3_C32_S4HV_LR150_DEEPCRACK_NATIVE384x544_EMA_BPCM.yaml"
)
LITE_CONFIG = REPO_ROOT / (
    "configs/experiments/lightweight_demambanet_lite/"
    "DE_MAMBANET_LITE_LR150_DEEPCRACK_NATIVE384x544_EMA_BPCM.yaml"
)
R4_CONFIG = REPO_ROOT / (
    "configs/experiments/lightweight_demambanet_lite/"
    "DE_MAMBANET_LITE_R4_LR150_DEEPCRACK_NATIVE384x544_EMA_BPCM.yaml"
)


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class DEMambaNetLiteTests(unittest.TestCase):
    @staticmethod
    def _small_model(*, use_decoder_route_supervision: bool = True) -> DEMambaNetLite:
        return DEMambaNetLite(
            encoder_widths=(8, 16, 24, 32),
            encoder_depths=(1, 1, 1, 1),
            decoder_channels=16,
            directional_kernel=7,
            mamba_channels=16,
            use_mamba_ssm=False,
            drop_path_rate=0.0,
            use_decoder_route_supervision=use_decoder_route_supervision,
        )

    def test_model_contract_and_native_output_size(self) -> None:
        model = self._small_model().eval()
        with torch.no_grad():
            outputs = model(torch.randn(1, 3, 64, 80))
        self.assertEqual(set(outputs), {"mask", "edge", "reliability", "route_s1", "route_s4"})
        for name in ("mask", "edge", "reliability"):
            self.assertEqual(tuple(outputs[name].shape), (1, 1, 64, 80))
            self.assertTrue(torch.isfinite(outputs[name]).all())
        self.assertEqual(tuple(outputs["route_s1"].shape), (1, 4, 16, 20))
        self.assertEqual(tuple(outputs["route_s4"].shape), (1, 4, 2, 3))

    def test_r4_omits_decoder_route_supervision_and_parameters(self) -> None:
        model = self._small_model(use_decoder_route_supervision=False).eval()
        with torch.no_grad():
            outputs = model(torch.randn(1, 3, 64, 80))
        self.assertEqual(set(outputs), {"mask", "edge", "reliability", "route_s4"})
        self.assertNotIn("route_s1", outputs)
        self.assertFalse(hasattr(model.decoder, "route_head"))
        self.assertFalse(any(name.startswith("decoder.route_head.") for name, _ in model.named_parameters()))

        cores = [module for module in model.modules() if isinstance(module, SequenceMambaCore)]
        self.assertEqual(len(cores), 1)
        self.assertEqual(model.global_context.scan_count, 2)
        self.assertEqual(model.global_context.active_mamba_directions, ("horizontal", "vertical"))

    def test_shared_axial_context_uses_one_core_for_two_traversals(self) -> None:
        context = SharedAxialMambaContext(32, 16, use_mamba_ssm=False).eval()
        cores = [module for module in context.modules() if isinstance(module, SequenceMambaCore)]
        self.assertEqual(len(cores), 1)
        self.assertTrue(context.shared_core)
        self.assertEqual(context.scan_count, 2)
        self.assertEqual(context.active_mamba_directions, ("horizontal", "vertical"))
        with torch.no_grad():
            feature, route_logits = context(torch.randn(2, 32, 5, 7))
        self.assertEqual(tuple(feature.shape), (2, 32, 5, 7))
        self.assertEqual(tuple(route_logits.shape), (2, 4, 5, 7))

    def test_diagonal_routes_cooperate_across_both_axial_scans(self) -> None:
        logits = torch.full((1, 4, 1, 1), -20.0)
        logits[:, 2] = 20.0
        horizontal, vertical = SharedAxialMambaContext.axial_weights(logits)
        self.assertTrue(torch.allclose(horizontal, torch.full_like(horizontal, 0.5), atol=1e-6))
        self.assertTrue(torch.allclose(vertical, torch.full_like(vertical, 0.5), atol=1e-6))
        self.assertTrue(torch.allclose(horizontal + vertical, torch.ones_like(horizontal), atol=1e-6))

    def test_directional_block_uses_cooperative_depthwise_branches(self) -> None:
        block = StructureDirectionalBlock(16, directional_kernel=13)
        self.assertEqual(block.horizontal.groups, block.horizontal.in_channels)
        self.assertEqual(block.vertical.groups, block.vertical.in_channels)
        self.assertEqual(block.oblique.groups, block.oblique.in_channels)
        self.assertEqual(block.oblique.dilation, (3, 3))
        with torch.no_grad():
            output = block.eval()(torch.randn(1, 16, 9, 11))
        self.assertEqual(tuple(output.shape), (1, 16, 9, 11))

    def test_registry_and_backend_contract(self) -> None:
        model = build_model(
            "DEMambaNetLite",
            encoder_widths=(8, 16, 24, 32),
            encoder_depths=(1, 1, 1, 1),
            decoder_channels=16,
            directional_kernel=7,
            mamba_channels=16,
            use_mamba_ssm=False,
        )
        self.assertIsInstance(model, DEMambaNetLite)
        self.assertEqual(model.mamba_backend_info(), {"global_context": ["fallback"]})

    def test_experiment_preserves_data_training_loss_and_evaluation_protocol(self) -> None:
        baseline = _load(BASE_CONFIG)
        candidate = _load(LITE_CONFIG)
        for section in ("data", "train", "loss", "evaluation"):
            self.assertEqual(candidate[section], baseline[section], section)
        self.assertEqual(candidate["model"]["name"], "DEMambaNetLite")
        self.assertNotEqual(candidate["model"], baseline["model"])
        self.assertEqual(
            candidate["save"]["save_dir"],
            "./runs/pilot/lightweight_demambanet_lite/"
            "DE_MAMBANET_LITE_LR150_DEEPCRACK_NATIVE384x544_EMA_BPCM",
        )

    def test_r4_config_changes_only_decoder_supervision_and_save_dir(self) -> None:
        lite = _load(LITE_CONFIG)
        r4 = _load(R4_CONFIG)
        for section in ("data", "train", "loss", "evaluation"):
            self.assertEqual(r4[section], lite[section], section)

        expected_model = dict(lite["model"])
        expected_model["use_decoder_route_supervision"] = False
        self.assertEqual(r4["model"], expected_model)
        self.assertEqual(r4["loss"]["lambda_tangent"], 0.2)
        self.assertNotEqual(r4["save"], lite["save"])
        self.assertEqual(
            r4["save"]["save_dir"],
            "./runs/pilot/lightweight_demambanet_lite/"
            "DE_MAMBANET_LITE_R4_LR150_DEEPCRACK_NATIVE384x544_EMA_BPCM",
        )

    def test_invalid_architecture_arguments_fail_early(self) -> None:
        with self.assertRaisesRegex(ValueError, "four values"):
            DEMambaNetLite(encoder_widths=(16, 32, 64))
        with self.assertRaisesRegex(ValueError, "increase strictly"):
            DEMambaNetLite(encoder_widths=(16, 32, 32, 64))
        with self.assertRaisesRegex(ValueError, "odd integer"):
            DEMambaNetLite(directional_kernel=12)


if __name__ == "__main__":
    unittest.main()
