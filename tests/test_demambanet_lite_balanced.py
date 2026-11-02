from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import DEMambaNetLiteBalanced, build_model
from models.de_mambanet_lite_balanced import ResidualBaselineGatedScaleFusion
from models.mamba_blocks import SequenceMambaCore


LITE_CONFIG = REPO_ROOT / (
    "configs/experiments/lightweight_demambanet_lite/"
    "DE_MAMBANET_LITE_LR150_DEEPCRACK_NATIVE384x544_EMA_BPCM.yaml"
)
BALANCED_CONFIG = REPO_ROOT / (
    "configs/experiments/lightweight_demambanet_lite_balanced/"
    "DE_MAMBANET_LITE_BALANCED_LR150_DEEPCRACK_NATIVE384x544_EMA_BPCM.yaml"
)


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class DEMambaNetLiteBalancedTests(unittest.TestCase):
    @staticmethod
    def _small_model() -> DEMambaNetLiteBalanced:
        return DEMambaNetLiteBalanced(
            encoder_widths=(8, 16, 24, 32),
            encoder_depths=(1, 1, 1, 1),
            decoder_channels=16,
            directional_kernel=7,
            mamba_channels=16,
            use_mamba_ssm=False,
            drop_path_rate=0.0,
        )

    def test_output_contract_and_half_upsample_geometry(self) -> None:
        model = self._small_model().eval()
        captured: dict[str, tuple[int, ...]] = {}

        def capture_half(_: torch.nn.Module, __: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            captured["half"] = tuple(output.shape)

        handle = model.decoder.half_upsample.register_forward_hook(capture_half)
        try:
            with torch.no_grad():
                outputs = model(torch.randn(1, 3, 64, 80))
        finally:
            handle.remove()

        self.assertEqual(set(outputs), {"mask", "edge", "reliability", "route_s1", "route_s4"})
        for name in ("mask", "edge", "reliability"):
            self.assertEqual(tuple(outputs[name].shape), (1, 1, 64, 80))
            self.assertTrue(torch.isfinite(outputs[name]).all())
        self.assertEqual(tuple(outputs["route_s1"].shape), (1, 4, 16, 20))
        self.assertEqual(tuple(outputs["route_s4"].shape), (1, 4, 2, 3))
        self.assertEqual(captured["half"], (1, 8, 32, 40))

    def test_one_shared_mamba_core_and_two_axial_traversals(self) -> None:
        model = self._small_model()
        cores = [module for module in model.modules() if isinstance(module, SequenceMambaCore)]
        self.assertEqual(len(cores), 1)
        self.assertTrue(model.global_context.shared_core)
        self.assertEqual(model.global_context.scan_count, 2)
        self.assertEqual(model.global_context.active_mamba_directions, ("horizontal", "vertical"))

    def test_residual_gate_is_zero_initialized_and_preserves_both_inputs(self) -> None:
        model = self._small_model().eval()
        fusions = [
            model.decoder.fuse3,
            model.decoder.fuse2,
            model.decoder.fuse1,
            model.decoder.half_fusion,
        ]
        for fusion in fusions:
            self.assertIsInstance(fusion, ResidualBaselineGatedScaleFusion)
            self.assertTrue(torch.count_nonzero(fusion.gate[-1].weight) == 0)
            self.assertTrue(torch.count_nonzero(fusion.gate[-1].bias) == 0)

        high = torch.randn(1, 16, 5, 7)
        lateral = torch.randn_like(high)
        with torch.no_grad():
            fused = model.decoder.fuse3.residual_fusion(high, lateral)
        self.assertTrue(torch.allclose(fused, high + lateral))

    def test_mask_refinement_is_private_to_mask_head(self) -> None:
        model = self._small_model().eval()
        head_inputs: dict[str, int] = {}

        def capture_input(name: str):
            def hook(_: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
                head_inputs[name] = inputs[0].data_ptr()

            return hook

        handles = [
            model.decoder.edge_head.register_forward_pre_hook(capture_input("edge")),
            model.decoder.reliability_head.register_forward_pre_hook(capture_input("reliability")),
            model.decoder.mask_head.register_forward_pre_hook(capture_input("mask")),
        ]
        try:
            with torch.no_grad():
                model(torch.randn(1, 3, 64, 80))
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(head_inputs["edge"], head_inputs["reliability"])
        self.assertNotEqual(head_inputs["mask"], head_inputs["edge"])
        self.assertEqual(len(model.decoder.mask_refine), 2)

    def test_registry_contract(self) -> None:
        model = build_model(
            "DEMambaNetLiteBalanced",
            encoder_widths=(8, 16, 24, 32),
            encoder_depths=(1, 1, 1, 1),
            decoder_channels=16,
            directional_kernel=7,
            mamba_channels=16,
            use_mamba_ssm=False,
        )
        self.assertIsInstance(model, DEMambaNetLiteBalanced)
        self.assertEqual(model.mamba_backend_info(), {"global_context": ["fallback"]})

    def test_config_preserves_lite_protocol(self) -> None:
        lite = _load(LITE_CONFIG)
        balanced = _load(BALANCED_CONFIG)
        for section in ("data", "train", "loss", "evaluation"):
            self.assertEqual(balanced[section], lite[section], section)

        expected_model = dict(lite["model"])
        expected_model["name"] = "DEMambaNetLiteBalanced"
        self.assertEqual(balanced["model"], expected_model)
        self.assertNotEqual(balanced["save"], lite["save"])
        self.assertEqual(
            balanced["save"]["save_dir"],
            "./runs/pilot/lightweight_demambanet_lite_balanced/"
            "DE_MAMBANET_LITE_BALANCED_LR150_DEEPCRACK_NATIVE384x544_EMA_BPCM",
        )


if __name__ == "__main__":
    unittest.main()
