from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import DEMambaNetLiteBalancedPlus, build_model
from models.de_mambanet_lite_balanced_plus import PartialDirectionalContinuityBlock
from models.mamba_blocks import SequenceMambaCore


BALANCED_CONFIG = REPO_ROOT / (
    "configs/experiments/lightweight_demambanet_lite_balanced/"
    "DE_MAMBANET_LITE_BALANCED_LR150_DEEPCRACK_NATIVE384x544_EMA_BPCM.yaml"
)
PLUS_CONFIG = REPO_ROOT / (
    "configs/experiments/lightweight_demambanet_lite_balanced_plus/"
    "DE_MAMBANET_LITE_BALANCED_PLUS_LR150_DEEPCRACK_NATIVE384x544_EMA_BPCM.yaml"
)


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class DEMambaNetLiteBalancedPlusTests(unittest.TestCase):
    @staticmethod
    def _small_model() -> DEMambaNetLiteBalancedPlus:
        return DEMambaNetLiteBalancedPlus(
            encoder_widths=(8, 16, 24, 32),
            encoder_depths=(1, 1, 1, 1),
            decoder_channels=16,
            directional_kernel=7,
            mamba_channels=16,
            use_mamba_ssm=False,
            drop_path_rate=0.0,
        )

    def test_output_contract_route_scales_and_single_mamba_core(self) -> None:
        model = self._small_model().eval()
        with torch.no_grad():
            outputs = model(torch.randn(1, 3, 64, 80))
        self.assertEqual(set(outputs), {"mask", "edge", "reliability", "route_s1", "route_s4"})
        for name in ("mask", "edge", "reliability"):
            self.assertEqual(tuple(outputs[name].shape), (1, 1, 64, 80))
            self.assertTrue(torch.isfinite(outputs[name]).all())
        self.assertEqual(tuple(outputs["route_s1"].shape), (1, 4, 16, 20))
        self.assertEqual(tuple(outputs["route_s4"].shape), (1, 4, 2, 3))

        cores = [module for module in model.modules() if isinstance(module, SequenceMambaCore)]
        self.assertEqual(len(cores), 1)
        self.assertEqual(model.global_context.scan_count, 2)
        self.assertEqual(model.global_context.active_mamba_directions, ("horizontal", "vertical"))

    def test_continuity_blocks_process_half_channels_and_preserve_bypass(self) -> None:
        model = self._small_model().eval()
        blocks = list(model.decoder.mask_continuity)
        self.assertEqual(len(blocks), 2)
        for block in blocks:
            self.assertIsInstance(block, PartialDirectionalContinuityBlock)
            self.assertEqual(block.channels, 48)
            self.assertEqual(block.active_channels, 24)
            self.assertEqual(block.bypass_channels, 24)
            for convolution in (block.local, block.horizontal, block.vertical, block.dilated):
                self.assertEqual(convolution.in_channels, 24)
                self.assertEqual(convolution.out_channels, 24)
                self.assertEqual(convolution.groups, 24)
            self.assertEqual(block.local.kernel_size, (3, 3))
            self.assertEqual(block.horizontal.kernel_size, (1, 9))
            self.assertEqual(block.vertical.kernel_size, (9, 1))
            self.assertEqual(block.dilated.kernel_size, (3, 3))
            self.assertEqual(block.dilated.dilation, (3, 3))

            sample = torch.randn(1, 48, 7, 9)
            with torch.no_grad():
                branch_features = block.branch_features(sample)
                output = block(sample)
            self.assertTrue(torch.equal(branch_features[:, 24:], sample[:, 24:]))
            self.assertTrue(torch.equal(output, sample))

    def test_zero_initialized_continuity_and_phase_residuals(self) -> None:
        model = self._small_model()
        for block in model.decoder.mask_continuity:
            self.assertEqual(torch.count_nonzero(block.residual_projection.weight).item(), 0)
            self.assertEqual(torch.count_nonzero(block.residual_projection.bias).item(), 0)
        phase_projection = model.decoder.phase_residual_head.projection
        self.assertEqual(torch.count_nonzero(phase_projection.weight).item(), 0)
        self.assertEqual(torch.count_nonzero(phase_projection.bias).item(), 0)

    def test_initial_mask_equals_bilinear_half_logits(self) -> None:
        model = self._small_model().eval()
        captured: dict[str, torch.Tensor] = {}

        def capture_half(_: torch.nn.Module, __: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            captured["mask_half"] = output.detach().clone()

        handle = model.decoder.mask_half_head.register_forward_hook(capture_half)
        try:
            with torch.no_grad():
                outputs = model(torch.randn(1, 3, 64, 80))
        finally:
            handle.remove()
        expected = F.interpolate(captured["mask_half"], size=(64, 80), mode="bilinear", align_corners=False)
        self.assertTrue(torch.equal(outputs["mask"], expected))

    def test_mask_continuity_does_not_change_edge_or_reliability_paths(self) -> None:
        model = self._small_model().eval()
        image = torch.randn(1, 3, 64, 80)
        with torch.no_grad():
            before = model(image)
            for block in model.decoder.mask_continuity:
                block.residual_projection.weight.fill_(0.05)
                block.residual_projection.bias.fill_(0.01)
            after = model(image)
        self.assertTrue(torch.equal(before["edge"], after["edge"]))
        self.assertTrue(torch.equal(before["reliability"], after["reliability"]))
        self.assertTrue(torch.equal(before["route_s1"], after["route_s1"]))
        self.assertTrue(torch.equal(before["route_s4"], after["route_s4"]))
        self.assertFalse(torch.equal(before["mask"], after["mask"]))

    def test_registry_contract(self) -> None:
        model = build_model(
            "DEMambaNetLiteBalancedPlus",
            encoder_widths=(8, 16, 24, 32),
            encoder_depths=(1, 1, 1, 1),
            decoder_channels=16,
            directional_kernel=7,
            mamba_channels=16,
            use_mamba_ssm=False,
        )
        self.assertIsInstance(model, DEMambaNetLiteBalancedPlus)
        self.assertEqual(model.mamba_backend_info(), {"global_context": ["fallback"]})

    def test_config_changes_only_model_name_and_save_dir(self) -> None:
        balanced = _load(BALANCED_CONFIG)
        plus = _load(PLUS_CONFIG)
        for section in ("data", "train", "loss", "evaluation"):
            self.assertEqual(plus[section], balanced[section], section)
        expected_model = dict(balanced["model"])
        expected_model["name"] = "DEMambaNetLiteBalancedPlus"
        self.assertEqual(plus["model"], expected_model)
        self.assertNotEqual(plus["save"], balanced["save"])
        self.assertEqual(
            plus["save"]["save_dir"],
            "./runs/pilot/lightweight_demambanet_lite_balanced_plus/"
            "DE_MAMBANET_LITE_BALANCED_PLUS_LR150_DEEPCRACK_NATIVE384x544_EMA_BPCM",
        )


if __name__ == "__main__":
    unittest.main()
