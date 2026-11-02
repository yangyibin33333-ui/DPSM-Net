from __future__ import annotations

import copy
import unittest
from pathlib import Path

import torch
import yaml

from models import DEMambaNet
from models.decoder import LowResolutionSubpixelAdditiveUpBlock, UpBlock


BASE_CONFIG = Path("configs/experiments/lightweight_s4hv/C3_C32_S4HV_LR150_DEEPCRACK_NATIVE384x544_EMA_BPCM.yaml")
LSAF_CONFIG = Path(
    "configs/experiments/lightweight_s4hv/"
    "C3_C32_S4HV_LR150_DEEPCRACK_NATIVE384x544_EMA_BPCM_LSAF.yaml"
)


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class LowResolutionSubpixelAdditiveFusionTests(unittest.TestCase):
    @staticmethod
    def _small_kwargs() -> dict:
        return {
            "base_channels": 8,
            "use_mamba_ssm": False,
            "mamba_stages": (),
            "transformer_stages": (),
        }

    def test_lsaf_block_reassembles_low_resolution_features_at_skip_size(self) -> None:
        block = LowResolutionSubpixelAdditiveUpBlock(8, 6, 4).eval()
        decoder_feature = torch.randn(2, 8, 7, 9)
        skip_feature = torch.randn(2, 6, 14, 18)
        with torch.no_grad():
            output = block(decoder_feature, skip_feature)
        self.assertEqual(tuple(output.shape), (2, 4, 14, 18))
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(tuple(block.pixel_shuffle(block.decoder_projection(decoder_feature)).shape), (2, 4, 14, 18))
        with self.assertRaisesRegex(ValueError, "exactly two times"):
            block(decoder_feature, skip_feature[..., :-1, :])

    def test_legacy_path_is_exact_when_lsaf_is_disabled(self) -> None:
        kwargs = self._small_kwargs()
        torch.manual_seed(42)
        implicit_default = DEMambaNet(**kwargs).eval()
        torch.manual_seed(42)
        explicit_default = DEMambaNet(**kwargs, use_low_resolution_subpixel_additive_fusion=False).eval()

        self.assertIsInstance(implicit_default.decoder.up3, UpBlock)
        self.assertIsInstance(implicit_default.decoder.up2, UpBlock)
        self.assertIsInstance(implicit_default.decoder.up1, UpBlock)
        self.assertIsNotNone(implicit_default.decoder.proj4)
        self.assertEqual(implicit_default.state_dict().keys(), explicit_default.state_dict().keys())
        for name, parameter in implicit_default.state_dict().items():
            self.assertTrue(torch.equal(parameter, explicit_default.state_dict()[name]), name)

        image = torch.randn(1, 3, 64, 80)
        with torch.no_grad():
            implicit_outputs = implicit_default(image)
            explicit_outputs = explicit_default(image)
        self.assertEqual(implicit_outputs.keys(), explicit_outputs.keys())
        for name in implicit_outputs:
            self.assertTrue(torch.equal(implicit_outputs[name], explicit_outputs[name]), name)

    def test_lsaf_model_contract_and_parameter_reduction(self) -> None:
        kwargs = self._small_kwargs()
        baseline = DEMambaNet(**kwargs).eval()
        lsaf = DEMambaNet(**kwargs, use_low_resolution_subpixel_additive_fusion=True).eval()

        self.assertIsNone(lsaf.decoder.proj4)
        self.assertIsInstance(lsaf.decoder.up3, LowResolutionSubpixelAdditiveUpBlock)
        self.assertIsInstance(lsaf.decoder.up2, LowResolutionSubpixelAdditiveUpBlock)
        self.assertIsInstance(lsaf.decoder.up1, LowResolutionSubpixelAdditiveUpBlock)
        self.assertLess(
            sum(parameter.numel() for parameter in lsaf.parameters()),
            sum(parameter.numel() for parameter in baseline.parameters()),
        )

        with torch.no_grad():
            outputs = lsaf(torch.randn(1, 3, 64, 80))
        for name in ("mask", "edge", "reliability"):
            self.assertEqual(tuple(outputs[name].shape), (1, 1, 64, 80))
            self.assertTrue(torch.isfinite(outputs[name]).all())

        with self.assertRaisesRegex(ValueError, "independent decoder ablation"):
            DEMambaNet(**kwargs, use_low_resolution_subpixel_additive_fusion=True, use_frequency_gate=True)

    def test_lsaf_config_is_single_decoder_variable(self) -> None:
        baseline = _load_config(BASE_CONFIG)
        candidate = _load_config(LSAF_CONFIG)
        expected = copy.deepcopy(baseline)
        expected["model"]["use_low_resolution_subpixel_additive_fusion"] = True
        expected["save"]["save_dir"] = (
            "./runs/pilot/lightweight_s4hv/C3_C32_S4HV_LR150_DEEPCRACK_NATIVE384x544_EMA_BPCM_LSAF"
        )
        self.assertEqual(candidate, expected)
        self.assertEqual(candidate["data"]["img_size"], [384, 544])
        self.assertEqual(candidate["train"]["batch_size"], 8)
        self.assertEqual(candidate["train"]["epochs"], 200)
        self.assertEqual(candidate["train"]["lr_scheduler"], "cosine")


if __name__ == "__main__":
    unittest.main()
