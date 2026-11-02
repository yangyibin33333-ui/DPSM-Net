from __future__ import annotations

import unittest
from pathlib import Path

import torch
import yaml

from models import DEMambaNet
from models.decoder import LocalSpectralConfidenceRejector


class LocalSpectralConfidenceRejectorTests(unittest.TestCase):
    def test_lscr_is_identity_at_init_and_local_dct_energy_has_no_global_dependency(self) -> None:
        torch.manual_seed(42)
        rejector = LocalSpectralConfidenceRejector(decoder_channels=40, stage3_channels=160).eval()
        detail = torch.randn(2, 40, 64, 80)
        stage3 = torch.randn(2, 160, 16, 20)
        mask_logits = torch.randn(2, 1, 64, 80)
        edge_logits = torch.randn(2, 1, 64, 80)
        reliability_logits = torch.randn(2, 1, 64, 80)

        with torch.no_grad():
            corrected = rejector(detail, stage3, mask_logits, edge_logits, reliability_logits)
        self.assertTrue(torch.equal(corrected, mask_logits))
        self.assertEqual(tuple(corrected.shape), tuple(mask_logits.shape))
        self.assertTrue(torch.isfinite(corrected).all())
        self.assertEqual(sum(parameter.numel() for parameter in rejector.parameters()), 2241)
        self.assertNotIn("dct_basis", dict(rejector.named_parameters()))
        self.assertIn("dct_basis", dict(rejector.named_buffers()))
        self.assertNotIn("dct_basis", rejector.state_dict())
        self.assertFalse(rejector.dct_basis.requires_grad)

        analysis_feature = torch.zeros(1, 8, 8, 8)
        analysis_feature[..., 1, 1] = 1.0
        local_energy = rejector._spectral_energy(analysis_feature)
        remote_change = analysis_feature.clone()
        remote_change[..., 5, 5] = 3.0
        changed_energy = rejector._spectral_energy(remote_change)
        self.assertEqual(tuple(local_energy.shape), (1, 15, 2, 2))
        self.assertTrue(torch.isfinite(local_energy).all())
        self.assertTrue(torch.equal(local_energy[..., 0, 0], changed_energy[..., 0, 0]))
        self.assertFalse(torch.equal(local_energy[..., 1, 1], changed_energy[..., 1, 1]))

        base_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml")
        experiment_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM_LSCR.yaml")
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
        self.assertEqual(experiment_cfg["save"]["save_dir"], "./runs/pilot/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM_LSCR")
        self.assertEqual(experiment_cfg["data"]["img_size"], [384, 544])
        self.assertEqual(experiment_cfg["train"]["epochs"], 200)
        self.assertEqual(experiment_cfg["train"]["batch_size"], 8)
        self.assertEqual(experiment_cfg["train"]["lr"], 0.00015)
        self.assertEqual(experiment_cfg["train"]["lr_scheduler"], "cosine")
        self.assertEqual(experiment_cfg["train"]["amp_dtype"], "fp16")
        enabled_optional_modules = {
            name
            for name, value in experiment_cfg["model"].items()
            if name.startswith("use_") and value and name != "use_mamba_ssm"
        }
        self.assertEqual(enabled_optional_modules, {"use_local_spectral_confidence_rejector"})

        base_model_cfg = dict(base_cfg["model"])
        base_model_cfg.pop("name")
        experiment_model_cfg = dict(experiment_cfg["model"])
        experiment_model_cfg.pop("name")
        torch.manual_seed(42)
        native = DEMambaNet(**base_model_cfg).eval()
        torch.manual_seed(42)
        lscr_model = DEMambaNet(**experiment_model_cfg).eval()
        self.assertIsNone(native.decoder.local_spectral_confidence_rejector)
        self.assertIsNotNone(lscr_model.decoder.local_spectral_confidence_rejector)
        self.assertEqual(sum(parameter.numel() for parameter in native.parameters()), 7026611)
        self.assertEqual(sum(parameter.numel() for parameter in lscr_model.parameters()), 7028852)
        self.assertEqual(
            sum(parameter.numel() for parameter in lscr_model.parameters())
            - sum(parameter.numel() for parameter in native.parameters()),
            2241,
        )
        native_state = native.state_dict()
        for name, value in lscr_model.state_dict().items():
            if name.startswith("decoder.local_spectral_confidence_rejector."):
                continue
            self.assertIn(name, native_state)
            self.assertTrue(torch.equal(value, native_state[name]), name)

        # The real mamba_ssm CUDA extension cannot execute CPU tensors. Keep
        # the parameter/state checks above on the configured Mamba backend,
        # then verify LSCR's zero-initialized forward identity with the
        # architecture-compatible CPU fallback.
        cpu_model_cfg = dict(base_model_cfg)
        cpu_model_cfg["use_mamba_ssm"] = False
        cpu_lscr_model_cfg = dict(cpu_model_cfg)
        cpu_lscr_model_cfg["use_local_spectral_confidence_rejector"] = True
        torch.manual_seed(42)
        cpu_native = DEMambaNet(**cpu_model_cfg).eval()
        torch.manual_seed(42)
        cpu_lscr = DEMambaNet(**cpu_lscr_model_cfg).eval()
        image = torch.randn(1, 3, 64, 80)
        with torch.no_grad():
            native_outputs = cpu_native(image)
            lscr_outputs = cpu_lscr(image)
        for name in ("mask", "edge", "reliability"):
            self.assertEqual(tuple(lscr_outputs[name].shape), tuple(native_outputs[name].shape))
            self.assertTrue(torch.isfinite(lscr_outputs[name]).all())
            self.assertTrue(torch.equal(lscr_outputs[name], native_outputs[name]), name)


if __name__ == "__main__":
    unittest.main()
