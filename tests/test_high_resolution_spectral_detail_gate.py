from __future__ import annotations

import inspect
import unittest
from pathlib import Path

import torch
import yaml

from models import DEMambaNet
from models.decoder import HighResolutionSpectralDetailGate


class HighResolutionSpectralDetailGateTests(unittest.TestCase):
    def test_hrsdg_is_identity_at_initialization_and_local_dct_energy_is_pixelwise_local(self) -> None:
        torch.manual_seed(42)
        gate = HighResolutionSpectralDetailGate(decoder_channels=40, stage3_channels=160).eval()
        decoder_feature = torch.randn(2, 40, 64, 80)
        stage3_feature = torch.randn(2, 160, 16, 20)
        with torch.no_grad():
            output = gate(decoder_feature, stage3_feature)
        self.assertTrue(torch.equal(output, decoder_feature))
        self.assertEqual(tuple(output.shape), tuple(decoder_feature.shape))
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(sum(parameter.numel() for parameter in gate.parameters()), 1360)
        self.assertNotIn("dct_basis", dict(gate.named_parameters()))
        self.assertIn("dct_basis", dict(gate.named_buffers()))
        self.assertNotIn("dct_basis", gate.state_dict())
        self.assertFalse(gate.dct_basis.requires_grad)
        self.assertEqual(
            tuple(name for name in inspect.signature(HighResolutionSpectralDetailGate.forward).parameters if name != "self"),
            ("decoder_feature", "stage3_feature"),
        )

        analysis_feature = torch.randn(1, 4, 9, 11)
        local_energy = gate._spectral_energy(analysis_feature)
        remote_change = analysis_feature.clone()
        remote_change[..., 0, 0] += 3.0
        changed_energy = gate._spectral_energy(remote_change)
        self.assertEqual(tuple(local_energy.shape), (1, 8, 9, 11))
        self.assertTrue(torch.isfinite(local_energy).all())
        self.assertTrue(torch.equal(local_energy[..., 4, 5], changed_energy[..., 4, 5]))
        self.assertFalse(torch.equal(local_energy[..., 0, 0], changed_energy[..., 0, 0]))

        base_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml")
        experiment_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM_HRSDG.yaml")
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
        self.assertEqual(experiment_cfg["save"]["save_dir"], "./runs/pilot/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM_HRSDG")
        enabled_optional_modules = {
            name
            for name, value in experiment_cfg["model"].items()
            if name.startswith("use_") and value and name != "use_mamba_ssm"
        }
        self.assertEqual(enabled_optional_modules, {"use_high_resolution_spectral_detail_gate"})

        base_model_cfg = dict(base_cfg["model"])
        base_model_cfg.pop("name")
        experiment_model_cfg = dict(experiment_cfg["model"])
        experiment_model_cfg.pop("name")
        torch.manual_seed(42)
        native = DEMambaNet(**base_model_cfg).eval()
        torch.manual_seed(42)
        hrsdg_model = DEMambaNet(**experiment_model_cfg).eval()
        self.assertIsNone(native.decoder.high_resolution_spectral_detail_gate)
        self.assertIsNotNone(hrsdg_model.decoder.high_resolution_spectral_detail_gate)
        self.assertEqual(sum(parameter.numel() for parameter in native.parameters()), 7026611)
        self.assertEqual(sum(parameter.numel() for parameter in hrsdg_model.parameters()), 7027971)
        self.assertEqual(
            sum(parameter.numel() for parameter in hrsdg_model.parameters())
            - sum(parameter.numel() for parameter in native.parameters()),
            1360,
        )
        native_state = native.state_dict()
        for name, value in hrsdg_model.state_dict().items():
            if name.startswith("decoder.high_resolution_spectral_detail_gate."):
                continue
            self.assertIn(name, native_state)
            self.assertTrue(torch.equal(value, native_state[name]), name)

        cpu_model_cfg = dict(base_model_cfg)
        cpu_model_cfg["use_mamba_ssm"] = False
        cpu_hrsdg_model_cfg = dict(cpu_model_cfg)
        cpu_hrsdg_model_cfg["use_high_resolution_spectral_detail_gate"] = True
        torch.manual_seed(42)
        cpu_native = DEMambaNet(**cpu_model_cfg).eval()
        torch.manual_seed(42)
        cpu_hrsdg = DEMambaNet(**cpu_hrsdg_model_cfg).eval()
        image = torch.randn(1, 3, 64, 80)
        with torch.no_grad():
            native_outputs = cpu_native(image)
            hrsdg_outputs = cpu_hrsdg(image)
        for name in ("mask", "edge", "reliability"):
            self.assertEqual(tuple(hrsdg_outputs[name].shape), tuple(native_outputs[name].shape))
            self.assertTrue(torch.isfinite(hrsdg_outputs[name]).all())
            self.assertTrue(torch.equal(hrsdg_outputs[name], native_outputs[name]), name)


if __name__ == "__main__":
    unittest.main()
