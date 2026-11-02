from __future__ import annotations

import unittest
from pathlib import Path

import torch
import yaml

from models import DEMambaNet
from models.blocks import ResidualBranchCompetitionFusion, TopoRouteBlock


class ResidualBranchCompetitionFusionTests(unittest.TestCase):
    def test_rbcf_is_identity_at_initialization_and_uses_only_same_location_branch_residuals(self) -> None:
        torch.manual_seed(42)
        fusion = ResidualBranchCompetitionFusion(channels=16).eval()
        x = torch.randn(2, 16, 7, 9)
        local = torch.randn_like(x)
        routed = torch.randn_like(x)
        global_feature = torch.randn_like(x)
        with torch.no_grad():
            local_calibrated, routed_calibrated, global_calibrated = fusion(x, local, routed, global_feature)
            initial_gate_delta = fusion.gate_delta(x, local, routed, global_feature)
        self.assertTrue(torch.equal(local_calibrated, local))
        self.assertTrue(torch.equal(routed_calibrated, routed))
        self.assertTrue(torch.equal(global_calibrated, global_feature))
        self.assertEqual(tuple(initial_gate_delta.shape), (2, 3, 7, 9))
        self.assertLessEqual(float(initial_gate_delta.abs().max()), 0.5)

        with torch.no_grad():
            fusion.local_proj.weight.zero_()
            fusion.mamba_proj.weight.zero_()
            fusion.transformer_proj.weight.zero_()
            for channel in range(fusion.local_proj.out_channels):
                fusion.local_proj.weight[channel, channel, 0, 0] = 0.1 * (channel + 1)
            fusion.gate_out.weight.zero_()
            fusion.gate_out.weight[:, 0, 0, 0] = 0.1
            fusion.gate_out.bias.fill_(0.1)
            nonzero_gate_delta = fusion.gate_delta(x, local, routed, global_feature)
        self.assertGreater(float(nonzero_gate_delta.abs().max()), 0.0)
        self.assertLessEqual(float(nonzero_gate_delta.abs().max()), 0.5)
        changed_local = local.clone()
        changed_local[:, :, 0, 0] += 5.0
        with torch.no_grad():
            changed_gate_delta = fusion.gate_delta(x, changed_local, routed, global_feature)
        self.assertTrue(torch.equal(nonzero_gate_delta[:, :, 5, 7], changed_gate_delta[:, :, 5, 7]))
        self.assertFalse(torch.equal(nonzero_gate_delta[:, :, 0, 0], changed_gate_delta[:, :, 0, 0]))

        self.assertEqual(fusion.local_proj.kernel_size, (1, 1))
        self.assertEqual(fusion.mamba_proj.kernel_size, (1, 1))
        self.assertEqual(fusion.transformer_proj.kernel_size, (1, 1))
        self.assertEqual(fusion.gate_out.kernel_size, (1, 1))
        self.assertTrue(
            all(module.kernel_size == (1, 1) for module in fusion.modules() if isinstance(module, torch.nn.Conv2d))
        )

        base_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml")
        experiment_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM_RBCF.yaml")
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
        self.assertEqual(experiment_cfg["save"]["save_dir"], "./runs/pilot/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM_RBCF")
        enabled_optional_modules = {
            name
            for name, value in experiment_cfg["model"].items()
            if name.startswith("use_") and value and name != "use_mamba_ssm"
        }
        self.assertEqual(enabled_optional_modules, {"use_residual_branch_competition_fusion"})

        base_model_cfg = dict(base_cfg["model"])
        base_model_cfg.pop("name")
        experiment_model_cfg = dict(experiment_cfg["model"])
        experiment_model_cfg.pop("name")
        torch.manual_seed(42)
        native = DEMambaNet(**base_model_cfg).eval()
        torch.manual_seed(42)
        rbcf_model = DEMambaNet(**experiment_model_cfg).eval()
        self.assertIsInstance(native.stage3, TopoRouteBlock)
        self.assertIsInstance(native.stage4, TopoRouteBlock)
        self.assertIsNone(native.stage3.residual_branch_competition_fusion)
        self.assertIsNone(native.stage4.residual_branch_competition_fusion)
        self.assertIsNotNone(rbcf_model.stage3.residual_branch_competition_fusion)
        self.assertIsNotNone(rbcf_model.stage4.residual_branch_competition_fusion)
        self.assertEqual(sum(parameter.numel() for parameter in native.parameters()), 7026611)
        self.assertEqual(sum(parameter.numel() for parameter in rbcf_model.parameters()), 7074917)
        native_state = native.state_dict()
        shared_state = {
            name: value
            for name, value in rbcf_model.state_dict().items()
            if ".residual_branch_competition_fusion." not in name
        }
        self.assertEqual(set(native_state), set(shared_state))
        for name, value in native_state.items():
            self.assertTrue(torch.equal(value, shared_state[name]), name)

        fallback_kwargs = {
            "base_channels": 8,
            "use_mamba_ssm": False,
            "transformer_stages": [],
        }
        torch.manual_seed(7)
        fallback_native = DEMambaNet(**fallback_kwargs).eval()
        torch.manual_seed(7)
        fallback_model = DEMambaNet(
            **fallback_kwargs,
            use_residual_branch_competition_fusion=True,
        ).eval()
        inputs = torch.randn(1, 3, 32, 40)
        with torch.no_grad():
            native_outputs = fallback_native(inputs)
            rbcf_outputs = fallback_model(inputs)
        for name, output in native_outputs.items():
            self.assertTrue(torch.equal(rbcf_outputs[name], output), name)

        self.assertEqual(fallback_model.mamba_backend_info(), {"stage3": ["fallback"] * 4, "stage4": ["fallback"] * 4})
        self.assertEqual(fallback_model.stage3.route_mamba.route_head[-1].out_channels, 4)
        self.assertEqual(fallback_model.stage4.route_mamba.route_head[-1].out_channels, 4)


if __name__ == "__main__":
    unittest.main()
