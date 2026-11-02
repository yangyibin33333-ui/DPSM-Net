from __future__ import annotations

import tempfile
import unittest
import inspect
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml
from PIL import Image

from datasets import CrackSegmentationDataset
from datasets.crack_dataset import _augment, _photometric_augment
from losses import BCEDiceLoss, DEMambaLoss, EdgeLoss, SCNP, TruncatedSDFLoss
from models import DEMambaNet
from models.blocks import (
    DensityBalancedPrototypeContext,
    LocalRouteConsistencyGate,
    RetinexDirectionalValleyPrior,
    SemanticConditionedLocalKernelResidual,
    TopoRouteBlock,
)
from models.decoder import (
    BoundaryLogitCalibrator,
    DetailResidualSkipGate,
    EdgeMaskConsistencyRefiner,
    LocalSemanticContinuityCalibrator,
    LocalPixelReassemblyResidual,
    RouteGuidedAnisotropicDiffusionRefiner,
    RouteGuidedDirectionalRefiner,
    SemanticSkipGate,
    StructureConsistentHarmonicFusion,
)
from models.mamba_blocks import TangentRoutedMambaBlock
from metrics import SegmentationMetric
from test import build_threshold_sweep, summarize_threshold_sweep, update_threshold_sweep
from train import (
    accumulate_epoch_timing,
    apply_overrides,
    create_run_dir,
    format_mamba_fallback_warning,
    format_duration,
    make_optimizer,
    make_scheduler,
    save_resolved_config,
    structural_score,
    summarize_mamba_backends,
)
from utils.structural_targets import build_structural_targets, build_truncated_sdf_targets, distance_transform_edt, skeletonize
from utils.visualize import save_diagnostic_bundle, write_diagnostics_readme


class LiteTopoRouteTests(unittest.TestCase):
    @unittest.skipIf(skeletonize is None, "scikit-image is not installed")
    def test_tangent_bins_for_synthetic_cracks(self) -> None:
        cases = [(np.eye(17, dtype=np.uint8), 2), (np.fliplr(np.eye(17, dtype=np.uint8)), 3)]
        horizontal = np.zeros((17, 17), dtype=np.uint8)
        horizontal[8] = 1
        vertical = horizontal.T
        cases.extend([(horizontal, 0), (vertical, 1)])
        for mask, expected in cases:
            targets = build_structural_targets(np.ascontiguousarray(mask), threshold=0)
            center = targets["tangent_gt"][:, 8, 8]
            self.assertEqual(int(center.argmax()), expected)
            self.assertEqual(float(targets["tangent_valid"][0, 8, 8]), 1.0)

    def test_route_logits_are_spatial_and_model_outputs_match_input(self) -> None:
        torch.manual_seed(0)
        block = TangentRoutedMambaBlock(8, use_mamba_ssm=False).eval()
        with torch.no_grad():
            _, routes = block(torch.randn(2, 8, 9, 11))
        self.assertEqual(tuple(routes.shape), (2, 4, 9, 11))
        self.assertGreater(float(routes.std(dim=(-2, -1)).mean()), 0.0)
        model = DEMambaNet(base_channels=8, use_mamba_ssm=False, transformer_stages=[]).eval()
        with torch.no_grad():
            outputs = model(torch.randn(1, 3, 32, 40))
        self.assertEqual(tuple(outputs["mask"].shape), (1, 1, 32, 40))
        self.assertIn("route_s3", outputs)
        self.assertIn("route_s4", outputs)

    def test_lpr2_is_zero_initialized_and_preserves_seeded_baseline(self) -> None:
        kwargs = {"base_channels": 8, "use_mamba_ssm": False, "transformer_stages": []}
        torch.manual_seed(42)
        native = DEMambaNet(**kwargs).eval()
        torch.manual_seed(42)
        lpr2 = DEMambaNet(**kwargs, use_local_pixel_reassembly_residual=True).eval()

        self.assertIsNone(native.decoder.local_pixel_reassembly_residual)
        self.assertIsNotNone(lpr2.decoder.local_pixel_reassembly_residual)
        native_state = native.state_dict()
        shared_state = {
            name: value
            for name, value in lpr2.state_dict().items()
            if not name.startswith("decoder.local_pixel_reassembly_residual.")
        }
        self.assertEqual(set(native_state), set(shared_state))
        for name, value in native_state.items():
            self.assertTrue(torch.equal(value, shared_state[name]), name)

        inputs = torch.randn(1, 3, 64, 80)
        with torch.no_grad():
            native_outputs = native(inputs)
            lpr2_outputs = lpr2(inputs)
        self.assertEqual(set(native_outputs), set(lpr2_outputs))
        for name, output in native_outputs.items():
            self.assertEqual(tuple(lpr2_outputs[name].shape), tuple(output.shape), name)
            self.assertTrue(torch.isfinite(lpr2_outputs[name]).all(), name)
            self.assertTrue(torch.equal(lpr2_outputs[name], output), name)

        native_count = sum(parameter.numel() for parameter in native.parameters())
        lpr2_count = sum(parameter.numel() for parameter in lpr2.parameters())
        module_count = sum(parameter.numel() for parameter in lpr2.decoder.local_pixel_reassembly_residual.parameters())
        self.assertEqual(lpr2_count - native_count, module_count)
        self.assertEqual(sum(parameter.numel() for parameter in LocalPixelReassemblyResidual(40).parameters()), 5121)

        locality_module = LocalPixelReassemblyResidual(1).eval()
        with torch.no_grad():
            locality_module.reduce.weight.fill_(1.0)
            locality_module.expand.weight.fill_(1.0)
            locality_module.gamma.fill_(1.0)
        reference = torch.zeros(1, 1, 4, 4)
        injected = reference.clone()
        injected[:, :, 1, 1] = 1.0
        with torch.no_grad():
            difference = locality_module(injected) - locality_module(reference)
        self.assertGreater(float(difference[:, :, :2, :2].abs().sum()), 0.0)
        outside_tile = difference.clone()
        outside_tile[:, :, :2, :2] = 0.0
        self.assertTrue(torch.equal(outside_tile, torch.zeros_like(outside_tile)))

        config_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_LPR2.yaml")
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        self.assertEqual(config["data"]["img_size"], [384, 544])
        self.assertEqual(config["train"]["epochs"], 200)
        self.assertEqual(config["train"]["batch_size"], 8)
        self.assertEqual(config["train"]["lr"], 0.00015)
        self.assertEqual(config["train"]["lr_scheduler"], "cosine")
        self.assertEqual(config["train"]["amp_dtype"], "fp16")
        self.assertTrue(config["model"]["use_local_pixel_reassembly_residual"])
        for name in (
            "use_frequency_gate",
            "use_semantic_skip_gate",
            "use_detail_residual_gate",
            "use_structure_harmonic_fusion",
            "use_boundary_logit_calibration",
            "use_sdf_head",
            "use_edge_mask_consistency_refiner",
            "use_deep_supervision",
            "use_retinex_directional_valley_prior",
            "use_route_guided_directional_refiner",
            "use_route_guided_anisotropic_diffusion",
            "use_density_balanced_prototype_context",
            "use_local_semantic_continuity_calibrator",
            "use_local_route_gate",
            "use_deep_identity_decay",
        ):
            self.assertFalse(config["model"][name], name)

    def test_sclkr_is_identity_deterministic_and_uses_only_local_detail_values(self) -> None:
        kwargs = {"base_channels": 8, "use_mamba_ssm": False, "transformer_stages": []}
        torch.manual_seed(42)
        native = DEMambaNet(**kwargs).eval()
        torch.manual_seed(42)
        sclkr_model = DEMambaNet(**kwargs, use_semantic_conditioned_local_kernel_residual=True).eval()

        self.assertIsNone(native.semantic_conditioned_local_kernel_residual)
        self.assertIsNotNone(sclkr_model.semantic_conditioned_local_kernel_residual)
        native_state = native.state_dict()
        shared_state = {
            name: value
            for name, value in sclkr_model.state_dict().items()
            if not name.startswith("semantic_conditioned_local_kernel_residual.")
        }
        self.assertEqual(set(native_state), set(shared_state))
        for name, value in native_state.items():
            self.assertTrue(torch.equal(value, shared_state[name]), name)

        model_input = torch.randn(1, 3, 64, 80)
        with torch.no_grad():
            native_outputs = native(model_input)
            sclkr_outputs = sclkr_model(model_input)
        self.assertEqual(set(native_outputs), set(sclkr_outputs))
        for name, output in native_outputs.items():
            self.assertEqual(tuple(sclkr_outputs[name].shape), tuple(output.shape), name)
            self.assertTrue(torch.isfinite(sclkr_outputs[name]).all(), name)
            self.assertTrue(torch.equal(sclkr_outputs[name], output), name)

        sclkr = SemanticConditionedLocalKernelResidual(160, 320).eval()
        self.assertEqual(sum(parameter.numel() for parameter in sclkr.parameters()), 33893)
        detail = torch.randn(1, 160, 12, 17)
        context = torch.randn(1, 320, 6, 9)
        with torch.no_grad():
            identity_output = sclkr(detail, context)
        self.assertEqual(tuple(identity_output.shape), tuple(detail.shape))
        self.assertTrue(torch.isfinite(identity_output).all())
        self.assertTrue(torch.equal(identity_output, detail))

        local_module = SemanticConditionedLocalKernelResidual(4, 4).eval()
        with torch.no_grad():
            local_module.residual_scale.fill_(1.0)
            local_module.q_proj.weight.fill_(0.25)
            local_module.k_proj.weight.fill_(0.5)
            local_module.affinity_to_kernel.zero_()
            local_module.affinity_to_kernel[:, :, 0].fill_(1.0)
            local_module.kernel_bias.zero_()
            local_module.out_proj.weight.copy_(torch.eye(4).view(4, 4, 1, 1))

        local_detail = torch.randn(1, 4, 7, 9)
        context_a = torch.ones(1, 4, 4, 5)
        context_b = torch.zeros_like(context_a)
        with torch.no_grad():
            _, logits_a = local_module.kernel_logits(local_detail, context_a)
            _, logits_b = local_module.kernel_logits(local_detail, context_b)
            self.assertFalse(torch.equal(torch.softmax(logits_a, dim=-1), torch.softmax(logits_b, dim=-1)))

            reference = local_module(local_detail, context_a)
            repeated_reference = local_module(local_detail, context_a)
            changed_detail = local_detail.clone()
            changed_detail[:, :, 0, 0] = changed_detail[:, :, 0, 0] + 10.0
            changed = local_module(changed_detail, context_a)
        self.assertTrue(torch.equal(reference, repeated_reference))
        self.assertTrue(torch.equal(reference[:, :, 4, 5], changed[:, :, 4, 5]))

        zero_detail = torch.zeros_like(local_detail)
        with torch.no_grad():
            context_only_a = local_module(zero_detail, context_a)
            context_only_b = local_module(zero_detail, context_b)
        self.assertTrue(torch.equal(context_only_a, zero_detail))
        self.assertTrue(torch.equal(context_only_b, zero_detail))

        torch.manual_seed(42)
        native_c40 = DEMambaNet(base_channels=40).eval()
        torch.manual_seed(42)
        sclkr_c40 = DEMambaNet(base_channels=40, use_semantic_conditioned_local_kernel_residual=True).eval()
        native_count = sum(parameter.numel() for parameter in native_c40.parameters())
        sclkr_count = sum(parameter.numel() for parameter in sclkr_c40.parameters())
        self.assertEqual(sclkr_count - native_count, 33893)
        if all(backend == "mamba_ssm" for backends in native_c40.mamba_backend_info().values() for backend in backends):
            self.assertEqual(native_count, 7026611)
            self.assertEqual(sclkr_count, 7060504)

        source = inspect.getsource(SemanticConditionedLocalKernelResidual)
        self.assertNotIn("F.unfold", source)
        self.assertNotIn("scaled_dot_product_attention", source)

        baseline_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544.yaml")
        with baseline_path.open(encoding="utf-8") as handle:
            baseline_config = yaml.safe_load(handle)
        self.assertFalse(baseline_config["model"].get("use_semantic_conditioned_local_kernel_residual", False))

        config_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_SCLKR.yaml")
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        self.assertEqual(config["data"]["img_size"], [384, 544])
        self.assertEqual(config["train"]["epochs"], 200)
        self.assertEqual(config["train"]["batch_size"], 8)
        self.assertEqual(config["train"]["lr"], 0.00015)
        self.assertEqual(config["train"]["lr_scheduler"], "cosine")
        self.assertEqual(config["train"]["amp_dtype"], "fp16")
        self.assertTrue(config["model"]["use_semantic_conditioned_local_kernel_residual"])
        for name in (
            "use_frequency_gate",
            "use_semantic_skip_gate",
            "use_detail_residual_gate",
            "use_structure_harmonic_fusion",
            "use_boundary_logit_calibration",
            "use_sdf_head",
            "use_edge_mask_consistency_refiner",
            "use_deep_supervision",
            "use_retinex_directional_valley_prior",
            "use_route_guided_directional_refiner",
            "use_route_guided_anisotropic_diffusion",
            "use_density_balanced_prototype_context",
            "use_local_semantic_continuity_calibrator",
            "use_local_pixel_reassembly_residual",
            "use_local_route_gate",
            "use_deep_identity_decay",
        ):
            self.assertFalse(config["model"][name], name)

    def test_dbpca_is_identity_deterministic_and_uses_compact_prototypes(self) -> None:
        kwargs = {"base_channels": 8, "use_mamba_ssm": False, "transformer_stages": []}
        torch.manual_seed(42)
        baseline = DEMambaNet(**kwargs).eval()
        torch.manual_seed(42)
        dbpca = DEMambaNet(**kwargs, use_density_balanced_prototype_context=True).eval()
        self.assertIsNone(baseline.density_balanced_prototype_context)
        self.assertIsNotNone(dbpca.density_balanced_prototype_context)

        baseline_state = baseline.state_dict()
        shared_state = {
            name: value
            for name, value in dbpca.state_dict().items()
            if not name.startswith("density_balanced_prototype_context.")
        }
        self.assertEqual(set(baseline_state), set(shared_state))
        for name, value in baseline_state.items():
            self.assertTrue(torch.equal(value, shared_state[name]), name)

        model_input = torch.randn(1, 3, 64, 80)
        with torch.no_grad():
            baseline_outputs = baseline(model_input)
            dbpca_outputs = dbpca(model_input)
        self.assertTrue(torch.equal(dbpca_outputs["mask"], baseline_outputs["mask"]))
        self.assertEqual(float((dbpca_outputs["mask"] - baseline_outputs["mask"]).abs().max()), 0.0)

        context = DensityBalancedPrototypeContext(160).eval()
        feature = torch.randn(2, 160, 96, 136)
        with torch.no_grad():
            prototypes_first, target_norms_first = context.prototypes_and_target_norms(feature)
            prototypes_second, target_norms_second = context.prototypes_and_target_norms(feature)
            output_first = context(feature)
            output_second = context(feature)
        self.assertEqual(tuple(prototypes_first.shape), (2, 16, 160))
        self.assertEqual(tuple(target_norms_first.shape), (2, 16))
        self.assertEqual(tuple(output_first.shape), (2, 160, 96, 136))
        self.assertTrue(torch.isfinite(prototypes_first).all())
        self.assertTrue(torch.isfinite(output_first).all())
        self.assertTrue(torch.equal(output_first, feature))
        self.assertTrue(torch.allclose(prototypes_first, prototypes_second, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(target_norms_first, target_norms_second, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(output_first, output_second, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(prototypes_first.norm(dim=-1), target_norms_first, atol=1e-5, rtol=1e-4))
        self.assertEqual(sum(parameter.numel() for parameter in context.parameters()), 41120)

        source = inspect.getsource(DensityBalancedPrototypeContext)
        self.assertIn("F.scaled_dot_product_attention", source)
        self.assertNotIn("one_hot", source)
        self.assertNotIn("rand", source)
        self.assertNotIn("matmul", source)

        config_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_DBPCA.yaml")
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        self.assertTrue(config["model"]["use_density_balanced_prototype_context"])
        for name in (
            "use_frequency_gate",
            "use_semantic_skip_gate",
            "use_detail_residual_gate",
            "use_structure_harmonic_fusion",
            "use_boundary_logit_calibration",
            "use_sdf_head",
            "use_edge_mask_consistency_refiner",
            "use_deep_supervision",
            "use_retinex_directional_valley_prior",
            "use_route_guided_directional_refiner",
            "use_route_guided_anisotropic_diffusion",
            "use_local_route_gate",
            "use_deep_identity_decay",
        ):
            self.assertFalse(config["model"][name], name)

    def test_rdvp_is_zero_initialized_and_preserves_seeded_baseline(self) -> None:
        kwargs = {"base_channels": 8, "use_mamba_ssm": False, "transformer_stages": []}
        torch.manual_seed(42)
        baseline = DEMambaNet(**kwargs).eval()
        torch.manual_seed(42)
        rdvp = DEMambaNet(**kwargs, use_retinex_directional_valley_prior=True).eval()

        self.assertIsNone(baseline.retinex_directional_valley_prior)
        self.assertIsNotNone(rdvp.retinex_directional_valley_prior)
        baseline_parameters = dict(baseline.named_parameters())
        shared_parameters = {
            name: parameter
            for name, parameter in rdvp.named_parameters()
            if not name.startswith("retinex_directional_valley_prior.")
        }
        self.assertEqual(set(baseline_parameters), set(shared_parameters))
        for name, parameter in baseline_parameters.items():
            self.assertTrue(torch.equal(parameter, shared_parameters[name]), name)
        baseline_count = sum(parameter.numel() for parameter in baseline.parameters())
        rdvp_count = sum(parameter.numel() for parameter in rdvp.parameters())
        prior_count = sum(parameter.numel() for parameter in rdvp.retinex_directional_valley_prior.parameters())
        self.assertEqual(rdvp_count - baseline_count, prior_count)

        torch.manual_seed(7)
        inputs = torch.randn(2, 3, 384, 544)
        with torch.no_grad():
            baseline_outputs = baseline(inputs)
            rdvp_outputs = rdvp(inputs)
        self.assertEqual(set(rdvp_outputs), set(baseline_outputs))
        self.assertEqual(tuple(rdvp_outputs["mask"].shape), (2, 1, 384, 544))
        for name, output in rdvp_outputs.items():
            self.assertTrue(torch.isfinite(output).all(), name)
            self.assertTrue(torch.equal(output, baseline_outputs[name]), name)
        self.assertEqual(float((rdvp_outputs["mask"] - baseline_outputs["mask"]).abs().max()), 0.0)

        prior = RetinexDirectionalValleyPrior(8).eval()
        feature = torch.randn(2, 8, 384, 544)
        with torch.no_grad():
            valley_strength, directional_coherence = prior.prior_maps(inputs)
            corrected_feature = prior(inputs, feature)
        self.assertEqual(tuple(valley_strength.shape), (2, 1, 384, 544))
        self.assertEqual(tuple(directional_coherence.shape), (2, 1, 384, 544))
        self.assertTrue(torch.isfinite(valley_strength).all())
        self.assertTrue(torch.isfinite(directional_coherence).all())
        self.assertGreaterEqual(float(directional_coherence.min()), 0.0)
        self.assertLessEqual(float(directional_coherence.max()), 1.0)
        self.assertTrue(torch.equal(corrected_feature, feature))

        config_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_RDVP.yaml")
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        self.assertTrue(config["model"]["use_retinex_directional_valley_prior"])
        for name in (
            "use_frequency_gate",
            "use_semantic_skip_gate",
            "use_detail_residual_gate",
            "use_structure_harmonic_fusion",
            "use_boundary_logit_calibration",
            "use_sdf_head",
            "use_edge_mask_consistency_refiner",
            "use_deep_supervision",
            "use_local_route_gate",
            "use_deep_identity_decay",
        ):
            self.assertFalse(config["model"].get(name, False), name)

    def test_rdvp_gated_directional_confidence_rejects_flat_regions(self) -> None:
        mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

        uniform_rgb = torch.full((1, 3, 64, 96), 0.5)
        uniform_input = (uniform_rgb - mean) / std
        uniform_valley, uniform_confidence = RetinexDirectionalValleyPrior(8).prior_maps(uniform_input)
        self.assertLess(float(uniform_valley.abs().max()), 1e-6)
        self.assertLess(float(uniform_confidence.abs().max()), 1e-6)

        crack_rgb = torch.full((1, 3, 64, 96), 0.6)
        crack_rgb[:, :, 32, 12:84] = 0.05
        crack_input = (crack_rgb - mean) / std
        crack_valley, crack_confidence = RetinexDirectionalValleyPrior(8).prior_maps(crack_input)
        line = (slice(None), slice(None), 32, slice(20, 76))
        background = (slice(None), slice(None), 8, slice(20, 76))
        self.assertGreater(float(crack_valley[line].mean()), float(crack_valley[background].mean()) + 0.05)
        self.assertGreater(float(crack_confidence[line].mean()), float(crack_confidence[background].mean()) + 0.05)

        inputs = torch.randn(2, 3, 384, 544)
        valley_strength, directional_coherence = RetinexDirectionalValleyPrior(8).prior_maps(inputs)
        self.assertEqual(tuple(valley_strength.shape), (2, 1, 384, 544))
        self.assertEqual(tuple(directional_coherence.shape), (2, 1, 384, 544))
        self.assertTrue(torch.isfinite(valley_strength).all())
        self.assertTrue(torch.isfinite(directional_coherence).all())
        self.assertGreaterEqual(float(directional_coherence.min()), 0.0)
        self.assertLessEqual(float(directional_coherence.max()), 1.0)

    def test_rgddp_is_zero_initialized_and_preserves_seeded_baseline(self) -> None:
        kwargs = {"base_channels": 8, "use_mamba_ssm": False, "transformer_stages": []}
        torch.manual_seed(42)
        baseline = DEMambaNet(**kwargs).eval()
        torch.manual_seed(42)
        rgddp = DEMambaNet(**kwargs, use_route_guided_directional_refiner=True).eval()

        self.assertIsNone(baseline.decoder.route_guided_directional_refiner)
        self.assertIsNotNone(rgddp.decoder.route_guided_directional_refiner)
        baseline_state = baseline.state_dict()
        shared_state = {
            name: value
            for name, value in rgddp.state_dict().items()
            if not name.startswith("decoder.route_guided_directional_refiner.")
        }
        self.assertEqual(set(baseline_state), set(shared_state))
        for name, value in baseline_state.items():
            self.assertTrue(torch.equal(value, shared_state[name]), name)

        torch.manual_seed(11)
        inputs = torch.randn(2, 3, 384, 544)
        with torch.no_grad():
            baseline_outputs = baseline(inputs)
            rgddp_outputs = rgddp(inputs)
        for name in ("mask", "edge", "reliability"):
            self.assertEqual(tuple(rgddp_outputs[name].shape), (2, 1, 384, 544), name)
            self.assertTrue(torch.isfinite(rgddp_outputs[name]).all(), name)
            self.assertTrue(torch.equal(rgddp_outputs[name], baseline_outputs[name]), name)
        self.assertEqual(float((rgddp_outputs["mask"] - baseline_outputs["mask"]).abs().max()), 0.0)

        refiner = RouteGuidedDirectionalRefiner(40).eval()
        decoder_feature = torch.randn(2, 40, 384, 544)
        route_logits = torch.randn(2, 4, 96, 136)
        with torch.no_grad():
            route_probs, route_confidence = refiner.route_probabilities_and_confidence(
                route_logits, decoder_feature.shape[-2:]
            )
            refined = refiner(decoder_feature, route_logits)
        self.assertEqual(tuple(route_probs.shape), (2, 4, 384, 544))
        self.assertEqual(tuple(route_confidence.shape), (2, 1, 384, 544))
        self.assertEqual(tuple(refined.shape), (2, 40, 384, 544))
        self.assertTrue(torch.isfinite(refined).all())
        self.assertGreaterEqual(float(route_confidence.min()), 0.0)
        self.assertLessEqual(float(route_confidence.max()), 1.0)
        self.assertTrue(torch.equal(refined, decoder_feature))
        self.assertEqual(sum(parameter.numel() for parameter in refiner.parameters()), 1380)

        config_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_RGDDP.yaml")
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        self.assertTrue(config["model"]["use_route_guided_directional_refiner"])
        for name in (
            "use_frequency_gate",
            "use_semantic_skip_gate",
            "use_detail_residual_gate",
            "use_structure_harmonic_fusion",
            "use_boundary_logit_calibration",
            "use_sdf_head",
            "use_edge_mask_consistency_refiner",
            "use_deep_supervision",
            "use_retinex_directional_valley_prior",
            "use_local_route_gate",
            "use_deep_identity_decay",
        ):
            self.assertFalse(config["model"][name], name)

    def test_rgafd_is_zero_initialized_and_preserves_feature_boundaries(self) -> None:
        kwargs = {"base_channels": 8, "use_mamba_ssm": False, "transformer_stages": []}
        torch.manual_seed(42)
        baseline = DEMambaNet(**kwargs).eval()
        torch.manual_seed(42)
        rgafd = DEMambaNet(**kwargs, use_route_guided_anisotropic_diffusion=True).eval()

        self.assertIsNone(baseline.decoder.route_guided_anisotropic_diffusion)
        self.assertIsNotNone(rgafd.decoder.route_guided_anisotropic_diffusion)
        baseline_state = baseline.state_dict()
        shared_state = {
            name: value
            for name, value in rgafd.state_dict().items()
            if not name.startswith("decoder.route_guided_anisotropic_diffusion.")
        }
        self.assertEqual(set(baseline_state), set(shared_state))
        for name, value in baseline_state.items():
            self.assertTrue(torch.equal(value, shared_state[name]), name)

        torch.manual_seed(13)
        inputs = torch.randn(2, 3, 384, 544)
        with torch.no_grad():
            baseline_outputs = baseline(inputs)
            rgafd_outputs = rgafd(inputs)
        for name in ("mask", "edge", "reliability"):
            self.assertEqual(tuple(rgafd_outputs[name].shape), (2, 1, 384, 544), name)
            self.assertTrue(torch.isfinite(rgafd_outputs[name]).all(), name)
            self.assertTrue(torch.equal(rgafd_outputs[name], baseline_outputs[name]), name)
        self.assertEqual(float((rgafd_outputs["mask"] - baseline_outputs["mask"]).abs().max()), 0.0)
        del baseline_outputs, rgafd_outputs, inputs

        refiner = RouteGuidedAnisotropicDiffusionRefiner(40).eval()
        feature = torch.randn(2, 40, 384, 544, requires_grad=True)
        route_logits = torch.randn(2, 4, 96, 136, requires_grad=True)
        directional_delta, route_confidence = refiner._detached_half_resolution_context(feature, route_logits)
        with torch.no_grad():
            refined = refiner(feature, route_logits)
        self.assertEqual(tuple(directional_delta.shape), (2, 40, 384, 544))
        self.assertEqual(tuple(route_confidence.shape), (2, 1, 384, 544))
        self.assertEqual(tuple(refined.shape), (2, 40, 384, 544))
        self.assertTrue(torch.isfinite(directional_delta).all())
        self.assertTrue(torch.isfinite(refined).all())
        self.assertFalse(directional_delta.requires_grad)
        self.assertFalse(route_confidence.requires_grad)
        self.assertGreaterEqual(float(route_confidence.min()), 0.0)
        self.assertLessEqual(float(route_confidence.max()), 1.0)
        self.assertTrue(torch.equal(refined, feature))
        self.assertEqual(sum(parameter.numel() for parameter in refiner.parameters()), 1380)
        del directional_delta, route_confidence, refined, feature, route_logits

        gradient_feature = torch.randn(1, 40, 64, 80, requires_grad=True)
        gradient_routes = torch.randn(1, 4, 16, 20, requires_grad=True)
        refiner(gradient_feature, gradient_routes).square().mean().backward()
        self.assertIsNotNone(gradient_feature.grad)
        self.assertTrue(torch.isfinite(gradient_feature.grad).all())
        self.assertIsNone(gradient_routes.grad)

        boundary_feature = torch.ones(1, 40, 12, 16)
        boundary_column = 8
        boundary_feature[..., boundary_column:] = -1.0
        directional_deltas, affinities = refiner.directional_deltas_and_affinities(boundary_feature)
        horizontal_delta = directional_deltas[0]
        left = refiner._replicate_shift(boundary_feature, 0, -1)
        right = refiner._replicate_shift(boundary_feature, 0, 1)
        naive_horizontal_delta = 0.5 * ((left - boundary_feature) + (right - boundary_feature))
        boundary_slice = (slice(None), slice(None), slice(None), slice(boundary_column - 1, boundary_column + 1))
        self.assertLess(float(affinities["right"][..., boundary_column - 1].max()), 1e-6)
        self.assertLess(float(affinities["left"][..., boundary_column].max()), 1e-6)
        self.assertLess(
            float(horizontal_delta[boundary_slice].abs().mean()),
            0.01 * float(naive_horizontal_delta[boundary_slice].abs().mean()),
        )

        horizontal_route_logits = torch.full((1, 4, 3, 4), -10.0)
        horizontal_route_logits[:, 0] = 10.0
        _, horizontal_confidence = refiner.route_probabilities_and_confidence(
            horizontal_route_logits, boundary_feature.shape[-2:]
        )
        self.assertGreater(float(horizontal_confidence.mean()), 0.99)

        with self.assertRaisesRegex(ValueError, "cannot be enabled together"):
            DEMambaNet(**kwargs, use_route_guided_directional_refiner=True, use_route_guided_anisotropic_diffusion=True)

        config_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_RGAFD.yaml")
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        self.assertTrue(config["model"]["use_route_guided_anisotropic_diffusion"])
        for name in (
            "use_frequency_gate",
            "use_semantic_skip_gate",
            "use_detail_residual_gate",
            "use_structure_harmonic_fusion",
            "use_boundary_logit_calibration",
            "use_sdf_head",
            "use_edge_mask_consistency_refiner",
            "use_deep_supervision",
            "use_retinex_directional_valley_prior",
            "use_route_guided_directional_refiner",
            "use_local_route_gate",
            "use_deep_identity_decay",
        ):
            self.assertFalse(config["model"][name], name)

    def test_semantic_skip_gate_is_initially_an_identity(self) -> None:
        gate = SemanticSkipGate(in_channels=16, skip_channels=8).eval()
        decoder_context = torch.randn(2, 16, 7, 9)
        skip = torch.randn(2, 8, 7, 9)

        with torch.no_grad():
            gated_skip = gate(decoder_context, skip)

        self.assertTrue(torch.equal(gated_skip, skip))

    def test_sgf_forward_matches_baseline_shapes_and_is_finite(self) -> None:
        baseline = DEMambaNet(
            base_channels=8,
            use_mamba_ssm=False,
            transformer_stages=[],
            use_semantic_skip_gate=False,
        ).eval()
        sgf = DEMambaNet(
            base_channels=8,
            use_mamba_ssm=False,
            transformer_stages=[],
            use_semantic_skip_gate=True,
        ).eval()
        x = torch.randn(1, 3, 32, 40)

        with torch.no_grad():
            baseline_outputs = baseline(x)
            sgf_outputs = sgf(x)

        for name in ("mask", "edge", "reliability", "route_s3", "route_s4"):
            self.assertEqual(sgf_outputs[name].shape, baseline_outputs[name].shape)
            self.assertTrue(torch.isfinite(sgf_outputs[name]).all())

        self.assertIsNone(baseline.decoder.up3.semantic_skip_gate)
        self.assertIsNone(baseline.decoder.up2.semantic_skip_gate)
        self.assertIsNone(baseline.decoder.up1.semantic_skip_gate)
        self.assertIsNotNone(sgf.decoder.up3.semantic_skip_gate)
        self.assertIsNotNone(sgf.decoder.up2.semantic_skip_gate)
        self.assertIsNotNone(sgf.decoder.up1.semantic_skip_gate)

        baseline_params = sum(parameter.numel() for parameter in baseline.parameters())
        sgf_params = sum(parameter.numel() for parameter in sgf.parameters())
        gate_params = sum(
            parameter.numel()
            for up_block in (sgf.decoder.up3, sgf.decoder.up2, sgf.decoder.up1)
            for parameter in up_block.semantic_skip_gate.parameters()
        )
        self.assertEqual(sgf_params - baseline_params, gate_params)
        self.assertGreater(gate_params, 0)

    def test_detail_residual_skip_gate_starts_as_identity_and_can_change_skip(self) -> None:
        gate = DetailResidualSkipGate(decoder_channels=16, skip_channels=8).eval()
        decoder_context = torch.randn(2, 16, 7, 9)
        skip = torch.randn(2, 8, 7, 9)

        with torch.no_grad():
            self.assertTrue(torch.equal(gate(decoder_context, skip), skip))
            gate.alpha.fill_(1.0)
            gate.gate_logits.bias.fill_(1.0)
            self.assertFalse(torch.equal(gate(decoder_context, skip), skip))

    def test_dsr_default_compatibility_and_seed_stable_shared_parameters(self) -> None:
        kwargs = {"base_channels": 8, "use_mamba_ssm": False, "transformer_stages": []}
        torch.manual_seed(42)
        baseline = DEMambaNet(**kwargs).eval()
        torch.manual_seed(42)
        dsr = DEMambaNet(**kwargs, use_detail_residual_gate=True).eval()

        self.assertIsNone(baseline.decoder.detail_gates)
        self.assertIsNotNone(dsr.decoder.detail_gates)
        baseline_parameters = dict(baseline.named_parameters())
        shared_dsr_parameters = {
            name: parameter
            for name, parameter in dsr.named_parameters()
            if not name.startswith("decoder.detail_gates.")
        }
        self.assertEqual(set(baseline_parameters), set(shared_dsr_parameters))
        for name, parameter in baseline_parameters.items():
            self.assertTrue(torch.equal(parameter, shared_dsr_parameters[name]), name)

        inputs = torch.randn(1, 3, 32, 40)
        with torch.no_grad():
            baseline_outputs = baseline(inputs)
            outputs = dsr(inputs)
        for name in ("mask", "edge", "reliability"):
            self.assertEqual(tuple(outputs[name].shape), (1, 1, 32, 40))
            self.assertTrue(torch.isfinite(outputs[name]).all())
        for name in ("mask", "edge", "reliability", "route_s3", "route_s4"):
            self.assertTrue(torch.equal(outputs[name], baseline_outputs[name]), name)

        baseline_params = sum(parameter.numel() for parameter in baseline.parameters())
        dsr_params = sum(parameter.numel() for parameter in dsr.parameters())
        detail_gate_params = sum(parameter.numel() for parameter in dsr.decoder.detail_gates.parameters())
        self.assertEqual(dsr_params - baseline_params, detail_gate_params)
        self.assertGreater(detail_gate_params, 0)
        self.assertLess(detail_gate_params, baseline_params)

    def test_harmonic_fusion_is_initially_identity_and_preserves_shared_seed_parameters(self) -> None:
        fusion = StructureConsistentHarmonicFusion([8, 16, 32, 64], decoder_channels=8).eval()
        decoder_feature = torch.randn(2, 8, 17, 19)
        features = [
            torch.randn(2, 8, 17, 19),
            torch.randn(2, 16, 9, 10),
            torch.randn(2, 32, 5, 5),
            torch.randn(2, 64, 3, 3),
        ]
        with torch.no_grad():
            self.assertTrue(torch.equal(fusion(decoder_feature, features), decoder_feature))
            fusion.residual.bias.fill_(1.0)
            changed = fusion(decoder_feature, features)
        self.assertFalse(torch.equal(changed, decoder_feature))
        self.assertTrue(torch.isfinite(changed).all())

        kwargs = {"base_channels": 8, "use_mamba_ssm": False, "transformer_stages": []}
        torch.manual_seed(42)
        baseline = DEMambaNet(**kwargs).eval()
        torch.manual_seed(42)
        schf = DEMambaNet(**kwargs, use_structure_harmonic_fusion=True).eval()
        self.assertIsNone(baseline.decoder.structure_harmonic_fusion)
        self.assertIsNotNone(schf.decoder.structure_harmonic_fusion)

        baseline_parameters = dict(baseline.named_parameters())
        shared_schf_parameters = {
            name: parameter
            for name, parameter in schf.named_parameters()
            if not name.startswith("decoder.structure_harmonic_fusion.")
        }
        self.assertEqual(set(baseline_parameters), set(shared_schf_parameters))
        for name, parameter in baseline_parameters.items():
            self.assertTrue(torch.equal(parameter, shared_schf_parameters[name]), name)

        inputs = torch.randn(1, 3, 32, 40)
        with torch.no_grad():
            baseline_outputs = baseline(inputs)
            outputs = schf(inputs)
        for name in ("mask", "edge", "reliability", "route_s3", "route_s4"):
            self.assertEqual(outputs[name].shape, baseline_outputs[name].shape)
            self.assertTrue(torch.isfinite(outputs[name]).all())
            self.assertTrue(torch.equal(outputs[name], baseline_outputs[name]), name)

        baseline_params = sum(parameter.numel() for parameter in baseline.parameters())
        schf_params = sum(parameter.numel() for parameter in schf.parameters())
        fusion_params = sum(parameter.numel() for parameter in schf.decoder.structure_harmonic_fusion.parameters())
        self.assertEqual(schf_params - baseline_params, fusion_params)
        self.assertGreater(fusion_params, 0)
        self.assertLess(fusion_params, baseline_params)

    def test_boundary_logit_calibration_identity_and_edge_gradient(self) -> None:
        calibrator = BoundaryLogitCalibrator(decoder_channels=8).eval()
        feature = torch.randn(2, 8, 9, 11)
        mask_logits = torch.randn(2, 1, 9, 11)
        edge_logits = torch.randn(2, 1, 9, 11)
        with torch.no_grad():
            self.assertTrue(torch.equal(calibrator(feature, mask_logits, edge_logits), mask_logits))
            calibrator.delta_out.bias.fill_(1.0)
            calibrated_logits = calibrator(feature, mask_logits, edge_logits)
        self.assertFalse(torch.equal(calibrated_logits, mask_logits))
        self.assertTrue(torch.isfinite(calibrated_logits).all())

        kwargs = {"base_channels": 8, "use_mamba_ssm": False, "transformer_stages": []}
        torch.manual_seed(42)
        baseline = DEMambaNet(**kwargs).eval()
        torch.manual_seed(42)
        blc = DEMambaNet(**kwargs, use_boundary_logit_calibration=True).eval()
        self.assertIsNone(baseline.decoder.boundary_logit_calibrator)
        self.assertIsNotNone(blc.decoder.boundary_logit_calibrator)

        baseline_parameters = dict(baseline.named_parameters())
        shared_blc_parameters = {
            name: parameter
            for name, parameter in blc.named_parameters()
            if not name.startswith("decoder.boundary_logit_calibrator.")
        }
        self.assertEqual(set(baseline_parameters), set(shared_blc_parameters))
        for name, parameter in baseline_parameters.items():
            self.assertTrue(torch.equal(parameter, shared_blc_parameters[name]), name)

        inputs = torch.randn(2, 3, 32, 40)
        with torch.no_grad():
            baseline_outputs = baseline(inputs)
            outputs = blc(inputs)
        for name in ("mask", "edge", "reliability", "route_s3", "route_s4"):
            self.assertEqual(outputs[name].shape, baseline_outputs[name].shape)
            self.assertTrue(torch.isfinite(outputs[name]).all())
            self.assertTrue(torch.equal(outputs[name], baseline_outputs[name]), name)

        baseline_params = sum(parameter.numel() for parameter in baseline.parameters())
        blc_params = sum(parameter.numel() for parameter in blc.parameters())
        calibrator_params = sum(parameter.numel() for parameter in blc.decoder.boundary_logit_calibrator.parameters())
        self.assertEqual(blc_params - baseline_params, calibrator_params)
        self.assertGreater(calibrator_params, 0)
        self.assertLess(calibrator_params, 2_000)

        blc.zero_grad(set_to_none=True)
        with torch.no_grad():
            blc.decoder.boundary_logit_calibrator.delta_out.bias.fill_(1.0)
        blc_outputs = blc(inputs)
        blc_outputs["mask"].square().mean().backward()
        edge_gradients = [parameter.grad for parameter in blc.decoder.edge_head.parameters()]
        self.assertTrue(any(gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0 for gradient in edge_gradients))

    def test_emcr_identity_seed_stability_and_blc_exclusivity(self) -> None:
        refiner = EdgeMaskConsistencyRefiner(decoder_channels=8).eval()
        decoder_feature = torch.randn(2, 8, 9, 11)
        mask_logits = torch.randn(2, 1, 9, 11)
        edge_logits = torch.randn(2, 1, 9, 11)
        with torch.no_grad():
            refined_logits = refiner(decoder_feature, mask_logits, edge_logits)
        self.assertEqual(refined_logits.shape, mask_logits.shape)
        self.assertTrue(torch.equal(refined_logits, mask_logits))

        kwargs = {"base_channels": 8, "use_mamba_ssm": False, "transformer_stages": []}
        torch.manual_seed(42)
        native = DEMambaNet(**kwargs).eval()
        torch.manual_seed(42)
        emcr = DEMambaNet(**kwargs, use_edge_mask_consistency_refiner=True).eval()
        self.assertIsNone(native.decoder.edge_mask_consistency_refiner)
        self.assertIsNotNone(emcr.decoder.edge_mask_consistency_refiner)
        native_parameters = dict(native.named_parameters())
        shared_emcr_parameters = {
            name: parameter
            for name, parameter in emcr.named_parameters()
            if not name.startswith("decoder.edge_mask_consistency_refiner.")
        }
        self.assertEqual(set(native_parameters), set(shared_emcr_parameters))
        for name, parameter in native_parameters.items():
            self.assertTrue(torch.equal(parameter, shared_emcr_parameters[name]), name)

        input_tensor = torch.randn(1, 3, 384, 544)
        with torch.no_grad():
            native_outputs = native(input_tensor)
            emcr_outputs = emcr(input_tensor)
        for name in ("mask", "edge", "reliability", "route_s3", "route_s4"):
            self.assertEqual(tuple(emcr_outputs[name].shape), tuple(native_outputs[name].shape), name)
            self.assertTrue(torch.isfinite(emcr_outputs[name]).all(), name)
            self.assertTrue(torch.equal(emcr_outputs[name], native_outputs[name]), name)

        native_params = sum(parameter.numel() for parameter in native.parameters())
        emcr_params = sum(parameter.numel() for parameter in emcr.parameters())
        refiner_params = sum(parameter.numel() for parameter in emcr.decoder.edge_mask_consistency_refiner.parameters())
        self.assertEqual(emcr_params - native_params, refiner_params)
        self.assertGreater(refiner_params, 0)

        with self.assertRaisesRegex(ValueError, "BoundaryLogitCalibrator and EdgeMaskConsistencyRefiner"):
            DEMambaNet(**kwargs, use_boundary_logit_calibration=True, use_edge_mask_consistency_refiner=True)

        config_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMCR.yaml")
        with config_path.open(encoding="utf-8") as handle:
            emcr_config = yaml.safe_load(handle)
        self.assertTrue(emcr_config["model"]["use_edge_mask_consistency_refiner"])
        self.assertFalse(emcr_config["model"]["use_boundary_logit_calibration"])

    def test_lscc_is_identity_and_rejects_semantic_boundary_bridges(self) -> None:
        refiner = LocalSemanticContinuityCalibrator(decoder_channels=8, semantic_channels=16).eval()
        decoder_feature = torch.randn(2, 8, 32, 40)
        deep_feature = torch.randn(2, 16, 8, 10)
        mask_logits = torch.randn(2, 1, 32, 40)
        edge_logits = torch.randn(2, 1, 32, 40)
        with torch.no_grad():
            refined_logits = refiner(decoder_feature, deep_feature, mask_logits, edge_logits)
        self.assertEqual(refined_logits.shape, mask_logits.shape)
        self.assertTrue(torch.equal(refined_logits, mask_logits))

        # Every local neighbor has a different semantic descriptor from the
        # center, although each neighbor holds high crack evidence.  The
        # absolute-agreement term must reject that possible background bridge.
        semantic = torch.zeros(1, 2, 5, 5)
        semantic[:, 0] = 1.0
        semantic[0, :, 2, 2] = torch.tensor([1.0, 0.0])
        for row in range(1, 4):
            for column in range(1, 4):
                if (row, column) != (2, 2):
                    semantic[0, :, row, column] = torch.tensor([0.0, 1.0])
        evidence = torch.zeros(1, 1, 5, 5)
        evidence[:, :, 1:4, 1:4] = 1.0
        evidence[:, :, 2, 2] = 0.0
        _, agreement = refiner.local_support_and_agreement(semantic, evidence)
        self.assertLess(float(agreement[0, 0, 2, 2]), 0.01)

        kwargs = {"base_channels": 8, "use_mamba_ssm": False, "transformer_stages": []}
        torch.manual_seed(42)
        native = DEMambaNet(**kwargs).eval()
        torch.manual_seed(42)
        lscc = DEMambaNet(**kwargs, use_local_semantic_continuity_calibrator=True).eval()
        self.assertIsNone(native.decoder.local_semantic_continuity_calibrator)
        self.assertIsNotNone(lscc.decoder.local_semantic_continuity_calibrator)
        native_state = native.state_dict()
        shared_state = {
            name: value
            for name, value in lscc.state_dict().items()
            if not name.startswith("decoder.local_semantic_continuity_calibrator.")
        }
        self.assertEqual(set(native_state), set(shared_state))
        for name, value in native_state.items():
            self.assertTrue(torch.equal(value, shared_state[name]), name)

        inputs = torch.randn(1, 3, 64, 80)
        with torch.no_grad():
            native_outputs = native(inputs)
            lscc_outputs = lscc(inputs)
        for name in native_outputs:
            self.assertTrue(torch.equal(lscc_outputs[name], native_outputs[name]), name)

        native_params = sum(parameter.numel() for parameter in native.parameters())
        lscc_params = sum(parameter.numel() for parameter in lscc.parameters())
        refiner_params = sum(parameter.numel() for parameter in lscc.decoder.local_semantic_continuity_calibrator.parameters())
        self.assertEqual(lscc_params - native_params, refiner_params)
        self.assertGreater(refiner_params, 0)
        self.assertLess(refiner_params, 5_000)

        config_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_LSCC.yaml")
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        self.assertTrue(config["model"]["use_local_semantic_continuity_calibrator"])
        for name in (
            "use_frequency_gate",
            "use_semantic_skip_gate",
            "use_detail_residual_gate",
            "use_structure_harmonic_fusion",
            "use_boundary_logit_calibration",
            "use_sdf_head",
            "use_edge_mask_consistency_refiner",
            "use_deep_supervision",
            "use_retinex_directional_valley_prior",
            "use_route_guided_directional_refiner",
            "use_route_guided_anisotropic_diffusion",
            "use_density_balanced_prototype_context",
            "use_local_route_gate",
            "use_deep_identity_decay",
        ):
            self.assertFalse(config["model"][name], name)

    def test_deep_supervision_is_train_only_and_auxiliary_loss_is_gated(self) -> None:
        kwargs = {"base_channels": 8, "use_mamba_ssm": False, "transformer_stages": []}
        torch.manual_seed(42)
        native = DEMambaNet(**kwargs).eval()
        torch.manual_seed(42)
        deep_supervision = DEMambaNet(**kwargs, use_deep_supervision=True).eval()
        for name in ("aux_mask_quarter", "aux_edge_quarter", "aux_mask_half", "aux_edge_half"):
            self.assertIsNone(getattr(native.decoder, name))
            self.assertIsNotNone(getattr(deep_supervision.decoder, name))

        native_parameters = dict(native.named_parameters())
        shared_parameters = {
            name: parameter for name, parameter in deep_supervision.named_parameters() if not name.startswith("decoder.aux_")
        }
        self.assertEqual(set(native_parameters), set(shared_parameters))
        for name, parameter in native_parameters.items():
            self.assertTrue(torch.equal(parameter, shared_parameters[name]), name)

        input_tensor = torch.randn(1, 3, 384, 544)
        with torch.no_grad():
            native_outputs = native(input_tensor)
            eval_outputs = deep_supervision(input_tensor)
        self.assertFalse(any(name.startswith("aux_") for name in eval_outputs))
        for name in ("mask", "edge", "reliability", "route_s3", "route_s4"):
            self.assertEqual(tuple(eval_outputs[name].shape), tuple(native_outputs[name].shape), name)
            self.assertTrue(torch.equal(eval_outputs[name], native_outputs[name]), name)

        deep_supervision.train()
        train_outputs = deep_supervision(input_tensor)
        for name in ("aux_mask_half", "aux_edge_half"):
            self.assertEqual(tuple(train_outputs[name].shape), (1, 1, 192, 272), name)
        for name in ("aux_mask_quarter", "aux_edge_quarter"):
            self.assertEqual(tuple(train_outputs[name].shape), (1, 1, 96, 136), name)
        self.assertEqual(tuple(train_outputs["mask"].shape), (1, 1, 384, 544))

        native_params = sum(parameter.numel() for parameter in native.parameters())
        deep_supervision_params = sum(parameter.numel() for parameter in deep_supervision.parameters())
        auxiliary_params = sum(
            parameter.numel()
            for name in ("aux_mask_quarter", "aux_edge_quarter", "aux_mask_half", "aux_edge_half")
            for parameter in getattr(deep_supervision.decoder, name).parameters()
        )
        self.assertEqual(deep_supervision_params - native_params, auxiliary_params)

        outputs = {
            "mask": torch.zeros(1, 1, 4, 6, requires_grad=True),
            "edge": torch.zeros(1, 1, 4, 6, requires_grad=True),
            "aux_mask_half": torch.randn(1, 1, 2, 3, requires_grad=True),
            "aux_mask_quarter": torch.randn(1, 1, 1, 2, requires_grad=True),
            "aux_edge_half": torch.randn(1, 1, 2, 3, requires_grad=True),
            "aux_edge_quarter": torch.randn(1, 1, 1, 2, requires_grad=True),
        }
        batch = {
            "mask": torch.randint(0, 2, (1, 1, 4, 6), dtype=torch.float32),
            "edge_gt": torch.randint(0, 2, (1, 1, 4, 6), dtype=torch.float32),
            "tangent_gt": torch.zeros(1, 4, 4, 6),
            "tangent_valid": torch.zeros(1, 1, 4, 6),
        }
        criterion = DEMambaLoss(
            lambda_edge=0.0,
            lambda_tangent=0.0,
            scnp_weight=0.0,
            lambda_aux_mask=0.15,
            lambda_aux_edge=0.05,
        )
        total, items = criterion(outputs, batch, epoch=0)
        self.assertTrue(torch.isfinite(total))
        self.assertGreater(items["aux_mask_loss"], 0.0)
        self.assertGreater(items["aux_edge_loss"], 0.0)
        self.assertEqual(items["lambda_aux_mask"], 0.15)
        self.assertEqual(items["lambda_aux_edge"], 0.05)
        total.backward()
        self.assertTrue(torch.isfinite(outputs["aux_mask_half"].grad).all())
        self.assertTrue(torch.isfinite(outputs["aux_edge_half"].grad).all())

        with self.assertRaisesRegex(RuntimeError, "lambda_aux_mask > 0"):
            DEMambaLoss(lambda_aux_mask=0.1)({key: value for key, value in outputs.items() if key != "aux_mask_half"}, batch)
        with self.assertRaisesRegex(RuntimeError, "lambda_aux_edge > 0"):
            DEMambaLoss(lambda_aux_edge=0.1)({key: value for key, value in outputs.items() if key != "aux_edge_half"}, batch)
        default_total, default_items = DEMambaLoss(lambda_edge=0.0, lambda_tangent=0.0, scnp_weight=0.0)(
            {key: value for key, value in outputs.items() if not key.startswith("aux_")}, batch
        )
        self.assertTrue(torch.isfinite(default_total))
        self.assertEqual(default_items["aux_mask_loss"], 0.0)
        self.assertEqual(default_items["aux_edge_loss"], 0.0)

        with self.assertRaisesRegex(ValueError, "lambda_aux_mask and lambda_aux_edge"):
            DEMambaLoss(lambda_aux_mask=-0.1)

        config_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_DEEPSUP.yaml")
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        self.assertTrue(config["model"]["use_deep_supervision"])
        self.assertEqual(config["loss"]["lambda_aux_mask"], 0.15)
        self.assertEqual(config["loss"]["lambda_aux_edge"], 0.05)

    def test_sdf_head_is_auxiliary_and_preserves_shared_seed_parameters(self) -> None:
        kwargs = {"base_channels": 8, "use_mamba_ssm": False, "transformer_stages": []}
        torch.manual_seed(42)
        baseline = DEMambaNet(**kwargs).eval()
        torch.manual_seed(42)
        sdf_model = DEMambaNet(**kwargs, use_sdf_head=True).eval()

        self.assertIsNone(baseline.decoder.sdf_head)
        self.assertIsNotNone(sdf_model.decoder.sdf_head)
        self.assertTrue(torch.equal(sdf_model.decoder.sdf_head.out.weight, torch.zeros_like(sdf_model.decoder.sdf_head.out.weight)))
        self.assertTrue(torch.equal(sdf_model.decoder.sdf_head.out.bias, torch.zeros_like(sdf_model.decoder.sdf_head.out.bias)))
        baseline_parameters = dict(baseline.named_parameters())
        shared_sdf_parameters = {
            name: parameter for name, parameter in sdf_model.named_parameters() if not name.startswith("decoder.sdf_head.")
        }
        self.assertEqual(set(baseline_parameters), set(shared_sdf_parameters))
        for name, parameter in baseline_parameters.items():
            self.assertTrue(torch.equal(parameter, shared_sdf_parameters[name]), name)

        inputs = torch.randn(1, 3, 32, 40)
        with torch.no_grad():
            baseline_outputs = baseline(inputs)
            sdf_outputs = sdf_model(inputs)
        self.assertNotIn("sdf", baseline_outputs)
        self.assertNotIn("sdf", sdf_outputs)
        for name in ("mask", "edge", "reliability", "route_s3", "route_s4"):
            self.assertTrue(torch.equal(baseline_outputs[name], sdf_outputs[name]), name)

        sdf_model.train()
        train_outputs = sdf_model(inputs)
        self.assertEqual(tuple(train_outputs["sdf"].shape), (1, 1, 32, 40))
        self.assertTrue(torch.isfinite(train_outputs["sdf"]).all())

        baseline_params = sum(parameter.numel() for parameter in baseline.parameters())
        sdf_params = sum(parameter.numel() for parameter in sdf_model.parameters())
        head_params = sum(parameter.numel() for parameter in sdf_model.decoder.sdf_head.parameters())
        self.assertEqual(sdf_params - baseline_params, head_params)

    @unittest.skipIf(distance_transform_edt is None, "SciPy is not installed")
    def test_truncated_sdf_targets_and_narrow_band_loss(self) -> None:
        mask = np.zeros((17, 19), dtype=np.float32)
        mask[7:10, 8:12] = 1.0
        sdf_target, sdf_valid = build_truncated_sdf_targets(mask, truncation=2.0)
        self.assertEqual(sdf_target.dtype, np.float32)
        self.assertEqual(sdf_valid.dtype, np.float32)
        self.assertEqual(sdf_target.shape, (1, 17, 19))
        self.assertEqual(sdf_valid.shape, (1, 17, 19))
        self.assertGreater(sdf_target[0, 8, 9], 0.0)
        self.assertLess(sdf_target[0, 0, 0], 0.0)
        self.assertEqual(sdf_valid[0, 0, 0], 0.0)
        self.assertGreater(sdf_valid.sum(), 0.0)

        prediction = torch.randn(1, 1, 17, 19, requires_grad=True)
        target_tensor = torch.from_numpy(sdf_target)[None]
        valid_tensor = torch.from_numpy(sdf_valid)[None]
        loss = TruncatedSDFLoss()(prediction, target_tensor, valid_tensor)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertEqual(TruncatedSDFLoss()(prediction.detach(), target_tensor, torch.zeros_like(valid_tensor)).item(), 0.0)

    @unittest.skipIf(distance_transform_edt is None, "SciPy is not installed")
    def test_sdf_dataset_target_uses_final_letterbox_mask_and_loss_is_gated(self) -> None:
        image = np.full((9, 13, 3), 127, dtype=np.uint8)
        mask = np.zeros((9, 13), dtype=np.uint8)
        mask[3:6, 5:8] = 255
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for kind in ("images", "masks", "targets"):
                (root / "train" / kind).mkdir(parents=True)
            Image.fromarray(image).save(root / "train" / "images" / "sample.png")
            Image.fromarray(mask).save(root / "train" / "masks" / "sample.png")
            np.savez_compressed(
                root / "train" / "targets" / "sample.npz",
                edge_gt=np.zeros((1, 9, 13), dtype=np.float32),
                tangent_gt=np.zeros((4, 9, 13), dtype=np.float32),
                tangent_valid=np.zeros((1, 9, 13), dtype=np.float32),
            )
            sdf_item = CrackSegmentationDataset(root, "train", img_size=(384, 544), augment=False, return_sdf_target=True)[0]
            self.assertEqual(tuple(sdf_item["image"].shape), (3, 384, 544))
            for key in ("mask", "sdf_target", "sdf_valid"):
                self.assertEqual(tuple(sdf_item[key].shape), (1, 384, 544), key)
            expected_sdf, expected_valid = build_truncated_sdf_targets(sdf_item["mask"][0].numpy())
            self.assertTrue(torch.equal(sdf_item["sdf_target"], torch.from_numpy(expected_sdf)))
            self.assertTrue(torch.equal(sdf_item["sdf_valid"], torch.from_numpy(expected_valid)))
            legacy_item = CrackSegmentationDataset(root, "train", img_size=(384, 544), augment=False)[0]
            self.assertNotIn("sdf_target", legacy_item)
            self.assertNotIn("sdf_valid", legacy_item)

        config_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_SDF.yaml")
        with config_path.open(encoding="utf-8") as handle:
            sdf_config = yaml.safe_load(handle)
        self.assertTrue(sdf_config["model"]["use_sdf_head"])
        self.assertEqual(sdf_config["loss"]["lambda_sdf"], 0.05)
        self.assertEqual(sdf_config["loss"]["sdf_truncation"], 8.0)
        with tempfile.TemporaryDirectory() as directory:
            config_path, saved = save_resolved_config(Path(directory), sdf_config)
            self.assertTrue(saved)
            with config_path.open(encoding="utf-8") as handle:
                resolved = yaml.safe_load(handle)
            self.assertTrue(resolved["model"]["use_sdf_head"])
            self.assertEqual(resolved["loss"]["lambda_sdf"], 0.05)
            self.assertEqual(resolved["loss"]["sdf_truncation"], 8.0)

        outputs = {
            "mask": torch.zeros(1, 1, 2, 3, requires_grad=True),
            "edge": torch.zeros(1, 1, 2, 3, requires_grad=True),
            "sdf": torch.ones(1, 1, 2, 3, requires_grad=True),
        }
        batch = {
            "mask": torch.zeros(1, 1, 2, 3),
            "edge_gt": torch.zeros(1, 1, 2, 3),
            "tangent_gt": torch.zeros(1, 4, 2, 3),
            "tangent_valid": torch.zeros(1, 1, 2, 3),
            "sdf_target": torch.zeros(1, 1, 2, 3),
            "sdf_valid": torch.ones(1, 1, 2, 3),
        }
        criterion = DEMambaLoss(lambda_edge=0.0, lambda_tangent=0.0, scnp_weight=0.0, lambda_sdf=0.05)
        total, items = criterion(outputs, batch, epoch=0)
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(np.isfinite(items["sdf_loss"]))
        self.assertEqual(items["lambda_sdf"], 0.05)
        total.backward()
        self.assertTrue(torch.isfinite(outputs["sdf"].grad).all())

        with self.assertRaisesRegex(RuntimeError, "outputs\\['sdf'\\]"):
            criterion({key: value for key, value in outputs.items() if key != "sdf"}, batch, epoch=0)
        default_total, default_items = DEMambaLoss(lambda_edge=0.0, lambda_tangent=0.0, scnp_weight=0.0)(
            {key: value for key, value in outputs.items() if key != "sdf"},
            {key: value for key, value in batch.items() if not key.startswith("sdf_")},
            epoch=0,
        )
        self.assertTrue(torch.isfinite(default_total))
        self.assertEqual(default_items["lambda_sdf"], 0.0)

    def test_local_route_gate_identity_and_seed_stable_model(self) -> None:
        gate = LocalRouteConsistencyGate(channels=16).eval()
        x = torch.randn(2, 16, 9, 11)
        local = x + torch.randn_like(x)
        routed = x + torch.randn_like(x)
        with torch.no_grad():
            self.assertTrue(torch.equal(gate(x, local, routed), routed))
            for bias in (1.0, -1.0):
                gate.gate_out.bias.fill_(bias)
                gate_scale = gate.gate_scale(x, local, routed)
                calibrated_routed = gate(x, local, routed)
                self.assertTrue(torch.all(gate_scale >= 0.5))
                self.assertTrue(torch.all(gate_scale <= 1.5))
                self.assertFalse(torch.equal(calibrated_routed, routed))
                self.assertTrue(torch.isfinite(calibrated_routed).all())

        kwargs = {"base_channels": 8, "use_mamba_ssm": False, "transformer_stages": []}
        torch.manual_seed(42)
        baseline = DEMambaNet(**kwargs).eval()
        torch.manual_seed(42)
        lrg = DEMambaNet(**kwargs, use_local_route_gate=True).eval()
        self.assertIsNone(baseline.local_route_gates)
        self.assertIsNotNone(lrg.local_route_gates)

        baseline_parameters = dict(baseline.named_parameters())
        shared_lrg_parameters = {
            name: parameter
            for name, parameter in lrg.named_parameters()
            if not name.startswith("local_route_gates.")
        }
        self.assertEqual(set(baseline_parameters), set(shared_lrg_parameters))
        for name, parameter in baseline_parameters.items():
            self.assertTrue(torch.equal(parameter, shared_lrg_parameters[name]), name)

        inputs = torch.randn(1, 3, 32, 40)
        with torch.no_grad():
            baseline_outputs = baseline(inputs)
            outputs = lrg(inputs)
        for name in ("mask", "edge", "reliability", "route_s3", "route_s4"):
            self.assertEqual(outputs[name].shape, baseline_outputs[name].shape)
            self.assertTrue(torch.isfinite(outputs[name]).all())
            self.assertTrue(torch.equal(outputs[name], baseline_outputs[name]), name)
        self.assertEqual(lrg.mamba_backend_info(), baseline.mamba_backend_info())
        self.assertEqual(set(lrg.mamba_backend_info()), {"stage3", "stage4"})

        baseline_params = sum(parameter.numel() for parameter in baseline.parameters())
        lrg_params = sum(parameter.numel() for parameter in lrg.parameters())
        gate_params = sum(parameter.numel() for parameter in lrg.local_route_gates.parameters())
        self.assertEqual(lrg_params - baseline_params, gate_params)
        self.assertGreater(gate_params, 0)
        self.assertLess(gate_params, 500_000)

    def test_deep_identity_decay_is_parameter_free_and_seed_stable(self) -> None:
        block = TopoRouteBlock(channels=8, use_mamba_ssm=False).eval()
        feature = torch.randn(1, 8, 9, 11)
        with torch.no_grad():
            original_output, original_routes = block(feature)
            identity_output, identity_routes = block(feature, identity_scale=1.0)
            decayed_output, decayed_routes = block(feature, identity_scale=0.90)
        self.assertTrue(torch.equal(identity_output, original_output))
        self.assertTrue(torch.equal(identity_routes, original_routes))
        self.assertEqual(decayed_output.shape, original_output.shape)
        self.assertTrue(torch.isfinite(decayed_output).all())
        self.assertFalse(torch.equal(decayed_output, original_output))
        self.assertTrue(torch.equal(decayed_routes, original_routes))
        with self.assertRaises(ValueError):
            block(feature, identity_scale=0.0)

        kwargs = {"base_channels": 8, "use_mamba_ssm": False, "transformer_stages": []}
        torch.manual_seed(42)
        baseline = DEMambaNet(**kwargs).eval()
        torch.manual_seed(42)
        did = DEMambaNet(
            **kwargs,
            use_deep_identity_decay=True,
            deep_identity_scales={"stage3": 0.98, "stage4": 0.90},
        ).eval()
        self.assertEqual(baseline.deep_identity_scales, {"stage3": 1.0, "stage4": 1.0})
        self.assertEqual(did.deep_identity_scales, {"stage3": 0.98, "stage4": 0.90})
        self.assertEqual(sum(parameter.numel() for parameter in did.parameters()), sum(parameter.numel() for parameter in baseline.parameters()))

        baseline_state = baseline.state_dict()
        did_state = did.state_dict()
        self.assertEqual(set(baseline_state), set(did_state))
        for name, value in baseline_state.items():
            self.assertTrue(torch.equal(value, did_state[name]), name)

        inputs = torch.randn(1, 3, 32, 40)
        with torch.no_grad():
            baseline_outputs = baseline(inputs)
            did_outputs = did(inputs)
        for name in ("mask", "edge", "reliability", "route_s3", "route_s4"):
            self.assertEqual(did_outputs[name].shape, baseline_outputs[name].shape)
            self.assertTrue(torch.isfinite(did_outputs[name]).all())
        self.assertEqual(did_outputs["route_s3"].shape, baseline_outputs["route_s3"].shape)
        self.assertEqual(did_outputs["route_s4"].shape, baseline_outputs["route_s4"].shape)
        self.assertEqual(did.mamba_backend_info(), baseline.mamba_backend_info())

        config_path = Path("configs/experiments/C3_DID.yaml")
        with config_path.open(encoding="utf-8") as handle:
            did_config = yaml.safe_load(handle)
        self.assertTrue(did_config["model"]["use_deep_identity_decay"])
        self.assertEqual(did_config["model"]["deep_identity_scales"], {"stage3": 0.98, "stage4": 0.90})

    def test_adamw_excludes_norm_and_bias_weight_decay_only_when_enabled(self) -> None:
        class OptimizerFixture(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 3)
                self.norm = torch.nn.LayerNorm(3)
                self.vector = torch.nn.Parameter(torch.ones(3))
                self.scalar = torch.nn.Parameter(torch.ones(()))
                self.frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)

        class Logger:
            def __init__(self) -> None:
                self.messages: list[str] = []

            def info(self, message: str, *args) -> None:
                self.messages.append(message % args if args else message)

            def warning(self, message: str, *args) -> None:
                self.messages.append(message % args if args else message)

        model = OptimizerFixture()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        base_cfg = {"lr": 1e-4, "weight_decay": 1e-4, "fused_adamw": False}
        default_logger = Logger()
        default_optimizer = make_optimizer(
            model,
            {**base_cfg, "exclude_norm_bias_from_weight_decay": False},
            torch.device("cpu"),
            default_logger,
        )
        self.assertEqual(len(default_optimizer.param_groups), 1)
        self.assertEqual(default_optimizer.param_groups[0]["weight_decay"], 1e-4)

        grouped_logger = Logger()
        grouped_optimizer = make_optimizer(
            model,
            {**base_cfg, "exclude_norm_bias_from_weight_decay": True},
            torch.device("cpu"),
            grouped_logger,
        )
        self.assertEqual(len(grouped_optimizer.param_groups), 2)
        decay_group, no_decay_group = grouped_optimizer.param_groups
        self.assertEqual(decay_group["weight_decay"], 1e-4)
        self.assertEqual(no_decay_group["weight_decay"], 0.0)

        trainable_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
        grouped_ids = [id(parameter) for group in grouped_optimizer.param_groups for parameter in group["params"]]
        self.assertEqual(len(grouped_ids), len(set(grouped_ids)))
        self.assertEqual(set(grouped_ids), trainable_ids)
        no_decay_ids = {id(parameter) for parameter in no_decay_group["params"]}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and (name.endswith(".bias") or parameter.ndim <= 1):
                self.assertIn(id(parameter), no_decay_ids, name)
        self.assertEqual(parameter_count, sum(parameter.numel() for parameter in model.parameters()))
        self.assertTrue(any("decay:" in message and "no_decay:" in message for message in grouped_logger.messages))

    def test_photometric_augment_is_train_only_and_preserves_targets(self) -> None:
        image = np.arange(9 * 13 * 3, dtype=np.uint8).reshape(9, 13, 3)
        disabled = {"enabled": False, "probability": 1.0, "brightness": 0.2, "contrast": 0.2, "gamma_range": [0.8, 1.2]}
        probability_zero = {**disabled, "enabled": True, "probability": 0.0}
        self.assertTrue(np.array_equal(_photometric_augment(image, disabled), image))
        self.assertTrue(np.array_equal(_photometric_augment(image, probability_zero), image))

        enabled = {"enabled": True, "probability": 1.0, "brightness": 0.12, "contrast": 0.12, "gamma_range": [0.9, 1.1]}
        with patch("datasets.crack_dataset.random.uniform", side_effect=[1.12, 0.88, 0.9]):
            augmented = _photometric_augment(image, enabled)
        self.assertEqual(augmented.dtype, np.uint8)
        self.assertEqual(augmented.shape, image.shape)
        self.assertGreaterEqual(int(augmented.min()), 0)
        self.assertLessEqual(int(augmented.max()), 255)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "val", "test"):
                for kind in ("images", "masks", "targets"):
                    (root / split / kind).mkdir(parents=True)
                Image.fromarray(image).save(root / split / "images" / "sample.png")
                Image.fromarray(np.zeros((9, 13), dtype=np.uint8)).save(root / split / "masks" / "sample.png")
                np.savez_compressed(
                    root / split / "targets" / "sample.npz",
                    edge_gt=np.zeros((1, 9, 13), dtype=np.float32),
                    tangent_gt=np.zeros((4, 9, 13), dtype=np.float32),
                    tangent_valid=np.zeros((1, 9, 13), dtype=np.float32),
                )

            baseline = CrackSegmentationDataset(root, "train", img_size=(16, 16), augment=False)[0]
            with patch("datasets.crack_dataset.random.uniform", side_effect=[1.12, 0.88, 0.9]):
                train_item = CrackSegmentationDataset(
                    root,
                    "train",
                    img_size=(16, 16),
                    augment=False,
                    photometric_augment=enabled,
                )[0]
            self.assertFalse(torch.equal(train_item["image"], baseline["image"]))
            for key in ("mask", "edge_gt", "tangent_gt", "tangent_valid"):
                self.assertTrue(torch.equal(train_item[key], baseline[key]), key)

            for split in ("val", "test"):
                reference = CrackSegmentationDataset(root, split, img_size=(16, 16), augment=False)[0]
                configured = CrackSegmentationDataset(
                    root,
                    split,
                    img_size=(16, 16),
                    augment=False,
                    photometric_augment=enabled,
                )[0]
                self.assertTrue(torch.equal(configured["image"], reference["image"]), split)
                for key in ("mask", "edge_gt", "tangent_gt", "tangent_valid"):
                    self.assertTrue(torch.equal(configured[key], reference[key]), f"{split}:{key}")

    def test_color_robust_photometric_augment_is_independent_and_legacy_compatible(self) -> None:
        image = np.array([[[100, 150, 200], [20, 40, 60]]], dtype=np.uint8)
        legacy = {"enabled": True, "probability": 0.5, "brightness": 0.12, "contrast": 0.12, "gamma_range": [0.9, 1.1]}
        with patch("datasets.crack_dataset.random.random", return_value=0.75) as random_value, patch(
            "datasets.crack_dataset.random.uniform"
        ) as uniform:
            self.assertTrue(np.array_equal(_photometric_augment(image, legacy), image))
        random_value.assert_called_once()
        uniform.assert_not_called()

        channel_gain = {
            "enabled": True,
            "probability": 0.0,
            "channel_gain_probability": 1.0,
            "channel_gain_range": [0.5, 1.5],
            "grayscale_probability": 0.0,
        }
        with patch("datasets.crack_dataset.random.random", return_value=0.0), patch(
            "datasets.crack_dataset.random.uniform", side_effect=[0.5, 1.0, 1.5]
        ):
            gained = _photometric_augment(image, channel_gain)
        self.assertEqual(gained.dtype, np.uint8)
        self.assertEqual(gained.shape, image.shape)
        self.assertTrue(np.array_equal(gained[0, 0], np.array([50, 150, 255], dtype=np.uint8)))

        grayscale = {"enabled": True, "probability": 0.0, "channel_gain_probability": 0.0, "grayscale_probability": 1.0}
        with patch("datasets.crack_dataset.random.random", return_value=0.0):
            gray = _photometric_augment(image, grayscale)
        self.assertEqual(gray.dtype, np.uint8)
        self.assertEqual(gray.shape, image.shape)
        self.assertTrue(np.array_equal(gray[..., 0], gray[..., 1]))
        self.assertTrue(np.array_equal(gray[..., 1], gray[..., 2]))

        config_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_COLORROBUST.yaml")
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        color_cfg = config["data"]["photometric_augment"]
        self.assertEqual(color_cfg["channel_gain_probability"], 0.30)
        self.assertEqual(color_cfg["channel_gain_range"], [0.90, 1.10])
        self.assertEqual(color_cfg["grayscale_probability"], 0.12)
        with tempfile.TemporaryDirectory() as directory:
            saved_path, saved = save_resolved_config(Path(directory), config)
            self.assertTrue(saved)
            with saved_path.open(encoding="utf-8") as handle:
                resolved = yaml.safe_load(handle)
            self.assertEqual(resolved["data"]["photometric_augment"], color_cfg)

    def test_diagnostic_bundle_exports_input_coordinate_artifacts(self) -> None:
        image = torch.zeros(3, 4, 6)
        target = torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]])
        mask_logits = torch.full((1, 4, 6), -10.0)
        mask_logits[0, 0, 0] = 10.0
        mask_logits[0, 0, 1] = 10.0
        edge_logits = torch.zeros(1, 4, 6)
        route_s3 = torch.randn(4, 2, 3)
        route_s4 = torch.randn(4, 1, 2)
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            write_diagnostics_readme(output_dir)
            save_diagnostic_bundle(
                image,
                target,
                mask_logits,
                edge_logits,
                "sample",
                output_dir,
                threshold=0.6,
                route_s3=route_s3,
                route_s4=route_s4,
            )
            for suffix in (
                "input.png",
                "target.png",
                "mask_probability.png",
                "mask_probability.npy",
                "edge_probability.png",
                "edge_probability.npy",
                "prediction_mask.png",
                "overlay.png",
                "error_map.png",
            ):
                self.assertTrue((output_dir / f"sample_{suffix}").is_file(), suffix)
            for stage in ("route_s3", "route_s4"):
                for direction in ("horizontal", "vertical", "diagonal", "anti_diagonal"):
                    png_path = output_dir / f"sample_{stage}_{direction}.png"
                    npy_path = output_dir / f"sample_{stage}_{direction}.npy"
                    self.assertTrue(png_path.is_file(), png_path.name)
                    self.assertTrue(npy_path.is_file(), npy_path.name)
                    array = np.load(npy_path)
                    self.assertEqual(array.dtype, np.float32)
                    self.assertEqual(array.shape, (4, 6))
            self.assertEqual(np.load(output_dir / "sample_mask_probability.npy").shape, (4, 6))
            error_map = np.asarray(Image.open(output_dir / "sample_error_map.png"))
            self.assertTrue(np.array_equal(error_map[0, 0], [0, 255, 0]))
            self.assertTrue(np.array_equal(error_map[0, 1], [255, 0, 0]))
            self.assertTrue(np.array_equal(error_map[3, 5], [0, 0, 255]))
            self.assertIn("Route channel order", (output_dir / "README.txt").read_text(encoding="utf-8"))

    def test_native_aspect_ratio_rotation_disables_quarter_turns(self) -> None:
        image = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
        mask = np.arange(3 * 5, dtype=np.float32).reshape(3, 5)
        edge = mask.copy()
        tangent = np.stack([mask] * 4)
        valid = mask[None]
        with patch("datasets.crack_dataset.random.random", side_effect=[1.0, 1.0]), patch(
            "datasets.crack_dataset.random.choice", return_value=2
        ) as choose_turn:
            augmented = _augment(image, mask, edge, tangent, valid, allow_quarter_turns=False)
        choose_turn.assert_called_once_with((0, 2))
        self.assertTrue(np.array_equal(augmented[0], np.rot90(image, 2)))
        self.assertTrue(np.array_equal(augmented[1], np.rot90(mask, 2)))

        with patch("datasets.crack_dataset.random.random", side_effect=[1.0, 1.0]), patch(
            "datasets.crack_dataset.random.randint", return_value=1
        ) as choose_turn:
            default_augmented = _augment(image, mask, edge, tangent, valid)
        choose_turn.assert_called_once_with(0, 3)
        self.assertEqual(default_augmented[0].shape[:2], (5, 3))

        config_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544.yaml")
        with config_path.open(encoding="utf-8") as handle:
            native_config = yaml.safe_load(handle)
        self.assertEqual(native_config["data"]["img_size"], [384, 544])
        self.assertFalse(native_config["data"]["allow_quarter_turns"])
        with tempfile.TemporaryDirectory() as directory:
            saved_path, saved = save_resolved_config(Path(directory), native_config)
            self.assertTrue(saved)
            with saved_path.open(encoding="utf-8") as handle:
                resolved = yaml.safe_load(handle)
            self.assertEqual(resolved["data"]["img_size"], [384, 544])
            self.assertFalse(resolved["data"]["allow_quarter_turns"])

    def test_mask_positive_weight_only_changes_primary_mask_bce(self) -> None:
        logits = torch.tensor([[[[-1.0, 0.5], [1.5, -0.5]]]])
        target = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
        default_loss = BCEDiceLoss()
        explicit_default_loss = BCEDiceLoss(pos_weight=1.0)
        weighted_loss = BCEDiceLoss(pos_weight=1.05)
        self.assertTrue(torch.equal(default_loss(logits, target), explicit_default_loss(logits, target)))
        self.assertTrue(torch.equal(default_loss.dice(logits, target), weighted_loss.dice(logits, target)))
        self.assertGreater(weighted_loss.bce(logits, target), default_loss.bce(logits, target))
        with self.assertRaises(ValueError):
            BCEDiceLoss(pos_weight=0.0)

        outputs = {
            "mask": logits,
            "edge": torch.tensor([[[[0.2, -0.3], [0.5, -0.4]]]]),
            "route_s3": torch.tensor([[[[0.1]], [[0.2]], [[-0.1]], [[0.0]]]]),
        }
        batch = {
            "mask": target,
            "edge_gt": torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]]),
            "tangent_gt": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]]]),
            "tangent_valid": torch.ones(1, 1, 2, 2),
        }
        unweighted = DEMambaLoss(mask_pos_weight=1.0, scnp_warmup_epochs=0)
        weighted = DEMambaLoss(mask_pos_weight=1.05, scnp_warmup_epochs=0)
        unweighted_total, unweighted_items = unweighted(outputs, batch, epoch=1)
        weighted_total, weighted_items = weighted(outputs, batch, epoch=1)
        self.assertGreater(weighted_items["mask_loss"], unweighted_items["mask_loss"])
        self.assertAlmostEqual(weighted_items["edge_loss"], unweighted_items["edge_loss"])
        self.assertAlmostEqual(weighted_items["scnp_loss"], unweighted_items["scnp_loss"])
        self.assertAlmostEqual(weighted_items["tangent_loss"], unweighted_items["tangent_loss"])
        self.assertAlmostEqual(
            float(weighted_total - unweighted_total),
            weighted_items["mask_loss"] - unweighted_items["mask_loss"],
            delta=1e-6,
        )
        self.assertTrue(torch.equal(weighted.edge_loss(outputs["edge"], batch["edge_gt"]), EdgeLoss()(outputs["edge"], batch["edge_gt"])))
        self.assertEqual(weighted.scnp_mask_loss.pos_weight, 1.0)

    def test_scheduler_factory_matches_cosine_and_poly_resumes(self) -> None:
        class Logger:
            def __init__(self) -> None:
                self.messages: list[str] = []

            def info(self, message: str, *args) -> None:
                self.messages.append(message % args if args else message)

        def optimizer() -> torch.optim.Optimizer:
            return torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=1e-3)

        cosine_cfg = {"epochs": 4, "lr_scheduler": "cosine", "min_lr": 0.0, "poly_power": 0.9}
        factory_optimizer, reference_optimizer = optimizer(), optimizer()
        factory_scheduler = make_scheduler(factory_optimizer, cosine_cfg, Logger())
        reference_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(reference_optimizer, T_max=4, eta_min=0.0)
        for _ in range(4):
            self.assertEqual(factory_optimizer.param_groups[0]["lr"], reference_optimizer.param_groups[0]["lr"])
            factory_optimizer.step()
            reference_optimizer.step()
            factory_scheduler.step()
            reference_scheduler.step()
        self.assertEqual(factory_optimizer.param_groups[0]["lr"], reference_optimizer.param_groups[0]["lr"])

        poly_cfg = {"epochs": 4, "lr_scheduler": "poly", "min_lr": 1e-5, "poly_power": 0.9}
        poly_optimizer = optimizer()
        poly_scheduler = make_scheduler(poly_optimizer, poly_cfg, Logger())
        lrs = [poly_optimizer.param_groups[0]["lr"]]
        for _ in range(4):
            poly_optimizer.step()
            poly_scheduler.step()
            lrs.append(poly_optimizer.param_groups[0]["lr"])
        self.assertEqual(lrs[0], 1e-3)
        self.assertTrue(all(next_lr <= current_lr for current_lr, next_lr in zip(lrs, lrs[1:])))
        self.assertTrue(all(lr >= 1e-5 for lr in lrs))
        self.assertAlmostEqual(lrs[-1], 1e-5)

        resume_optimizer = optimizer()
        resume_scheduler = make_scheduler(resume_optimizer, poly_cfg, Logger())
        for _ in range(2):
            resume_optimizer.step()
            resume_scheduler.step()
        saved_optimizer_state = resume_optimizer.state_dict()
        saved_scheduler_state = resume_scheduler.state_dict()
        restored_optimizer = optimizer()
        restored_scheduler = make_scheduler(restored_optimizer, poly_cfg, Logger())
        restored_optimizer.load_state_dict(saved_optimizer_state)
        restored_scheduler.load_state_dict(saved_scheduler_state)
        for _ in range(2):
            resume_optimizer.step()
            resume_scheduler.step()
            restored_optimizer.step()
            restored_scheduler.step()
            self.assertEqual(resume_optimizer.param_groups[0]["lr"], restored_optimizer.param_groups[0]["lr"])

        with self.assertRaisesRegex(ValueError, "Unsupported lr_scheduler"):
            make_scheduler(optimizer(), {**cosine_cfg, "lr_scheduler": "invalid"}, Logger())

    def test_scnp_is_finite_and_differentiable(self) -> None:
        logits = torch.randn(2, 1, 9, 11, requires_grad=True)
        target = torch.randint(0, 2, (2, 1, 9, 11), dtype=torch.float32)
        output = SCNP(3)(logits, target)
        self.assertEqual(tuple(output.shape), tuple(logits.shape))
        self.assertTrue(torch.isfinite(output).all())
        output.mean().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    @patch("metrics.metrics._topology_scores", return_value=(1.0, 0.0))
    def test_boundary_f1_and_structural_score_follow_prediction_quality(self, _topology_scores) -> None:
        target = torch.zeros(1, 1, 20, 20)
        target[:, :, 3:8, 3:8] = 1
        matching_logits = torch.where(target.bool(), torch.full_like(target, 10.0), torch.full_like(target, -10.0))
        mismatching_mask = torch.zeros_like(target)
        mismatching_mask[:, :, 12:17, 12:17] = 1
        mismatching_logits = torch.where(mismatching_mask.bool(), torch.full_like(target, 10.0), torch.full_like(target, -10.0))

        matching_metric = SegmentationMetric()
        matching_metric.update(matching_logits, target)
        mismatching_metric = SegmentationMetric()
        mismatching_metric.update(mismatching_logits, target)
        matching_boundary = matching_metric.compute()["BoundaryF1"]
        mismatching_boundary = mismatching_metric.compute()["BoundaryF1"]

        self.assertGreater(matching_boundary, 0.99)
        self.assertLess(mismatching_boundary, 0.01)
        evaluation_cfg = {"selection_metric": "structural_score", "structural_score_weights": {"f1": 0.5, "cldice": 0.3, "boundary_f1": 0.2}}
        baseline = {"F1": 0.8, "clDice": 0.7, "BoundaryF1": matching_boundary}
        degraded = {**baseline, "BoundaryF1": mismatching_boundary}
        self.assertGreater(structural_score(baseline, evaluation_cfg), structural_score(degraded, evaluation_cfg))

    @patch("metrics.metrics._topology_scores", return_value=(1.0, 0.0))
    def test_primary_threshold_changes_main_metrics_but_not_ods_or_ois(self, _topology_scores) -> None:
        target = torch.tensor([[[[1.0, 0.0]]]])
        logits = torch.tensor([[[[0.4, -0.4]]]])
        default_metric = SegmentationMetric()
        explicit_default_metric = SegmentationMetric(primary_threshold=0.5)
        strict_metric = SegmentationMetric(primary_threshold=0.7)
        default_metric.update(logits, target)
        explicit_default_metric.update(logits, target)
        strict_metric.update(logits, target)
        default_results = default_metric.compute()
        self.assertEqual(default_results, explicit_default_metric.compute())
        strict_results = strict_metric.compute()

        self.assertGreater(default_results["F1"], strict_results["F1"])
        self.assertEqual(default_results["ODS"], strict_results["ODS"])
        self.assertEqual(default_results["OIS"], strict_results["OIS"])
        with self.assertRaisesRegex(ValueError, "primary_threshold"):
            SegmentationMetric(primary_threshold=1.1)

    def test_validation_threshold_sweep_is_inclusive_and_prefers_higher_tie(self) -> None:
        thresholds = build_threshold_sweep(0.2, 0.6, 0.2)
        self.assertEqual(thresholds, [0.2, 0.4, 0.6])
        sweep = {threshold: [torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)] for threshold in thresholds}
        logits = torch.zeros(1, 1, 1, 1)
        target = torch.ones(1, 1, 1, 1)
        update_threshold_sweep(sweep, logits, target)
        results = summarize_threshold_sweep(sweep)
        self.assertEqual([threshold for threshold, _ in results], thresholds)
        best_threshold, best_metrics = max(results, key=lambda item: (item[1]["F1"], item[0]))
        self.assertEqual(best_threshold, 0.4)
        self.assertGreater(best_metrics["F1"], 0.99)
        self.assertEqual(build_threshold_sweep(0.2, 0.5, 0.2), [0.2, 0.4, 0.5])
        with self.assertRaisesRegex(ValueError, "STEP"):
            build_threshold_sweep(0.2, 0.5, 0.0)

    def test_average_timing_uses_batch_weighted_epoch_means(self) -> None:
        totals = {"data": 0.0, "forward_loss": 0.0, "backward_optimizer": 0.0}
        batches = 0.0
        batches += accumulate_epoch_timing(totals, {"data": 1.0, "forward_loss": 2.0, "backward_optimizer": 3.0, "batches": 2.0})
        batches += accumulate_epoch_timing(totals, {"data": 3.0, "forward_loss": 4.0, "backward_optimizer": 5.0, "batches": 6.0})
        self.assertEqual(batches, 8.0)
        self.assertAlmostEqual(totals["data"] / batches, 2.5)
        self.assertAlmostEqual(totals["forward_loss"] / batches, 3.5)
        self.assertAlmostEqual(totals["backward_optimizer"] / batches, 4.5)

    def test_resolved_config_snapshot_uses_cli_overrides_and_is_not_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            cfg = {"data": {"data_root": "./DeepCrack", "img_size": [512, 512]}, "train": {"batch_size": 8, "epochs": 10}, "save": {"save_dir": "runs"}}
            args = type("Args", (), {"data_root": "./custom", "epochs": None, "batch_size": 2, "lr": None, "img_size": [384, 384], "num_workers": None, "save_dir": None})()
            resolved = apply_overrides(cfg, args)
            config_path, saved = save_resolved_config(run_dir, resolved)
            self.assertTrue(saved)
            with config_path.open(encoding="utf-8") as handle:
                snapshot = yaml.safe_load(handle)
            self.assertEqual(snapshot["data"]["data_root"], "./custom")
            self.assertEqual(snapshot["data"]["img_size"], [384, 384])
            self.assertEqual(snapshot["train"]["batch_size"], 2)

            resolved["train"]["batch_size"] = 99
            _, saved_again = save_resolved_config(run_dir, resolved)
            self.assertFalse(saved_again)
            with config_path.open(encoding="utf-8") as handle:
                self.assertEqual(yaml.safe_load(handle)["train"]["batch_size"], 2)

    def test_mamba_backend_summary_and_fallback_locations(self) -> None:
        summary, fallback_locations = summarize_mamba_backends(
            {"stage3": ["mamba_ssm"] * 4, "stage4": ["mamba_ssm"] * 4}
        )
        self.assertEqual(summary, "Mamba backend | stage3: mamba_ssm ×4; stage4: mamba_ssm ×4")
        self.assertEqual(fallback_locations, [])

        summary, fallback_locations = summarize_mamba_backends(
            {"stage3": ["mamba_ssm", "fallback", "mamba_ssm", "mamba_ssm"]}
        )
        self.assertEqual(summary, "Mamba backend | stage3: mamba_ssm ×3, fallback ×1")
        self.assertEqual(fallback_locations, ["stage3: vertical"])
        self.assertEqual(format_mamba_fallback_warning(fallback_locations), "Mamba fallback detected | stage3: vertical")
        self.assertEqual(summarize_mamba_backends({}), (None, []))

    def test_cached_dataset_and_run_time_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for kind in ("images", "masks", "targets"):
                (root / "train" / kind).mkdir(parents=True)
            Image.fromarray(np.zeros((9, 13, 3), dtype=np.uint8)).save(root / "train" / "images" / "sample.png")
            Image.fromarray(np.zeros((9, 13), dtype=np.uint8)).save(root / "train" / "masks" / "sample.png")
            np.savez_compressed(
                root / "train" / "targets" / "sample.npz",
                edge_gt=np.zeros((1, 9, 13), dtype=np.float32),
                tangent_gt=np.zeros((4, 9, 13), dtype=np.float32),
                tangent_valid=np.zeros((1, 9, 13), dtype=np.float32),
            )
            item = CrackSegmentationDataset(root, "train", img_size=(16, 16), augment=False)[0]
            self.assertEqual(tuple(item["image"].shape), (3, 16, 16))
            self.assertEqual(tuple(item["tangent_gt"].shape), (4, 16, 16))
            first = create_run_dir(root / "runs", None)
            first.mkdir(parents=True)
            second = create_run_dir(root / "runs", None)
            self.assertNotEqual(first, second)
            self.assertEqual(create_run_dir(root, str(root / "latest.pth")), root.resolve())
            self.assertEqual(format_duration(3661), "01:01:01")


if __name__ == "__main__":
    unittest.main()
