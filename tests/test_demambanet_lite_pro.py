from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import DEMambaNetLitePro, build_model
from models.de_mambanet_lite import SharedAxialMambaContext
from models.mamba_blocks import SequenceMambaCore


BALANCED_CONFIG = REPO_ROOT / (
    "configs/experiments/lightweight_demambanet_lite_balanced/"
    "DE_MAMBANET_LITE_BALANCED_LR150_DEEPCRACK_NATIVE384x544_EMA_BPCM.yaml"
)
PRO_CONFIG = REPO_ROOT / (
    "configs/experiments/lightweight_demambanet_lite_pro/"
    "DE_MAMBANET_LITE_PRO_LR150_DEEPCRACK_NATIVE384x544_EMA_BPCM.yaml"
)


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class DEMambaNetLiteProTests(unittest.TestCase):
    @staticmethod
    def _model() -> DEMambaNetLitePro:
        return DEMambaNetLitePro(use_mamba_ssm=False, drop_path_rate=0.0)

    def test_output_contract_and_route_scales(self) -> None:
        model = self._model().eval()
        with torch.no_grad():
            outputs = model(torch.randn(1, 3, 64, 80))
        self.assertEqual(
            set(outputs),
            {"mask", "edge", "reliability", "route_s1", "route_s3", "route_s4"},
        )
        for name in ("mask", "edge", "reliability"):
            self.assertEqual(tuple(outputs[name].shape), (1, 1, 64, 80))
            self.assertTrue(torch.isfinite(outputs[name]).all())
        self.assertEqual(tuple(outputs["route_s1"].shape), (1, 4, 16, 20))
        self.assertEqual(tuple(outputs["route_s3"].shape), (1, 4, 4, 5))
        self.assertEqual(tuple(outputs["route_s4"].shape), (1, 4, 2, 3))

    def test_exactly_two_shared_axial_mamba_cores(self) -> None:
        model = self._model()
        contexts = [model.stage3_context, model.stage4_context]
        self.assertTrue(all(isinstance(context, SharedAxialMambaContext) for context in contexts))
        all_cores = [module for module in model.modules() if isinstance(module, SequenceMambaCore)]
        self.assertEqual(len(all_cores), 2)
        for context in contexts:
            cores = [module for module in context.modules() if isinstance(module, SequenceMambaCore)]
            self.assertEqual(len(cores), 1)
            self.assertTrue(context.shared_core)
            self.assertEqual(context.scan_count, 2)
            self.assertEqual(context.active_mamba_directions, ("horizontal", "vertical"))

    def test_encoder_decoder_and_head_widths(self) -> None:
        model = self._model()
        self.assertEqual(model.encoder_widths, (40, 80, 160, 240))
        self.assertEqual(model.encoder_depths, (2, 2, 4, 2))
        self.assertEqual(len(model.stage1), 2)
        self.assertEqual(len(model.stage2), 2)
        self.assertEqual(len(model.stage3), 4)
        self.assertEqual(len(model.stage4), 2)
        self.assertEqual(model.stage3[0].input_norm.num_channels, 160)
        self.assertEqual(model.stage4[0].input_norm.num_channels, 240)
        self.assertEqual(model.stage3_context.input_projection.block[0].in_channels, 160)
        self.assertEqual(model.stage4_context.input_projection.block[0].in_channels, 240)

        self.assertEqual(model.decoder.proj1.block[0].out_channels, 80)
        self.assertEqual(model.decoder.route_head.in_channels, 80)
        self.assertEqual(model.decoder.half_upsample[0].out_channels, 40 * 4)
        self.assertEqual(model.detail_channels, 40)
        self.assertEqual(model.decoder.edge_head.in_channels, 40)
        self.assertEqual(model.decoder.reliability_head.in_channels, 40)
        self.assertEqual(model.decoder.mask_projection.block[0].out_channels, 64)
        self.assertEqual(model.decoder.mask_head.in_channels, 64)

    def test_pro_excludes_plus_and_unrequested_architectures(self) -> None:
        model = self._model()
        forbidden_fragments = (
            "PartialDirectionalContinuityBlock",
            "PhaseAwareMaskResidual",
            "Transformer",
        )
        module_names = [type(module).__name__ for module in model.modules()]
        for fragment in forbidden_fragments:
            self.assertFalse(any(fragment in name for name in module_names), fragment)

    def test_all_residual_baseline_gates_are_zero_initialized(self) -> None:
        model = self._model()
        fusions = (
            model.decoder.fuse3,
            model.decoder.fuse2,
            model.decoder.fuse1,
            model.decoder.half_fusion,
        )
        for fusion in fusions:
            self.assertEqual(torch.count_nonzero(fusion.gate[-1].weight).item(), 0)
            self.assertEqual(torch.count_nonzero(fusion.gate[-1].bias).item(), 0)

    def test_registry_and_backend_contract(self) -> None:
        model = build_model("DEMambaNetLitePro", use_mamba_ssm=False)
        self.assertIsInstance(model, DEMambaNetLitePro)
        self.assertEqual(
            model.mamba_backend_info(),
            {"stage3_context": ["fallback"], "stage4_context": ["fallback"]},
        )

    def test_config_preserves_balanced_protocol(self) -> None:
        balanced = _load(BALANCED_CONFIG)
        pro = _load(PRO_CONFIG)
        for section in ("data", "train", "loss", "evaluation"):
            self.assertEqual(pro[section], balanced[section], section)
        self.assertEqual(pro["model"]["name"], "DEMambaNetLitePro")
        self.assertNotEqual(pro["model"], balanced["model"])
        self.assertNotEqual(pro["save"], balanced["save"])
        self.assertEqual(
            pro["save"]["save_dir"],
            "./runs/pilot/lightweight_demambanet_lite_pro/"
            "DE_MAMBANET_LITE_PRO_LR150_DEEPCRACK_NATIVE384x544_EMA_BPCM",
        )


if __name__ == "__main__":
    unittest.main()
