from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml
from PIL import Image

from datasets import CrackSegmentationDataset
from datasets.crack_dataset import _native_scale_jitter
from models import DEMambaNet


class NativeScaleJitterTests(unittest.TestCase):
    @staticmethod
    def _arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        height, width = 20, 24
        image = (np.arange(height * width * 3, dtype=np.uint16).reshape(height, width, 3) % 251 + 1).astype(np.uint8)
        mask = np.zeros((height, width), dtype=np.float32)
        mask[8:13, 10:15] = 1.0
        edge = mask.copy()
        tangent = np.zeros((4, height, width), dtype=np.float32)
        tangent[2] = mask
        valid = mask[None].copy()
        return image, mask, edge, tangent, valid

    @staticmethod
    def _assert_exactly_equal(
        expected: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        actual: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        for expected_value, actual_value in zip(expected, actual):
            np.testing.assert_array_equal(actual_value, expected_value)
            assert actual_value.dtype == expected_value.dtype

    def test_centered_scaling_keeps_structural_targets_aligned(self) -> None:
        inputs = self._arrays()
        for cfg in (
            None,
            {"enabled": False, "probability": 1.0, "scale_range": [0.9, 1.1]},
            {"enabled": True, "probability": 0.0, "scale_range": [0.9, 1.1]},
        ):
            self._assert_exactly_equal(inputs, _native_scale_jitter(*inputs, cfg))
        with patch("datasets.crack_dataset.random.random", return_value=0.0), patch(
            "datasets.crack_dataset.random.uniform", return_value=1.0
        ):
            self._assert_exactly_equal(
                inputs,
                _native_scale_jitter(*inputs, {"enabled": True, "probability": 1.0, "scale_range": [1.0, 1.0]}),
            )

        with patch("datasets.crack_dataset.random.random", return_value=0.0), patch(
            "datasets.crack_dataset.random.uniform", return_value=1.10
        ):
            image, mask, edge, tangent, valid = _native_scale_jitter(
                *inputs,
                {"enabled": True, "probability": 1.0, "scale_range": [0.90, 1.10]},
            )
        self.assertEqual(image.shape, inputs[0].shape)
        self.assertEqual(mask.shape, inputs[1].shape)
        self.assertEqual(tangent.shape, inputs[3].shape)
        self.assertEqual(image.dtype, np.uint8)
        self.assertTrue(all(value.dtype == np.float32 for value in (mask, edge, tangent, valid)))
        self.assertTrue(all(value.flags.c_contiguous for value in (image, mask, edge, tangent, valid)))
        self.assertTrue(np.array_equal(mask, edge))
        self.assertTrue(np.array_equal(mask, valid[0]))
        self.assertEqual(int(np.count_nonzero(tangent[[0, 1, 3]])), 0)
        normalized_tangent = tangent * valid
        normalized_tangent /= normalized_tangent.sum(axis=0, keepdims=True).clip(min=1e-6)
        tangent_sum = normalized_tangent.sum(axis=0)
        self.assertTrue(np.allclose(tangent_sum[valid[0] > 0.5], 1.0))
        self.assertTrue(np.allclose(tangent_sum[valid[0] <= 0.5], 0.0))
        self.assertEqual(float(mask[10, 12]), 1.0)

    def test_scale_down_uses_reflection_for_images_and_zero_padding_for_targets(self) -> None:
        inputs = self._arrays()
        with patch("datasets.crack_dataset.random.random", return_value=0.0), patch(
            "datasets.crack_dataset.random.uniform", return_value=0.90
        ):
            image, mask, edge, tangent, valid = _native_scale_jitter(
                *inputs,
                {"enabled": True, "probability": 1.0, "scale_range": [0.90, 1.10]},
            )
        self.assertTrue(np.all(mask[0] == 0.0))
        self.assertTrue(np.all(mask[-1] == 0.0))
        self.assertTrue(np.all(mask[:, 0] == 0.0))
        self.assertTrue(np.all(mask[:, -1] == 0.0))
        self.assertTrue(np.all(edge == mask))
        self.assertTrue(np.all(valid[:, 0] == 0.0))
        self.assertTrue(np.all(tangent[:, 0] == 0.0))
        self.assertTrue(np.any(image[0] != 0))
        self.assertTrue(np.any(image[:, 0] != 0))
        self.assertEqual(float(mask[10, 12]), 1.0)

    def test_train_only_dataset_path_and_config_leave_model_unchanged(self) -> None:
        jitter_cfg = {"enabled": True, "probability": 1.0, "scale_range": [0.90, 0.90]}
        image, mask, edge, tangent, valid = self._arrays()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "val", "test"):
                for directory_name in ("images", "masks", "targets"):
                    (root / split / directory_name).mkdir(parents=True)
                Image.fromarray(image).save(root / split / "images" / "sample.png")
                Image.fromarray((mask * 255).astype(np.uint8)).save(root / split / "masks" / "sample.png")
                np.savez_compressed(
                    root / split / "targets" / "sample.npz",
                    edge_gt=edge[None],
                    tangent_gt=tangent,
                    tangent_valid=valid,
                )

            reference_train = CrackSegmentationDataset(
                root, "train", img_size=(20, 24), augment=True, allow_quarter_turns=False
            )
            configured_train = CrackSegmentationDataset(
                root, "train", img_size=(20, 24), augment=True, allow_quarter_turns=False, scale_jitter=jitter_cfg
            )
            with patch("datasets.crack_dataset.random.random", side_effect=[1.0, 1.0]), patch(
                "datasets.crack_dataset.random.choice", return_value=0
            ):
                reference_item = reference_train[0]
            with patch("datasets.crack_dataset.random.random", side_effect=[1.0, 1.0, 0.0]), patch(
                "datasets.crack_dataset.random.choice", return_value=0
            ), patch("datasets.crack_dataset.random.uniform", return_value=0.90):
                configured_item = configured_train[0]
            self.assertFalse(torch.equal(configured_item["image"], reference_item["image"]))

            disabled_train = CrackSegmentationDataset(
                root, "train", img_size=(20, 24), augment=False, scale_jitter=jitter_cfg
            )[0]
            reference_disabled_train = CrackSegmentationDataset(root, "train", img_size=(20, 24), augment=False)[0]
            for key in ("image", "mask", "edge_gt", "tangent_gt", "tangent_valid"):
                self.assertTrue(torch.equal(disabled_train[key], reference_disabled_train[key]), f"disabled:{key}")

            for split in ("val", "test"):
                reference = CrackSegmentationDataset(root, split, img_size=(20, 24), augment=True)[0]
                configured = CrackSegmentationDataset(root, split, img_size=(20, 24), augment=True, scale_jitter=jitter_cfg)[0]
                for key in ("image", "mask", "edge_gt", "tangent_gt", "tangent_valid"):
                    self.assertTrue(torch.equal(configured[key], reference[key]), f"{split}:{key}")

        base_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml")
        experiment_path = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM_NSJ.yaml")
        with base_path.open(encoding="utf-8") as handle:
            base_cfg = yaml.safe_load(handle)
        with experiment_path.open(encoding="utf-8") as handle:
            experiment_cfg = yaml.safe_load(handle)
        self.assertEqual({key: value for key, value in experiment_cfg["data"].items() if key != "scale_jitter"}, base_cfg["data"])
        self.assertEqual(experiment_cfg["data"]["scale_jitter"], {"enabled": True, "probability": 0.35, "scale_range": [0.90, 1.10]})
        self.assertEqual(experiment_cfg["train"], base_cfg["train"])
        self.assertEqual(experiment_cfg["evaluation"], base_cfg["evaluation"])
        for name, value in base_cfg["model"].items():
            self.assertEqual(experiment_cfg["model"][name], value)
        for name, value in base_cfg["loss"].items():
            self.assertEqual(experiment_cfg["loss"][name], value)
        self.assertEqual(experiment_cfg["model"]["stage3_mamba_window_size"], None)
        self.assertFalse(experiment_cfg["model"]["stage3_bidirectional_mamba"])
        self.assertFalse(experiment_cfg["model"]["return_training_mask_feature"])
        self.assertEqual(experiment_cfg["loss"]["lambda_ep_hcfc"], 0.0)
        self.assertEqual(experiment_cfg["loss"]["lambda_fbhn"], 0.0)
        self.assertEqual(experiment_cfg["save"]["save_dir"], "./runs/pilot/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM_NSJ")
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
        base_model = DEMambaNet(**base_model_cfg).eval()
        torch.manual_seed(42)
        nsj_model = DEMambaNet(**experiment_model_cfg).eval()
        self.assertEqual(sum(parameter.numel() for parameter in base_model.parameters()), 7026611)
        self.assertEqual(sum(parameter.numel() for parameter in nsj_model.parameters()), 7026611)
        for index, (base_parameter, nsj_parameter) in enumerate(zip(base_model.parameters(), nsj_model.parameters(), strict=True)):
            self.assertTrue(torch.equal(base_parameter, nsj_parameter), index)
        self.assertTrue(all(value == "mamba_ssm" for values in nsj_model.mamba_backend_info().values() for value in values))

        # NSJ is dataset-only. Use the lightweight CPU fallback path to prove
        # that its configuration does not alter any model construction or output.
        cpu_model_cfg = {
            "in_channels": 3,
            "num_classes": 1,
            "base_channels": 8,
            "use_mamba_ssm": False,
            "mamba_stages": (),
            "transformer_stages": (),
        }
        torch.manual_seed(42)
        cpu_base_model = DEMambaNet(**cpu_model_cfg).eval()
        torch.manual_seed(42)
        cpu_nsj_model = DEMambaNet(**cpu_model_cfg).eval()
        probe = torch.randn(1, 3, 64, 80)
        with torch.no_grad():
            base_outputs = cpu_base_model(probe)
            nsj_outputs = cpu_nsj_model(probe)
        self.assertEqual(base_outputs.keys(), nsj_outputs.keys())
        for name in base_outputs:
            self.assertTrue(torch.equal(base_outputs[name], nsj_outputs[name]), name)


if __name__ == "__main__":
    unittest.main()
