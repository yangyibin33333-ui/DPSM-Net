from __future__ import annotations

import copy
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

import torch
import yaml

from models import DEMambaNet
from models.masked_reconstruction import (
    ENCODER_PREFIXES,
    MaskedC40Reconstruction,
    is_encoder_state_key,
    masked_smooth_l1_rgb_loss,
    sample_patch_mask,
)
from pretrain_masked_reconstruction import validate_pretraining_config
from train import load_encoder_initialization, parse_args


BASE_CONFIG = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml")
SSL_CONFIG = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM_TRAINONLY_SSL.yaml")
PRETRAIN_CONFIG = Path("configs/pretraining/C40_MASKED_RGB_TRAINONLY_384x544.yaml")


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _c40_kwargs() -> dict:
    kwargs = dict(_load_yaml(BASE_CONFIG)["model"])
    kwargs.pop("name", None)
    return kwargs


class MaskedReconstructionPretrainingTests(unittest.TestCase):
    def test_patch_mask_is_repeatable_patch_aligned_and_nearest_sixty_percent(self) -> None:
        first = sample_patch_mask(2, 384, 544, generator=torch.Generator().manual_seed(17))
        second = sample_patch_mask(2, 384, 544, generator=torch.Generator().manual_seed(17))
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(tuple(first.shape), (2, 1, 384, 544))
        patch_area = 16 * 16
        expected_masked_patches = round((384 // 16) * (544 // 16) * 0.60)
        self.assertTrue(torch.all(first.sum(dim=(1, 2, 3)) == expected_masked_patches * patch_area))
        tiles = first.reshape(2, 1, 24, 16, 34, 16)
        self.assertTrue(torch.all(tiles == tiles[:, :, :, :1, :, :1]))

    def test_masked_smooth_l1_ignores_unmasked_rgb_pixels(self) -> None:
        target = torch.zeros(1, 3, 32, 32)
        patch_mask = torch.zeros(1, 1, 32, 32, dtype=torch.bool)
        patch_mask[:, :, :16, :16] = True
        reconstruction = torch.zeros_like(target)
        reconstruction[:, :, :16, :16] = 1.0
        masked_loss = masked_smooth_l1_rgb_loss(reconstruction, target, patch_mask)
        reconstruction[:, :, 16:, 16:] = 1_000.0
        self.assertTrue(torch.equal(masked_loss, masked_smooth_l1_rgb_loss(reconstruction, target, patch_mask)))
        self.assertTrue(torch.isfinite(masked_loss))
        self.assertGreater(float(masked_loss), 0.0)

    def test_encoder_feature_interface_matches_standard_forward_and_ssl_export(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("The real mamba_ssm backend requires CUDA for its forward path.")
        device = torch.device("cuda")
        torch.manual_seed(42)
        model = DEMambaNet(**_c40_kwargs()).to(device).eval()
        image = torch.randn(1, 3, 64, 80, device=device)
        with torch.no_grad():
            outputs = model(image)
            features = model.extract_encoder_features(image)
            reconstructed_outputs = model.decoder(
                [features["f1"], features["f2"], features["f3"], features["f4"]],  # type: ignore[list-item]
                output_size=image.shape[-2:],
            )
            for name in ("route_s1", "route_s2", "route_s3", "route_s4"):
                if features[name] is not None:
                    reconstructed_outputs[name] = features[name]  # type: ignore[assignment]
        self.assertEqual(set(outputs), set(reconstructed_outputs))
        for name in outputs:
            self.assertTrue(torch.equal(outputs[name], reconstructed_outputs[name]), name)
        self.assertTrue(all(backend == "mamba_ssm" for values in model.mamba_backend_info().values() for backend in values))

        torch.manual_seed(42)
        pretraining_model = MaskedC40Reconstruction(_c40_kwargs()).to(device).eval()
        exported = pretraining_model.encoder_state_dict()
        self.assertTrue(exported)
        self.assertTrue(all(is_encoder_state_key(name) for name in exported))
        self.assertTrue(all(any(name == prefix or name.startswith(f"{prefix}.") for name in exported) for prefix in ENCODER_PREFIXES))
        self.assertFalse(any("decoder" in name or "reconstruction" in name for name in exported))
        patch_mask = sample_patch_mask(1, 64, 80, generator=torch.Generator().manual_seed(5)).to(device)
        with torch.no_grad():
            reconstruction_outputs = pretraining_model(torch.randn(1, 3, 64, 80, device=device), patch_mask)
        self.assertEqual(tuple(reconstruction_outputs["reconstruction"].shape), (1, 3, 64, 80))
        self.assertTrue(torch.isfinite(reconstruction_outputs["reconstruction"]).all())

    def test_pretraining_checkpoint_initializes_only_encoder_and_preserves_decoder_seed(self) -> None:
        torch.manual_seed(42)
        pretraining_model = MaskedC40Reconstruction(_c40_kwargs())
        metadata = {
            "seed": 42,
            "epochs": 100,
            "split": "train",
            "image_count": 368,
            "img_size": [384, 544],
            "mask_ratio": 0.60,
            "patch_size": 16,
            "objective": "masked_smooth_l1_rgb_reconstruction",
            "mamba_backend": pretraining_model.encoder.mamba_backend_info(),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "encoder_pretrain_final.pth"
            torch.save({"encoder": pretraining_model.encoder_state_dict(), "metadata": metadata}, path)
            self.assertEqual(set(torch.load(path)), {"encoder", "metadata"})

            torch.manual_seed(42)
            native_model = DEMambaNet(**_c40_kwargs())
            torch.manual_seed(42)
            initialized_model = DEMambaNet(**_c40_kwargs())
            loaded = load_encoder_initialization(path, initialized_model)
            self.assertEqual(loaded, len(pretraining_model.encoder_state_dict()))
            for name, tensor in native_model.state_dict().items():
                if name.startswith("decoder."):
                    self.assertTrue(torch.equal(tensor, initialized_model.state_dict()[name]), name)
            for name, tensor in pretraining_model.encoder_state_dict().items():
                self.assertTrue(torch.equal(tensor, initialized_model.state_dict()[name]), name)

            malformed = {"encoder": copy.deepcopy(pretraining_model.encoder_state_dict()), "metadata": metadata}
            malformed["encoder"]["decoder.mask_head.weight"] = torch.zeros(1)
            bad_path = Path(directory) / "bad_encoder.pth"
            torch.save(malformed, bad_path)
            with self.assertRaisesRegex(ValueError, "unknown encoder key"):
                load_encoder_initialization(bad_path, initialized_model)

            missing = {"encoder": copy.deepcopy(pretraining_model.encoder_state_dict()), "metadata": metadata}
            missing["encoder"].pop(next(iter(missing["encoder"])))
            torch.save(missing, bad_path)
            with self.assertRaisesRegex(ValueError, "missing encoder key"):
                load_encoder_initialization(bad_path, initialized_model)

            shape_mismatch = {"encoder": copy.deepcopy(pretraining_model.encoder_state_dict()), "metadata": metadata}
            first_key = next(iter(shape_mismatch["encoder"]))
            shape_mismatch["encoder"][first_key] = torch.zeros(1)
            torch.save(shape_mismatch, bad_path)
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                load_encoder_initialization(bad_path, initialized_model)

            metadata_mismatch = {"encoder": pretraining_model.encoder_state_dict(), "metadata": dict(metadata, epochs=99)}
            torch.save(metadata_mismatch, bad_path)
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                load_encoder_initialization(bad_path, initialized_model)

    def test_ssl_configuration_is_only_a_pretraining_annotation_and_cli_is_exclusive(self) -> None:
        baseline = _load_yaml(BASE_CONFIG)
        ssl = _load_yaml(SSL_CONFIG)
        expected = copy.deepcopy(baseline)
        expected["save"]["save_dir"] = "./runs/pilot/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM_TRAINONLY_SSL"
        expected["pretraining"] = {
            "protocol": "train_only_masked_rgb_reconstruction",
            "checkpoint": "./runs/pretrain/C40_MASKED_RGB_TRAINONLY_384x544/encoder_pretrain_final.pth",
            "source_split": "train",
            "source_image_count": 368,
            "external_data": False,
        }
        self.assertEqual(ssl, expected)
        pretraining = _load_yaml(PRETRAIN_CONFIG)
        validate_pretraining_config(pretraining)
        self.assertEqual(pretraining["data"]["split"], "train")
        self.assertEqual(pretraining["data"]["img_size"], [384, 544])
        self.assertEqual(pretraining["masking"], {"patch_size": 16, "mask_ratio": 0.60})
        self.assertEqual(pretraining["train"]["epochs"], 100)
        self.assertIn("from models import build_model", Path("test.py").read_text(encoding="utf-8"))
        with mock.patch.object(sys, "argv", ["train.py", "--resume", "old.pth", "--init_encoder_checkpoint", "ssl.pth"]):
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args()


if __name__ == "__main__":
    unittest.main()
