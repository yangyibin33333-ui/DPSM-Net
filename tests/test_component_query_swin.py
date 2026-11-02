from __future__ import annotations

import copy
import unittest
from pathlib import Path

import torch
import yaml

from datasets import build_component_id_map
from losses import ComponentQuerySetLoss, DEMambaLoss
from models import ComponentQuerySwin, DEMambaNet, HRTopoMambaNet, build_model


BASELINE_CONFIG = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml")
EXPERIMENT_CONFIG = Path("configs/experiments/COMPONENT_QUERY_SWIN_NATIVE384x544_EMA_BPCM.yaml")
HRTOPO_CONFIG = Path("configs/experiments/C3_C40_LR150_PHOTO_NATIVE384x544_EMA_BPCM_HRTOPO.yaml")


class ComponentQuerySwinTests(unittest.TestCase):
    @staticmethod
    def _load_config(path: Path) -> dict:
        with path.open(encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    @classmethod
    def _model_from_config(cls) -> ComponentQuerySwin:
        config = cls._load_config(EXPERIMENT_CONFIG)
        model_cfg = dict(config["model"])
        model = build_model(str(model_cfg.pop("name")), **model_cfg)
        if not isinstance(model, ComponentQuerySwin):
            raise AssertionError("Component-query config did not select ComponentQuerySwin.")
        return model

    @staticmethod
    def _component_map(height: int = 64, width: int = 80) -> torch.Tensor:
        mask = torch.zeros(1, 1, height, width)
        mask[..., 8:24, 10:24] = 1.0
        mask[..., 38:54, 50:68] = 1.0
        return torch.from_numpy(build_component_id_map(mask[0, 0].numpy())).unsqueeze(0).unsqueeze(0).long()

    def test_factory_small_forward_and_mamba_free_contract(self) -> None:
        model = self._model_from_config().eval()
        with torch.no_grad():
            outputs = model(torch.randn(1, 3, 64, 80))
        self.assertEqual(
            set(outputs),
            {"mask", "edge", "reliability", "route_s1", "component_mask_logits", "component_class_logits"},
        )
        self.assertEqual(outputs["mask"].shape, (1, 1, 64, 80))
        self.assertEqual(outputs["edge"].shape, (1, 1, 64, 80))
        self.assertEqual(outputs["reliability"].shape, (1, 1, 64, 80))
        self.assertEqual(outputs["route_s1"].shape, (1, 4, 32, 40))
        self.assertEqual(outputs["component_mask_logits"].shape, (1, 64, 32, 40))
        self.assertEqual(outputs["component_class_logits"].shape, (1, 64, 2))
        self.assertTrue(all(torch.isfinite(value).all() for value in outputs.values()))
        self.assertTrue(model.is_mamba_free)
        self.assertEqual(model.mamba_backend_info(), {})

    def test_unknown_model_name_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown model.name 'not-a-model'"):
            build_model("not-a-model")

    def test_native_resolution_shape_contract(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("Native-resolution ComponentQuerySwin contract is checked on CUDA when available.")
        model = self._model_from_config().to("cuda").eval()
        with torch.no_grad():
            outputs = model(torch.randn(1, 3, 384, 544, device="cuda"))
        self.assertEqual(outputs["mask"].shape, (1, 1, 384, 544))
        self.assertEqual(outputs["edge"].shape, (1, 1, 384, 544))
        self.assertEqual(outputs["reliability"].shape, (1, 1, 384, 544))
        self.assertEqual(outputs["route_s1"].shape, (1, 4, 192, 272))
        self.assertEqual(outputs["component_mask_logits"].shape, (1, 64, 192, 272))
        self.assertEqual(outputs["component_class_logits"].shape, (1, 64, 2))
        self.assertTrue(all(torch.isfinite(value).all() for value in outputs.values()))

    def test_fused_adamw_layout_regression(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("Fused AdamW layout regression is checked on CUDA when available.")
        model = self._model_from_config().to("cuda").train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-4, fused=True)
        outputs = model(torch.randn(1, 3, 64, 80, device="cuda"))
        loss = sum(value.float().square().mean() for value in outputs.values())
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertEqual(parameter.stride(), parameter.grad.stride())
        optimizer.step()

    def test_component_union_is_differentiable_and_not_post_processing(self) -> None:
        component_masks = torch.full((1, 64, 4, 5), -12.0, requires_grad=True)
        component_classes = torch.full((1, 64, 2), -12.0, requires_grad=True)
        with torch.no_grad():
            component_masks[0, 0, 1:3, 2:4] = 12.0
            component_classes[..., 0] = 12.0
            component_classes[0, 0, 0] = -12.0
            component_classes[0, 0, 1] = 12.0
        one_query = ComponentQuerySwin._component_union(component_masks, component_classes)
        self.assertGreater(float(one_query[0, 0, 1, 2]), float(one_query[0, 0, 0, 0]))
        with torch.no_grad():
            component_masks[0, 1, 0, 0] = 12.0
            component_classes[0, 1, 0] = -12.0
            component_classes[0, 1, 1] = 12.0
        two_queries = ComponentQuerySwin._component_union(component_masks, component_classes)
        self.assertGreater(float(two_queries[0, 0, 0, 0]), float(one_query[0, 0, 0, 0]))
        two_queries.sum().backward()
        self.assertTrue(torch.isfinite(component_masks.grad).all())
        self.assertTrue(torch.isfinite(component_classes.grad).all())

    def test_component_loss_handles_empty_maps_and_unique_hungarian_matching(self) -> None:
        loss_fn = ComponentQuerySetLoss()
        diagonal = torch.zeros(4, 4).numpy()
        diagonal[1, 1] = 1.0
        diagonal[2, 2] = 1.0
        self.assertEqual(int(build_component_id_map(diagonal).max()), 1)
        empty_masks = torch.zeros(1, 64, 8, 10, requires_grad=True)
        empty_classes = torch.zeros(1, 64, 2, requires_grad=True)
        empty_ids = torch.zeros(1, 1, 16, 20, dtype=torch.long)
        empty_loss = loss_fn(empty_masks, empty_classes, empty_ids)
        self.assertTrue(torch.isfinite(empty_loss))
        empty_loss.backward()
        self.assertTrue(torch.isfinite(empty_classes.grad).all())
        self.assertNotEqual(float(empty_classes.grad.abs().sum()), 0.0)

        component_ids = self._component_map()
        target = torch.nn.functional.interpolate(component_ids.float(), size=(32, 40), mode="nearest").long()[0, 0]
        perfect_masks = torch.full((1, 64, 32, 40), -8.0)
        perfect_classes = torch.full((1, 64, 2), -8.0)
        perfect_classes[..., 0] = 8.0
        for index, component_id in enumerate(torch.unique(target)[1:]):
            perfect_masks[0, index][target == component_id] = 8.0
            perfect_classes[0, index, 0] = -8.0
            perfect_classes[0, index, 1] = 8.0
        query_indices, target_indices, _ = loss_fn.match_single(perfect_masks[0], perfect_classes[0], component_ids[0])
        self.assertEqual(query_indices.numel(), 2)
        self.assertEqual(query_indices.numel(), torch.unique(query_indices).numel())
        self.assertEqual(target_indices.numel(), torch.unique(target_indices).numel())
        perfect_loss = loss_fn(perfect_masks, perfect_classes, component_ids)
        wrong_loss = loss_fn(torch.zeros_like(perfect_masks), perfect_classes, component_ids)
        self.assertLess(float(perfect_loss), float(wrong_loss))

    def test_component_limit_and_full_existing_loss_backward_contract(self) -> None:
        loss_fn = ComponentQuerySetLoss()
        excessive_ids = torch.zeros(1, 1, 9, 8, dtype=torch.long)
        excessive_ids.flatten()[1:66] = torch.arange(1, 66)
        with self.assertRaisesRegex(ValueError, "at most 64 components"):
            loss_fn(torch.zeros(1, 64, 9, 8), torch.zeros(1, 64, 2), excessive_ids)

        torch.manual_seed(13)
        config = self._load_config(EXPERIMENT_CONFIG)
        model = self._model_from_config().train()
        image = torch.randn(1, 3, 64, 80)
        mask = torch.zeros(1, 1, 64, 80)
        mask[..., 8:24, 10:24] = 1.0
        mask[..., 38:54, 50:68] = 1.0
        batch = {
            "mask": mask,
            "edge_gt": torch.zeros_like(mask),
            "tangent_gt": torch.nn.functional.one_hot(torch.zeros(1, 64, 80, dtype=torch.long), 4)
            .permute(0, 3, 1, 2)
            .float(),
            "tangent_valid": torch.ones_like(mask),
            "component_id_map": torch.from_numpy(build_component_id_map(mask[0, 0].numpy())).unsqueeze(0).unsqueeze(0).long(),
        }
        total_loss, items = DEMambaLoss(**config["loss"])(model(image), batch, epoch=100)
        self.assertTrue(torch.isfinite(total_loss))
        self.assertIn("component_query_loss", items)
        self.assertTrue(torch.isfinite(torch.tensor(items["component_query_loss"])))
        total_loss.backward()
        self.assertIsNotNone(model.patch_embed.weight.grad)
        self.assertTrue(torch.isfinite(model.patch_embed.weight.grad).all())

    def test_zero_component_weight_is_legacy_loss_compatible(self) -> None:
        torch.manual_seed(19)
        outputs = {
            "mask": torch.randn(1, 1, 32, 40, requires_grad=True),
            "edge": torch.randn(1, 1, 32, 40, requires_grad=True),
            "route_s1": torch.randn(1, 4, 16, 20, requires_grad=True),
        }
        batch = {
            "mask": torch.randint(0, 2, (1, 1, 32, 40), dtype=torch.float32),
            "edge_gt": torch.randint(0, 2, (1, 1, 32, 40), dtype=torch.float32),
            "tangent_gt": torch.nn.functional.one_hot(torch.zeros(1, 32, 40, dtype=torch.long), 4)
            .permute(0, 3, 1, 2)
            .float(),
            "tangent_valid": torch.ones(1, 1, 32, 40),
        }
        legacy_loss, legacy_items = DEMambaLoss(lambda_bpcm=0.025)(outputs, batch, epoch=100)
        component_disabled_loss, component_disabled_items = DEMambaLoss(
            lambda_bpcm=0.025, lambda_component_query=0.0
        )(outputs, batch, epoch=100)
        self.assertTrue(torch.equal(legacy_loss, component_disabled_loss))
        self.assertEqual(legacy_items, component_disabled_items)

    def test_parameter_budget_config_and_shared_factory_path(self) -> None:
        model = self._model_from_config()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        self.assertEqual(parameter_count, 7_196_120)
        self.assertGreaterEqual(parameter_count, 6_675_280)
        self.assertLessEqual(parameter_count, 7_377_942)
        source = Path("models/component_query_swin.py").read_text(encoding="utf-8")
        self.assertNotIn("from .blocks", source)
        self.assertNotIn("from .decoder", source)

        baseline = self._load_config(BASELINE_CONFIG)
        experiment = self._load_config(EXPERIMENT_CONFIG)
        expected = copy.deepcopy(baseline)
        expected["model"] = experiment["model"]
        expected["loss"]["lambda_component_query"] = 1.0
        expected["save"]["save_dir"] = "./runs/pilot/COMPONENT_QUERY_SWIN_NATIVE384x544_EMA_BPCM"
        self.assertEqual(experiment, expected)
        self.assertEqual(experiment["data"], baseline["data"])
        self.assertEqual(experiment["train"], baseline["train"])
        self.assertEqual(experiment["evaluation"], baseline["evaluation"])

        test_source = Path("test.py").read_text(encoding="utf-8")
        self.assertIn("from models import build_model", test_source)
        self.assertIn('model_name = str(model_cfg.pop("name", "DEMambaNet"))', test_source)
        self.assertIn("model = build_model(model_name, **model_cfg).to(device)", test_source)
        for config_path, expected_type in (
            (BASELINE_CONFIG, DEMambaNet),
            (HRTOPO_CONFIG, HRTopoMambaNet),
            (EXPERIMENT_CONFIG, ComponentQuerySwin),
        ):
            config = self._load_config(config_path)
            model_cfg = dict(config["model"])
            self.assertIsInstance(build_model(str(model_cfg.pop("name", "DEMambaNet")), **model_cfg), expected_type)


if __name__ == "__main__":
    unittest.main()
