"""Synthetic regression checks for the validation-only P0 branch/route diagnostic.

Run from the repository root: ``python tests/test_stage4_branch_route_diagnostic.py``.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import build_model

SPEC = importlib.util.spec_from_file_location("branch_route_diagnostic", ROOT / "scripts" / "diagnose_stage4_branch_route.py")
assert SPEC is not None and SPEC.loader is not None
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


def _p0_model() -> nn.Module:
    cfg = yaml.safe_load((ROOT / "configs" / "experiments" / "stage4_order_ablation" / "tut" / "P0_Parallel.yaml").read_text(encoding="utf-8"))
    model_cfg = dict(cfg["model"])
    model_cfg["use_mamba_ssm"] = False  # CPU synthetic backend only; it never changes P0 configuration files.
    name = model_cfg.pop("name")
    torch.manual_seed(17)
    return build_model(name, **model_cfg).eval()


def test_split_entropy_fusion_and_targets() -> None:
    assert diagnostic.validate_val_split("val") == "val"
    for split in ("train", "test"):
        try:
            diagnostic.validate_val_split(split)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{split} was accepted")
    logits = torch.zeros(1, 4, 2, 2)
    tangent = torch.zeros(1, 4, 8, 8); tangent[:, 2] = 1
    valid = torch.ones(1, 1, 8, 8)
    target, pooled_valid = diagnostic.aligned_tangent_targets(logits, tangent, valid)
    assert torch.equal(pooled_valid, torch.ones_like(pooled_valid)) and torch.allclose(target[:, 2], torch.ones_like(target[:, 2]))
    stats, confusion = diagnostic.route_statistics(logits, tangent, valid, torch.ones(1, 1, 8, 8), torch.ones(1, 1, 8, 8))
    assert abs(stats["route_all_entropy_normalized"] - 1.0) < 1e-6
    assert int(confusion.sum()) == 4 and int(confusion[2, 0]) == 4
    assert stats["route_tangent_valid_top1_accuracy"] == 0.0
    assert abs(stats["route_tangent_valid_target_mass"] - 0.25) < 1e-6
    assert abs(stats["route_tangent_valid_soft_ce"] - torch.log(torch.tensor(4.0)).item()) < 1e-6


def test_wrapper_capture_and_read_only_state() -> None:
    model = _p0_model()
    state = diagnostic._state_snapshot(model)
    modes = [(module, module.training) for module in model.modules()]
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        original, capture = diagnostic._capture_normal(model, image)
    block = model.stage4
    normal = diagnostic.Stage4CounterfactualWrapper(block, "normal")
    with torch.no_grad():
        stage_output, route = normal(capture["x"])
    assert torch.equal(stage_output, capture["final"]) and torch.equal(route, capture["route_logits"])
    for mode in ("remove_local_delta", "remove_dpsm_delta", "remove_transformer_delta"):
        wrapper = diagnostic.Stage4CounterfactualWrapper(block, mode)
        with torch.no_grad():
            _, route_cf = wrapper(capture["x"])
        assert torch.equal(route_cf, capture["route_logits"]), mode
        assert block.route_mamba.last_shared_mamba_call_count == 1, mode
    diagnostic._assert_state_unchanged(model, state)
    assert [(module, module.training) for module in model.modules()] == modes
    parts, summary = diagnostic.fusion_contributions(block, capture["local"], capture["routed"], capture["global"], capture["pre_bn"])
    assert set(parts) == {"local", "dpsm", "transformer"} and summary["fusion_decomposition_max_abs"] < 1e-5
    branch = diagnostic.branch_statistics(capture["x"], capture["local"], capture["routed"], capture["global"])
    assert all(torch.isfinite(torch.tensor(value)) for value in branch.values())


def test_validation_loader_only_and_atomic_outputs() -> None:
    cfg = {"data": {"data_root": "synthetic", "img_size": [64, 64], "threshold": 127}, "train": {"persistent_workers": False, "prefetch_factor": 2}}
    calls: list[str] = []
    original = diagnostic.build_dataloader
    try:
        diagnostic.build_dataloader = lambda _root, split, *_args, **_kwargs: calls.append(split) or []
        assert diagnostic.build_val_loader(cfg, 1, 0) == []
    finally:
        diagnostic.build_dataloader = original
    assert calls == ["val"]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        diagnostic.atomic_csv(root / "per_image.csv", [{"stem": "a", "value": 1.0}])
        diagnostic.atomic_json(root / "summary.json", {"test_loader_constructed": False})
        diagnostic.atomic_text(root / "summary.md", "read-only\n")
        assert (root / "per_image.csv").is_file() and (root / "summary.json").is_file() and (root / "summary.md").is_file()


def check() -> None:
    for function in (test_split_entropy_fusion_and_targets, test_wrapper_capture_and_read_only_state, test_validation_loader_only_and_atomic_outputs):
        function(); print(f"PASS: {function.__name__}")


if __name__ == "__main__":
    check()
