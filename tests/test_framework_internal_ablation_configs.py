"""Read-only audit for the Crack500 F0--F3 Framework internal chain.

Run from repository root:
    python tests/test_framework_internal_ablation_configs.py

No dataloader, Auto Protocol dry-run, optimizer update, validation, test split,
or formal training is used here.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    torch = None
    TORCH_AVAILABLE = False
else:
    TORCH_AVAILABLE = True
    from losses import DEMambaLoss
    from models import build_model
    from models.blocks import (
        DetailPreservingHalfStem,
        HalfResolutionStage1Block,
        NoDPSMTopoBlock,
        RIFTStructuralDirectionalModelingBlock,
    )
    from models.decoder import LowResolutionSubpixelSeparableConcatUpBlock
    from models.mamba_blocks import SequenceMambaCore


M_REGISTRY = ROOT / "configs" / "experiments" / "paper_dependency_aware_ablation" / "crack500"
REGISTRY = ROOT / "configs" / "experiments" / "paper_framework_internal_ablation" / "crack500"
ROWS = {
    "F0": M_REGISTRY / "M0_FrameworkOff_DPSMOff_StructSupOff.yaml",
    "F1": REGISTRY / "F1_LSSCFOnly_DPSMOff_StructSupOff.yaml",
    "F2": REGISTRY / "F2_LSSCF_DPER_DPSMOff_StructSupOff.yaml",
    "F3": M_REGISTRY / "M1_FrameworkOn_DPSMOff_StructSupOff.yaml",
}
EXPECTED_NEW = {
    "F1_LSSCFOnly_DPSMOff_StructSupOff.yaml",
    "F2_LSSCF_DPER_DPSMOff_StructSupOff.yaml",
}
SAVE_ROOT = "runs/paper/framework_internal_ablation_strict_v2/crack500"
LSSCF = ("model", "use_low_resolution_subpixel_separable_concat_fusion")
DPER = ("model", "use_detail_preserving_early_rebase")
RSDM = ("model", "use_rsdm_stage1")
PROBE_DEVICE = torch.device("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    assert isinstance(value, dict), path
    return value


def _rows() -> dict[str, dict]:
    return {identifier: _load(path) for identifier, path in ROWS.items()}


def _flatten(value: object, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], object]:
    if isinstance(value, dict):
        result: dict[tuple[str, ...], object] = {}
        for key, child in value.items():
            result.update(_flatten(child, prefix + (str(key),)))
        return result
    return {prefix: value}


def _without_save(config: dict) -> dict:
    result = copy.deepcopy(config)
    result.pop("save", None)
    return result


def _diff(left: dict, right: dict) -> set[tuple[str, ...]]:
    left_flat, right_flat = _flatten(left), _flatten(right)
    return {key for key in left_flat.keys() | right_flat.keys() if left_flat.get(key) != right_flat.get(key)}


def _only(left: dict, right: dict, expected: set[tuple[str, ...]], label: str) -> None:
    actual = _diff(_without_save(left), _without_save(right))
    assert actual == expected, f"{label}: expected {sorted(expected)}, got {sorted(actual)}"


def _build(config: dict):
    assert TORCH_AVAILABLE
    model_cfg = copy.deepcopy(config["model"])
    name = model_cfg.pop("name")
    # No-Mamba rows can be smoke-tested on CPU without changing their YAML
    # contract, which still declares the formal server backend request.
    model_cfg["use_mamba_ssm"] = bool(torch.cuda.is_available())
    return build_model(name, **model_cfg).to(PROBE_DEVICE).eval()


def _criterion(config: dict):
    assert TORCH_AVAILABLE
    loss_cfg = copy.deepcopy(config["loss"])
    if loss_cfg["mask_pos_weight"] == "auto":
        loss_cfg["mask_pos_weight"] = 1.0
    return DEMambaLoss(**loss_cfg).to(PROBE_DEVICE)


def _batch(height: int = 64, width: int = 96) -> dict[str, torch.Tensor]:
    assert TORCH_AVAILABLE
    tangent = torch.zeros(1, 4, height, width, device=PROBE_DEVICE)
    tangent[:, 0] = 1.0
    return {
        "image": torch.randn(1, 3, height, width, device=PROBE_DEVICE),
        "mask": (torch.rand(1, 1, height, width, device=PROBE_DEVICE) < 0.1).float(),
        "edge_gt": (torch.rand(1, 1, height, width, device=PROBE_DEVICE) < 0.04).float(),
        "tangent_gt": tangent,
        "tangent_valid": torch.ones(1, 1, height, width, device=PROBE_DEVICE),
    }


def test_reuse_mapping_diffs_and_frozen_protocol() -> None:
    assert {path.name for path in REGISTRY.glob("F*.yaml")} == EXPECTED_NEW
    assert ROWS["F0"].name.startswith("M0_") and ROWS["F3"].name.startswith("M1_")
    configs = _rows()
    _only(configs["F0"], configs["F1"], {LSSCF}, "F0->F1")
    _only(configs["F1"], configs["F2"], {DPER}, "F1->F2")
    _only(configs["F2"], configs["F3"], {RSDM}, "F2->F3")
    for identifier, config in configs.items():
        model, loss, auto = config["model"], config["loss"], config["auto_protocol"]
        assert config["data"]["data_root"] == "./Crack500" and config["data"]["img_size"] is None
        assert model["base_channels"] == 40 and model["transformer_stages"] == [4]
        assert model["use_transformer"] is True and model["use_topology_matched_no_dpsm_stage4"] is True
        assert model["mamba_stages"] == [] and model["use_direction_partitioned_shared_mamba"] is False
        assert loss["lambda_tangent"] == 0.0 and loss["lambda_bpcm"] == 0.0
        assert config["distillation"]["enabled"] is False
        assert auto["version"] == 2 and auto["enabled"] is True and auto["dry_run"] is False
        assert auto["regimes"][-1]["effective_batch_size"] == 8 and auto["regimes"][-1]["target_epochs"] == 64
        assert auto["validation"]["interval_updates"] == 400
        assert auto["early_stopping"]["min_delta"] == 0.0005
        assert config["evaluation"]["structural_score_weights"] == {"f1": 0.5, "cldice": 0.3, "boundary_f1": 0.2}
        assert not any("checkpoint" in ".".join(path).lower() or "resume" in ".".join(path).lower() for path in _flatten(config))
    assert configs["F1"]["save"]["save_dir"] == f"{SAVE_ROOT}/F1_LSSCFOnly_DPSMOff_StructSupOff"
    assert configs["F2"]["save"]["save_dir"] == f"{SAVE_ROOT}/F2_LSSCF_DPER_DPSMOff_StructSupOff"


def test_no_dpsm_structure_and_framework_registration() -> None:
    if not TORCH_AVAILABLE:
        print("SKIP: PyTorch unavailable; run model checks on demamba21.")
        return
    configs = _rows()
    models = {identifier: _build(config) for identifier, config in configs.items()}
    counts = {identifier: sum(parameter.numel() for parameter in model.parameters()) for identifier, model in models.items()}
    for identifier, model in models.items():
        state = model.state_dict()
        assert isinstance(model.stage4, NoDPSMTopoBlock)
        assert model.route_stage4 is False and model.mamba_backend_info() == {}
        assert sum(isinstance(module, SequenceMambaCore) for module in model.modules()) == 0
        assert not any(token in key.lower() for key in state for token in ("route_mamba", "route_head", "shared_mamba"))
    assert not isinstance(models["F0"].decoder.up1, LowResolutionSubpixelSeparableConcatUpBlock)
    for identifier in ("F1", "F2", "F3"):
        assert isinstance(models[identifier].decoder.up1, LowResolutionSubpixelSeparableConcatUpBlock), identifier
    for identifier in ("F0", "F1"):
        assert not isinstance(models[identifier].stem, DetailPreservingHalfStem), identifier
        assert not isinstance(models[identifier].stage1, HalfResolutionStage1Block), identifier
    for identifier in ("F2", "F3"):
        assert isinstance(models[identifier].stem, DetailPreservingHalfStem), identifier
        assert isinstance(models[identifier].stage1, HalfResolutionStage1Block), identifier
    for identifier in ("F0", "F1", "F2"):
        assert models[identifier].rsdm_stage1 is None
    assert isinstance(models["F3"].rsdm_stage1, RIFTStructuralDirectionalModelingBlock)
    print(f"{PROBE_DEVICE.type} Framework internal parameter counts: {counts}")


def _assert_equal_prefixes(left, right, prefixes: tuple[str, ...], label: str) -> None:
    left_state, right_state = left.state_dict(), right.state_dict()
    for prefix in prefixes:
        keys = [key for key in left_state.keys() & right_state.keys() if key.startswith(prefix)]
        assert keys, f"{label}: no common keys under {prefix}"
        for key in keys:
            assert torch.equal(left_state[key], right_state[key]), f"{label}: shared tensor changed: {key}"


def test_adjacent_shared_initialization_and_forward_loss() -> None:
    if not TORCH_AVAILABLE:
        print("SKIP: PyTorch unavailable; run initialization/forward checks on demamba21.")
        return
    configs = _rows()
    pairs = (
        ("F0", "F1", ("stage2.", "stage3.", "stage4.local.", "stage4.global_branch.", "stage4.refine.")),
        ("F1", "F2", ("stage2.", "stage3.", "stage4.local.", "stage4.global_branch.", "stage4.refine.", "decoder.")),
        ("F2", "F3", ("stem.", "stage1.", "down1.", "stage2.", "stage3.", "stage4.local.", "stage4.global_branch.", "stage4.refine.", "decoder.")),
    )
    for left_id, right_id, prefixes in pairs:
        torch.manual_seed(2909)
        left = _build(configs[left_id])
        torch.manual_seed(2909)
        right = _build(configs[right_id])
        _assert_equal_prefixes(left, right, prefixes, f"{left_id}<->{right_id}")
    batch = _batch()
    for identifier, config in configs.items():
        output = _build(config)(batch["image"])
        loss, items = _criterion(config)(output, batch, epoch=100)
        assert torch.isfinite(loss) and torch.isfinite(output["mask"]).all(), identifier
        assert output["mask"].shape == (1, 1, 64, 96)
        assert output.get("route_s4") is None
        assert items["tangent_loss"] == 0.0 and items["bpcm_loss"] == 0.0


def test() -> None:
    for check in (
        test_reuse_mapping_diffs_and_frozen_protocol,
        test_no_dpsm_structure_and_framework_registration,
        test_adjacent_shared_initialization_and_forward_loss,
    ):
        check()
        print(f"PASS: {check.__name__}")


if __name__ == "__main__":
    test()
