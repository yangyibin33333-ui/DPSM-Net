"""Read-only R0/R1 supervised-route ablation audit.

Run from the repository root:
    python tests/test_paper_supervised_route_ablation_configs.py

No dataset, optimizer, formal Auto Protocol dry-run, validation/test split, or
formal training is used. CPU fallback validates control flow only; the checksum
anchor is asserted solely with CUDA + real mamba_ssm on demamba21.
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
    from models.mamba_blocks import DirectionPartitionedSharedMamba
    from train import register_first_dpsm_execution_hook, validate_dpsm_scan_route_config
    from utils import ModelEMA, model_checksum


R0_PATH = (
    ROOT
    / "configs"
    / "experiments"
    / "paper_supervised_route_ablation"
    / "crack500"
    / "R0_FourDirection_Uniform_TangentOn.yaml"
)
D1_PATH = (
    ROOT
    / "configs"
    / "experiments"
    / "paper_dependency_aware_ablation"
    / "crack500"
    / "loss_2x2"
    / "D1_TangentOn_BPCMOff.yaml"
)
EXPECTED_SAVE_DIR = "runs/paper/supervised_route_ablation_strict_v2/crack500/R0_FourDirection_Uniform_TangentOn"
EXPECTED_CHECKSUM = "3f7cd9a6ef3e73c2ddb51ceac09d64fbd9353ac336b97fad3cc1990e11246a4a"
DEVICE = torch.device("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    assert isinstance(config, dict), path
    return config


def _normalize(config: dict) -> dict:
    result = copy.deepcopy(config)
    result["model"].setdefault("dpsm_scan_mode", "four_direction")
    result["model"].setdefault("dpsm_route_mode", "learned")
    return result


def _flatten(value: object, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], object]:
    if isinstance(value, dict):
        result: dict[tuple[str, ...], object] = {}
        for key, child in value.items():
            result.update(_flatten(child, prefix + (str(key),)))
        return result
    return {prefix: value}


def _diff(left: dict, right: dict) -> set[tuple[str, ...]]:
    left_flat, right_flat = _flatten(left), _flatten(right)
    return {key for key in left_flat.keys() | right_flat.keys() if left_flat.get(key) != right_flat.get(key)}


def _without_save(config: dict) -> dict:
    result = _normalize(config)
    result.pop("save", None)
    return result


def _build(config: dict) -> torch.nn.Module:
    assert TORCH_AVAILABLE
    model_cfg = copy.deepcopy(config["model"])
    model_name = model_cfg.pop("name")
    # CPU fallback is deliberately labeled logic-only by this test.
    model_cfg["use_mamba_ssm"] = bool(torch.cuda.is_available())
    return build_model(model_name, **model_cfg).to(DEVICE)


def _criterion(config: dict) -> DEMambaLoss:
    assert TORCH_AVAILABLE
    loss_cfg = copy.deepcopy(config["loss"])
    if loss_cfg.get("mask_pos_weight") == "auto":
        # Crack500 xlarge V2 resolves this pre-registered value before loss construction.
        loss_cfg["mask_pos_weight"] = 1.0
    return DEMambaLoss(**loss_cfg).to(DEVICE)


def _batch(height: int = 64, width: int = 96) -> dict[str, torch.Tensor]:
    assert TORCH_AVAILABLE
    directions = torch.randint(0, 4, (1, 1, height, width), device=DEVICE)
    tangent = torch.zeros(1, 4, height, width, device=DEVICE)
    tangent.scatter_(1, directions, 1.0)
    return {
        "image": torch.randn(1, 3, height, width, device=DEVICE),
        "mask": (torch.rand(1, 1, height, width, device=DEVICE) < 0.1).float(),
        "edge_gt": (torch.rand(1, 1, height, width, device=DEVICE) < 0.04).float(),
        "tangent_gt": tangent,
        "tangent_valid": torch.ones(1, 1, height, width, device=DEVICE),
    }


def _assert_same_state(left: torch.nn.Module, right: torch.nn.Module, label: str) -> None:
    left_state, right_state = left.state_dict(), right.state_dict()
    assert left_state.keys() == right_state.keys(), label
    for key, value in left_state.items():
        assert torch.equal(value, right_state[key]), f"{label}: {key}"


def _route_has_nonzero_gradient(model: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in model.stage4.route_mamba.route_head.parameters()
    )


def test_r0_d1_normalized_config_diff_and_protocol() -> None:
    r0, d1 = _load(R0_PATH), _load(D1_PATH)
    # No R1 YAML is allowed: it is the historic D1 run, not a duplicate run.
    assert list(R0_PATH.parent.glob("R1*.yaml")) == []
    assert _diff(_without_save(r0), _without_save(d1)) == {("model", "dpsm_route_mode")}
    assert r0["save"]["save_dir"] == EXPECTED_SAVE_DIR
    model, loss, auto = r0["model"], r0["loss"], r0["auto_protocol"]
    assert model["dpsm_scan_mode"] == "four_direction" and model["dpsm_route_mode"] == "uniform"
    assert model["use_direction_partitioned_shared_mamba"] is True
    assert model["stage4_direction_group_channels"] == 80
    assert model["use_detail_preserving_early_rebase"] is True
    assert model["use_rsdm_stage1"] is True
    assert model["use_low_resolution_subpixel_separable_concat_fusion"] is True
    assert loss["lambda_tangent"] == 0.2 and loss["lambda_bpcm"] == 0.0
    assert r0["data"] == d1["data"] and r0["train"] == d1["train"]
    assert r0["evaluation"] == d1["evaluation"] and r0["distillation"] == d1["distillation"]
    assert r0["auto_protocol"] == d1["auto_protocol"]
    assert auto["regimes"][-1]["effective_batch_size"] == 8 and auto["regimes"][-1]["target_epochs"] == 64
    assert (2357 // 8) * 64 == 18_816


def test_registration_default_regression_and_checksum() -> None:
    if not TORCH_AVAILABLE:
        print("SKIP: PyTorch unavailable; run model registration checks on demamba21.")
        return
    r0, d1 = _load(R0_PATH), _load(D1_PATH)
    torch.manual_seed(42)
    uniform = _build(r0).eval()
    torch.manual_seed(42)
    learned = _build(d1).eval()
    _assert_same_state(uniform, learned, "R0/D1 registered initialization")
    assert sum(parameter.numel() for parameter in uniform.parameters()) == 3_193_751
    assert sum(parameter.numel() for parameter in learned.parameters()) == 3_193_751
    dpsm = uniform.stage4.route_mamba
    assert isinstance(dpsm, DirectionPartitionedSharedMamba)
    assert dpsm.scan_mode == "four_direction" and dpsm.route_mode == "uniform"
    assert dpsm.group_channels == 80 and dpsm.dim == 320 and dpsm.instantiated_core_count == 1

    # Default D1 and explicit frozen four_direction+learned must remain exact.
    explicit_d1 = _normalize(d1)
    torch.manual_seed(42)
    implicit_default = _build(d1).eval()
    torch.manual_seed(42)
    explicit_default = _build(explicit_d1).eval()
    _assert_same_state(implicit_default, explicit_default, "D1 implicit/default route modes")
    image = torch.randn(1, 3, 64, 96, device=DEVICE)
    with torch.inference_mode():
        expected, actual = implicit_default(image), explicit_default(image)
    assert expected.keys() == actual.keys()
    for key, value in expected.items():
        torch.testing.assert_close(value, actual[key], rtol=0.0, atol=0.0)

    if DEVICE.type == "cuda":
        assert dpsm.shared_mamba.backend == "mamba_ssm", dpsm.shared_mamba.backend
        torch.manual_seed(42)
        assert model_checksum(_build(d1)) == EXPECTED_CHECKSUM
        torch.manual_seed(42)
        assert model_checksum(_build(r0)) == EXPECTED_CHECKSUM
        checksum_summary = EXPECTED_CHECKSUM
    else:
        checksum_summary = "CPU fallback (not a CUDA/mamba_ssm checksum)"
        print("SKIP: CUDA+mamba_ssm checksum regression requires demamba21; CPU fallback is logic-only.")
    print(
        f"{DEVICE.type} R0 model audit | scan_mode: {dpsm.scan_mode} | route_mode: {dpsm.route_mode} | "
        "tangent: 0.2000 | BPCM: 0.0000 | groups: 4 × 80 | width: 320 | shared cores: 1 | "
        f"Model parameters: {sum(parameter.numel() for parameter in uniform.parameters())} | initial checksum: {checksum_summary}"
    )


def test_uniform_mask_independence_tangent_gradient_and_mask_only() -> None:
    if not TORCH_AVAILABLE:
        print("SKIP: PyTorch unavailable; run R0 forward/backward checks on demamba21.")
        return
    r0 = _load(R0_PATH)
    batch = _batch()
    uniform = _build(r0).eval()
    with torch.inference_mode():
        before = uniform(batch["image"])
        for parameter in uniform.stage4.route_mamba.route_head.parameters():
            parameter.add_(17.0)
        after = uniform(batch["image"])
        mask_only = uniform.forward_mask_only(batch["image"])
    assert before["route_s4"].shape == (1, 4, 8, 12)
    assert not torch.equal(before["route_s4"], after["route_s4"])
    torch.testing.assert_close(before["mask"], after["mask"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(after["mask"], mask_only["mask"], rtol=1e-5, atol=1e-6)
    assert torch.isfinite(after["mask"]).all() and torch.isfinite(after["route_s4"]).all()

    restored = torch.ones(4, 1, 80, 2, 3, device=DEVICE)
    uniform_weights = torch.full((1, 4, 2, 3), 0.25, device=DEVICE)
    torch.testing.assert_close(
        DirectionPartitionedSharedMamba.apply_route_weights(restored, uniform_weights), restored, rtol=0.0, atol=0.0
    )

    torch.manual_seed(71)
    supervised = _build(r0).train()
    supervised.zero_grad(set_to_none=True)
    outputs = supervised(batch["image"])
    total, items = _criterion(r0)(outputs, batch, epoch=100)
    assert torch.isfinite(total) and items["tangent_loss"] > 0.0 and items["bpcm_loss"] == 0.0
    total.backward()
    assert _route_has_nonzero_gradient(supervised), "R0 route head did not receive tangent gradient"

    # With tangent disabled, the identical uniform model has no mask/edge path
    # to route_head. This isolates the preceding R0 gradient to supervision.
    mask_only_loss_cfg = copy.deepcopy(r0)
    mask_only_loss_cfg["loss"]["lambda_tangent"] = 0.0
    torch.manual_seed(71)
    unsupervised = _build(mask_only_loss_cfg).train()
    unsupervised.zero_grad(set_to_none=True)
    unsupervised_outputs = unsupervised(batch["image"])
    unsupervised_total, _ = _criterion(mask_only_loss_cfg)(unsupervised_outputs, batch, epoch=100)
    assert torch.isfinite(unsupervised_total)
    unsupervised_total.backward()
    assert not _route_has_nonzero_gradient(unsupervised), "uniform route head received a mask/edge gradient"


def test_d1_learned_mask_gradient_and_guard_matrix() -> None:
    if not TORCH_AVAILABLE:
        print("SKIP: PyTorch unavailable; run learned-route/guard checks on demamba21.")
        return
    r0, d1 = _load(R0_PATH), _load(D1_PATH)
    batch = _batch()
    torch.manual_seed(81)
    learned = _build(d1).train()
    learned.zero_grad(set_to_none=True)
    learned_output = learned(batch["image"])
    (learned_output["mask"].square().mean() + learned_output["edge"].square().mean()).backward()
    assert _route_has_nonzero_gradient(learned), "D1 learned route lost its mask-path gradient"

    # The R0 registration is the sole newly allowed case.
    validate_dpsm_scan_route_config(r0)
    for route_mode in ("uniform", "learned"):
        invalid = copy.deepcopy(r0)
        invalid["model"].update(dpsm_scan_mode="axial_repeated", dpsm_route_mode=route_mode)
        try:
            validate_dpsm_scan_route_config(invalid)
        except ValueError as error:
            assert "axial_repeated" in str(error) and "four_direction" in str(error)
        else:
            raise AssertionError(f"axial+tangent was accepted for {route_mode}")
    invalid = copy.deepcopy(r0)
    invalid["loss"]["lambda_bpcm"] = 0.025
    try:
        validate_dpsm_scan_route_config(invalid)
    except ValueError as error:
        assert "lambda_bpcm=0" in str(error)
    else:
        raise AssertionError("uniform+BPCM was accepted")
    for overrides, label in (
        ({"use_direction_partitioned_shared_mamba": False}, "DPSM-off"),
        ({"use_direction_partitioned_shared_mamba": False, "use_topology_matched_no_dpsm_stage4": True}, "NoDPSM"),
        ({"use_direction_partitioned_shared_mamba": False, "use_common_private_direction_partitioned_shared_mamba": True}, "CP-DPSM"),
        ({"stage4_routed_branch": "single_raster_mamba80"}, "alternate branch"),
    ):
        invalid = copy.deepcopy(r0)
        invalid["model"].update(overrides)
        try:
            validate_dpsm_scan_route_config(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{label} accepted non-default route mode")


def test_ema_does_not_inherit_first_execution_hook() -> None:
    if not TORCH_AVAILABLE:
        print("SKIP: PyTorch unavailable; run hook lifecycle check on demamba21.")
        return

    class Logger:
        def info(self, *_args, **_kwargs) -> None:
            pass

    torch.manual_seed(113)
    raw_model = _build(_load(R0_PATH)).eval()
    raw_dpsm = raw_model.stage4.route_mamba
    assert not raw_dpsm._forward_hooks
    ema = ModelEMA(raw_model)
    ema_dpsm = ema.module.stage4.route_mamba
    assert not ema_dpsm._forward_hooks
    handle = register_first_dpsm_execution_hook(raw_model, Logger())
    try:
        assert handle is not None and len(raw_dpsm._forward_hooks) == 1
        assert not ema_dpsm._forward_hooks, "EMA inherited the raw-model first-forward hook"
    finally:
        if handle is not None:
            handle.remove()
    assert not raw_dpsm._forward_hooks and not ema_dpsm._forward_hooks


def test() -> None:
    checks = (
        test_r0_d1_normalized_config_diff_and_protocol,
        test_registration_default_regression_and_checksum,
        test_uniform_mask_independence_tangent_gradient_and_mask_only,
        test_d1_learned_mask_gradient_and_guard_matrix,
        test_ema_does_not_inherit_first_execution_hook,
    )
    for check in checks:
        check()
        print(f"PASS: {check.__name__}")


if __name__ == "__main__":
    test()
