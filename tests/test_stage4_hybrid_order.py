"""Read-only Stage4 DPSM80--Transformer position diagnostics.

Run from repository root:
    python tests/test_stage4_hybrid_order.py
"""

from __future__ import annotations

import copy
import sys
import tempfile
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
    import train as train_module
    from models import build_model
    from models.blocks import TopoRouteBlock
    from models.mamba_blocks import DirectionPartitionedSharedMamba, SequenceMambaCore


CONFIG_DIR = ROOT / "configs" / "experiments" / "stage4_order_ablation" / "tut"
CONFIGS = {
    "P0": CONFIG_DIR / "P0_Parallel.yaml",
    "P1": CONFIG_DIR / "P1_DPSMThenTransformer.yaml",
    "P2": CONFIG_DIR / "P2_TransformerThenDPSM.yaml",
}
ORDERS = {"P0": "parallel", "P1": "dpsm_then_transformer", "P2": "transformer_then_dpsm"}
DEVICE = torch.device("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    assert isinstance(result, dict)
    return result


def _flatten(value: object, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], object]:
    if isinstance(value, dict):
        result: dict[tuple[str, ...], object] = {}
        for key, child in value.items():
            result.update(_flatten(child, prefix + (str(key),)))
        return result
    return {prefix: value}


def _build(config: dict) -> torch.nn.Module:
    assert TORCH_AVAILABLE
    model_cfg = copy.deepcopy(config["model"])
    name = model_cfg.pop("name")
    # CPU coverage intentionally uses the repository fallback; formal YAML
    # retains use_mamba_ssm=true and CUDA audits require real mamba_ssm.
    model_cfg["use_mamba_ssm"] = bool(torch.cuda.is_available())
    return build_model(name, **model_cfg).to(DEVICE)


def _state_equal(left: torch.nn.Module, right: torch.nn.Module) -> None:
    left_state, right_state = left.state_dict(), right.state_dict()
    assert left_state.keys() == right_state.keys()
    for key, value in left_state.items():
        assert value.shape == right_state[key].shape, key
        assert torch.equal(value, right_state[key]), key


def _order_model(order: str) -> torch.nn.Module:
    config = _load(CONFIGS["P0"])
    config["model"]["stage4_hybrid_order"] = order
    return _build(config)


def _capture_stage4(block: TopoRouteBlock, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    captured: dict[str, torch.Tensor] = {}
    def record_input(name: str):
        def hook(_module, args):
            captured.setdefault(name, args[0].detach().clone())
            return None
        return hook

    def record_output(name: str, *, tuple_first: bool = False):
        def hook(_module, _args, output):
            value = output[0] if tuple_first else output
            captured.setdefault(name, value.detach().clone())
            return None
        return hook

    handles = [
        block.route_mamba.register_forward_pre_hook(record_input("route_input")),
        block.route_mamba.register_forward_hook(record_output("route_output", tuple_first=True)),
        block.global_branch.register_forward_pre_hook(record_input("global_input")),
        block.global_branch.register_forward_hook(record_output("global_output")),
        block.local.register_forward_hook(record_output("local_output")),
        block.fuse.register_forward_pre_hook(record_input("fusion_input")),
    ]
    try:
        output, route = block(x)
    finally:
        for handle in handles:
            handle.remove()
    return output, route, captured


def test_default_parallel_and_old_formula_equivalence() -> None:
    if not TORCH_AVAILABLE:
        print("SKIP: PyTorch unavailable; run model checks on demamba21.")
        return
    config = _load(CONFIGS["P0"])
    implicit = copy.deepcopy(config)
    implicit["model"].pop("stage4_hybrid_order")
    torch.manual_seed(101)
    old_default = _build(implicit).eval()
    torch.manual_seed(101)
    explicit = _build(config).eval()
    _state_equal(old_default, explicit)
    image = torch.randn(1, 3, 64, 96, device=DEVICE)
    with torch.inference_mode():
        old_output, explicit_output = old_default(image), explicit(image)
    assert old_output.keys() == explicit_output.keys()
    for key, value in old_output.items():
        torch.testing.assert_close(value, explicit_output[key], rtol=0.0, atol=0.0)

    block = explicit.stage4
    assert isinstance(block, TopoRouteBlock) and block.hybrid_order == "parallel"
    x = torch.randn(1, 320, 8, 12, device=DEVICE)
    with torch.inference_mode():
        local = block.local(x)
        routed, expected_route = block.route_mamba(x)
        global_feature = block.global_branch(x)
        expected = block.refine(x + block.fuse(torch.cat([local, routed, global_feature], dim=1)))
        actual, actual_route = block(x)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_route, expected_route, rtol=0.0, atol=0.0)


def test_three_orders_initialization_dependencies_and_outputs() -> None:
    if not TORCH_AVAILABLE:
        print("SKIP: PyTorch unavailable; run model checks on demamba21.")
        return
    models = {}
    for identifier, order in ORDERS.items():
        torch.manual_seed(202)
        models[identifier] = _order_model(order).eval()
    _state_equal(models["P0"], models["P1"])
    _state_equal(models["P0"], models["P2"])
    counts = {key: sum(parameter.numel() for parameter in model.parameters()) for key, model in models.items()}
    assert len(set(counts.values())) == 1
    for model in models.values():
        assert isinstance(model.stage4, TopoRouteBlock)
        assert isinstance(model.stage4.route_mamba, DirectionPartitionedSharedMamba)
        assert sum(isinstance(module, SequenceMambaCore) for module in model.stage4.modules()) == 1
        assert model.stage4.route_mamba.instantiated_core_count == 1

    x = torch.randn(1, 320, 8, 12, device=DEVICE)
    captures = {}
    for identifier, model in models.items():
        output, route, captures[identifier] = _capture_stage4(model.stage4, x)
        assert output.shape == x.shape and route.shape == (1, 4, 8, 12)
        captured = captures[identifier]
        if identifier == "P0":
            torch.testing.assert_close(captured["route_input"], x, rtol=0.0, atol=0.0)
            torch.testing.assert_close(captured["global_input"], x, rtol=0.0, atol=0.0)
        elif identifier == "P1":
            torch.testing.assert_close(captured["route_input"], x, rtol=0.0, atol=0.0)
            torch.testing.assert_close(captured["global_input"], captured["route_output"], rtol=0.0, atol=0.0)
        else:
            torch.testing.assert_close(captured["global_input"], x, rtol=0.0, atol=0.0)
            torch.testing.assert_close(captured["route_input"], captured["global_output"], rtol=0.0, atol=0.0)
        fusion = captured["fusion_input"]
        torch.testing.assert_close(fusion[:, :320], captured["local_output"], rtol=0.0, atol=0.0)
        torch.testing.assert_close(fusion[:, 320:640], captured["route_output"], rtol=0.0, atol=0.0)
        torch.testing.assert_close(fusion[:, 640:], captured["global_output"], rtol=0.0, atol=0.0)

    image = torch.randn(1, 3, 64, 96, device=DEVICE)
    with torch.inference_mode():
        outputs = {key: model(image) for key, model in models.items()}
    for output in outputs.values():
        assert output["mask"].shape == output["edge"].shape == (1, 1, 64, 96)
        assert output["route_s4"].shape == (1, 4, 8, 12)
        assert all(torch.isfinite(value).all() for value in output.values())
    assert not (torch.equal(outputs["P0"]["mask"], outputs["P1"]["mask"]) and torch.equal(outputs["P0"]["mask"], outputs["P2"]["mask"]))
    print(f"{DEVICE.type} hybrid-order parameter counts: {counts}")


def test_three_orders_backward_and_illegal_guards() -> None:
    if not TORCH_AVAILABLE:
        print("SKIP: PyTorch unavailable; run model checks on demamba21.")
        return
    for order in ORDERS.values():
        torch.manual_seed(303)
        model = _order_model(order).train()
        output = model(torch.randn(1, 3, 64, 96, device=DEVICE))
        (output["mask"].mean() + output["edge"].mean() + output["route_s4"].mean()).backward()
        required = (
            "stage4.route_mamba.shared_mamba",
            "stage4.global_branch.attn.qkv",
            "stage4.fuse",
            "stage4.refine",
        )
        for prefix in required:
            grads = [parameter.grad for name, parameter in model.named_parameters() if name.startswith(prefix)]
            assert grads and all(grad is not None and torch.isfinite(grad).all() for grad in grads), (order, prefix)

    config = _load(CONFIGS["P0"])
    cases = (
        ({"stage4_hybrid_order": "unknown"}, "unknown"),
        ({"stage4_hybrid_order": "dpsm_then_transformer", "use_topology_matched_no_dpsm_stage4": True}, "NoDPSMTopoBlock"),
        ({"stage4_hybrid_order": "dpsm_then_transformer", "use_direction_partitioned_shared_mamba": False}, "DPSM80"),
        ({"stage4_hybrid_order": "dpsm_then_transformer", "transformer_stages": []}, "transformer_stages"),
        ({"stage4_hybrid_order": "dpsm_then_transformer", "use_common_private_direction_partitioned_shared_mamba": True}, "CP-DPSM64"),
        ({"stage4_hybrid_order": "dpsm_then_transformer", "use_multiscale_directional_reallocation": True}, "MSDR"),
    )
    for overrides, expected in cases:
        invalid = copy.deepcopy(config)
        invalid["model"].update(overrides)
        try:
            _build(invalid)
        except ValueError as error:
            assert expected.lower() in str(error).lower(), (overrides, error)
        else:
            raise AssertionError(f"invalid hybrid-order config accepted: {overrides}")


def test_tut_yaml_diff_and_skip_final_test() -> None:
    configs = {identifier: _load(path) for identifier, path in CONFIGS.items()}
    expected_orders = list(ORDERS.values())
    for identifier, config in configs.items():
        assert config["model"]["stage4_hybrid_order"] == ORDERS[identifier]
        assert config["save"]["save_dir"] != "auto"
        assert config["data"] == {**config["data"], "data_root": "./TUT", "img_size": None}
    for other in ("P1", "P2"):
        left, right = _flatten(configs["P0"]), _flatten(configs[other])
        changed = {key for key in left.keys() | right.keys() if left.get(key) != right.get(key)}
        assert changed == {("model", "stage4_hybrid_order"), ("save", "save_dir")}, (other, changed)
    assert len(expected_orders) == 3

    if not TORCH_AVAILABLE:
        print("SKIP: PyTorch unavailable; run skip-final-test checks on demamba21.")
        return
    calls: list[str] = []
    class Logger:
        def info(self, message, *args):
            calls.append(message % args if args else message)
    with tempfile.TemporaryDirectory() as temporary:
        config_path = Path(temporary) / "resolved.yaml"
        config_path.write_text("auto_protocol_resolved: {}\n", encoding="utf-8")
        cfg = {"auto_protocol_resolved": {}}
        original_loader, original_validate = train_module.build_loader, train_module.validate
        try:
            train_module.build_loader = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("test loader built"))
            train_module.validate = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("test validation called"))
            assert train_module.run_final_test(skip_final_test=True, checkpoint_model=torch.nn.Identity(), cfg=cfg, device=DEVICE, auto_enabled=True, config_path=config_path, logger=Logger()) is None
            assert cfg["auto_protocol_resolved"]["test_calls"] == 0 and "Final test skipped by request" in calls
            loader = object()
            train_module.build_loader = lambda *_args, **_kwargs: loader
            train_module.validate = lambda *_args, **_kwargs: {"F1": 0.5}
            assert train_module.run_final_test(skip_final_test=False, checkpoint_model=torch.nn.Identity(), cfg=cfg, device=DEVICE, auto_enabled=False, config_path=None, logger=Logger()) == {"F1": 0.5}
        finally:
            train_module.build_loader, train_module.validate = original_loader, original_validate


def test() -> None:
    checks = (
        test_default_parallel_and_old_formula_equivalence,
        test_three_orders_initialization_dependencies_and_outputs,
        test_three_orders_backward_and_illegal_guards,
        test_tut_yaml_diff_and_skip_final_test,
    )
    for check in checks:
        check()
        print(f"PASS: {check.__name__}")


if __name__ == "__main__":
    test()
