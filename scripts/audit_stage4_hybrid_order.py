"""Read-only structural audit for the TUT Stage4 hybrid-order pilot.

Example:
    python scripts/audit_stage4_hybrid_order.py --device cuda --seed 42 --img_size 640 640
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import build_model
from models.blocks import TopoRouteBlock
from models.mamba_blocks import SequenceMambaCore


CONFIG_DIR = ROOT / "configs" / "experiments" / "stage4_order_ablation" / "tut"
CONFIGS = {
    "P0": CONFIG_DIR / "P0_Parallel.yaml",
    "P1": CONFIG_DIR / "P1_DPSMThenTransformer.yaml",
    "P2": CONFIG_DIR / "P2_TransformerThenDPSM.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Audit Stage4 DPSM80--Transformer order")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img_size", type=int, nargs=2, default=[640, 640])
    parser.add_argument("--output_dir", default="runs/audits/stage4_hybrid_order")
    parser.add_argument(
        "--resolved-config",
        nargs=3,
        metavar=("P0_CONFIG", "P1_CONFIG", "P2_CONFIG"),
        help="Optional resolved config.yaml files from the three server dry-runs, in P0/P1/P2 order.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return result


def build(config: dict, device: torch.device) -> torch.nn.Module:
    model_cfg = copy.deepcopy(config["model"])
    name = model_cfg.pop("name")
    model_cfg["use_mamba_ssm"] = device.type == "cuda"
    return build_model(name, **model_cfg).to(device)


def state_signature(model: torch.nn.Module) -> dict[str, list[int]]:
    return {key: list(value.shape) for key, value in model.state_dict().items()}


def convolution_flops_proxy(model: torch.nn.Module, image: torch.Tensor) -> int:
    """Same-operation proxy: Conv2d/Linear FLOPs only, excluding unsupported Mamba kernels."""

    total = 0
    handles = []
    def conv_hook(module: torch.nn.Conv2d, _args, output: torch.Tensor) -> None:
        nonlocal total
        kh, kw = module.kernel_size
        total += int(output.numel() * (module.in_channels // module.groups) * kh * kw)
    def linear_hook(module: torch.nn.Linear, _args, output: torch.Tensor) -> None:
        nonlocal total
        total += int(output.numel() * module.in_features)
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    try:
        with torch.inference_mode():
            model(image)
    finally:
        for handle in handles:
            handle.remove()
    return total


def capture_stage4(block: TopoRouteBlock, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    values: dict[str, torch.Tensor] = {}
    def pre(name: str):
        def hook(_module, args):
            values[name] = args[0].detach().clone()
            return None
        return hook
    def post(name: str, tuple_first: bool = False):
        def hook(_module, _args, output):
            values[name] = (output[0] if tuple_first else output).detach().clone()
            return None
        return hook
    handles = [
        block.route_mamba.register_forward_pre_hook(pre("route_input")),
        block.route_mamba.register_forward_hook(post("route_output", tuple_first=True)),
        block.global_branch.register_forward_pre_hook(pre("global_input")),
        block.global_branch.register_forward_hook(post("global_output")),
        block.local.register_forward_hook(post("local_output")),
        block.fuse.register_forward_pre_hook(pre("fusion_input")),
    ]
    try:
        output, route = block(x)
    finally:
        for handle in handles:
            handle.remove()
    return output, route, values


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def audit_resolved_configs(paths: list[str]) -> dict[str, object]:
    """Require the frozen Auto Protocol fields to agree after three dry-runs."""

    resolved_configs = {key: load_config(Path(path)) for key, path in zip(("P0", "P1", "P2"), paths)}
    required = (
        "dataset_fingerprint", "train_samples", "selected_regime", "selected_img_size",
        "effective_batch_size", "target_epochs", "target_updates", "resolved_mask_pos_weight",
        "loss_phases", "validation_interval_updates", "min_early_stop_update", "patience_updates",
    )
    reference = resolved_configs["P0"]
    reference_resolved = reference.get("auto_protocol_resolved", {})
    for key, config in resolved_configs.items():
        resolved = config.get("auto_protocol_resolved", {})
        require(isinstance(resolved, dict), f"missing auto_protocol_resolved: {key}")
        for field in required:
            require(resolved.get(field) == reference_resolved.get(field), f"resolved field differs {field}: {key}")
        for field in ("train", "loss", "evaluation"):
            require(config.get(field) == reference.get(field), f"frozen config section differs {field}: {key}")
        require(config.get("data", {}).get("photometric_augment") == reference.get("data", {}).get("photometric_augment"), f"augmentation differs: {key}")
        require(resolved.get("test_calls") == 0, f"skip-final-test did not record test_calls=0: {key}")
        micro, accumulation = resolved.get("micro_batch"), resolved.get("accumulation_steps")
        require(isinstance(micro, int) and isinstance(accumulation, int), f"missing micro/accumulation: {key}")
        require(micro * accumulation == int(resolved["effective_batch_size"]), f"effective batch mismatch: {key}")
    return {
        "resolved_auto_protocol_equal": True,
        "resolved_fields": {field: reference_resolved[field] for field in required},
        "micro_batch": {key: config["auto_protocol_resolved"]["micro_batch"] for key, config in resolved_configs.items()},
        "accumulation_steps": {key: config["auto_protocol_resolved"]["accumulation_steps"] for key, config in resolved_configs.items()},
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable.")
    configs = {key: load_config(path) for key, path in CONFIGS.items()}
    models = {}
    for key, config in configs.items():
        torch.manual_seed(args.seed)
        models[key] = build(config, device).eval()
    signatures = {key: state_signature(model) for key, model in models.items()}
    require(signatures["P0"] == signatures["P1"] == signatures["P2"], "state_dict keys/shapes differ")
    reference_state = models["P0"].state_dict()
    for key in ("P1", "P2"):
        require(all(torch.equal(value, models[key].state_dict()[name]) for name, value in reference_state.items()), f"same-seed initialization differs: {key}")
    counts = {key: sum(parameter.numel() for parameter in model.parameters()) for key, model in models.items()}
    require(len(set(counts.values())) == 1, f"parameter counts differ: {counts}")
    core_counts = {key: sum(isinstance(module, SequenceMambaCore) for module in model.stage4.modules()) for key, model in models.items()}
    require(set(core_counts.values()) == {1}, f"unexpected Stage4 Mamba cores: {core_counts}")
    backends = {key: model.mamba_backend_info().get("stage4", []) for key, model in models.items()}
    if device.type == "cuda":
        require(all(value == ["mamba_ssm"] for value in backends.values()), f"expected mamba_ssm: {backends}")

    h, w = args.img_size
    image = torch.randn(1, 3, h, w, device=device)
    stage_input = torch.randn(1, 320, h // 8, w // 8, device=device)
    output_shapes, dependencies = {}, {}
    for key, model in models.items():
        stage_output, route, captured = capture_stage4(model.stage4, stage_input)
        require(route.shape == (1, 4, h // 8, w // 8), f"bad route shape: {key} {tuple(route.shape)}")
        fusion = captured["fusion_input"]
        require(torch.equal(fusion[:, :320], captured["local_output"]), f"fusion local order changed: {key}")
        require(torch.equal(fusion[:, 320:640], captured["route_output"]), f"fusion route order changed: {key}")
        require(torch.equal(fusion[:, 640:], captured["global_output"]), f"fusion global order changed: {key}")
        if key == "P0":
            require(torch.equal(captured["route_input"], stage_input) and torch.equal(captured["global_input"], stage_input), "P0 dependency mismatch")
        elif key == "P1":
            require(torch.equal(captured["route_input"], stage_input) and torch.equal(captured["global_input"], captured["route_output"]), "P1 dependency mismatch")
        else:
            require(torch.equal(captured["global_input"], stage_input) and torch.equal(captured["route_input"], captured["global_output"]), "P2 dependency mismatch")
        with torch.inference_mode():
            result = model(image)
        require(all(torch.isfinite(value).all() for value in result.values()), f"non-finite output: {key}")
        output_shapes[key] = {name: list(value.shape) for name, value in result.items()}
        dependencies[key] = model.stage4.hybrid_order
        require(stage_output.shape == stage_input.shape, f"bad stage shape: {key}")

    block = models["P0"].stage4
    with torch.inference_mode():
        local = block.local(stage_input)
        routed, expected_route = block.route_mamba(stage_input)
        global_feature = block.global_branch(stage_input)
        expected = block.refine(stage_input + block.fuse(torch.cat([local, routed, global_feature], dim=1)))
        actual, route = block(stage_input)
    require(torch.equal(actual, expected) and torch.equal(route, expected_route), "P0 no longer equals the old parallel formula")
    flops_proxy = {key: convolution_flops_proxy(model, image) for key, model in models.items()}
    require(len(set(flops_proxy.values())) == 1, f"Conv/Linear FLOPs proxy differs: {flops_proxy}")
    for key, model in models.items():
        model.train()
        model.zero_grad(set_to_none=True)
        result = model(image)
        (result["mask"].mean() + result["edge"].mean() + result["route_s4"].mean()).backward()
        for prefix in ("stage4.route_mamba.shared_mamba", "stage4.global_branch.attn.qkv", "stage4.fuse", "stage4.refine"):
            gradients = [parameter.grad for name, parameter in model.named_parameters() if name.startswith(prefix)]
            require(bool(gradients) and all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients), f"missing/non-finite gradient {key}:{prefix}")

    summary = {
        "seed": args.seed,
        "device": str(device),
        "img_size": [h, w],
        "parameter_counts": counts,
        "mamba_core_counts": core_counts,
        "mamba_backends": backends,
        "conv_linear_flops_proxy": flops_proxy,
        "stage4_orders": dependencies,
        "output_shapes": output_shapes,
        "all_hard_conditions_passed": True,
    }
    if args.resolved_config:
        summary["resolved_config_audit"] = audit_resolved_configs(args.resolved_config)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"audit_seed{args.seed}.json"
    if output.exists():
        output = output_dir / f"audit_seed{args.seed}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print("Stage4 hybrid-order audit")
    print("ID | params | Stage4 Mamba cores | backend | Conv/Linear FLOPs proxy")
    for key in ("P0", "P1", "P2"):
        print(f"{key} | {counts[key]} | {core_counts[key]} | {backends[key]} | {flops_proxy[key]}")
    print(f"PASS: saved {output}")


if __name__ == "__main__":
    main()
