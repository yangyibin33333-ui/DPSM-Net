"""Read-only P0 validation diagnosis for Stage4 branch contributions and DPSM routes.

This is deliberately a post-training sensitivity probe, not a training-time
ablation.  It accepts *only* the validation split and restores the model object,
module modes, hooks, and state dictionary before it returns.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import build_dataloader
from metrics import SegmentationMetric
from models import build_model
from models.blocks import DirectionPartitionedSharedMamba, TopoRouteBlock
from utils import load_checkpoint, set_seed


METRIC_KEYS = ("mIoU", "F1", "Precision", "Recall", "ODS", "OIS", "BoundaryF1", "clDice", "ComponentError")
BRANCHES = ("local", "dpsm", "transformer")
MODES = ("normal", "remove_local_delta", "remove_dpsm_delta", "remove_transformer_delta")


def validate_val_split(split: str) -> str:
    """Keep this diagnostic incapable of constructing a train/test loader."""
    if split != "val":
        raise ValueError("diagnose_stage4_branch_route is validation-only; --split must be exactly 'val'.")
    return split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Read-only P0 Stage4 branch and DPSM route diagnosis")
    parser.add_argument("--config", required=True, help="Resolved training config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Selected P0 checkpoint")
    parser.add_argument("--split", default="val", help="Must be val; train/test are deliberately rejected")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    validate_val_split(args.split)
    if args.batch_size != 1:
        parser.error("--batch_size must be 1 so every CSV row is one unambiguous validation image.")
    if args.num_workers < 0:
        parser.error("--num_workers must be non-negative")
    return args


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"Expected mapping config at {path}")
    return result


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    if not fields:
        fields = ["empty"]
        rows = [{"empty": ""}]
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def tensor_summary(values: list[float]) -> dict[str, float | int]:
    clean = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if clean.size == 0:
        return {key: 0.0 for key in ("mean", "std", "median", "p10", "p25", "p75", "p90", "min", "max")} | {"count": 0}
    return {
        "mean": float(clean.mean()), "std": float(clean.std()), "median": float(np.median(clean)),
        "p10": float(np.percentile(clean, 10)), "p25": float(np.percentile(clean, 25)),
        "p75": float(np.percentile(clean, 75)), "p90": float(np.percentile(clean, 90)),
        "min": float(clean.min()), "max": float(clean.max()), "count": int(clean.size),
    }


def _flat_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach().float().reshape(-1), b.detach().float().reshape(-1)
    return float(F.cosine_similarity(a[None], b[None], dim=1, eps=1e-12).item())


def branch_statistics(x: torch.Tensor, local: torch.Tensor, routed: torch.Tensor, global_feature: torch.Tensor) -> dict[str, float]:
    """Per-image quantities; dataset aggregation never pools pixels between images."""
    values = {"local": local, "dpsm": routed, "transformer": global_feature}
    result: dict[str, float] = {}
    x_norm = float(torch.linalg.vector_norm(x.float()).item())
    deltas: dict[str, torch.Tensor] = {}
    for name, value in values.items():
        delta = value - x
        deltas[name] = delta
        result[f"{name}_relative_delta_l2"] = float(torch.linalg.vector_norm(delta.float()).item() / max(x_norm, 1e-12))
        result[f"{name}_raw_cosine_with_x"] = _flat_cosine(value, x)
    for left, right in (("local", "dpsm"), ("local", "transformer"), ("dpsm", "transformer")):
        result[f"raw_cosine_{left}_{right}"] = _flat_cosine(values[left], values[right])
        result[f"delta_cosine_{left}_{right}"] = _flat_cosine(deltas[left], deltas[right])
    return result


def fusion_weight_groups(block: TopoRouteBlock) -> list[dict[str, Any]]:
    conv = block.fuse.block[0]
    if not isinstance(conv, nn.Conv2d) or conv.in_channels != block.channels * 3:
        raise ValueError("P0 Stage4 fuse must be the frozen 960-to-320 Conv2d.")
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(BRANCHES):
        weight = conv.weight[:, index * block.channels : (index + 1) * block.channels]
        rows.append({"kind": "weight", "branch": name, "l1": float(weight.abs().sum()), "l2": float(torch.linalg.vector_norm(weight)), "rms": float(weight.square().mean().sqrt()), "parameters": int(weight.numel())})
    return rows


def fusion_contributions(block: TopoRouteBlock, local: torch.Tensor, routed: torch.Tensor, global_feature: torch.Tensor, expected_pre_bn: torch.Tensor | None = None, *, include_bias: bool = True) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Exact grouped 1x1-convolution decomposition immediately before fuse BN."""
    conv = block.fuse.block[0]
    if not isinstance(conv, nn.Conv2d):
        raise TypeError("Stage4 fuse first module must be Conv2d")
    source = {"local": local, "dpsm": routed, "transformer": global_feature}
    parts: dict[str, torch.Tensor] = {}
    for index, name in enumerate(BRANCHES):
        weight = conv.weight[:, index * block.channels : (index + 1) * block.channels]
        parts[name] = F.conv2d(source[name], weight, None, conv.stride, conv.padding, conv.dilation, conv.groups)
    sum_parts = sum(parts.values())
    if include_bias and conv.bias is not None:
        sum_parts = sum_parts + conv.bias.view(1, -1, 1, 1)
    official = expected_pre_bn if expected_pre_bn is not None else conv(torch.cat([local, routed, global_feature], dim=1))
    if not include_bias and conv.bias is not None:
        official = official - conv.bias.view(1, -1, 1, 1)
    max_abs = float((sum_parts - official).abs().max().item())
    # CUDA's grouped slices perform three independent GEMM reductions whereas
    # the official 960-channel Conv2d has one reduction order.  Accept only a
    # documented float32 rounding envelope, scaled by the observed pre-BN
    # magnitude; a material topology/fusion mismatch remains a hard failure.
    max_reference = float(official.detach().float().abs().max().item())
    tolerance = 5e-5 + 1e-5 * max_reference
    if max_abs > tolerance:
        raise RuntimeError(
            "Stage4 fusion decomposition mismatch "
            f"(max abs {max_abs:.3e}, tolerance {tolerance:.3e}, reference {max_reference:.3e})."
        )
    norms = {name: float(torch.linalg.vector_norm(value.float()).item()) for name, value in parts.items()}
    denom = max(sum(norms.values()), 1e-12)
    summary: dict[str, float] = {"fusion_decomposition_max_abs": max_abs, "fusion_decomposition_tolerance": tolerance, "fusion_cancellation": 1.0 - float(torch.linalg.vector_norm(sum_parts.float()).item()) / denom}
    for name, value in parts.items():
        summary[f"fusion_{name}_l1"] = float(value.abs().sum())
        summary[f"fusion_{name}_l2"] = norms[name]
        summary[f"fusion_{name}_rms"] = float(value.float().square().mean().sqrt())
        summary[f"fusion_{name}_l2_share"] = norms[name] / denom
    for left, right in (("local", "dpsm"), ("local", "transformer"), ("dpsm", "transformer")):
        summary[f"fusion_cosine_{left}_{right}"] = _flat_cosine(parts[left], parts[right])
    return parts, summary


def aligned_tangent_targets(logits: torch.Tensor, tangent_gt: torch.Tensor, tangent_valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exactly TangentRouteLoss._single's adaptive target/valid construction."""
    size = logits.shape[-2:]
    valid = F.adaptive_max_pool2d(tangent_valid.float(), size)
    target = F.adaptive_avg_pool2d(tangent_gt.float() * tangent_valid.float(), size)
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return target, valid > 0.5


def route_statistics(logits: torch.Tensor, tangent_gt: torch.Tensor, tangent_valid: torch.Tensor, mask: torch.Tensor, edge_gt: torch.Tensor) -> tuple[dict[str, float], torch.Tensor]:
    probability = logits.detach().float().softmax(dim=1)
    target, valid = aligned_tangent_targets(logits, tangent_gt, tangent_valid)
    eps = 1e-12
    entropy = -(probability * probability.clamp_min(eps).log()).sum(1) / math.log(probability.shape[1])
    max_probability, prediction = probability.max(dim=1)
    top2 = probability.topk(2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    effective = torch.exp(-(probability * probability.clamp_min(eps).log()).sum(1))
    pooled_mask = F.adaptive_max_pool2d(mask.float(), logits.shape[-2:]) > 0.5
    pooled_edge = F.adaptive_max_pool2d(edge_gt.float(), logits.shape[-2:]) > 0.5
    target_has_mass = target.sum(dim=1) > 0
    regions = {
        "all": torch.ones_like(valid[:, 0], dtype=torch.bool), "fg": pooled_mask[:, 0], "bg": ~pooled_mask[:, 0],
        "edge_gt": pooled_edge[:, 0], "tangent_valid": valid[:, 0], "fg_valid": pooled_mask[:, 0] & valid[:, 0],
        "fg_invalid": pooled_mask[:, 0] & ~valid[:, 0],
    }
    row: dict[str, float] = {}
    for name, region in regions.items():
        count = int(region.sum())
        row[f"route_{name}_pixels"] = count
        for metric_name, tensor in (("entropy_normalized", entropy), ("max_probability", max_probability), ("margin", margin), ("effective_directions", effective)):
            row[f"route_{name}_{metric_name}"] = float(tensor[region].mean()) if count else float("nan")
    for direction in range(4):
        row[f"route_soft_use_d{direction}"] = float(probability[:, direction].mean())
        row[f"route_argmax_share_d{direction}"] = float((prediction == direction).float().mean())
    row["route_collapse_max_soft_use"] = max(row[f"route_soft_use_d{i}"] for i in range(4))
    row["route_collapse_max_argmax_share"] = max(row[f"route_argmax_share_d{i}"] for i in range(4))
    route_valid = valid[:, 0] & target_has_mass
    if int(route_valid.sum()):
        target_label = target.argmax(dim=1)
        pred_label = prediction
        # Probability mass assigned to the (possibly soft) tangent target.
        # This is a calibrated route-quality quantity; it must not be the
        # target value selected by the model's argmax direction.
        target_mass = (probability * target).sum(dim=1)
        log_probability = probability.clamp_min(eps).log()
        ce = -(target * log_probability).sum(dim=1)
        kl = (target * (target.clamp_min(eps).log() - log_probability)).sum(dim=1)
        brier = (probability - target).square().sum(dim=1)
        row.update({
            "route_tangent_valid_top1_accuracy": float((pred_label[route_valid] == target_label[route_valid]).float().mean()),
            "route_tangent_valid_target_mass": float(target_mass[route_valid].mean()),
            "route_tangent_valid_soft_ce": float(ce[route_valid].mean()), "route_tangent_valid_kl": float(kl[route_valid].mean()),
            "route_tangent_valid_brier": float(brier[route_valid].mean()),
        })
    else:
        for name in ("top1_accuracy", "target_mass", "soft_ce", "kl", "brier"):
            row[f"route_tangent_valid_{name}"] = float("nan")
    return row, _route_confusion(prediction, target.argmax(dim=1), route_valid)


def _route_confusion(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    result = torch.zeros((4, 4), dtype=torch.int64)
    for expected in range(4):
        for observed in range(4):
            result[expected, observed] = ((target == expected) & (prediction == observed) & valid).sum().cpu()
    return result


def route_confusion_summary(confusion: torch.Tensor) -> tuple[list[dict[str, Any]], dict[str, float]]:
    support = confusion.sum(1)
    total = int(confusion.sum())
    majority = float(support.max() / max(total, 1))
    per_direction: dict[int, dict[str, float | int]] = {}
    for direction in range(4):
        tp = int(confusion[direction, direction])
        fp = int(confusion[:, direction].sum() - tp)
        fn = int(confusion[direction].sum() - tp)
        precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        per_direction[direction] = {"tp": tp, "support": int(support[direction]), "precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-12)}
    rows: list[dict[str, Any]] = []
    for expected in range(4):
        for observed in range(4):
            rows.append({"target_direction": expected, "predicted_direction": observed, "count": int(confusion[expected, observed]), **per_direction[expected]})
    return rows, {"route_tangent_valid_pixels": total, "route_majority_baseline": majority, "route_random_baseline": 0.25}


class Stage4CounterfactualWrapper(nn.Module):
    """Script-local P0 wrapper; it never changes the checkpoint or block math."""
    def __init__(self, original: TopoRouteBlock, mode: str):
        super().__init__()
        if original.hybrid_order != "parallel":
            raise ValueError("Branch-route diagnostic supports frozen P0 parallel Stage4 only.")
        if mode not in MODES:
            raise ValueError(f"Unknown diagnostic mode: {mode}")
        self.original, self.mode = original, mode
        self.last: dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if kwargs:
            raise ValueError("P0 diagnostic does not support auxiliary Stage4 gates/scales.")
        local = self.original.local(x)
        routed, route_logits = self.original.route_mamba(x)  # Still runs for every counterfactual.
        global_feature = self.original.global_branch(x)
        selected = {"local": local, "dpsm": routed, "transformer": global_feature}
        if self.mode == "remove_local_delta": selected["local"] = x
        elif self.mode == "remove_dpsm_delta": selected["dpsm"] = x
        elif self.mode == "remove_transformer_delta": selected["transformer"] = x
        concat = torch.cat([selected["local"], selected["dpsm"], selected["transformer"]], dim=1)
        fused = self.original.fuse(concat)
        final = self.original.refine(x + fused)
        self.last = {"x": x.detach(), "local": local.detach(), "routed": routed.detach(), "global": global_feature.detach(), "route_logits": route_logits.detach(), "pre_bn": self.original.fuse.block[0](concat).detach(), "final": final.detach()}
        return final, route_logits


def _capture_normal(model: nn.Module, images: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    block = model.stage4
    if not isinstance(block, TopoRouteBlock):
        raise TypeError("P0 diagnostic requires TopoRouteBlock at model.stage4")
    captured: dict[str, torch.Tensor] = {}
    hooks = []
    def save_input(name: str):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            captured[name] = inputs[0].detach()
        return hook
    def save_output(name: str):
        def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: Any) -> None:
            captured[name] = (output[0] if isinstance(output, tuple) else output).detach()
            if name == "route_mamba": captured["route_logits"] = output[1].detach()
        return hook
    hooks.extend([block.register_forward_pre_hook(save_input("x")), block.local.register_forward_hook(save_output("local")), block.route_mamba.register_forward_hook(save_output("route_mamba")), block.global_branch.register_forward_hook(save_output("global")), block.fuse.block[0].register_forward_hook(save_output("pre_bn")), block.register_forward_hook(save_output("final"))])
    try:
        outputs = model(images)
    finally:
        for hook in hooks: hook.remove()
    captured["routed"] = captured.pop("route_mamba")
    required = {"x", "local", "routed", "global", "route_logits", "pre_bn", "final"}
    if set(captured) != required:
        raise RuntimeError(f"Incomplete Stage4 capture; missing {required - set(captured)}")
    return outputs, captured


def _evaluate(model: nn.Module, loader: Any, device: torch.device, mode: str, expected_routes: dict[str, torch.Tensor]) -> dict[str, float]:
    original = model.stage4
    if not isinstance(original, TopoRouteBlock):
        raise TypeError("Expected TopoRouteBlock")
    wrapper = Stage4CounterfactualWrapper(original, mode)
    model.stage4 = wrapper
    metric = SegmentationMetric()
    try:
        with torch.inference_mode():
            for batch in loader:
                output = model(batch["image"].to(device))
                image_path = str(batch["image_path"][0])
                if not torch.equal(output["route_s4"].detach().cpu(), expected_routes[image_path]):
                    raise RuntimeError(f"DPSM route logits changed during {mode}; the route branch must still execute unchanged.")
                metric.update(output["mask"], batch["mask"].to(device))
    finally:
        model.stage4 = original
    return metric.compute()


def _state_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _assert_state_unchanged(model: nn.Module, before: dict[str, torch.Tensor]) -> None:
    after = model.state_dict()
    if before.keys() != after.keys() or any(not torch.equal(value, after[name].detach().cpu()) for name, value in before.items()):
        raise RuntimeError("Read-only diagnostic changed model state_dict; refusing to report results.")


def build_val_loader(cfg: dict[str, Any], batch_size: int, num_workers: int):
    """The only data loader entry point in this module: explicitly validation-only."""
    validate_val_split("val")
    data, train = cfg["data"], cfg["train"]
    return build_dataloader(data["data_root"], "val", data.get("img_size"), int(data["threshold"]), batch_size, num_workers, shuffle=False, target_dir_name=data.get("target_dir_name", "targets"), persistent_workers=bool(train.get("persistent_workers", False)) and num_workers > 0, prefetch_factor=int(train.get("prefetch_factor", 2)), allow_quarter_turns=bool(data.get("allow_quarter_turns", True)))


def diagnose(model: nn.Module, loader: Any, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run normal capture, route/branch tables, then four post-training sensitivity replays."""
    if not isinstance(model.stage4, TopoRouteBlock) or not isinstance(model.stage4.route_mamba, DirectionPartitionedSharedMamba):
        raise ValueError("Diagnostic requires P0 TopoRouteBlock with DirectionPartitionedSharedMamba.")
    if model.stage4.hybrid_order != "parallel":
        raise ValueError("Diagnostic requires frozen P0 stage4_hybrid_order=parallel.")
    per_image: list[dict[str, Any]] = []
    fusion_rows = fusion_weight_groups(model.stage4)
    confusion = torch.zeros((4, 4), dtype=torch.int64)
    normal_metric = SegmentationMetric()
    baseline_logits: dict[str, torch.Tensor] = {}
    baseline_routes: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            outputs, capture = _capture_normal(model, images)
            route_logits = outputs.get("route_s4")
            if route_logits is None or not torch.equal(route_logits, capture["route_logits"]):
                raise RuntimeError("Captured Stage4 route logits do not match model route_s4 output.")
            normal_metric.update(outputs["mask"], batch["mask"].to(device))
            image_path = str(batch["image_path"][0])
            baseline_logits[image_path] = outputs["mask"].detach().cpu().clone()
            baseline_routes[image_path] = route_logits.detach().cpu().clone()
            row: dict[str, Any] = {"image_path": image_path, "stem": Path(image_path).stem}
            row.update(branch_statistics(capture["x"], capture["local"], capture["routed"], capture["global"]))
            parts, fusion = fusion_contributions(model.stage4, capture["local"], capture["routed"], capture["global"], capture["pre_bn"])
            row.update(fusion)
            delta_parts, delta_fusion = fusion_contributions(model.stage4, capture["local"] - capture["x"], capture["routed"] - capture["x"], capture["global"] - capture["x"], include_bias=False)
            row.update({f"delta_{key}": value for key, value in delta_fusion.items()})
            route_row, image_confusion = route_statistics(route_logits, batch["tangent_gt"].to(device), batch["tangent_valid"].to(device), batch["mask"].to(device), batch["edge_gt"].to(device))
            row.update(route_row); confusion += image_confusion
            per_image.append(row)
            for kind, values in (("activation", parts), ("delta_activation", delta_parts)):
                for branch, value in values.items():
                    fusion_rows.append({"kind": kind, "image_path": image_path, "stem": Path(image_path).stem, "branch": branch, "l1": float(value.abs().sum()), "l2": float(torch.linalg.vector_norm(value.float())), "rms": float(value.float().square().mean().sqrt())})
    metrics = {"normal": normal_metric.compute()}
    for mode in MODES[1:]: metrics[mode] = _evaluate(model, loader, device, mode, baseline_routes)
    # Exact replay guards the temporary wrapper against an accidental semantic drift.
    original = model.stage4; wrapper = Stage4CounterfactualWrapper(original, "normal"); model.stage4 = wrapper
    try:
        with torch.inference_mode():
            for batch in loader:
                path = str(batch["image_path"][0]); replay = model(batch["image"].to(device))
                if not torch.equal(replay["mask"].detach().cpu(), baseline_logits[path]):
                    raise RuntimeError("Counterfactual normal replay differs from the original P0 output.")
                if not torch.equal(replay["route_s4"].detach().cpu(), baseline_routes[path]):
                    raise RuntimeError("Counterfactual normal replay route logits differ from original P0.")
    finally:
        model.stage4 = original
    confusion_rows, confusion_summary = route_confusion_summary(confusion)
    metric_summary = {mode: {key: float(value) for key, value in values.items() if key in METRIC_KEYS} for mode, values in metrics.items()}
    deltas = {mode: {key: metric_summary[mode][key] - metric_summary["normal"][key] for key in METRIC_KEYS} for mode in MODES[1:]}
    def aggregate_fields(predicate: Any) -> dict[str, dict[str, float | int]]:
        return {key: tensor_summary([float(row[key]) for row in per_image]) for key in sorted(key for key in per_image[0] if predicate(key))}
    aggregate = {
        "branch_statistics": aggregate_fields(lambda key: key.endswith("l2") or "cosine" in key),
        "fusion_statistics": aggregate_fields(lambda key: key.startswith("fusion_") or key.startswith("delta_fusion_")),
        "route_statistics": aggregate_fields(lambda key: key.startswith("route_")),
        "route_confusion": confusion_summary, "metrics": metric_summary,
        "counterfactual_delta_from_normal": deltas, "normal_replay_exact": True,
        "route_mamba_runs_in_each_counterfactual": True,
    }
    return per_image, fusion_rows, confusion_rows, aggregate


def _summary_markdown(summary: dict[str, Any]) -> str:
    return "\n".join([
        "# P0 Stage4 branch contribution and DPSM route diagnosis", "",
        "This is a validation-only, post-training sensitivity diagnosis. It does not constitute a new formal ablation and must not automatically change the model or training protocol.", "",
        f"- Validation images: {summary['validation']['samples']}",
        f"- Normal counterfactual replay exact: {summary['diagnostic']['normal_replay_exact']}",
        f"- Test loader constructed: {summary['validation']['test_loader_constructed']}",
        "- `remove_*_delta` replaces one branch feature with Stage4 input `x`; the branch is still executed, so this measures post-training dependence on its learned residual rather than compute removal.", "",
        "Review `summary.json`, `per_image.csv`, `fusion_contributions.csv`, and `route_confusion.csv` together. Judge redundancy from raw and delta cosine distributions, fusion contribution shares/cancellation, counterfactual metric deltas, and route-target agreement; do not infer a training-time causal conclusion from this checkpoint-only probe.", "",
    ])


def main() -> int:
    args = parse_args(); validate_val_split(args.split)
    config_path, checkpoint_path = Path(args.config), Path(args.checkpoint)
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("Both --config and --checkpoint must exist; this diagnostic never trains a replacement.")
    cfg = load_yaml(config_path)
    if str(cfg.get("model", {}).get("stage4_hybrid_order", "parallel")) != "parallel":
        raise ValueError("The requested diagnosis is P0 only: resolved config must set stage4_hybrid_order: parallel.")
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("--device cuda was requested but CUDA is unavailable.")
    model_cfg = dict(cfg["model"]); model_name = str(model_cfg.pop("name", "DEMambaNet"))
    model = build_model(model_name, **model_cfg).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    state_before, modes_before = _state_snapshot(model), [(module, module.training) for module in model.modules()]
    model.eval(); loader = build_val_loader(cfg, args.batch_size, args.num_workers)
    try:
        per_image, fusion_rows, confusion_rows, diagnostic = diagnose(model, loader, device)
        _assert_state_unchanged(model, state_before)
    finally:
        for module, was_training in modes_before: module.train(was_training)
    output_dir = Path(args.output_dir)
    info = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device), "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "config_sha256": sha256_file(config_path), "checkpoint_sha256": sha256_file(checkpoint_path), "mamba_backend": model.mamba_backend_info(), "seed": args.seed}
    summary = {"inputs": {"config": str(config_path), "checkpoint": str(checkpoint_path), **info}, "validation": {"split": "val", "samples": len(per_image), "batch_size": args.batch_size, "test_loader_constructed": False, "test_samples": 0}, "diagnostic": diagnostic}
    atomic_csv(output_dir / "per_image.csv", per_image)
    atomic_csv(output_dir / "fusion_contributions.csv", fusion_rows)
    atomic_csv(output_dir / "route_confusion.csv", confusion_rows)
    atomic_json(output_dir / "summary.json", summary)
    atomic_text(output_dir / "summary.md", _summary_markdown(summary))
    print(f"Diagnostic manifest | torch={info['torch']} | cuda={info['cuda']} | device={info['device']} | seed={args.seed}")
    print(f"Diagnostic hashes | config={info['config_sha256']} | checkpoint={info['checkpoint_sha256']}")
    print(f"PASS: validation-only P0 diagnostic | samples={len(per_image)} | output={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
