"""How much accuracy is already on the table without retraining anything?

Three zero-training measurements on existing checkpoints, reported separately so
that none of them can be confused with a model contribution:

1. ``val-threshold``: sweep the decision threshold on the validation split,
   select the one that maximises validation F1, and report the test F1 at that
   threshold next to the fixed 0.5 used in the paper.  This is the value of
   threshold calibration for *this* checkpoint, and it is only admissible if the
   selection never looks at the test split.
2. ``ensemble``: average the predicted probabilities of the available seeds and
   report the test F1 of the average.  This bounds what seed variance costs and
   what a three-checkpoint deployment would buy, at three times the inference
   cost.
3. ``d4``: average the predictions over the dihedral group acting on the input
   (four flips, times two quarter turns when the input is square) and report the
   test F1 of the average.  For a model whose directional claims imply a
   preferred orientation, this measures how much of the score is orientation
   variance rather than orientation competence.

Nothing is trained, no threshold is tuned on the test split, and no metric is
reported for a configuration that was selected on the test split.

Runtime: the val and test splits are forwarded once per seed and the resulting
logits are kept in host memory, so every threshold sweep is free.  ``--d4``
costs eight extra forwards per test image (four when the input is not square).

Examples
--------
    # the reported DeepCrack model, one seed
    python scripts/diagnose_accuracy_headroom.py \\
        --run DeepCrack=runs/pilot/auto_protocol_v2/C40_DPSM80_DeepCrack_AUTO_PROTOCOL_V2/20260827-160052

    # a three-seed Crack500 table entry, with the dihedral check
    python scripts/diagnose_accuracy_headroom.py \\
        --run Crack500=runs/paper/dependency_aware_ablation_strict_v2/crack500/M5_FrameworkOn_DPSMOn_StructSupOn \\
        --d4
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from datasets.crack_dataset import build_dataloader  # noqa: E402
from evaluate_orientation_strata import (  # noqa: E402
    discover_seed_dirs,
    per_image_metrics,
    read_config,
)
from models import build_model  # noqa: E402
from utils.checkpoint import load_checkpoint  # noqa: E402


def parse_run(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError(f"--run expects DATASET=DIR, got {value!r}")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"run directory does not exist: {path}")
    return label.strip(), path


def threshold_grid(spec: str) -> list[float]:
    values: list[float] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            parts = [piece for piece in chunk.split(":") if piece != ""]
            if len(parts) not in (2, 3):
                raise argparse.ArgumentTypeError(f"bad threshold range {chunk!r}")
            start, stop = float(parts[0]), float(parts[1])
            step = float(parts[2]) if len(parts) == 3 else 0.05
            count = int(round((stop - start) / step))
            values.extend(round(start + index * step, 4) for index in range(count + 1))
        else:
            values.append(float(chunk))
    grid = sorted({value for value in values if 0.0 < value < 1.0})
    if not grid:
        raise argparse.ArgumentTypeError("empty threshold grid")
    return grid


def load_model(seed_dir: Path, device: torch.device) -> tuple[torch.nn.Module, dict, dict]:
    config = read_config(seed_dir / "config.yaml")
    model_config = dict(config["model"])
    model_name = str(model_config.pop("name", "DEMambaNet"))
    model = build_model(model_name, **model_config).to(device)
    load_checkpoint(seed_dir / "best_structural_score.pth", model, map_location=device)
    model.eval()
    return model, config, model_config


def make_loader(config: dict, split: str, *, batch_size: int, num_workers: int,
                data_root: Path | None):
    data_config = dict(config.get("data", {}))
    root = data_root if data_root is not None else Path(data_config.get("data_root", "./Crack500"))
    if not Path(root).is_absolute():
        root = (ROOT / root).resolve()
    img_size = data_config.get("img_size")
    img_size = tuple(int(value) for value in img_size) if img_size else None
    loader = build_dataloader(
        root, split, img_size, int(data_config.get("threshold", 127)),
        batch_size, num_workers, shuffle=False,
        target_dir_name=str(data_config.get("target_dir_name", "targets")),
    )
    return loader, root, img_size


def _flip_permutations(height: int, width: int) -> list[tuple[str, object, object]]:
    """Dihedral group as ``(name, forward, inverse)`` triples valid for this shape."""
    def identity(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def flip_h(tensor: torch.Tensor) -> torch.Tensor:
        return torch.flip(tensor, dims=[-1])

    def flip_v(tensor: torch.Tensor) -> torch.Tensor:
        return torch.flip(tensor, dims=[-2])

    def rotate(tensor: torch.Tensor) -> torch.Tensor:
        return torch.rot90(tensor, 1, dims=[-2, -1])

    def rotate_inverse(tensor: torch.Tensor) -> torch.Tensor:
        return torch.rot90(tensor, -1, dims=[-2, -1])

    group = [
        ("id", identity, identity),
        ("flip_h", flip_h, flip_h),
        ("flip_v", flip_v, flip_v),
        ("flip_hv", lambda tensor: flip_h(flip_v(tensor)), lambda tensor: flip_v(flip_h(tensor))),
    ]
    if height == width:
        base = list(group)
        for name, forward, inverse in base:
            group.append((
                f"{name}_rot90",
                lambda tensor, f=forward: rotate(f(tensor)),
                lambda tensor, i=inverse: i(rotate_inverse(tensor)),
            ))
    return group


def collect(
    model: torch.nn.Module,
    loader,
    *,
    device: torch.device,
    mask_only: bool,
    max_images: int | None,
    d4: bool,
) -> dict[str, object]:
    """Forward one split once and keep per-image logits plus targets on the host."""
    stems: list[str] = []
    logits_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    d4_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            outputs = model.forward_mask_only(images) if mask_only else model(images)
            logits = outputs["mask"].detach().float().cpu()
            d4_logits = logits
            if d4:
                height, width = images.shape[-2], images.shape[-1]
                accumulator = torch.zeros_like(logits)
                group = _flip_permutations(height, width)
                for _, forward, inverse in group:
                    transformed = forward(images)
                    transformed_outputs = (
                        model.forward_mask_only(transformed) if mask_only else model(transformed)
                    )
                    accumulator += inverse(transformed_outputs["mask"].detach().float().cpu())
                d4_logits = accumulator / len(group)
            for index, path in enumerate(batch["image_path"]):
                stems.append(Path(path).stem)
                logits_chunks.append(logits[index:index + 1].half())
                target_chunks.append(batch["mask"][index:index + 1].detach().float().cpu().half())
                if d4:
                    d4_chunks.append(d4_logits[index:index + 1].half())
            if max_images is not None and len(stems) >= max_images:
                stems = stems[:max_images]
                logits_chunks = logits_chunks[:max_images]
                target_chunks = target_chunks[:max_images]
                d4_chunks = d4_chunks[:max_images]
                break
    result: dict[str, object] = {
        "stems": stems,
        "logits": torch.cat(logits_chunks, dim=0) if logits_chunks else torch.empty(0),
        "target": torch.cat(target_chunks, dim=0) if target_chunks else torch.empty(0),
    }
    if d4:
        result["d4_logits"] = torch.cat(d4_chunks, dim=0) if d4_chunks else torch.empty(0)
    return result


def pooled_from_probability(probability: torch.Tensor, target: torch.Tensor,
                            threshold: float) -> dict[str, float]:
    prediction = probability > threshold
    truth = target.float() > 0.5
    tp = float((prediction & truth).sum())
    fp = float((prediction & ~truth).sum())
    fn = float((~prediction & truth).sum())
    f1 = 2.0 * tp / (2.0 * tp + fp + fn + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    return {"f1": f1, "iou": iou, "precision": precision, "recall": recall,
            "tp": tp, "fp": fp, "fn": fn}


def sweep(probability: torch.Tensor, target: torch.Tensor,
          thresholds: list[float]) -> list[dict[str, float]]:
    return [{"threshold": threshold, **pooled_from_probability(probability, target, threshold)}
            for threshold in thresholds]


def best_threshold(rows: list[dict[str, float]]) -> dict[str, float]:
    return max(rows, key=lambda row: (row["f1"], -abs(row["threshold"] - 0.5)))


def cross_check(logits: torch.Tensor, target: torch.Tensor, threshold: float) -> float:
    """Pooled F1 through ``per_image_metrics`` minus through the fast sweep path."""
    fast = pooled_from_probability(torch.sigmoid(logits.float()), target, threshold)["f1"]
    total_tp = total_fp = total_fn = 0.0
    for index in range(logits.shape[0]):
        metrics = per_image_metrics(logits[index:index + 1], target[index:index + 1],
                                    threshold=threshold)
        total_tp += float(metrics["tp"])
        total_fp += float(metrics["fp"])
        total_fn += float(metrics["fn"])
    reference = 2.0 * total_tp / (2.0 * total_tp + total_fp + total_fn + 1e-7)
    return fast - reference


def run_dataset(label: str, run_dir: Path, args: argparse.Namespace,
                device: torch.device) -> dict[str, object]:
    seeds = discover_seed_dirs(run_dir)
    if args.seeds != "all":
        wanted = {int(piece) for piece in args.seeds.split(",") if piece.strip()}
        seeds = [entry for entry in seeds if entry[0] in wanted]
    if not seeds:
        raise RuntimeError(f"{label}: no seed directory selected under {run_dir}")
    print(f"evaluating {label}: {run_dir}  seeds={[seed for seed, _ in seeds]}", flush=True)

    per_seed: dict[str, object] = {}
    val_probability: list[torch.Tensor] = []
    test_probability: list[torch.Tensor] = []
    test_d4_probability: list[torch.Tensor] = []
    target_val = target_test = None
    stems_test: list[str] = []

    for seed, seed_dir in seeds:
        model, config, _ = load_model(seed_dir, device)
        val_loader, root, img_size = make_loader(
            config, args.val_split, batch_size=args.batch_size,
            num_workers=args.num_workers, data_root=args.data_root_override,
        )
        val = collect(model, val_loader, device=device, mask_only=args.mask_only,
                      max_images=args.max_images, d4=False)
        if target_val is None:
            target_val = val["target"]
        val_prob = torch.sigmoid(val["logits"].float())
        val_rows = sweep(val_prob, target_val, args.grid)
        val_best = best_threshold(val_rows)
        val_probability.append(val_prob)

        test_loader, _, _ = make_loader(
            config, args.test_split, batch_size=args.batch_size,
            num_workers=args.num_workers, data_root=args.data_root_override,
        )
        test = collect(model, test_loader, device=device, mask_only=args.mask_only,
                       max_images=args.max_images, d4=args.d4)
        if target_test is None:
            target_test = test["target"]
            stems_test = list(test["stems"])
        test_prob = torch.sigmoid(test["logits"].float())
        test_probability.append(test_prob)
        discrepancy = cross_check(test["logits"], target_test, 0.5)
        test_rows = sweep(test_prob, target_test, [0.5, val_best["threshold"]])
        entry: dict[str, object] = {
            "seed": seed,
            "seed_dir": str(seed_dir),
            "data_root": str(root),
            "img_size": list(img_size) if img_size else None,
            "val_threshold_selected": val_best["threshold"],
            "val_f1_at_selected": val_best["f1"],
            "val_f1_at_0.5": pooled_from_probability(val_prob, target_val, 0.5)["f1"],
            "test_at_0.5": test_rows[0],
            "test_at_val_threshold": test_rows[1],
            "test_f1_gain_from_threshold": test_rows[1]["f1"] - test_rows[0]["f1"],
            "fast_vs_reference_f1_difference": discrepancy,
            "n_val": int(val_prob.shape[0]),
            "n_test": int(test_prob.shape[0]),
            "val_sweep": val_rows,
        }
        if args.d4 and "d4_logits" in test:
            d4_prob = torch.sigmoid(test["d4_logits"].float())
            test_d4_probability.append(d4_prob)
            d4_rows = sweep(d4_prob, target_test, [0.5, val_best["threshold"]])
            entry["test_d4_at_0.5"] = d4_rows[0]
            entry["test_d4_at_val_threshold"] = d4_rows[1]
            entry["test_f1_gain_from_d4"] = d4_rows[0]["f1"] - test_rows[0]["f1"]
        per_seed[str(seed)] = entry
        print(f"  seed {seed}: val-best t={val_best['threshold']:.2f} "
              f"test@0.5={test_rows[0]['f1']:.4f} test@{val_best['threshold']:.2f}="
              f"{test_rows[1]['f1']:.4f}", flush=True)

    summary: dict[str, object] = {"runs": per_seed, "n_seeds": len(seeds)}

    single_f1 = [entry["test_at_0.5"]["f1"] for entry in per_seed.values()]
    summary["single_seed_test_f1"] = {
        "mean": statistics.fmean(single_f1),
        "sd": statistics.stdev(single_f1) if len(single_f1) > 1 else 0.0,
        "values": single_f1,
    }
    threshold_gains = [entry["test_f1_gain_from_threshold"] for entry in per_seed.values()]
    summary["threshold_gain"] = {
        "mean": statistics.fmean(threshold_gains),
        "sd": statistics.stdev(threshold_gains) if len(threshold_gains) > 1 else 0.0,
        "positive_in": sum(1 for gain in threshold_gains if gain > 0),
        "values": threshold_gains,
        "selected_thresholds": [entry["val_threshold_selected"] for entry in per_seed.values()],
    }

    if len(seeds) > 1:
        ensemble_prob = torch.stack(test_probability, dim=0).mean(dim=0)
        ensemble_val = torch.stack(val_probability, dim=0).mean(dim=0)
        ensemble_val_rows = sweep(ensemble_val, target_val, args.grid)
        ensemble_best = best_threshold(ensemble_val_rows)
        ensemble_rows = sweep(ensemble_prob, target_test, [0.5, ensemble_best["threshold"]])
        summary["ensemble"] = {
            "test_at_0.5": ensemble_rows[0],
            "test_at_val_threshold": ensemble_rows[1],
            "val_threshold_selected": ensemble_best["threshold"],
            "test_f1_gain_over_best_single": (
                ensemble_rows[0]["f1"] - max(single_f1)
            ),
            "test_f1_gain_over_mean_single": (
                ensemble_rows[0]["f1"] - statistics.fmean(single_f1)
            ),
        }
        if args.d4 and test_d4_probability:
            d4_ensemble = torch.stack(test_d4_probability, dim=0).mean(dim=0)
            d4_ensemble_rows = sweep(d4_ensemble, target_test, [0.5, ensemble_best["threshold"]])
            summary["ensemble_d4"] = {
                "test_at_0.5": d4_ensemble_rows[0],
                "test_at_val_threshold": d4_ensemble_rows[1],
                "test_f1_gain_over_ensemble": (
                    d4_ensemble_rows[0]["f1"] - ensemble_rows[0]["f1"]
                ),
            }

    if args.d4 and test_d4_probability:
        gains = [entry["test_f1_gain_from_d4"] for entry in per_seed.values()
                 if "test_f1_gain_from_d4" in entry]
        summary["d4_gain"] = {
            "mean": statistics.fmean(gains),
            "sd": statistics.stdev(gains) if len(gains) > 1 else 0.0,
            "positive_in": sum(1 for gain in gains if gain > 0),
            "values": gains,
        }

    summary["stems"] = stems_test[:5]
    return summary


def render_markdown(table: dict[str, object], metadata: dict[str, object]) -> list[str]:
    lines = [
        "# Accuracy headroom without retraining",
        "",
        f"Protocol: val split `{metadata['val_split']}`, test split `{metadata['test_split']}`, "
        f"threshold grid {metadata['grid'][0]:.2f}..{metadata['grid'][-1]:.2f} "
        f"({len(metadata['grid'])} points), fixed paper threshold 0.5.",
        "",
        "## 1. Per-seed results",
        "",
        "| dataset | seed | test F1 @0.5 | val-selected t | test F1 @t | gain (pp) | D4 gain (pp) "
        "| fast-vs-ref F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, entry in table.items():
        for seed, run in entry["runs"].items():
            d4 = run.get("test_f1_gain_from_d4")
            d4_text = f"{d4 * 100:+.2f}" if d4 is not None else "--"
            lines.append(
                f"| {label} | {seed} | {run['test_at_0.5']['f1']:.4f} "
                f"| {run['val_threshold_selected']:.2f} "
                f"| {run['test_at_val_threshold']['f1']:.4f} "
                f"| {run['test_f1_gain_from_threshold'] * 100:+.2f} "
                f"| {d4_text} "
                f"| {run['fast_vs_reference_f1_difference']:+.2e} |"
            )

    lines += ["", "## 2. Aggregates", "",
              "| dataset | seeds | mean single F1 | threshold gain (pp) | positive seeds "
              "| ensemble gain vs mean (pp) | D4 gain (pp) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for label, entry in table.items():
        threshold = entry["threshold_gain"]
        ensemble = entry.get("ensemble")
        ensemble_text = (
            f"{ensemble['test_f1_gain_over_mean_single'] * 100:+.2f}" if ensemble else "--"
        )
        d4 = entry.get("d4_gain")
        d4_text = f"{d4['mean'] * 100:+.2f} ({d4['positive_in']}/{len(d4['values'])})" if d4 else "--"
        lines.append(
            f"| {label} | {entry['n_seeds']} "
            f"| {entry['single_seed_test_f1']['mean']:.4f} "
            f"| {threshold['mean'] * 100:+.2f} "
            f"| {threshold['positive_in']}/{len(threshold['values'])} "
            f"| {ensemble_text} | {d4_text} |"
        )

    lines += [
        "",
        "## Reading",
        "",
        "`fast-vs-ref F1` is the difference between the vectorised sweep path and "
        "`per_image_metrics` at threshold 0.5 on the same logits; it must be ~0, "
        "otherwise the two paths disagree and the table must not be used.",
        "",
        "`threshold gain` is what a validation-selected operating point buys for the "
        "checkpoints that were trained at a fixed 0.5. It is a protocol change, not a "
        "model change: it is admissible only if the selection stays on the validation "
        "split and the change is disclosed the same way the competitor rows are.",
        "",
        "`ensemble gain` is the mean probability over seeds; it costs N times the "
        "inference, so it is an upper bound for what seed variance hides rather than a "
        "deployable configuration under this paper's efficiency protocol.",
        "",
        "`D4 gain` is the mean over the dihedral group acting on the input. A model "
        "whose four-direction scan genuinely carries orientation information should "
        "not gain from orientation averaging; a positive value means part of the score "
        "was orientation variance.",
    ]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", action="append", type=parse_run, required=True,
                        metavar="DATASET=DIR",
                        help="Dataset label and a run/seed directory (repeatable)")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--thresholds", default="0.20:0.80:0.02",
                        help="Threshold grid: 'start:stop:step' or a comma list")
    parser.add_argument("--seeds", default="all", help="'all' or a comma list of seeds")
    parser.add_argument("--d4", action="store_true", help="Add the dihedral averaging pass")
    parser.add_argument("--data-root-override", type=Path, default=None,
                        help="Override data_root for every run")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mask-only", action="store_true",
                        help="Use the eval-only mask path (bitwise identical mask, faster)")
    parser.add_argument("--max-images", type=int, default=None, help="Debug subset size")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "runs/diagnostics/accuracy_headroom")
    args = parser.parse_args()

    args.grid = threshold_grid(args.thresholds)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    metadata = {
        "created": datetime.now().astimezone().isoformat(),
        "script": "diagnose_accuracy_headroom.py",
        "question": "How much test F1 is available from threshold calibration, seed "
                    "averaging, and orientation averaging, on the existing checkpoints?",
        "val_split": args.val_split,
        "test_split": args.test_split,
        "grid": args.grid,
        "fixed_threshold": 0.5,
        "d4": bool(args.d4),
        "mask_only": bool(args.mask_only),
        "max_images": args.max_images,
    }

    table: dict[str, object] = {}
    for label, run_dir in args.run:
        table[label] = run_dataset(label, run_dir, args, device)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_root / f"HEADROOM_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "headroom.json").write_text(
        json.dumps({"metadata": metadata, "datasets": table}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "headroom.md").write_text(
        "\n".join(render_markdown(table, metadata)) + "\n", encoding="utf-8"
    )

    print()
    for label, entry in table.items():
        threshold = entry["threshold_gain"]
        line = (f"{label:>12s} single F1={entry['single_seed_test_f1']['mean']:.4f} "
                f"threshold={threshold['mean'] * 100:+.2f}pp "
                f"({threshold['positive_in']}/{len(threshold['values'])})")
        ensemble = entry.get("ensemble")
        if ensemble:
            line += f" ensemble={ensemble['test_f1_gain_over_mean_single'] * 100:+.2f}pp"
        d4 = entry.get("d4_gain")
        if d4:
            line += f" d4={d4['mean'] * 100:+.2f}pp ({d4['positive_in']}/{len(d4['values'])})"
        print(line)
        for seed, run in entry["runs"].items():
            if abs(run["fast_vs_reference_f1_difference"]) > 1e-6:
                print(f"  WARNING seed {seed}: sweep path and per_image_metrics disagree by "
                      f"{run['fast_vs_reference_f1_difference']:.2e}")
    print(f"results written: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
