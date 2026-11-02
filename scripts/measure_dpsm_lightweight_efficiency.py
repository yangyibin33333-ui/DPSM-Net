"""Run the immutable L0--L4 paper efficiency protocol in an isolated directory.

This is an inference-only orchestrator.  It never reads data, checkpoints, or
training state, and it deliberately does not overwrite earlier measurements.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from math import isclose
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MEASURE = ROOT / "scripts" / "measure_model_complexity.py"
CONFIG_DIR = ROOT / "configs" / "experiments" / "dpsm_lightweight_ablation" / "crack500"
VARIANTS = (
    ("L0", "L0_NoMamba.yaml", 0),
    ("L1", "L1_SingleRasterMamba80.yaml", 1),
    ("L2", "L2_FourIndependentMamba80.yaml", 4),
    ("L3", "L3_DPSM80.yaml", 1),
    ("L4", "L4_FourIndependentMamba320.yaml", 4),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run formal L0--L4 512x512 mask-only efficiency measurements.")
    parser.add_argument("--device", default="cuda", help="Formal protocol expects the RTX 4090 D CUDA device.")
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "runs" / "paper" / "dpsm_lightweight_ablation" / "crack500" / "efficiency_mask_only_512"),
        help="A new timestamped child directory is created here.",
    )
    return parser.parse_args()


def mamba_version() -> str | None:
    try:
        return metadata.version("mamba-ssm")
    except metadata.PackageNotFoundError:
        return None


def write_markdown(path: Path, summary: list[dict[str, object]], metadata_row: dict[str, object]) -> None:
    lines = [
        "# L0--L4 formal mask-only efficiency summary",
        "",
        "Protocol: 512x512, batch 1, FP16, eval + inference_mode, 100 warmups, 300 timed forwards, three repeats.",
        "",
        "```json",
        json.dumps(metadata_row, ensure_ascii=False, indent=2),
        "```",
        "",
        "| Variant | Expected cores | Status | FLOPs status | fvcore-counted FLOPs | Mamba fused-missing estimated | Estimated FLOPs† | FPS | Log |",
        "|---|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for item in summary:
        row = item.get("report") if isinstance(item.get("report"), dict) else {}
        lines.append(
            "| {variant} | {expected} | {status} | {flops_status} | {lower} | {mamba} | {total} | {fps} | {log} |".format(
                variant=item["variant"], expected=item["expected_mamba_cores"], status=item["status"],
                flops_status=("-" if not row else row.get("flops_status")),
                lower=("-" if not row else f"{float(row['fvcore_counted_flops']) / 1e9:.4f} G"),
                mamba=("-" if not row else f"{float(row['mamba_fused_missing_flops_estimated']) / 1e9:.4f} G"),
                total=("-" if not row else f"{float(row['corrected_flops_estimated']) / 1e9:.4f} G"),
                fps=("-" if not row else f"{float(row['fps']):.2f}"), log=item["log"],
            )
        )
    lines.extend([
        "",
        "† fvcore-counted operations plus an analytical estimate for the fused Mamba operator; one fused multiply-add is counted as one FLOP.",
        "Remaining elementwise, reduction, and data-rearrangement unsupported operators are retained in JSON; these are estimated-under-declared-convention values, not exact FLOPs.",
    ])
    lines.extend([
        "",
        "## Stage4 equivalence audit",
        "",
        "| Variant | Stage4 class | route_mamba | route head | route_s4 | instantiated cores | fused calls | routed sequences | token slots | diagonal padding slots | packed | backend | input shapes | Params |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|---:|",
    ])
    for item in summary:
        row = item.get("report") if isinstance(item.get("report"), dict) else {}
        if not row:
            lines.append(f"| {item['variant']} | - | - | - | - | - | - | - | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| {item['variant']} | {row.get('stage4_class')} | {row.get('route_mamba_present')} | "
            f"{row.get('route_head_present')} | {row.get('route_s4_present')} | "
            f"{row.get('mamba_instantiated_core_count')} | {row.get('mamba_forward_call_count')} | "
            f"{row.get('mamba_routed_sequence_count')} | {row.get('mamba_token_slots')} | "
            f"{row.get('mamba_diagonal_padding_token_slots')} | {row.get('mamba_packed_execution')} | {row.get('mamba_backends')} | "
            f"{row.get('mamba_input_shapes')} | {row.get('params')} |"
        )
    lines.extend([
        "",
        "Interpretation: DPSM80 combines core sharing, packed directional execution, and padding-free ordered traversals.",
        "It must not be described as a speed gain caused solely by shared parameters.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_csv(path: Path, summary: list[dict[str, object]]) -> None:
    rows: list[dict[str, object]] = []
    for item in summary:
        report = item.get("report") if isinstance(item.get("report"), dict) else {}
        rows.append({
            "variant": item["variant"], "expected_mamba_cores": item["expected_mamba_cores"],
            "status": item["status"], "returncode": item["returncode"], "log": item["log"],
            "fvcore_counted_flops": report.get("fvcore_counted_flops"),
            "mamba_fused_missing_flops_estimated": report.get("mamba_fused_missing_flops_estimated"),
            "corrected_flops_estimated": report.get("corrected_flops_estimated"),
            "flops_status": report.get("flops_status"), "fps": report.get("fps"),
        })
    fields = list(rows[0]) if rows else ["variant", "expected_mamba_cores", "status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def validate_l2_l3_execution_disclosure(summary: list[dict[str, object]]) -> dict[str, object]:
    """Check the measured, not hard-coded, L1/L2/L3 token-slot relationships."""

    reports = {item["variant"]: item.get("report") for item in summary if isinstance(item.get("report"), dict)}
    if not {"L1", "L2", "L3"}.issubset(reports):
        return {"status": "incomplete"}
    l1, l2, l3 = (reports[name] for name in ("L1", "L2", "L3"))
    l1_slots, l2_slots, l3_slots = (int(report["mamba_token_slots"]) for report in (l1, l2, l3))
    result = {
        "status": "complete", "L1_token_slots": l1_slots, "L2_token_slots": l2_slots, "L3_token_slots": l3_slots,
        "L2_to_L3_token_slot_ratio": l2_slots / l3_slots, "L3_to_L1_token_slot_ratio": l3_slots / l1_slots,
        "L2_statement": "four independent calls with diagonal padding; packed execution=False",
        "L3_statement": "one shared packed call without diagonal padding; packed execution=True",
    }
    valid = (
        l3_slots == 4 * l1_slots
        and isclose(l2_slots / l3_slots, 24448 / 16384, rel_tol=1e-6, abs_tol=1e-6)
        and int(l2["mamba_diagonal_padding_token_slots"]) > 0
        and not bool(l2["mamba_packed_execution"])
        and int(l3["mamba_diagonal_padding_token_slots"]) == 0
        and bool(l3["mamba_packed_execution"])
    )
    if not valid:
        result["status"] = "failed"
        result["error"] = "L2/L3 captured token-slot or packing disclosure does not match the actual required topology."
    return result


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_root) / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata_row: dict[str, object] = {
        "timestamp": timestamp,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(torch.device(args.device)) if str(args.device).startswith("cuda") and torch.cuda.is_available() else None,
        "pytorch": torch.__version__, "cuda": torch.version.cuda, "mamba_ssm": mamba_version(),
        "precision": "fp16", "input": [1, 3, 512, 512], "warmup": 100, "iterations": 300,
        "benchmark_repeats": 3, "inference_path": "mask-only",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata_row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary: list[dict[str, object]] = []
    for index, (variant, filename, expected) in enumerate(VARIANTS):
        config = CONFIG_DIR / filename
        csv_path, json_path, log_path = output_dir / f"{variant}.csv", output_dir / f"{variant}.json", output_dir / f"{variant}.log"
        boundary_path = output_dir / f"{variant}.mamba_boundary.json"
        command = [
            sys.executable, str(MEASURE), "--config", str(config), "--experiment", variant,
            "--img_size", "512", "512", "--batch_size", "1", "--device", args.device,
            "--amp", "fp16", "--warmup", "100", "--iterations", "300", "--benchmark_repeats", "3",
            "--inference_path", "mask-only", "--require_mamba_ssm", "--expected-mamba-cores", str(expected),
            "--report_csv", str(csv_path), "--report_json", str(json_path), "--mamba-boundary-report", str(boundary_path),
        ]
        completed = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log_path.write_text(completed.stdout, encoding="utf-8")
        row: dict[str, object] = {
            "variant": variant, "expected_mamba_cores": expected, "status": "complete" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode, "log": log_path.name, "command": command,
        }
        if completed.returncode == 0 and json_path.exists():
            records = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(records, list) or len(records) != 1:
                row.update({"status": "failed", "returncode": 1, "error": "missing or ambiguous machine report"})
            else:
                row["report"] = records[0]
        elif completed.returncode != 0:
            row["error"] = "see per-variant log"
        summary.append(row)
        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_summary_csv(output_dir / "summary.csv", summary)
        write_markdown(output_dir / "summary.md", summary, metadata_row)
        if row["status"] != "complete":
            for pending_variant, _pending_filename, pending_expected in VARIANTS[index + 1:]:
                summary.append({
                    "variant": pending_variant, "expected_mamba_cores": pending_expected, "status": "not_run",
                    "returncode": None, "log": "-", "error": "previous formal variant failed",
                })
            (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            write_summary_csv(output_dir / "summary.csv", summary)
            write_markdown(output_dir / "summary.md", summary, metadata_row)
            print(f"{variant} failed; formal efficiency suite is incomplete. See {log_path}")
            return 1
    execution_disclosure = validate_l2_l3_execution_disclosure(summary)
    metadata_row["l2_l3_execution_disclosure"] = execution_disclosure
    (output_dir / "metadata.json").write_text(json.dumps(metadata_row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if execution_disclosure["status"] == "failed":
        write_markdown(output_dir / "summary.md", summary, metadata_row)
        print(f"L2/L3 execution disclosure failed; formal efficiency suite is incomplete: {output_dir}")
        return 1
    print(f"Formal L0--L4 efficiency suite complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
