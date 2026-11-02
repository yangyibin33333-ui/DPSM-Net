"""Check that a candidate run is protocol-identical to a reference run.

The paper's ablations are only pairable if every run in a cell resolves to the
*same* auto-protocol plan: same model graph, same data contract, same optimiser
hyper-parameters, same resolved update budget / loss-phase boundaries and the
same seed.  This tool diffs two (or more) saved ``config.yaml`` files on
exactly those items and prints one PASS/FAIL line per candidate plus a
``reference -> candidate`` line per offending key.

Only what is listed below is compared; anything else in the config (paths,
timestamps, telemetry, checkpoints, per-run bookkeeping) is deliberately
ignored so that a fresh run can be compared against an archived one.

  model                         entire section (flattened, key by key)
  data.data_root
  data.threshold
  data.target_dir_name
  train.batch_size
  train.lr
  train.weight_decay
  train.amp
  train.amp_dtype
  auto_protocol_resolved.train_samples
  auto_protocol_resolved.selected_regime
  auto_protocol_resolved.selected_img_size
  auto_protocol_resolved.effective_batch_size
  auto_protocol_resolved.target_updates
  auto_protocol_resolved.resolved_epochs
  auto_protocol_resolved.scnp_warmup_updates
  auto_protocol_resolved.bpcm_start_update
  auto_protocol_resolved.bpcm_ramp_updates
  auto_protocol_resolved.resolved_validation_interval_updates
  auto_protocol_resolved.min_early_stop_update
  auto_protocol_resolved.patience_updates
  auto_protocol_resolved.training_seed
  loss                          entire section (flattened, key by key)
  save.save_dir

Everything that is compared must be equal unless it is explicitly excused: the
two sanctioned exemptions are the *intended* loss knobs of an ablation cell
(``--allow-loss-key``) and the per-run output directory (``--allow-save-dir``).
The model and loss sections are compared as a whole precisely so that an
accidental change of a loss weight, a margin or a schedule cannot slip through
unnoticed.

A key that is missing on either side counts as a mismatch.  The single nuance
applies to the two whole sections: a key written on one side only with a
*neutral* value (``0`` / ``false`` / ``null`` / ``""``) is reported as a note
instead of a failure, because every loss weight defaults to zero and every model
flag to off, so ``lambda_cfps: 0.0`` and no such key at all describe the same
protocol.  An archived run may legitimately carry such an explicit zero while a
new cell omits it.  A one-sided key with a non-neutral value is still a failure.

The script is read-only: it never writes any file.

Typical call for one loss-ablation cell (flags shown one per line; put them on a
single line when typing):

  python scripts/verify_run_protocol_identity.py
      --reference <archived A run>/config.yaml
      --candidate <new run>/config.yaml
      --allow-loss-key lambda_skel_recall
      --allow-loss-key skel_recall_band
      --allow-save-dir

Exit code is 0 only when every candidate passes, 1 when at least one does not.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import yaml

#: Sentinel for "the key is absent from this config".
MISSING = object()

MODEL_PREFIX = "model"

LOSS_PREFIX = "loss"

DATA_KEYS: tuple[str, ...] = (
    "data.data_root",
    "data.threshold",
    "data.target_dir_name",
)

TRAIN_KEYS: tuple[str, ...] = (
    "train.batch_size",
    "train.lr",
    "train.weight_decay",
    "train.amp",
    "train.amp_dtype",
)

RESOLVED_FIELDS: tuple[str, ...] = (
    "train_samples",
    "selected_regime",
    "selected_img_size",
    "effective_batch_size",
    "target_updates",
    "resolved_epochs",
    "scnp_warmup_updates",
    "bpcm_start_update",
    "bpcm_ramp_updates",
    "resolved_validation_interval_updates",
    "min_early_stop_update",
    "patience_updates",
    "training_seed",
)

RESOLVED_KEYS: tuple[str, ...] = tuple(f"auto_protocol_resolved.{name}" for name in RESOLVED_FIELDS)

SEED_KEY = "auto_protocol_resolved.training_seed"

SAVE_DIR_KEY = "save.save_dir"


def flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested mapping into ``{"a.b.c": value}`` leaves.

    Empty mappings are kept as leaves so that "the whole section is empty here
    but populated there" still shows up as a difference instead of vanishing.
    """
    flat: dict[str, Any] = {}
    if isinstance(node, dict):
        if not node:
            flat[prefix] = {}
            return flat
        for key, value in node.items():
            flat.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
        return flat
    flat[prefix] = node
    return flat


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("top level of the document is not a YAML mapping")
    return config


def is_model_key(key: str) -> bool:
    return key == MODEL_PREFIX or key.startswith(MODEL_PREFIX + ".")


def is_loss_key(key: str) -> bool:
    return key == LOSS_PREFIX or key.startswith(LOSS_PREFIX + ".")


def normalise_loss_keys(raw_keys: Iterable[str]) -> set[str]:
    """Accept ``scnp_weight`` or ``loss.scnp_weight``; reject everything else."""
    allowed: set[str] = set()
    for raw in raw_keys:
        key = raw.strip()
        if not key:
            raise ValueError("--allow-loss-key got an empty key")
        if not key.startswith("loss."):
            key = f"loss.{key}"
        if not key.startswith("loss.") or key == "loss.":
            raise ValueError(f"--allow-loss-key expects a loss.* key, got {raw!r}")
        allowed.add(key)
    return allowed


def section_keys(
    reference_flat: dict[str, Any],
    candidate_flat: dict[str, Any],
    predicate: Callable[[str], bool],
) -> list[str]:
    """Sorted union of the keys of one whole section, from both configs."""
    return sorted(
        {key for key in reference_flat if predicate(key)}
        | {key for key in candidate_flat if predicate(key)}
    )


def build_key_list(reference_flat: dict[str, Any], candidate_flat: dict[str, Any]) -> list[str]:
    """Every compared key, in a stable order (whole sections first)."""
    model_keys = section_keys(reference_flat, candidate_flat, is_model_key)
    loss_keys = section_keys(reference_flat, candidate_flat, is_loss_key)
    return (
        model_keys
        + list(DATA_KEYS)
        + list(TRAIN_KEYS)
        + list(RESOLVED_KEYS)
        + loss_keys
        + [SAVE_DIR_KEY]
    )


def format_value(value: Any) -> str:
    if value is MISSING:
        return "<missing>"
    return repr(value)


def is_neutral(value: Any) -> bool:
    """True when an explicit value is behaviourally identical to omitting the key.

    Loss weights default to 0 and model switches default to False, so an archived
    config that spells ``lambda_cfps: 0.0`` out describes the same objective as a
    config that simply does not mention it.
    """
    if value is None or value is False:
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return float(value) == 0.0
    if isinstance(value, str):
        return not value.strip()
    return False


def in_whole_section(key: str) -> bool:
    """Keys of the sections that are compared in full (model / loss)."""
    return is_model_key(key) or is_loss_key(key)


def compare(
    reference_flat: dict[str, Any],
    candidate_flat: dict[str, Any],
    allowed_keys: set[str],
    ignore_seed: bool,
) -> tuple[list[tuple[str, Any, Any]], list[tuple[str, Any, Any]]]:
    """Return ``(mismatches, notes)`` as ``(key, reference_value, candidate_value)``.

    ``mismatches`` decide PASS/FAIL; ``notes`` are one-sided keys of the two whole
    sections whose present value is neutral, i.e. no behavioural difference.
    """
    mismatches: list[tuple[str, Any, Any]] = []
    notes: list[tuple[str, Any, Any]] = []
    for key in build_key_list(reference_flat, candidate_flat):
        if ignore_seed and key == SEED_KEY:
            continue
        if key in allowed_keys:
            continue
        reference_value = reference_flat.get(key, MISSING)
        candidate_value = candidate_flat.get(key, MISSING)
        if reference_value is MISSING or candidate_value is MISSING:
            present = candidate_value if reference_value is MISSING else reference_value
            if in_whole_section(key) and is_neutral(present):
                notes.append((key, reference_value, candidate_value))
            else:
                mismatches.append((key, reference_value, candidate_value))
        elif reference_value != candidate_value:
            mismatches.append((key, reference_value, candidate_value))
    return mismatches, notes


def describe(
    candidate: Path,
    mismatches: Sequence[tuple[str, Any, Any]],
    notes: Sequence[tuple[str, Any, Any]],
) -> list[str]:
    verdict = "PASS" if not mismatches else "FAIL"
    lines = [f"{verdict} {candidate}"]
    for key, reference_value, candidate_value in mismatches:
        lines.append(
            f"    {key}: {format_value(reference_value)} -> {format_value(candidate_value)}"
        )
    for key, reference_value, candidate_value in notes:
        lines.append(
            f"    note {key}: {format_value(reference_value)} -> {format_value(candidate_value)}"
            " (neutral value, identical behaviour)"
        )
    return lines


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that candidate run config(s) resolve to the same training protocol "
            "as a reference run config."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--reference",
        required=True,
        type=Path,
        help="config.yaml of the reference (e.g. the archived A cell) run",
    )
    parser.add_argument(
        "--candidate",
        required=True,
        action="append",
        default=[],
        type=Path,
        metavar="PATH",
        help="config.yaml of a candidate run; repeat the flag for several runs",
    )
    parser.add_argument(
        "--allow-loss-key",
        action="append",
        default=[],
        dest="allow_loss_keys",
        metavar="KEY",
        help=(
            "loss.* key that is allowed to differ; accepts 'lambda_skel_recall' or "
            "'loss.lambda_skel_recall'. Repeat for several keys."
        ),
    )
    parser.add_argument(
        "--allow-save-dir",
        action="store_true",
        help="allow save.save_dir to differ (every run writes to its own directory)",
    )
    parser.add_argument(
        "--ignore-seed",
        action="store_true",
        help="do not require auto_protocol_resolved.training_seed to match",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        allowed_keys = normalise_loss_keys(args.allow_loss_keys)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.allow_save_dir:
        allowed_keys.add(SAVE_DIR_KEY)

    try:
        reference_config = load_config(args.reference)
    except (OSError, yaml.YAMLError, ValueError) as error:
        print(f"error: cannot read reference config {args.reference}: {error}", file=sys.stderr)
        return 2
    reference_flat = flatten(reference_config)

    print(f"reference: {args.reference}")
    print(
        "exempted keys: "
        + (", ".join(sorted(allowed_keys)) if allowed_keys else "<none>")
        + (" | training_seed ignored" if args.ignore_seed else "")
    )

    failed = 0
    for candidate in args.candidate:
        try:
            candidate_config = load_config(candidate)
        except (OSError, yaml.YAMLError, ValueError) as error:
            failed += 1
            print(f"FAIL {candidate}")
            print(f"    <unreadable config>: {error}")
            continue
        mismatches, notes = compare(
            reference_flat,
            flatten(candidate_config),
            allowed_keys,
            args.ignore_seed,
        )
        if mismatches:
            failed += 1
        for line in describe(candidate, mismatches, notes):
            print(line)

    print()
    total = len(args.candidate)
    if failed:
        print(f"{failed} of {total} candidate(s) FAILED protocol identity")
        return 1
    print(f"{total} of {total} candidate(s) PASS protocol identity")
    return 0


if __name__ == "__main__":
    sys.exit(main())
