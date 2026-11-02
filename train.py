from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import gc
from contextlib import nullcontext
from datetime import datetime
import math
from pathlib import Path
from time import perf_counter

import torch
import yaml
from torch.utils.data import DataLoader

from datasets import build_dataloader, scan_train_component_counts
from losses import ConfidenceBalancedDistillationLoss, DEMambaLoss, distillation_weight_for_epoch
from metrics import SegmentationMetric
from models import build_model
from models.masked_reconstruction import is_encoder_state_key
from utils import (
    ModelEMA,
    AutoProtocolCheckpointState,
    UpdateEarlyStopping,
    accumulation_steps,
    auto_protocol_enabled,
    capture_rng_state,
    is_new_best_checkpoint,
    load_checkpoint,
    load_checkpoint_metadata,
    model_checksum,
    resolve_auto_protocol,
    restore_rng_state,
    save_checkpoint,
    select_auto_protocol_final_checkpoint,
    set_seed,
    setup_logger,
    update_resolved_run_state,
    validate_auto_protocol_resume,
)


TIMING_KEYS = ("data", "forward_loss", "backward_optimizer")
MAMBA_PATH_NAMES = ("horizontal", "vertical", "diagonal", "anti_diagonal")
PRETRAINING_OBJECTIVE = "masked_smooth_l1_rgb_reconstruction"
PRETRAINING_METADATA = {
    "seed": 42,
    "epochs": 100,
    "split": "train",
    "image_count": 368,
    "img_size": [384, 544],
    "mask_ratio": 0.60,
    "patch_size": 16,
    "objective": PRETRAINING_OBJECTIVE,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train Lite-TopoRoute-Mamba")
    parser.add_argument("--config", default="configs/demambanet.yaml")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--img_size", type=int, nargs=2, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--save_dir", default=None)
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--resume", default=None)
    checkpoint_group.add_argument("--init_encoder_checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-final-test",
        action="store_true",
        help="Train and validate normally, but do not construct or evaluate the final test loader.",
    )
    parser.add_argument(
        "--auto-protocol-dry-run",
        "--auto_protocol_dry_run",
        dest="auto_protocol_dry_run",
        action="store_true",
        help="For an enabled Auto Protocol config, resolve/probe then run exactly one optimizer update and exit.",
    )
    parser.add_argument(
        "--auto-protocol-unsafe-resume",
        action="store_true",
        help="Allow a non-reproducible Auto Protocol V2 resume when the train-only protocol identity changed.",
    )
    parser.add_argument(
        "--non-finite-diagnostic",
        action="store_true",
        help=(
            "Log the first non-finite training event and later event counts without changing "
            "loss, gradients, optimizer, scheduler, or EMA behavior."
        ),
    )
    parser.add_argument(
        "--teacher_checkpoint",
        default=None,
        help="Override distillation.teacher_checkpoint for an enabled CBSD run.",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    for section, key, value in (
        ("data", "data_root", args.data_root),
        ("train", "epochs", args.epochs),
        ("train", "batch_size", args.batch_size),
        ("train", "lr", args.lr),
        ("data", "img_size", args.img_size),
        ("train", "num_workers", args.num_workers),
        ("save", "save_dir", args.save_dir),
    ):
        if value is not None:
            cfg[section][key] = value
    if args.teacher_checkpoint is not None:
        if not isinstance(cfg.get("distillation"), dict):
            raise ValueError("--teacher_checkpoint requires a config containing a distillation section.")
        cfg["distillation"]["teacher_checkpoint"] = args.teacher_checkpoint
    return cfg


def validate_no_dpsm_topology_config(cfg: dict) -> None:
    """Reject invalid strict no-DPSM registrations before run setup.

    Historical ordinary-Stage4 configs retain their existing behavior.  This
    applies only to the explicit topology-matched deletion switch, where
    structural supervision cannot exist without DPSM route logits.
    """

    model_cfg = cfg.get("model", {})
    if not bool(model_cfg.get("use_topology_matched_no_dpsm_stage4", False)):
        return
    loss_cfg = cfg.get("loss", {})
    if float(loss_cfg.get("lambda_tangent", 0.0)) > 0.0:
        raise ValueError("NoDPSMTopoBlock has no route_s4; lambda_tangent must be 0 before training starts.")
    if float(loss_cfg.get("lambda_bpcm", 0.0)) > 0.0:
        raise ValueError("NoDPSMTopoBlock cannot be registered as StructSup; lambda_bpcm must be 0.")


def validate_dpsm_scan_route_config(cfg: dict) -> None:
    """Reject scan/route mechanism registrations outside the frozen C-table contract.

    The model constructor repeats structural validation for direct callers.  This
    training-level check additionally owns the C-table/R0 loss restrictions, so
    invalid non-default configurations fail before Auto Protocol probing or
    training.
    """

    model_cfg = cfg.get("model", {})
    scan_mode = model_cfg.get("dpsm_scan_mode", "four_direction")
    route_mode = model_cfg.get("dpsm_route_mode", "learned")
    valid_scan_modes = ("four_direction", "axial_repeated")
    valid_route_modes = ("learned", "uniform")
    if scan_mode not in valid_scan_modes:
        raise ValueError(f"Invalid dpsm_scan_mode {scan_mode!r}; expected one of {valid_scan_modes}.")
    if route_mode not in valid_route_modes:
        raise ValueError(f"Invalid dpsm_route_mode {route_mode!r}; expected one of {valid_route_modes}.")
    if scan_mode == "four_direction" and route_mode == "learned":
        return
    if not bool(model_cfg.get("use_direction_partitioned_shared_mamba", False)):
        raise ValueError(
            "Non-default DPSM80 scan/route modes require use_direction_partitioned_shared_mamba=true; "
            "they cannot be silently ignored by a DPSM-off or alternate Stage4 variant."
        )
    if bool(model_cfg.get("use_topology_matched_no_dpsm_stage4", False)) or bool(
        model_cfg.get("use_common_private_direction_partitioned_shared_mamba", False)
    ):
        raise ValueError("Non-default DPSM80 scan/route modes are valid only for original DPSM80, not NoDPSMTopoBlock/CP-DPSM64.")
    if str(model_cfg.get("stage4_routed_branch", "standard")) != "standard":
        raise ValueError("Non-default DPSM80 scan/route modes require stage4_routed_branch='standard'.")
    loss_cfg = cfg.get("loss", {})
    tangent_weight = float(loss_cfg.get("lambda_tangent", 0.0))
    bpcm_weight = float(loss_cfg.get("lambda_bpcm", 0.0))
    if bpcm_weight > 0.0:
        raise ValueError(
            "Non-default DPSM80 scan/route experiments require lambda_bpcm=0; "
            "this registered table does not combine altered scan/route mechanisms with BPCM."
        )
    if scan_mode == "axial_repeated" and tangent_weight > 0.0:
        raise ValueError(
            "axial_repeated DPSM80 cannot use lambda_tangent>0: the frozen tangent labels describe "
            "four_direction H/V/D/AD route logits, not repeated H/V group semantics."
        )
    # four_direction + uniform retains raw route_logits and is intentionally
    # valid with tangent supervision: only predicted route *weights* are kept
    # out of mask-feature modulation, not out of TangentRouteLoss.


def register_first_dpsm_execution_hook(model: torch.nn.Module, logger) -> object | None:
    """Log one existing raw-model DPSM forward without creating a probe.

    This must be called only after ``make_model_ema``.  ``ModelEMA`` deep-copies
    the model, so registering first would duplicate the hook into the EMA
    validation model and repeatedly log its forwards.
    """

    if not bool(getattr(model, "use_direction_partitioned_shared_mamba", False)):
        return None
    dpsm = model.stage4.route_mamba
    hook_state: dict[str, object] = {}

    def _log_first_dpsm_execution(module, _inputs, _outputs) -> None:
        logger.info(
            "DPSM80 first actual execution | packed: %s | shared core calls: %d | group sequences: %s | restored: %s",
            module.last_packed_shape,
            module.last_shared_mamba_call_count,
            module.last_group_sequence_shapes,
            module.last_group_restored_shapes,
        )
        handle = hook_state.pop("handle", None)
        if handle is not None:
            handle.remove()

    hook_state["handle"] = dpsm.register_forward_hook(_log_first_dpsm_execution)
    return hook_state["handle"]


def create_run_dir(base_dir: str | Path, resume_path: str | None) -> Path:
    if resume_path:
        return Path(resume_path).resolve().parent
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base_dir) / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = Path(base_dir) / f"{timestamp}-{suffix:02d}"
        suffix += 1
    return run_dir


def save_resolved_config(run_dir: str | Path, cfg: dict) -> tuple[Path, bool]:
    """Persist the merged configuration once without replacing a resume snapshot."""
    config_path = Path(run_dir) / "config.yaml"
    if config_path.exists():
        return config_path, False
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)
    return config_path, True


def refresh_resolved_config(config_path: str | Path, cfg: dict) -> None:
    """Rewrite the mutable auto-protocol run state without touching legacy runs."""

    with Path(config_path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)


def _validate_pretraining_metadata(metadata: object, expected_backend: dict[str, list[str]]) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("Encoder initialization checkpoint is missing metadata.")
    missing = set(PRETRAINING_METADATA) - set(metadata)
    unexpected = set(metadata) - (set(PRETRAINING_METADATA) | {"mamba_backend"})
    if missing or unexpected:
        raise ValueError(
            "Encoder initialization checkpoint metadata mismatch: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}."
        )
    for name, expected in PRETRAINING_METADATA.items():
        if metadata[name] != expected:
            raise ValueError(
                "Encoder initialization checkpoint metadata mismatch for "
                f"'{name}': expected {expected!r}, got {metadata[name]!r}."
            )
    if metadata.get("mamba_backend") != expected_backend:
        raise ValueError(
            "Encoder initialization checkpoint metadata mismatch for 'mamba_backend': "
            f"expected {expected_backend!r}, got {metadata.get('mamba_backend')!r}."
        )


def load_encoder_initialization(
    path: str | Path,
    model: torch.nn.Module,
    *,
    map_location: str | torch.device = "cpu",
) -> int:
    """Strictly load only a train-only masked-pretraining C40 encoder state."""

    checkpoint_path = Path(path)
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"encoder", "metadata"}:
        raise ValueError("Encoder initialization checkpoint must contain exactly 'encoder' and 'metadata'.")
    backend_info = model.mamba_backend_info()
    _validate_pretraining_metadata(checkpoint["metadata"], backend_info)
    incoming = checkpoint["encoder"]
    if not isinstance(incoming, dict):
        raise ValueError("Encoder initialization checkpoint field 'encoder' must be a state dictionary.")
    expected = {name: tensor for name, tensor in model.state_dict().items() if is_encoder_state_key(name)}
    unknown = set(incoming) - set(expected)
    missing = set(expected) - set(incoming)
    if unknown:
        raise ValueError(f"Encoder initialization checkpoint contains unknown encoder key(s): {sorted(unknown)}.")
    if missing:
        raise ValueError(f"Encoder initialization checkpoint is missing encoder key(s): {sorted(missing)}.")
    for name, tensor in incoming.items():
        if not torch.is_tensor(tensor):
            raise ValueError(f"Encoder initialization checkpoint key '{name}' is not a tensor.")
        if tensor.shape != expected[name].shape:
            raise ValueError(
                f"Encoder initialization checkpoint shape mismatch for '{name}': "
                f"expected {tuple(expected[name].shape)}, got {tuple(tensor.shape)}."
            )
    incompatible = model.load_state_dict(incoming, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected keys while loading encoder initialization: {incompatible.unexpected_keys}")
    return len(incoming)


def summarize_mamba_backends(backend_info: dict[str, list[str]]) -> tuple[str | None, list[str]]:
    """Summarize actual routed-stage backends and report fallback path locations."""
    if not backend_info:
        return None, []
    stage_summaries: list[str] = []
    fallback_locations: list[str] = []
    for stage, backends in backend_info.items():
        counts = Counter(backends)
        stage_summaries.append(f"{stage}: " + ", ".join(f"{backend} ×{count}" for backend, count in counts.items()))
        fallback_paths = [
            MAMBA_PATH_NAMES[index] if index < len(MAMBA_PATH_NAMES) else f"path{index}"
            for index, backend in enumerate(backends)
            if backend == "fallback"
        ]
        if fallback_paths:
            fallback_locations.append(f"{stage}: {', '.join(fallback_paths)}")
    return "Mamba backend | " + "; ".join(stage_summaries), fallback_locations


def format_mamba_fallback_warning(fallback_locations: list[str]) -> str:
    return "Mamba fallback detected | " + "; ".join(fallback_locations)


def _require_real_mamba_backends(model: torch.nn.Module, *, role: str) -> dict[str, list[str]]:
    """Return Mamba backend metadata or reject fallback/disabled teacher paths."""

    backend_info = model.mamba_backend_info() if hasattr(model, "mamba_backend_info") else {}
    if not backend_info:
        raise RuntimeError(f"{role} must report Mamba backends; got no backend metadata.")
    invalid = [
        f"{stage}.{MAMBA_PATH_NAMES[index] if index < len(MAMBA_PATH_NAMES) else f'path{index}'}: {backend}"
        for stage, backends in backend_info.items()
        for index, backend in enumerate(backends)
        if backend != "mamba_ssm"
    ]
    if invalid:
        raise RuntimeError(f"{role} Mamba backend validation failed: " + "; ".join(invalid))
    return backend_info


def _load_frozen_teacher_checkpoint(
    checkpoint_path: str | Path,
    teacher: torch.nn.Module,
    *,
    map_location: str | torch.device,
) -> None:
    """Strictly load the EMA ``model`` state from a formal teacher checkpoint."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"CBSD teacher checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError("CBSD teacher checkpoint must be a checkpoint dictionary.")
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise ValueError("CBSD teacher checkpoint must contain the EMA/model state under 'model'.")
    teacher.load_state_dict(state, strict=True)


def _validate_cbsd_teacher_variant(
    teacher_model_cfg: dict,
    teacher: torch.nn.Module,
) -> str:
    """Accept only the formal C40 teacher or its isolated DPER-RSDM variant."""

    if int(teacher_model_cfg.get("base_channels", 0)) != 40:
        raise ValueError("CBSD teacher must use the formal C40 width (base_channels=40).")
    if bool(teacher_model_cfg.get("use_rsdm_stage1", False)):
        required = {
            "use_detail_preserving_early_rebase": True,
            "use_low_resolution_subpixel_separable_concat_fusion": True,
        }
        for option, expected in required.items():
            if bool(teacher_model_cfg.get(option, False)) is not expected:
                raise ValueError(f"DPER-RSDM CBSD teacher requires {option}={expected}.")
        forbidden = (
            "use_direction_partitioned_shared_mamba",
            "use_narrow_transport_decoder",
            "use_partial_narrow_transport_decoder",
            "use_multiscale_directional_reallocation",
            "use_depthwise_separable_mask_head",
        )
        enabled = [option for option in forbidden if bool(teacher_model_cfg.get(option, False))]
        if enabled:
            raise ValueError(
                "DPER-RSDM CBSD teacher must be the complete teacher, not another ablation: "
                + ", ".join(enabled)
                + "."
            )
        if getattr(teacher, "rsdm_stage1", None) is None:
            raise ValueError("DPER-RSDM CBSD teacher must contain exactly one RSDM block.")
        rsdm_count = sum(
            module.__class__.__name__ == "RIFTStructuralDirectionalModelingBlock"
            for module in teacher.modules()
        )
        if rsdm_count != 1:
            raise ValueError(f"DPER-RSDM CBSD teacher must contain exactly one RSDM block; got {rsdm_count}.")
        stage4 = getattr(teacher, "stage4", None)
        route_mamba = getattr(stage4, "route_mamba", None)
        if route_mamba is None or route_mamba.__class__.__name__ != "TangentRoutedMambaBlock":
            raise ValueError("DPER-RSDM CBSD teacher must retain four independent full-width Stage4 Mamba paths.")
        if (
            int(getattr(route_mamba, "active_mamba_path_count", 0)) != 4
            or int(getattr(route_mamba, "mamba_channels", 0)) != 320
            or int(getattr(route_mamba, "spatial_reduction", 0)) != 1
        ):
            raise ValueError(
                "DPER-RSDM CBSD teacher must use four full-resolution, full-width Stage4 Mamba paths."
            )
        global_branch = getattr(stage4, "global_branch", None)
        if global_branch is None or global_branch.__class__.__name__ != "TransformerGlobalBlock":
            raise ValueError("DPER-RSDM CBSD teacher must retain the Stage4 Transformer.")
        return "DPER-RSDM"

    disallowed = (
        "use_detail_preserving_early_rebase",
        "use_direction_partitioned_shared_mamba",
        "use_narrow_transport_decoder",
        "use_partial_narrow_transport_decoder",
    )
    enabled = [option for option in disallowed if bool(teacher_model_cfg.get(option, False))]
    if enabled:
        raise ValueError(
            "Non-RSDM CBSD teacher must be the formal C40 model; unsupported option(s): "
            + ", ".join(enabled)
            + "."
        )
    return "C40"


def build_distillation_teacher(
    distillation_cfg: dict | None,
    *,
    device: torch.device,
    logger=None,
) -> torch.nn.Module | None:
    """Build, validate, strictly load, and freeze an enabled CBSD teacher."""

    if not isinstance(distillation_cfg, dict) or not bool(distillation_cfg.get("enabled", False)):
        return None
    teacher_config_path = distillation_cfg.get("teacher_config")
    teacher_checkpoint_path = distillation_cfg.get("teacher_checkpoint")
    if not teacher_config_path:
        raise ValueError("Enabled CBSD requires distillation.teacher_config.")
    if not teacher_checkpoint_path:
        raise ValueError("Enabled CBSD requires distillation.teacher_checkpoint or --teacher_checkpoint.")
    config_path = Path(teacher_config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"CBSD teacher config does not exist: {config_path}")
    teacher_cfg = load_config(config_path)
    if not isinstance(teacher_cfg.get("model"), dict):
        raise ValueError("CBSD teacher config must contain a model mapping.")
    teacher_model_cfg = dict(teacher_cfg["model"])
    teacher_name = str(teacher_model_cfg.pop("name", "DEMambaNet"))
    if teacher_name != "DEMambaNet":
        raise ValueError("CBSD teacher must be a formal C40-width DEMambaNet configuration.")
    teacher = build_model(teacher_name, **teacher_model_cfg).to(device)
    teacher_variant = _validate_cbsd_teacher_variant(teacher_model_cfg, teacher)
    backend_info = _require_real_mamba_backends(teacher, role="CBSD teacher")
    _load_frozen_teacher_checkpoint(teacher_checkpoint_path, teacher, map_location=device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    if logger is not None:
        backend_summary, _ = summarize_mamba_backends(backend_info)
        logger.info("CBSD teacher config: %s", config_path)
        logger.info("CBSD teacher checkpoint: %s", teacher_checkpoint_path)
        logger.info("CBSD teacher checkpoint | strict load: success (EMA/model weights)")
        logger.info("CBSD teacher variant | %s", teacher_variant)
        logger.info("CBSD teacher model: %s | parameters: %d", teacher_name, sum(p.numel() for p in teacher.parameters()))
        logger.info("CBSD teacher %s", backend_summary)
        if teacher_variant == "DPER-RSDM":
            logger.info("CBSD teacher Mamba backend | stage4: mamba_ssm ×4")
            logger.info("CBSD teacher RSDM | enabled; exactly 1 block")
            logger.info("CBSD teacher mode | eval; all parameters frozen")
    return teacher


def format_duration(seconds: float) -> str:
    total = max(0, round(seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def build_loader(cfg: dict, split: str, shuffle: bool, generator: torch.Generator | None = None) -> DataLoader:
    train_cfg, data_cfg = cfg["train"], cfg["data"]
    return build_dataloader(
        data_cfg["data_root"],
        split,
        data_cfg.get("img_size"),
        int(data_cfg["threshold"]),
        int(train_cfg["batch_size"]),
        int(train_cfg["num_workers"]),
        shuffle=shuffle,
        target_dir_name=data_cfg.get("target_dir_name", "targets"),
        persistent_workers=bool(train_cfg.get("persistent_workers", False)),
        prefetch_factor=int(train_cfg.get("prefetch_factor", 2)),
        photometric_augment=data_cfg.get("photometric_augment"),
        allow_quarter_turns=bool(data_cfg.get("allow_quarter_turns", True)),
        scale_jitter=data_cfg.get("scale_jitter"),
        return_sdf_target=split == "train" and float(cfg["loss"].get("lambda_sdf", 0.0)) > 0.0,
        sdf_truncation=float(cfg["loss"].get("sdf_truncation", 8.0)),
        return_component_id_map=split == "train" and float(cfg["loss"].get("lambda_component_query", 0.0)) > 0.0,
        generator=generator,
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast_context(device: torch.device, enabled: bool, dtype: torch.dtype):
    if enabled:
        return torch.autocast(device_type=device.type, dtype=dtype)
    return nullcontext()


def _is_cuda_oom(error: RuntimeError) -> bool:
    return "out of memory" in str(error).lower() and "cuda" in str(error).lower()


def resolve_auto_micro_batch(cfg: dict, device: torch.device, logger) -> tuple[int, int, int, int]:
    """Probe only CUDA OOMs, then discard all temporary training state.

    The formal model/optimizer are constructed *after* this function returns.
    ``main`` reseeds before that construction so the probe cannot change formal
    parameter initialization or the train-data augmentation random stream.
    """

    protocol = cfg.get("auto_protocol")
    resolved = cfg.get("auto_protocol_resolved")
    if not isinstance(protocol, dict) or not isinstance(resolved, dict):
        raise ValueError("Auto micro-batch resolution requires an enabled resolved auto_protocol.")
    batch_cfg = protocol.get("batch")
    if not isinstance(batch_cfg, dict):
        raise ValueError("auto_protocol.batch must be a mapping.")
    effective_batch = int(resolved["effective_batch_size"])
    requested = protocol.get("micro_batch_request", batch_cfg.get("micro_batch", "auto"))
    candidates = [int(value) for value in batch_cfg.get("candidates", [8, 4, 2, 1])]
    if isinstance(requested, int):
        candidates = [requested]
    elif requested != "auto":
        raise ValueError("auto_protocol.batch.micro_batch must be an integer or 'auto'.")
    candidates = [candidate for candidate in candidates if candidate > 0 and candidate <= effective_batch and effective_batch % candidate == 0]
    if not candidates:
        raise ValueError("No auto_protocol.batch candidate divides effective_batch_size.")

    if device.type != "cuda":
        selected = candidates[-1] if requested == "auto" else candidates[0]
        accumulation = accumulation_steps(effective_batch, selected)
        logger.warning(
            "Auto Protocol | CUDA unavailable; using micro batch %d and accumulation %d without a memory probe.",
            selected,
            accumulation,
        )
        cfg["train"]["batch_size"] = selected
        update_resolved_run_state(cfg, micro_batch=selected, accumulation=accumulation)
        return selected, accumulation, 0, 0

    fraction = float(batch_cfg.get("max_memory_fraction", 0.85))
    if not 0.0 < fraction <= 1.0:
        raise ValueError("auto_protocol.batch.max_memory_fraction must be in (0, 1].")
    capacity = int(torch.cuda.get_device_properties(device).total_memory * fraction)
    amp_name = str(cfg["train"].get("amp_dtype", "fp16")).lower()
    amp_dtype = torch.bfloat16 if amp_name == "bf16" else torch.float16
    amp_enabled = bool(cfg["train"].get("amp", True))
    last_peaks = (0, 0)
    rng_state = capture_rng_state()

    try:
        for candidate in candidates:
            temporary_cfg = deepcopy(cfg)
            temporary_cfg["train"]["batch_size"] = candidate
            temporary_cfg["train"]["num_workers"] = 0
            temporary_cfg["train"]["persistent_workers"] = False
            temporary_model = None
            temporary_optimizer = None
            temporary_criterion = None
            probe_loader = None
            probe_generator = torch.Generator(device="cpu").manual_seed(
                int(protocol.get("probe_seed", 104729)) + candidate
            )
            batch = outputs = loss = None
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                # Keep *all* probe work that can consume Torch RNG inside the
                # fork, including temporary module construction. Python/NumPy
                # state is restored by the outer finally block as well.
                cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
                with torch.random.fork_rng(devices=cuda_devices):
                    temporary_model_cfg = dict(temporary_cfg["model"])
                    temporary_model_name = str(temporary_model_cfg.pop("name", "DEMambaNet"))
                    temporary_model = build_model(temporary_model_name, **temporary_model_cfg).to(device)
                    temporary_optimizer = torch.optim.AdamW(
                        temporary_model.parameters(),
                        lr=float(temporary_cfg["train"].get("lr", 1e-4)),
                        weight_decay=float(temporary_cfg["train"].get("weight_decay", 0.0)),
                    )
                    temporary_criterion = DEMambaLoss(**temporary_cfg["loss"]).to(device)
                    probe_loader = build_loader(temporary_cfg, "train", True, generator=probe_generator)
                    batch = move_batch_to_device(next(iter(probe_loader)), device)
                    temporary_optimizer.zero_grad(set_to_none=True)
                    with autocast_context(device, amp_enabled, amp_dtype):
                        outputs = temporary_model(batch["image"])
                        loss, _ = temporary_criterion(outputs, batch, epoch=0)
                    loss.backward()
                synchronize(device)
                allocated = int(torch.cuda.max_memory_allocated(device))
                reserved = int(torch.cuda.max_memory_reserved(device))
                last_peaks = (allocated, reserved)
                if allocated <= capacity:
                    accumulation = accumulation_steps(effective_batch, candidate)
                    cfg["train"]["batch_size"] = candidate
                    update_resolved_run_state(
                        cfg,
                        micro_batch=candidate,
                        accumulation=accumulation,
                        peak_allocated_bytes=allocated,
                        peak_reserved_bytes=reserved,
                    )
                    logger.info(
                        "Auto Protocol batch probe | micro batch: %d | accumulation: %d | effective batch: %d | peak allocated: %.2f GiB | peak reserved: %.2f GiB",
                        candidate,
                        accumulation,
                        effective_batch,
                        allocated / 2**30,
                        reserved / 2**30,
                    )
                    return candidate, accumulation, allocated, reserved
                logger.info(
                    "Auto Protocol batch probe | micro batch: %d exceeds %.0f%% memory limit (allocated %.2f GiB / limit %.2f GiB).",
                    candidate,
                    fraction * 100,
                    allocated / 2**30,
                    capacity / 2**30,
                )
            except RuntimeError as error:
                if not _is_cuda_oom(error):
                    raise
                logger.info("Auto Protocol batch probe | micro batch: %d | CUDA OOM", candidate)
            finally:
                del batch, outputs, loss, probe_loader, probe_generator, temporary_criterion, temporary_optimizer, temporary_model
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        restore_rng_state(rng_state)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    allocated, reserved = last_peaks
    raise RuntimeError(
        "Auto Protocol could not fit any micro-batch candidate within the configured CUDA memory limit; "
        f"last peak allocated/reserved={allocated}/{reserved} bytes."
    )


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    metric = SegmentationMetric()
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        metric.update(model(batch["image"])["mask"], batch["mask"])
    return metric.compute()


def run_final_test(
    *,
    skip_final_test: bool,
    checkpoint_model: torch.nn.Module,
    cfg: dict,
    device: torch.device,
    auto_enabled: bool,
    config_path: Path | None,
    logger,
) -> dict[str, float] | None:
    """Run the final test only when explicitly permitted by the caller.

    The skip path intentionally returns before ``build_loader`` is referenced,
    so no test split dataset, sampler, or image is constructed/read.
    """

    if skip_final_test:
        if auto_enabled:
            cfg["auto_protocol_resolved"]["test_calls"] = 0
            if config_path is None:
                raise RuntimeError("Auto Protocol final-test skip requires a resolved config path.")
            refresh_resolved_config(config_path, cfg)
        logger.info("Final test skipped by request")
        return None
    return validate(checkpoint_model, build_loader(cfg, "test", False), device)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: DEMambaLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    epoch: int,
    model_ema: ModelEMA | None = None,
    ema_source_model: torch.nn.Module | None = None,
    teacher: torch.nn.Module | None = None,
    distillation_criterion: ConfidenceBalancedDistillationLoss | None = None,
    distillation_cfg: dict | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    accumulation: int = 1,
    global_update: int = 0,
    target_updates: int | None = None,
    use_update_clock: bool = False,
    next_validation_update: int | None = None,
    validation_interval_updates: int | None = None,
    non_finite_diagnostic: bool = False,
    logger=None,
) -> tuple[float, dict[str, float]]:
    model.train()
    if (teacher is None) != (distillation_criterion is None):
        raise ValueError("CBSD teacher and distillation criterion must be both set or both unset.")
    if accumulation <= 0:
        raise ValueError("Gradient accumulation must be positive.")
    distillation_enabled = teacher is not None
    if distillation_enabled:
        assert distillation_criterion is not None
        settings = distillation_cfg or {}
        lambda_kd = distillation_weight_for_epoch(
            epoch + 1,
            lambda_max=float(settings.get("lambda_max", 0.25)),
            warmup_epochs=int(settings.get("warmup_epochs", 10)),
            ramp_end_epoch=int(settings.get("ramp_end_epoch", 40)),
        )
    else:
        lambda_kd = 0.0
    loss_total = 0.0
    original_loss_total = 0.0
    cbkd_raw_loss_total = 0.0
    cbkd_weighted_loss_total = 0.0
    cbkd_stat_totals = {
        "teacher_foreground_probability": 0.0,
        "teacher_background_probability": 0.0,
        "foreground_confidence": 0.0,
        "background_confidence": 0.0,
    }
    cfps_loss_total = 0.0
    cfps_weighted_loss_total = 0.0
    cfps_selected_pixels_total = 0.0
    cfps_active_image_ratio_total = 0.0
    aabfp_loss_total = 0.0
    aabfp_boundary_selected_pixels_total = 0.0
    aabfp_bridge_selected_pixels_total = 0.0
    timings = {"data": 0.0, "forward_loss": 0.0, "backward_optimizer": 0.0, "batches": 0.0}
    validation_crossed = False
    reached_target = False
    optimizer_updates = 0
    micro_batches_in_accumulation = 0
    non_finite_events = 0
    first_non_finite_pending_step = False
    optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    ready_at = perf_counter()
    for batch_index, batch in enumerate(loader):
        if target_updates is not None and global_update >= target_updates:
            reached_target = True
            break
        data_ready_at = perf_counter()
        timings["data"] += data_ready_at - ready_at
        batch = move_batch_to_device(batch, device)
        synchronize(device)
        forward_start = perf_counter()
        with autocast_context(device, amp_enabled, amp_dtype):
            outputs = model(batch["image"])
            original_loss, loss_items = criterion(
                outputs,
                batch,
                epoch=epoch,
                cfps_epoch=epoch + 1,
                global_update=global_update if use_update_clock else None,
            )
            if distillation_enabled:
                assert teacher is not None and distillation_criterion is not None
                with torch.no_grad():
                    teacher_outputs = teacher(batch["image"])
                raw_cbkd_loss, cbkd_stats = distillation_criterion(
                    outputs["mask"],
                    teacher_outputs["mask"],
                    batch["mask"],
                    return_stats=True,
                )
                del teacher_outputs
                weighted_cbkd_loss = raw_cbkd_loss * lambda_kd
                loss = original_loss + weighted_cbkd_loss
            else:
                raw_cbkd_loss = original_loss.detach() * 0.0
                weighted_cbkd_loss = raw_cbkd_loss
                cbkd_stats = {}
                loss = original_loss
        synchronize(device)
        timings["forward_loss"] += perf_counter() - forward_start
        if non_finite_diagnostic:
            output_finite = {
                key: (bool(torch.isfinite(value.detach()).all().item()) if isinstance(value, torch.Tensor) else None)
                for key, value in ((key, outputs.get(key)) for key in ("mask", "edge", "route_s4"))
            }
            loss_items_finite = {
                key: math.isfinite(float(value)) for key, value in loss_items.items() if isinstance(value, (int, float))
            }
            loss_is_finite = bool(torch.isfinite(loss.detach()).all().item())
            if not loss_is_finite or not all(value is not False for value in output_finite.values()) or not all(loss_items_finite.values()):
                non_finite_events += 1
                if non_finite_events == 1:
                    first_non_finite_pending_step = True
                    if logger is not None:
                        logger.error(
                            "Non-finite diagnostic FIRST | epoch=%d batch=%d global_update=%d "
                            "total_loss=%s original_loss=%s loss_items=%s loss_items_finite=%s "
                            "output_finite=%s scaler_scale_before=pending scaler_scale_after=pending "
                            "optimizer_stepped=pending lr=%.10g scnp_weight=%.10g bpcm_weight=%.10g",
                            epoch + 1,
                            batch_index,
                            global_update,
                            float(loss.detach().cpu()),
                            float(original_loss.detach().cpu()),
                            loss_items,
                            loss_items_finite,
                            output_finite,
                            optimizer.param_groups[0]["lr"],
                            criterion._scnp_scale(epoch, global_update if use_update_clock else None),
                            criterion._bpcm_scale(epoch, global_update if use_update_clock else None),
                        )
                elif logger is not None:
                    logger.warning(
                        "Non-finite diagnostic count=%d | epoch=%d batch=%d global_update=%d",
                        non_finite_events,
                        epoch + 1,
                        batch_index,
                        global_update,
                    )
        backward_start = perf_counter()
        scaler.scale(loss / accumulation).backward()
        micro_batches_in_accumulation += 1
        if micro_batches_in_accumulation == accumulation:
            scale_before_step = scaler.get_scale() if scaler.is_enabled() else None
            scaler.step(optimizer)
            scaler.update()
            optimizer_stepped = scale_before_step is None or scaler.get_scale() >= scale_before_step
            if first_non_finite_pending_step and logger is not None:
                logger.error(
                    "Non-finite diagnostic FIRST step-result | epoch=%d batch=%d global_update=%d "
                    "scaler_scale_before=%s scaler_scale_after=%s optimizer_stepped=%s",
                    epoch + 1,
                    batch_index,
                    global_update,
                    scale_before_step,
                    scaler.get_scale() if scaler.is_enabled() else None,
                    optimizer_stepped,
                )
                first_non_finite_pending_step = False
            if optimizer_stepped:
                if model_ema is not None:
                    model_ema.update(ema_source_model if ema_source_model is not None else model)
                if scheduler is not None:
                    scheduler.step()
                global_update += 1
                optimizer_updates += 1
            micro_batches_in_accumulation = 0
            optimizer.zero_grad(set_to_none=True)
            if (
                next_validation_update is not None
                and validation_interval_updates is not None
                and global_update >= next_validation_update
            ):
                validation_crossed = True
                while global_update >= next_validation_update:
                    next_validation_update += validation_interval_updates
            if target_updates is not None and global_update >= target_updates:
                reached_target = True
        synchronize(device)
        timings["backward_optimizer"] += perf_counter() - backward_start
        loss_total += float(loss.detach().cpu())
        original_loss_total += float(original_loss.detach().cpu())
        cbkd_raw_loss_total += float(raw_cbkd_loss.detach().cpu())
        cbkd_weighted_loss_total += float(weighted_cbkd_loss.detach().cpu())
        for key in cbkd_stat_totals:
            if key in cbkd_stats:
                cbkd_stat_totals[key] += float(cbkd_stats[key].detach().cpu())
        cfps_loss_total += loss_items.get("cfps_loss", 0.0)
        cfps_weighted_loss_total += loss_items.get("cfps_weighted_loss", 0.0)
        cfps_selected_pixels_total += loss_items.get("cfps_selected_pixels_per_image", 0.0)
        cfps_active_image_ratio_total += loss_items.get("cfps_active_image_ratio", 0.0)
        aabfp_loss_total += loss_items.get("aabfp_loss", 0.0)
        aabfp_boundary_selected_pixels_total += loss_items.get("aabfp_boundary_selected_pixels_per_image", 0.0)
        aabfp_bridge_selected_pixels_total += loss_items.get("aabfp_bridge_selected_pixels_per_image", 0.0)
        timings["batches"] += 1
        ready_at = perf_counter()
        if reached_target:
            break
    # Never execute a partial accumulation at an epoch boundary: that would
    # change the requested effective batch and violate Auto Protocol V1.
    if micro_batches_in_accumulation:
        optimizer.zero_grad(set_to_none=True)
    if non_finite_diagnostic and first_non_finite_pending_step and logger is not None:
        logger.error(
            "Non-finite diagnostic FIRST step-result unavailable | epoch=%d global_update=%d "
            "because the epoch ended before a complete accumulation.",
            epoch + 1,
            global_update,
        )
    count = max(timings["batches"], 1.0)
    return loss_total / count, {key: value / count for key, value in timings.items() if key != "batches"} | {
        "batches": timings["batches"],
        "cfps_loss": cfps_loss_total / count,
        "cfps_weighted_loss": cfps_weighted_loss_total / count,
        "cfps_selected_pixels_per_image": cfps_selected_pixels_total / count,
        "cfps_active_image_ratio": cfps_active_image_ratio_total / count,
        "cfps_scale": criterion._cfps_scale(epoch + 1) if hasattr(criterion, "_cfps_scale") else 0.0,
        "aabfp_loss": aabfp_loss_total / count,
        "aabfp_boundary_selected_pixels_per_image": aabfp_boundary_selected_pixels_total / count,
        "aabfp_bridge_selected_pixels_per_image": aabfp_bridge_selected_pixels_total / count,
        "aabfp_scale": criterion._aabfp_scale(epoch) if hasattr(criterion, "_aabfp_scale") else 0.0,
        "original_loss": original_loss_total / count,
        "cbkd_raw_loss": cbkd_raw_loss_total / count,
        "cbkd_weighted_loss": cbkd_weighted_loss_total / count,
        "lambda_kd": lambda_kd,
        "global_update": global_update,
        "optimizer_updates": optimizer_updates,
        "reached_target": reached_target,
        "validation_crossed": validation_crossed,
        "next_validation_update": next_validation_update,
        **{key: value / count for key, value in cbkd_stat_totals.items()},
    }


def structural_score(metrics: dict[str, float], evaluation_cfg: dict) -> float:
    selected = str(evaluation_cfg.get("selection_metric", "structural_score"))
    if selected != "structural_score":
        if selected not in metrics:
            raise KeyError(f"Unknown selection metric: {selected}")
        return metrics[selected]
    weights = evaluation_cfg.get("structural_score_weights", {})
    return (
        float(weights.get("f1", 0.5)) * metrics["F1"]
        + float(weights.get("cldice", 0.3)) * metrics["clDice"]
        + float(weights.get("boundary_f1", 0.2)) * metrics["BoundaryF1"]
    )


def accumulate_epoch_timing(total_timing: dict[str, float], epoch_stats: dict[str, float]) -> float:
    """Convert one epoch's per-batch means back to totals before aggregation."""
    batches = epoch_stats["batches"]
    for key in TIMING_KEYS:
        total_timing[key] += epoch_stats[key] * batches
    return batches


def make_optimizer(model: torch.nn.Module, cfg: dict, device: torch.device, logger) -> torch.optim.Optimizer:
    weight_decay = float(cfg["weight_decay"])
    exclude_norm_bias = bool(cfg.get("exclude_norm_bias_from_weight_decay", False))
    escs_parameter_ids = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith("early_serpentine_connectivity_ssm.")
    }
    strict_fused_candidate = bool(
        getattr(model, "use_multiscale_directional_reallocation", False)
        or getattr(model, "use_depthwise_separable_mask_head", False)
        or getattr(model, "use_full_width_residual_mask_head", False)
        or getattr(model, "use_difference_aware_up1_fusion", False)
    )
    require_fused_adamw = bool(
        cfg.get("fused_adamw", False)
        and device.type == "cuda"
        and strict_fused_candidate
    )
    if exclude_norm_bias:
        decay_parameters: list[torch.nn.Parameter] = []
        no_decay_parameters: list[torch.nn.Parameter] = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.endswith(".bias") or parameter.ndim <= 1:
                no_decay_parameters.append(parameter)
            else:
                decay_parameters.append(parameter)
        trainable_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
        grouped_ids = [id(parameter) for parameter in decay_parameters + no_decay_parameters]
        if len(grouped_ids) != len(set(grouped_ids)) or set(grouped_ids) != trainable_ids:
            raise RuntimeError("AdamW weight-decay groups must cover each trainable parameter exactly once.")
        optimizer_parameters: object = [
            {"params": decay_parameters, "weight_decay": weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ]
        kwargs = {"lr": float(cfg["lr"])}
    else:
        optimizer_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        kwargs = {"lr": float(cfg["lr"]), "weight_decay": weight_decay}

    def log_optimizer(implementation: str) -> None:
        if exclude_norm_bias:
            decay_count = sum(parameter.numel() for parameter in decay_parameters)
            no_decay_count = sum(parameter.numel() for parameter in no_decay_parameters)
            logger.info(
                "Optimizer: %s | decay: %d tensors / %d params | no_decay: %d tensors / %d params",
                implementation,
                len(decay_parameters),
                decay_count,
                len(no_decay_parameters),
                no_decay_count,
            )
        else:
            logger.info("Optimizer: %s", implementation)

    if bool(cfg.get("fused_adamw", False)) and device.type == "cuda":
        try:
            # PyTorch's GradScaler cannot dispatch fused and non-fused AdamW
            # parameter groups through one optimizer: it forwards fused-only
            # grad_scale/found_inf arguments to the standard group.  The ESCS
            # shared Mamba needs the standard kernel in this environment, so
            # use standard AdamW for that *candidate* as one coherent optimizer.
            # All existing models have no ESCS parameters and retain fused
            # AdamW exactly as before; learning-rate and decay settings remain
            # those declared by the experiment config.
            if escs_parameter_ids:
                optimizer = torch.optim.AdamW(optimizer_parameters, **kwargs)
                logger.info(
                    "Optimizer: standard AdamW | ESCS mamba_ssm AMP compatibility (%d ESCS tensors / %d params); "
                    "all AdamW hyperparameters unchanged",
                    len(escs_parameter_ids),
                    sum(parameter.numel() for name, parameter in model.named_parameters()
                        if id(parameter) in escs_parameter_ids),
                )
                return optimizer
            optimizer = torch.optim.AdamW(optimizer_parameters, fused=True, **kwargs)
            log_optimizer("fused AdamW")
            if require_fused_adamw:
                logger.info("Optimizer | fused AdamW")
            return optimizer
        except (RuntimeError, TypeError) as error:
            if require_fused_adamw:
                raise RuntimeError(
                    "This isolated DPER candidate requires fused AdamW; optimizer construction failed and "
                    "automatic fallback is forbidden."
                ) from error
            logger.warning("Fused AdamW unavailable; using standard AdamW (%s)", error)
    optimizer = torch.optim.AdamW(optimizer_parameters, **kwargs)
    log_optimizer("standard AdamW")
    return optimizer


class PolyLR(torch.optim.lr_scheduler._LRScheduler):
    """Polynomial epoch scheduler with the same step timing as the training loop."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_epochs: int,
        power: float = 0.9,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        if total_epochs <= 0:
            raise ValueError("PolyLR total_epochs must be positive.")
        if power <= 0:
            raise ValueError("PolyLR power must be positive.")
        if eta_min < 0:
            raise ValueError("PolyLR eta_min must be non-negative.")
        if any(eta_min > group["lr"] for group in optimizer.param_groups):
            raise ValueError("PolyLR eta_min must not exceed an optimizer base learning rate.")
        self.total_epochs = int(total_epochs)
        self.power = float(power)
        self.eta_min = float(eta_min)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        progress = min(max(self.last_epoch, 0), self.total_epochs) / self.total_epochs
        factor = (1.0 - progress) ** self.power
        return [self.eta_min + (base_lr - self.eta_min) * factor for base_lr in self.base_lrs]


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    """Linear epoch warmup followed by cosine decay, including both endpoints."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_epochs: int,
        warmup_epochs: int,
        start_factor: float,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        if total_epochs < 2:
            raise ValueError("WarmupCosineLR total_epochs must be at least 2.")
        if not 1 <= warmup_epochs < total_epochs:
            raise ValueError("WarmupCosineLR warmup_epochs must be in [1, total_epochs).")
        if not 0.0 < start_factor <= 1.0:
            raise ValueError("WarmupCosineLR start_factor must be in (0, 1].")
        if eta_min < 0.0:
            raise ValueError("WarmupCosineLR eta_min must be non-negative.")
        if any(eta_min > group["lr"] for group in optimizer.param_groups):
            raise ValueError("WarmupCosineLR eta_min must not exceed an optimizer base learning rate.")
        self.total_epochs = int(total_epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.start_factor = float(start_factor)
        self.eta_min = float(eta_min)
        # Epochs 6..200 contain 195 learning-rate values, so there are 194
        # cosine intervals between its maximum and eta_min endpoints.
        self.cosine_intervals = self.total_epochs - self.warmup_epochs - 1
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        epoch_index = min(max(self.last_epoch, 0), self.total_epochs - 1)
        if epoch_index < self.warmup_epochs:
            if self.warmup_epochs == 1:
                factor = self.start_factor
            else:
                factor = self.start_factor + (1.0 - self.start_factor) * epoch_index / (self.warmup_epochs - 1)
            return [base_lr * factor for base_lr in self.base_lrs]

        cosine_step = min(epoch_index - self.warmup_epochs, self.cosine_intervals)
        if self.cosine_intervals == 0:
            cosine_factor = 1.0
        else:
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * cosine_step / self.cosine_intervals))
        return [self.eta_min + (base_lr - self.eta_min) * cosine_factor for base_lr in self.base_lrs]


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    logger,
    *,
    auto_target_updates: int | None = None,
) -> torch.optim.lr_scheduler._LRScheduler:
    train_cfg = cfg["train"] if isinstance(cfg.get("train"), dict) else cfg
    scheduler_cfg = cfg.get("scheduler") if isinstance(cfg.get("scheduler"), dict) else None
    epochs = int(train_cfg["epochs"])
    eta_min = float(train_cfg.get("min_lr", 0.0))
    if auto_target_updates is not None:
        if str(train_cfg.get("lr_scheduler", "cosine")).lower() != "cosine" or scheduler_cfg is not None:
            raise ValueError("Auto Protocol currently requires train.lr_scheduler='cosine'.")
        logger.info(
            "Scheduler: CosineAnnealingLR | T_max: %d optimizer updates | eta_min: %s",
            auto_target_updates,
            eta_min,
        )
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=auto_target_updates,
            eta_min=eta_min,
        )
    if scheduler_cfg is not None:
        scheduler_name = str(scheduler_cfg.get("name", "")).lower()
        if scheduler_name == "warmup_cosine":
            warmup_epochs = int(scheduler_cfg.get("warmup_epochs", 0))
            start_factor = float(scheduler_cfg.get("warmup_start_factor", 0.0))
            logger.info(
                "Scheduler: WarmupCosineLR | warmup_epochs: %d | start_factor: %.4f | total_epochs: %d | eta_min: %.6f",
                warmup_epochs,
                start_factor,
                epochs,
                eta_min,
            )
            return WarmupCosineLR(
                optimizer,
                total_epochs=epochs,
                warmup_epochs=warmup_epochs,
                start_factor=start_factor,
                eta_min=eta_min,
            )
        raise ValueError("Unsupported scheduler.name '%s'. Expected 'warmup_cosine'." % scheduler_name)

    scheduler_name = str(train_cfg.get("lr_scheduler", "cosine")).lower()
    if scheduler_name == "cosine":
        logger.info("Scheduler: CosineAnnealingLR | T_max: %d | eta_min: %s", epochs, eta_min)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
    if scheduler_name == "poly":
        power = float(train_cfg.get("poly_power", 0.9))
        logger.info("Scheduler: PolyLR | total_epochs: %d | power: %s | eta_min: %.6f", epochs, power, eta_min)
        return PolyLR(optimizer, total_epochs=epochs, power=power, eta_min=eta_min)
    raise ValueError(f"Unsupported lr_scheduler '{scheduler_name}'. Expected 'cosine' or 'poly'.")


def make_model_ema(model: torch.nn.Module, cfg: dict) -> ModelEMA | None:
    if not bool(cfg.get("use_model_ema", False)):
        return None
    return ModelEMA(model, decay=float(cfg.get("model_ema_decay", 0.999)))


def auto_protocol_checkpoint_payload(
    selector: AutoProtocolCheckpointState,
    early_stopping: UpdateEarlyStopping | None,
    global_update: int,
    formal_train_generator: torch.Generator | None = None,
    resume_identity: dict | None = None,
) -> dict:
    """State required for an interrupted auto-protocol run to resume exactly."""

    payload = selector.state_dict()
    payload["global_update"] = int(global_update)
    if formal_train_generator is not None:
        # This is the state *after* the completed epoch's sampler traversal.
        # Restoring it makes resumed shuffling match uninterrupted training.
        payload["formal_train_generator_state"] = formal_train_generator.get_state()
    if resume_identity is not None:
        payload["resume_identity"] = resume_identity
    if early_stopping is not None:
        payload["early_stopping"] = {
            "min_update": early_stopping.min_update,
            "patience_updates": early_stopping.patience_updates,
            "min_delta": early_stopping.min_delta,
            "best_significant_score": early_stopping.best_significant_score,
            "last_significant_update": early_stopping.last_significant_update,
        }
    return payload


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    validate_no_dpsm_topology_config(cfg)
    validate_dpsm_scan_route_config(cfg)
    auto_plan = resolve_auto_protocol(
        cfg,
        cli_img_size=args.img_size is not None,
        cli_batch_size=args.batch_size is not None,
        cli_epochs=args.epochs is not None,
        cli_data_root=args.data_root is not None,
        cli_save_dir=args.save_dir is not None,
    )
    auto_enabled = auto_plan is not None
    auto_protocol_version = int(cfg.get("auto_protocol", {}).get("version", 1)) if auto_enabled else None
    dry_run_requested = args.auto_protocol_dry_run or (
        auto_enabled and bool(cfg.get("auto_protocol", {}).get("dry_run", False))
    )
    if dry_run_requested and not auto_enabled:
        raise ValueError("--auto_protocol_dry_run requires auto_protocol.enabled=true.")
    if auto_enabled and isinstance(cfg.get("distillation"), dict) and bool(cfg["distillation"].get("enabled", False)):
        raise ValueError("Auto Protocol forbids distillation.")
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(cfg["train"].get("cudnn_benchmark", True))
    save_dir = create_run_dir(cfg["save"]["save_dir"], args.resume)
    logger = setup_logger(save_dir)
    logger.info("Using device: %s", device)
    logger.info("Run directory: %s", save_dir)
    if args.non_finite_diagnostic:
        logger.warning(
            "Non-finite diagnostic enabled | read-only checks; fail-fast is disabled and normal training behavior is unchanged."
        )
    if not auto_enabled:
        config_path, config_saved = save_resolved_config(save_dir, cfg)
        if config_saved:
            logger.info("Resolved config saved: %s", config_path)
        else:
            logger.info("Resolved config preserved: %s", config_path)

    use_sdf_head = bool(cfg["model"].get("use_sdf_head", False))
    lambda_sdf = float(cfg["loss"].get("lambda_sdf", 0.0))
    if use_sdf_head and lambda_sdf <= 0:
        raise ValueError("model.use_sdf_head=true requires loss.lambda_sdf > 0.")
    if lambda_sdf > 0 and not use_sdf_head:
        raise ValueError("loss.lambda_sdf > 0 requires model.use_sdf_head=true.")
    if bool(cfg["model"].get("use_common_private_direction_partitioned_shared_mamba", False)) or bool(
        cfg["model"].get("use_full_width_residual_mask_head", False)
    ) or bool(
        cfg["model"].get("use_difference_aware_up1_fusion", False)
    ):
        distillation_cfg = cfg.get("distillation")
        if isinstance(distillation_cfg, dict) and bool(distillation_cfg.get("enabled", False)):
            raise ValueError("This structural-only experiment forbids distillation.")
        forbidden_training_losses = (
            "lambda_cfps",
            "lambda_fbhn",
            "lambda_aabfp",
            "lambda_ep_hcfc",
            "lambda_sdf",
            "lambda_aux_mask",
            "lambda_aux_edge",
            "lambda_component_query",
        )
        enabled_losses = [
            name for name in forbidden_training_losses if float(cfg["loss"].get(name, 0.0)) != 0.0
        ]
        if enabled_losses:
            raise ValueError(
                "This structural-only experiment must retain the original DPSM80 loss protocol; disabled loss(es) were enabled: "
                + ", ".join(enabled_losses)
                + "."
            )

    if auto_enabled:
        assert auto_plan is not None
        micro_batch, accumulation, peak_allocated, peak_reserved = resolve_auto_micro_batch(cfg, device, logger)
        # The probe creates a temporary model and consumes data augmentation
        # randomness.  Re-seed before formal model/optimizer construction.
        set_seed(args.seed)
        update_resolved_run_state(cfg)
        cfg["auto_protocol_resolved"].update(
            rng_isolation_restored=True,
            training_seed=args.seed,
            formal_dataloader_seed=args.seed,
            dry_run=dry_run_requested,
        )
        logger.info("Auto Protocol RNG isolation | restored: true")
        logger.info("Training seed reapplied after probe: %d", args.seed)
        logger.info(
            "Auto Protocol V%s dataset audit | train samples: %d | native size distribution: %s | selected img_size: %s",
            auto_protocol_version,
            auto_plan.train_samples,
            auto_plan.native_size_distribution,
            list(auto_plan.selected_img_size),
        )
        logger.info(
            "Auto Protocol V%s statistical batch | micro batch: %d | accumulation: %d | effective batch: %d | steps/epoch: %d | target updates: %d | epochs: %d",
            auto_protocol_version,
            micro_batch,
            accumulation,
            auto_plan.effective_batch_size,
            auto_plan.optimizer_steps_per_epoch,
            auto_plan.target_updates,
            auto_plan.resolved_epochs,
        )
        logger.info(
            "Auto Protocol V%s loss phases | SCNP warmup: %d updates | BPCM start: %d updates | BPCM ramp: %d updates",
            auto_protocol_version,
            auto_plan.scnp_warmup_updates,
            auto_plan.bpcm_start_update,
            auto_plan.bpcm_ramp_updates,
        )
        if auto_protocol_version == 2:
            resolved = cfg["auto_protocol_resolved"]
            audit = resolved["foreground_audit"]
            logger.info(
                "Auto Protocol V2 selected regime | name: %s | condition: %s | overrides: %s",
                resolved["selected_regime"],
                resolved["selected_regime_condition"],
                resolved["overrides"] or "none",
            )
            if resolved["overrides"]:
                logger.info("Auto policy overridden | %s", ", ".join(resolved["overrides"]))
            logger.info(
                "Auto Protocol V2 foreground audit | fraction: %.8f | neg/pos: %.4f | mask_pos_weight: %.4f (%s) | empty: %d | fingerprint: %s",
                audit["foreground_fraction"],
                audit["negative_positive_ratio"],
                resolved["resolved_mask_pos_weight"],
                resolved["mask_pos_weight_source"],
                audit["empty_mask_count"],
                audit["dataset_fingerprint"],
            )
            logger.info(
                "Auto Protocol V2 exposure budget | planned exposures: %d | approximate dataset passes: %.3f | min early exposures: %d | patience exposures: %d",
                resolved["planned_sample_exposures"],
                resolved["approximate_dataset_passes"],
                resolved["minimum_early_stop_sample_exposures"],
                resolved["patience_sample_exposures"],
            )
            logger.info(
                "Auto Protocol V2 validation/early-stop plan | validation interval: %d | min early update: %d | patience updates: %d | min_delta: %.6f",
                resolved["resolved_validation_interval_updates"],
                resolved["min_early_stop_update"],
                resolved["patience_updates"],
                float(cfg["auto_protocol"]["early_stopping"].get("min_delta", 0.0005)),
            )
        logger.info(
            "Configured validation interval updates: %d | Resolved validation interval updates: %d | Approximate validation interval epochs: %.3f",
            cfg["auto_protocol_resolved"]["configured_validation_interval_updates"],
            cfg["auto_protocol_resolved"]["resolved_validation_interval_updates"],
            cfg["auto_protocol_resolved"]["approximate_validation_interval_epochs"],
        )
        if device.type == "cuda":
            logger.info(
                "Auto Protocol V%s memory micro-batch probe | peak allocated/reserved: %.2f/%.2f GiB",
                auto_protocol_version,
                peak_allocated / 2**30,
                peak_reserved / 2**30,
            )
        config_path, config_saved = save_resolved_config(save_dir, cfg)
        if config_saved:
            logger.info("Resolved auto-protocol config saved: %s", config_path)
        else:
            logger.info("Resolved auto-protocol config preserved: %s", config_path)

    model_cfg = dict(cfg["model"])
    model_name = str(model_cfg.pop("name", "DEMambaNet"))
    raw_model = build_model(model_name, **model_cfg).to(device)
    if auto_enabled:
        checksum = model_checksum(raw_model)
        cfg["auto_protocol_resolved"]["initial_model_checksum"] = checksum
        refresh_resolved_config(config_path, cfg)
        logger.info("Initial model checksum: %s", checksum)
    backend_info = raw_model.mamba_backend_info()
    if backend_info:
        invalid_backends = [
            f"{stage}.{MAMBA_PATH_NAMES[index] if index < len(MAMBA_PATH_NAMES) else f'path{index}'}: {backend}"
            for stage, backends in backend_info.items()
            for index, backend in enumerate(backends)
            if backend != "mamba_ssm"
            and not (backend == "disabled" and not hasattr(raw_model, stage))
        ]
        if invalid_backends:
            raise RuntimeError("Mamba backend validation failed: " + "; ".join(invalid_backends))
        backend_summary, _ = summarize_mamba_backends(backend_info)
        logger.info("%s", backend_summary)
    else:
        logger.info("Mamba backend: not applicable (Mamba-free model)")
    logger.info(
        "Framework internal state | LSSCF: %s | DPER: %s | RSDM: %s",
        "on" if bool(getattr(raw_model.decoder, "use_low_resolution_subpixel_separable_concat_fusion", False)) else "off",
        "on" if bool(getattr(raw_model, "use_detail_preserving_early_rebase", False)) else "off",
        "on" if bool(getattr(raw_model, "use_rsdm_stage1", False)) else "off",
    )
    if bool(getattr(raw_model, "use_topology_matched_no_dpsm_stage4", False)):
        logger.info("Stage4 variant | NoDPSMTopoBlock")
        logger.info("Stage4 local branch | DWConvBlock + StripConvBlock")
        logger.info("Stage4 global branch | TransformerGlobalBlock")
        logger.info("Stage4 fusion | local + transformer; 2 × 320 -> 320")
        logger.info("Stage4 route branch | absent")
    stage4_routed_branch = str(getattr(raw_model, "stage4_routed_branch", "standard"))
    if stage4_routed_branch == "single_raster_mamba80":
        logger.info("Stage4 Mamba reference | single raster core | channels 80 | route disabled")
        logger.info("Stage4 routed branch | x 320 -> 1x1 80 -> raster SequenceMambaCore -> 1x1 320 + x")
        logger.info("Stage4 directional scans / pixel-wise route weights / local compensation | absent")
    elif stage4_routed_branch == "four_independent_mamba80":
        logger.info("Stage4 Mamba reference | four independent directional Mamba cores x4 | channels 80")
        logger.info("Stage4 directions | horizontal, vertical, diagonal, anti_diagonal | local compensation absent")
    elif stage4_routed_branch == "standard" and bool(getattr(raw_model, "route_stage4", False)):
        stage4_route = getattr(raw_model.stage4, "route_mamba", None)
        if getattr(stage4_route, "mamba_channels", None) == 320 and not bool(getattr(raw_model, "use_direction_partitioned_shared_mamba", False)):
            logger.info("Stage4 Mamba reference | four independent full-width directional Mamba cores x4 | channels 320")
        logger.info("Stage4 route supervision | absent")
    if bool(getattr(raw_model, "use_haar_topology_preserving_rebase", False)):
        logger.info("Early rebase | DPER-HTPR: fixed Haar LL/LH/HL/HH + gated half-resolution HF injection")
        logger.info("SPCR | disabled")
    if bool(getattr(raw_model, "use_narrow_dual_resolution_stage1", False)):
        logger.info(
            "Early rebase | DPER-NDRS: narrow full-resolution DW3x3 detail residual + one-way detail-to-semantic exchange"
        )
        logger.info("SPCR | disabled; HTPR | disabled")
    if bool(getattr(raw_model, "use_full_resolution_joint_exchange", False)):
        logger.info(
            "Early rebase | DPER-FJSE: dense full-resolution 3x3 joint modeling + phase-preserving one-way exchange"
        )
        logger.info("NDRS | disabled; SPCR | disabled; HTPR | disabled")
    if bool(getattr(raw_model, "use_partial_full_resolution_stage1", False)):
        logger.info("Early full-resolution stage | DPER-PFRS: LocalCrackBlock 16ch")
        logger.info("High-to-low exchange | residual-only PixelUnshuffle ×2, 64→40")
    if bool(getattr(raw_model, "use_multiscale_directional_reallocation", False)):
        stage3_route = raw_model.stage3.route_mamba
        stage4_route = raw_model.stage4.route_mamba
        assert stage3_route.local_compensations is not None
        assert stage4_route.local_compensations is not None
        logger.info("Directional reallocation | DPER-MSDR: Stage3 ×2 + Stage4 ×2")
        logger.info(
            "Stage3 Mamba | %s | %s ×%d | channels: %d/160 | spatial reduction: %d",
            ", ".join(stage3_route.active_mamba_directions),
            stage3_route.backends[0],
            len(stage3_route.backends),
            stage3_route.mamba_channels,
            stage3_route.spatial_reduction,
        )
        logger.info(
            "Stage4 Mamba | %s | %s ×%d | channels: %d/320 | spatial reduction: %d",
            ", ".join(stage4_route.active_mamba_directions),
            stage4_route.backends[0],
            len(stage4_route.backends),
            stage4_route.mamba_channels,
            stage4_route.spatial_reduction,
        )
        logger.info("Stage3 Transformer | disabled")
        logger.info("Stage4 Transformer | enabled")
        logger.info("Total Mamba | mamba_ssm ×4 independent cores")
        logger.info(
            "Local directional compensation | Stage3: %s",
            ", ".join(stage3_route.local_compensations.keys()),
        )
        logger.info(
            "Local directional compensation | Stage4: %s",
            ", ".join(stage4_route.local_compensations.keys()),
        )
    if bool(getattr(raw_model, "use_depthwise_separable_mask_head", False)):
        logger.info(
            "Mask head | DPER-DSMH: depthwise 3×3 (40 groups) → "
            "pointwise 1×1 (40→40) → output 1×1 (40→1)"
        )
        logger.info("Decoder fusion | LSSCF unchanged")
    if bool(getattr(raw_model, "use_full_width_residual_mask_head", False)):
        logger.info("Model variant | DPSM80-FWRMH")
        logger.info("Mask head | full-width 40ch identity bypass + residual 40→16→DW3×3→40 correction")
        logger.info("Mask classifier | original-width 1×1 (40→1)")
        logger.info("DPSM80 Stage4 / Transformer / LSSCF decoder / edge and reliability heads | unchanged")
        logger.info("DSMH / distillation / CFPS | disabled")
    if bool(getattr(raw_model, "use_difference_aware_up1_fusion", False)):
        logger.info("Model variant | DPSM80-DAFU1")
        logger.info("Decoder up1 | full-width sum path + full-width difference correction")
        logger.info("Decoder up2/up3 | original LSSCF concat fusion retained")
        logger.info("Mask head | original dense 40→40 3×3 retained")
        logger.info("DPSM80 Stage4 / Transformer / RSDM / decoder refine / edge and reliability heads | unchanged")
        logger.info("FWRMH / DSMH / distillation / CFPS | disabled")
    if bool(getattr(raw_model, "use_rsdm_stage1", False)):
        rsdm_module = getattr(raw_model, "rsdm_stage1", None)
        if rsdm_module is None:
            raise RuntimeError("RSDM is enabled but rsdm_stage1 was not constructed.")
        logger.info("DPER-RSDM | half-resolution semantic Stage1 structural-directional modeling")
        logger.info("SPCR | disabled")
        logger.info("HTPR | disabled")
        logger.info("Full-resolution refinement | disabled")
        logger.info("Early Mamba | disabled")
        logger.info("Separable mask head | disabled")
        logger.info("RSDM blocks | 1")
        logger.info("RSDM kernel | %d", rsdm_module.kernel_size)
        logger.info("RSDM dilation | %d", rsdm_module.dilation)
        logger.info("RSDM hidden ratio | %.1f", rsdm_module.hidden_ratio)
    if bool(getattr(raw_model, "use_narrow_transport_decoder", False)):
        logger.info("Model variant | DPER-RSDM-NTD")
        logger.info("RSDM | enabled")
        logger.info("RSDM blocks | 1")
        logger.info("Narrow transport decoder | enabled")
        logger.info("Transport dimension | %d", raw_model.decoder_transport_dim)
        logger.info("Original dense mask head | retained")
        logger.info("SPCR | disabled")
        logger.info("HTPR | disabled")
        logger.info("Full-resolution refinement | disabled")
        logger.info("Early Mamba | disabled")
        logger.info("Separable mask head | disabled")
        stage4_backends = raw_model.mamba_backend_info().get("stage4", ())
        logger.info(
            "Stage4 backend | %s ×%d",
            stage4_backends[0] if stage4_backends else "unavailable",
            len(stage4_backends),
        )
    if bool(getattr(raw_model, "use_partial_narrow_transport_decoder", False)):
        logger.info("Model variant | DPER-RSDM-PNTD")
        logger.info("RSDM | enabled | blocks: 1 | half-resolution Stage1")
        logger.info("Partial narrow transport | enabled | dimension: %d", raw_model.partial_transport_dim)
        logger.info("PNTD deep region | F4/F3 narrow concat; F3→F2 decoder-side narrow transport")
        logger.info("F2 native skip | original independent 80-channel LSSCF projection retained")
        logger.info("F1 fusion/final_refine/dense output heads | original modules retained")
        logger.info("All-scale NTD24 | disabled")
        logger.info("SPCR/HTPR/DSMH/other failed candidates | disabled")
        stage4_backends = raw_model.mamba_backend_info().get("stage4", ())
        logger.info(
            "Mamba backend | stage4: %s ×%d",
            stage4_backends[0] if stage4_backends else "unavailable",
            len(stage4_backends),
        )
    if bool(getattr(raw_model, "use_common_private_direction_partitioned_shared_mamba", False)):
        cpdpsm = raw_model.stage4.route_mamba
        logger.info("Model variant | DPER-RSDM-CPDPSM64")
        logger.info("RSDM | enabled; exactly 1 block")
        logger.info("Stage4 common-private partition | shared: 64; private: 4×64")
        logger.info("Per-direction Mamba width | 128")
        logger.info("Stage4 directional traversals | horizontal, vertical, diagonal, anti_diagonal")
        logger.info("Stage4 Mamba core | shared %s ×1", cpdpsm.shared_mamba.backend)
        logger.info("Packed Mamba input | batch multiplier: 4; sequence length: H×W; channels: 128")
        logger.info("Stage4 spatial reduction | 1")
        logger.info("Stage4 Transformer | enabled")
        logger.info("Original LSSCF decoder | retained")
        logger.info("Original dense mask head | retained")
    if bool(getattr(raw_model, "use_direction_partitioned_shared_mamba", False)):
        dpsm = raw_model.stage4.route_mamba
        framework_enabled = bool(getattr(raw_model, "use_rsdm_stage1", False))
        decoder_uses_lsscf = bool(
            getattr(raw_model.decoder, "use_low_resolution_subpixel_separable_concat_fusion", False)
        )
        logger.info("Model variant | %s-DPSM80", "DPER-RSDM" if framework_enabled else "C40-S4")
        logger.info("Stage4 variant | DPSM80")
        logger.info("Stage4 hybrid order: %s", raw_model.stage4_hybrid_order)
        logger.info(
            "Framework | %s | RSDM blocks: %d | decoder: %s",
            "DPER-RSDM-LSSCF" if framework_enabled else "ordinary C40-S4",
            1 if framework_enabled else 0,
            "LSSCF" if decoder_uses_lsscf else "ordinary",
        )
        group_mapping = dpsm.group_scan_mapping
        logger.info("DPSM80 mechanism | scan_mode: %s | route_mode: %s", dpsm.scan_mode, dpsm.route_mode)
        logger.info(
            "DPSM80 group scans | %s",
            " | ".join(f"{group}: {direction}" for group, direction in group_mapping.items()),
        )
        logger.info(
            "Stage4 directional grouping | %d × %d channels; concatenated width %d; shared cores: %d",
            dpsm.route_bins,
            dpsm.group_channels,
            dpsm.dim,
            dpsm.instantiated_core_count,
        )
        logger.info("Stage4 Mamba core | shared %s ×1", dpsm.shared_mamba.backend)
        logger.info("Packed Mamba input | batch multiplier: 4; sequence length: H×W; channels: 80")
        logger.info("DPSM80 route modulation | fixed scale: 4.0")
        if dpsm.route_mode == "uniform":
            logger.info(
                "DPSM80 route mode | uniform 1/4 per group; route_head/raw logits retained but do not modulate mask features"
            )
        else:
            logger.info("DPSM80 route mode | learned softmax(route_logits) per pixel and group")
        logger.info(
            "DPSM80 structural supervision | tangent: %.4f | BPCM: %.4f",
            float(cfg["loss"].get("lambda_tangent", 0.0)),
            float(cfg["loss"].get("lambda_bpcm", 0.0)),
        )
        logger.info("Stage4 spatial reduction | 1")
        logger.info("Stage4 Transformer | enabled")
        logger.info("Old full-width Mamba paths/local compensation | absent")
        logger.info("Stage4 route supervision | available")
        logger.info("Original decoder fusion and dense mask head | retained")
    if bool(getattr(raw_model, "use_early_serpentine_connectivity_ssm", False)):
        early_module = raw_model.early_serpentine_connectivity_ssm
        assert early_module is not None
        logger.info("Early connectivity | DPER-ESCS: horizontal snake ×2 + diagonal snake ×2")
        logger.info(
            "Early Mamba | %s ×%d shared core | traversals: %d batched",
            early_module.backend,
            early_module.instantiated_core_count,
            early_module.traversal_count,
        )
        stage4_backends = raw_model.mamba_backend_info().get("stage4", ())
        logger.info(
            "Stage4 Mamba | %s ×%d independent paths",
            stage4_backends[0] if stage4_backends else "unavailable",
            len(stage4_backends),
        )
    loaded_encoder_key_count: int | None = None
    if args.init_encoder_checkpoint:
        if model_name != "DEMambaNet" or int(model_cfg.get("base_channels", 32)) != 40:
            raise ValueError("--init_encoder_checkpoint is supported only for the standard C40 DEMambaNet fine-tune model.")
        loaded_encoder_key_count = load_encoder_initialization(
            args.init_encoder_checkpoint,
            raw_model,
            map_location=device,
        )
        logger.info(
            "Initialized C40 encoder from train-only SSL checkpoint: %s | encoder keys: %d",
            args.init_encoder_checkpoint,
            loaded_encoder_key_count,
        )
    logger.info("Model parameters: %d", sum(parameter.numel() for parameter in raw_model.parameters()))
    model_ema = make_model_ema(raw_model, cfg["train"])
    if model_ema is not None:
        logger.info("Model EMA | enabled: True | decay: %.4f", model_ema.decay)
        logger.info("Validation model: EMA shadow weights")
        logger.info("Checkpoint model: EMA shadow weights")
    # Register after ModelEMA has deep-copied raw_model.  The hook observes only
    # the first existing student forward, cannot be inherited by the EMA model,
    # and never creates a diagnostic probe.
    register_first_dpsm_execution_hook(raw_model, logger)
    if loaded_encoder_key_count is not None:
        logger.info("Fine-tuning starts at epoch 0 with fresh optimizer, scheduler, scaler, and EMA.")

    distillation_cfg = cfg.get("distillation")
    distillation_enabled = isinstance(distillation_cfg, dict) and bool(distillation_cfg.get("enabled", False))
    teacher: torch.nn.Module | None = None
    distillation_criterion: ConfidenceBalancedDistillationLoss | None = None
    if distillation_enabled:
        assert isinstance(distillation_cfg, dict)
        distillation_criterion = ConfidenceBalancedDistillationLoss(
            temperature=float(distillation_cfg.get("temperature", 2.0)),
            foreground_weight=float(distillation_cfg.get("foreground_weight", 0.45)),
            background_weight=float(distillation_cfg.get("background_weight", 0.55)),
            confidence_power=float(distillation_cfg.get("confidence_power", 2.0)),
        ).to(device)
        teacher = build_distillation_teacher(distillation_cfg, device=device, logger=logger)
        if bool(getattr(raw_model, "use_direction_partitioned_shared_mamba", False)):
            logger.info("CBSD student variant | DPER-RSDM-DPSM80")
            logger.info("CBSD student Mamba backend | stage4: shared mamba_ssm ×1")
        logger.info(
            "CBSD | enabled: True | temperature: %.3f | lambda_max: %.3f | fg/bg: %.3f/%.3f | confidence_power: %.3f | warmup/ramp_end: %d/%d",
            distillation_criterion.temperature,
            float(distillation_cfg.get("lambda_max", 0.25)),
            distillation_criterion.foreground_weight,
            distillation_criterion.background_weight,
            distillation_criterion.confidence_power,
            int(distillation_cfg.get("warmup_epochs", 10)),
            int(distillation_cfg.get("ramp_end_epoch", 40)),
        )
    else:
        logger.info("CBSD | enabled: False")
        logger.info("Distillation | disabled")

    amp_name = str(cfg["train"].get("amp_dtype", "fp16")).lower()
    amp_dtype = torch.bfloat16 if amp_name == "bf16" else torch.float16
    amp_enabled = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    logger.info("Batch size: %d | AMP: %s", int(cfg["train"]["batch_size"]), amp_name if amp_enabled else "disabled")
    photometric_cfg = cfg["data"].get("photometric_augment", {})
    logger.info(
        "Photometric augment | enabled: %s | probability: %s | brightness: %s | contrast: %s | gamma_range: %s | channel_gain_probability: %s | channel_gain_range: %s | grayscale_probability: %s",
        bool(photometric_cfg.get("enabled", False)),
        photometric_cfg.get("probability", 0.0),
        photometric_cfg.get("brightness", 0.0),
        photometric_cfg.get("contrast", 0.0),
        photometric_cfg.get("gamma_range", [1.0, 1.0]),
        photometric_cfg.get("channel_gain_probability", 0.0),
        photometric_cfg.get("channel_gain_range", [1.0, 1.0]),
        photometric_cfg.get("grayscale_probability", 0.0),
    )
    logger.info(
        "Geometric augment | quarter-turn rotations: %s",
        "enabled" if bool(cfg["data"].get("allow_quarter_turns", True)) else "disabled",
    )
    scale_jitter_cfg = cfg["data"].get("scale_jitter") or {}
    logger.info(
        "Native scale jitter | enabled: %s | probability: %s | scale_range: %s | train only",
        bool(scale_jitter_cfg.get("enabled", False)),
        scale_jitter_cfg.get("probability", 0.0),
        scale_jitter_cfg.get("scale_range", [1.0, 1.0]),
    )
    criterion_cfg = dict(cfg["loss"])
    if auto_enabled:
        assert auto_plan is not None
        criterion_cfg.update(
            use_update_clock=True,
            scnp_warmup_updates=auto_plan.scnp_warmup_updates,
            bpcm_start_update=auto_plan.bpcm_start_update,
            bpcm_ramp_updates=auto_plan.bpcm_ramp_updates,
        )
    criterion = DEMambaLoss(**criterion_cfg).to(device)
    logger.info(
        "Loss config | mask_pos_weight: %.4f | edge_pos_weight: 1.0000 | scnp_pos_weight: 1.0000",
        criterion.mask_pos_weight,
    )
    logger.info(
        "Deep supervision | lambda_aux_mask: %.4f | lambda_aux_edge: %.4f",
        criterion.lambda_aux_mask,
        criterion.lambda_aux_edge,
    )
    if criterion.lambda_cfps > 0:
        logger.info(
            "CFPS | enabled; train-only | lambda: %.4f | start_epoch: %d | ramp_epochs: %d",
            criterion.lambda_cfps,
            criterion.cfps_start_epoch,
            criterion.cfps_ramp_epochs,
        )
    elif bool(getattr(raw_model, "use_common_private_direction_partitioned_shared_mamba", False)):
        logger.info("CFPS | disabled")
    if use_sdf_head:
        logger.info(
            "SDF auxiliary | enabled: True | lambda: %.4f | truncation: %.1fpx | supervision: narrow-band",
            criterion.lambda_sdf,
            criterion.sdf_truncation,
        )
    if criterion.lambda_component_query > 0:
        component_counts = scan_train_component_counts(
            cfg["data"]["data_root"],
            cfg["data"].get("img_size"),
            int(cfg["data"]["threshold"]),
        )
        max_components = max((count for _, count in component_counts), default=0)
        max_stems = [stem for stem, count in component_counts if count == max_components]
        if max_components > 64:
            raise ValueError(
                "Component-query training supports at most 64 connected components per image; "
                f"found {max_components} in: {', '.join(max_stems)}."
            )
        logger.info(
            "Component-query targets | train images: %d | max components: %d | samples: %s",
            len(component_counts),
            max_components,
            ", ".join(max_stems),
        )
    optimizer = make_optimizer(raw_model, cfg["train"], device, logger)
    scheduler = make_scheduler(
        optimizer,
        cfg,
        logger,
        auto_target_updates=auto_plan.target_updates if auto_enabled and auto_plan is not None else None,
    )
    start_epoch, best_score = 0, float("-inf")
    resume_auto_state: dict | None = None
    if args.resume:
        resume_metadata = load_checkpoint_metadata(args.resume, map_location=device) if auto_enabled else None
        start_epoch, best_score = load_checkpoint(
            args.resume,
            raw_model,
            optimizer,
            scheduler,
            map_location=device,
            prefer_raw_model=model_ema is not None,
        )
        if model_ema is not None:
            load_checkpoint(args.resume, model_ema.module, map_location=device)
        if auto_enabled:
            resume_auto_state = resume_metadata.get("auto_protocol_state") if resume_metadata is not None else None
            if auto_protocol_version == 2 and isinstance(resume_auto_state, dict):
                validate_auto_protocol_resume(
                    cfg["auto_protocol_resolved"],
                    resume_auto_state,
                    unsafe=bool(getattr(args, "auto_protocol_unsafe_resume", False)),
                )
            if not isinstance(resume_auto_state, dict):
                logger.warning(
                    "Auto Protocol resume checkpoint has no auto_protocol_state; using its checkpoint score as both initial tracks."
                )
        logger.info("Resumed from %s at epoch %d", args.resume, start_epoch)

    model: torch.nn.Module = raw_model
    if bool(cfg["train"].get("torch_compile", False)):
        try:
            model = torch.compile(raw_model)
            logger.info("torch.compile enabled")
        except Exception as error:  # pragma: no cover - depends on local compiler support
            logger.warning("torch.compile unavailable; continuing eagerly (%s)", error)

    formal_train_generator = torch.Generator(device="cpu").manual_seed(args.seed) if auto_enabled else None
    train_loader = build_loader(cfg, "train", True, generator=formal_train_generator)
    val_loader = build_loader(cfg, "val", False)
    if auto_enabled:
        logger.info("Formal dataloader generator seed: %d", args.seed)
        if isinstance(resume_auto_state, dict) and resume_auto_state.get("formal_train_generator_state") is not None:
            assert formal_train_generator is not None
            formal_train_generator.set_state(resume_auto_state["formal_train_generator_state"])
            logger.info("Auto Protocol resume | restored formal dataloader generator state")
    epochs, val_interval = int(cfg["train"]["epochs"]), max(1, int(cfg["train"].get("val_interval", 1)))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)
    validation_model = model_ema.module if model_ema is not None else model
    checkpoint_model = model_ema.module if model_ema is not None else raw_model
    run_start, smoothed_epoch_seconds = perf_counter(), None
    total_timing = {key: 0.0 for key in TIMING_KEYS}
    total_batches = 0.0
    global_update = 0
    auto_accumulation = accumulation if auto_enabled else 1
    auto_target_updates = auto_plan.target_updates if auto_enabled and auto_plan is not None else None
    run_target_updates = 1 if dry_run_requested and auto_enabled else auto_target_updates
    next_validation_update = auto_plan.validation_interval_updates if auto_enabled and auto_plan is not None else None
    early_cfg = cfg.get("auto_protocol", {}).get("early_stopping", {}) if auto_enabled else {}
    early_stopping = (
        UpdateEarlyStopping(
            min_update=auto_plan.min_early_stop_update,
            patience_updates=auto_plan.patience_updates,
            min_delta=float(early_cfg.get("min_delta", 0.0005)),
        )
        if auto_enabled and auto_plan is not None and bool(early_cfg.get("enabled", True))
        else None
    )
    checkpoint_selector = (
        AutoProtocolCheckpointState(min_delta=float(early_cfg.get("min_delta", 0.0005)))
        if auto_enabled
        else None
    )
    absolute_checkpoint = save_dir / "best_absolute_score.pth"
    significant_checkpoint = save_dir / "best_structural_score.pth"
    if auto_enabled and checkpoint_selector is not None:
        if isinstance(resume_auto_state, dict):
            checkpoint_selector.load_state_dict(resume_auto_state)
            global_update = int(resume_auto_state.get("global_update", 0))
        elif args.resume:
            checkpoint_selector.absolute_best_score = best_score
            checkpoint_selector.significant_best_score = best_score
            checkpoint_selector.absolute_best_update = global_update
            checkpoint_selector.significant_best_update = global_update
            checkpoint_selector.last_significant_update = global_update
        if early_stopping is not None:
            early_stopping.best_significant_score = checkpoint_selector.significant_best_score
            early_stopping.last_significant_update = checkpoint_selector.last_significant_update
        interval = auto_plan.validation_interval_updates
        next_validation_update = (global_update // interval + 1) * interval
        best_score = checkpoint_selector.significant_best_score
        update_resolved_run_state(
            cfg,
            best_update=checkpoint_selector.significant_best_update,
            best_epoch=start_epoch if checkpoint_selector.significant_best_update is not None else None,
        )
    stop_reason: str | None = None
    best_update: int | None = (
        checkpoint_selector.significant_best_update if checkpoint_selector is not None else None
    )
    best_epoch: int | None = (
        start_epoch if checkpoint_selector is not None and checkpoint_selector.significant_best_update is not None else None
    )
    validation_calls = 0
    cfps_diagnostic_epochs = {
        "best_structural": (float("-inf"), None),
        "Precision": (float("-inf"), None),
        "mIoU": (float("-inf"), None),
        "BoundaryF1": (float("-inf"), None),
        "ComponentError": (float("inf"), None),
    }
    for epoch in range(start_epoch, epochs):
        epoch_start = perf_counter()
        train_loss, epoch_stats = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            amp_enabled,
            amp_dtype,
            epoch,
            model_ema=model_ema,
            ema_source_model=raw_model,
            teacher=teacher,
            distillation_criterion=distillation_criterion,
            distillation_cfg=distillation_cfg if distillation_enabled else None,
            scheduler=scheduler if auto_enabled else None,
            accumulation=auto_accumulation,
            global_update=global_update,
            target_updates=run_target_updates,
            use_update_clock=auto_enabled,
            next_validation_update=next_validation_update,
            validation_interval_updates=auto_plan.validation_interval_updates if auto_enabled and auto_plan is not None else None,
            non_finite_diagnostic=bool(args.non_finite_diagnostic),
            logger=logger,
        )
        global_update = int(epoch_stats["global_update"])
        next_validation_update = epoch_stats["next_validation_update"]
        total_batches += accumulate_epoch_timing(total_timing, epoch_stats)
        if not auto_enabled:
            scheduler.step()
        logger.info("Learning rate: %.8f", optimizer.param_groups[0]["lr"])
        should_validate = (
            bool(epoch_stats["validation_crossed"])
            or bool(epoch_stats["reached_target"])
            if auto_enabled
            else (epoch + 1) % val_interval == 0 or epoch + 1 == epochs
        )
        if dry_run_requested and auto_enabled:
            should_validate = False
        val_metrics: dict[str, float] | None = validate(validation_model, val_loader, device) if should_validate else None
        if val_metrics is not None:
            validation_calls += 1
        score = (
            structural_score(val_metrics, cfg["evaluation"])
            if val_metrics is not None
            else (checkpoint_selector.significant_best_score if checkpoint_selector is not None else best_score)
        )
        if criterion.lambda_cfps > 0 and val_metrics is not None:
            if score >= cfps_diagnostic_epochs["best_structural"][0]:
                cfps_diagnostic_epochs["best_structural"] = (score, epoch + 1)
            for metric_name in ("Precision", "mIoU", "BoundaryF1"):
                if val_metrics[metric_name] >= cfps_diagnostic_epochs[metric_name][0]:
                    cfps_diagnostic_epochs[metric_name] = (val_metrics[metric_name], epoch + 1)
            if val_metrics["ComponentError"] <= cfps_diagnostic_epochs["ComponentError"][0]:
                cfps_diagnostic_epochs["ComponentError"] = (val_metrics["ComponentError"], epoch + 1)
        if val_metrics is not None and auto_enabled:
            assert checkpoint_selector is not None
            absolute_updated, significant_updated, delta_from_significant = checkpoint_selector.consider(
                score,
                global_update,
            )
            if early_stopping is not None:
                early_stopping.best_significant_score = checkpoint_selector.significant_best_score
                early_stopping.last_significant_update = checkpoint_selector.last_significant_update
            checkpoint_state = auto_protocol_checkpoint_payload(
                checkpoint_selector,
                early_stopping,
                global_update,
                formal_train_generator,
                cfg["auto_protocol_resolved"].get("resume_identity"),
            )
            if absolute_updated:
                save_checkpoint(
                    absolute_checkpoint,
                    checkpoint_model,
                    optimizer,
                    scheduler,
                    epoch,
                    checkpoint_selector.absolute_best_score,
                    val_metrics["mIoU"],
                    raw_model=raw_model if model_ema is not None else None,
                    checkpoint_type="absolute_best_diagnostic",
                    auto_protocol_state=checkpoint_state,
                )
            if significant_updated:
                best_score = checkpoint_selector.significant_best_score
                best_update = checkpoint_selector.significant_best_update
                best_epoch = epoch + 1
                save_checkpoint(
                    significant_checkpoint,
                    checkpoint_model,
                    optimizer,
                    scheduler,
                    epoch,
                    checkpoint_selector.significant_best_score,
                    val_metrics["mIoU"],
                    raw_model=raw_model if model_ema is not None else None,
                    checkpoint_type="significant_best",
                    auto_protocol_state=checkpoint_state,
                )
            patience_elapsed = global_update - checkpoint_selector.last_significant_update
            cfg["auto_protocol_resolved"].update(
                best_checkpoint_score=checkpoint_selector.significant_best_score,
                best_checkpoint_update=checkpoint_selector.significant_best_update,
                best_checkpoint_epoch=best_epoch,
                absolute_best_score=checkpoint_selector.absolute_best_score,
                absolute_best_update=checkpoint_selector.absolute_best_update,
                absolute_best_checkpoint=str(absolute_checkpoint),
                significant_best_score=checkpoint_selector.significant_best_score,
                significant_best_checkpoint=str(significant_checkpoint),
                last_significant_improvement_update=checkpoint_selector.last_significant_update,
            )
            logger.info(
                "Auto Protocol validation | current structural: %.6f | absolute best: %.6f @ update %s | significant best: %.6f @ update %s | delta from significant: %.6f | min_delta: %.6f | absolute_updated: %s | significant_updated: %s | patience_reset: %s | updates since significant: %d",
                score,
                checkpoint_selector.absolute_best_score,
                checkpoint_selector.absolute_best_update,
                checkpoint_selector.significant_best_score,
                checkpoint_selector.significant_best_update,
                delta_from_significant,
                checkpoint_selector.min_delta,
                absolute_updated,
                significant_updated,
                significant_updated,
                patience_elapsed,
            )
            if early_stopping is not None and early_stopping.should_stop(global_update):
                stop_reason = "early_stopping"
                logger.info("Auto Protocol early stopping triggered")
                logger.info("Stop update: %d | Target updates: %d", global_update, auto_plan.target_updates)
                logger.info(
                    "Last significant update: %d | Patience updates: %d | Minimum delta: %.6f",
                    checkpoint_selector.last_significant_update,
                    early_stopping.patience_updates,
                    early_stopping.min_delta,
                )
                logger.info(
                    "Significant-best update/score: %s/%.6f | absolute-best update/score: %s/%.6f",
                    checkpoint_selector.significant_best_update,
                    checkpoint_selector.significant_best_score,
                    checkpoint_selector.absolute_best_update,
                    checkpoint_selector.absolute_best_score,
                )
                logger.info("Final checkpoint selection pending | type=significant_best | path=%s", significant_checkpoint)
                logger.info("Stop reason: no structural improvement larger than min_delta")
        elif val_metrics is not None and score >= best_score:
            # Legacy non-AutoProtocol behaviour stays exactly as before.
            best_score = score
            best_epoch = epoch + 1
            save_checkpoint(
                significant_checkpoint,
                checkpoint_model,
                optimizer,
                scheduler,
                epoch,
                best_score,
                val_metrics["mIoU"],
                raw_model=raw_model if model_ema is not None else None,
            )
        # A protocol dry-run is an execution audit, not a resumable training
        # artifact.  Its resolved config is intentionally the only output.
        if not (dry_run_requested and auto_enabled):
            latest_auto_state = (
                auto_protocol_checkpoint_payload(
                    checkpoint_selector,
                    early_stopping,
                    global_update,
                    formal_train_generator,
                    cfg["auto_protocol_resolved"].get("resume_identity"),
                )
                if checkpoint_selector is not None
                else None
            )
            save_checkpoint(
                save_dir / "latest.pth",
                checkpoint_model,
                optimizer,
                scheduler,
                epoch,
                best_score,
                val_metrics["mIoU"] if val_metrics else None,
                raw_model=raw_model if model_ema is not None else None,
                checkpoint_type="latest" if auto_enabled else None,
                auto_protocol_state=latest_auto_state,
            )
        epoch_seconds = perf_counter() - epoch_start
        smoothed_epoch_seconds = epoch_seconds if smoothed_epoch_seconds is None else 0.8 * smoothed_epoch_seconds + 0.2 * epoch_seconds
        elapsed = perf_counter() - run_start
        eta = smoothed_epoch_seconds * (epochs - epoch - 1)
        if val_metrics is None:
            validation_text = "val skipped"
        else:
            validation_text = f"val mIoU {val_metrics['mIoU']:.4f} | F1 {val_metrics['F1']:.4f} | clDice {val_metrics['clDice']:.4f} | BoundaryF1 {val_metrics['BoundaryF1']:.4f} | ComponentError {val_metrics['ComponentError']:.3f} | structural {score:.4f}"
        logger.info(
            "Epoch %03d/%03d | loss %.4f | %s | epoch %s | elapsed %s | ETA %s",
            epoch + 1, epochs, train_loss, validation_text, format_duration(epoch_seconds), format_duration(elapsed), format_duration(eta),
        )
        if auto_enabled and auto_plan is not None:
            next_validation_text = next_validation_update if next_validation_update is not None else "complete"
            patience_remaining = (
                max(0, early_stopping.patience_updates - (global_update - early_stopping.last_significant_update))
                if early_stopping is not None and global_update >= early_stopping.min_update
                else "inactive"
            )
            logger.info(
                "Auto Protocol | global update %d/%d | effective batch %d | SCNP %.4f | BPCM %.4f | next validation: %s | checkpoint best %.4f @ update %s | significant best %.4f @ update %s | early-stop patience remaining: %s",
                global_update,
                auto_plan.target_updates,
                auto_plan.effective_batch_size,
                criterion._scnp_scale(epoch, global_update),
                criterion._bpcm_scale(epoch, global_update),
                next_validation_text,
                best_score,
                best_update,
                early_stopping.best_significant_score if early_stopping is not None else float("-inf"),
                early_stopping.last_significant_update if early_stopping is not None else None,
                patience_remaining,
            )
            if early_stopping is not None and global_update < early_stopping.min_update:
                logger.info("Early stopping: inactive until update %d", early_stopping.min_update)
        if criterion.lambda_cfps > 0:
            logger.info(
                "CFPS | scale: %.4f | raw_loss: %.4f | weighted_loss: %.6f | selected_pixels_per_image: %.2f | active_image_ratio: %.4f",
                epoch_stats["cfps_scale"],
                epoch_stats["cfps_loss"],
                epoch_stats["cfps_weighted_loss"],
                epoch_stats["cfps_selected_pixels_per_image"],
                epoch_stats["cfps_active_image_ratio"],
            )
        if criterion.lambda_aabfp > 0:
            logger.info(
                "AABFP | scale: %.4f | loss: %.4f | boundary_pixels_per_image: %.2f | bridge_pixels_per_image: %.2f",
                epoch_stats["aabfp_scale"],
                epoch_stats["aabfp_loss"],
                epoch_stats["aabfp_boundary_selected_pixels_per_image"],
                epoch_stats["aabfp_bridge_selected_pixels_per_image"],
            )
        if distillation_enabled:
            logger.info(
                "CBSD | original_loss: %.4f | raw_distillation_loss: %.4f | lambda_kd: %.4f | weighted_distillation_loss: %.4f | total_loss: %.4f",
                epoch_stats["original_loss"],
                epoch_stats["cbkd_raw_loss"],
                epoch_stats["lambda_kd"],
                epoch_stats["cbkd_weighted_loss"],
                train_loss,
            )
            logger.info(
                "CBSD teacher statistics | fg_probability: %.4f | bg_probability: %.4f | fg_confidence: %.4f | bg_confidence: %.4f",
                epoch_stats["teacher_foreground_probability"],
                epoch_stats["teacher_background_probability"],
                epoch_stats["foreground_confidence"],
                epoch_stats["background_confidence"],
            )
        logger.info("Performance | data %.3fs | forward+loss %.3fs | backward+optimizer %.3fs", epoch_stats["data"], epoch_stats["forward_loss"], epoch_stats["backward_optimizer"])
        if dry_run_requested and auto_enabled and global_update >= 1:
            stop_reason = "dry_run_one_optimizer_update"
            break
        if auto_enabled and (bool(epoch_stats["reached_target"]) or stop_reason is not None):
            if stop_reason is None:
                stop_reason = "target_updates_reached"
                logger.info("Stop reason: target_updates reached")
            break

    if auto_enabled:
        update_resolved_run_state(
            cfg,
            stop_reason=stop_reason or "epochs_exhausted",
            best_update=best_update,
            best_epoch=best_epoch,
        )
        cfg["auto_protocol_resolved"].update(
            final_global_update=global_update,
            validation_calls=validation_calls,
            best_checkpoint_score=(
                None
                if checkpoint_selector is None or checkpoint_selector.significant_best_score == float("-inf")
                else checkpoint_selector.significant_best_score
            ),
            absolute_best_score=(
                None
                if checkpoint_selector is None or checkpoint_selector.absolute_best_score == float("-inf")
                else checkpoint_selector.absolute_best_score
            ),
            absolute_best_update=None if checkpoint_selector is None else checkpoint_selector.absolute_best_update,
            significant_best_score=(
                None
                if checkpoint_selector is None or checkpoint_selector.significant_best_score == float("-inf")
                else checkpoint_selector.significant_best_score
            ),
            last_significant_improvement_update=(
                None if checkpoint_selector is None else checkpoint_selector.last_significant_update
            ),
        )
        refresh_resolved_config(config_path, cfg)
    if dry_run_requested and auto_enabled:
        cfg["auto_protocol_resolved"].update(
            final_global_update=global_update,
            dry_run_optimizer_updates=global_update,
            validation_calls=0,
            test_calls=0,
        )
        refresh_resolved_config(config_path, cfg)
        logger.info("Auto Protocol dry-run completed successfully")
        logger.info("Micro batches processed: %d", int(epoch_stats["batches"]))
        logger.info("Optimizer updates performed: %d", global_update)
        logger.info("Validation calls: 0")
        logger.info("Test calls: 0")
        logger.info("Exiting without formal training")
        return

    if criterion.lambda_cfps > 0:
        logger.info(
            "CFPS validation diagnostics | best_structural_epoch: %s | best_precision_epoch: %s | best_mIoU_epoch: %s | best_BoundaryF1_epoch: %s | lowest_ComponentError_epoch: %s",
            cfps_diagnostic_epochs["best_structural"][1],
            cfps_diagnostic_epochs["Precision"][1],
            cfps_diagnostic_epochs["mIoU"][1],
            cfps_diagnostic_epochs["BoundaryF1"][1],
            cfps_diagnostic_epochs["ComponentError"][1],
        )
    if auto_enabled:
        final_checkpoint, final_checkpoint_type = select_auto_protocol_final_checkpoint(
            significant_checkpoint,
            absolute_checkpoint,
        )
        if final_checkpoint_type == "absolute_best_fallback":
            logger.warning(
                "Auto Protocol final selection fallback: significant-best checkpoint is missing; using diagnostic absolute-best %s.",
                absolute_checkpoint,
            )
        load_checkpoint(final_checkpoint, checkpoint_model, map_location=device)
        assert checkpoint_selector is not None
        final_score = (
            checkpoint_selector.significant_best_score
            if final_checkpoint_type == "significant_best"
            else checkpoint_selector.absolute_best_score
        )
        final_update = (
            checkpoint_selector.significant_best_update
            if final_checkpoint_type == "significant_best"
            else checkpoint_selector.absolute_best_update
        )
        logger.info(
            "Final checkpoint selection | type=%s | update=%s | score=%.6f | path=%s",
            final_checkpoint_type,
            final_update,
            final_score,
            final_checkpoint,
        )
        cfg["auto_protocol_resolved"].update(
            final_checkpoint_type=final_checkpoint_type,
            final_checkpoint_path=str(final_checkpoint),
            final_checkpoint_score=final_score,
            final_checkpoint_update=final_update,
        )
        refresh_resolved_config(config_path, cfg)
    else:
        best_checkpoint = save_dir / "best_structural_score.pth"
        if best_checkpoint.exists():
            load_checkpoint(best_checkpoint, checkpoint_model, map_location=device)
    test_metrics = run_final_test(
        skip_final_test=args.skip_final_test,
        checkpoint_model=checkpoint_model,
        cfg=cfg,
        device=device,
        auto_enabled=auto_enabled,
        config_path=config_path if auto_enabled else None,
        logger=logger,
    )
    if test_metrics is None:
        return
    if auto_enabled:
        cfg["auto_protocol_resolved"]["test_calls"] = 1
        refresh_resolved_config(config_path, cfg)
    batches = max(total_batches, 1.0)
    logger.info("Training time: %s", format_duration(perf_counter() - run_start))
    logger.info("Average performance | data %.3fs | forward+loss %.3fs | backward+optimizer %.3fs", total_timing["data"] / batches, total_timing["forward_loss"] / batches, total_timing["backward_optimizer"] / batches)
    logger.info("Final test metrics (best structural-score checkpoint):")
    for key in ["IoU", "mIoU", "Dice", "F1", "Precision", "Recall", "ODS", "OIS", "BoundaryF1", "clDice", "ComponentError"]:
        logger.info("%s: %.4f", key, test_metrics[key])


if __name__ == "__main__":
    main()
