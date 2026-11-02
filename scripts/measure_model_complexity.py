"""Measure audited inference efficiency, parameter count, and model size.

The default input is one 3x512x512 image because recent crack-segmentation
complexity tables normalize all methods to 512x512.  Use --img_size to profile
the deployment resolution instead.  fvcore omits Mamba's fused
``MambaInnerFn``; this script therefore reports its standard lower bound, a
captured-input Mamba estimate, and their explicitly labelled sum.  The lower
bound is never presented as total FLOPs.

``Size`` is intentionally reproducible: it is the uncompressed FP32
``torch.save(model.state_dict())`` size, excluding optimizer state.  The script
also prints raw parameter bytes because papers occasionally use that different
definition without saying so.  FPS follows the reproducible RIFT protocol:
``eval()``, ``torch.inference_mode()``, fixed GPU-resident input, warm-up, then
the mean throughput of repeated forward passes.  It intentionally excludes
data loading, preprocessing, postprocessing, and host-to-device transfer.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import io
import json
import os
import sys
from collections import Counter
from contextlib import nullcontext
from importlib import metadata
from pathlib import Path
from statistics import mean, median, pstdev
from time import perf_counter
from typing import Any

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import build_model
from models.blocks import (
    DWConvBlock, StripConvBlock, NoDPSMTopoBlock,
    RIFTStructuralDirectionalModelingBlock, TopoRouteBlock,
    Stage4LocalOnly, Stage4LocalDPSM80,
)
from models.mamba_blocks import (
    CommonPrivateDirectionPartitionedSharedMamba,
    DirectionPartitionedSharedMamba,
    SequenceMambaCore,
    SingleRasterMamba80,
    TangentRoutedMambaBlock,
)
from models.transformer_blocks import TransformerGlobalBlock
from models.decoder import (
    DepthwiseSeparableMaskHead,
    DifferenceAwareSubpixelUpBlock,
    FullWidthResidualMaskHead,
    LowResolutionSubpixelSeparableConcatUpBlock,
    NarrowTransportDecoder,
    PartialNarrowTransportDecoder,
)


MIB = 1024 * 1024

MODULE_ANALYSIS_ALIASES = {
    "decoder.mask_input_projection": "decoder.mask_continuity_head.input_projection",
    "decoder.mask_continuity": "decoder.mask_continuity_head.continuity",
    "decoder.mask_continuity.0": "decoder.mask_continuity_head.continuity.0",
    "decoder.mask_continuity.1": "decoder.mask_continuity_head.continuity.1",
    "decoder.mask_half_head": "decoder.mask_continuity_head.half_head",
    "decoder.phase_residual_head": "decoder.mask_continuity_head.phase_residual",
}

FLOP_MODULE_NAMES = (
    "stem",
    "stem.full_detail_projection",
    "stem.haar_rebase",
    "stem.semantic_projection",
    "stem.hf_projection",
    "stem.hf_depthwise",
    "stem.low_frequency_gate",
    "narrow_dual_resolution_stage1",
    "narrow_dual_resolution_stage1.local_depthwise",
    "narrow_dual_resolution_stage1.local_norm",
    "narrow_dual_resolution_stage1.local_activation",
    "narrow_dual_resolution_stage1.local_pointwise",
    "narrow_dual_resolution_stage1.exchange_depthwise",
    "narrow_dual_resolution_stage1.exchange_norm",
    "narrow_dual_resolution_stage1.exchange_activation",
    "narrow_dual_resolution_stage1.exchange_projection",
    "full_resolution_joint_exchange",
    "full_resolution_joint_exchange.local_dense",
    "full_resolution_joint_exchange.local_norm",
    "full_resolution_joint_exchange.local_activation",
    "full_resolution_joint_exchange.phase_unshuffle",
    "full_resolution_joint_exchange.exchange_projection",
    "full_resolution_joint_exchange.exchange_norm",
    "partial_full_resolution_stage1",
    "partial_full_resolution_stage1.full_stage1",
    "partial_full_resolution_stage1.full_stage1.local",
    "partial_full_resolution_stage1.full_stage1.refine",
    "partial_full_resolution_stage1.phase_unshuffle",
    "partial_full_resolution_stage1.exchange_projection",
    "early_serpentine_connectivity_ssm",
    "early_serpentine_connectivity_ssm.input_projection",
    "early_serpentine_connectivity_ssm.shared_mamba",
    "early_serpentine_connectivity_ssm.output_projection",
    "early_serpentine_connectivity_ssm.scan_reorder",
    "early_serpentine_connectivity_ssm.inverse_reorder",
    "early_serpentine_connectivity_ssm.path_fusion",
    "stem_half",
    "stem_quarter",
    "detail_branch",
    "detail_refine",
    "stage1",
    "f1_reconstruction",
    "f1_reconstruction.phase_refiner",
    "rsdm_stage1",
    "rsdm_stage1.pre_norm",
    "rsdm_stage1.input_projection",
    "rsdm_stage1.local_structure",
    "rsdm_stage1.horizontal",
    "rsdm_stage1.vertical",
    "rsdm_stage1.diagonal",
    "rsdm_stage1.structure_gate",
    "rsdm_stage1.output_norm",
    "rsdm_stage1.output_projection",
    "down1",
    "stage2",
    "stage2.0.active_branch",
    "down2",
    "stage3",
    "stage3.local",
    "stage3.route_mamba",
    "stage3.route_mamba.route_head",
    "stage3.route_mamba.input_projection",
    "stage3.route_mamba.paths.0",
    "stage3.route_mamba.paths.1",
    "stage3.route_mamba.local_compensations.diagonal",
    "stage3.route_mamba.local_compensations.anti_diagonal",
    "stage3.route_mamba.out_proj",
    "stage3.global_branch",
    "stage3.fuse",
    "stage3.refine",
    "stage3.0.active_branch",
    "stage3_context",
    "stage3_ptca",
    "stage3_context.route_head",
    "stage3_context.core",
    "route_s3_head",
    "down3",
    "stage4",
    "stage4.0.active_branch",
    "stage4_context",
    "stage4_context.route_head",
    "stage4_context.core",
    "global_context",
    "global_context.route_head",
    "global_context.input_projection",
    "global_context.core",
    "global_context.output_projection",
    "stage4.local",
    "stage4.route_mamba",
    "stage4.route_mamba.route_head",
    "stage4.route_mamba.input_projection",
    "stage4.route_mamba.paths.0",
    "stage4.route_mamba.paths.1",
    "stage4.route_mamba.local_compensations.horizontal",
    "stage4.route_mamba.local_compensations.vertical",
    "stage4.route_mamba.out_proj",
    "stage4.route_mamba.pre_mix",
    "stage4.route_mamba.scans",
    "stage4.route_mamba.shared_mamba",
    "stage4.route_mamba.shared_mamba.core",
    "stage4.route_mamba.output_mix",
    "stage4.global_branch",
    "stage4.fuse",
    "stage4.refine",
    "decoder",
    "decoder.proj4",
    "decoder.proj3",
    "decoder.proj2",
    "decoder.proj1",
    "decoder.up3",
    "decoder.up2",
    "decoder.up1",
    "decoder.up1.decoder_projection",
    "decoder.up1.pixel_shuffle",
    "decoder.up1.skip_projection",
    "decoder.up1.delta_depthwise",
    "decoder.up1.delta_depthwise_norm",
    "decoder.up1.delta_pointwise",
    "decoder.up1.delta_pointwise_norm",
    "decoder.fuse3",
    "decoder.fuse2",
    "decoder.fuse1",
    "decoder.final_refine",
    "decoder.refine",
    "decoder.narrow_transport_decoder",
    "decoder.narrow_transport_decoder.proj1",
    "decoder.narrow_transport_decoder.proj2",
    "decoder.narrow_transport_decoder.proj3",
    "decoder.narrow_transport_decoder.proj4",
    "decoder.narrow_transport_decoder.refine4",
    "decoder.narrow_transport_decoder.refine3",
    "decoder.narrow_transport_decoder.refine2",
    "decoder.narrow_transport_decoder.refine1",
    "decoder.narrow_transport_decoder.adapter",
    "decoder.partial_narrow_transport_decoder",
    "decoder.partial_narrow_transport_decoder.f4_projection",
    "decoder.partial_narrow_transport_decoder.f3_projection",
    "decoder.partial_narrow_transport_decoder.f3_depthwise",
    "decoder.partial_narrow_transport_decoder.f3_pointwise",
    "decoder.partial_narrow_transport_decoder.f2_decoder_projection",
    "decoder.partial_narrow_transport_decoder.f2_transport_adapter",
    "decoder.partial_narrow_transport_decoder.f2_skip_projection",
    "decoder.partial_narrow_transport_decoder.f2_depthwise",
    "decoder.partial_narrow_transport_decoder.f2_pointwise",
    "decoder.half_upsample",
    "decoder.half_fusion",
    "decoder.mask_refine",
    "decoder.mask_projection",
    "decoder.mask_input_projection",
    "decoder.mask_continuity",
    "decoder.mask_continuity.0",
    "decoder.mask_continuity.1",
    "decoder.mask_half_head",
    "decoder.phase_residual_head",
    "decoder.mask_head",
    "decoder.mask_head.reduce",
    "decoder.mask_head.spatial",
    "decoder.mask_head.expand",
    "decoder.mask_head.classifier",
    "decoder.mask_head.depthwise",
    "decoder.mask_head.pointwise",
    "decoder.mask_head.output",
    "decoder.lsra",
    "decoder.lsra.detail_downsample",
    "decoder.lsra.local_attention",
    "decoder.lsra.upsample",
    "decoder.edge_head",
    "decoder.reliability_head",
    "decoder.route_head",
    "segr",
    "segr.semantic_projection",
    "segr.detail_projection",
    "segr.route_projection",
    "segr.recovery_blocks",
    "segr.delta_head",
    "csdr_decoder",
    "csdr_decoder.fuse3",
    "csdr_decoder.fuse2",
    "csdr_decoder.fuse1",
    "csdr_decoder.semantic_upsample",
    "csdr_decoder.detail_gate",
    "csdr_decoder.bottom_up_verification",
    "csdr_decoder.final_refine",
    "csdr_decoder.mask_head",
    "csdr_decoder.edge_head",
    "csdr_decoder.reliability_head",
    "decoder.proto_feature",
    "decoder.per_image_prototypes",
    "decoder.prototype_verifier",
    "decoder.prototype_mask_head",
)

TIMED_MODULE_NAMES = (
    "stem",
    "stem.full_detail_projection",
    "stem.haar_rebase",
    "stem.semantic_projection",
    "stem.hf_projection",
    "stem.hf_depthwise",
    "stem.low_frequency_gate",
    "narrow_dual_resolution_stage1",
    "narrow_dual_resolution_stage1.local_depthwise",
    "narrow_dual_resolution_stage1.local_norm",
    "narrow_dual_resolution_stage1.local_activation",
    "narrow_dual_resolution_stage1.local_pointwise",
    "narrow_dual_resolution_stage1.exchange_depthwise",
    "narrow_dual_resolution_stage1.exchange_norm",
    "narrow_dual_resolution_stage1.exchange_activation",
    "narrow_dual_resolution_stage1.exchange_projection",
    "full_resolution_joint_exchange",
    "full_resolution_joint_exchange.local_dense",
    "full_resolution_joint_exchange.local_norm",
    "full_resolution_joint_exchange.local_activation",
    "full_resolution_joint_exchange.phase_unshuffle",
    "full_resolution_joint_exchange.exchange_projection",
    "full_resolution_joint_exchange.exchange_norm",
    "partial_full_resolution_stage1",
    "partial_full_resolution_stage1.full_stage1",
    "partial_full_resolution_stage1.full_stage1.local",
    "partial_full_resolution_stage1.full_stage1.refine",
    "partial_full_resolution_stage1.phase_unshuffle",
    "partial_full_resolution_stage1.exchange_projection",
    "early_serpentine_connectivity_ssm",
    "early_serpentine_connectivity_ssm.input_projection",
    "early_serpentine_connectivity_ssm.shared_mamba",
    "early_serpentine_connectivity_ssm.output_projection",
    "early_serpentine_connectivity_ssm.scan_reorder",
    "early_serpentine_connectivity_ssm.inverse_reorder",
    "early_serpentine_connectivity_ssm.path_fusion",
    "stem_half",
    "stem_quarter",
    "detail_branch",
    "detail_refine",
    "stage1",
    "f1_reconstruction",
    "f1_reconstruction.phase_refiner",
    "rsdm_stage1",
    "rsdm_stage1.pre_norm",
    "rsdm_stage1.input_projection",
    "rsdm_stage1.local_structure",
    "rsdm_stage1.horizontal",
    "rsdm_stage1.vertical",
    "rsdm_stage1.diagonal",
    "rsdm_stage1.structure_gate",
    "rsdm_stage1.output_norm",
    "rsdm_stage1.output_projection",
    "down1",
    "stage2",
    "stage2.0.active_branch",
    "down2",
    "stage3",
    "stage3.local",
    "stage3.route_mamba",
    "stage3.route_mamba.route_head",
    "stage3.route_mamba.input_projection",
    "stage3.route_mamba.paths.0",
    "stage3.route_mamba.paths.1",
    "stage3.route_mamba.local_compensations.diagonal",
    "stage3.route_mamba.local_compensations.anti_diagonal",
    "stage3.route_mamba.out_proj",
    "stage3.global_branch",
    "stage3.fuse",
    "stage3.refine",
    "stage3.0.active_branch",
    "stage3_context",
    "stage3_ptca",
    "stage3_context.route_head",
    "stage3_context.core",
    "route_s3_head",
    "down3",
    "stage4",
    "stage4.local",
    "stage4.route_mamba",
    "stage4.route_mamba.route_head",
    "stage4.route_mamba.input_projection",
    "stage4.route_mamba.paths.0",
    "stage4.route_mamba.paths.1",
    "stage4.route_mamba.local_compensations.horizontal",
    "stage4.route_mamba.local_compensations.vertical",
    "stage4.route_mamba.out_proj",
    "stage4.route_mamba.pre_mix",
    "stage4.route_mamba.scans",
    "stage4.route_mamba.shared_mamba",
    "stage4.route_mamba.shared_mamba.core",
    "stage4.route_mamba.output_mix",
    "stage4.global_branch",
    "stage4.fuse",
    "stage4.refine",
    "stage4.0.active_branch",
    "stage4_context",
    "stage4_context.route_head",
    "stage4_context.core",
    "global_context",
    "global_context.route_head",
    "decoder",
    "decoder.up3",
    "decoder.up2",
    "decoder.up1",
    "decoder.up1.decoder_projection",
    "decoder.up1.pixel_shuffle",
    "decoder.up1.skip_projection",
    "decoder.up1.delta_depthwise",
    "decoder.up1.delta_depthwise_norm",
    "decoder.up1.delta_pointwise",
    "decoder.up1.delta_pointwise_norm",
    "decoder.fuse3",
    "decoder.fuse2",
    "decoder.fuse1",
    "decoder.final_refine",
    "decoder.refine",
    "decoder.narrow_transport_decoder",
    "decoder.narrow_transport_decoder.proj1",
    "decoder.narrow_transport_decoder.proj2",
    "decoder.narrow_transport_decoder.proj3",
    "decoder.narrow_transport_decoder.proj4",
    "decoder.narrow_transport_decoder.refine4",
    "decoder.narrow_transport_decoder.refine3",
    "decoder.narrow_transport_decoder.refine2",
    "decoder.narrow_transport_decoder.refine1",
    "decoder.narrow_transport_decoder.adapter",
    "decoder.partial_narrow_transport_decoder",
    "decoder.partial_narrow_transport_decoder.f4_projection",
    "decoder.partial_narrow_transport_decoder.f3_projection",
    "decoder.partial_narrow_transport_decoder.f3_depthwise",
    "decoder.partial_narrow_transport_decoder.f3_pointwise",
    "decoder.partial_narrow_transport_decoder.f2_decoder_projection",
    "decoder.partial_narrow_transport_decoder.f2_transport_adapter",
    "decoder.partial_narrow_transport_decoder.f2_skip_projection",
    "decoder.partial_narrow_transport_decoder.f2_depthwise",
    "decoder.partial_narrow_transport_decoder.f2_pointwise",
    "decoder.half_upsample",
    "decoder.half_fusion",
    "decoder.mask_refine",
    "decoder.mask_projection",
    "decoder.mask_input_projection",
    "decoder.mask_continuity",
    "decoder.mask_continuity.0",
    "decoder.mask_continuity.1",
    "decoder.mask_half_head",
    "decoder.phase_residual_head",
    "decoder.mask_head",
    "decoder.mask_head.reduce",
    "decoder.mask_head.spatial",
    "decoder.mask_head.expand",
    "decoder.mask_head.classifier",
    "decoder.mask_head.depthwise",
    "decoder.mask_head.pointwise",
    "decoder.mask_head.output",
    "decoder.lsra",
    "decoder.lsra.detail_downsample",
    "decoder.lsra.local_attention",
    "decoder.lsra.upsample",
    "decoder.edge_head",
    "decoder.reliability_head",
    "decoder.route_head",
    "segr",
    "segr.semantic_projection",
    "segr.detail_projection",
    "segr.route_projection",
    "segr.recovery_blocks",
    "segr.delta_head",
    "csdr_decoder",
    "csdr_decoder.fuse3",
    "csdr_decoder.fuse2",
    "csdr_decoder.fuse1",
    "csdr_decoder.semantic_upsample",
    "csdr_decoder.detail_gate",
    "csdr_decoder.bottom_up_verification",
    "csdr_decoder.final_refine",
    "csdr_decoder.mask_head",
    "csdr_decoder.edge_head",
    "csdr_decoder.reliability_head",
    "decoder.proto_feature",
    "decoder.per_image_prototypes",
    "decoder.prototype_verifier",
    "decoder.prototype_mask_head",
)


class _FullMultiOutputModel(torch.nn.Module):
    """Keep the three C40 prediction heads live for FLOP tracing."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.model(image)
        try:
            return outputs["mask"], outputs["edge"], outputs["reliability"]
        except KeyError as error:
            raise RuntimeError("Full multi-output FLOPs require mask, edge, and reliability predictions.") from error


class _MaskOutputEffectiveModel(torch.nn.Module):
    """Trace only the mask dataflow of the normal full forward method."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)["mask"]


class _MaskOnlyModel(torch.nn.Module):
    """Trace the real eval-only deployment entry point as a Tensor output."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        forward_mask_only = getattr(self.model, "forward_mask_only", None)
        if not callable(forward_mask_only):
            raise RuntimeError("--mask_only requires a model with forward_mask_only(image).")
        return forward_mask_only(image)["mask"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure FLOPs, parameters, and serialized FP32 model size.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Experiment YAML containing the model section.")
    parser.add_argument(
        "--img_size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=(512, 512),
        help="Profile input height and width. Table-2 comparison default is 512 512.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size used only for FLOPs/memory profiling.")
    parser.add_argument("--device", default=None, help="For example cuda, cuda:0, or cpu.")
    parser.add_argument("--warmup", type=int, default=50, help="Untimed inference iterations before FPS measurement.")
    parser.add_argument("--iterations", type=int, default=200, help="Timed inference iterations used for FPS measurement.")
    parser.add_argument(
        "--benchmark_repeats",
        type=int,
        default=1,
        help="Independent warmup/timed FPS repetitions; use 3 for formal DPSM admission.",
    )
    parser.add_argument(
        "--amp",
        choices=("off", "fp16", "bf16"),
        default="off",
        help="Autocast precision used only for the FPS benchmark; Params/Size stay FP32.",
    )
    parser.add_argument(
        "--strict_flops",
        action="store_true",
        help="Return a nonzero exit code when fvcore reports unsupported operators.",
    )
    parser.add_argument(
        "--by_module",
        action="store_true",
        help="Report selected module FLOPs and CUDA forward timing diagnostics.",
    )
    parser.add_argument(
        "--mask_only",
        action="store_true",
        help="Profile the eval-only deployment path that computes only the mask prediction.",
    )
    parser.add_argument(
        "--inference_path",
        choices=("full", "mask-effective", "mask-only"),
        default=None,
        help="Selected latency/FPS path; --mask_only remains a backward-compatible alias for mask-only.",
    )
    parser.add_argument(
        "--require_mamba_ssm",
        action="store_true",
        help="Reject fallback/disabled Stage4 Mamba paths when one is instantiated.",
    )
    parser.add_argument(
        "--expected-mamba-cores",
        type=int,
        default=None,
        help=(
            "Formal Stage4 audit contract. 0 requires a genuine no-Mamba Stage4; a positive value requires "
            "exactly that many instantiated Stage4 SequenceMambaCore modules and mamba_ssm backends."
        ),
    )
    parser.add_argument("--report_csv", default=None, help="Append one unified machine-readable measurement row.")
    parser.add_argument("--report_json", default=None, help="Write/append the same measurement row as JSON.")
    parser.add_argument(
        "--mamba-boundary-report",
        default=None,
        help="Write actual mamba_ssm source and single-core fvcore boundary evidence as JSON.",
    )
    parser.add_argument("--experiment", default=None, help="Report label; defaults to the config stem.")
    return parser.parse_args()


def load_model(config_path: str | Path, device: torch.device) -> tuple[str, torch.nn.Module]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        cfg: dict[str, Any] = yaml.safe_load(handle)

    try:
        model_cfg = dict(cfg["model"])
        model_name = model_cfg.pop("name")
    except (KeyError, TypeError) as error:
        raise ValueError(f"{config_path} must contain model.name and its model arguments.") from error

    model = build_model(model_name, **model_cfg).to(device).eval()
    return str(model_name), model


def tensor_bytes(tensors: list[torch.Tensor]) -> int:
    """Return tensor element bytes; parameter aliases are counted only once."""

    seen: set[tuple[int, int, torch.dtype]] = set()
    total = 0
    for tensor in tensors:
        if tensor.device.type == "meta":
            continue
        storage = tensor.untyped_storage()
        key = (storage.data_ptr(), storage.nbytes(), tensor.dtype)
        if key not in seen:
            seen.add(key)
            total += tensor.numel() * tensor.element_size()
    return total


def state_dict_size_bytes(model: torch.nn.Module) -> int:
    """Serialize only model state, never optimizer or training checkpoint data."""

    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    return stream.tell()


def autocast_context(device: torch.device, amp: str):
    if amp == "off":
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("--amp fp16/bf16 requires a CUDA device.")
    dtype = torch.float16 if amp == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_fps(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
    amp: str,
) -> tuple[float, float, int | None]:
    """Return image throughput, mean batch latency, and inference peak CUDA memory.

    The input is allocated on the target device before this function so the
    measurement excludes host-to-device transfer and data pipeline overhead.
    """

    with torch.inference_mode():
        with autocast_context(device, amp):
            for _ in range(warmup):
                model(image)
            synchronize(device)

            if device.type == "cuda":
                # Reset after warm-up so one-time kernel/autotune allocations are
                # not reported as inference peak memory.
                torch.cuda.reset_peak_memory_stats(device)

            if device.type == "cuda":
                start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start_event.record()
                for _ in range(iterations):
                    model(image)
                end_event.record()
                synchronize(device)
                elapsed = start_event.elapsed_time(end_event) / 1000.0
            else:
                start = perf_counter()
                for _ in range(iterations):
                    model(image)
                elapsed = perf_counter() - start

    fps = image.shape[0] * iterations / elapsed
    batch_latency_ms = elapsed * 1000.0 / iterations
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return fps, batch_latency_ms, peak_memory


def measure_flops(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    inference_path: str,
) -> tuple[float, Counter[str]]:
    """Return FLOPs for ``full``, ``mask-effective``, or real ``mask-only``."""

    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError as error:
        raise RuntimeError(
            "FLOP measurement requires fvcore. Install it in the training environment with: "
            "pip install fvcore"
        ) from error

    wrappers: dict[str, torch.nn.Module] = {
        "full": _FullMultiOutputModel(model),
        "mask-effective": _MaskOutputEffectiveModel(model),
        "mask-only": _MaskOnlyModel(model),
    }
    try:
        wrapper = wrappers[inference_path]
    except KeyError as error:
        raise ValueError(f"Unsupported FLOP inference path: {inference_path!r}.") from error
    analysis = FlopCountAnalysis(wrapper, (image,))
    analysis.unsupported_ops_warnings(False)
    analysis.uncalled_modules_warnings(False)
    total_flops = float(analysis.total())
    unsupported = Counter({str(name): int(count) for name, count in analysis.unsupported_ops().items()})
    return total_flops, unsupported


def _stage4_sequence_cores(model: torch.nn.Module) -> list[SequenceMambaCore]:
    """Return actual Stage4 core modules, never inferred from a config name."""

    route = getattr(getattr(model, "stage4", None), "route_mamba", None)
    if not isinstance(route, torch.nn.Module):
        return []
    return [module for module in route.modules() if isinstance(module, SequenceMambaCore)]


def _stage4_backend_details(model: torch.nn.Module) -> list[str]:
    return [str(core.backend) for core in _stage4_sequence_cores(model)]


def capture_stage4_execution(
    model: torch.nn.Module,
    image: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[int, list[tuple[int, ...]]]]:
    """Run one normal forward and capture true inputs of every Stage4 Mamba core."""

    captured: dict[int, list[tuple[int, ...]]] = {}
    handles = []
    for index, core in enumerate(_stage4_sequence_cores(model)):
        captured[index] = []

        def hook(_module: torch.nn.Module, inputs: tuple[object, ...], *, _index: int = index) -> None:
            if inputs and isinstance(inputs[0], torch.Tensor):
                captured[_index].append(tuple(int(value) for value in inputs[0].shape))

        # Hook the backend module rather than its SequenceMambaCore wrapper:
        # for formal mamba_ssm runs this is one call to MambaInnerFn.
        handles.append(core.core.register_forward_pre_hook(hook))
    try:
        outputs = model(image)
    finally:
        for handle in handles:
            handle.remove()
    if not isinstance(outputs, dict):
        raise RuntimeError("Normal model forward must return its output dictionary during the Stage4 audit.")
    return outputs, captured


def audit_strict_stage4_deletion(model: torch.nn.Module) -> dict[str, object]:
    """Accept the two strict deletions without weakening historical contracts."""
    stage = getattr(model, "stage4", None)
    variant = getattr(model, "stage4_variant", None)
    if variant is None and not isinstance(stage, (Stage4LocalOnly, Stage4LocalDPSM80)):
        return {}
    expected = {"local_only": Stage4LocalOnly, "local_dpsm80": Stage4LocalDPSM80}
    if variant not in expected or not isinstance(stage, expected[variant]):
        raise RuntimeError("Strict Stage4 selector does not match the real module class.")
    if any(isinstance(m, (TransformerGlobalBlock, torch.nn.Identity)) for m in stage.modules()) or hasattr(stage, "global_branch"):
        raise RuntimeError("Strict Stage4 must physically delete Transformer, without Identity placeholders.")
    if not (len(stage.local) == 2 and isinstance(stage.local[0], DWConvBlock)
            and isinstance(stage.local[1], StripConvBlock) and isinstance(stage.refine, DWConvBlock)):
        raise RuntimeError("Strict Stage4 must retain the original local branch and refine.")
    conv = next(m for m in stage.fuse.modules() if isinstance(m, torch.nn.Conv2d))
    inputs = 320 if variant == "local_only" else 640
    if (stage.channels, conv.in_channels, conv.out_channels, conv.kernel_size) != (320, inputs, 320, (1, 1)):
        raise RuntimeError("Strict Stage4 fusion shape differs from T0/T2.")
    if variant == "local_only":
        if hasattr(stage, "route_mamba") or any(isinstance(m, SequenceMambaCore) for m in stage.modules()):
            raise RuntimeError("T0 must physically delete routed/Mamba modules.")
    else:
        route = getattr(stage, "route_mamba", None)
        if not (isinstance(route, DirectionPartitionedSharedMamba) and route.dim == 320
                and route.group_channels == 80 and route.scan_mode == "four_direction"
                and route.route_mode == "learned"):
            raise RuntimeError("T2 requires the unchanged formal four-direction DPSM80.")
    return {"stage4_strict_variant": variant, "stage4_transformer_present": False,
            "stage4_fusion_in_channels": inputs, "stage4_fusion_out_channels": 320,
            "stage4_local_params": sum(p.numel() for p in stage.local.parameters())}


def measurement_environment(device: torch.device) -> dict[str, object]:
    """Observe performance settings without changing any of them."""
    def version(package: str) -> str | None:
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            return None
    return {
        "python": sys.version, "platform": sys.platform, "pytorch": torch.__version__,
        "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" and torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version(), "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cpu_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "mamba_ssm": version("mamba-ssm"), "causal_conv1d": version("causal-conv1d"), "fvcore": version("fvcore"),
        "environment_variables": {k: os.environ.get(k) for k in
            ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE", "PYTORCH_CUDA_ALLOC_CONF")},
        "weights": "random initialization; no checkpoint loaded", "torch_compile": False,
        "model_parameter_dtype": "float32", "timing_autocast_dtype": "specified by --amp",
    }


def audit_stage4_mamba(
    model: torch.nn.Module,
    *,
    variant: str,
    expected_cores: int | None,
    outputs: dict[str, torch.Tensor],
    captured_inputs: dict[int, list[tuple[int, ...]]],
    formal_backend: bool,
) -> dict[str, object]:
    """Validate the real Stage4 topology and return reproducible audit metadata.

    ``expected_cores`` is deliberately explicit.  Empty backend metadata is accepted
    only for an audited, real no-Mamba Stage4; it is never used as a proxy for that
    topology.
    """

    if expected_cores is not None and expected_cores < 0:
        raise ValueError("--expected-mamba-cores must be non-negative.")
    stage4 = getattr(model, "stage4", None)
    route = getattr(stage4, "route_mamba", None)
    cores = _stage4_sequence_cores(model)
    actual_cores = len(cores)
    backends = _stage4_backend_details(model)
    forward_calls = sum(len(shapes) for shapes in captured_inputs.values())
    routed_sequences = int(getattr(route, "active_mamba_path_count", 0)) if route is not None else 0
    route_s4_present = "route_s4" in outputs and outputs["route_s4"] is not None
    input_shapes = [shape for shapes in captured_inputs.values() for shape in shapes]
    token_slots = sum(shape[0] * shape[1] for shape in input_shapes)
    packed_execution = isinstance(route, DirectionPartitionedSharedMamba)
    diagonal_padding_slots = 0
    route_s4 = outputs.get("route_s4")
    if route_s4_present and isinstance(route_s4, torch.Tensor) and isinstance(route, TangentRoutedMambaBlock):
        unpadded_direction_slots = int(route_s4.shape[0] * route_s4.shape[-2] * route_s4.shape[-1])
        for direction, shapes in zip(route.active_mamba_directions, captured_inputs.values(), strict=True):
            if direction in ("diagonal", "anti_diagonal"):
                diagonal_padding_slots += sum(shape[0] * shape[1] - unpadded_direction_slots for shape in shapes)
    details = {
        **audit_strict_stage4_deletion(model),
        "variant": variant,
        "stage4_class": type(stage4).__name__ if stage4 is not None else None,
        "route_mamba_present": route is not None,
        "route_head_present": bool(getattr(route, "route_head", None) is not None),
        "route_s4_present": route_s4_present,
        "mamba_instantiated_core_count": actual_cores,
        "mamba_forward_call_count": forward_calls,
        "mamba_routed_sequence_count": routed_sequences,
        "mamba_backends": backends,
        "mamba_input_shapes": input_shapes,
        "mamba_token_slots": token_slots,
        "mamba_diagonal_padding_token_slots": diagonal_padding_slots,
        "mamba_packed_execution": packed_execution,
    }

    def failure(reason: str) -> RuntimeError:
        return RuntimeError(
            "Stage4 Mamba audit failed | "
            f"variant={variant} | expected={expected_cores} | actual={actual_cores} | "
            f"stage4={details['stage4_class']} | backends={backends or ['<none>']} | {reason}"
        )

    if expected_cores is not None and actual_cores != expected_cores:
        raise failure("instantiated core count mismatch")
    if expected_cores == 0:
        if not isinstance(stage4, (NoDPSMTopoBlock, Stage4LocalOnly)):
            raise failure("expected zero cores requires NoDPSMTopoBlock or strict Stage4LocalOnly")
        if route is not None or hasattr(stage4, "route_head") or route_s4_present:
            raise failure("zero-core Stage4 must not retain route_mamba, route head, or route_s4")
        if forward_calls != 0 or routed_sequences != 0:
            raise failure("zero-core Stage4 executed a Mamba path")
    elif expected_cores is not None:
        if not isinstance(stage4, (TopoRouteBlock, Stage4LocalDPSM80)) or route is None or not route_s4_present:
            raise failure("positive core count requires routed TopoRouteBlock/Stage4LocalDPSM80 and route_s4 output")
        if not backends or len(backends) != actual_cores or any(backend != "mamba_ssm" for backend in backends):
            raise failure("formal positive-core measurement requires mamba_ssm metadata for every core")
    elif formal_backend and actual_cores > 0:
        if not backends or len(backends) != actual_cores or any(backend != "mamba_ssm" for backend in backends):
            raise failure("--require-mamba-ssm requires mamba_ssm metadata for every actual Stage4 core")
    return details


def validate_mask_only_equivalence(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    device: torch.device,
    amp: str,
) -> dict[str, object]:
    """Fail closed unless the actual deployment path returns the same mask."""

    if not callable(getattr(model, "forward_mask_only", None)):
        raise RuntimeError("Formal mask-only measurement requires forward_mask_only(image).")
    with torch.inference_mode(), autocast_context(device, amp):
        full = model(image)
        if not isinstance(full, dict) or not isinstance(full.get("mask"), torch.Tensor):
            raise RuntimeError("Full forward did not return a Tensor mask for mask-only equivalence.")
        mask_only = model.forward_mask_only(image)
    if not isinstance(mask_only, dict) or not isinstance(mask_only.get("mask"), torch.Tensor):
        raise RuntimeError("forward_mask_only did not return a Tensor mask.")
    full_mask = full["mask"]
    only_mask = mask_only["mask"]
    if full_mask.shape != only_mask.shape:
        raise RuntimeError(
            f"Mask-only equivalence failed: full shape {tuple(full_mask.shape)} != mask-only {tuple(only_mask.shape)}."
        )
    difference = (full_mask.float() - only_mask.float()).abs()
    result = {
        "mask_only_shape": tuple(int(value) for value in full_mask.shape),
        "mask_only_max_abs_error": float(difference.max().item()),
        "mask_only_mean_abs_error": float(difference.mean().item()),
        "mask_only_allclose": bool(torch.allclose(full_mask, only_mask, atol=1e-3, rtol=1e-3)),
    }
    if not result["mask_only_allclose"]:
        raise RuntimeError(
            "Mask-only equivalence failed at FP16 acceptance tolerance "
            f"(max_abs={result['mask_only_max_abs_error']:.3e}, mean_abs={result['mask_only_mean_abs_error']:.3e})."
        )
    return result


class _MambaCoreProbe(torch.nn.Module):
    """Trace exactly the mamba_ssm module, excluding SequenceMambaCore norms."""

    def __init__(self, core: SequenceMambaCore) -> None:
        super().__init__()
        self.core = core.core

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.core(sequence)


def _mamba_component_costs(core: SequenceMambaCore) -> dict[str, int]:
    """Per-token analytical components using fvcore's one-MAC-is-one-FLOP convention."""

    d_model = int(core.dim)
    d_inner = int(getattr(core.core, "d_inner", d_model * int(core.expand)))
    d_state = int(getattr(core.core, "d_state", core.d_state))
    d_conv = int(getattr(core.core, "d_conv", core.d_conv))
    dt_rank = int(getattr(core.core, "dt_rank", max(1, (d_model + 15) // 16)))
    return {
        "in_proj": d_model * (2 * d_inner),
        "conv": d_inner * d_conv,
        "x_proj": d_inner * (dt_rank + 2 * d_state),
        "dt_proj": dt_rank * d_inner,
        "out_proj": d_inner * d_model,
        "selective_scan_and_skip": 6 * d_inner * d_state + 2 * d_inner,
    }


def diagnose_mamba_fused_boundary(core: SequenceMambaCore, shape: tuple[int, int, int]) -> dict[str, object]:
    """Saveable evidence for which Mamba work fvcore actually sees.

    This is deliberately conservative: a component is manually added only when
    the installed source passes it into ``mamba_inner_fn`` *and* its isolated
    fvcore module count is zero.  Any ambiguous boundary becomes NOT FOR PAPER.
    """

    if core.backend != "mamba_ssm":
        return {"flops_status": "NOT_FOR_PAPER_non_mamba_ssm_backend", "backend": core.backend}
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError as error:
        return {"flops_status": "NOT_FOR_PAPER_fvcore_missing", "error": str(error)}
    mamba = core.core
    try:
        source_file = inspect.getsourcefile(mamba.forward)
        source_lines, source_start_line = inspect.getsourcelines(mamba.forward)
        source = "".join(source_lines)
    except (OSError, TypeError):
        source_file = inspect.getsourcefile(type(mamba))
        source_start_line = None
        try:
            source = Path(source_file).read_text(encoding="utf-8") if source_file is not None else ""
        except OSError:
            source = ""
    source_lower = source.lower()
    uses_fast_path = bool(getattr(mamba, "use_fast_path", False))
    inner_present = "mamba_inner_fn" in source_lower
    inner_relative_line = next((index + 1 for index, line in enumerate(source.splitlines()) if "mamba_inner_fn" in line), None)
    call_start = source.find("mamba_inner_fn(")
    call_source = ""
    if call_start >= 0:
        depth = 0
        for offset, character in enumerate(source[call_start:]):
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0:
                    call_source = source[call_start:call_start + offset + 1]
                    break
    # The source excerpt is written verbatim to JSON for human review. These
    # booleans are only gates; they are not a replacement for the trace below.
    source_passes = {
        "in_proj": "self.in_proj" in call_source,
        "conv": "self.conv1d" in call_source,
        "x_proj": "self.x_proj" in call_source,
        "dt_proj": "self.dt_proj" in call_source,
        "out_proj": "self.out_proj" in call_source,
    }
    device = next(mamba.parameters()).device
    sample = torch.zeros(shape, device=device, dtype=next(mamba.parameters()).dtype)
    probe = _MambaCoreProbe(core).eval()
    analysis = FlopCountAnalysis(probe, (sample,))
    analysis.unsupported_ops_warnings(False); analysis.uncalled_modules_warnings(False)
    fvcore_total = float(analysis.total())
    by_operator = {str(name): float(value) for name, value in analysis.by_operator().items()}
    by_module = {str(name): float(value) for name, value in analysis.by_module().items()}
    unsupported = {str(name): int(value) for name, value in analysis.unsupported_ops().items()}
    counted_by_component: dict[str, float] = {}
    module_names = {"in_proj": "in_proj", "conv": "conv1d", "x_proj": "x_proj", "dt_proj": "dt_proj", "out_proj": "out_proj"}
    for component, suffix in module_names.items():
        counted_by_component[component] = sum(value for name, value in by_module.items() if name.endswith(suffix))
    counted_matmul = sum(value for name, value in by_operator.items() if "matmul" in name.lower())
    source_projects_before_fused = call_start >= 0 and "self.in_proj" in source[:call_start] and not source_passes["in_proj"]
    has_inner = any("MambaInnerFn" in name for name in unsupported)
    boundary: dict[str, str] = {}
    for component in module_names:
        if component == "in_proj" and source_projects_before_fused and counted_matmul > 0:
            # mamba-ssm 2.0.3 executes self.in_proj.weight @ hidden_states
            # before mamba_inner_fn(xz, ...). This is a raw matmul rather than
            # a module call, so by_module is zero but by_operator proves fvcore
            # already counted it.
            boundary[component] = "fvcore_counted_outside_fused"
            counted_by_component[component] = counted_matmul
        elif counted_by_component[component] > 0:
            boundary[component] = "fvcore_counted_outside_fused"
        elif uses_fast_path and inner_present and source_passes[component] and has_inner:
            boundary[component] = "inside_MambaInnerFn_missing"
        else:
            boundary[component] = "ambiguous"
    boundary["selective_scan_and_skip"] = "inside_MambaInnerFn_missing" if uses_fast_path and inner_present and has_inner else "ambiguous"
    allowed_unsupported = (
        "MambaInnerFn", "add", "mul", "sub", "div", "neg", "square", "sqrt", "clamp_min", "gelu", "silu", "relu", "sigmoid", "softmax", "exp", "sum", "mean",
        "amax", "reshape", "view", "scatter", "empty_like", "pixel_shuffle", "pixel_unshuffle", "detach",
    )
    unknown_unsupported = [name for name in unsupported if not any(token.lower() in name.lower() for token in allowed_unsupported)]
    status = "estimated_under_declared_convention"
    if not has_inner or unknown_unsupported or any(value == "ambiguous" for value in boundary.values()):
        status = "NOT_FOR_PAPER_missing_fused_boundary_evidence"
    costs = _mamba_component_costs(core)
    probe_slots = int(shape[0] * shape[1])
    missing_for_probe = {
        name: float(probe_slots * cost)
        for name, cost in costs.items()
        if boundary.get(name) == "inside_MambaInnerFn_missing"
    }
    try:
        package_path = str(Path(inspect.getfile(type(mamba))).parent)
    except (OSError, TypeError):
        package_path = None
    return {
        "mamba_ssm_source_file": source_file,
        "mamba_package_path": package_path,
        "mamba_forward_source": source,
        "mamba_forward_source_start_line": source_start_line,
        "mamba_inner_fn_relative_source_line": inner_relative_line,
        "mamba_inner_fn_call_source": call_source,
        "mamba_use_fast_path": uses_fast_path,
        "causal_conv1d_available": "causal_conv1d" in source_lower,
        "fvcore_total": fvcore_total,
        "fvcore_by_operator": by_operator,
        "fvcore_by_module": by_module,
        "fvcore_unsupported": unsupported,
        "fvcore_counted_in_proj_matmul_flops": counted_matmul,
        "source_projects_in_proj_before_fused": source_projects_before_fused,
        "fvcore_counted_component_flops": counted_by_component,
        "mamba_component_boundary": boundary,
        "mamba_component_cost_per_token": costs,
        "mamba_inner_fn_missing_flops_for_probe": missing_for_probe,
        "mamba_inner_fn_missing_flops_for_probe_total": sum(missing_for_probe.values()),
        "unknown_unsupported_ops": unknown_unsupported,
        "flops_status": status,
    }


def estimate_mamba_inner_flops(
    model: torch.nn.Module,
    captured_inputs: dict[int, list[tuple[int, ...]]],
    boundary_evidence: dict[str, object] | None,
) -> tuple[float | None, str, dict[str, float]]:
    """Add only component work proved missing from fvcore's fused-op trace."""

    total = 0.0
    components_total: dict[str, float] = {}
    cores = _stage4_sequence_cores(model)
    if not cores:
        return 0.0, "no Stage4 Mamba", components_total
    if not boundary_evidence or boundary_evidence.get("flops_status") != "estimated_under_declared_convention":
        return None, "NOT FOR PAPER: fused Mamba boundary evidence is incomplete", components_total
    boundary = boundary_evidence["mamba_component_boundary"]
    for index, core in enumerate(cores):
        if core.backend != "mamba_ssm":
            return None, "NOT FOR PAPER: non-mamba_ssm backend cannot use the MambaInnerFn estimate", components_total
        costs = _mamba_component_costs(core)
        missing_components = {name: cost for name, cost in costs.items() if boundary.get(name) == "inside_MambaInnerFn_missing"}
        if not missing_components:
            return None, "NOT FOR PAPER: no verified missing MambaInnerFn components", components_total
        for shape in captured_inputs.get(index, []):
            if len(shape) != 3 or shape[2] != core.dim:
                raise RuntimeError(
                    f"Captured Stage4 Mamba shape {shape} is incompatible with core d_model={core.dim}."
                )
            slots = shape[0] * shape[1]
            for name, cost in missing_components.items():
                value = float(slots * cost)
                total += value
                components_total[name] = components_total.get(name, 0.0) + value
    return total, "estimated missing fused MambaInnerFn work only; one multiply-add=one FLOP", components_total


def append_machine_report(path: str | None, row: dict[str, object], *, json_output: bool) -> None:
    if path is None:
        return
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    if json_output:
        old = json.loads(target.read_text(encoding="utf-8")) if target.exists() else []
        if not isinstance(old, list): raise ValueError("--report_json target must contain a JSON list.")
        old.append(row); target.write_text(json.dumps(old, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return
    old_rows: list[dict[str, str]] = []
    if target.exists():
        with target.open(newline="", encoding="utf-8") as handle: old_rows = list(csv.DictReader(handle))
    fields = sorted(set(row) | {key for item in old_rows for key in item})
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(old_rows); writer.writerow(row)


def find_module(model: torch.nn.Module, module_name: str) -> torch.nn.Module | None:
    """Resolve a dotted module path without assuming every model has it."""

    current: object = model
    for part in module_name.split("."):
        if not isinstance(current, torch.nn.Module) or not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current if isinstance(current, torch.nn.Module) else None


def measure_flops_by_module(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    inference_path: str,
) -> tuple[float, Counter[str], dict[str, float]]:
    """Profile one explicit graph while retaining normal module paths."""

    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError as error:
        raise RuntimeError(
            "Module FLOP measurement requires fvcore. Install it in the training environment with: "
            "pip install fvcore"
        ) from error

    def analyze(module: torch.nn.Module) -> tuple[float, Counter[str], dict[str, float]]:
        analysis = FlopCountAnalysis(module, (image,))
        analysis.unsupported_ops_warnings(False)
        analysis.uncalled_modules_warnings(False)
        total_flops = float(analysis.total())
        unsupported = Counter({str(name): int(count) for name, count in analysis.unsupported_ops().items()})
        return total_flops, unsupported, {str(name): float(value) for name, value in analysis.by_module().items()}

    wrappers: dict[str, torch.nn.Module] = {
        "full": _FullMultiOutputModel(model),
        "mask-effective": _MaskOutputEffectiveModel(model),
        "mask-only": _MaskOnlyModel(model),
    }
    try:
        wrapper = wrappers[inference_path]
    except KeyError as error:
        raise ValueError(f"Unsupported FLOP inference path: {inference_path!r}.") from error
    total_flops, unsupported, wrapped_by_module = analyze(wrapper)
    return (
        total_flops,
        unsupported,
        {name.removeprefix("model."): flops for name, flops in wrapped_by_module.items()},
    )


def print_module_flops(model: torch.nn.Module, image: torch.Tensor, *, inference_path: str) -> None:
    """Print selected registered-module FLOPs, retaining unsupported operators."""

    print(f"\nModule FLOPs (fvcore {inference_path} path)")
    try:
        total_flops, unsupported, by_module = measure_flops_by_module(
            model, image, inference_path=inference_path
        )
    except Exception as error:
        print(f"Module FLOPs: unavailable ({type(error).__name__}: {error})")
        return

    print(f"Module-analysis total FLOPs: {total_flops / 1e9:.4f} G")
    print(f"Module-analysis unsupported operators: {format_unsupported(unsupported)}")
    if unsupported:
        print("WARNING: module-analysis FLOPs are a lower bound because fvcore omitted the listed operators.")
    print("| Module | FLOPs (fvcore) | Share of total |")
    print("|---|---:|---:|")
    for module_name in FLOP_MODULE_NAMES:
        if find_module(model, module_name) is None:
            print(f"| {module_name} | N/A | N/A |")
            continue
        analysis_name = MODULE_ANALYSIS_ALIASES.get(module_name, module_name)
        module_flops = by_module.get(analysis_name, 0.0)
        share = 100.0 * module_flops / total_flops if total_flops > 0 else 0.0
        print(f"| {module_name} | {module_flops / 1e9:.4f} G | {share:.2f}% |")


def benchmark_modules_cuda(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
    amp: str,
    mask_only: bool = False,
) -> tuple[float, dict[str, float | None]] | None:
    """Measure selected module forwards with CUDA events on a fixed GPU input."""

    if device.type != "cuda":
        return None

    modules = {name: find_module(model, name) for name in TIMED_MODULE_NAMES}
    inference_model: torch.nn.Module = _MaskOnlyModel(model) if mask_only else model
    handles: list[torch.utils.hooks.RemovableHandle] = []
    pending: dict[str, list[torch.cuda.Event]] = {name: [] for name in TIMED_MODULE_NAMES}
    events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
        name: [] for name in TIMED_MODULE_NAMES
    }
    total_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    state = {"recording": False}

    def make_pre_hook(name: str):
        def pre_hook(_: torch.nn.Module, __: tuple[object, ...]) -> None:
            if not state["recording"]:
                return
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            pending[name].append(start)

        return pre_hook

    def make_post_hook(name: str):
        def post_hook(_: torch.nn.Module, __: tuple[object, ...], ___: object) -> None:
            if not state["recording"]:
                return
            start = pending[name].pop()
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            events[name].append((start, end))

        return post_hook

    try:
        for name, module in modules.items():
            if module is not None:
                handles.append(module.register_forward_pre_hook(make_pre_hook(name)))
                handles.append(module.register_forward_hook(make_post_hook(name)))

        with torch.cuda.device(device), torch.inference_mode(), autocast_context(device, amp):
            for _ in range(warmup):
                inference_model(image)
            synchronize(device)

            state["recording"] = True
            for _ in range(iterations):
                total_start = torch.cuda.Event(enable_timing=True)
                total_start.record()
                inference_model(image)
                total_end = torch.cuda.Event(enable_timing=True)
                total_end.record()
                total_events.append((total_start, total_end))
            state["recording"] = False
            synchronize(device)
    finally:
        for handle in handles:
            handle.remove()

    total_latency_ms = sum(start.elapsed_time(end) for start, end in total_events) / len(total_events)
    module_latency_ms: dict[str, float | None] = {}
    for name in TIMED_MODULE_NAMES:
        recorded_events = events[name]
        module_latency_ms[name] = (
            sum(start.elapsed_time(end) for start, end in recorded_events) / len(recorded_events)
            if recorded_events
            else None
        )
    return total_latency_ms, module_latency_ms


def print_module_timings(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
    amp: str,
    mask_only: bool = False,
) -> None:
    """Print CUDA-event module timing or explicit N/A entries on CPU."""

    print("\nCUDA module forward timing")
    measurements = benchmark_modules_cuda(
        model,
        image,
        device=device,
        warmup=warmup,
        iterations=iterations,
        amp=amp,
        mask_only=mask_only,
    )
    if measurements is None:
        print("CUDA module timing: N/A (requires --device cuda)")
        print("| Module | Mean forward latency | Share of model forward |")
        print("|---|---:|---:|")
        for module_name in TIMED_MODULE_NAMES:
            print(f"| {module_name} | N/A | N/A |")
        return

    total_latency_ms, module_latencies = measurements
    print(
        f"Model forward latency (CUDA events; warmup={warmup}, iterations={iterations}, {amp}): "
        f"{total_latency_ms:.3f} ms"
    )
    print("Note: parent and child module rows overlap and their shares must not be summed.")
    print("| Module | Mean forward latency | Share of model forward |")
    print("|---|---:|---:|")
    for module_name in TIMED_MODULE_NAMES:
        module_latency_ms = module_latencies[module_name]
        if module_latency_ms is None:
            print(f"| {module_name} | N/A | N/A |")
            continue
        share = 100.0 * module_latency_ms / total_latency_ms if total_latency_ms > 0 else 0.0
        print(f"| {module_name} | {module_latency_ms:.3f} ms | {share:.2f}% |")


def format_unsupported(unsupported: Counter[str]) -> str:
    if not unsupported:
        return "none"
    return ", ".join(f"{name} x{count}" for name, count in sorted(unsupported.items()))


def describe_mamba_execution(model: torch.nn.Module) -> list[str]:
    """Report instantiated Mamba paths without changing complexity accounting."""

    backend_info = model.mamba_backend_info() if hasattr(model, "mamba_backend_info") else {}
    early_rebase_descriptions: list[str] = []
    if bool(getattr(model, "use_haar_topology_preserving_rebase", False)):
        early_rebase_descriptions.extend(
            (
                "Early rebase | DPER-HTPR: fixed Haar LL/LH/HL/HH + gated half-resolution HF injection",
                "SPCR | disabled",
            )
        )
    if bool(getattr(model, "use_narrow_dual_resolution_stage1", False)):
        early_rebase_descriptions.append(
            "Early rebase | DPER-NDRS: narrow full-resolution DW3x3 detail residual + one-way detail-to-semantic exchange"
        )
    if bool(getattr(model, "use_full_resolution_joint_exchange", False)):
        early_rebase_descriptions.append(
            "Early rebase | DPER-FJSE: dense full-resolution 3x3 joint modeling + phase-preserving one-way exchange"
        )
        early_rebase_descriptions.append("NDRS | disabled; SPCR | disabled; HTPR | disabled")
    pfrs = getattr(model, "partial_full_resolution_stage1", None)
    if pfrs is not None:
        early_rebase_descriptions.extend(
            (
                "Early full-resolution stage | DPER-PFRS: LocalCrackBlock 16ch",
                "High-to-low exchange | residual-only PixelUnshuffle ×2, 64→40",
            )
        )
    early_escs = getattr(model, "early_serpentine_connectivity_ssm", None)
    if early_escs is not None:
        early_rebase_descriptions.extend(
            (
                "Early connectivity | DPER-ESCS: horizontal snake ×2 + diagonal snake ×2",
                f"Early Mamba | {early_escs.backend} ×{early_escs.instantiated_core_count} shared core | "
                f"traversals: {early_escs.traversal_count} batched",
            )
        )
    if not backend_info:
        return [*early_rebase_descriptions, "Mamba execution: not applicable"]
    summary_parts: list[str] = []
    for stage_name, backends in backend_info.items():
        if backends == ["disabled"]:
            summary_parts.append(f"{stage_name}: disabled")
            continue
        counts = Counter(backends)
        backend_summary = ", ".join(
            f"{backend} ×{count}" for backend, count in counts.items()
        )
        summary_parts.append(f"{stage_name}: {backend_summary}")

    descriptions: list[str] = [*early_rebase_descriptions, f"Mamba backend | {'; '.join(summary_parts)}"]
    for stage_name, backends in backend_info.items():
        if backends == ["disabled"]:
            continue
        stage = getattr(model, stage_name, None)
        if stage_name == "early_connectivity":
            descriptions.append(
                f"Mamba execution | early_connectivity: {len(backends)} shared core(s) | traversals: 4 batched"
            )
            continue
        route_mamba = getattr(stage, "route_mamba", None)
        direction_owner = route_mamba if route_mamba is not None else stage
        directions = getattr(direction_owner, "active_mamba_directions", ())
        spatial_reduction = int(getattr(direction_owner, "spatial_reduction", 1))
        channel_description = ""
        if hasattr(direction_owner, "mamba_channels"):
            mamba_channels = int(getattr(direction_owner, "mamba_channels"))
            route_head = getattr(direction_owner, "route_head", None)
            full_channels = mamba_channels
            if isinstance(route_head, torch.nn.Sequential) and len(route_head) > 0:
                full_channels = int(getattr(route_head[0], "in_channels", mamba_channels))
            channel_description = f" | channels: {mamba_channels}/{full_channels}"
        if bool(getattr(direction_owner, "shared_core", False)):
            descriptions.append(
                f"Mamba execution | {stage_name}: {len(backends)} shared core(s) | "
                f"traversals: {', '.join(directions)}"
            )
            continue
        descriptions.append(
            f"Mamba execution | {stage_name}: {len(backends)} instantiated path(s) "
            f"({', '.join(directions)}) | backends: {', '.join(backends)} | "
            f"spatial reduction: {spatial_reduction}{channel_description}"
        )
        local_compensations = getattr(direction_owner, "local_compensations", None)
        if local_compensations is not None:
            descriptions.append(
                f"Local directional compensation | {stage_name}: "
                + ", ".join(local_compensations.keys())
            )
        # TopoRouteBlock keeps its three-branch fusion contract when the
        # Transformer is disabled: ``global_branch`` is then an Identity
        # branch rather than a silently omitted input.  Report that state so
        # a no-Transformer complexity run is unambiguous.
        if stage_name == "stage4":
            global_branch = getattr(stage, "global_branch", None)
            if isinstance(global_branch, torch.nn.Identity):
                descriptions.append("Transformer execution | stage4: disabled (Identity)")
            elif global_branch is not None:
                descriptions.append(
                    "Transformer execution | stage4: enabled "
                    f"({type(global_branch).__name__})"
                )
    return descriptions


def main() -> int:
    args = parse_args()
    if args.mask_only and args.inference_path not in (None, "mask-only"):
        raise ValueError("--mask_only can only be combined with --inference_path mask-only.")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be at least 1.")
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative.")
    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1.")
    if args.benchmark_repeats < 1:
        raise ValueError("--benchmark_repeats must be at least 1.")
    height, width = args.img_size
    if height < 1 or width < 1:
        raise ValueError("--img_size values must be positive.")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_name, model = load_model(args.config, device)
    strict_stage4 = audit_strict_stage4_deletion(model)
    escs_config = "ESCS" in str(args.config).upper()
    pfrs_config = "PFRS" in str(args.config).upper()
    msdr_config = "MSDR" in str(args.config).upper()
    dsmh_config = "DSMH" in str(args.config).upper()
    fwrmh_config = "FWRMH" in str(args.config).upper()
    dafu1_config = "DAFU1" in str(args.config).upper()
    # Never infer architecture from a directory name.  In particular the
    # lightweight-ablation directory deliberately contains "dpsm" while L0,
    # L1, L2 and L4 are not frozen DPSM80 models.
    cpdpsm_config = bool(getattr(model, "use_common_private_direction_partitioned_shared_mamba", False))
    dpsm_config = (
        bool(getattr(model, "use_direction_partitioned_shared_mamba", False))
        and not cpdpsm_config and not fwrmh_config and not dafu1_config
    )
    pntd_config = "PNTD" in str(args.config).upper()
    ntd_config = "NTD" in str(args.config).upper() and not pntd_config
    rsdm_config = (
        "RSDM" in str(args.config).upper()
        and not ntd_config
        and not pntd_config
        and not dpsm_config
        and not cpdpsm_config
        and not fwrmh_config
        and not dafu1_config
    )
    early_escs = getattr(model, "early_serpentine_connectivity_ssm", None)
    pfrs = getattr(model, "partial_full_resolution_stage1", None)
    msdr_stage3_route = getattr(getattr(model, "stage3", None), "route_mamba", None)
    msdr_stage4_route = getattr(getattr(model, "stage4", None), "route_mamba", None)
    dsmh = getattr(getattr(model, "decoder", None), "mask_head", None)
    rsdm = getattr(model, "rsdm_stage1", None)
    ntd = getattr(getattr(model, "decoder", None), "narrow_transport_decoder", None)
    pntd = getattr(getattr(model, "decoder", None), "partial_narrow_transport_decoder", None)
    dpsm = getattr(getattr(model, "stage4", None), "route_mamba", None)
    if escs_config and (not bool(getattr(model, "use_early_serpentine_connectivity_ssm", False)) or early_escs is None):
        raise RuntimeError(
            "ESCS configuration validation failed: use_early_serpentine_connectivity_ssm must be true "
            "and early_serpentine_connectivity_ssm must be present."
        )
    if pfrs_config and (
        not bool(getattr(model, "use_partial_full_resolution_stage1", False))
        or pfrs is None
        or early_escs is not None
    ):
        raise RuntimeError(
            "PFRS configuration validation failed: use_partial_full_resolution_stage1 must be true, "
            "partial_full_resolution_stage1 must be present, and ESCS must be absent."
        )
    if msdr_config:
        failed_candidate_flags = (
            "use_subpixel_phase_consistency_refiner",
            "use_haar_topology_preserving_rebase",
            "use_narrow_dual_resolution_stage1",
            "use_full_resolution_joint_exchange",
            "use_early_serpentine_connectivity_ssm",
            "use_partial_full_resolution_stage1",
        )
        invalid_failed_candidates = [name for name in failed_candidate_flags if bool(getattr(model, name, False))]
        if (
            not bool(getattr(model, "use_multiscale_directional_reallocation", False))
            or not bool(getattr(model, "route_stage3", False))
            or not bool(getattr(model, "route_stage4", False))
            or msdr_stage3_route is None
            or msdr_stage4_route is None
            or tuple(msdr_stage3_route.active_mamba_directions) != ("horizontal", "vertical")
            or tuple(msdr_stage4_route.active_mamba_directions) != ("diagonal", "anti_diagonal")
            or msdr_stage3_route.active_mamba_path_count != 2
            or msdr_stage4_route.active_mamba_path_count != 2
            or invalid_failed_candidates
        ):
            raise RuntimeError(
                "MSDR configuration validation failed: expected isolated Stage3 horizontal/vertical ×2 and "
                "Stage4 diagonal/anti_diagonal ×2 routed blocks with all failed candidates absent."
            )
        msdr_backends = [*msdr_stage3_route.backends, *msdr_stage4_route.backends]
        if msdr_backends != ["mamba_ssm"] * 4:
            raise RuntimeError(
                "MSDR formal complexity measurement requires four mamba_ssm cores; got "
                + ", ".join(msdr_backends)
                + "."
            )
    if dsmh_config:
        decoder = getattr(model, "decoder", None)
        stage4_backends = model.mamba_backend_info().get("stage4", []) if hasattr(model, "mamba_backend_info") else []
        failed_candidate_flags = (
            "use_subpixel_phase_consistency_refiner",
            "use_haar_topology_preserving_rebase",
            "use_narrow_dual_resolution_stage1",
            "use_full_resolution_joint_exchange",
            "use_early_serpentine_connectivity_ssm",
            "use_partial_full_resolution_stage1",
            "use_multiscale_directional_reallocation",
        )
        failed_candidates = [name for name in failed_candidate_flags if bool(getattr(model, name, False))]
        required_module_names = {
            "decoder.mask_head",
            "decoder.mask_head.depthwise",
            "decoder.mask_head.pointwise",
            "decoder.mask_head.output",
        }
        registered_module_names = set(dict(model.named_modules()))
        if (
            not bool(getattr(model, "use_depthwise_separable_mask_head", False))
            or decoder is None
            or not isinstance(dsmh, DepthwiseSeparableMaskHead)
            or not isinstance(getattr(decoder, "up1", None), LowResolutionSubpixelSeparableConcatUpBlock)
            or not required_module_names.issubset(registered_module_names)
            or failed_candidates
        ):
            raise RuntimeError(
                "DSMH configuration validation failed: expected an isolated DepthwiseSeparableMaskHead, "
                "unchanged LSSCF up1, registered child modules, and no failed DPER candidates."
            )
        depthwise = dsmh.depthwise.block[0]
        pointwise = dsmh.pointwise.block[0]
        output_head = dsmh.output
        if not (
            isinstance(depthwise, torch.nn.Conv2d)
            and depthwise.in_channels == depthwise.out_channels == depthwise.groups == 40
            and depthwise.kernel_size == (3, 3)
            and isinstance(pointwise, torch.nn.Conv2d)
            and pointwise.in_channels == pointwise.out_channels == 40
            and pointwise.groups == 1
            and pointwise.kernel_size == (1, 1)
            and isinstance(output_head, torch.nn.Conv2d)
            and output_head.in_channels == 40
            and output_head.out_channels == 1
            and output_head.kernel_size == (1, 1)
        ):
            raise RuntimeError("DSMH configuration validation failed: mask-head channel/group/kernel contract changed.")
        dense_spatial = [
            module
            for module in dsmh.modules()
            if isinstance(module, torch.nn.Conv2d)
            and module.in_channels == module.out_channels == 40
            and module.kernel_size == (3, 3)
            and module.groups == 1
        ]
        if dense_spatial:
            raise RuntimeError("DSMH configuration validation failed: dense 40→40 3×3 mask convolution remains.")
        if stage4_backends != ["mamba_ssm"] * 4:
            raise RuntimeError(
                "DSMH formal complexity measurement requires Stage4 mamba_ssm ×4; got "
                + ", ".join(stage4_backends)
                + "."
            )
    if rsdm_config:
        decoder = getattr(model, "decoder", None)
        stage4_backends = model.mamba_backend_info().get("stage4", []) if hasattr(model, "mamba_backend_info") else []
        failed_candidate_flags = (
            "use_subpixel_phase_consistency_refiner",
            "use_haar_topology_preserving_rebase",
            "use_narrow_dual_resolution_stage1",
            "use_full_resolution_joint_exchange",
            "use_early_serpentine_connectivity_ssm",
            "use_partial_full_resolution_stage1",
            "use_multiscale_directional_reallocation",
            "use_depthwise_separable_mask_head",
        )
        failed_candidates = [name for name in failed_candidate_flags if bool(getattr(model, name, False))]
        rsdm_count = sum(
            isinstance(module, RIFTStructuralDirectionalModelingBlock) for module in model.modules()
        )
        dense_mask_conv = None
        mask_head = getattr(decoder, "mask_head", None)
        if isinstance(mask_head, torch.nn.Sequential) and len(mask_head) > 0:
            first = mask_head[0]
            block = getattr(first, "block", None)
            if isinstance(block, torch.nn.Sequential) and len(block) > 0:
                dense_mask_conv = block[0]
        if (
            not bool(getattr(model, "use_rsdm_stage1", False))
            or not isinstance(rsdm, RIFTStructuralDirectionalModelingBlock)
            or rsdm_count != 1
            or decoder is None
            or not isinstance(getattr(decoder, "up1", None), LowResolutionSubpixelSeparableConcatUpBlock)
            or not isinstance(dense_mask_conv, torch.nn.Conv2d)
            or dense_mask_conv.in_channels != 40
            or dense_mask_conv.out_channels != 40
            or dense_mask_conv.kernel_size != (3, 3)
            or dense_mask_conv.groups != 1
            or failed_candidates
        ):
            raise RuntimeError(
                "RSDM configuration validation failed: expected one isolated half-resolution RSDM, "
                "unchanged LSSCF and dense mask head, and no failed DPER candidates."
            )
        if not (
            rsdm.channels == 40
            and rsdm.kernel_size == 13
            and rsdm.dilation == 3
            and rsdm.hidden_ratio == 1.0
            and rsdm.horizontal.groups == rsdm.vertical.groups == rsdm.diagonal.groups == 40
        ):
            raise RuntimeError("RSDM configuration validation failed: fixed channel/kernel/dilation contract changed.")
        if stage4_backends != ["mamba_ssm"] * 4:
            raise RuntimeError(
                "RSDM formal complexity measurement requires Stage4 mamba_ssm ×4; got "
                + ", ".join(stage4_backends)
                + "."
            )
    if ntd_config:
        decoder = getattr(model, "decoder", None)
        stage4_backends = model.mamba_backend_info().get("stage4", []) if hasattr(model, "mamba_backend_info") else []
        mask_head = getattr(decoder, "mask_head", None)
        dense_mask_conv = None
        if isinstance(mask_head, torch.nn.Sequential) and len(mask_head) > 0:
            block = getattr(mask_head[0], "block", None)
            if isinstance(block, torch.nn.Sequential) and len(block) > 0:
                dense_mask_conv = block[0]
        failed_candidate_flags = (
            "use_subpixel_phase_consistency_refiner",
            "use_haar_topology_preserving_rebase",
            "use_narrow_dual_resolution_stage1",
            "use_full_resolution_joint_exchange",
            "use_early_serpentine_connectivity_ssm",
            "use_partial_full_resolution_stage1",
            "use_multiscale_directional_reallocation",
            "use_depthwise_separable_mask_head",
        )
        failed_candidates = [name for name in failed_candidate_flags if bool(getattr(model, name, False))]
        if (
            not bool(getattr(model, "use_narrow_transport_decoder", False))
            or not bool(getattr(model, "use_rsdm_stage1", False))
            or not isinstance(rsdm, RIFTStructuralDirectionalModelingBlock)
            or not isinstance(ntd, NarrowTransportDecoder)
            or ntd.transport_dim not in (16, 24)
            or decoder is None
            or any(getattr(decoder, name, None) is not None for name in ("proj4", "up3", "up2", "up1", "refine"))
            or not isinstance(dense_mask_conv, torch.nn.Conv2d)
            or dense_mask_conv.in_channels != dense_mask_conv.out_channels
            or dense_mask_conv.in_channels != 40
            or dense_mask_conv.kernel_size != (3, 3)
            or dense_mask_conv.groups != 1
            or failed_candidates
        ):
            raise RuntimeError(
                "NTD configuration validation failed: expected one preserved RSDM, only the narrow transport "
                "decoder, no registered LSSCF transport/refine, and the original dense mask head."
            )
        if stage4_backends != ["mamba_ssm"] * 4:
            raise RuntimeError(
                "NTD formal complexity measurement requires Stage4 mamba_ssm ×4; got "
                + ", ".join(stage4_backends)
                + "."
            )
    if pntd_config:
        decoder = getattr(model, "decoder", None)
        stage4_backends = model.mamba_backend_info().get("stage4", []) if hasattr(model, "mamba_backend_info") else []
        mask_head = getattr(decoder, "mask_head", None)
        dense_mask_conv = None
        if isinstance(mask_head, torch.nn.Sequential) and len(mask_head) > 0:
            block = getattr(mask_head[0], "block", None)
            if isinstance(block, torch.nn.Sequential) and len(block) > 0:
                dense_mask_conv = block[0]
        failed_candidate_flags = (
            "use_narrow_transport_decoder",
            "use_subpixel_phase_consistency_refiner",
            "use_haar_topology_preserving_rebase",
            "use_narrow_dual_resolution_stage1",
            "use_full_resolution_joint_exchange",
            "use_early_serpentine_connectivity_ssm",
            "use_partial_full_resolution_stage1",
            "use_multiscale_directional_reallocation",
            "use_depthwise_separable_mask_head",
        )
        failed_candidates = [name for name in failed_candidate_flags if bool(getattr(model, name, False))]
        rsdm_count = sum(isinstance(module, RIFTStructuralDirectionalModelingBlock) for module in model.modules())
        if (
            not bool(getattr(model, "use_partial_narrow_transport_decoder", False))
            or not bool(getattr(model, "use_rsdm_stage1", False))
            or not isinstance(rsdm, RIFTStructuralDirectionalModelingBlock)
            or rsdm_count != 1
            or not isinstance(pntd, PartialNarrowTransportDecoder)
            or pntd.transport_dim != 24
            or decoder is None
            or getattr(decoder, "up3", None) is not None
            or getattr(decoder, "up2", None) is not None
            or not isinstance(getattr(decoder, "up1", None), LowResolutionSubpixelSeparableConcatUpBlock)
            or getattr(decoder, "refine", None) is None
            or not isinstance(dense_mask_conv, torch.nn.Conv2d)
            or dense_mask_conv.in_channels != dense_mask_conv.out_channels
            or dense_mask_conv.in_channels != 40
            or dense_mask_conv.kernel_size != (3, 3)
            or dense_mask_conv.groups != 1
            or failed_candidates
        ):
            raise RuntimeError(
                "PNTD validation failed: expected one RSDM, isolated F4/F3 narrow transport, preserved "
                "native F2 fusion children, unchanged LSSCF up1/refine/dense head, and no failed candidates."
            )
        if stage4_backends != ["mamba_ssm"] * 4:
            raise RuntimeError(
                "PNTD formal complexity measurement requires Stage4 mamba_ssm ×4; got "
                + ", ".join(stage4_backends)
                + "."
            )
    if dpsm_config:
        decoder = getattr(model, "decoder", None)
        stage4 = getattr(model, "stage4", None)
        stage4_backends = model.mamba_backend_info().get("stage4", []) if hasattr(model, "mamba_backend_info") else []
        mask_head = getattr(decoder, "mask_head", None)
        dense_mask_conv = None
        if isinstance(mask_head, torch.nn.Sequential) and len(mask_head) > 0:
            block = getattr(mask_head[0], "block", None)
            if isinstance(block, torch.nn.Sequential) and len(block) > 0:
                dense_mask_conv = block[0]
        failed_flags = (
            "use_narrow_transport_decoder",
            "use_partial_narrow_transport_decoder",
            "use_subpixel_phase_consistency_refiner",
            "use_haar_topology_preserving_rebase",
            "use_narrow_dual_resolution_stage1",
            "use_full_resolution_joint_exchange",
            "use_early_serpentine_connectivity_ssm",
            "use_partial_full_resolution_stage1",
            "use_multiscale_directional_reallocation",
            "use_depthwise_separable_mask_head",
        )
        failed = [name for name in failed_flags if bool(getattr(model, name, False))]
        rsdm_count = sum(isinstance(module, RIFTStructuralDirectionalModelingBlock) for module in model.modules())
        core_count = sum(isinstance(module, SequenceMambaCore) for module in dpsm.modules()) if isinstance(dpsm, torch.nn.Module) else 0
        if (
            not bool(getattr(model, "use_direction_partitioned_shared_mamba", False))
            or not isinstance(dpsm, DirectionPartitionedSharedMamba)
            or dpsm.dim != 320
            or dpsm.group_channels != 80
            or tuple(dpsm.active_mamba_directions) != DirectionPartitionedSharedMamba.directions
            or dpsm.spatial_reduction != 1
            or dpsm.instantiated_core_count != 1
            or core_count != 1
            or hasattr(dpsm, "paths")
            or hasattr(dpsm, "local_compensations")
            or not (isinstance(getattr(stage4, "global_branch", None), TransformerGlobalBlock)
                    or (isinstance(stage4, Stage4LocalDPSM80) and strict_stage4.get("stage4_strict_variant") == "local_dpsm80"))
            or not bool(getattr(model, "use_rsdm_stage1", False))
            or rsdm_count != 1
            or decoder is None
            or not isinstance(getattr(decoder, "up3", None), LowResolutionSubpixelSeparableConcatUpBlock)
            or getattr(decoder, "narrow_transport_decoder", None) is not None
            or getattr(decoder, "partial_narrow_transport_decoder", None) is not None
            or not isinstance(dense_mask_conv, torch.nn.Conv2d)
            or dense_mask_conv.in_channels != dense_mask_conv.out_channels
            or dense_mask_conv.in_channels != 40
            or dense_mask_conv.kernel_size != (3, 3)
            or dense_mask_conv.groups != 1
            or failed
        ):
            raise RuntimeError(
                "DPSM80 validation failed: expected one 80-channel shared core, four full-grid direction "
                "groups, retained Transformer/RSDM/LSSCF/dense head, and no old/failure paths."
            )
        if stage4_backends != ["mamba_ssm"]:
            raise RuntimeError(
                "DPSM80 formal complexity measurement requires one shared mamba_ssm core; got "
                + ", ".join(stage4_backends)
                + "."
            )
    if fwrmh_config:
        decoder = getattr(model, "decoder", None)
        stage4 = getattr(model, "stage4", None)
        stage4_backends = model.mamba_backend_info().get("stage4", []) if hasattr(model, "mamba_backend_info") else []
        failed_flags = (
            "use_common_private_direction_partitioned_shared_mamba",
            "use_narrow_transport_decoder",
            "use_partial_narrow_transport_decoder",
            "use_subpixel_phase_consistency_refiner",
            "use_haar_topology_preserving_rebase",
            "use_narrow_dual_resolution_stage1",
            "use_full_resolution_joint_exchange",
            "use_early_serpentine_connectivity_ssm",
            "use_partial_full_resolution_stage1",
            "use_multiscale_directional_reallocation",
            "use_depthwise_separable_mask_head",
        )
        failed = [name for name in failed_flags if bool(getattr(model, name, False))]
        rsdm_count = sum(isinstance(module, RIFTStructuralDirectionalModelingBlock) for module in model.modules())
        core_count = sum(isinstance(module, SequenceMambaCore) for module in dpsm.modules()) if isinstance(dpsm, torch.nn.Module) else 0
        if (
            not bool(getattr(model, "use_full_width_residual_mask_head", False))
            or not bool(getattr(model, "use_direction_partitioned_shared_mamba", False))
            or not isinstance(dpsm, DirectionPartitionedSharedMamba)
            or dpsm.dim != 320
            or dpsm.group_channels != 80
            or dpsm.instantiated_core_count != 1
            or core_count != 1
            or tuple(dpsm.active_mamba_directions) != DirectionPartitionedSharedMamba.directions
            or not isinstance(getattr(stage4, "global_branch", None), TransformerGlobalBlock)
            or not bool(getattr(model, "use_rsdm_stage1", False))
            or rsdm_count != 1
            or decoder is None
            or not isinstance(getattr(decoder, "mask_head", None), FullWidthResidualMaskHead)
            or getattr(decoder.mask_head, "channels", None) != 40
            or getattr(decoder.mask_head, "hidden_channels", None) != 16
            or not isinstance(getattr(decoder, "up3", None), LowResolutionSubpixelSeparableConcatUpBlock)
            or getattr(decoder, "narrow_transport_decoder", None) is not None
            or getattr(decoder, "partial_narrow_transport_decoder", None) is not None
            or failed
        ):
            raise RuntimeError(
                "DPSM80-FWRMH validation failed: expected isolated 40→16 depthwise residual correction, "
                "the DPSM80 shared Stage4 core, and unchanged RSDM/LSSCF/auxiliary heads."
            )
        dense_3x3 = [
            module
            for module in decoder.mask_head.modules()
            if isinstance(module, torch.nn.Conv2d)
            and module.in_channels == module.out_channels == 40
            and module.kernel_size == (3, 3)
            and module.groups == 1
        ]
        if dense_3x3:
            raise RuntimeError("DPSM80-FWRMH validation failed: the original dense 40→40 3×3 mask convolution remains.")
        if stage4_backends != ["mamba_ssm"]:
            raise RuntimeError(
                "DPSM80-FWRMH formal complexity measurement requires Stage4 mamba_ssm ×1; got "
                + ", ".join(stage4_backends)
                + "."
            )
    if dafu1_config:
        decoder = getattr(model, "decoder", None)
        stage4 = getattr(model, "stage4", None)
        stage4_backends = model.mamba_backend_info().get("stage4", []) if hasattr(model, "mamba_backend_info") else []
        failed_flags = (
            "use_common_private_direction_partitioned_shared_mamba",
            "use_narrow_transport_decoder",
            "use_partial_narrow_transport_decoder",
            "use_subpixel_phase_consistency_refiner",
            "use_haar_topology_preserving_rebase",
            "use_narrow_dual_resolution_stage1",
            "use_full_resolution_joint_exchange",
            "use_early_serpentine_connectivity_ssm",
            "use_partial_full_resolution_stage1",
            "use_multiscale_directional_reallocation",
            "use_depthwise_separable_mask_head",
            "use_full_width_residual_mask_head",
        )
        failed = [name for name in failed_flags if bool(getattr(model, name, False))]
        rsdm_count = sum(isinstance(module, RIFTStructuralDirectionalModelingBlock) for module in model.modules())
        core_count = sum(isinstance(module, SequenceMambaCore) for module in dpsm.modules()) if isinstance(dpsm, torch.nn.Module) else 0
        head = getattr(decoder, "mask_head", None) if decoder is not None else None
        dense_mask_conv = None
        if isinstance(head, torch.nn.Sequential) and len(head) > 0:
            block = getattr(head[0], "block", None)
            if isinstance(block, torch.nn.Sequential) and len(block) > 0:
                dense_mask_conv = block[0]
        if (
            not bool(getattr(model, "use_difference_aware_up1_fusion", False))
            or not bool(getattr(model, "use_direction_partitioned_shared_mamba", False))
            or not isinstance(dpsm, DirectionPartitionedSharedMamba)
            or dpsm.dim != 320
            or dpsm.group_channels != 80
            or dpsm.instantiated_core_count != 1
            or core_count != 1
            or tuple(dpsm.active_mamba_directions) != DirectionPartitionedSharedMamba.directions
            or not isinstance(getattr(stage4, "global_branch", None), TransformerGlobalBlock)
            or not bool(getattr(model, "use_rsdm_stage1", False))
            or rsdm_count != 1
            or decoder is None
            or not isinstance(getattr(decoder, "up1", None), DifferenceAwareSubpixelUpBlock)
            or not isinstance(getattr(decoder, "up2", None), LowResolutionSubpixelSeparableConcatUpBlock)
            or not isinstance(getattr(decoder, "up3", None), LowResolutionSubpixelSeparableConcatUpBlock)
            or getattr(decoder.up1, "out_channels", None) != 40
            or getattr(decoder.up1, "delta_depthwise", None).groups != 40
            or not isinstance(dense_mask_conv, torch.nn.Conv2d)
            or dense_mask_conv.in_channels != 40
            or dense_mask_conv.out_channels != 40
            or dense_mask_conv.kernel_size != (3, 3)
            or dense_mask_conv.groups != 1
            or getattr(decoder, "narrow_transport_decoder", None) is not None
            or getattr(decoder, "partial_narrow_transport_decoder", None) is not None
            or failed
        ):
            raise RuntimeError(
                "DPSM80-DAFU1 validation failed: expected only full-width difference-aware Up1, retained "
                "DPSM80/RSDM/LSSCF Up2-Up3/dense mask head, and no failed paths."
            )
        if stage4_backends != ["mamba_ssm"]:
            raise RuntimeError(
                "DPSM80-DAFU1 formal complexity measurement requires Stage4 mamba_ssm ×1; got "
                + ", ".join(stage4_backends)
                + "."
            )
    if cpdpsm_config:
        decoder = getattr(model, "decoder", None)
        stage4 = getattr(model, "stage4", None)
        stage4_backends = model.mamba_backend_info().get("stage4", []) if hasattr(model, "mamba_backend_info") else []
        mask_head = getattr(decoder, "mask_head", None)
        dense_mask_conv = None
        if isinstance(mask_head, torch.nn.Sequential) and len(mask_head) > 0:
            block = getattr(mask_head[0], "block", None)
            if isinstance(block, torch.nn.Sequential) and len(block) > 0:
                dense_mask_conv = block[0]
        failed_flags = (
            "use_direction_partitioned_shared_mamba",
            "use_narrow_transport_decoder",
            "use_partial_narrow_transport_decoder",
            "use_subpixel_phase_consistency_refiner",
            "use_haar_topology_preserving_rebase",
            "use_narrow_dual_resolution_stage1",
            "use_full_resolution_joint_exchange",
            "use_early_serpentine_connectivity_ssm",
            "use_partial_full_resolution_stage1",
            "use_multiscale_directional_reallocation",
            "use_depthwise_separable_mask_head",
        )
        failed = [name for name in failed_flags if bool(getattr(model, name, False))]
        rsdm_count = sum(isinstance(module, RIFTStructuralDirectionalModelingBlock) for module in model.modules())
        core_count = sum(isinstance(module, SequenceMambaCore) for module in dpsm.modules()) if isinstance(dpsm, torch.nn.Module) else 0
        if (
            not bool(getattr(model, "use_common_private_direction_partitioned_shared_mamba", False))
            or not isinstance(dpsm, CommonPrivateDirectionPartitionedSharedMamba)
            or dpsm.dim != 320
            or dpsm.shared_channels != 64
            or dpsm.private_channels != 64
            or dpsm.per_direction_channels != 128
            or tuple(dpsm.active_mamba_directions) != CommonPrivateDirectionPartitionedSharedMamba.directions
            or dpsm.spatial_reduction != 1
            or dpsm.instantiated_core_count != 1
            or core_count != 1
            or hasattr(dpsm, "paths")
            or hasattr(dpsm, "local_compensations")
            or not isinstance(getattr(stage4, "global_branch", None), TransformerGlobalBlock)
            or not bool(getattr(model, "use_rsdm_stage1", False))
            or rsdm_count != 1
            or decoder is None
            or not isinstance(getattr(decoder, "up3", None), LowResolutionSubpixelSeparableConcatUpBlock)
            or getattr(decoder, "narrow_transport_decoder", None) is not None
            or getattr(decoder, "partial_narrow_transport_decoder", None) is not None
            or not isinstance(dense_mask_conv, torch.nn.Conv2d)
            or dense_mask_conv.in_channels != dense_mask_conv.out_channels
            or dense_mask_conv.in_channels != 40
            or dense_mask_conv.kernel_size != (3, 3)
            or dense_mask_conv.groups != 1
            or failed
        ):
            raise RuntimeError(
                "CP-DPSM64 validation failed: expected one 128-channel shared core, 64 common + four 64-channel "
                "private groups, retained Transformer/RSDM/LSSCF/dense head, and no failed paths."
            )
        if stage4_backends != ["mamba_ssm"]:
            raise RuntimeError(
                "CP-DPSM64 formal complexity measurement requires one shared mamba_ssm core; got "
                + ", ".join(stage4_backends)
                + "."
            )
    selected_path = args.inference_path or ("mask-only" if args.mask_only else "full")
    mask_only_selected = selected_path == "mask-only"
    if mask_only_selected and not callable(getattr(model, "forward_mask_only", None)):
        raise ValueError(f"{model_name} does not implement eval-only forward_mask_only(image).")
    image = torch.randn(args.batch_size, 3, height, width, device=device)
    inference_model: torch.nn.Module = _MaskOnlyModel(model) if mask_only_selected else model

    # Audit a normal forward before timing.  Core counts and Mamba sequence
    # shapes are observed from the actual modules, not inferred from YAML names.
    with torch.inference_mode():
        normal_output, captured_mamba_inputs = capture_stage4_execution(model, image)
    stage4_audit = audit_stage4_mamba(
        model,
        variant=args.experiment or Path(args.config).stem,
        expected_cores=args.expected_mamba_cores,
        outputs=normal_output,
        captured_inputs=captured_mamba_inputs,
        formal_backend=args.require_mamba_ssm,
    )
    boundary_evidence: dict[str, object] | None = None
    if captured_mamba_inputs:
        first_core = _stage4_sequence_cores(model)[0]
        first_shape = captured_mamba_inputs[0][0]
        boundary_evidence = diagnose_mamba_fused_boundary(first_core, first_shape)
        try:
            boundary_evidence["mamba_ssm_version"] = metadata.version("mamba-ssm")
        except metadata.PackageNotFoundError:
            boundary_evidence["mamba_ssm_version"] = None
        try:
            boundary_evidence["causal_conv1d_version"] = metadata.version("causal-conv1d")
        except metadata.PackageNotFoundError:
            boundary_evidence["causal_conv1d_version"] = None
        try:
            boundary_evidence["fvcore_version"] = metadata.version("fvcore")
        except metadata.PackageNotFoundError:
            boundary_evidence["fvcore_version"] = None
        if args.mamba_boundary_report:
            target = Path(args.mamba_boundary_report); target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(boundary_evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if (args.require_mamba_ssm or args.expected_mamba_cores is not None) and boundary_evidence.get("flops_status") != "estimated_under_declared_convention":
            raise RuntimeError(
                "Formal FLOP measurement stopped: actual mamba_ssm/fvcore fused-boundary evidence is insufficient; "
                f"status={boundary_evidence.get('flops_status')}"
            )
    elif args.mamba_boundary_report:
        target = Path(args.mamba_boundary_report); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"flops_status": "estimated_under_declared_convention", "mamba_fused_missing_flops_estimated": 0.0}, indent=2) + "\n", encoding="utf-8")
    mask_only_audit = validate_mask_only_equivalence(model, image, device=device, amp=args.amp)

    # Validate the exact selected inference path before passing it to fvcore.
    with torch.inference_mode():
        output = inference_model(image)
    if mask_only_selected:
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("forward_mask_only must return a dictionary containing a Tensor 'mask' prediction.")
    elif not isinstance(output, dict) or "mask" not in output:
        raise RuntimeError("The model must return a dictionary containing the 'mask' prediction.")
    del output
    if dpsm_config:
        assert isinstance(dpsm, DirectionPartitionedSharedMamba)
        expected_packed = (4 * args.batch_size, (height // 8) * (width // 8), 80)
        if dpsm.last_shared_mamba_call_count != 1 or dpsm.last_packed_shape != expected_packed:
            raise RuntimeError(
                "DPSM80 execution validation failed: expected one shared-core call with packed shape "
                f"{expected_packed}, got calls={dpsm.last_shared_mamba_call_count}, "
                f"shape={dpsm.last_packed_shape}."
            )
    if fwrmh_config:
        assert isinstance(dpsm, DirectionPartitionedSharedMamba)
        expected_packed = (4 * args.batch_size, (height // 8) * (width // 8), 80)
        if dpsm.last_shared_mamba_call_count != 1 or dpsm.last_packed_shape != expected_packed:
            raise RuntimeError(
                "DPSM80-FWRMH execution validation failed: expected one shared-core call with packed shape "
                f"{expected_packed}, got calls={dpsm.last_shared_mamba_call_count}, "
                f"shape={dpsm.last_packed_shape}."
            )
    if dafu1_config:
        assert isinstance(dpsm, DirectionPartitionedSharedMamba)
        expected_packed = (4 * args.batch_size, (height // 8) * (width // 8), 80)
        if dpsm.last_shared_mamba_call_count != 1 or dpsm.last_packed_shape != expected_packed:
            raise RuntimeError(
                "DPSM80-DAFU1 execution validation failed: expected one shared-core call with packed shape "
                f"{expected_packed}, got calls={dpsm.last_shared_mamba_call_count}, "
                f"shape={dpsm.last_packed_shape}."
            )
    if cpdpsm_config:
        assert isinstance(dpsm, CommonPrivateDirectionPartitionedSharedMamba)
        expected_packed = (4 * args.batch_size, (height // 8) * (width // 8), 128)
        if dpsm.last_shared_mamba_call_count != 1 or dpsm.last_packed_shape != expected_packed:
            raise RuntimeError(
                "CP-DPSM64 execution validation failed: expected one shared-core call with packed shape "
                f"{expected_packed}, got calls={dpsm.last_shared_mamba_call_count}, "
                f"shape={dpsm.last_packed_shape}."
            )

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if escs_config and total_params <= 5_740_751:
        raise RuntimeError(
            "ESCS configuration validation failed: parameter count did not exceed the original DPER "
            "count (5,740,751), so the ESCS module was likely not profiled."
        )
    if pfrs_config and total_params <= 5_740_751:
        raise RuntimeError(
            "PFRS configuration validation failed: parameter count did not exceed the original DPER "
            "count (5,740,751), so the PFRS module was likely not profiled."
        )
    if msdr_config and total_params == 5_740_751:
        raise RuntimeError(
            "MSDR configuration validation failed: parameter count equals original DPER, so the candidate "
            "Stage3/Stage4 replacement was likely not profiled."
        )
    if dsmh_config and total_params >= 5_740_751:
        raise RuntimeError(
            "DSMH configuration validation failed: parameter count must be below original DPER "
            "(5,740,751), so the dense mask head may still be registered."
        )
    if rsdm_config and total_params != 5_747_511:
        raise RuntimeError(
            "RSDM configuration validation failed: expected 5,747,511 parameters "
            "(original DPER 5,740,751 + one 6,760-parameter RSDM)."
        )
    if ntd_config:
        expected_params = 5_403_471 if getattr(ntd, "transport_dim", None) == 24 else None
        if expected_params is not None and total_params != expected_params:
            raise RuntimeError(
                f"NTD24 configuration validation failed: expected {expected_params:,} parameters, "
                f"got {total_params:,}."
            )
        if total_params > 5_750_000:
            raise RuntimeError("NTD parameter count exceeds the 5.75M admission limit.")
    if pntd_config:
        expected_params = 5_436_983
        if total_params != expected_params:
            raise RuntimeError(
                f"PNTD24 configuration validation failed: expected {expected_params:,} parameters, "
                f"got {total_params:,}."
            )
        if not 5_400_000 <= total_params <= 5_600_000:
            raise RuntimeError("PNTD parameter count is outside the planned 5.4M–5.6M interval.")
    if dpsm_config and total_params > 5_500_000:
        raise RuntimeError(f"DPSM80 parameter count exceeds 5.5M: {total_params:,}.")
    if fwrmh_config and total_params > 3_193_800:
        raise RuntimeError(f"DPSM80-FWRMH parameter count exceeds the DPSM80 baseline: {total_params:,}.")
    if dafu1_config and total_params > 3_193_800:
        raise RuntimeError(f"DPSM80-DAFU1 parameter count exceeds the DPSM80 baseline: {total_params:,}.")
    if cpdpsm_config and total_params > 4_200_000:
        raise RuntimeError(f"CP-DPSM64 parameter count exceeds 4.2M: {total_params:,}.")
    raw_weight_bytes = tensor_bytes(list(model.parameters()))
    serialized_size = state_dict_size_bytes(model)
    benchmark_runs = [
        benchmark_fps(
            inference_model,
            image,
            device=device,
            warmup=args.warmup,
            iterations=args.iterations,
            amp=args.amp,
        )
        for _ in range(args.benchmark_repeats)
    ]
    fps_values = [result[0] for result in benchmark_runs]
    latency_values = [result[1] for result in benchmark_runs]
    memory_values = [result[2] for result in benchmark_runs if result[2] is not None]
    fps = float(median(fps_values))
    batch_latency_ms = float(median(latency_values))
    peak_cuda_memory = max(memory_values) if memory_values else None

    flop_results: dict[str, tuple[float, Counter[str]]] = {}
    flop_errors: dict[str, str] = {}
    for flop_path in ("full", "mask-effective", "mask-only"):
        if flop_path == "mask-only" and not callable(getattr(model, "forward_mask_only", None)):
            flop_errors[flop_path] = "model does not implement forward_mask_only(image)"
            continue
        try:
            flop_results[flop_path] = measure_flops(model, image, inference_path=flop_path)
        except Exception as error:  # Keep Params/Size available if fvcore cannot trace a graph.
            flop_errors[flop_path] = f"{type(error).__name__}: {error}"

    print("Table-2-style complexity (inference mode)")
    print(f"Model: {model_name}")
    print(f"Inference path: {selected_path}")
    print(f"Input: batch={args.batch_size}, channels=3, height={height}, width={width}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
    print("Stage4 formal audit")
    print(f"- variant: {stage4_audit['variant']}; class: {stage4_audit['stage4_class']}")
    print(
        "- route_mamba: "
        f"{stage4_audit['route_mamba_present']}; route head: {stage4_audit['route_head_present']}; "
        f"route_s4 output: {stage4_audit['route_s4_present']}"
    )
    if int(stage4_audit["mamba_instantiated_core_count"]) == 0:
        print("- Mamba execution: none; instantiated cores: 0")
    else:
        print(
            "- Mamba execution | instantiated cores: "
            f"{stage4_audit['mamba_instantiated_core_count']}; fused calls: "
            f"{stage4_audit['mamba_forward_call_count']}; routed sequences: "
            f"{stage4_audit['mamba_routed_sequence_count']}; backends: {stage4_audit['mamba_backends']}"
        )
        print(f"- captured Mamba input shapes: {stage4_audit['mamba_input_shapes']}")
    print(
        "- mask-only equivalence | "
        f"shape={mask_only_audit['mask_only_shape']} | max_abs={mask_only_audit['mask_only_max_abs_error']:.3e} | "
        f"mean_abs={mask_only_audit['mask_only_mean_abs_error']:.3e} | allclose={mask_only_audit['mask_only_allclose']}"
    )
    if escs_config:
        assert early_escs is not None
        print("Candidate flag | use_early_serpentine_connectivity_ssm: true")
        print("Early ESCS module | present")
        print(f"Early Mamba backend | {early_escs.backend}")
        print(f"Early Mamba instantiated cores | {early_escs.instantiated_core_count}")
        print(f"Early Mamba traversal count | {early_escs.traversal_count}")
        print(f"Early Mamba calls in last forward | {early_escs.last_shared_mamba_call_count}")
    if pfrs_config:
        assert pfrs is not None
        print("Candidate flag | use_partial_full_resolution_stage1: true")
        print("PFRS module | present")
        print(f"full-resolution LocalCrackBlock | present ({pfrs.detail_channels}ch)")
        print(
            "residual-only phase exchange | present "
            f"(PixelUnshuffle ×2, {pfrs.detail_channels * 4}→{pfrs.semantic_channels})"
        )
        print("ESCS module | absent")
    if msdr_config:
        assert msdr_stage3_route is not None and msdr_stage4_route is not None
        total_cores = msdr_stage3_route.active_mamba_path_count + msdr_stage4_route.active_mamba_path_count
        total_calls = len(msdr_stage3_route.last_executed_mamba_directions) + len(
            msdr_stage4_route.last_executed_mamba_directions
        )
        print("Candidate flag | use_multiscale_directional_reallocation: true")
        print("MSDR structure | present")
        print("Stage3 routed block | present")
        print("Stage4 routed block | present")
        print("Stage3 active Mamba directions | horizontal, vertical")
        print("Stage4 active Mamba directions | diagonal, anti_diagonal")
        print("Stage3 Transformer | disabled")
        print("Stage4 Transformer | enabled")
        print(f"Total instantiated Mamba cores | {total_cores}")
        print(f"Total Mamba calls in last forward | {total_calls}")
        print("Optimizer expected | fused AdamW")
        print("Failed DPER candidate modules | absent")
    if dsmh_config:
        assert isinstance(dsmh, DepthwiseSeparableMaskHead)
        print("Candidate flag | use_depthwise_separable_mask_head: true")
        print("DSMH module | present")
        print("Dense mask head | absent")
        print("Depthwise groups | 40")
        print("Pointwise channels | 40→40")
        print("Output channels | 40→1")
        print("LSSCF up1 | unchanged")
        print("Stage4 Mamba | mamba_ssm ×4")
        print("Failed candidate modules | absent")
    if rsdm_config:
        assert isinstance(rsdm, RIFTStructuralDirectionalModelingBlock)
        rsdm_params = sum(parameter.numel() for parameter in rsdm.parameters())
        print("Candidate flag | use_rsdm_stage1: true")
        print("RSDM module | present (exactly 1)")
        print("RSDM location | half-resolution Stage1 semantic path")
        print("RSDM channels | 40→40 | hidden ratio: 1.0")
        print("RSDM directions | depthwise 1×13 + 13×1 + dilated 3×3")
        print("RSDM gate | structural pointwise sigmoid")
        print(f"RSDM Params | {rsdm_params:,}")
        print("Full-detail refinement | absent")
        print("Early Mamba | absent")
        print("Separable mask head | absent; original dense mask head retained")
        print("LSSCF decoder | unchanged")
        print("Stage4 Mamba | mamba_ssm ×4")
        print("Failed candidate modules | absent")
    if ntd_config:
        assert isinstance(ntd, NarrowTransportDecoder)
        ntd_params = sum(parameter.numel() for parameter in ntd.parameters())
        mask_head_params = sum(parameter.numel() for parameter in model.decoder.mask_head.parameters())
        print("Model variant | DPER-RSDM-NTD")
        print("Candidate flag | use_narrow_transport_decoder: true")
        print("RSDM | enabled; exactly 1 block")
        print(f"Transport dimension | {ntd.transport_dim}")
        print("Native projections | F1/F2/F3/F4 → transport dimension")
        print("Fusion | bilinear upsample + elementwise addition")
        print("Refine blocks | dense 3×3 Conv-GN-SiLU ×4")
        print("Wide LSSCF transport modules | unregistered")
        print("Original dense mask head | retained")
        print(f"Original dense mask head Params | {mask_head_params:,}")
        print(f"Narrow transport decoder Params | {ntd_params:,}")
        print("SPCR/HTPR/full-resolution refinement/early Mamba/DSMH | absent")
        print("Stage4 Mamba | mamba_ssm ×4")
    if pntd_config:
        assert isinstance(pntd, PartialNarrowTransportDecoder)
        pntd_params = sum(parameter.numel() for parameter in pntd.parameters())
        mask_head_params = sum(parameter.numel() for parameter in model.decoder.mask_head.parameters())
        print("Model variant | DPER-RSDM-PNTD")
        print("Candidate flag | use_partial_narrow_transport_decoder: true")
        print("RSDM | enabled; exactly 1 block")
        print(f"Partial transport dimension | {pntd.transport_dim}")
        print("Deep transport | F4→24 and F3→24, concat fusion to 160")
        print("F3→F2 decoder transport | 160→24→80")
        print("Native F2 fusion | independent 80-channel skip + preserved LSSCF concat fusion")
        print("F1 fusion/final refine | original LSSCF modules retained")
        print("Old LSSCF up3/up2 containers | unregistered")
        print("All-scale NTD24 | absent")
        print("Original dense mask head | retained")
        print(f"Original dense mask head Params | {mask_head_params:,}")
        print(f"PNTD deep/F2 transport Params | {pntd_params:,}")
        print("SPCR/HTPR/full-resolution refinement/early Mamba/DSMH | absent")
        print("Stage4 Mamba | mamba_ssm ×4")
    if dpsm_config:
        assert isinstance(dpsm, DirectionPartitionedSharedMamba)
        dpsm_params = sum(parameter.numel() for parameter in dpsm.parameters())
        print("Model variant | DPER-RSDM-DPSM80")
        print("RSDM | enabled; exactly 1 block")
        print("Stage4 directional execution | horizontal, vertical, diagonal, anti_diagonal")
        print("Stage4 directional grouping | 4 × 80 channels; concatenated width 320")
        print("Stage4 Mamba core | shared mamba_ssm ×1")
        print("Packed Mamba input | batch multiplier: 4; sequence length: H×W; channels: 80")
        print("Stage4 spatial reduction | 1")
        print("Stage4 Transformer | " + ("deleted (strict T2)" if isinstance(model.stage4, Stage4LocalDPSM80) else "enabled"))
        print("Old full-width Mamba paths/local compensation/input_projection/out_proj | absent")
        print("Original LSSCF decoder | retained")
        print("Original dense mask head | retained")
        print(f"DPSM route module Params | {dpsm_params:,}")
    if fwrmh_config:
        assert isinstance(dpsm, DirectionPartitionedSharedMamba)
        fwrmh = model.decoder.mask_head
        assert isinstance(fwrmh, FullWidthResidualMaskHead)
        fwrmh_params = sum(parameter.numel() for parameter in fwrmh.parameters())
        print("Model variant | DPSM80-FWRMH")
        print("Candidate flag | use_full_width_residual_mask_head: true")
        print("Mask head | 40ch identity bypass + 40→16→DW3×3→40 residual correction + 40→1 classifier")
        print("Dense 40→40 3×3 mask convolution | absent")
        print("DPSM80 shared Stage4 core | mamba_ssm ×1")
        print("RSDM / Stage4 Transformer / LSSCF / edge and reliability heads | retained")
        print(f"FWRMH Params | {fwrmh_params:,}")
    if dafu1_config:
        assert isinstance(dpsm, DirectionPartitionedSharedMamba)
        dafu1 = model.decoder.up1
        assert isinstance(dafu1, DifferenceAwareSubpixelUpBlock)
        dafu1_params = sum(parameter.numel() for parameter in dafu1.parameters())
        print("Model variant | DPSM80-DAFU1")
        print("Candidate flag | use_difference_aware_up1_fusion: true")
        print("Decoder up1 | full-width sum path + full-width difference correction")
        print("Decoder up2/up3 | original LSSCF concat fusion retained")
        print("Mask head | original dense 40→40 3×3 retained")
        print("DPSM80 shared Stage4 core | mamba_ssm ×1")
        print(f"DAFU1 Up1 Params | {dafu1_params:,}")
    if cpdpsm_config:
        assert isinstance(dpsm, CommonPrivateDirectionPartitionedSharedMamba)
        cpdpsm_params = sum(parameter.numel() for parameter in dpsm.parameters())
        print("Model variant | DPER-RSDM-CPDPSM64")
        print("RSDM | enabled; exactly 1 block")
        print("Stage4 common-private partition | shared: 64; private: 4×64")
        print("Per-direction Mamba width | 128")
        print("Stage4 directional traversals | horizontal, vertical, diagonal, anti_diagonal")
        print("Stage4 Mamba core | shared mamba_ssm ×1")
        print("Packed Mamba input | batch multiplier: 4; sequence length: H×W; channels: 128")
        print("Stage4 spatial reduction | 1")
        print("Stage4 Transformer | enabled")
        print("Distillation | disabled")
        print("CFPS | disabled")
        print("Original LSSCF decoder | retained")
        print("Original dense mask head | retained")
        print(f"CP-DPSM route module Params | {cpdpsm_params:,}")
    for description in describe_mamba_execution(model):
        print(description)
    backend_info = model.mamba_backend_info() if hasattr(model, "mamba_backend_info") else {}
    if any(backend != "mamba_ssm" for backends in backend_info.values() for backend in backends):
        print("WARNING: fallback/disabled Mamba backend detected; FLOPs, latency, and FPS are non-formal debug results.")
    print(f"Params: {total_params / 1e6:.4f} M ({total_params:,})")
    print(f"Trainable Params: {trainable_params / 1e6:.4f} M ({trainable_params:,})")
    print(f"Size (FP32 state_dict, optimizer excluded): {serialized_size / MIB:.2f} MiB")
    print(f"Raw parameter tensors: {raw_weight_bytes / MIB:.2f} MiB")
    if early_escs is not None:
        analytical = early_escs.analytical_mamba_estimate_flops(args.batch_size, height // 2, width // 2)
        print(
            "ESCS Mamba analytical estimate (projection/temporal-conv/state-update lower-order model): "
            f"{analytical / 1e9:.4f} G"
        )
    print(
        f"FPS ({selected_path} forward-only, {args.amp}, warmup={args.warmup}, "
        f"iterations={args.iterations}): {fps:.2f}"
    )
    print(f"Mean batch latency: {batch_latency_ms:.3f} ms")
    if args.benchmark_repeats > 1:
        print("FPS runs | " + ", ".join(f"{value:.2f}" for value in fps_values))
        print(f"FPS median | {fps:.2f}")
        print("Mean latency runs (ms) | " + ", ".join(f"{value:.3f}" for value in latency_values))
        print(f"Mean latency median (ms) | {batch_latency_ms:.3f}")
    if peak_cuda_memory is None:
        print("Peak CUDA memory: unavailable (CPU device)")
    else:
        print(f"Peak CUDA memory (post-warm-up inference): {peak_cuda_memory / MIB:.2f} MiB")

    print("\nFLOP definitions (fvcore lower bound; 1 fused multiply-add = 1 FLOP)")
    print("- Full multi-output: graph retains mask, edge, and reliability outputs.")
    print("- Mask-output effective: normal forward traced only through its mask output.")
    print("- Mask-only: eval-only forward_mask_only deployment graph.")
    flop_labels = {
        "full": "Full multi-output FLOPs",
        "mask-effective": "Mask-output effective FLOPs",
        "mask-only": "Mask-only FLOPs",
    }
    unsupported_paths: list[str] = []
    for flop_path in ("full", "mask-effective", "mask-only"):
        label = flop_labels[flop_path]
        if flop_path in flop_errors:
            print(f"{label}: unavailable ({flop_errors[flop_path]})")
            continue
        flops, unsupported = flop_results[flop_path]
        print(f"{label} (fvcore lower-bound FLOPs): {flops / 1e9:.4f} G")
        print(f"{label} (multiply-add as 2 ops): {2.0 * flops / 1e9:.4f} G")
        print(f"{label} unsupported operators: {format_unsupported(unsupported)}")
        if unsupported:
            unsupported_paths.append(flop_path)
            print("WARNING: this FLOP value is a lower bound because fvcore omitted the listed operators.")

    if selected_path not in flop_results:
        return 1
    selected_flops, _ = flop_results[selected_path]
    selected_unsupported = flop_results[selected_path][1]
    declared_non_core_unsupported = (
        "MambaInnerFn", "add", "mul", "sub", "div", "neg", "square", "sqrt", "clamp_min", "gelu", "sigmoid", "softmax", "exp", "sum", "mean",
        "amax", "reshape", "view", "scatter", "empty_like", "pixel_shuffle", "pixel_unshuffle", "upsample",
        "interpolate", "detach", "contiguous", "clone", "flatten", "permute", "transpose",
    )
    unknown_selected_unsupported = [
        name for name in selected_unsupported
        if not any(token.lower() in name.lower() for token in declared_non_core_unsupported)
    ]
    if unknown_selected_unsupported and (args.require_mamba_ssm or args.expected_mamba_cores is not None):
        raise RuntimeError(
            "Formal FLOP measurement stopped: unknown potentially material unsupported operators: "
            + ", ".join(unknown_selected_unsupported)
        )
    mamba_flops_estimated, mamba_flop_method, mamba_component_flops = estimate_mamba_inner_flops(
        model, captured_mamba_inputs, boundary_evidence
    )
    has_mamba_inner_unsupported = any("MambaInnerFn" in name for name in selected_unsupported)
    if int(stage4_audit["mamba_instantiated_core_count"]) == 0:
        if mamba_flops_estimated != 0.0:
            raise RuntimeError("Zero-core Stage4 unexpectedly produced a nonzero Mamba FLOP estimate.")
    else:
        if mamba_flops_estimated is None:
            if args.require_mamba_ssm or args.expected_mamba_cores is not None:
                raise RuntimeError("Formal Mamba FLOP measurement cannot use a fallback or missing backend.")
        elif mamba_flops_estimated <= 0.0:
            raise RuntimeError("Instantiated Stage4 Mamba produced no captured estimated FLOPs.")
        elif not has_mamba_inner_unsupported:
            raise RuntimeError(
                "Cannot produce paper FLOPs: fvcore did not report the expected unsupported "
                "prim::PythonOp.MambaInnerFn, so double-counting cannot be ruled out."
            )
    corrected_flops_estimated = (
        selected_flops + mamba_flops_estimated if mamba_flops_estimated is not None else None
    )
    if args.strict_flops and unsupported_paths:
        return 2

    if args.by_module:
        print_module_flops(model, image, inference_path=selected_path)
        print_module_timings(
            model,
            image,
            device=device,
            warmup=args.warmup,
            iterations=args.iterations,
            amp=args.amp,
            mask_only=mask_only_selected,
        )

    memory_cell = "-" if peak_cuda_memory is None else f"{peak_cuda_memory / MIB:.2f} MiB"
    print("\nSelected-path FLOPs")
    print(f"fvcore-counted FLOPs: {selected_flops / 1e9:.4f} G")
    print("Mamba fused-missing FLOPs estimated: " + (f"{mamba_flops_estimated / 1e9:.4f} G" if mamba_flops_estimated is not None else "NOT FOR PAPER"))
    print("Corrected FLOPs estimated: " + (f"{corrected_flops_estimated / 1e9:.4f} G" if corrected_flops_estimated is not None else "NOT FOR PAPER"))
    print(f"Mamba estimate method: {mamba_flop_method}")
    print("\n| Model | Inference path | Input | fvcore-counted FLOPs | Mamba fused-missing estimated | Corrected FLOPs estimated | Params | Size (FP32 state_dict) | Peak CUDA memory | FPS |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    estimated_mamba_cell = f"{mamba_flops_estimated / 1e9:.2f} G" if mamba_flops_estimated is not None else "NOT FOR PAPER"
    estimated_total_cell = f"{corrected_flops_estimated / 1e9:.2f} G" if corrected_flops_estimated is not None else "NOT FOR PAPER"
    print(
        f"| {model_name} | {selected_path} | {height}x{width} | {selected_flops / 1e9:.2f} G | "
        f"{estimated_mamba_cell} | {estimated_total_cell} | {total_params / 1e6:.2f} M | "
        f"{serialized_size / MIB:.2f} MiB | {memory_cell} | {fps:.2f} |"
    )
    report_row: dict[str, object] = {
        "experiment": args.experiment or Path(args.config).stem,
        "config": str(args.config), "params": total_params, "trainable_params": trainable_params,
        "fvcore_counted_flops": selected_flops,
        "mamba_fused_missing_flops_estimated": mamba_flops_estimated,
        "corrected_flops_estimated": corrected_flops_estimated,
        "remaining_unsupported_ops": dict(selected_unsupported),
        "flops_convention": "fvcore counted operations plus verified missing fused MambaInnerFn estimate; one multiply-add=one FLOP",
        "flops_status": (boundary_evidence or {"flops_status": "estimated_under_declared_convention"})["flops_status"],
        "mamba_fused_component_flops_estimated": mamba_component_flops,
        "mamba_flop_method": mamba_flop_method,
        # Deprecated aliases retained only for old CSV readers. New reports never use them.
        "deprecated_standard_flops_lower_bound": selected_flops,
        "deprecated_mamba_flops_estimated": mamba_flops_estimated,
        "deprecated_total_flops_estimated": corrected_flops_estimated,
        "input_height": height, "input_width": width, "batch_size": args.batch_size, "dtype": args.amp,
        "warmup": args.warmup, "timed_iterations": args.iterations, "benchmark_repeats": args.benchmark_repeats,
        "latency_mean_ms": float(mean(latency_values)), "latency_round_std_ms": float(pstdev(latency_values)),
        "latency_p50_round_ms": float(median(latency_values)), "fps": float(median(fps_values)),
        "latency_round_means_ms": latency_values, "fps_rounds": fps_values,
        "latency_statistics_scope": "mean/median/population std of repeated 300-forward batch means; not individual-frame percentiles",
        "inference_path": selected_path, "weights": "random initialization; no checkpoint loaded",
        "runtime_environment": measurement_environment(device),
        "peak_allocated_bytes": peak_cuda_memory, "peak_reserved_bytes": (torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "pytorch": torch.__version__, "cuda": torch.version.cuda,
        "mamba_ssm_backend_required": bool(args.require_mamba_ssm or args.expected_mamba_cores is not None),
        "expected_mamba_cores": args.expected_mamba_cores,
        **stage4_audit, **mask_only_audit,
    }
    append_machine_report(args.report_csv, report_row, json_output=False)
    append_machine_report(args.report_json, report_row, json_output=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
