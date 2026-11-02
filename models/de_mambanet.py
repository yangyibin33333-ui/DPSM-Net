from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import (
    CNNEdgeStem,
    ConvBNAct,
    DensityBalancedPrototypeContext,
    DetailPreservingHalfStem,
    FullResolutionJointSpatialExchange,
    HaarTopologyPreservingRebase,
    HalfResolutionStage1Block,
    LocalCrackBlock,
    LocalRouteConsistencyGate,
    NarrowDualResolutionStage1,
    NoDPSMTopoBlock,
    Stage4LocalOnly,
    Stage4LocalDPSM80,
    PartialFullResolutionStage1,
    RetinexDirectionalValleyPrior,
    RIFTStructuralDirectionalModelingBlock,
    SemanticConditionedLocalKernelResidual,
    SubpixelPhaseConsistencyRefiner,
    SubpixelDetailReconstruction,
    TopoRouteBlock,
)
from .decoder import BoundaryAwareDecoder
from .mamba_blocks import EarlySerpentineConnectivitySSM


class DEMambaNet(nn.Module):
    """Lite-TopoRoute-Mamba, retaining the original public model class name."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 32,
        use_mamba_ssm: bool = True,
        use_transformer: bool | None = None,
        mamba_stages: list[int] | tuple[int, ...] = (3, 4),
        transformer_stages: list[int] | tuple[int, ...] = (4,),
        route_bins: int = 4,
        stage3_mamba_window_size: int | None = None,
        stage3_bidirectional_mamba: bool = False,
        stage3_active_mamba_directions: tuple[str, ...] | list[str] | None = None,
        stage4_active_mamba_directions: tuple[str, ...] | list[str] | None = None,
        use_frequency_gate: bool = False,
        use_semantic_skip_gate: bool = False,
        use_detail_residual_gate: bool = False,
        use_structure_harmonic_fusion: bool = False,
        use_boundary_logit_calibration: bool = False,
        use_sdf_head: bool = False,
        use_edge_mask_consistency_refiner: bool = False,
        use_deep_supervision: bool = False,
        use_retinex_directional_valley_prior: bool = False,
        use_route_guided_directional_refiner: bool = False,
        use_route_guided_anisotropic_diffusion: bool = False,
        use_density_balanced_prototype_context: bool = False,
        use_local_semantic_continuity_calibrator: bool = False,
        use_local_pixel_reassembly_residual: bool = False,
        use_local_spectral_confidence_rejector: bool = False,
        use_high_resolution_spectral_detail_gate: bool = False,
        use_low_resolution_subpixel_additive_fusion: bool = False,
        return_training_mask_feature: bool = False,
        use_semantic_conditioned_local_kernel_residual: bool = False,
        use_local_route_gate: bool = False,
        use_deep_identity_decay: bool = False,
        deep_identity_scales: dict[str, float] | None = None,
        use_residual_branch_competition_fusion: bool = False,
        use_low_resolution_subpixel_separable_concat_fusion: bool = False,
        stage4_mamba_spatial_reduction: int = 1,
        stage4_mamba_channels: int | None = None,
        use_detail_preserving_early_rebase: bool = False,
        early_detail_channels: int = 16,
        use_subpixel_phase_consistency_refiner: bool = False,
        use_haar_topology_preserving_rebase: bool = False,
        use_narrow_dual_resolution_stage1: bool = False,
        use_full_resolution_joint_exchange: bool = False,
        use_early_serpentine_connectivity_ssm: bool = False,
        use_partial_full_resolution_stage1: bool = False,
        use_multiscale_directional_reallocation: bool = False,
        use_depthwise_separable_mask_head: bool = False,
        use_full_width_residual_mask_head: bool = False,
        use_difference_aware_up1_fusion: bool = False,
        use_rsdm_stage1: bool = False,
        use_narrow_transport_decoder: bool = False,
        decoder_transport_dim: int = 24,
        use_partial_narrow_transport_decoder: bool = False,
        partial_transport_dim: int = 24,
        use_topology_matched_no_dpsm_stage4: bool = False,
        use_direction_partitioned_shared_mamba: bool = False,
        stage4_direction_group_channels: int = 80,
        stage4_routed_branch: str = "standard",
        stage4_hybrid_order: str = "parallel",
        use_common_private_direction_partitioned_shared_mamba: bool = False,
        stage4_common_direction_channels: int = 64,
        stage4_private_direction_channels: int = 64,
        dpsm_scan_mode: str = "four_direction",
        dpsm_route_mode: str = "learned",
        rsdm_kernel_size: int = 13,
        rsdm_dilation: int = 3,
        stage4_variant: str | None = None,
    ):
        super().__init__()
        if num_classes != 1:
            raise ValueError("DE-MambaNet currently implements binary crack segmentation.")
        if stage3_mamba_window_size is not None and (
            not isinstance(stage3_mamba_window_size, int)
            or isinstance(stage3_mamba_window_size, bool)
            or stage3_mamba_window_size <= 0
        ):
            raise ValueError(
                "stage3_mamba_window_size must be a positive integer or None, "
                f"got {stage3_mamba_window_size!r}."
            )
        if (
            not isinstance(stage4_mamba_spatial_reduction, int)
            or isinstance(stage4_mamba_spatial_reduction, bool)
            or stage4_mamba_spatial_reduction < 1
        ):
            raise ValueError(
                "stage4_mamba_spatial_reduction must be a positive integer, "
                f"got {stage4_mamba_spatial_reduction!r}."
            )
        self.mamba_stages = set(mamba_stages)
        self.transformer_stages = set(transformer_stages)
        if use_transformer is False:
            self.transformer_stages.clear()
        # None preserves historical defaults, names and initialization order.
        if stage4_variant not in (None, "local_only", "local_dpsm80"):
            raise ValueError(f"Unknown stage4_variant: {stage4_variant!r}.")
        self.stage4_variant = stage4_variant
        if stage4_variant is not None:
            dpsm_on = stage4_variant == "local_dpsm80"
            if self.transformer_stages or self.mamba_stages != ({4} if dpsm_on else set()):
                raise ValueError("Strict Stage4 variants require no Transformer and Mamba only in T2 Stage4.")
            if (dpsm_on and use_topology_matched_no_dpsm_stage4) or (not dpsm_on and use_direction_partitioned_shared_mamba):
                raise ValueError("stage4_variant conflicts with the historical DPSM/NoDPSM switches.")
            use_direction_partitioned_shared_mamba = dpsm_on
            use_topology_matched_no_dpsm_stage4 = not dpsm_on
            if not (base_channels == 40 and use_detail_preserving_early_rebase and use_rsdm_stage1 and use_low_resolution_subpixel_separable_concat_fusion):
                raise ValueError("Strict T0/T2 require the C40 DPER/RSDM/LSSCF framework.")
            if (use_local_route_gate or use_deep_identity_decay or use_residual_branch_competition_fusion
                    or stage4_hybrid_order != "parallel" or dpsm_scan_mode != "four_direction"
                    or dpsm_route_mode != "learned"):
                raise ValueError("Strict T0/T2 preserve plain parallel fusion, residual and formal DPSM80.")
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        valid_stage4_hybrid_orders = TopoRouteBlock.valid_hybrid_orders
        if not isinstance(stage4_hybrid_order, str) or stage4_hybrid_order not in valid_stage4_hybrid_orders:
            raise ValueError(
                f"Invalid stage4_hybrid_order {stage4_hybrid_order!r}; "
                f"expected one of {valid_stage4_hybrid_orders}."
            )
        valid_dpsm_scan_modes = ("four_direction", "axial_repeated")
        valid_dpsm_route_modes = ("learned", "uniform")
        if not isinstance(dpsm_scan_mode, str) or dpsm_scan_mode not in valid_dpsm_scan_modes:
            raise ValueError(
                f"Invalid dpsm_scan_mode {dpsm_scan_mode!r}; expected one of {valid_dpsm_scan_modes}."
            )
        if not isinstance(dpsm_route_mode, str) or dpsm_route_mode not in valid_dpsm_route_modes:
            raise ValueError(
                f"Invalid dpsm_route_mode {dpsm_route_mode!r}; expected one of {valid_dpsm_route_modes}."
            )
        dpsm_mechanism_nondefault = dpsm_scan_mode != "four_direction" or dpsm_route_mode != "learned"
        if dpsm_mechanism_nondefault:
            if not use_direction_partitioned_shared_mamba:
                if use_topology_matched_no_dpsm_stage4:
                    variant = "NoDPSMTopoBlock"
                elif use_common_private_direction_partitioned_shared_mamba:
                    variant = "CP-DPSM64"
                elif stage4_routed_branch != "standard":
                    variant = stage4_routed_branch
                else:
                    variant = "DPSM-off Stage4"
                raise ValueError(
                    "Non-default DPSM80 scan/route modes require the original DPSM80 replacement "
                    f"(use_direction_partitioned_shared_mamba=true), not {variant}."
                )
            if use_topology_matched_no_dpsm_stage4 or use_common_private_direction_partitioned_shared_mamba:
                raise ValueError("Non-default DPSM80 scan/route modes cannot be combined with NoDPSMTopoBlock or CP-DPSM64.")
            if stage4_routed_branch != "standard":
                raise ValueError("Non-default DPSM80 scan/route modes require stage4_routed_branch='standard'.")
        if stage4_hybrid_order != "parallel":
            if set(mamba_stages) != {4}:
                raise ValueError("Non-parallel Stage4 hybrid order requires mamba_stages=[4].")
            if set(transformer_stages) != {4} or use_transformer is False:
                raise ValueError("Non-parallel Stage4 hybrid order requires transformer_stages=[4].")
            if not use_direction_partitioned_shared_mamba:
                raise ValueError("Non-parallel Stage4 hybrid order requires DPSM80.")
            if use_topology_matched_no_dpsm_stage4:
                raise ValueError("Non-parallel Stage4 hybrid order is incompatible with NoDPSMTopoBlock.")
            if use_common_private_direction_partitioned_shared_mamba:
                raise ValueError("Non-parallel Stage4 hybrid order is incompatible with CP-DPSM64.")
            if use_multiscale_directional_reallocation:
                raise ValueError("Non-parallel Stage4 hybrid order is incompatible with MSDR.")
        if stage4_mamba_channels is not None and (
            not isinstance(stage4_mamba_channels, int)
            or isinstance(stage4_mamba_channels, bool)
            or stage4_mamba_channels < 1
            or stage4_mamba_channels > c4
        ):
            raise ValueError(
                f"stage4_mamba_channels must be an integer in [1, {c4}], got {stage4_mamba_channels!r}."
            )
        valid_stage4_routed_branches = ("standard", "single_raster_mamba80", "four_independent_mamba80")
        if stage4_routed_branch not in valid_stage4_routed_branches:
            raise ValueError(
                f"Invalid stage4_routed_branch {stage4_routed_branch!r}; "
                f"expected one of {valid_stage4_routed_branches}."
            )
        if stage4_routed_branch != "standard":
            if set(mamba_stages) != {4} or set(transformer_stages) != {4} or use_transformer is False:
                raise ValueError("Lightweight Stage4 routed branches require mamba_stages=[4] and transformer_stages=[4].")
            if use_topology_matched_no_dpsm_stage4 or use_direction_partitioned_shared_mamba:
                raise ValueError("Lightweight Stage4 routed branches are mutually exclusive with NoDPSMTopoBlock and DPSM80.")
            if use_common_private_direction_partitioned_shared_mamba:
                raise ValueError("Lightweight Stage4 routed branches are mutually exclusive with CP-DPSM64.")
            if stage4_hybrid_order != "parallel":
                raise ValueError("Lightweight Stage4 routed branches require frozen parallel hybrid order.")
            if stage3_active_mamba_directions is not None or stage4_active_mamba_directions is not None:
                raise ValueError("Lightweight Stage4 routed branches require the unfused four-direction scaffold.")
            if stage4_mamba_spatial_reduction != 1:
                raise ValueError("Lightweight Stage4 routed branches require spatial reduction 1.")
            if stage4_routed_branch == "single_raster_mamba80" and stage4_mamba_channels not in (None, 80):
                raise ValueError("SingleRasterMamba80 requires stage4_mamba_channels=80.")
            if stage4_routed_branch == "four_independent_mamba80" and stage4_mamba_channels != 80:
                raise ValueError("FourIndependentMamba80 requires stage4_mamba_channels=80.")
        if (
            stage4_mamba_channels is not None
            and stage4_mamba_channels < c4
            and stage4_mamba_spatial_reduction > 1
        ):
            raise ValueError(
                "Stage4 Mamba channel bottleneck and spatial reduction are independent ablations and cannot be combined."
            )
        if use_detail_preserving_early_rebase and (
            not isinstance(early_detail_channels, int)
            or isinstance(early_detail_channels, bool)
            or early_detail_channels < 1
            or early_detail_channels >= c1
        ):
            raise ValueError(
                f"early_detail_channels must be an integer in [1, {c1 - 1}] when DPER is enabled, "
                f"got {early_detail_channels!r}."
            )
        if use_rsdm_stage1 and (
            not isinstance(rsdm_kernel_size, int)
            or isinstance(rsdm_kernel_size, bool)
            or rsdm_kernel_size < 3
            or rsdm_kernel_size % 2 == 0
        ):
            raise ValueError(
                "rsdm_kernel_size must be an odd integer greater than or equal to 3 when RSDM is enabled, "
                f"got {rsdm_kernel_size!r}."
            )
        if use_rsdm_stage1 and (
            not isinstance(rsdm_dilation, int)
            or isinstance(rsdm_dilation, bool)
            or rsdm_dilation < 1
        ):
            raise ValueError(
                "rsdm_dilation must be an integer greater than or equal to 1 when RSDM is enabled, "
                f"got {rsdm_dilation!r}."
            )
        if use_detail_preserving_early_rebase and 1 in self.mamba_stages:
            raise ValueError("DPER replaces Stage1 with its fixed half-resolution residual 3-by-3 block.")
        if use_detail_preserving_early_rebase and use_retinex_directional_valley_prior:
            raise ValueError("DPER is incompatible with the retired Retinex directional-valley prior.")
        if use_subpixel_phase_consistency_refiner and not use_detail_preserving_early_rebase:
            raise ValueError(
                "use_subpixel_phase_consistency_refiner requires "
                "use_detail_preserving_early_rebase=True."
            )
        if use_subpixel_phase_consistency_refiner and early_detail_channels != 16:
            raise ValueError(
                "SPCR is defined only for DPER's fixed 16/24 full-detail/semantic-detail channel split."
            )
        if use_haar_topology_preserving_rebase and not use_detail_preserving_early_rebase:
            raise ValueError(
                "use_haar_topology_preserving_rebase requires "
                "use_detail_preserving_early_rebase=True."
            )
        if use_haar_topology_preserving_rebase and use_subpixel_phase_consistency_refiner:
            raise ValueError("HTPR and SPCR are independent DPER ablations and cannot be enabled together.")
        if use_narrow_dual_resolution_stage1 and not use_detail_preserving_early_rebase:
            raise ValueError(
                "use_narrow_dual_resolution_stage1 requires "
                "use_detail_preserving_early_rebase=True."
            )
        if use_narrow_dual_resolution_stage1 and use_subpixel_phase_consistency_refiner:
            raise ValueError("NDRS and SPCR are independent DPER ablations and cannot be enabled together.")
        if use_narrow_dual_resolution_stage1 and use_haar_topology_preserving_rebase:
            raise ValueError("NDRS and HTPR are independent DPER ablations and cannot be enabled together.")
        if use_full_resolution_joint_exchange and not use_detail_preserving_early_rebase:
            raise ValueError(
                "use_full_resolution_joint_exchange requires "
                "use_detail_preserving_early_rebase=True."
            )
        if use_full_resolution_joint_exchange and use_narrow_dual_resolution_stage1:
            raise ValueError("FJSE and NDRS are independent DPER ablations and cannot be enabled together.")
        if use_full_resolution_joint_exchange and use_subpixel_phase_consistency_refiner:
            raise ValueError("FJSE and SPCR are independent DPER ablations and cannot be enabled together.")
        if use_full_resolution_joint_exchange and use_haar_topology_preserving_rebase:
            raise ValueError("FJSE and HTPR are independent DPER ablations and cannot be enabled together.")
        if use_early_serpentine_connectivity_ssm and not use_detail_preserving_early_rebase:
            raise ValueError(
                "use_early_serpentine_connectivity_ssm requires "
                "use_detail_preserving_early_rebase=True."
            )
        if use_early_serpentine_connectivity_ssm and use_subpixel_phase_consistency_refiner:
            raise ValueError("ESCS and SPCR are independent DPER ablations and cannot be enabled together.")
        if use_early_serpentine_connectivity_ssm and use_haar_topology_preserving_rebase:
            raise ValueError("ESCS and HTPR are independent DPER ablations and cannot be enabled together.")
        if use_early_serpentine_connectivity_ssm and use_narrow_dual_resolution_stage1:
            raise ValueError("ESCS and NDRS are independent DPER ablations and cannot be enabled together.")
        if use_early_serpentine_connectivity_ssm and use_full_resolution_joint_exchange:
            raise ValueError("ESCS and FJSE are independent DPER ablations and cannot be enabled together.")
        if use_partial_full_resolution_stage1 and not use_detail_preserving_early_rebase:
            raise ValueError(
                "use_partial_full_resolution_stage1 requires use_detail_preserving_early_rebase=True."
            )
        if use_partial_full_resolution_stage1 and use_subpixel_phase_consistency_refiner:
            raise ValueError("PFRS and SPCR are independent DPER ablations and cannot be enabled together.")
        if use_partial_full_resolution_stage1 and use_haar_topology_preserving_rebase:
            raise ValueError("PFRS and HTPR are independent DPER ablations and cannot be enabled together.")
        if use_partial_full_resolution_stage1 and use_narrow_dual_resolution_stage1:
            raise ValueError("PFRS and NDRS are independent DPER ablations and cannot be enabled together.")
        if use_partial_full_resolution_stage1 and use_full_resolution_joint_exchange:
            raise ValueError("PFRS and FJSE are independent DPER ablations and cannot be enabled together.")
        if use_partial_full_resolution_stage1 and use_early_serpentine_connectivity_ssm:
            raise ValueError("PFRS and ESCS are independent DPER ablations and cannot be enabled together.")
        msdr_stage3_directions = ("horizontal", "vertical")
        msdr_stage4_directions = ("diagonal", "anti_diagonal")
        if use_multiscale_directional_reallocation:
            if not use_detail_preserving_early_rebase:
                raise ValueError(
                    "use_multiscale_directional_reallocation requires use_detail_preserving_early_rebase=True."
                )
            conflicting_candidates = {
                "SPCR": use_subpixel_phase_consistency_refiner,
                "HTPR": use_haar_topology_preserving_rebase,
                "NDRS": use_narrow_dual_resolution_stage1,
                "FJSE": use_full_resolution_joint_exchange,
                "ESCS": use_early_serpentine_connectivity_ssm,
                "PFRS": use_partial_full_resolution_stage1,
            }
            enabled_conflicts = [name for name, enabled in conflicting_candidates.items() if enabled]
            if enabled_conflicts:
                raise ValueError(
                    "MSDR is an independent DPER ablation and cannot be combined with: "
                    + ", ".join(enabled_conflicts)
                    + "."
                )
            if set(mamba_stages) != {4}:
                raise ValueError("MSDR must start from the DPER baseline mamba_stages=[4].")
            if set(transformer_stages) != {4} or use_transformer is False:
                raise ValueError("MSDR requires Stage3 Transformer disabled and Stage4 Transformer enabled.")
            if tuple(stage3_active_mamba_directions or ()) != msdr_stage3_directions:
                raise ValueError("MSDR Stage3 directions must be exactly: horizontal, vertical.")
            if tuple(stage4_active_mamba_directions or ()) != msdr_stage4_directions:
                raise ValueError("MSDR Stage4 directions must be exactly: diagonal, anti_diagonal.")
            if stage3_mamba_window_size is not None:
                raise ValueError("MSDR does not allow Stage3 windowed Mamba.")
            if stage3_bidirectional_mamba:
                raise ValueError("MSDR does not allow Stage3 bidirectional Mamba.")
            if stage4_mamba_spatial_reduction != 1:
                raise ValueError("MSDR requires Stage4 Mamba spatial reduction=1.")
            if stage4_mamba_channels not in (None, c4):
                raise ValueError(f"MSDR requires full Stage4 Mamba channels ({c4}/{c4}).")
            if use_local_route_gate:
                raise ValueError("MSDR does not allow local route gates.")
            if use_residual_branch_competition_fusion:
                raise ValueError("MSDR does not allow residual branch competition fusion.")
            if use_deep_identity_decay:
                raise ValueError("MSDR does not allow deep identity decay.")
            if use_density_balanced_prototype_context:
                raise ValueError("MSDR does not allow density-balanced Stage3 prototype context.")
            if use_semantic_conditioned_local_kernel_residual:
                raise ValueError("MSDR does not allow semantic-conditioned Stage3/Stage4 residual fusion.")
        if use_depthwise_separable_mask_head:
            if use_full_width_residual_mask_head:
                raise ValueError("DPER-DSMH and DPSM80-FWRMH are mutually exclusive mask-head replacements.")
            if not use_detail_preserving_early_rebase:
                raise ValueError("use_depthwise_separable_mask_head requires the original DPER early rebase.")
            if base_channels != 40:
                raise ValueError("DPER-DSMH is fixed to base_channels=decoder_channels=40.")
            if not use_low_resolution_subpixel_separable_concat_fusion:
                raise ValueError("DPER-DSMH requires use_low_resolution_subpixel_separable_concat_fusion=True.")
            conflicting_candidates = {
                "SPCR": use_subpixel_phase_consistency_refiner,
                "HTPR": use_haar_topology_preserving_rebase,
                "NDRS": use_narrow_dual_resolution_stage1,
                "FJSE": use_full_resolution_joint_exchange,
                "ESCS": use_early_serpentine_connectivity_ssm,
                "PFRS": use_partial_full_resolution_stage1,
                "MSDR": use_multiscale_directional_reallocation,
            }
            enabled_candidates = [name for name, enabled in conflicting_candidates.items() if enabled]
            if enabled_candidates:
                raise ValueError(
                    "DSMH is an independent DPER ablation and cannot be combined with: "
                    + ", ".join(enabled_candidates)
                    + "."
                )
            conflicting_decoder_options = {
                "LSAF": use_low_resolution_subpixel_additive_fusion,
                "frequency gate": use_frequency_gate,
                "semantic skip gate": use_semantic_skip_gate,
                "detail residual gate": use_detail_residual_gate,
                "structure harmonic fusion": use_structure_harmonic_fusion,
                "boundary logit calibration": use_boundary_logit_calibration,
                "SDF head": use_sdf_head,
                "edge-mask consistency refiner": use_edge_mask_consistency_refiner,
                "deep supervision": use_deep_supervision,
                "route-guided directional refiner": use_route_guided_directional_refiner,
                "route-guided anisotropic diffusion": use_route_guided_anisotropic_diffusion,
                "local semantic continuity calibrator": use_local_semantic_continuity_calibrator,
                "local pixel reassembly residual": use_local_pixel_reassembly_residual,
                "local spectral confidence rejector": use_local_spectral_confidence_rejector,
                "high-resolution spectral detail gate": use_high_resolution_spectral_detail_gate,
                "training mask feature": return_training_mask_feature,
            }
            enabled_decoder_options = [
                name for name, enabled in conflicting_decoder_options.items() if enabled
            ]
            if enabled_decoder_options:
                raise ValueError(
                    "DPER-DSMH cannot be combined with optional decoder modules: "
                    + ", ".join(enabled_decoder_options)
                    + "."
                )
            conflicting_encoder_options = {
                "local route gate": use_local_route_gate,
                "deep identity decay": use_deep_identity_decay,
                "residual branch competition": use_residual_branch_competition_fusion,
                "density-balanced prototype context": use_density_balanced_prototype_context,
                "semantic-conditioned residual": use_semantic_conditioned_local_kernel_residual,
            }
            enabled_encoder_options = [
                name for name, enabled in conflicting_encoder_options.items() if enabled
            ]
            if enabled_encoder_options:
                raise ValueError(
                    "DPER-DSMH changes only mask_head and cannot be combined with encoder ablations: "
                    + ", ".join(enabled_encoder_options)
                    + "."
                )
            if set(mamba_stages) != {4} or set(transformer_stages) != {4} or use_transformer is False:
                raise ValueError("DPER-DSMH requires the original DPER Stage4 Mamba and Transformer topology.")
            if stage3_mamba_window_size is not None or stage3_bidirectional_mamba:
                raise ValueError("DPER-DSMH does not allow Stage3 Mamba ablations.")
            if stage3_active_mamba_directions is not None or stage4_active_mamba_directions is not None:
                raise ValueError("DPER-DSMH requires the original four-direction Stage4 routing configuration.")
            if stage4_mamba_spatial_reduction != 1 or stage4_mamba_channels is not None:
                raise ValueError("DPER-DSMH requires full-channel Stage4 Mamba at spatial reduction 1.")
        if use_rsdm_stage1:
            if not use_detail_preserving_early_rebase:
                raise ValueError("use_rsdm_stage1 requires the original DPER early rebase.")
            if base_channels != 40:
                raise ValueError("DPER-RSDM is fixed to the original C40 DPER semantic width.")
            if not use_low_resolution_subpixel_separable_concat_fusion:
                raise ValueError("DPER-RSDM requires the unchanged LSSCF decoder.")
            conflicting_candidates = {
                "SPCR": use_subpixel_phase_consistency_refiner,
                "HTPR": use_haar_topology_preserving_rebase,
                "NDRS": use_narrow_dual_resolution_stage1,
                "FJSE": use_full_resolution_joint_exchange,
                "ESCS": use_early_serpentine_connectivity_ssm,
                "PFRS": use_partial_full_resolution_stage1,
                "MSDR": use_multiscale_directional_reallocation,
                "DSMH": use_depthwise_separable_mask_head,
            }
            enabled_candidates = [name for name, enabled in conflicting_candidates.items() if enabled]
            if enabled_candidates:
                raise ValueError(
                    "RSDM is an independent DPER ablation and cannot be combined with: "
                    + ", ".join(enabled_candidates)
                    + "."
                )
            conflicting_decoder_options = {
                "LSAF": use_low_resolution_subpixel_additive_fusion,
                "frequency gate": use_frequency_gate,
                "semantic skip gate": use_semantic_skip_gate,
                "detail residual gate": use_detail_residual_gate,
                "structure harmonic fusion": use_structure_harmonic_fusion,
                "boundary logit calibration": use_boundary_logit_calibration,
                "SDF head": use_sdf_head,
                "edge-mask consistency refiner": use_edge_mask_consistency_refiner,
                # Deep supervision is deliberately composed with LSSCF/DPER-RSDM for the
                # experiment in runs/experiments/deep_supervision; the isolation rule above
                # still applies to every other optional decoder module.
                "route-guided directional refiner": use_route_guided_directional_refiner,
                "route-guided anisotropic diffusion": use_route_guided_anisotropic_diffusion,
                "local semantic continuity calibrator": use_local_semantic_continuity_calibrator,
                "local pixel reassembly residual": use_local_pixel_reassembly_residual,
                "local spectral confidence rejector": use_local_spectral_confidence_rejector,
                "high-resolution spectral detail gate": use_high_resolution_spectral_detail_gate,
                "training mask feature": return_training_mask_feature,
            }
            enabled_decoder_options = [
                name for name, enabled in conflicting_decoder_options.items() if enabled
            ]
            if enabled_decoder_options:
                raise ValueError(
                    "DPER-RSDM keeps the original decoder and cannot be combined with: "
                    + ", ".join(enabled_decoder_options)
                    + "."
                )
            conflicting_encoder_options = {
                "local route gate": use_local_route_gate,
                "deep identity decay": use_deep_identity_decay,
                "residual branch competition": use_residual_branch_competition_fusion,
                "density-balanced prototype context": use_density_balanced_prototype_context,
                "semantic-conditioned residual": use_semantic_conditioned_local_kernel_residual,
            }
            enabled_encoder_options = [
                name for name, enabled in conflicting_encoder_options.items() if enabled
            ]
            if enabled_encoder_options:
                raise ValueError(
                    "DPER-RSDM changes only half-resolution Stage1 semantics and cannot be combined with: "
                    + ", ".join(enabled_encoder_options)
                    + "."
                )
            # RSDM consumes only DPER's half-resolution 40-channel semantic
            # stream. Preserve historical topology guards unless the explicit
            # strict Stage4 deletion path has already validated its framework.
            if stage4_variant is None and (set(mamba_stages) not in ({4}, set()) or set(transformer_stages) != {4} or use_transformer is False):
                raise ValueError("DPER-RSDM requires the frozen Stage4 Transformer topology (with or without routed Mamba).")
            if stage3_mamba_window_size is not None or stage3_bidirectional_mamba:
                raise ValueError("DPER-RSDM does not allow Stage3 Mamba ablations.")
            if stage3_active_mamba_directions is not None or stage4_active_mamba_directions is not None:
                raise ValueError("DPER-RSDM requires the frozen four-direction Stage4 topology; direction-subset variants are not allowed.")
            if stage4_mamba_spatial_reduction != 1 or (
                stage4_mamba_channels is not None and stage4_routed_branch not in ("single_raster_mamba80", "four_independent_mamba80")
            ):
                raise ValueError("DPER-RSDM requires frozen full-channel Stage4 settings (spatial reduction 1, no bottleneck).")
        if use_narrow_transport_decoder:
            if use_partial_narrow_transport_decoder:
                raise ValueError("NTD and PNTD are mutually exclusive decoder candidates.")
            if not use_rsdm_stage1:
                raise ValueError("use_narrow_transport_decoder requires the successful DPER-RSDM baseline.")
            if not use_detail_preserving_early_rebase or base_channels != 40:
                raise ValueError("DPER-RSDM-NTD requires the original C40 DPER early rebase.")
            if not use_low_resolution_subpixel_separable_concat_fusion:
                raise ValueError("DPER-RSDM-NTD requires LSSCF as the transport-only replacement baseline.")
            if decoder_transport_dim not in (16, 24):
                raise ValueError("decoder_transport_dim must be 24, or 16 for one untrained fallback measurement.")
            conflicting_candidates = {
                "SPCR": use_subpixel_phase_consistency_refiner,
                "HTPR": use_haar_topology_preserving_rebase,
                "NDRS": use_narrow_dual_resolution_stage1,
                "FJSE": use_full_resolution_joint_exchange,
                "ESCS": use_early_serpentine_connectivity_ssm,
                "PFRS": use_partial_full_resolution_stage1,
                "MSDR": use_multiscale_directional_reallocation,
                "DSMH": use_depthwise_separable_mask_head,
            }
            enabled_candidates = [name for name, enabled in conflicting_candidates.items() if enabled]
            if enabled_candidates:
                raise ValueError(
                    "NTD is an independent DPER-RSDM decoder ablation and cannot be combined with: "
                    + ", ".join(enabled_candidates)
                    + "."
                )
            conflicting_decoder_options = {
                "LSAF": use_low_resolution_subpixel_additive_fusion,
                "frequency gate": use_frequency_gate,
                "semantic skip gate": use_semantic_skip_gate,
                "detail residual gate": use_detail_residual_gate,
                "structure harmonic fusion": use_structure_harmonic_fusion,
                "boundary logit calibration": use_boundary_logit_calibration,
                "SDF head": use_sdf_head,
                "edge-mask consistency refiner": use_edge_mask_consistency_refiner,
                "deep supervision": use_deep_supervision,
                "route-guided directional refiner": use_route_guided_directional_refiner,
                "route-guided anisotropic diffusion": use_route_guided_anisotropic_diffusion,
                "local semantic continuity calibrator": use_local_semantic_continuity_calibrator,
                "local pixel reassembly residual": use_local_pixel_reassembly_residual,
                "local spectral confidence rejector": use_local_spectral_confidence_rejector,
                "high-resolution spectral detail gate": use_high_resolution_spectral_detail_gate,
                "training mask feature": return_training_mask_feature,
            }
            enabled_decoder_options = [
                name for name, enabled in conflicting_decoder_options.items() if enabled
            ]
            if enabled_decoder_options:
                raise ValueError(
                    "DPER-RSDM-NTD cannot be combined with optional decoder modules: "
                    + ", ".join(enabled_decoder_options)
                    + "."
                )
            conflicting_encoder_options = {
                "local route gate": use_local_route_gate,
                "deep identity decay": use_deep_identity_decay,
                "residual branch competition": use_residual_branch_competition_fusion,
                "density-balanced prototype context": use_density_balanced_prototype_context,
                "semantic-conditioned residual": use_semantic_conditioned_local_kernel_residual,
            }
            enabled_encoder_options = [
                name for name, enabled in conflicting_encoder_options.items() if enabled
            ]
            if enabled_encoder_options:
                raise ValueError(
                    "DPER-RSDM-NTD changes only decoder transport and cannot be combined with: "
                    + ", ".join(enabled_encoder_options)
                    + "."
                )
            if set(mamba_stages) != {4} or set(transformer_stages) != {4} or use_transformer is False:
                raise ValueError("DPER-RSDM-NTD requires the original Stage4 Mamba and Transformer topology.")
            if stage3_mamba_window_size is not None or stage3_bidirectional_mamba:
                raise ValueError("DPER-RSDM-NTD does not allow Stage3 Mamba ablations.")
            if stage3_active_mamba_directions is not None or stage4_active_mamba_directions is not None:
                raise ValueError("DPER-RSDM-NTD requires the original four-direction Stage4 routing.")
            if stage4_mamba_spatial_reduction != 1 or stage4_mamba_channels is not None:
                raise ValueError("DPER-RSDM-NTD requires full-channel Stage4 Mamba at spatial reduction 1.")
        if use_partial_narrow_transport_decoder:
            if not use_rsdm_stage1:
                raise ValueError("use_partial_narrow_transport_decoder requires the successful DPER-RSDM baseline.")
            if not use_detail_preserving_early_rebase or base_channels != 40:
                raise ValueError("DPER-RSDM-PNTD requires the original C40 DPER early rebase.")
            if not use_low_resolution_subpixel_separable_concat_fusion:
                raise ValueError("DPER-RSDM-PNTD requires LSSCF as its decoder baseline.")
            if partial_transport_dim != 24:
                raise ValueError("partial_transport_dim is fixed to 24 for this single candidate.")
            if use_narrow_transport_decoder:
                raise ValueError("PNTD and the all-scale NTD are mutually exclusive.")
            conflicting_candidates = {
                "SPCR": use_subpixel_phase_consistency_refiner,
                "HTPR": use_haar_topology_preserving_rebase,
                "NDRS": use_narrow_dual_resolution_stage1,
                "FJSE": use_full_resolution_joint_exchange,
                "ESCS": use_early_serpentine_connectivity_ssm,
                "PFRS": use_partial_full_resolution_stage1,
                "MSDR": use_multiscale_directional_reallocation,
                "DSMH": use_depthwise_separable_mask_head,
            }
            enabled_candidates = [name for name, enabled in conflicting_candidates.items() if enabled]
            if enabled_candidates:
                raise ValueError(
                    "PNTD is an independent DPER-RSDM decoder ablation and cannot be combined with: "
                    + ", ".join(enabled_candidates)
                    + "."
                )
            conflicting_decoder_options = {
                "LSAF": use_low_resolution_subpixel_additive_fusion,
                "frequency gate": use_frequency_gate,
                "semantic skip gate": use_semantic_skip_gate,
                "detail residual gate": use_detail_residual_gate,
                "structure harmonic fusion": use_structure_harmonic_fusion,
                "boundary logit calibration": use_boundary_logit_calibration,
                "SDF head": use_sdf_head,
                "edge-mask consistency refiner": use_edge_mask_consistency_refiner,
                "deep supervision": use_deep_supervision,
                "route-guided directional refiner": use_route_guided_directional_refiner,
                "route-guided anisotropic diffusion": use_route_guided_anisotropic_diffusion,
                "local semantic continuity calibrator": use_local_semantic_continuity_calibrator,
                "local pixel reassembly residual": use_local_pixel_reassembly_residual,
                "local spectral confidence rejector": use_local_spectral_confidence_rejector,
                "high-resolution spectral detail gate": use_high_resolution_spectral_detail_gate,
                "training mask feature": return_training_mask_feature,
            }
            enabled_decoder_options = [name for name, enabled in conflicting_decoder_options.items() if enabled]
            if enabled_decoder_options:
                raise ValueError(
                    "DPER-RSDM-PNTD changes only deep decoder transport and cannot be combined with: "
                    + ", ".join(enabled_decoder_options)
                    + "."
                )
            conflicting_encoder_options = {
                "local route gate": use_local_route_gate,
                "deep identity decay": use_deep_identity_decay,
                "residual branch competition": use_residual_branch_competition_fusion,
                "density-balanced prototype context": use_density_balanced_prototype_context,
                "semantic-conditioned residual": use_semantic_conditioned_local_kernel_residual,
            }
            enabled_encoder_options = [name for name, enabled in conflicting_encoder_options.items() if enabled]
            if enabled_encoder_options:
                raise ValueError(
                    "DPER-RSDM-PNTD preserves the original encoder and cannot be combined with: "
                    + ", ".join(enabled_encoder_options)
                    + "."
                )
            if set(mamba_stages) != {4} or set(transformer_stages) != {4} or use_transformer is False:
                raise ValueError("DPER-RSDM-PNTD requires the original Stage4 Mamba and Transformer topology.")
            if stage3_mamba_window_size is not None or stage3_bidirectional_mamba:
                raise ValueError("DPER-RSDM-PNTD does not allow Stage3 Mamba ablations.")
            if stage3_active_mamba_directions is not None or stage4_active_mamba_directions is not None:
                raise ValueError("DPER-RSDM-PNTD requires the original four-direction Stage4 routing.")
            if stage4_mamba_spatial_reduction != 1 or stage4_mamba_channels is not None:
                raise ValueError("DPER-RSDM-PNTD requires full-channel Stage4 Mamba at spatial reduction 1.")
        if use_topology_matched_no_dpsm_stage4 and (
            use_direction_partitioned_shared_mamba
            or use_common_private_direction_partitioned_shared_mamba
        ):
            raise ValueError("NoDPSMTopoBlock is mutually exclusive with DPSM80 and CP-DPSM64.")
        if use_topology_matched_no_dpsm_stage4:
            if self.mamba_stages:
                raise ValueError("NoDPSMTopoBlock requires mamba_stages=[]; it cannot contain any Mamba module.")
            if stage4_variant is None and (set(transformer_stages) != {4} or use_transformer is False):
                raise ValueError("NoDPSMTopoBlock requires the frozen Stage4 TransformerGlobalBlock.")
            if stage3_mamba_window_size is not None or stage3_bidirectional_mamba:
                raise ValueError("NoDPSMTopoBlock does not allow Stage3 Mamba variants.")
            if stage3_active_mamba_directions is not None or stage4_active_mamba_directions is not None:
                raise ValueError("NoDPSMTopoBlock does not allow routed direction subsets.")
            if stage4_mamba_spatial_reduction != 1 or stage4_mamba_channels is not None:
                raise ValueError("NoDPSMTopoBlock requires the frozen full-width Stage4 settings.")
            if use_residual_branch_competition_fusion or use_local_route_gate or use_deep_identity_decay:
                raise ValueError("NoDPSMTopoBlock cannot retain routed Stage4 auxiliary branches.")
        if use_direction_partitioned_shared_mamba and use_common_private_direction_partitioned_shared_mamba:
            raise ValueError("DPSM80 and CP-DPSM64 are mutually exclusive Stage4 route-Mamba replacements.")
        if use_common_private_direction_partitioned_shared_mamba:
            if not use_rsdm_stage1:
                raise ValueError("CP-DPSM64 requires the successful DPER-RSDM baseline.")
            if not use_detail_preserving_early_rebase or base_channels != 40:
                raise ValueError("CP-DPSM64 requires the original C40 DPER early rebase.")
            if not use_low_resolution_subpixel_separable_concat_fusion:
                raise ValueError("CP-DPSM64 requires the unchanged original LSSCF decoder.")
            if (
                stage4_common_direction_channels != 64
                or stage4_private_direction_channels != 64
                or stage4_common_direction_channels + 4 * stage4_private_direction_channels != c4
            ):
                raise ValueError("CP-DPSM64 requires 64 common + four 64-channel private groups totaling 320.")
            conflicting_candidates = {
                "NTD": use_narrow_transport_decoder,
                "PNTD": use_partial_narrow_transport_decoder,
                "SPCR": use_subpixel_phase_consistency_refiner,
                "HTPR": use_haar_topology_preserving_rebase,
                "NDRS": use_narrow_dual_resolution_stage1,
                "FJSE": use_full_resolution_joint_exchange,
                "ESCS": use_early_serpentine_connectivity_ssm,
                "PFRS": use_partial_full_resolution_stage1,
                "MSDR": use_multiscale_directional_reallocation,
                "DSMH": use_depthwise_separable_mask_head,
            }
            enabled_candidates = [name for name, enabled in conflicting_candidates.items() if enabled]
            if enabled_candidates:
                raise ValueError(
                    "CP-DPSM64 is an isolated Stage4 route_mamba replacement and cannot be combined with: "
                    + ", ".join(enabled_candidates)
                    + "."
                )
            conflicting_options = {
                "LSAF": use_low_resolution_subpixel_additive_fusion,
                "frequency gate": use_frequency_gate,
                "semantic skip gate": use_semantic_skip_gate,
                "detail residual gate": use_detail_residual_gate,
                "structure harmonic fusion": use_structure_harmonic_fusion,
                "boundary logit calibration": use_boundary_logit_calibration,
                "SDF head": use_sdf_head,
                "edge-mask consistency refiner": use_edge_mask_consistency_refiner,
                "deep supervision": use_deep_supervision,
                "route-guided directional refiner": use_route_guided_directional_refiner,
                "route-guided anisotropic diffusion": use_route_guided_anisotropic_diffusion,
                "local semantic continuity calibrator": use_local_semantic_continuity_calibrator,
                "local pixel reassembly residual": use_local_pixel_reassembly_residual,
                "local spectral confidence rejector": use_local_spectral_confidence_rejector,
                "high-resolution spectral detail gate": use_high_resolution_spectral_detail_gate,
                "training mask feature": return_training_mask_feature,
                "local route gate": use_local_route_gate,
                "deep identity decay": use_deep_identity_decay,
                "residual branch competition": use_residual_branch_competition_fusion,
                "density-balanced prototype context": use_density_balanced_prototype_context,
                "semantic-conditioned residual": use_semantic_conditioned_local_kernel_residual,
            }
            enabled_options = [name for name, enabled in conflicting_options.items() if enabled]
            if enabled_options:
                raise ValueError(
                    "CP-DPSM64 preserves the original DPER-RSDM decoder/Stage4 branches and cannot be combined with: "
                    + ", ".join(enabled_options)
                    + "."
                )
            if set(mamba_stages) != {4}:
                raise ValueError("CP-DPSM64 requires Stage4 as the only routed Mamba stage.")
            if set(transformer_stages) != {4} or use_transformer is False:
                raise ValueError("CP-DPSM64 requires the original Stage4 TransformerGlobalBlock.")
            if stage3_mamba_window_size is not None or stage3_bidirectional_mamba:
                raise ValueError("CP-DPSM64 does not allow Stage3 Mamba ablations.")
            if stage3_active_mamba_directions is not None or stage4_active_mamba_directions is not None:
                raise ValueError("CP-DPSM64 fixes all four Stage4 directions and does not accept direction subsets.")
            if stage4_mamba_spatial_reduction != 1:
                raise ValueError("CP-DPSM64 requires Stage4 spatial reduction 1.")
            if stage4_mamba_channels not in (None, c4):
                raise ValueError("CP-DPSM64 forbids a global Stage4 channel bottleneck.")
        if use_direction_partitioned_shared_mamba:
            # DPSM80 replaces only the fixed-width routed Stage4 Mamba.  Its
            # 320-channel input, route head, local/global branches, and output
            # fusion exist in both the ordinary C40-S4 decoder and the frozen
            # DPER-RSDM framework.  Keep the C40/Stage4 topology guards below,
            # but do not require the unrelated early-rebase/RSDM/LSSCF path.
            if base_channels != 40:
                raise ValueError("DPSM80 requires the original C40 Stage4 width.")
            if stage4_direction_group_channels != 80 or 4 * stage4_direction_group_channels != c4:
                raise ValueError("DPSM80 requires four 80-channel groups with total Stage4 width 320.")
            conflicting_candidates = {
                "NTD": use_narrow_transport_decoder,
                "PNTD": use_partial_narrow_transport_decoder,
                "SPCR": use_subpixel_phase_consistency_refiner,
                "HTPR": use_haar_topology_preserving_rebase,
                "NDRS": use_narrow_dual_resolution_stage1,
                "FJSE": use_full_resolution_joint_exchange,
                "ESCS": use_early_serpentine_connectivity_ssm,
                "PFRS": use_partial_full_resolution_stage1,
                "MSDR": use_multiscale_directional_reallocation,
                "DSMH": use_depthwise_separable_mask_head,
            }
            enabled_candidates = [name for name, enabled in conflicting_candidates.items() if enabled]
            if enabled_candidates:
                raise ValueError(
                    "DPSM80 is an isolated Stage4 route_mamba replacement and cannot be combined with: "
                    + ", ".join(enabled_candidates)
                    + "."
                )
            conflicting_decoder_options = {
                "LSAF": use_low_resolution_subpixel_additive_fusion,
                "frequency gate": use_frequency_gate,
                "semantic skip gate": use_semantic_skip_gate,
                "detail residual gate": use_detail_residual_gate,
                "structure harmonic fusion": use_structure_harmonic_fusion,
                "boundary logit calibration": use_boundary_logit_calibration,
                "SDF head": use_sdf_head,
                "edge-mask consistency refiner": use_edge_mask_consistency_refiner,
                # Deep supervision is deliberately composed with LSSCF/DPER-RSDM for the
                # experiment in runs/experiments/deep_supervision; the isolation rule above
                # still applies to every other optional decoder module.
                "route-guided directional refiner": use_route_guided_directional_refiner,
                "route-guided anisotropic diffusion": use_route_guided_anisotropic_diffusion,
                "local semantic continuity calibrator": use_local_semantic_continuity_calibrator,
                "local pixel reassembly residual": use_local_pixel_reassembly_residual,
                "local spectral confidence rejector": use_local_spectral_confidence_rejector,
                "high-resolution spectral detail gate": use_high_resolution_spectral_detail_gate,
                "training mask feature": return_training_mask_feature,
            }
            enabled_decoder_options = [name for name, enabled in conflicting_decoder_options.items() if enabled]
            if enabled_decoder_options:
                raise ValueError(
                    "DPSM80 preserves the original decoder and cannot be combined with: "
                    + ", ".join(enabled_decoder_options)
                    + "."
                )
            conflicting_encoder_options = {
                "local route gate": use_local_route_gate,
                "deep identity decay": use_deep_identity_decay,
                "residual branch competition": use_residual_branch_competition_fusion,
                "density-balanced prototype context": use_density_balanced_prototype_context,
                "semantic-conditioned residual": use_semantic_conditioned_local_kernel_residual,
            }
            enabled_encoder_options = [name for name, enabled in conflicting_encoder_options.items() if enabled]
            if enabled_encoder_options:
                raise ValueError(
                    "DPSM80 preserves the original Stage4 branches and cannot be combined with: "
                    + ", ".join(enabled_encoder_options)
                    + "."
                )
            if set(mamba_stages) != {4}:
                raise ValueError("DPSM80 requires Stage4 as the only routed Mamba stage.")
            if stage4_variant is None and (set(transformer_stages) != {4} or use_transformer is False):
                raise ValueError("DPSM80 requires the original Stage4 TransformerGlobalBlock.")
            if stage3_mamba_window_size is not None or stage3_bidirectional_mamba:
                raise ValueError("DPSM80 does not allow Stage3 Mamba ablations.")
            if stage3_active_mamba_directions is not None or stage4_active_mamba_directions is not None:
                raise ValueError("DPSM80 fixes all four Stage4 directions and does not accept direction subsets.")
            if stage4_mamba_spatial_reduction != 1:
                raise ValueError("DPSM80 requires Stage4 spatial reduction 1.")
            if stage4_mamba_channels not in (None, c4):
                raise ValueError("DPSM80 forbids a global Stage4 channel bottleneck.")
        if use_full_width_residual_mask_head:
            if use_depthwise_separable_mask_head:
                raise ValueError("DPSM80-FWRMH and DPER-DSMH are mutually exclusive mask-head replacements.")
            if not use_direction_partitioned_shared_mamba or use_common_private_direction_partitioned_shared_mamba:
                raise ValueError("DPSM80-FWRMH requires the original DPSM80 Stage4 replacement only.")
            if not use_rsdm_stage1 or not use_detail_preserving_early_rebase or base_channels != 40:
                raise ValueError("DPSM80-FWRMH requires the original DPER-RSDM C40 early encoder.")
            if not use_low_resolution_subpixel_separable_concat_fusion:
                raise ValueError("DPSM80-FWRMH requires the unchanged LSSCF decoder fusion.")
            conflicting_candidates = {
                "NTD": use_narrow_transport_decoder,
                "PNTD": use_partial_narrow_transport_decoder,
                "SPCR": use_subpixel_phase_consistency_refiner,
                "HTPR": use_haar_topology_preserving_rebase,
                "NDRS": use_narrow_dual_resolution_stage1,
                "FJSE": use_full_resolution_joint_exchange,
                "ESCS": use_early_serpentine_connectivity_ssm,
                "PFRS": use_partial_full_resolution_stage1,
                "MSDR": use_multiscale_directional_reallocation,
            }
            enabled_candidates = [name for name, enabled in conflicting_candidates.items() if enabled]
            if enabled_candidates:
                raise ValueError(
                    "DPSM80-FWRMH is an isolated mask-head replacement and cannot be combined with: "
                    + ", ".join(enabled_candidates)
                    + "."
                )
            conflicting_decoder_options = {
                "LSAF": use_low_resolution_subpixel_additive_fusion,
                "frequency gate": use_frequency_gate,
                "semantic skip gate": use_semantic_skip_gate,
                "detail residual gate": use_detail_residual_gate,
                "structure harmonic fusion": use_structure_harmonic_fusion,
                "boundary logit calibration": use_boundary_logit_calibration,
                "SDF head": use_sdf_head,
                "edge-mask consistency refiner": use_edge_mask_consistency_refiner,
                "deep supervision": use_deep_supervision,
                "route-guided directional refiner": use_route_guided_directional_refiner,
                "route-guided anisotropic diffusion": use_route_guided_anisotropic_diffusion,
                "local semantic continuity calibrator": use_local_semantic_continuity_calibrator,
                "local pixel reassembly residual": use_local_pixel_reassembly_residual,
                "local spectral confidence rejector": use_local_spectral_confidence_rejector,
                "high-resolution spectral detail gate": use_high_resolution_spectral_detail_gate,
                "training mask feature": return_training_mask_feature,
            }
            enabled_decoder_options = [name for name, enabled in conflicting_decoder_options.items() if enabled]
            if enabled_decoder_options:
                raise ValueError(
                    "DPSM80-FWRMH cannot be combined with optional decoder modules: "
                    + ", ".join(enabled_decoder_options)
                    + "."
                )
            conflicting_encoder_options = {
                "local route gate": use_local_route_gate,
                "deep identity decay": use_deep_identity_decay,
                "residual branch competition": use_residual_branch_competition_fusion,
                "density-balanced prototype context": use_density_balanced_prototype_context,
                "semantic-conditioned residual": use_semantic_conditioned_local_kernel_residual,
            }
            enabled_encoder_options = [name for name, enabled in conflicting_encoder_options.items() if enabled]
            if enabled_encoder_options:
                raise ValueError(
                    "DPSM80-FWRMH preserves the DPSM80 encoder and cannot be combined with: "
                    + ", ".join(enabled_encoder_options)
                    + "."
                )
            if set(mamba_stages) != {4} or set(transformer_stages) != {4} or use_transformer is False:
                raise ValueError("DPSM80-FWRMH requires the original DPSM80 Stage4 Mamba and Transformer topology.")
            if stage3_mamba_window_size is not None or stage3_bidirectional_mamba:
                raise ValueError("DPSM80-FWRMH does not allow Stage3 Mamba ablations.")
            if stage3_active_mamba_directions is not None or stage4_active_mamba_directions is not None:
                raise ValueError("DPSM80-FWRMH requires all four original Stage4 directions.")
            if stage4_mamba_spatial_reduction != 1 or stage4_mamba_channels not in (None, c4):
                raise ValueError("DPSM80-FWRMH requires full-channel Stage4 Mamba at spatial reduction 1.")
        if use_difference_aware_up1_fusion:
            if use_depthwise_separable_mask_head or use_full_width_residual_mask_head:
                raise ValueError("DAFU1 is mutually exclusive with DSMH and FWRMH mask-head replacements.")
            if not use_direction_partitioned_shared_mamba or use_common_private_direction_partitioned_shared_mamba:
                raise ValueError("DAFU1 requires the original DPSM80 Stage4 replacement only.")
            if not use_rsdm_stage1 or not use_detail_preserving_early_rebase or base_channels != 40:
                raise ValueError("DAFU1 requires the original DPER-RSDM C40 early encoder.")
            if not use_low_resolution_subpixel_separable_concat_fusion:
                raise ValueError("DAFU1 requires the unchanged LSSCF decoder fusion.")
            conflicting_candidates = {
                "NTD": use_narrow_transport_decoder,
                "PNTD": use_partial_narrow_transport_decoder,
                "SPCR": use_subpixel_phase_consistency_refiner,
                "HTPR": use_haar_topology_preserving_rebase,
                "NDRS": use_narrow_dual_resolution_stage1,
                "FJSE": use_full_resolution_joint_exchange,
                "ESCS": use_early_serpentine_connectivity_ssm,
                "PFRS": use_partial_full_resolution_stage1,
                "MSDR": use_multiscale_directional_reallocation,
            }
            enabled_candidates = [name for name, enabled in conflicting_candidates.items() if enabled]
            if enabled_candidates:
                raise ValueError(
                    "DAFU1 is an isolated Up1 replacement and cannot be combined with: "
                    + ", ".join(enabled_candidates)
                    + "."
                )
            conflicting_decoder_options = {
                "LSAF": use_low_resolution_subpixel_additive_fusion,
                "frequency gate": use_frequency_gate,
                "semantic skip gate": use_semantic_skip_gate,
                "detail residual gate": use_detail_residual_gate,
                "structure harmonic fusion": use_structure_harmonic_fusion,
                "boundary logit calibration": use_boundary_logit_calibration,
                "SDF head": use_sdf_head,
                "edge-mask consistency refiner": use_edge_mask_consistency_refiner,
                "deep supervision": use_deep_supervision,
                "route-guided directional refiner": use_route_guided_directional_refiner,
                "route-guided anisotropic diffusion": use_route_guided_anisotropic_diffusion,
                "local semantic continuity calibrator": use_local_semantic_continuity_calibrator,
                "local pixel reassembly residual": use_local_pixel_reassembly_residual,
                "local spectral confidence rejector": use_local_spectral_confidence_rejector,
                "high-resolution spectral detail gate": use_high_resolution_spectral_detail_gate,
                "training mask feature": return_training_mask_feature,
            }
            enabled_decoder_options = [name for name, enabled in conflicting_decoder_options.items() if enabled]
            if enabled_decoder_options:
                raise ValueError(
                    "DAFU1 cannot be combined with optional decoder modules: "
                    + ", ".join(enabled_decoder_options)
                    + "."
                )
            conflicting_encoder_options = {
                "local route gate": use_local_route_gate,
                "deep identity decay": use_deep_identity_decay,
                "residual branch competition": use_residual_branch_competition_fusion,
                "density-balanced prototype context": use_density_balanced_prototype_context,
                "semantic-conditioned residual": use_semantic_conditioned_local_kernel_residual,
            }
            enabled_encoder_options = [name for name, enabled in conflicting_encoder_options.items() if enabled]
            if enabled_encoder_options:
                raise ValueError(
                    "DAFU1 preserves the DPSM80 encoder and cannot be combined with: "
                    + ", ".join(enabled_encoder_options)
                    + "."
                )
            if set(mamba_stages) != {4} or set(transformer_stages) != {4} or use_transformer is False:
                raise ValueError("DAFU1 requires the original DPSM80 Stage4 Mamba and Transformer topology.")
            if stage3_mamba_window_size is not None or stage3_bidirectional_mamba:
                raise ValueError("DAFU1 does not allow Stage3 Mamba ablations.")
            if stage3_active_mamba_directions is not None or stage4_active_mamba_directions is not None:
                raise ValueError("DAFU1 requires all four original Stage4 directions.")
            if stage4_mamba_spatial_reduction != 1 or stage4_mamba_channels not in (None, c4):
                raise ValueError("DAFU1 requires full-channel Stage4 Mamba at spatial reduction 1.")
        self.stem = CNNEdgeStem(in_channels, c1)
        self.stage1, self.route_stage1 = self._make_stage(c1, 1, route_bins, use_mamba_ssm)
        self.down1 = ConvBNAct(c1, c2, stride=2)
        self.stage2, self.route_stage2 = self._make_stage(c2, 2, route_bins, use_mamba_ssm)
        self.down2 = ConvBNAct(c2, c3, stride=2)
        self.stage3, self.route_stage3 = self._make_stage(
            c3,
            3,
            route_bins,
            use_mamba_ssm,
            mamba_window_size=stage3_mamba_window_size,
            mamba_bidirectional=stage3_bidirectional_mamba,
            active_mamba_directions=(None if use_multiscale_directional_reallocation else stage3_active_mamba_directions),
        )
        if stage3_bidirectional_mamba and not self.route_stage3:
            raise ValueError("stage3_bidirectional_mamba requires stage3 to be a routed TopoRouteBlock.")
        self.down3 = ConvBNAct(c3, c4, stride=2)
        self.stage4, self.route_stage4 = self._make_stage(
            c4,
            4,
            route_bins,
            use_mamba_ssm,
            active_mamba_directions=(None if use_multiscale_directional_reallocation else stage4_active_mamba_directions),
            mamba_spatial_reduction=stage4_mamba_spatial_reduction,
            # Lightweight L1/L2 first construct the exact full-width
            # TopoRoute scaffold, then replace only route_mamba after all
            # shared model modules are initialized.
            mamba_channels=(None if stage4_routed_branch != "standard" else stage4_mamba_channels),
            force_routed_stage=use_topology_matched_no_dpsm_stage4,
        )
        self.decoder = BoundaryAwareDecoder(
            [c1, c2, c3, c4],
            c1,
            use_frequency_gate=use_frequency_gate,
            use_semantic_skip_gate=use_semantic_skip_gate,
            use_detail_residual_gate=use_detail_residual_gate,
            use_structure_harmonic_fusion=use_structure_harmonic_fusion,
            use_boundary_logit_calibration=use_boundary_logit_calibration,
            use_sdf_head=use_sdf_head,
            use_edge_mask_consistency_refiner=use_edge_mask_consistency_refiner,
            use_deep_supervision=use_deep_supervision,
            use_route_guided_directional_refiner=use_route_guided_directional_refiner,
            use_route_guided_anisotropic_diffusion=use_route_guided_anisotropic_diffusion,
            use_local_semantic_continuity_calibrator=use_local_semantic_continuity_calibrator,
            use_local_pixel_reassembly_residual=use_local_pixel_reassembly_residual,
            use_local_spectral_confidence_rejector=use_local_spectral_confidence_rejector,
            use_high_resolution_spectral_detail_gate=use_high_resolution_spectral_detail_gate,
            use_low_resolution_subpixel_additive_fusion=use_low_resolution_subpixel_additive_fusion,
            use_low_resolution_subpixel_separable_concat_fusion=(
                use_low_resolution_subpixel_separable_concat_fusion
            ),
            return_training_mask_feature=return_training_mask_feature,
        )
        self.local_route_gates: nn.ModuleDict | None = None
        if use_local_route_gate:
            # Create gates after every C3 module so shared parameters retain
            # the exact initialization produced by an identical random seed.
            self.local_route_gates = nn.ModuleDict(
                {
                    "stage3": LocalRouteConsistencyGate(c3),
                    "stage4": LocalRouteConsistencyGate(c4),
                }
            )
        self.deep_identity_scales = {"stage3": 1.0, "stage4": 1.0}
        if use_deep_identity_decay:
            configured_scales = deep_identity_scales or {"stage3": 0.98, "stage4": 0.90}
            self.deep_identity_scales = {
                "stage3": float(configured_scales.get("stage3", 0.98)),
                "stage4": float(configured_scales.get("stage4", 0.90)),
            }
        # Create this optional prior after every existing module so enabling it
        # does not perturb the seeded initialization of shared C3 parameters.
        self.retinex_directional_valley_prior: RetinexDirectionalValleyPrior | None = None
        if use_retinex_directional_valley_prior:
            self.retinex_directional_valley_prior = RetinexDirectionalValleyPrior(c1)
        # Create after every existing model module so DB-PCA cannot affect any
        # shared parameter initialization under the same random seed.
        self.density_balanced_prototype_context: DensityBalancedPrototypeContext | None = None
        if use_density_balanced_prototype_context:
            self.density_balanced_prototype_context = DensityBalancedPrototypeContext(c3)
        # Create last so the optional residual cannot perturb any existing
        # component's seeded initialization.
        self.semantic_conditioned_local_kernel_residual: SemanticConditionedLocalKernelResidual | None = None
        if use_semantic_conditioned_local_kernel_residual:
            self.semantic_conditioned_local_kernel_residual = SemanticConditionedLocalKernelResidual(c3, c4)
        if use_residual_branch_competition_fusion:
            for stage_name in ("stage3", "stage4"):
                stage = getattr(self, stage_name)
                if not isinstance(stage, TopoRouteBlock):
                    raise ValueError(
                        "Residual branch competition fusion requires routed "
                        f"TopoRouteBlock at {stage_name}, got {type(stage).__name__}."
                    )
                stage.enable_residual_branch_competition_fusion()

        # DPER is deliberately constructed last.  The baseline stem/Stage1/
        # down1 above establish exactly the original random-number sequence
        # for Stage2 through decoder; replacing those modules unregisters the
        # old parameters so they cannot remain in state_dict or FLOP totals.
        self.use_detail_preserving_early_rebase = bool(use_detail_preserving_early_rebase)
        self.early_detail_channels = int(early_detail_channels)
        self.use_subpixel_phase_consistency_refiner = bool(use_subpixel_phase_consistency_refiner)
        self.use_haar_topology_preserving_rebase = bool(use_haar_topology_preserving_rebase)
        self.use_narrow_dual_resolution_stage1 = bool(use_narrow_dual_resolution_stage1)
        self.use_full_resolution_joint_exchange = bool(use_full_resolution_joint_exchange)
        self.use_early_serpentine_connectivity_ssm = bool(use_early_serpentine_connectivity_ssm)
        self.use_partial_full_resolution_stage1 = bool(use_partial_full_resolution_stage1)
        self.use_multiscale_directional_reallocation = bool(use_multiscale_directional_reallocation)
        self.use_depthwise_separable_mask_head = bool(use_depthwise_separable_mask_head)
        self.use_full_width_residual_mask_head = bool(use_full_width_residual_mask_head)
        self.use_difference_aware_up1_fusion = bool(use_difference_aware_up1_fusion)
        self.use_rsdm_stage1 = bool(use_rsdm_stage1)
        # Plain configuration metadata: no parameter/buffer registration and
        # no effect on an RSDM-off model.  Defaults preserve frozen M5.
        self.rsdm_kernel_size = rsdm_kernel_size
        self.rsdm_dilation = rsdm_dilation
        self.use_narrow_transport_decoder = bool(use_narrow_transport_decoder)
        self.decoder_transport_dim = int(decoder_transport_dim)
        self.use_partial_narrow_transport_decoder = bool(use_partial_narrow_transport_decoder)
        self.partial_transport_dim = int(partial_transport_dim)
        self.use_topology_matched_no_dpsm_stage4 = bool(use_topology_matched_no_dpsm_stage4)
        self.use_direction_partitioned_shared_mamba = bool(use_direction_partitioned_shared_mamba)
        self.stage4_direction_group_channels = int(stage4_direction_group_channels)
        # Plain configuration metadata: these values are deliberately not
        # parameters/buffers, so the frozen default DPSM80 checkpoint contract
        # remains byte-for-byte unchanged.
        self.dpsm_scan_mode = str(dpsm_scan_mode)
        self.dpsm_route_mode = str(dpsm_route_mode)
        self.stage4_routed_branch = str(stage4_routed_branch)
        self.stage4_hybrid_order = str(stage4_hybrid_order)
        self.use_common_private_direction_partitioned_shared_mamba = bool(
            use_common_private_direction_partitioned_shared_mamba
        )
        self.stage4_common_direction_channels = int(stage4_common_direction_channels)
        self.stage4_private_direction_channels = int(stage4_private_direction_channels)
        self.f1_reconstruction: SubpixelDetailReconstruction | None = None
        self.narrow_dual_resolution_stage1: NarrowDualResolutionStage1 | None = None
        self.full_resolution_joint_exchange: FullResolutionJointSpatialExchange | None = None
        self.early_serpentine_connectivity_ssm: EarlySerpentineConnectivitySSM | None = None
        self.partial_full_resolution_stage1: PartialFullResolutionStage1 | None = None
        self.rsdm_stage1: RIFTStructuralDirectionalModelingBlock | None = None
        if self.use_detail_preserving_early_rebase:
            self.stem = DetailPreservingHalfStem(in_channels, c1, self.early_detail_channels)
            self.stage1 = HalfResolutionStage1Block(c1)
            self.route_stage1 = False
            self.down1 = ConvBNAct(c1, c2, kernel_size=1, stride=1, padding=0)
            self.f1_reconstruction = SubpixelDetailReconstruction(c1, self.early_detail_channels)
            if self.use_subpixel_phase_consistency_refiner:
                self.f1_reconstruction.phase_refiner = SubpixelPhaseConsistencyRefiner(
                    self.f1_reconstruction.reconstructed_channels
                )
            if self.use_haar_topology_preserving_rebase:
                assert isinstance(self.stem, DetailPreservingHalfStem)
                self.stem = HaarTopologyPreservingRebase(
                    in_channels,
                    c1,
                    self.early_detail_channels,
                    full_detail_projection=self.stem.full_detail_projection,
                )
            # Create NDRS only after every original DPER module has been
            # constructed.  It therefore cannot perturb seeded initialization
            # of DPER, Stage2-4, or the decoder.
            if self.use_narrow_dual_resolution_stage1:
                self.narrow_dual_resolution_stage1 = NarrowDualResolutionStage1(
                    self.early_detail_channels,
                    c1,
                )
            # FJSE is deliberately registered after the complete DPER path,
            # so it cannot perturb same-seed initialization of any shared
            # module.  It never replaces the DPER stem, Stage1, or decoder.
            if self.use_full_resolution_joint_exchange:
                self.full_resolution_joint_exchange = FullResolutionJointSpatialExchange(
                    self.early_detail_channels,
                    c1,
                )
            # ESCS is registered last: its single shared narrow Mamba cannot
            # change seeded initialization of DPER or any shared encoder and
            # decoder parameter.
            if self.use_early_serpentine_connectivity_ssm:
                self.early_serpentine_connectivity_ssm = EarlySerpentineConnectivitySSM(
                    c1,
                    bottleneck_channels=16,
                    use_mamba_ssm=use_mamba_ssm,
                )
            # Register PFRS only after the complete original DPER path so the
            # same seed leaves every shared DPER/encoder/decoder parameter
            # initialized bitwise identically.
            if self.use_partial_full_resolution_stage1:
                self.partial_full_resolution_stage1 = PartialFullResolutionStage1(
                    self.early_detail_channels,
                    c1,
                )
        # MSDR stages are constructed only after the complete original DPER,
        # decoder, and every shared optional slot. Reassignment unregisters the
        # original Stage3/4 so no replaced parameters survive in state_dict,
        # FLOP totals, or optimizer groups.
        if self.use_multiscale_directional_reallocation:
            self.stage3 = TopoRouteBlock(
                c3,
                route_bins,
                use_transformer=False,
                use_mamba_ssm=use_mamba_ssm,
                mamba_window_size=None,
                mamba_bidirectional=False,
                active_mamba_directions=msdr_stage3_directions,
                mamba_spatial_reduction=1,
                mamba_channels=None,
            )
            self.route_stage3 = True
            self.stage4 = TopoRouteBlock(
                c4,
                route_bins,
                use_transformer=True,
                use_mamba_ssm=use_mamba_ssm,
                mamba_window_size=None,
                mamba_bidirectional=False,
                active_mamba_directions=msdr_stage4_directions,
                mamba_spatial_reduction=1,
                mamba_channels=None,
            )
            self.route_stage4 = True
        # DSMH is created after the complete original DPER and every shared
        # module. This preserves same-seed initialization outside mask_head;
        # assignment unregisters the replaced dense head completely.
        if self.use_depthwise_separable_mask_head:
            self.decoder.enable_depthwise_separable_mask_head()
        # RSDM is registered after the complete original DPER so it cannot
        # perturb same-seed initialization of any shared encoder or decoder
        # parameter. It processes only Stage1 half-resolution semantics.
        #
        # DPSM80 is enabled below and consequently allocates its route-Mamba
        # modules after RSDM.  A non-default RSDM strip kernel has a different
        # parameter count.  For a non-default RSDM, replay the *default* 13/3
        # initialization after constructing the requested module so every
        # downstream frozen module receives exactly the RNG state used by M5.
        # Both CPU and CUDA generators are preserved: third-party Mamba
        # constructors may allocate on either generator depending on their
        # installed backend.  The temporary replay module is never registered.
        # The default 13/3 path intentionally does not snapshot, restore, or
        # replay RNG, preserving its exact historical construction sequence.
        if self.use_rsdm_stage1:
            restore_downstream_rng = (
                self.rsdm_kernel_size != RIFTStructuralDirectionalModelingBlock.kernel_size
                or self.rsdm_dilation != RIFTStructuralDirectionalModelingBlock.dilation
            )
            downstream_rng_state = torch.get_rng_state() if restore_downstream_rng else None
            downstream_cuda_rng_states = (
                torch.cuda.get_rng_state_all() if restore_downstream_rng and torch.cuda.is_available() else None
            )
            self.rsdm_stage1 = RIFTStructuralDirectionalModelingBlock(
                c1,
                kernel_size=self.rsdm_kernel_size,
                dilation=self.rsdm_dilation,
            )
            if downstream_rng_state is not None:
                torch.set_rng_state(downstream_rng_state)
                if downstream_cuda_rng_states is not None:
                    torch.cuda.set_rng_state_all(downstream_cuda_rng_states)
                RIFTStructuralDirectionalModelingBlock(c1)
        if self.stage4_routed_branch == "single_raster_mamba80":
            if not isinstance(self.stage4, TopoRouteBlock):
                raise RuntimeError("SingleRasterMamba80 expected the original routed Stage4 TopoRouteBlock scaffold.")
            self.stage4.enable_single_raster_mamba80(use_mamba_ssm)
        if self.stage4_routed_branch == "four_independent_mamba80":
            if not isinstance(self.stage4, TopoRouteBlock):
                raise RuntimeError("FourIndependentMamba80 expected the original routed Stage4 TopoRouteBlock scaffold.")
            self.stage4.enable_four_independent_mamba80(use_mamba_ssm)
        if self.use_narrow_transport_decoder:
            self.decoder.enable_narrow_transport_decoder(self.decoder_transport_dim)
        if self.use_partial_narrow_transport_decoder:
            self.decoder.enable_partial_narrow_transport_decoder(self.partial_transport_dim)
        if self.use_direction_partitioned_shared_mamba:
            if not isinstance(self.stage4, TopoRouteBlock):
                raise RuntimeError("DPSM80 expected the original routed Stage4 TopoRouteBlock.")
            self.stage4.enable_direction_partitioned_shared_mamba(
                self.stage4_direction_group_channels,
                use_mamba_ssm,
                scan_mode=self.dpsm_scan_mode,
                route_mode=self.dpsm_route_mode,
            )
            self.stage4.set_hybrid_order(self.stage4_hybrid_order)
        if self.use_topology_matched_no_dpsm_stage4:
            if not isinstance(self.stage4, TopoRouteBlock) or not self.route_stage4:
                raise RuntimeError("NoDPSMTopoBlock expected the original routed Stage4 TopoRouteBlock scaffold.")
            # Convert only after all shared modules (including optional DPER
            # and RSDM) have been initialized.  Assignment unregisters the
            # original route head/Mamba and 3-way fusion from this model.
            self.stage4 = NoDPSMTopoBlock.from_topo_route(self.stage4)
            self.route_stage4 = False
        if self.use_common_private_direction_partitioned_shared_mamba:
            if not isinstance(self.stage4, TopoRouteBlock):
                raise RuntimeError("CP-DPSM64 expected the original routed Stage4 TopoRouteBlock.")
            self.stage4.enable_common_private_direction_partitioned_shared_mamba(
                self.stage4_common_direction_channels,
                self.stage4_private_direction_channels,
                use_mamba_ssm,
            )
        # FWRMH is deliberately the final optional module.  Replacing only
        # decoder.mask_head here keeps all DPER-RSDM-DPSM80 shared parameters
        # bitwise identical under a common construction seed.
        if self.use_full_width_residual_mask_head:
            self.decoder.enable_full_width_residual_mask_head()
        # DAFU1 is final so all retained DPSM80 parameters, including Up1's
        # two projections, retain their same-seed initialization. Only the
        # original Up1 high-resolution concat-fusion children are unregistered.
        if self.use_difference_aware_up1_fusion:
            self.decoder.enable_difference_aware_up1_fusion()
        # Retain the historical shared initialization, then register only
        # active branches. No removed branch/Identity survives or executes.
        if stage4_variant == "local_only":
            self.stage4 = Stage4LocalOnly(self.stage4)
        elif stage4_variant == "local_dpsm80":
            self.stage4 = Stage4LocalDPSM80(self.stage4)

    def _make_stage(
        self,
        channels: int,
        stage: int,
        route_bins: int,
        use_mamba_ssm: bool,
        mamba_window_size: int | None = None,
        mamba_bidirectional: bool = False,
        active_mamba_directions: tuple[str, ...] | list[str] | None = None,
        mamba_spatial_reduction: int = 1,
        mamba_channels: int | None = None,
        force_routed_stage: bool = False,
    ) -> tuple[nn.Module, bool]:
        use_transformer = stage in self.transformer_stages or (stage == 4 and self.stage4_variant is not None)
        if stage in self.mamba_stages or force_routed_stage:
            return TopoRouteBlock(
                channels,
                route_bins,
                use_transformer,
                use_mamba_ssm,
                mamba_window_size=mamba_window_size if stage == 3 else None,
                mamba_bidirectional=mamba_bidirectional if stage == 3 else False,
                active_mamba_directions=active_mamba_directions,
                mamba_spatial_reduction=mamba_spatial_reduction,
                mamba_channels=mamba_channels,
            ), True
        return LocalCrackBlock(channels, channels, use_transformer), False

    @staticmethod
    def _run_stage(
        stage: nn.Module,
        routed: bool,
        x: torch.Tensor,
        local_route_gate: LocalRouteConsistencyGate | None = None,
        identity_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if routed:
            if local_route_gate is not None or identity_scale != 1.0:
                return stage(  # type: ignore[call-arg, return-value]
                    x,
                    local_route_gate=local_route_gate,
                    identity_scale=identity_scale,
                )
            return stage(x)  # type: ignore[return-value]
        return stage(x), None

    def mamba_backend_info(self) -> dict[str, list[str]]:
        info: dict[str, list[str]] = {}
        if self.early_serpentine_connectivity_ssm is not None:
            info["early_connectivity"] = [self.early_serpentine_connectivity_ssm.backend]
        for index in range(1, 5):
            stage = getattr(self, f"stage{index}")
            if getattr(self, f"route_stage{index}"):
                info[f"stage{index}"] = stage.mamba_backends  # type: ignore[attr-defined]
        return info

    def extract_encoder_features(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        """Run the exact encoder path used by forward, without invoking the segmentation decoder."""
        if self.use_detail_preserving_early_rebase:
            assert isinstance(self.stem, (DetailPreservingHalfStem, HaarTopologyPreservingRebase))
            assert isinstance(self.stage1, HalfResolutionStage1Block)
            assert self.f1_reconstruction is not None
            full_detail, semantic_half = self.stem(x)
            if self.narrow_dual_resolution_stage1 is not None:
                full_detail, semantic_half = self.narrow_dual_resolution_stage1(full_detail, semantic_half)
            if self.full_resolution_joint_exchange is not None:
                full_detail, semantic_half = self.full_resolution_joint_exchange(full_detail, semantic_half)
            stage1_half = self.stage1(semantic_half)
            if self.rsdm_stage1 is not None:
                stage1_half = self.rsdm_stage1(stage1_half)
            if self.partial_full_resolution_stage1 is not None:
                detail_stage1, enhanced_stage1_half = self.partial_full_resolution_stage1(full_detail, stage1_half)
            else:
                detail_stage1 = full_detail
                enhanced_stage1_half = (
                    self.early_serpentine_connectivity_ssm(stage1_half)
                    if self.early_serpentine_connectivity_ssm is not None
                    else stage1_half
                )
            f1 = self.f1_reconstruction(detail_stage1, enhanced_stage1_half)
            route_s1 = None
            f2, route_s2 = self._run_stage(self.stage2, self.route_stage2, self.down1(enhanced_stage1_half))
        else:
            stem_feature = self.stem(x)
            if self.retinex_directional_valley_prior is not None:
                stem_feature = self.retinex_directional_valley_prior(x, stem_feature)
            f1, route_s1 = self._run_stage(self.stage1, self.route_stage1, stem_feature)
            f2, route_s2 = self._run_stage(self.stage2, self.route_stage2, self.down1(f1))
        local_route_gates = self.local_route_gates
        f3_base, route_s3 = self._run_stage(
            self.stage3,
            self.route_stage3,
            self.down2(f2),
            None if local_route_gates is None else local_route_gates["stage3"],
            self.deep_identity_scales["stage3"],
        )
        f3 = f3_base
        if self.density_balanced_prototype_context is not None:
            f3 = self.density_balanced_prototype_context(f3)
        f3_for_stage4 = f3_base if self.semantic_conditioned_local_kernel_residual is not None else f3
        f4, route_s4 = self._run_stage(
            self.stage4,
            self.route_stage4,
            self.down3(f3_for_stage4),
            None if local_route_gates is None else local_route_gates["stage4"],
            self.deep_identity_scales["stage4"],
        )
        if self.semantic_conditioned_local_kernel_residual is not None:
            f3 = self.semantic_conditioned_local_kernel_residual(f3_base, f4)
        return {
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "f4": f4,
            "route_s1": route_s1,
            "route_s2": route_s2,
            "route_s3": route_s3,
            "route_s4": route_s4,
        }

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        output_size = x.shape[-2:]
        features = self.extract_encoder_features(x)
        f1 = features["f1"]
        f2 = features["f2"]
        f3 = features["f3"]
        f4 = features["f4"]
        route_s1 = features["route_s1"]
        route_s2 = features["route_s2"]
        route_s3 = features["route_s3"]
        route_s4 = features["route_s4"]
        outputs = self.decoder(
            [f1, f2, f3, f4],  # type: ignore[list-item]
            output_size=output_size,
            route_s3=(
                route_s3
                if self.decoder.route_guided_directional_refiner is not None
                or self.decoder.route_guided_anisotropic_diffusion is not None
                else None
            ),
        )
        for name, route_logits in (("route_s1", route_s1), ("route_s2", route_s2), ("route_s3", route_s3), ("route_s4", route_s4)):
            if route_logits is not None:
                outputs[name] = route_logits
        return outputs

    def forward_mask_only(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the eval-only deployment path that omits auxiliary predictions."""

        if self.training:
            raise RuntimeError("DEMambaNet.forward_mask_only is only available in eval mode.")
        output_size = x.shape[-2:]
        features = self.extract_encoder_features(x)
        route_s3 = features["route_s3"]
        return self.decoder.forward_mask_only(
            [features["f1"], features["f2"], features["f3"], features["f4"]],  # type: ignore[list-item]
            output_size=output_size,
            route_s3=(
                route_s3
                if self.decoder.route_guided_directional_refiner is not None
                or self.decoder.route_guided_anisotropic_diffusion is not None
                else None
            ),
        )
