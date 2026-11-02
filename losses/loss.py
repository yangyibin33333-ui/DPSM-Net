from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.fp_geometry import bridge_false_positive_mask
from .skeleton_recall import SkeletonRecallLoss


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = torch.sigmoid(logits)
        target = target.float()
        intersection = (prediction * target).sum(dim=(1, 2, 3))
        union = prediction.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        return 1.0 - ((2.0 * intersection + self.eps) / (union + self.eps)).mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        if pos_weight <= 0:
            raise ValueError("mask_pos_weight must be greater than 0.")
        self.pos_weight = float(pos_weight)
        self.bce = (
            nn.BCEWithLogitsLoss()
            if self.pos_weight == 1.0
            else nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.pos_weight]))
        )
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, target.float()) + self.dice(logits, target)


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, target.float())


class ComponentQuerySetLoss(nn.Module):
    """DETR-style one-to-one supervision for differentiable crack-component queries."""

    @staticmethod
    def _hungarian(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as error:
            raise RuntimeError(
                "Component-query supervision requires scipy.optimize.linear_sum_assignment for Hungarian matching."
            ) from error
        query_indices, target_indices = linear_sum_assignment(cost.detach().float().cpu().numpy())
        return (
            torch.as_tensor(query_indices, dtype=torch.long, device=cost.device),
            torch.as_tensor(target_indices, dtype=torch.long, device=cost.device),
        )

    @staticmethod
    def _dice_cost(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        intersection = (probability * target).sum(dim=(1, 2))
        denominator = probability.sum(dim=(1, 2)) + target.sum()
        return 1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)

    def match_single(
        self,
        component_mask_logits: torch.Tensor,
        component_class_logits: torch.Tensor,
        component_id_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return unique Hungarian query/target pairs and resized target masks for one sample."""
        if component_mask_logits.ndim != 3:
            raise ValueError("component mask logits must have shape [Q, H, W].")
        if component_class_logits.shape != (component_mask_logits.size(0), 2):
            raise ValueError("component class logits must have shape [Q, 2].")
        if component_id_map.ndim != 3 or component_id_map.size(0) != 1:
            raise ValueError("component_id_map must have shape [1, H, W].")
        query_count, height, width = component_mask_logits.shape
        if query_count != 64:
            raise ValueError("Component-query supervision requires exactly 64 queries.")
        resized_ids = F.interpolate(component_id_map.unsqueeze(0).float(), size=(height, width), mode="nearest")[0, 0].long()
        component_ids = torch.unique(resized_ids)
        component_ids = component_ids[component_ids > 0]
        if component_ids.numel() > query_count:
            raise ValueError(
                f"Component-query supervision supports at most {query_count} components, got {component_ids.numel()}."
            )
        if component_ids.numel() == 0:
            target_masks = component_mask_logits.new_zeros((0, height, width))
            return (
                torch.empty(0, dtype=torch.long, device=component_mask_logits.device),
                torch.empty(0, dtype=torch.long, device=component_mask_logits.device),
                target_masks,
            )
        target_masks = torch.stack([(resized_ids == component_id).float() for component_id in component_ids], dim=0)

        class_cost = -F.log_softmax(component_class_logits, dim=-1)[:, 1]
        probability = component_mask_logits.sigmoid()
        bce_costs, dice_costs = [], []
        for target in target_masks:
            expanded_target = target.unsqueeze(0).expand_as(component_mask_logits)
            bce_costs.append(F.binary_cross_entropy_with_logits(component_mask_logits, expanded_target, reduction="none").mean(dim=(1, 2)))
            dice_costs.append(self._dice_cost(probability, target.unsqueeze(0)))
        bce_cost = torch.stack(bce_costs, dim=1)
        dice_cost = torch.stack(dice_costs, dim=1)
        cost = class_cost.unsqueeze(1) + 2.0 * bce_cost + 2.0 * dice_cost
        query_indices, target_indices = self._hungarian(cost)
        return query_indices, target_indices, target_masks

    def forward(
        self,
        component_mask_logits: torch.Tensor,
        component_class_logits: torch.Tensor,
        component_id_map: torch.Tensor,
    ) -> torch.Tensor:
        if component_mask_logits.ndim != 4 or component_class_logits.ndim != 3:
            raise ValueError("Component-query outputs must have shapes [B, Q, H, W] and [B, Q, 2].")
        if component_mask_logits.size(0) != component_class_logits.size(0) or component_id_map.size(0) != component_mask_logits.size(0):
            raise ValueError("Component-query outputs and component_id_map must share a batch dimension.")
        if component_mask_logits.size(1) != 64 or component_class_logits.shape[1:] != (64, 2):
            raise ValueError("Component-query supervision requires component logits [B, 64, H, W] and [B, 64, 2].")

        image_losses = []
        for image_masks, image_classes, image_ids in zip(
            component_mask_logits.unbind(0), component_class_logits.unbind(0), component_id_map.unbind(0)
        ):
            query_indices, target_indices, target_masks = self.match_single(image_masks, image_classes, image_ids)
            class_targets = torch.zeros(64, dtype=torch.long, device=image_classes.device)
            if query_indices.numel() > 0:
                class_targets[query_indices] = 1
                matched_logits = image_masks.index_select(0, query_indices)
                matched_targets = target_masks.index_select(0, target_indices)
                mask_bce = F.binary_cross_entropy_with_logits(matched_logits, matched_targets)
                mask_dice = self._dice_cost(matched_logits.sigmoid(), matched_targets).mean()
            else:
                mask_bce = image_masks.new_zeros(())
                mask_dice = image_masks.new_zeros(())
            class_ce = F.cross_entropy(image_classes, class_targets)
            image_losses.append(mask_bce + mask_dice + 0.5 * class_ce)
        return torch.stack(image_losses).mean() if image_losses else component_mask_logits.new_zeros(())


class TruncatedSDFLoss(nn.Module):
    """Smooth-L1 regression restricted to the truncated SDF boundary band."""

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        per_pixel = F.smooth_l1_loss(prediction, target.float(), reduction="none")
        valid = valid.float()
        # clamp_min returns an exact zero scalar when no valid pixels exist,
        # without a device synchronization or a torch.compile graph break.
        return (per_pixel * valid).sum() / valid.sum().clamp_min(1.0)


class SCNP(nn.Module):
    """Same-class neighborhood pooling for binary mask logits."""

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("scnp_kernel_size must be a positive odd number.")
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        foreground = target > 0.5
        positive_values = logits.masked_fill(~foreground, float("inf"))
        positive_min = -F.max_pool2d(-positive_values, self.kernel_size, stride=1, padding=self.padding)
        negative_values = logits.masked_fill(foreground, float("-inf"))
        negative_max = F.max_pool2d(negative_values, self.kernel_size, stride=1, padding=self.padding)
        # Each center belongs to its own class, so finite values always exist at valid pixels.
        return torch.where(foreground, positive_min, negative_max)


class TangentRouteLoss(nn.Module):
    """Soft route cross-entropy, restricted to cached valid tangent positions."""

    def _single(self, logits: torch.Tensor, tangent_gt: torch.Tensor, tangent_valid: torch.Tensor) -> torch.Tensor:
        size = logits.shape[-2:]
        valid = F.adaptive_max_pool2d(tangent_valid.float(), size)
        target = F.adaptive_avg_pool2d(tangent_gt.float() * tangent_valid.float(), size)
        target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-6)
        per_pixel = -(target * F.log_softmax(logits, dim=1)).sum(dim=1, keepdim=True)
        return (per_pixel * valid).sum() / valid.sum().clamp_min(1.0)

    def forward(self, outputs: dict[str, torch.Tensor], tangent_gt: torch.Tensor, tangent_valid: torch.Tensor) -> torch.Tensor:
        routes = [value for key, value in outputs.items() if key.startswith("route_s")]
        if not routes:
            return tangent_gt.new_zeros(())
        return torch.stack([self._single(route, tangent_gt, tangent_valid) for route in routes]).mean()


class TangentNormalHardNegativeCalibrationLoss(nn.Module):
    """Suppress high-logit exterior background selected along GT skeleton normals."""

    NORMAL_OFFSETS = {
        0: ((-1, 0), (1, 0)),
        1: ((0, -1), (0, 1)),
        2: ((-1, 1), (1, -1)),
        3: ((-1, -1), (1, 1)),
    }

    def __init__(self, margin: float = 0.5, temperature: float = 0.25, max_normal_offset: int = 6):
        super().__init__()
        if temperature <= 0:
            raise ValueError("tnhnc_temperature must be greater than 0.")
        if max_normal_offset < 1:
            raise ValueError("tnhnc_max_normal_offset must be at least 1.")
        self.margin = float(margin)
        self.temperature = float(temperature)
        self.max_normal_offset = int(max_normal_offset)

    @staticmethod
    def _shift_with_valid(
        tensor: torch.Tensor,
        row_offset: int,
        column_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Align a shifted source tensor to anchors without treating padding as data."""
        batch, _, height, width = tensor.shape
        shifted = torch.zeros_like(tensor)
        valid = torch.zeros((batch, 1, height, width), dtype=torch.bool, device=tensor.device)
        source_top, source_bottom = max(0, row_offset), min(height, height + row_offset)
        source_left, source_right = max(0, column_offset), min(width, width + column_offset)
        target_top, target_bottom = max(0, -row_offset), min(height, height - row_offset)
        target_left, target_right = max(0, -column_offset), min(width, width - column_offset)
        if source_top >= source_bottom or source_left >= source_right:
            return shifted, valid
        shifted[..., target_top:target_bottom, target_left:target_right] = tensor[
            ..., source_top:source_bottom, source_left:source_right
        ]
        valid[..., target_top:target_bottom, target_left:target_right] = True
        return shifted, valid

    def _hard_negative(
        self,
        mask_logits: torch.Tensor,
        mask_gt: torch.Tensor,
        anchors: torch.Tensor,
        normal_direction: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        best_logit = mask_logits.new_full(mask_logits.shape, float("-inf"))
        has_candidate = torch.zeros_like(anchors, dtype=torch.bool)
        row_step, column_step = normal_direction
        for offset in range(1, self.max_normal_offset + 1):
            shifted_logits, in_bounds = self._shift_with_valid(mask_logits, row_step * offset, column_step * offset)
            shifted_mask, _ = self._shift_with_valid(mask_gt, row_step * offset, column_step * offset)
            candidate = anchors & in_bounds & (shifted_mask <= 0.5)
            replace = candidate & (~has_candidate | (shifted_logits > best_logit))
            best_logit = torch.where(replace, shifted_logits, best_logit)
            has_candidate = has_candidate | candidate
        return best_logit, has_candidate

    def forward(
        self,
        mask_logits: torch.Tensor,
        mask_gt: torch.Tensor,
        tangent_gt: torch.Tensor,
        tangent_valid: torch.Tensor,
    ) -> torch.Tensor:
        if mask_logits.shape != mask_gt.shape or mask_logits.ndim != 4 or mask_logits.size(1) != 1:
            raise ValueError("TNHNC mask_logits and mask_gt must have matching shape [B, 1, H, W].")
        if tangent_gt.shape != (mask_logits.size(0), 4, *mask_logits.shape[-2:]):
            raise ValueError("TNHNC tangent_gt must have shape [B, 4, H, W] matching mask logits.")
        if tangent_valid.shape != mask_logits.shape:
            raise ValueError("TNHNC tangent_valid must have shape [B, 1, H, W] matching mask logits.")

        anchors = (tangent_valid > 0.5) & (mask_gt > 0.5)
        weighted_loss = mask_logits.new_zeros(())
        weight_sum = mask_logits.new_zeros(())
        z_pos = mask_logits
        for tangent_bin, normal_sides in self.NORMAL_OFFSETS.items():
            tangent_weight = tangent_gt[:, tangent_bin : tangent_bin + 1].float()
            for normal_side in normal_sides:
                z_neg, has_negative = self._hard_negative(mask_logits, mask_gt, anchors, normal_side)
                pair_weight = tangent_weight * (anchors & has_negative).to(dtype=mask_logits.dtype)
                hard_negative_weight = torch.sigmoid(z_neg.detach() / self.temperature)
                pair_loss = F.softplus(self.margin - z_pos) + hard_negative_weight * F.softplus(self.margin + z_neg)
                weighted_loss = weighted_loss + (pair_weight * pair_loss).sum()
                weight_sum = weight_sum + pair_weight.sum()
        return weighted_loss / weight_sum.clamp_min(1.0)


class BoundaryPairConfidenceMarginLoss(nn.Module):
    """Separate 4-neighbor GT boundary pairs without changing inference."""

    def __init__(self, margin: float = 0.5, neg_temperature: float = 0.5):
        super().__init__()
        if neg_temperature <= 0:
            raise ValueError("bpcm_neg_temperature must be greater than 0.")
        self.margin = float(margin)
        self.neg_temperature = float(neg_temperature)

    @staticmethod
    def _cross_class_pair_count(mask_gt: torch.Tensor) -> torch.Tensor:
        """Count right/down cross-class neighbors exactly once for diagnostics/tests."""
        labels = mask_gt > 0.5
        horizontal = labels[..., :, :-1] != labels[..., :, 1:]
        vertical = labels[..., :-1, :] != labels[..., 1:, :]
        return horizontal.sum() + vertical.sum()

    def _pair_loss(
        self,
        first_logits: torch.Tensor,
        second_logits: torch.Tensor,
        first_labels: torch.Tensor,
        second_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        first_is_foreground = first_labels > 0.5
        second_is_foreground = second_labels > 0.5
        different_class = first_is_foreground != second_is_foreground
        z_fg = torch.where(first_is_foreground, first_logits, second_logits)
        z_bg = torch.where(first_is_foreground, second_logits, first_logits)
        # The weight tracks background confidence but never changes the
        # weighting policy through its own gradient path.
        w_bg = torch.sigmoid(z_bg.detach() / self.neg_temperature)
        per_pair = F.softplus(self.margin - (z_fg - z_bg)) + 0.5 * w_bg * F.softplus(self.margin + z_bg)
        pair_weight = different_class.to(dtype=first_logits.dtype)
        return (per_pair * pair_weight).sum(), pair_weight.sum()

    def forward(self, mask_logits: torch.Tensor, mask_gt: torch.Tensor) -> torch.Tensor:
        if mask_logits.shape != mask_gt.shape or mask_logits.ndim != 4 or mask_logits.size(1) != 1:
            raise ValueError("BPCM mask_logits and mask_gt must have matching shape [B, 1, H, W].")
        horizontal_loss, horizontal_count = self._pair_loss(
            mask_logits[..., :, :-1],
            mask_logits[..., :, 1:],
            mask_gt[..., :, :-1],
            mask_gt[..., :, 1:],
        )
        vertical_loss, vertical_count = self._pair_loss(
            mask_logits[..., :-1, :],
            mask_logits[..., 1:, :],
            mask_gt[..., :-1, :],
            mask_gt[..., 1:, :],
        )
        return (horizontal_loss + vertical_loss) / (horizontal_count + vertical_count).clamp_min(1.0)


class FarBackgroundHardNegativeLoss(nn.Module):
    """Penalize only the highest-logit background far outside GT crack bands."""

    def __init__(self, exclusion_kernel_size: int = 5, topk_per_image: int = 256, margin: float = 0.25):
        super().__init__()
        if exclusion_kernel_size < 1 or exclusion_kernel_size % 2 == 0:
            raise ValueError("fbhn_exclusion_kernel_size must be a positive odd number.")
        if topk_per_image <= 0:
            raise ValueError("fbhn_topk_per_image must be greater than 0.")
        self.exclusion_kernel_size = int(exclusion_kernel_size)
        self.topk_per_image = int(topk_per_image)
        self.margin = float(margin)

    def _far_background_mask(self, mask_gt: torch.Tensor) -> torch.Tensor:
        foreground = mask_gt > 0.5
        dilated_foreground = F.max_pool2d(
            foreground.float(),
            kernel_size=self.exclusion_kernel_size,
            stride=1,
            padding=self.exclusion_kernel_size // 2,
        ) > 0.5
        return (~foreground) & (~dilated_foreground)

    def forward(self, mask_logits: torch.Tensor, mask_gt: torch.Tensor) -> torch.Tensor:
        if mask_logits.shape != mask_gt.shape or mask_logits.ndim != 4 or mask_logits.size(1) != 1:
            raise ValueError("FBHN mask_logits and mask_gt must have matching shape [B, 1, H, W].")
        far_background = self._far_background_mask(mask_gt)
        loss_sum = mask_logits.new_zeros(())
        selected_count = mask_logits.new_zeros(())
        for image_logits, image_far_background in zip(mask_logits.unbind(0), far_background.unbind(0)):
            candidates = image_logits[image_far_background]
            if candidates.numel() == 0:
                continue
            hard_logits = torch.topk(candidates, k=min(self.topk_per_image, candidates.numel())).values
            loss_sum = loss_sum + F.softplus(self.margin + hard_logits).sum()
            selected_count = selected_count + hard_logits.numel()
        return loss_sum / selected_count.clamp_min(1.0)


class CohesiveFalsePositiveSuppressionLoss(nn.Module):
    """Suppress cohesive, far-background false positives without top-k mining."""

    def __init__(
        self,
        exclusion_kernel_size: int = 5,
        cohesion_kernel_size: int = 3,
        min_active_neighbors: int = 3,
        margin: float = 0.25,
    ):
        super().__init__()
        if exclusion_kernel_size < 1 or exclusion_kernel_size % 2 == 0:
            raise ValueError("cfps_exclusion_kernel_size must be a positive odd number.")
        if cohesion_kernel_size < 1 or cohesion_kernel_size % 2 == 0:
            raise ValueError("cfps_cohesion_kernel_size must be a positive odd number.")
        if not 1 <= min_active_neighbors <= cohesion_kernel_size * cohesion_kernel_size:
            raise ValueError("cfps_min_active_neighbors must be in [1, cohesion_kernel_size ** 2].")
        self.exclusion_kernel_size = int(exclusion_kernel_size)
        self.cohesion_kernel_size = int(cohesion_kernel_size)
        self.min_active_neighbors = int(min_active_neighbors)
        self.margin = float(margin)
        self.register_buffer("cohesion_kernel", torch.ones(1, 1, cohesion_kernel_size, cohesion_kernel_size))

    def selected_mask(self, logits: torch.Tensor, mask_gt: torch.Tensor) -> torch.Tensor:
        if logits.shape != mask_gt.shape or logits.ndim != 4 or logits.size(1) != 1:
            raise ValueError("CFPS logits and mask_gt must have matching shape [B, 1, H, W].")
        with torch.no_grad():
            predicted_positive = torch.sigmoid(logits.detach()) >= 0.5
            dilated_gt = F.max_pool2d(
                mask_gt.float(),
                kernel_size=self.exclusion_kernel_size,
                stride=1,
                padding=self.exclusion_kernel_size // 2,
            ) > 0.5
            far_background = (mask_gt <= 0.5) & (~dilated_gt)
            cohesion_count = F.conv2d(
                predicted_positive.float(),
                self.cohesion_kernel,
                padding=self.cohesion_kernel_size // 2,
            )
            cohesive_positive = predicted_positive & (cohesion_count >= self.min_active_neighbors)
            return cohesive_positive & far_background

    def forward(self, logits: torch.Tensor, mask_gt: torch.Tensor) -> torch.Tensor:
        selected = self.selected_mask(logits, mask_gt)
        return self.loss_from_selected(logits, selected)

    def loss_from_selected(self, logits: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        if selected.shape != logits.shape or selected.dtype != torch.bool:
            raise ValueError("CFPS selected mask must be boolean and match logits shape.")
        selected_logits = logits.masked_select(selected)
        # The sum of an empty masked selection remains connected to logits and
        # backpropagates exact zeros. This also avoids a GPU synchronization.
        denominator = selected.sum().clamp_min(1).to(dtype=logits.dtype)
        return F.softplus(selected_logits + self.margin).sum() / denominator


class AuditAlignedBoundaryBridgeFalsePositiveLoss(nn.Module):
    """Suppress fixed-threshold boundary-overextension and bridge false positives."""

    def __init__(self, boundary_kernel_size: int = 5, margin: float = 0.25):
        super().__init__()
        if boundary_kernel_size < 1 or boundary_kernel_size % 2 == 0:
            raise ValueError("aabfp_boundary_kernel_size must be a positive odd number.")
        self.boundary_kernel_size = int(boundary_kernel_size)
        self.margin = float(margin)

    def selected_masks(
        self,
        logits: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return detached selected, boundary, and bridge masks in audit order."""
        if logits.shape != mask_gt.shape or logits.ndim != 4 or logits.size(1) != 1:
            raise ValueError("AABFP logits and mask_gt must have matching shape [B, 1, H, W].")
        with torch.no_grad():
            predicted_positive = torch.sigmoid(logits.detach()) > 0.5
            foreground = mask_gt.detach() > 0.5
            false_positive = predicted_positive & (~foreground)
            bridge_per_image = []
            for prediction, target, fp in zip(
                predicted_positive[:, 0], foreground[:, 0], false_positive[:, 0]
            ):
                bridge_per_image.append(
                    torch.from_numpy(
                        bridge_false_positive_mask(
                            prediction.cpu().numpy(),
                            target.cpu().numpy(),
                            fp.cpu().numpy(),
                        )
                    )
                )
            bridge = torch.stack(bridge_per_image, dim=0).unsqueeze(1).to(device=logits.device, dtype=torch.bool)
            dilated_gt = F.max_pool2d(
                foreground.float(),
                kernel_size=self.boundary_kernel_size,
                stride=1,
                padding=self.boundary_kernel_size // 2,
            ) > 0.5
            boundary = false_positive & dilated_gt & (~bridge)
            selected = boundary | bridge
        return selected, boundary, bridge

    def loss_from_selected(self, logits: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        if not bool(selected.any()):
            return logits.new_zeros(())
        return F.softplus(logits[selected] + self.margin).mean()

    def forward(self, logits: torch.Tensor, mask_gt: torch.Tensor) -> torch.Tensor:
        selected, _, _ = self.selected_masks(logits, mask_gt)
        return self.loss_from_selected(logits, selected)


class EndpointProtectedHardConfuserFeatureContrastLoss(nn.Module):
    """Contrast weak crack features with reliable crack and far-confuser features."""

    def __init__(
        self,
        temperature: float = 0.20,
        exclusion_kernel_size: int = 7,
        endpoint_anchors: int = 48,
        foreground_anchors: int = 80,
        positive_supports: int = 128,
        hard_negatives: int = 128,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("ep_hcfc_temperature must be greater than 0.")
        if exclusion_kernel_size < 1 or exclusion_kernel_size % 2 == 0:
            raise ValueError("ep_hcfc_exclusion_kernel_size must be a positive odd number.")
        counts = {
            "ep_hcfc_endpoint_anchors": endpoint_anchors,
            "ep_hcfc_foreground_anchors": foreground_anchors,
            "ep_hcfc_positive_supports": positive_supports,
            "ep_hcfc_hard_negatives": hard_negatives,
        }
        for name, value in counts.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than 0.")
        self.temperature = float(temperature)
        self.exclusion_kernel_size = int(exclusion_kernel_size)
        self.endpoint_anchors = int(endpoint_anchors)
        self.foreground_anchors = int(foreground_anchors)
        self.positive_supports = int(positive_supports)
        self.hard_negatives = int(hard_negatives)

    @staticmethod
    def _resized_target(target: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return target if target.shape[-2:] == size else F.interpolate(target.float(), size=size, mode="nearest")

    @staticmethod
    def _topk_indices(
        logits: torch.Tensor,
        candidates: torch.Tensor,
        requested_count: int,
        largest: bool,
    ) -> torch.Tensor:
        flat_candidates = candidates.reshape(-1)
        candidate_indices = torch.nonzero(flat_candidates, as_tuple=False).squeeze(1)
        if candidate_indices.numel() == 0:
            return candidate_indices
        candidate_logits = logits.reshape(-1).index_select(0, candidate_indices)
        selected = torch.topk(
            candidate_logits,
            k=min(requested_count, candidate_indices.numel()),
            largest=largest,
            sorted=False,
        ).indices
        return candidate_indices.index_select(0, selected)

    @staticmethod
    def _endpoint_mask(foreground: torch.Tensor, tangent_valid: torch.Tensor) -> torch.Tensor:
        skeleton = foreground & (tangent_valid > 0.5)
        neighborhood = F.conv2d(
            skeleton.float(),
            torch.ones((1, 1, 3, 3), device=skeleton.device, dtype=torch.float32),
            padding=1,
        ) - skeleton.float()
        return skeleton & (neighborhood <= 1.0)

    def _hard_negative_mask(self, mask_gt: torch.Tensor, mask_logits: torch.Tensor) -> torch.Tensor:
        foreground = mask_gt > 0.5
        dilated_foreground = F.max_pool2d(
            foreground.float(),
            kernel_size=self.exclusion_kernel_size,
            stride=1,
            padding=self.exclusion_kernel_size // 2,
        ) > 0.5
        return (~foreground) & (~dilated_foreground) & (mask_logits.detach() >= 0.0)

    def _anchor_indices(
        self,
        image_logits: torch.Tensor,
        image_foreground: torch.Tensor,
        image_tangent_valid: torch.Tensor,
    ) -> torch.Tensor:
        endpoint_indices = self._topk_indices(
            image_logits,
            self._endpoint_mask(image_foreground.unsqueeze(0).unsqueeze(0), image_tangent_valid.unsqueeze(0).unsqueeze(0))[0, 0],
            self.endpoint_anchors,
            largest=False,
        )
        foreground_indices = self._topk_indices(
            image_logits,
            image_foreground,
            self.foreground_anchors,
            largest=False,
        )
        if endpoint_indices.numel() == 0:
            return foreground_indices
        if foreground_indices.numel() == 0:
            return endpoint_indices
        return torch.unique(torch.cat((endpoint_indices, foreground_indices)), sorted=True)

    def forward(
        self,
        mask_feature: torch.Tensor,
        mask_logits: torch.Tensor,
        mask_gt: torch.Tensor,
        tangent_valid: torch.Tensor,
    ) -> torch.Tensor:
        if mask_feature.ndim != 4:
            raise ValueError("EP-HCFC mask_feature must have shape [B, C, H, W].")
        if mask_logits.ndim != 4 or mask_logits.size(1) != 1:
            raise ValueError("EP-HCFC mask_logits must have shape [B, 1, H, W].")
        if mask_feature.size(0) != mask_logits.size(0) or mask_feature.shape[-2:] != mask_logits.shape[-2:]:
            raise ValueError("EP-HCFC mask_feature and mask_logits must share batch and spatial dimensions.")
        if mask_gt.ndim != 4 or tangent_valid.ndim != 4 or mask_gt.size(1) != 1 or tangent_valid.size(1) != 1:
            raise ValueError("EP-HCFC mask_gt and tangent_valid must have shape [B, 1, H, W].")
        if mask_gt.size(0) != mask_feature.size(0) or tangent_valid.size(0) != mask_feature.size(0):
            raise ValueError("EP-HCFC targets must share the feature batch dimension.")

        size = mask_feature.shape[-2:]
        mask_gt = self._resized_target(mask_gt, size)
        tangent_valid = self._resized_target(tangent_valid, size)
        feature = F.normalize(mask_feature.float(), p=2, dim=1, eps=1e-6)
        mining_logits = mask_logits.detach()
        foreground = mask_gt > 0.5
        hard_negative = self._hard_negative_mask(mask_gt, mining_logits)
        image_losses: list[torch.Tensor] = []

        for image_feature, image_logits, image_foreground, image_tangent_valid, image_hard_negative in zip(
            feature.unbind(0),
            mining_logits.unbind(0),
            foreground.unbind(0),
            tangent_valid.unbind(0),
            hard_negative.unbind(0),
        ):
            image_logits = image_logits[0]
            image_foreground = image_foreground[0]
            image_tangent_valid = image_tangent_valid[0]
            image_hard_negative = image_hard_negative[0]
            anchor_indices = self._anchor_indices(image_logits, image_foreground, image_tangent_valid)
            support_indices = self._topk_indices(
                image_logits,
                image_foreground,
                self.positive_supports,
                largest=True,
            )
            negative_indices = self._topk_indices(
                image_logits,
                image_hard_negative,
                self.hard_negatives,
                largest=True,
            )
            if anchor_indices.numel() == 0 or support_indices.numel() == 0 or negative_indices.numel() == 0:
                continue
            tokens = image_feature.permute(1, 2, 0).reshape(-1, image_feature.size(0))
            anchors = tokens.index_select(0, anchor_indices)
            supports = tokens.index_select(0, support_indices)
            negatives = tokens.index_select(0, negative_indices)
            positive_score = torch.logsumexp(anchors @ supports.transpose(0, 1) / self.temperature, dim=1)
            negative_score = torch.logsumexp(anchors @ negatives.transpose(0, 1) / self.temperature, dim=1)
            image_losses.append(F.softplus(negative_score - positive_score).mean())

        if not image_losses:
            return mask_feature.new_zeros(())
        return torch.stack(image_losses).mean()


class DEMambaLoss(nn.Module):
    def __init__(
        self,
        lambda_edge: float = 0.3,
        lambda_tangent: float = 0.2,
        scnp_weight: float = 0.2,
        scnp_warmup_epochs: int = 10,
        scnp_kernel_size: int = 3,
        mask_pos_weight: float = 1.0,
        lambda_sdf: float = 0.0,
        sdf_truncation: float = 8.0,
        lambda_aux_mask: float = 0.0,
        lambda_aux_edge: float = 0.0,
        lambda_tnhnc: float = 0.0,
        tnhnc_start_epoch: int = 30,
        tnhnc_ramp_epochs: int = 50,
        tnhnc_margin: float = 0.5,
        tnhnc_temperature: float = 0.25,
        tnhnc_max_normal_offset: int = 6,
        lambda_bpcm: float = 0.0,
        bpcm_start_epoch: int = 60,
        bpcm_ramp_epochs: int = 40,
        bpcm_margin: float = 0.5,
        bpcm_neg_temperature: float = 0.5,
        lambda_fbhn: float = 0.0,
        fbhn_start_epoch: int = 100,
        fbhn_ramp_epochs: int = 40,
        fbhn_exclusion_kernel_size: int = 5,
        fbhn_topk_per_image: int = 256,
        fbhn_margin: float = 0.25,
        lambda_ep_hcfc: float = 0.0,
        ep_hcfc_start_epoch: int = 120,
        ep_hcfc_ramp_epochs: int = 30,
        ep_hcfc_temperature: float = 0.20,
        ep_hcfc_exclusion_kernel_size: int = 7,
        ep_hcfc_endpoint_anchors: int = 48,
        ep_hcfc_foreground_anchors: int = 80,
        ep_hcfc_positive_supports: int = 128,
        ep_hcfc_hard_negatives: int = 128,
        lambda_cfps: float = 0.0,
        cfps_start_epoch: int = 120,
        cfps_ramp_epochs: int = 30,
        cfps_exclusion_kernel_size: int = 5,
        cfps_cohesion_kernel_size: int = 3,
        cfps_min_active_neighbors: int = 3,
        cfps_margin: float = 0.25,
        lambda_aabfp: float = 0.0,
        aabfp_start_epoch: int = 120,
        aabfp_ramp_epochs: int = 30,
        aabfp_boundary_kernel_size: int = 5,
        aabfp_margin: float = 0.25,
        lambda_component_query: float = 0.0,
        lambda_skel_recall: float = 0.0,
        skel_recall_band: int = 0,
        lambda_dir: float | None = None,
        use_update_clock: bool = False,
        scnp_warmup_updates: int | None = None,
        bpcm_start_update: int | None = None,
        bpcm_ramp_updates: int | None = None,
    ):
        super().__init__()
        self.lambda_edge = lambda_edge
        self.lambda_tangent = lambda_tangent if lambda_dir is None else lambda_dir
        self.scnp_weight = scnp_weight
        self.scnp_warmup_epochs = max(int(scnp_warmup_epochs), 0)
        self.mask_pos_weight = float(mask_pos_weight)
        self.lambda_sdf = float(lambda_sdf)
        self.sdf_truncation = float(sdf_truncation)
        self.lambda_aux_mask = float(lambda_aux_mask)
        self.lambda_aux_edge = float(lambda_aux_edge)
        self.lambda_tnhnc = float(lambda_tnhnc)
        self.tnhnc_start_epoch = int(tnhnc_start_epoch)
        self.tnhnc_ramp_epochs = int(tnhnc_ramp_epochs)
        self.lambda_bpcm = float(lambda_bpcm)
        self.bpcm_start_epoch = int(bpcm_start_epoch)
        self.bpcm_ramp_epochs = int(bpcm_ramp_epochs)
        self.lambda_fbhn = float(lambda_fbhn)
        self.fbhn_start_epoch = int(fbhn_start_epoch)
        self.fbhn_ramp_epochs = int(fbhn_ramp_epochs)
        self.lambda_ep_hcfc = float(lambda_ep_hcfc)
        self.ep_hcfc_start_epoch = int(ep_hcfc_start_epoch)
        self.ep_hcfc_ramp_epochs = int(ep_hcfc_ramp_epochs)
        self.lambda_cfps = float(lambda_cfps)
        self.cfps_start_epoch = int(cfps_start_epoch)
        self.cfps_ramp_epochs = int(cfps_ramp_epochs)
        self.lambda_aabfp = float(lambda_aabfp)
        self.aabfp_start_epoch = int(aabfp_start_epoch)
        self.aabfp_ramp_epochs = int(aabfp_ramp_epochs)
        self.lambda_component_query = float(lambda_component_query)
        self.lambda_skel_recall = float(lambda_skel_recall)
        self.skel_recall_band = int(skel_recall_band)
        self.use_update_clock = bool(use_update_clock)
        self.scnp_warmup_updates = None if scnp_warmup_updates is None else max(int(scnp_warmup_updates), 0)
        self.bpcm_start_update = None if bpcm_start_update is None else int(bpcm_start_update)
        self.bpcm_ramp_updates = None if bpcm_ramp_updates is None else int(bpcm_ramp_updates)
        if self.lambda_sdf < 0:
            raise ValueError("lambda_sdf must be non-negative.")
        if self.lambda_aux_mask < 0 or self.lambda_aux_edge < 0:
            raise ValueError("lambda_aux_mask and lambda_aux_edge must be non-negative.")
        if self.sdf_truncation <= 0:
            raise ValueError("sdf_truncation must be greater than 0.")
        if self.tnhnc_ramp_epochs < 0:
            raise ValueError("tnhnc_ramp_epochs must be non-negative.")
        if self.bpcm_ramp_epochs < 0:
            raise ValueError("bpcm_ramp_epochs must be non-negative.")
        if self.bpcm_ramp_updates is not None and self.bpcm_ramp_updates < 0:
            raise ValueError("bpcm_ramp_updates must be non-negative.")
        if self.fbhn_ramp_epochs < 0:
            raise ValueError("fbhn_ramp_epochs must be non-negative.")
        if self.lambda_ep_hcfc < 0:
            raise ValueError("lambda_ep_hcfc must be non-negative.")
        if self.ep_hcfc_ramp_epochs < 0:
            raise ValueError("ep_hcfc_ramp_epochs must be non-negative.")
        if self.lambda_cfps < 0:
            raise ValueError("lambda_cfps must be non-negative.")
        if self.cfps_ramp_epochs < 0:
            raise ValueError("cfps_ramp_epochs must be non-negative.")
        if self.lambda_aabfp < 0:
            raise ValueError("lambda_aabfp must be non-negative.")
        if self.aabfp_ramp_epochs < 0:
            raise ValueError("aabfp_ramp_epochs must be non-negative.")
        if self.lambda_component_query < 0:
            raise ValueError("lambda_component_query must be non-negative.")
        if self.lambda_skel_recall < 0:
            raise ValueError("lambda_skel_recall must be non-negative.")
        if self.skel_recall_band < 0:
            raise ValueError("skel_recall_band must be non-negative.")
        self.primary_mask_loss = BCEDiceLoss(pos_weight=self.mask_pos_weight)
        self.scnp_mask_loss = BCEDiceLoss(pos_weight=1.0)
        self.edge_loss = EdgeLoss()
        self.scnp = SCNP(scnp_kernel_size)
        self.tangent_loss = TangentRouteLoss()
        self.sdf_loss = TruncatedSDFLoss()
        self.tnhnc_loss = TangentNormalHardNegativeCalibrationLoss(
            margin=tnhnc_margin,
            temperature=tnhnc_temperature,
            max_normal_offset=tnhnc_max_normal_offset,
        )
        self.bpcm_loss = BoundaryPairConfidenceMarginLoss(
            margin=bpcm_margin,
            neg_temperature=bpcm_neg_temperature,
        )
        self.fbhn_loss = FarBackgroundHardNegativeLoss(
            exclusion_kernel_size=fbhn_exclusion_kernel_size,
            topk_per_image=fbhn_topk_per_image,
            margin=fbhn_margin,
        )
        self.ep_hcfc_loss = EndpointProtectedHardConfuserFeatureContrastLoss(
            temperature=ep_hcfc_temperature,
            exclusion_kernel_size=ep_hcfc_exclusion_kernel_size,
            endpoint_anchors=ep_hcfc_endpoint_anchors,
            foreground_anchors=ep_hcfc_foreground_anchors,
            positive_supports=ep_hcfc_positive_supports,
            hard_negatives=ep_hcfc_hard_negatives,
        )
        self.cfps_loss = CohesiveFalsePositiveSuppressionLoss(
            exclusion_kernel_size=cfps_exclusion_kernel_size,
            cohesion_kernel_size=cfps_cohesion_kernel_size,
            min_active_neighbors=cfps_min_active_neighbors,
            margin=cfps_margin,
        )
        self.aabfp_loss = AuditAlignedBoundaryBridgeFalsePositiveLoss(
            boundary_kernel_size=aabfp_boundary_kernel_size,
            margin=aabfp_margin,
        )
        self.component_query_loss = ComponentQuerySetLoss()
        self.skel_recall_loss = SkeletonRecallLoss(band=self.skel_recall_band)

    def _scnp_scale(self, epoch: int | None, global_update: int | None = None) -> float:
        if self.scnp_weight <= 0:
            return 0.0
        if self.use_update_clock and self.scnp_warmup_updates is not None:
            if self.scnp_warmup_updates == 0 or global_update is None:
                return self.scnp_weight
            return self.scnp_weight * min(1.0, max(0.0, global_update / self.scnp_warmup_updates))
        if self.scnp_warmup_epochs == 0 or epoch is None:
            return self.scnp_weight
        return self.scnp_weight * min(1.0, max(0.0, epoch / self.scnp_warmup_epochs))

    def _tnhnc_scale(self, epoch: int | None) -> float:
        if self.lambda_tnhnc <= 0:
            return 0.0
        if epoch is None or self.tnhnc_ramp_epochs == 0:
            return self.lambda_tnhnc
        progress = (epoch - self.tnhnc_start_epoch) / self.tnhnc_ramp_epochs
        return self.lambda_tnhnc * min(1.0, max(0.0, progress))

    def _bpcm_scale(self, epoch: int | None, global_update: int | None = None) -> float:
        if self.lambda_bpcm <= 0:
            return 0.0
        if self.use_update_clock and self.bpcm_start_update is not None and self.bpcm_ramp_updates is not None:
            if global_update is None or self.bpcm_ramp_updates == 0:
                return self.lambda_bpcm
            progress = (global_update - self.bpcm_start_update) / self.bpcm_ramp_updates
            return self.lambda_bpcm * min(1.0, max(0.0, progress))
        if epoch is None or self.bpcm_ramp_epochs == 0:
            return self.lambda_bpcm
        progress = (epoch - self.bpcm_start_epoch) / self.bpcm_ramp_epochs
        return self.lambda_bpcm * min(1.0, max(0.0, progress))

    def _fbhn_scale(self, epoch: int | None) -> float:
        if self.lambda_fbhn <= 0:
            return 0.0
        if epoch is None or self.fbhn_ramp_epochs == 0:
            return self.lambda_fbhn
        progress = (epoch - self.fbhn_start_epoch) / self.fbhn_ramp_epochs
        return self.lambda_fbhn * min(1.0, max(0.0, progress))

    def _ep_hcfc_scale(self, epoch: int | None) -> float:
        if self.lambda_ep_hcfc <= 0:
            return 0.0
        if epoch is None:
            return self.lambda_ep_hcfc
        if epoch < self.ep_hcfc_start_epoch:
            return 0.0
        if self.ep_hcfc_ramp_epochs == 0:
            return self.lambda_ep_hcfc
        progress = (epoch - self.ep_hcfc_start_epoch) / self.ep_hcfc_ramp_epochs
        return self.lambda_ep_hcfc * min(1.0, max(0.0, progress))

    def _cfps_scale(self, epoch: int | None) -> float:
        if self.lambda_cfps <= 0:
            return 0.0
        if epoch is None:
            return self.lambda_cfps
        if epoch <= self.cfps_start_epoch:
            return 0.0
        if self.cfps_ramp_epochs == 0:
            return self.lambda_cfps
        progress = (epoch - self.cfps_start_epoch) / self.cfps_ramp_epochs
        return self.lambda_cfps * min(1.0, max(0.0, progress))

    def _aabfp_scale(self, epoch: int | None) -> float:
        if self.lambda_aabfp <= 0:
            return 0.0
        if epoch is None:
            return self.lambda_aabfp
        if epoch <= self.aabfp_start_epoch:
            return 0.0
        if self.aabfp_ramp_epochs == 0:
            return self.lambda_aabfp
        progress = (epoch - self.aabfp_start_epoch) / self.aabfp_ramp_epochs
        return self.lambda_aabfp * min(1.0, max(0.0, progress))

    @staticmethod
    def _resized_target(target: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        return F.interpolate(target.float(), size=logits.shape[-2:], mode="nearest")

    def _aux_mask_loss(self, outputs: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        required = ("aux_mask_half", "aux_mask_quarter")
        missing = [name for name in required if name not in outputs]
        if missing:
            raise RuntimeError(f"lambda_aux_mask > 0 requires auxiliary outputs: {', '.join(missing)}.")
        half = self.primary_mask_loss(outputs["aux_mask_half"], self._resized_target(mask, outputs["aux_mask_half"]))
        quarter = self.primary_mask_loss(outputs["aux_mask_quarter"], self._resized_target(mask, outputs["aux_mask_quarter"]))
        return half + 0.5 * quarter

    def _aux_edge_loss(self, outputs: dict[str, torch.Tensor], edge: torch.Tensor) -> torch.Tensor:
        required = ("aux_edge_half", "aux_edge_quarter")
        missing = [name for name in required if name not in outputs]
        if missing:
            raise RuntimeError(f"lambda_aux_edge > 0 requires auxiliary outputs: {', '.join(missing)}.")
        half = self.edge_loss(outputs["aux_edge_half"], self._resized_target(edge, outputs["aux_edge_half"]))
        quarter = self.edge_loss(outputs["aux_edge_quarter"], self._resized_target(edge, outputs["aux_edge_quarter"]))
        return half + 0.5 * quarter

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        epoch: int | None = None,
        cfps_epoch: int | None = None,
        global_update: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        mask_loss = self.primary_mask_loss(outputs["mask"], batch["mask"])
        edge_loss = self.edge_loss(outputs["edge"], batch["edge_gt"])
        scnp_loss = self.scnp_mask_loss(self.scnp(outputs["mask"], batch["mask"]), batch["mask"])
        tangent_loss = outputs["mask"].new_zeros(())
        if self.lambda_tangent > 0:
            tangent_loss = self.tangent_loss(outputs, batch["tangent_gt"], batch["tangent_valid"])
        scnp_scale = self._scnp_scale(epoch, global_update)
        total = mask_loss + scnp_scale * scnp_loss + self.lambda_edge * edge_loss + self.lambda_tangent * tangent_loss
        tnhnc_scale = self._tnhnc_scale(epoch)
        tnhnc_loss = outputs["mask"].new_zeros(())
        if tnhnc_scale > 0:
            tnhnc_loss = self.tnhnc_loss(
                outputs["mask"],
                batch["mask"],
                batch["tangent_gt"],
                batch["tangent_valid"],
            )
            total = total + tnhnc_scale * tnhnc_loss
        bpcm_scale = self._bpcm_scale(epoch, global_update)
        bpcm_loss = outputs["mask"].new_zeros(())
        if bpcm_scale > 0:
            bpcm_loss = self.bpcm_loss(outputs["mask"], batch["mask"])
            total = total + bpcm_scale * bpcm_loss
        fbhn_scale = self._fbhn_scale(epoch)
        fbhn_loss = outputs["mask"].new_zeros(())
        if fbhn_scale > 0:
            fbhn_loss = self.fbhn_loss(outputs["mask"], batch["mask"])
            total = total + fbhn_scale * fbhn_loss
        ep_hcfc_scale = self._ep_hcfc_scale(epoch)
        ep_hcfc_loss = outputs["mask"].new_zeros(())
        if ep_hcfc_scale > 0:
            if "mask_feature" not in outputs:
                raise RuntimeError(
                    "lambda_ep_hcfc requires outputs['mask_feature']; enable "
                    "model.return_training_mask_feature=true."
                )
            ep_hcfc_loss = self.ep_hcfc_loss(
                outputs["mask_feature"],
                outputs["mask"],
                batch["mask"],
                batch["tangent_valid"],
            )
            total = total + ep_hcfc_scale * ep_hcfc_loss
        # train.py supplies the one-based logged epoch explicitly for CFPS;
        # direct callers retain the historical `epoch` behavior.
        cfps_scale = self._cfps_scale(epoch if cfps_epoch is None else cfps_epoch)
        cfps_loss = outputs["mask"].sum() * 0.0
        cfps_selected_pixels_per_image = 0.0
        cfps_active_image_ratio = 0.0
        if cfps_scale > 0:
            selected_cfps = self.cfps_loss.selected_mask(outputs["mask"], batch["mask"])
            selected_per_image = selected_cfps.flatten(1).sum(dim=1)
            cfps_selected_pixels_per_image = float(selected_per_image.float().mean().detach().cpu())
            cfps_active_image_ratio = float((selected_per_image > 0).float().mean().detach().cpu())
            cfps_loss = self.cfps_loss.loss_from_selected(outputs["mask"], selected_cfps)
            total = total + cfps_scale * cfps_loss
        aabfp_scale = self._aabfp_scale(epoch)
        aabfp_loss = outputs["mask"].new_zeros(())
        aabfp_boundary_selected_pixels_per_image = 0.0
        aabfp_bridge_selected_pixels_per_image = 0.0
        if aabfp_scale > 0:
            selected_aabfp, boundary_aabfp, bridge_aabfp = self.aabfp_loss.selected_masks(
                outputs["mask"], batch["mask"]
            )
            if bool(selected_aabfp.any()):
                aabfp_loss = self.aabfp_loss.loss_from_selected(outputs["mask"], selected_aabfp)
                aabfp_boundary_selected_pixels_per_image = float(
                    boundary_aabfp.flatten(1).sum(dim=1).float().mean().detach().cpu()
                )
                aabfp_bridge_selected_pixels_per_image = float(
                    bridge_aabfp.flatten(1).sum(dim=1).float().mean().detach().cpu()
                )
                total = total + aabfp_scale * aabfp_loss
        sdf_loss = outputs["mask"].new_zeros(())
        if self.lambda_sdf > 0:
            if "sdf" not in outputs:
                raise RuntimeError("lambda_sdf > 0 requires the model to output outputs['sdf'].")
            if "sdf_target" not in batch or "sdf_valid" not in batch:
                raise RuntimeError("lambda_sdf > 0 requires batch['sdf_target'] and batch['sdf_valid'].")
            sdf_loss = self.sdf_loss(outputs["sdf"], batch["sdf_target"], batch["sdf_valid"])
            total = total + self.lambda_sdf * sdf_loss
        aux_mask_loss = outputs["mask"].new_zeros(())
        aux_edge_loss = outputs["mask"].new_zeros(())
        if self.lambda_aux_mask > 0:
            aux_mask_loss = self._aux_mask_loss(outputs, batch["mask"])
            total = total + self.lambda_aux_mask * aux_mask_loss
        if self.lambda_aux_edge > 0:
            aux_edge_loss = self._aux_edge_loss(outputs, batch["edge_gt"])
            total = total + self.lambda_aux_edge * aux_edge_loss
        component_query_loss = outputs["mask"].new_zeros(())
        if self.lambda_component_query > 0:
            required_outputs = ("component_mask_logits", "component_class_logits")
            missing_outputs = [name for name in required_outputs if name not in outputs]
            if missing_outputs:
                raise RuntimeError(
                    "lambda_component_query > 0 requires model outputs: " + ", ".join(missing_outputs) + "."
                )
            if "component_id_map" not in batch:
                raise RuntimeError("lambda_component_query > 0 requires batch['component_id_map'].")
            component_query_loss = self.component_query_loss(
                outputs["component_mask_logits"],
                outputs["component_class_logits"],
                batch["component_id_map"],
            )
            total = total + self.lambda_component_query * component_query_loss
        skel_recall_loss = outputs["mask"].new_zeros(())
        skel_recall_value = outputs["mask"].new_zeros(())
        if self.lambda_skel_recall > 0:
            if "tangent_valid" not in batch:
                raise RuntimeError(
                    "lambda_skel_recall > 0 requires batch['tangent_valid'], i.e. the cached "
                    "structural targets (scripts/precompute_structural_targets.py)."
                )
            skel_recall_loss, skel_recall_value = self.skel_recall_loss(
                outputs["mask"], batch["tangent_valid"]
            )
            total = total + self.lambda_skel_recall * skel_recall_loss
        items = {
            "loss": float(total.detach().cpu()),
            "mask_loss": float(mask_loss.detach().cpu()),
            "edge_loss": float(edge_loss.detach().cpu()),
            "scnp_loss": float(scnp_loss.detach().cpu()),
            "tangent_loss": float(tangent_loss.detach().cpu()),
            "scnp_scale": scnp_scale,
            "tnhnc_loss": float(tnhnc_loss.detach().cpu()),
            "tnhnc_scale": tnhnc_scale,
            "lambda_tnhnc": self.lambda_tnhnc,
            "bpcm_loss": float(bpcm_loss.detach().cpu()),
            "bpcm_scale": bpcm_scale,
            "lambda_bpcm": self.lambda_bpcm,
            "fbhn_loss": float(fbhn_loss.detach().cpu()),
            "fbhn_scale": fbhn_scale,
            "lambda_fbhn": self.lambda_fbhn,
            "ep_hcfc_loss": float(ep_hcfc_loss.detach().cpu()),
            "ep_hcfc_scale": ep_hcfc_scale,
            "lambda_ep_hcfc": self.lambda_ep_hcfc,
            "cfps_loss": float(cfps_loss.detach().cpu()),
            "cfps_weighted_loss": float((cfps_scale * cfps_loss).detach().cpu()),
            "cfps_scale": cfps_scale,
            "lambda_cfps": self.lambda_cfps,
            "cfps_selected_pixels_per_image": cfps_selected_pixels_per_image,
            "cfps_active_image_ratio": cfps_active_image_ratio,
            "aabfp_loss": float(aabfp_loss.detach().cpu()),
            "aabfp_scale": aabfp_scale,
            "lambda_aabfp": self.lambda_aabfp,
            "aabfp_boundary_selected_pixels_per_image": aabfp_boundary_selected_pixels_per_image,
            "aabfp_bridge_selected_pixels_per_image": aabfp_bridge_selected_pixels_per_image,
            "sdf_loss": float(sdf_loss.detach().cpu()),
            "lambda_sdf": self.lambda_sdf,
            "aux_mask_loss": float(aux_mask_loss.detach().cpu()),
            "aux_edge_loss": float(aux_edge_loss.detach().cpu()),
            "lambda_aux_mask": self.lambda_aux_mask,
            "lambda_aux_edge": self.lambda_aux_edge,
            "skel_recall_loss": float(skel_recall_loss.detach().cpu()),
            "skel_recall_value": float(skel_recall_value.detach().cpu()),
            "lambda_skel_recall": self.lambda_skel_recall,
        }
        if self.lambda_component_query > 0:
            items["component_query_loss"] = float(component_query_loss.detach().cpu())
            items["lambda_component_query"] = self.lambda_component_query
        return total, items


# Compatibility alias; the new route loss supersedes hard-label direction CE.
DirectionLoss = TangentRouteLoss
