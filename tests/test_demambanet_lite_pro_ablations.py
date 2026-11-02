from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import (
    DEMambaNetLitePro,
    DEMambaNetLiteProNarrow,
    DEMambaNetLiteProNoS3,
    build_model,
)
from models.mamba_blocks import SequenceMambaCore


OUTPUT_KEYS = {"mask", "edge", "reliability", "route_s1", "route_s3", "route_s4"}


def _model(name: str) -> nn.Module:
    return build_model(name, use_mamba_ssm=False, drop_path_rate=0.0).eval()


def test_registry_builds_both_ablations() -> None:
    assert isinstance(_model("DEMambaNetLiteProNoS3"), DEMambaNetLiteProNoS3)
    assert isinstance(_model("DEMambaNetLiteProNarrow"), DEMambaNetLiteProNarrow)


@torch.no_grad()
def test_512_forward_contract_for_both_ablations() -> None:
    image = torch.randn(1, 3, 512, 512)
    for name in ("DEMambaNetLiteProNoS3", "DEMambaNetLiteProNarrow"):
        model = _model(name)
        outputs = model(image)
        assert set(outputs) == OUTPUT_KEYS
        for output_name in ("mask", "edge", "reliability"):
            assert outputs[output_name].shape == (1, 1, 512, 512)
            assert torch.isfinite(outputs[output_name]).all()
        assert outputs["route_s1"].shape == (1, 4, 128, 128)
        assert outputs["route_s3"].shape == (1, 4, 32, 32)
        assert outputs["route_s4"].shape == (1, 4, 16, 16)
        del model, outputs


def test_mamba_core_counts_and_context_topology() -> None:
    no_s3 = _model("DEMambaNetLiteProNoS3")
    assert not hasattr(no_s3, "stage3_context")
    assert len([module for module in no_s3.modules() if isinstance(module, SequenceMambaCore)]) == 1
    assert no_s3.stage4_context.shared_core
    assert no_s3.stage4_context.scan_count == 2
    assert no_s3.stage4_context.active_mamba_directions == ("horizontal", "vertical")
    assert no_s3.mamba_backend_info() == {
        "stage3_context": ["disabled"],
        "stage4_context": ["fallback"],
    }

    narrow = _model("DEMambaNetLiteProNarrow")
    assert len([module for module in narrow.modules() if isinstance(module, SequenceMambaCore)]) == 2
    for context in (narrow.stage3_context, narrow.stage4_context):
        assert len([module for module in context.modules() if isinstance(module, SequenceMambaCore)]) == 1
        assert context.shared_core
        assert context.scan_count == 2
        assert context.active_mamba_directions == ("horizontal", "vertical")


@torch.no_grad()
def test_no_s3_route_head_does_not_feed_the_main_path() -> None:
    model = _model("DEMambaNetLiteProNoS3")
    image = torch.randn(1, 3, 64, 80)
    pointers: dict[str, int] = {}

    def capture_stage3(_: nn.Module, __: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        pointers["stage3"] = output.data_ptr()

    def capture_input(name: str):
        def hook(_: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            pointers[name] = inputs[0].data_ptr()

        return hook

    handles = (
        model.stage3.register_forward_hook(capture_stage3),
        model.down3.register_forward_pre_hook(capture_input("down3")),
        model.route_s3_head.register_forward_pre_hook(capture_input("route")),
    )
    try:
        before = model(image)
    finally:
        for handle in handles:
            handle.remove()
    assert pointers["stage3"] == pointers["down3"] == pointers["route"]

    model.route_s3_head.projection.bias.add_(1.0)
    after = model(image)
    for name in ("mask", "edge", "reliability", "route_s1", "route_s4"):
        assert torch.equal(before[name], after[name])
    assert not torch.equal(before["route_s3"], after["route_s3"])


def test_narrow_widths_and_group_norm_are_valid() -> None:
    model = _model("DEMambaNetLiteProNarrow")
    assert model.encoder_widths == (32, 64, 128, 192)
    assert model.encoder_depths == (2, 2, 4, 2)
    assert model.stage3_context.input_projection.block[0].out_channels == 80
    assert model.stage4_context.input_projection.block[0].out_channels == 96
    assert model.decoder.proj1.block[0].out_channels == 64
    assert model.decoder.half_upsample[0].out_channels == 32 * 4
    assert model.decoder.mask_projection.block[0].out_channels == 48
    for module in model.modules():
        if isinstance(module, nn.GroupNorm):
            assert module.num_channels % module.num_groups == 0


def test_both_ablations_backpropagate_finite_gradients() -> None:
    for name in ("DEMambaNetLiteProNoS3", "DEMambaNetLiteProNarrow"):
        model = _model(name).train()
        outputs = model(torch.randn(1, 3, 64, 80))
        loss = sum(value.float().mean() for value in outputs.values())
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        del model, outputs, loss, gradients


@torch.no_grad()
def test_original_pro_structure_parameters_and_forward_remain_stable() -> None:
    torch.manual_seed(123)
    before = DEMambaNetLitePro(use_mamba_ssm=False, drop_path_rate=0.0).eval()
    before_shapes = {name: tuple(value.shape) for name, value in before.state_dict().items()}
    before_count = sum(parameter.numel() for parameter in before.parameters())
    del before

    _model("DEMambaNetLiteProNoS3")
    _model("DEMambaNetLiteProNarrow")

    torch.manual_seed(123)
    after = DEMambaNetLitePro(use_mamba_ssm=False, drop_path_rate=0.0).eval()
    after_shapes = {name: tuple(value.shape) for name, value in after.state_dict().items()}
    after_count = sum(parameter.numel() for parameter in after.parameters())
    outputs = after(torch.randn(1, 3, 64, 80))
    assert before_shapes == after_shapes
    assert before_count == after_count
    assert set(outputs) == OUTPUT_KEYS
    assert len([module for module in after.modules() if isinstance(module, SequenceMambaCore)]) == 2


def test_configs_change_only_model_and_save_sections() -> None:
    import yaml

    root = Path(__file__).resolve().parents[1]
    paths = (
        root / "configs/experiments/lightweight_demambanet_lite_pro/DE_MAMBANET_LITE_PRO_LR150_DEEPCRACK_NATIVE384x544_EMA_BPCM.yaml",
        root / "configs/experiments/lightweight_demambanet_lite_pro_ablations/DE_MAMBANET_LITE_PRO_NOS3_LR150_DEEPCRACK_NATIVE384x544_EMA_BPCM.yaml",
        root / "configs/experiments/lightweight_demambanet_lite_pro_ablations/DE_MAMBANET_LITE_PRO_NARROW_LR150_DEEPCRACK_NATIVE384x544_EMA_BPCM.yaml",
    )
    configs = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            configs.append(yaml.safe_load(handle))
    baseline, no_s3, narrow = configs
    for candidate in (no_s3, narrow):
        for section in ("data", "train", "loss", "evaluation"):
            assert candidate[section] == baseline[section], section
        assert candidate["model"] != baseline["model"]
        assert candidate["save"] != baseline["save"]


if __name__ == "__main__":
    tests = (
        test_registry_builds_both_ablations,
        test_512_forward_contract_for_both_ablations,
        test_mamba_core_counts_and_context_topology,
        test_no_s3_route_head_does_not_feed_the_main_path,
        test_narrow_widths_and_group_norm_are_valid,
        test_both_ablations_backpropagate_finite_gradients,
        test_original_pro_structure_parameters_and_forward_remain_stable,
        test_configs_change_only_model_and_save_sections,
    )
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
