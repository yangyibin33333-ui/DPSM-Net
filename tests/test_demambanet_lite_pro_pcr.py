from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import DEMambaNetLiteProPCR, build_model
from models.de_mambanet_lite import StructureDirectionalBlock
from models.de_mambanet_lite_pro_pcr import (
    LiteDepthwiseDownsample,
    PartialStructureDirectionalBlock,
)
from models.mamba_blocks import SequenceMambaCore


OUTPUT_KEYS = {"mask", "edge", "reliability", "route_s1", "route_s3", "route_s4"}
CONFIG_PATH = (
    REPO_ROOT
    / "configs/experiments/lightweight_demambanet_lite_pro_pcr"
    / "DE_MAMBANET_LITE_PRO_PCR_LR150_DEEPCRACK_NATIVE384x544_EMA_BPCM.yaml"
)
BASELINE_CONFIG_PATH = (
    REPO_ROOT
    / "configs/experiments/lightweight_demambanet_lite_pro"
    / "DE_MAMBANET_LITE_PRO_LR150_DEEPCRACK_NATIVE384x544_EMA_BPCM.yaml"
)


def _model(*, use_mamba_ssm: bool = False) -> DEMambaNetLiteProPCR:
    return build_model(
        "DEMambaNetLiteProPCR",
        use_mamba_ssm=use_mamba_ssm,
        drop_path_rate=0.0,
    ).eval()


def test_registry_builds_pcr_model() -> None:
    assert isinstance(_model(), DEMambaNetLiteProPCR)


@torch.no_grad()
def test_forward_contract_at_native_and_table_sizes() -> None:
    model = _model()
    for height, width in ((384, 544), (512, 512)):
        outputs = model(torch.randn(1, 3, height, width))
        assert set(outputs) == OUTPUT_KEYS
        for name in ("mask", "edge", "reliability"):
            assert outputs[name].shape == (1, 1, height, width)
            assert torch.isfinite(outputs[name]).all()
        assert outputs["route_s1"].shape == (1, 4, height // 4, width // 4)
        assert outputs["route_s3"].shape == (1, 4, height // 16, width // 16)
        assert outputs["route_s4"].shape == (1, 4, height // 32, width // 32)


def test_stage_types_active_channels_and_bypass_layout() -> None:
    model = _model()
    assert all(isinstance(block, StructureDirectionalBlock) for block in model.stage1)
    expected = ((model.stage2, 72, 24), (model.stage3, 96, 96), (model.stage4, 128, 128))
    for stage, active_channels, bypass_channels in expected:
        assert all(isinstance(block, PartialStructureDirectionalBlock) for block in stage)
        for block in stage:
            assert block.active_channels == active_channels
            assert block.bypass_channels == bypass_channels
            assert block.active_channels % 8 == 0
            assert block.active_branch.input_norm.num_channels == active_channels
            assert block.active_branch.output_projection.out_channels == active_channels
            assert block.channel_mixer[0].in_channels == block.channels


@torch.no_grad()
def test_bypass_channels_skip_directional_branch_and_downsample_shapes() -> None:
    block = PartialStructureDirectionalBlock(96, active_ratio=0.75, drop_path=0.0).eval()
    observed: dict[str, tuple[int, ...]] = {}

    def capture_active_input(_: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        observed["active"] = tuple(inputs[0].shape)

    handle = block.active_branch.register_forward_pre_hook(capture_active_input)
    try:
        block(torch.randn(1, 96, 8, 10))
    finally:
        handle.remove()
    assert observed["active"] == (1, 72, 8, 10)
    assert block.bypass_channels == 24

    model = _model()
    assert isinstance(model.down1, LiteDepthwiseDownsample)
    assert isinstance(model.down2, LiteDepthwiseDownsample)
    assert isinstance(model.down3, LiteDepthwiseDownsample)
    features = model.extract_encoder_features(torch.randn(1, 3, 384, 544))
    assert features["f1"].shape == (1, 40, 96, 136)
    assert features["f2"].shape == (1, 96, 48, 68)
    assert features["f3"].shape == (1, 192, 24, 34)
    assert features["f4"].shape == (1, 256, 12, 17)


def test_requested_mamba_backend_and_topology() -> None:
    model = _model(use_mamba_ssm=True)
    cores = [module for module in model.modules() if isinstance(module, SequenceMambaCore)]
    assert len(cores) == 2
    assert model.stage3_context.backend == "mamba_ssm"
    assert model.stage4_context.backend == "mamba_ssm"
    assert model.mamba_backend_info() == {
        "stage3_context": ["mamba_ssm"],
        "stage4_context": ["mamba_ssm"],
    }


def test_backward_has_finite_active_branch_and_mixer_gradients() -> None:
    model = _model().train()
    outputs = model(torch.randn(1, 3, 64, 80))
    loss = sum(value.float().mean() for value in outputs.values())
    loss.backward()
    parameters = (
        model.stage2[0].active_branch.input_projection.weight,
        model.stage3[0].active_branch.direction_gate.weight,
        model.stage4[0].channel_mixer[0].weight,
    )
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum().item() > 0.0


def test_parameter_budget_is_bounded() -> None:
    model = _model()
    assert sum(parameter.numel() for parameter in model.parameters()) < 3_800_000


def test_config_changes_only_model_and_save_sections() -> None:
    import yaml

    with BASELINE_CONFIG_PATH.open(encoding="utf-8") as handle:
        baseline = yaml.safe_load(handle)
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        candidate = yaml.safe_load(handle)
    for section in ("data", "train", "loss", "evaluation"):
        assert candidate[section] == baseline[section], section
    candidate_model = dict(candidate["model"])
    assert candidate_model.pop("name") == "DEMambaNetLiteProPCR"
    for name, value in {
        "partial_expansion_ratio": 1.5,
        "stage2_active_ratio": 0.75,
        "stage3_active_ratio": 0.5,
        "stage4_active_ratio": 0.5,
    }.items():
        assert candidate_model.pop(name) == value
    assert candidate_model["encoder_widths"] == [40, 96, 192, 256]
    assert candidate_model["stage3_mamba_channels"] == 112
    assert candidate_model["stage4_mamba_channels"] == 144
    assert candidate["save"] != baseline["save"]


if __name__ == "__main__":
    tests = (
        test_registry_builds_pcr_model,
        test_forward_contract_at_native_and_table_sizes,
        test_stage_types_active_channels_and_bypass_layout,
        test_bypass_channels_skip_directional_branch_and_downsample_shapes,
        test_requested_mamba_backend_and_topology,
        test_backward_has_finite_active_branch_and_mixer_gradients,
        test_parameter_budget_is_bounded,
        test_config_changes_only_model_and_save_sections,
    )
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
