from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import DEMambaNetLitePro, DEMambaNetLiteProPTCA, build_model
from models.de_mambanet_lite_pro_ptca import ParallelTransposedChannelAttention
from models.mamba_blocks import SequenceMambaCore


OUTPUT_KEYS = {"mask", "edge", "reliability", "route_s1", "route_s3", "route_s4"}
CONFIG_PATH = (
    REPO_ROOT
    / "configs/experiments/lightweight_demambanet_lite_pro_ptca"
    / "DE_MAMBANET_LITE_PRO_PTCA_LR150_DEEPCRACK_NATIVE384x544_EMA_BPCM.yaml"
)
BASELINE_CONFIG_PATH = (
    REPO_ROOT
    / "configs/experiments/lightweight_demambanet_lite_pro"
    / "DE_MAMBANET_LITE_PRO_LR150_DEEPCRACK_NATIVE384x544_EMA_BPCM.yaml"
)


def _model(*, use_mamba_ssm: bool = False) -> DEMambaNetLiteProPTCA:
    return build_model(
        "DEMambaNetLiteProPTCA",
        use_mamba_ssm=use_mamba_ssm,
        drop_path_rate=0.0,
    ).eval()


def test_registry_builds_ptca_model() -> None:
    assert isinstance(_model(), DEMambaNetLiteProPTCA)


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


@torch.no_grad()
def test_ptca_uses_transposed_channel_attention_not_spatial_attention() -> None:
    ptca = ParallelTransposedChannelAttention(160, attention_channels=40, num_heads=4).eval()
    output, attention = ptca(torch.randn(2, 160, 24, 34), return_attention=True)
    assert output.shape == (2, 160, 24, 34)
    assert ptca.head_dim == 10
    assert attention.shape == (2, 4, 10, 10)
    assert attention.shape[-1] != 24 * 34
    assert torch.allclose(attention.sum(dim=-1), torch.ones_like(attention[..., 0]))


def test_ptca_requires_channels_divisible_by_heads() -> None:
    try:
        ParallelTransposedChannelAttention(160, attention_channels=40, num_heads=3)
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("PTCA must reject attention channels that cannot be split across heads.")


@torch.no_grad()
def test_zero_alpha_exactly_preserves_stage3_mamba_output() -> None:
    model = _model()
    assert model.alpha.item() == 0.0
    image = torch.randn(1, 3, 64, 80)
    stem_half = model.stem_half(image)
    detail_half = model.detail_refine(stem_half + model.detail_branch(image, stem_half.shape[-2:]))
    f1 = model.stage1(model.stem_quarter(detail_half))
    f2 = model.stage2(model.down1(f1))
    f3_local = model.stage3(model.down2(f2))
    expected_f3, _ = model.stage3_context(f3_local)
    actual_f3 = model.extract_encoder_features(image)["f3"]
    assert torch.equal(actual_f3, expected_f3)


def test_only_two_shared_mamba_context_cores_remain() -> None:
    model = _model()
    cores = [module for module in model.modules() if isinstance(module, SequenceMambaCore)]
    assert len(cores) == 2
    for context in (model.stage3_context, model.stage4_context):
        assert context.shared_core
        assert context.scan_count == 2
        assert context.active_mamba_directions == ("horizontal", "vertical")


def test_backward_is_finite() -> None:
    model = _model().train()
    outputs = model(torch.randn(1, 3, 64, 80))
    loss = sum(value.float().mean() for value in outputs.values())
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_requested_mamba_backend_is_reported() -> None:
    model = _model(use_mamba_ssm=True)
    backend_info = model.mamba_backend_info()
    assert backend_info["stage3_context"] == [model.stage3_context.backend]
    assert backend_info["stage4_context"] == [model.stage4_context.backend]
    assert set(backend_info["stage3_context"] + backend_info["stage4_context"]).issubset(
        {"mamba_ssm", "fallback"}
    )


def test_parameter_growth_is_bounded() -> None:
    baseline = DEMambaNetLitePro(use_mamba_ssm=False, drop_path_rate=0.0)
    candidate = _model()
    added = sum(parameter.numel() for parameter in candidate.parameters()) - sum(
        parameter.numel() for parameter in baseline.parameters()
    )
    assert 0 < added < 100_000
    assert sum(parameter.numel() for parameter in candidate.parameters()) < 3_400_000


def test_config_changes_only_model_and_save_sections() -> None:
    import yaml

    with BASELINE_CONFIG_PATH.open(encoding="utf-8") as handle:
        baseline = yaml.safe_load(handle)
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        candidate = yaml.safe_load(handle)
    for section in ("data", "train", "loss", "evaluation"):
        assert candidate[section] == baseline[section], section
    candidate_model = dict(candidate["model"])
    candidate_model.pop("name")
    assert candidate_model.pop("ptca_channels") == 40
    assert candidate_model.pop("ptca_heads") == 4
    baseline_model = dict(baseline["model"])
    baseline_model.pop("name")
    assert candidate_model == baseline_model
    assert candidate["save"] != baseline["save"]


if __name__ == "__main__":
    tests = (
        test_registry_builds_ptca_model,
        test_forward_contract_at_native_and_table_sizes,
        test_ptca_uses_transposed_channel_attention_not_spatial_attention,
        test_ptca_requires_channels_divisible_by_heads,
        test_zero_alpha_exactly_preserves_stage3_mamba_output,
        test_only_two_shared_mamba_context_cores_remain,
        test_backward_is_finite,
        test_requested_mamba_backend_is_reported,
        test_parameter_growth_is_bounded,
        test_config_changes_only_model_and_save_sections,
    )
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
