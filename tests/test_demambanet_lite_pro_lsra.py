from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import DEMambaNetLiteProLSRA, build_model
from models.de_mambanet_lite_pro_lsra import LocalSemanticReassemblyAttention
from models.mamba_blocks import SequenceMambaCore


OUTPUT_KEYS = {"mask", "edge", "reliability", "route_s1", "route_s3", "route_s4"}
CONFIG_PATH = (
    REPO_ROOT
    / "configs/experiments/lightweight_demambanet_lite_pro_lsra"
    / "DE_MAMBANET_LITE_PRO_LSRA_LR150_DEEPCRACK_NATIVE384x544_EMA_BPCM.yaml"
)
BASELINE_CONFIG_PATH = (
    REPO_ROOT
    / "configs/experiments/lightweight_demambanet_lite_pro"
    / "DE_MAMBANET_LITE_PRO_LR150_DEEPCRACK_NATIVE384x544_EMA_BPCM.yaml"
)


def _model(*, use_mamba_ssm: bool = False) -> DEMambaNetLiteProLSRA:
    return build_model(
        "DEMambaNetLiteProLSRA",
        use_mamba_ssm=use_mamba_ssm,
        drop_path_rate=0.0,
    ).eval()


def test_registry_builds_lsra_model() -> None:
    assert isinstance(_model(), DEMambaNetLiteProLSRA)


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
def test_local_attention_contract_and_center_bias() -> None:
    lsra = LocalSemanticReassemblyAttention().eval()
    local_half, attention = lsra(
        torch.randn(2, 80, 16, 20),
        torch.randn(2, 40, 32, 40),
        return_attention=True,
    )
    assert local_half.shape == (2, 40, 32, 40)
    assert attention.shape == (2, 4, 9, 16 * 20)
    assert attention.shape[-2] == 9  # local neighborhood, never an HW-by-HW matrix
    assert torch.allclose(attention.sum(dim=2), torch.ones_like(attention[:, :, 0, :]), atol=1e-6)
    center = lsra.local_attention.neighborhood // 2
    assert torch.equal(lsra.local_attention.relative_bias[:, center], torch.full((4,), 2.0))
    assert torch.equal(
        lsra.local_attention.relative_bias[:, [index for index in range(9) if index != center]],
        torch.zeros(4, 8),
    )


@torch.no_grad()
def test_residual_scale_and_debug_values_are_bounded_and_detached() -> None:
    model = _model()
    scale = model.decoder.lsra.residual_scale()
    assert torch.isclose(scale, torch.tensor(0.1), atol=1e-6)
    assert 0.0 < scale.item() < 0.5
    outputs, debug = model(torch.randn(1, 3, 64, 80), return_debug=True)
    assert set(outputs) == OUTPUT_KEYS
    assert set(debug) == {
        "residual_scale",
        "attention_center_mean",
        "attention_entropy",
        "correction_to_base_norm_ratio",
    }
    assert all(not value.requires_grad and torch.isfinite(value).all() for value in debug.values())


def test_qkv_output_projection_and_scale_receive_finite_nonzero_gradients() -> None:
    model = _model().train()
    outputs = model(torch.randn(1, 3, 64, 80))
    loss = sum(value.float().mean() for value in outputs.values())
    loss.backward()
    lsra = model.decoder.lsra
    parameters = (
        lsra.local_attention.query_projection.weight,
        lsra.local_attention.key_projection.weight,
        lsra.local_attention.value_projection.weight,
        lsra.output_projection.weight,
        lsra.scale_parameter,
    )
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum().item() > 0.0


@torch.no_grad()
def test_autocast_fp16_has_finite_outputs() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _model().to(device)
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
    with autocast:
        outputs = model(torch.randn(1, 3, 64, 80, device=device))
    assert all(torch.isfinite(value).all() for value in outputs.values())


def test_context_topology_and_requested_mamba_backend() -> None:
    model = _model(use_mamba_ssm=True)
    cores = [module for module in model.modules() if isinstance(module, SequenceMambaCore)]
    assert len(cores) == 2
    backend_info = model.mamba_backend_info()
    assert backend_info == {
        "stage3_context": [model.stage3_context.backend],
        "stage4_context": [model.stage4_context.backend],
    }
    assert model.stage3_context.backend == "mamba_ssm"
    assert model.stage4_context.backend == "mamba_ssm"


def test_parameter_growth_is_bounded() -> None:
    model = _model()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert parameter_count < 3_300_000


def test_config_changes_only_model_and_save_sections() -> None:
    import yaml

    with BASELINE_CONFIG_PATH.open(encoding="utf-8") as handle:
        baseline = yaml.safe_load(handle)
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        candidate = yaml.safe_load(handle)
    for section in ("data", "train", "loss", "evaluation"):
        assert candidate[section] == baseline[section], section
    candidate_model = dict(candidate["model"])
    assert candidate_model.pop("name") == "DEMambaNetLiteProLSRA"
    assert candidate_model.pop("lsra_channels") == 24
    assert candidate_model.pop("lsra_heads") == 4
    assert candidate_model.pop("lsra_kernel_size") == 3
    assert candidate_model.pop("lsra_initial_scale") == 0.1
    baseline_model = dict(baseline["model"])
    baseline_model.pop("name")
    assert candidate_model == baseline_model
    assert candidate["save"] != baseline["save"]


if __name__ == "__main__":
    tests = (
        test_registry_builds_lsra_model,
        test_forward_contract_at_native_and_table_sizes,
        test_local_attention_contract_and_center_bias,
        test_residual_scale_and_debug_values_are_bounded_and_detached,
        test_qkv_output_projection_and_scale_receive_finite_nonzero_gradients,
        test_autocast_fp16_has_finite_outputs,
        test_context_topology_and_requested_mamba_backend,
        test_parameter_growth_is_bounded,
        test_config_changes_only_model_and_save_sections,
    )
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
