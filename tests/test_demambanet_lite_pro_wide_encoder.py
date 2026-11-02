from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import DEMambaNetLitePro, DEMambaNetLiteProWideEncoder, build_model
from models.de_mambanet_lite import ConvGNAct, StructureDirectionalBlock
from models.de_mambanet_lite_pro import ProStructuralDecoder
from models.de_mambanet_lite_pro_pcr import PartialStructureDirectionalBlock
from models.mamba_blocks import SequenceMambaCore


OUTPUT_KEYS = {"mask", "edge", "reliability", "route_s1", "route_s3", "route_s4"}
CONFIG_PATH = (
    REPO_ROOT
    / "configs/experiments/lightweight_demambanet_lite_pro_wide_encoder"
    / "DE_MAMBANET_LITE_PRO_WIDE_ENCODER_LR150_DEEPCRACK_NATIVE384x544_EMA_BPCM.yaml"
)
BASELINE_CONFIG_PATH = (
    REPO_ROOT
    / "configs/experiments/lightweight_demambanet_lite_pro"
    / "DE_MAMBANET_LITE_PRO_LR150_DEEPCRACK_NATIVE384x544_EMA_BPCM.yaml"
)


def _model(*, use_mamba_ssm: bool = False) -> DEMambaNetLiteProWideEncoder:
    return build_model(
        "DEMambaNetLiteProWideEncoder",
        use_mamba_ssm=use_mamba_ssm,
        drop_path_rate=0.0,
    ).eval()


def test_registry_builds_wide_encoder_model() -> None:
    assert isinstance(_model(), DEMambaNetLiteProWideEncoder)


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


def test_full_encoder_and_original_pro_decoder_topology() -> None:
    model = _model()
    assert model.encoder_widths == (48, 96, 192, 256)
    assert model.encoder_depths == (2, 2, 4, 2)
    for stage in (model.stage1, model.stage2, model.stage3, model.stage4):
        assert all(isinstance(block, StructureDirectionalBlock) for block in stage)
        assert not any(isinstance(block, PartialStructureDirectionalBlock) for block in stage)
    for downsample, in_channels, out_channels in (
        (model.down1, 48, 96),
        (model.down2, 96, 192),
        (model.down3, 192, 256),
    ):
        assert isinstance(downsample, ConvGNAct)
        convolution = downsample.block[0]
        assert convolution.kernel_size == (3, 3)
        assert convolution.stride == (2, 2)
        assert convolution.in_channels == in_channels
        assert convolution.out_channels == out_channels
    assert isinstance(model.decoder, ProStructuralDecoder)
    assert model.decoder.proj1.block[0].out_channels == 80
    assert model.decoder.mask_projection.block[0].in_channels == 48
    assert model.decoder.mask_projection.block[0].out_channels == 64


def test_no_retired_decoder_or_attention_modules_are_present() -> None:
    model = _model()
    forbidden_fragments = ("CSDR", "LSRA", "PTCA", "SEGR", "FBPV", "Partial")
    names = [module.__class__.__name__ for module in model.modules()]
    assert not any(fragment in name for name in names for fragment in forbidden_fragments)


def test_requested_mamba_backend_and_backward_are_valid() -> None:
    mamba_model = _model(use_mamba_ssm=True)
    cores = [module for module in mamba_model.modules() if isinstance(module, SequenceMambaCore)]
    assert len(cores) == 2
    assert mamba_model.mamba_backend_info() == {
        "stage3_context": ["mamba_ssm"],
        "stage4_context": ["mamba_ssm"],
    }

    model = _model().train()
    outputs = model(torch.randn(1, 3, 64, 80))
    loss = sum(value.float().mean() for value in outputs.values())
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_parameter_capacity_is_between_litepro_and_csdr_budget() -> None:
    lite_pro_params = sum(parameter.numel() for parameter in DEMambaNetLitePro(use_mamba_ssm=False).parameters())
    wide_params = sum(parameter.numel() for parameter in _model().parameters())
    assert wide_params > lite_pro_params
    assert wide_params < 4_000_000


def test_config_preserves_litepro_protocol() -> None:
    import yaml

    with BASELINE_CONFIG_PATH.open(encoding="utf-8") as handle:
        baseline = yaml.safe_load(handle)
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        candidate = yaml.safe_load(handle)
    for section in ("data", "train", "loss", "evaluation"):
        assert candidate[section] == baseline[section], section
    assert candidate["model"]["name"] == "DEMambaNetLiteProWideEncoder"
    assert candidate["model"]["encoder_widths"] == [48, 96, 192, 256]
    assert candidate["model"]["decoder_channels"] == 80
    assert candidate["save"] != baseline["save"]


if __name__ == "__main__":
    tests = (
        test_registry_builds_wide_encoder_model,
        test_forward_contract_at_native_and_table_sizes,
        test_full_encoder_and_original_pro_decoder_topology,
        test_no_retired_decoder_or_attention_modules_are_present,
        test_requested_mamba_backend_and_backward_are_valid,
        test_parameter_capacity_is_between_litepro_and_csdr_budget,
        test_config_preserves_litepro_protocol,
    )
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
