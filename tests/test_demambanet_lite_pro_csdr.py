from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import DEMambaNetLitePro, DEMambaNetLiteProCSDR, build_model
from models.de_mambanet_lite_pro_csdr import CSDRFusion
from models.mamba_blocks import SequenceMambaCore


OUTPUT_KEYS = {"mask", "edge", "reliability", "route_s1", "route_s3", "route_s4"}


def _model() -> DEMambaNetLiteProCSDR:
    return DEMambaNetLiteProCSDR(use_mamba_ssm=False, drop_path_rate=0.0).eval()


def test_registry_contract() -> None:
    assert isinstance(build_model("DEMambaNetLiteProCSDR", use_mamba_ssm=False), DEMambaNetLiteProCSDR)


@torch.no_grad()
def test_native_forward_contracts() -> None:
    for height, width in ((512, 512), (384, 544)):
        model = _model()
        outputs = model(torch.randn(1, 3, height, width))
        assert set(outputs) == OUTPUT_KEYS
        for name in ("mask", "edge", "reliability"):
            assert outputs[name].shape == (1, 1, height, width)
        assert outputs["route_s1"].shape[-2:] == (height // 4, width // 4)
        assert outputs["route_s3"].shape[-2:] == (height // 16, width // 16)
        assert outputs["route_s4"].shape[-2:] == (height // 32, width // 32)


@torch.no_grad()
def test_csdr_fusion_weights_and_detail_gate_ranges() -> None:
    model = _model()
    fusion = model.csdr_decoder.fuse1
    assert isinstance(fusion, CSDRFusion)
    deep = torch.randn(2, 96, 8, 10)
    weights = fusion.branch_weights(deep)
    assert weights.shape == (2, 3, 1, 1)
    assert torch.allclose(weights.sum(dim=1), torch.ones_like(weights[:, :1]), atol=1e-6)
    for branch in (fusion.local, fusion.horizontal, fusion.vertical):
        assert branch.groups == 96
    assert fusion.horizontal.kernel_size == (1, 7)
    assert fusion.vertical.kernel_size == (7, 1)

    _, debug = model(torch.randn(1, 3, 64, 80), return_debug=True)
    assert torch.isfinite(debug["detail_gate"]).all()
    assert torch.all(debug["detail_gate"] >= 0.125 - 1e-6)
    assert torch.all(debug["detail_gate"] <= 1.0 + 1e-6)


def test_two_mamba_cores_and_no_csdr_mamba_or_attention() -> None:
    model = _model()
    assert len([module for module in model.modules() if isinstance(module, SequenceMambaCore)]) == 2
    forbidden = ("Mamba", "Attention", "Transformer")
    assert not any(
        any(fragment in type(module).__name__ for fragment in forbidden)
        for module in model.csdr_decoder.modules()
    )


def test_mask_loss_cannot_update_route_or_edge_heads() -> None:
    model = _model().train()
    outputs = model(torch.randn(1, 3, 64, 80))
    outputs["mask"].mean().backward()
    assert model.csdr_decoder.mask_head.weight.grad is not None
    assert torch.isfinite(model.csdr_decoder.mask_head.weight.grad).all()
    assert model.csdr_decoder.route_head.weight.grad is None
    assert model.csdr_decoder.edge_head.weight.grad is None


@torch.no_grad()
def test_mask_does_not_depend_on_edge_logits() -> None:
    model = _model()
    image = torch.randn(1, 3, 64, 80)
    before = model(image)
    model.csdr_decoder.edge_head.bias.add_(10.0)
    after = model(image)
    assert torch.equal(before["mask"], after["mask"])
    assert not torch.equal(before["edge"], after["edge"])


def test_joint_loss_backpropagates_finite_gradients() -> None:
    model = _model().train()
    outputs = model(torch.randn(1, 3, 64, 80))
    sum(value.float().mean() for value in outputs.values()).backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@torch.no_grad()
def test_original_pro_is_unchanged() -> None:
    torch.manual_seed(177)
    before = DEMambaNetLitePro(use_mamba_ssm=False, drop_path_rate=0.0).eval()
    before_shapes = {name: tuple(value.shape) for name, value in before.state_dict().items()}
    before_count = sum(parameter.numel() for parameter in before.parameters())
    del before
    _model()
    torch.manual_seed(177)
    after = DEMambaNetLitePro(use_mamba_ssm=False, drop_path_rate=0.0).eval()
    assert before_shapes == {name: tuple(value.shape) for name, value in after.state_dict().items()}
    assert before_count == sum(parameter.numel() for parameter in after.parameters())
    assert set(after(torch.randn(1, 3, 64, 80))) == OUTPUT_KEYS


def test_config_preserves_pro_protocol() -> None:
    import yaml

    paths = (
        REPO_ROOT / "configs/experiments/lightweight_demambanet_lite_pro/DE_MAMBANET_LITE_PRO_LR150_DEEPCRACK_NATIVE384x544_EMA_BPCM.yaml",
        REPO_ROOT / "configs/experiments/lightweight_demambanet_lite_pro_csdr/DE_MAMBANET_LITE_PRO_CSDR_LR150_DEEPCRACK_NATIVE384x544_EMA_BPCM.yaml",
    )
    configs = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            configs.append(yaml.safe_load(handle))
    baseline, csdr = configs
    for section in ("data", "train", "loss", "evaluation"):
        assert csdr[section] == baseline[section], section
    assert csdr["model"]["name"] == "DEMambaNetLiteProCSDR"
    assert csdr["save"] != baseline["save"]


if __name__ == "__main__":
    tests = (
        test_registry_contract,
        test_native_forward_contracts,
        test_csdr_fusion_weights_and_detail_gate_ranges,
        test_two_mamba_cores_and_no_csdr_mamba_or_attention,
        test_mask_loss_cannot_update_route_or_edge_heads,
        test_mask_does_not_depend_on_edge_logits,
        test_joint_loss_backpropagates_finite_gradients,
        test_original_pro_is_unchanged,
        test_config_preserves_pro_protocol,
    )
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
