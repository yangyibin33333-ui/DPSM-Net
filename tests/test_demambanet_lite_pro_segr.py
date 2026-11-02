from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import DEMambaNetLitePro, DEMambaNetLiteProSEGR, build_model
from models.de_mambanet_lite_pro_segr import LightweightDirectionalRecoveryBlock
from models.mamba_blocks import SequenceMambaCore


OUTPUT_KEYS = {"mask", "edge", "reliability", "route_s1", "route_s3", "route_s4"}


def _model() -> DEMambaNetLiteProSEGR:
    return DEMambaNetLiteProSEGR(use_mamba_ssm=False, drop_path_rate=0.0).eval()


def test_registry_contract() -> None:
    assert isinstance(build_model("DEMambaNetLiteProSEGR", use_mamba_ssm=False), DEMambaNetLiteProSEGR)


@torch.no_grad()
def test_native_forward_contracts() -> None:
    for height, width in ((512, 512), (384, 544)):
        model = _model()
        outputs = model(torch.randn(1, 3, height, width))
        assert set(outputs) == OUTPUT_KEYS
        for name in ("mask", "edge", "reliability"):
            assert outputs[name].shape == (1, 1, height, width)
            assert torch.isfinite(outputs[name]).all()
        assert outputs["route_s1"].shape[-2:] == (height // 4, width // 4)
        assert outputs["route_s3"].shape[-2:] == (height // 16, width // 16)
        assert outputs["route_s4"].shape[-2:] == (height // 32, width // 32)
        del model, outputs


@torch.no_grad()
def test_gate_delta_initial_equivalence_and_signed_residual() -> None:
    model = _model()
    outputs, debug = model(torch.randn(1, 3, 64, 80), return_debug=True)
    assert set(outputs) == OUTPUT_KEYS
    gate = debug["recovery_gate"]
    assert torch.all(gate >= 0.0)
    assert torch.all(gate <= 1.0)
    assert torch.count_nonzero(model.segr.delta_head.weight).item() == 0
    assert torch.count_nonzero(model.segr.delta_head.bias).item() == 0
    assert torch.allclose(model.segr.residual_scale(), torch.tensor(1.0), atol=1e-6)
    assert torch.equal(debug["delta_logit"], torch.zeros_like(debug["delta_logit"]))

    expected = F.interpolate(
        debug["base_mask_logit_half"],
        size=(64, 80),
        mode="bilinear",
        align_corners=False,
    )
    assert torch.equal(outputs["mask"], expected)

    model.segr.delta_head.bias.fill_(1.0)
    _, positive = model(torch.randn(1, 3, 64, 80), return_debug=True)
    assert torch.all(positive["delta_logit"] > 0)
    model.segr.delta_head.bias.fill_(-1.0)
    _, negative = model(torch.randn(1, 3, 64, 80), return_debug=True)
    assert torch.all(negative["delta_logit"] < 0)


def test_recovery_blocks_are_lightweight_and_model_has_two_cores() -> None:
    model = _model()
    cores = [module for module in model.modules() if isinstance(module, SequenceMambaCore)]
    assert len(cores) == 2
    assert not any(isinstance(module, SequenceMambaCore) for module in model.segr.modules())
    assert len(model.segr.recovery_blocks) == 2
    for block in model.segr.recovery_blocks:
        assert isinstance(block, LightweightDirectionalRecoveryBlock)
        for convolution in (block.horizontal, block.vertical, block.local):
            assert convolution.in_channels == 48
            assert convolution.groups == 48
        assert block.horizontal.kernel_size == (1, 9)
        assert block.vertical.kernel_size == (9, 1)
        assert block.local.kernel_size == (3, 3)
    forbidden = ("Attention", "Transformer", "Mamba")
    for module in model.segr.modules():
        assert not any(fragment in type(module).__name__ for fragment in forbidden)


def test_mask_only_gradient_isolation_and_base_path_gradient() -> None:
    model = _model().train()
    outputs = model(torch.randn(1, 3, 64, 80))
    outputs["mask"].mean().backward()
    assert model.decoder.mask_head.weight.grad is not None
    assert torch.isfinite(model.decoder.mask_head.weight.grad).all()
    assert model.decoder.edge_head.weight.grad is None
    assert model.decoder.route_head.weight.grad is None


def test_joint_outputs_backpropagate_finite_gradients() -> None:
    model = _model().train()
    outputs = model(torch.randn(1, 3, 64, 80))
    loss = sum(value.float().mean() for value in outputs.values())
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@torch.no_grad()
def test_original_pro_remains_unchanged() -> None:
    torch.manual_seed(321)
    before = DEMambaNetLitePro(use_mamba_ssm=False, drop_path_rate=0.0).eval()
    shapes_before = {name: tuple(value.shape) for name, value in before.state_dict().items()}
    count_before = sum(parameter.numel() for parameter in before.parameters())
    del before
    _model()
    torch.manual_seed(321)
    after = DEMambaNetLitePro(use_mamba_ssm=False, drop_path_rate=0.0).eval()
    shapes_after = {name: tuple(value.shape) for name, value in after.state_dict().items()}
    count_after = sum(parameter.numel() for parameter in after.parameters())
    assert shapes_before == shapes_after
    assert count_before == count_after
    assert set(after(torch.randn(1, 3, 64, 80))) == OUTPUT_KEYS


def test_config_preserves_pro_protocol() -> None:
    import yaml

    paths = (
        REPO_ROOT / "configs/experiments/lightweight_demambanet_lite_pro/DE_MAMBANET_LITE_PRO_LR150_DEEPCRACK_NATIVE384x544_EMA_BPCM.yaml",
        REPO_ROOT / "configs/experiments/lightweight_demambanet_lite_pro_segr/DE_MAMBANET_LITE_PRO_SEGR_LR150_DEEPCRACK_NATIVE384x544_EMA_BPCM.yaml",
    )
    configs = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            configs.append(yaml.safe_load(handle))
    baseline, segr = configs
    for section in ("data", "train", "loss", "evaluation"):
        assert segr[section] == baseline[section], section
    expected_model = dict(baseline["model"])
    expected_model.update(
        name="DEMambaNetLiteProSEGR",
        recovery_channels=48,
        recovery_depth=2,
        recovery_kernel_size=9,
        recovery_drop_path=0.05,
        residual_scale_max=3.0,
        detach_recovery_gate_inputs=True,
    )
    assert segr["model"] == expected_model
    assert segr["save"] != baseline["save"]


if __name__ == "__main__":
    tests = (
        test_registry_contract,
        test_native_forward_contracts,
        test_gate_delta_initial_equivalence_and_signed_residual,
        test_recovery_blocks_are_lightweight_and_model_has_two_cores,
        test_mask_only_gradient_isolation_and_base_path_gradient,
        test_joint_outputs_backpropagate_finite_gradients,
        test_original_pro_remains_unchanged,
        test_config_preserves_pro_protocol,
    )
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
