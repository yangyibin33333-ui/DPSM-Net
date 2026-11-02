from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import DEMambaNetLitePro, DEMambaNetLiteProFBPV, build_model
from models.de_mambanet_lite_pro_fbpv import PerImageDualPrototype
from models.mamba_blocks import SequenceMambaCore

KEYS = {"mask", "edge", "reliability", "route_s1", "route_s3", "route_s4"}


def _model() -> DEMambaNetLiteProFBPV:
    return DEMambaNetLiteProFBPV(use_mamba_ssm=False, drop_path_rate=0.0).eval()


def test_registry_and_native_shapes() -> None:
    assert isinstance(build_model("DEMambaNetLiteProFBPV", use_mamba_ssm=False), DEMambaNetLiteProFBPV)
    for size in ((512, 512), (384, 544)):
        with torch.no_grad(): outputs = _model()(torch.randn(1, 3, *size))
        assert set(outputs) == KEYS
        assert all(outputs[name].shape == (1, 1, *size) for name in ("mask", "edge", "reliability"))


@torch.no_grad()
def test_prototypes_are_per_image_fp32_finite_and_stable() -> None:
    module = PerImageDualPrototype(64).eval()
    feature = torch.randn(1, 64, 7, 9)
    logit = torch.randn(1, 1, 7, 9)
    fg1, bg1 = module(feature, logit)
    fg2, bg2 = module(torch.cat([feature, torch.randn_like(feature)], 0), torch.cat([logit, torch.randn_like(logit)], 0))
    assert fg1.dtype == torch.float32 and bg1.dtype == torch.float32
    assert torch.allclose(fg1, fg2[:1]) and torch.allclose(bg1, bg2[:1])
    assert torch.allclose(fg1.norm(dim=1), torch.ones_like(fg1[:, :1, :1, :1]))
    zero_fg, zero_bg = module(feature, torch.full_like(logit, -100.0))
    one_fg, one_bg = module(feature, torch.full_like(logit, 100.0))
    assert all(torch.isfinite(value).all() for value in (zero_fg, zero_bg, one_fg, one_bg))


@torch.no_grad()
def test_verification_ranges_and_two_cores() -> None:
    model = _model(); outputs, debug = model(torch.randn(1, 3, 64, 80), return_debug=True)
    assert set(outputs) == KEYS
    assert len([m for m in model.modules() if isinstance(m, SequenceMambaCore)]) == 2
    assert torch.all((debug["verification_probability"] >= 0) & (debug["verification_probability"] <= 1))
    assert torch.all((debug["feature_scale"] >= .75) & (debug["feature_scale"] <= 1.25))
    assert .10 <= model.decoder.prototype_verifier.temperature().item() <= .50
    assert -1.5 <= model.decoder.prototype_mask_head.correction_scale().item() <= 1.5
    assert torch.count_nonzero(model.decoder.prototype_mask_head.detail_delta_head.weight) == 0


def test_mask_gradient_paths_and_edge_independence() -> None:
    model = _model().train(); image = torch.randn(1, 3, 64, 80)
    # At the exact zero-start state, the delta head receives the first mask
    # update while its zero weight blocks gradients to its inputs. A tiny
    # post-start value verifies the intended verifier gradient path.
    with torch.no_grad(): model.decoder.prototype_mask_head.detail_delta_head.weight.fill_(0.01)
    outputs = model(image); outputs["mask"].mean().backward()
    assert model.decoder.coarse_mask_head.weight.grad is not None
    assert model.decoder.prototype_mask_head.detail_delta_head.weight.grad is not None
    assert model.decoder.prototype_verifier.temperature_parameter.grad is not None
    assert model.decoder.edge_head.weight.grad is None
    with torch.no_grad():
        model.eval(); before = model(image); model.decoder.edge_head.bias.add_(10); after = model(image)
    assert torch.equal(before["mask"], after["mask"])


def test_joint_backward_and_original_pro() -> None:
    model = _model().train(); outputs = model(torch.randn(1, 3, 64, 80)); sum(v.float().mean() for v in outputs.values()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    before = DEMambaNetLitePro(use_mamba_ssm=False, drop_path_rate=0.0); count = sum(p.numel() for p in before.parameters()); _model(); after = DEMambaNetLitePro(use_mamba_ssm=False, drop_path_rate=0.0)
    assert count == sum(p.numel() for p in after.parameters())


if __name__ == "__main__":
    for test in (test_registry_and_native_shapes, test_prototypes_are_per_image_fp32_finite_and_stable, test_verification_ranges_and_two_cores, test_mask_gradient_paths_and_edge_independence, test_joint_backward_and_original_pro):
        test(); print(f"PASS: {test.__name__}")
