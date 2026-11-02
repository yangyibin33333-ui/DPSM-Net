"""Regression checks for the independent DPER-HTPR early-rebase ablation.

Run with either ``python tests/test_haar_topology_preserving_rebase.py`` or
``python -m pytest tests/test_haar_topology_preserving_rebase.py -q``.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import build_model
from models.blocks import HaarTopologyPreservingRebase, SubpixelPhaseConsistencyRefiner, TopoRouteBlock, TransformerGlobalBlock
from models.decoder import LowResolutionSubpixelSeparableConcatUpBlock


DPER_CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "lightweight_c40_s4mamba_lsscf_dper"
    / "C3_C40_S4MAMBA_LSSCF_DPER_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
HTPR_CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "lightweight_c40_s4mamba_lsscf_dper_htpr"
    / "C3_C40_S4MAMBA_LSSCF_DPER_HTPR_LR150_PHOTO_NATIVE384x544_EMA_BPCM.yaml"
)
ALL_DIRECTIONS = ("horizontal", "vertical", "diagonal", "anti_diagonal")
RUNTIME_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build(path: Path, **overrides: object) -> torch.nn.Module:
    model_cfg = copy.deepcopy(_load_config(path)["model"])
    model_cfg.update(overrides)
    model_name = model_cfg.pop("name")
    # Selective scan is CUDA-only; direct CPU regression runs exercise the
    # repository fallback while retaining four-path Stage4 topology.
    if RUNTIME_DEVICE.type != "cuda":
        model_cfg["use_mamba_ssm"] = False
    return build_model(model_name, **model_cfg).to(RUNTIME_DEVICE)


def test_htpr_config_is_isolated_from_original_dper() -> None:
    dper = _load_config(DPER_CONFIG)
    htpr = _load_config(HTPR_CONFIG)
    assert {key: value for key, value in htpr.items() if key not in {"model", "save"}} == {
        key: value for key, value in dper.items() if key not in {"model", "save"}
    }
    expected_model = copy.deepcopy(dper["model"])
    expected_model["use_haar_topology_preserving_rebase"] = True
    expected_model["use_subpixel_phase_consistency_refiner"] = False
    assert htpr["model"] == expected_model
    assert htpr["save"]["save_dir"] == (
        "./runs/pilot/lightweight_c40_s4mamba_lsscf_dper_htpr/"
        "C3_C40_S4MAMBA_LSSCF_DPER_HTPR_LR150_PHOTO_NATIVE384x544_EMA_BPCM"
    )


def test_fixed_haar_decomposition_is_invertible() -> None:
    torch.manual_seed(301)
    rebase = HaarTopologyPreservingRebase(3, 40, 16)
    image = torch.randn(2, 3, 64, 80)
    ll, lh, hl, hh = rebase.haar_rebase(image)
    assert ll.shape == lh.shape == hl.shape == hh.shape == (2, 3, 32, 40)
    restored = rebase.inverse_haar(ll, lh, hl, hh)
    assert restored.shape == image.shape
    assert (restored - image).abs().max().item() < 1e-6


def test_htpr_dependency_guards_and_non_htpr_compatibility() -> None:
    for overrides, expected_text in (
        ({"use_detail_preserving_early_rebase": False, "use_haar_topology_preserving_rebase": True}, "requires"),
        ({"use_haar_topology_preserving_rebase": True, "use_subpixel_phase_consistency_refiner": True}, "cannot"),
    ):
        try:
            _build(DPER_CONFIG, **overrides)
        except ValueError as error:
            assert expected_text in str(error)
        else:
            raise AssertionError("HTPR dependency validation did not reject an invalid configuration.")

    torch.manual_seed(302)
    dper = _build(DPER_CONFIG).eval()
    torch.manual_seed(302)
    false_flag = _build(DPER_CONFIG, use_haar_topology_preserving_rebase=False).eval()
    assert set(dper.state_dict()) == set(false_flag.state_dict())
    for key, value in dper.state_dict().items():
        assert torch.equal(value, false_flag.state_dict()[key]), key
    image = torch.randn(1, 3, 64, 80, device=RUNTIME_DEVICE)
    with torch.inference_mode():
        expected = dper(image)
        actual = false_flag(image)
    for key in expected:
        assert torch.equal(expected[key], actual[key]), key


def test_htpr_model_forward_backward_and_preserved_topology() -> None:
    torch.manual_seed(303)
    dper = _build(DPER_CONFIG).eval()
    torch.manual_seed(303)
    model = _build(HTPR_CONFIG).eval()
    assert model.use_detail_preserving_early_rebase
    assert model.use_haar_topology_preserving_rebase
    assert not model.use_subpixel_phase_consistency_refiner
    assert isinstance(model.stem, HaarTopologyPreservingRebase)
    assert model.stem.alpha.shape == torch.Size([])
    assert torch.isclose(model.stem.alpha.detach(), torch.tensor(0.1, device=RUNTIME_DEVICE))
    assert model.f1_reconstruction is not None and model.f1_reconstruction.phase_refiner is None
    assert not any(isinstance(module, SubpixelPhaseConsistencyRefiner) for module in model.modules())
    assert model.stem.semantic_projection.block[0].in_channels == 12
    assert model.stem.hf_projection.in_channels == 9
    assert model.stem.hf_depthwise.groups == 40
    assert model.stem.low_frequency_gate.in_channels == 3

    dper_state = dper.state_dict()
    htpr_state = model.state_dict()
    shared_prefixes = ("stem.full_detail_projection.", "stage1.", "down1.", "stage2.", "down2.", "stage3.", "down3.", "stage4.", "decoder.", "f1_reconstruction.")
    for key, value in dper_state.items():
        if key.startswith(shared_prefixes):
            assert key in htpr_state, key
            assert torch.equal(value, htpr_state[key]), key

    assert isinstance(model.stage4, TopoRouteBlock)
    assert isinstance(model.stage4.global_branch, TransformerGlobalBlock)
    assert model.stage4.channels == 320
    assert model.stage4.route_mamba.spatial_reduction == 1
    assert model.stage4.route_mamba.mamba_channels == 320
    assert model.stage4.route_mamba.active_mamba_directions == ALL_DIRECTIONS
    assert model.stage4.route_mamba.active_mamba_path_count == 4
    stage4_backends = model.mamba_backend_info()["stage4"]
    assert len(stage4_backends) == 4
    # CUDA availability alone does not guarantee that mamba_ssm is installed
    # or usable on a particular host.  This regression test validates the
    # four-path topology and reports the model's actual backend faithfully;
    # train.py still rejects fallback before any formal experiment can start.
    assert len(set(stage4_backends)) == 1
    assert stage4_backends[0] in {"mamba_ssm", "fallback"}
    assert isinstance(model.decoder.up3, LowResolutionSubpixelSeparableConcatUpBlock)
    assert isinstance(model.decoder.up2, LowResolutionSubpixelSeparableConcatUpBlock)
    assert isinstance(model.decoder.up1, LowResolutionSubpixelSeparableConcatUpBlock)

    calls = {"semantic": 0, "hf_projection": 0, "hf_depthwise": 0, "gate": 0}
    handles = [
        model.stem.semantic_projection.register_forward_hook(
            lambda _module, _inputs, _output: calls.__setitem__("semantic", calls["semantic"] + 1)
        ),
        model.stem.hf_projection.register_forward_hook(
            lambda _module, _inputs, _output: calls.__setitem__("hf_projection", calls["hf_projection"] + 1)
        ),
        model.stem.hf_depthwise.register_forward_hook(
            lambda _module, _inputs, _output: calls.__setitem__("hf_depthwise", calls["hf_depthwise"] + 1)
        ),
        model.stem.low_frequency_gate.register_forward_hook(
            lambda _module, _inputs, _output: calls.__setitem__("gate", calls["gate"] + 1)
        ),
    ]
    image = torch.randn(1, 3, 64, 80, device=RUNTIME_DEVICE)
    try:
        with torch.inference_mode():
            features = model.extract_encoder_features(image)
            full_outputs = model(image)
            mask_only = model.forward_mask_only(image)
    finally:
        for handle in handles:
            handle.remove()
    assert calls == {"semantic": 3, "hf_projection": 3, "hf_depthwise": 3, "gate": 3}
    assert features["f1"].shape == (1, 40, 64, 80)
    assert features["f2"].shape == (1, 80, 32, 40)
    assert features["f3"].shape == (1, 160, 16, 20)
    assert features["f4"].shape == (1, 320, 8, 10)
    assert full_outputs["mask"].shape == (1, 1, 64, 80)
    assert torch.equal(full_outputs["mask"], mask_only["mask"])

    model.train()
    outputs = model(image)
    (outputs["mask"].mean() + outputs["edge"].mean() + outputs["reliability"].mean()).backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


def test_htpr_native_and_table2_forward_shapes() -> None:
    torch.manual_seed(304)
    model = _build(HTPR_CONFIG).eval()
    with torch.inference_mode():
        for height, width in ((384, 544), (512, 512)):
            output = model.forward_mask_only(torch.randn(1, 3, height, width, device=RUNTIME_DEVICE))
            assert output["mask"].shape == (1, 1, height, width)


def run_all() -> None:
    tests = (
        test_htpr_config_is_isolated_from_original_dper,
        test_fixed_haar_decomposition_is_invertible,
        test_htpr_dependency_guards_and_non_htpr_compatibility,
        test_htpr_model_forward_backward_and_preserved_topology,
        test_htpr_native_and_table2_forward_shapes,
    )
    for check in tests:
        check()
        print(f"PASS: {check.__name__}")


if __name__ == "__main__":
    run_all()
