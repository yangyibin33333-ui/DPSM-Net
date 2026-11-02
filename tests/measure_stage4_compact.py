"""GPU-only synthetic efficiency check for the new T0/T2 classes.

Reuses repository measurement primitives, avoiding the old CLI's frozen
Transformer class guard. Never reads datasets or training checkpoints.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import yaml
from models import build_model
from models.blocks import Stage4LocalOnly, Stage4LocalDPSM80
from models.mamba_blocks import SequenceMambaCore
from scripts.measure_model_complexity import (
    _MaskOnlyModel, benchmark_fps, capture_stage4_execution,
    diagnose_mamba_fused_boundary, estimate_mamba_inner_flops,
    measure_flops, validate_mask_only_equivalence,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--img-size', nargs=2, type=int, default=[512, 512])
    p.add_argument('--output-dir', type=Path)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Formal efficiency measurement requires CUDA.')
    out = a.output_dir or ROOT / 'tests/stage4_reports' / ('efficiency-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
    out.mkdir(parents=True, exist_ok=False)
    options = yaml.safe_load(a.config.read_text(encoding='utf-8'))['model']
    model = build_model(options.pop('name'), **options).cuda().eval()
    assert isinstance(model.stage4, (Stage4LocalOnly, Stage4LocalDPSM80))
    cores = [m for m in model.modules() if isinstance(m, SequenceMambaCore)]
    assert len(cores) == int(isinstance(model.stage4, Stage4LocalDPSM80))
    assert all(c.backend == 'mamba_ssm' for c in cores), 'Fallback is not admissible'
    image = torch.randn(1, 3, *a.img_size, device='cuda')
    device = torch.device('cuda')
    with torch.inference_mode():
        outputs, captured = capture_stage4_execution(model, image)
        equivalence = validate_mask_only_equivalence(model, image, device=device, amp='fp16')
        counted, unsupported = measure_flops(model, image, inference_path='mask-only')
        boundary = diagnose_mamba_fused_boundary(cores[0], captured[0][0]) if cores else None
        missing, status, components = estimate_mamba_inner_flops(model, captured, boundary)
    if cores and missing is None:
        raise RuntimeError(f'Formal Mamba FLOPs boundary incomplete: {status}')
    rows = []
    for _ in range(3):
        fps, latency, memory = benchmark_fps(_MaskOnlyModel(model), image, device=device,
                                            warmup=100, iterations=300, amp='fp16')
        rows.append({'fps': fps, 'latency_ms': latency, 'peak_memory_bytes': memory})
    report = {'config': str(a.config), 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
        'input': list(image.shape), 'precision': 'fp16', 'warmup': 100, 'iterations': 300,
        'repeats': 3, 'path': 'mask-only', 'params': sum(p.numel() for p in model.parameters()),
        'backends': model.mamba_backend_info(), 'mask_only_equivalence': equivalence,
        'fvcore_counted_flops': counted, 'unsupported_ops': dict(unsupported),
        'mamba_missing_estimated': missing, 'estimated_flops': counted + float(missing or 0),
        'mamba_estimation_status': status, 'components': components,
        'convention': 'one multiply-add = one FLOP; unsupported non-Mamba ops remain omitted',
        'core_inputs': captured, 'measurements': rows}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if boundary:
        (out / 'boundary.json').write_text(json.dumps(boundary, indent=2), encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
