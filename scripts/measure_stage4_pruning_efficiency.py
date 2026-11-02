"""Run the historical paper measurement engine for strict T0--T3.

No new timing implementation: each variant is a fresh child process invoking
measure_model_complexity.py with the original L0--L4 command-line protocol.
No dataset, training checkpoint, training, or segmentation evaluation is used.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.measure_model_complexity import benchmark_fps, measurement_environment

ENGINE = ROOT / 'scripts/measure_model_complexity.py'
HISTORY = ROOT / 'runs/paper/dpsm_lightweight_ablation/crack500/efficiency_mask_only_512/20260901-073039'
NEW = 'configs/experiments/stage4_compact_ablation/crack500/'
OLD = 'configs/experiments/paper_dependency_aware_ablation/crack500/'
VARIANTS = {
    'T0': (NEW + 'T0_LocalOnly_StrictV2.yaml', 0, 'Local'),
    'T1': (OLD + 'M1_FrameworkOn_DPSMOff_StructSupOff.yaml', 0, 'Local+Transformer'),
    'T2': (NEW + 'T2_LocalDPSM80_StrictV2.yaml', 1, 'Local+DPSM'),
    'T3': (OLD + 'M3_FrameworkOn_DPSMOn_StructSupOff.yaml', 1, 'Local+Transformer+DPSM'),
}
# Segmentation values supplied by the user; this program never measures them.
ACCURACY = {'T0': (78.77, 54.76, 82.00), 'T1': (79.39, 55.25, 83.18),
            'T2': (79.25, 55.27, 82.54), 'T3': (79.48, 55.54, 83.31)}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def one_record(path):
    rows = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError(f'Expected one unambiguous historical-engine record: {path}')
    return rows[0]


def history_comparison(current, historical):
    keys = ('pytorch', 'cuda', 'mamba_ssm')
    diffs = {k: {'current': current.get(k), 'historical': historical.get(k)}
             for k in keys if current.get(k) != historical.get(k)}
    normalize = lambda x: re.sub('[^a-z0-9]', '', str(x).lower())
    if normalize(current.get('gpu')) != normalize(historical.get('gpu')):
        diffs['gpu'] = {'current': current.get('gpu'), 'historical': historical.get('gpu')}
    unknown = [k for k in ('tf32_matmul', 'tf32_cudnn', 'cudnn_benchmark', 'cudnn_deterministic',
                           'cudnn_version', 'driver_version', 'causal_conv1d') if k not in historical]
    return {'recorded_differences': diffs, 'unrecorded_historical_settings': unknown,
            'fully_verified_historical_equivalence': not diffs and not unknown,
            'recommendation': 'measure all four in one environment; keep historical values separately'}


def command(tag, out, device):
    config, count, _ = VARIANTS[tag]
    return [sys.executable, str(ENGINE), '--config', str(ROOT / config), '--experiment', tag,
            '--img_size', '512', '512', '--batch_size', '1', '--device', device,
            '--amp', 'fp16', '--warmup', '100', '--iterations', '300', '--benchmark_repeats', '3',
            '--inference_path', 'mask-only', '--require_mamba_ssm', '--expected-mamba-cores', str(count),
            '--report_json', str(out / f'{tag}.json'), '--report_csv', str(out / f'{tag}.csv'),
            '--mamba-boundary-report', str(out / f'{tag}.mamba_boundary.json')]


def provenance_snapshot(out, tags):
    files = [ENGINE, Path(__file__).resolve(), ROOT / 'scripts/measure_dpsm_lightweight_efficiency.py']
    files += list((ROOT / 'models').glob('*.py'))
    files += [ROOT / VARIANTS[t][0] for t in tags]
    hashes = {}
    for path in files:
        relative = path.relative_to(ROOT)
        target = out / 'reproduction' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative).replace('\\', '/')] = sha(path)
    return hashes


def comparison_rows(records, historical):
    result = []
    for tag, (_, _, structure) in VARIANTS.items():
        record = records.get(tag)
        source = 'current measured'
        if record is None:
            record = historical.get(tag)
            source = 'historical reference only' if record else 'not measured'
        record = record or {}
        value = lambda key, scale=1: None if record.get(key) is None else record[key] / scale
        result.append({'Model': tag, 'Structure': structure, 'Source': source,
            'Params(M)': value('params', 1e6), 'Estimated FLOPs(G)': value('corrected_flops_estimated', 1e9),
            'FPS': value('fps'), 'Mean Latency(ms)': value('latency_mean_ms'),
            'Median Latency(ms)': value('latency_p50_round_ms'),
            'Latency round std(ms)': value('latency_round_std_ms'),
            'Memory(MiB)': value('peak_allocated_bytes', 1024**2),
            'F1(%)': ACCURACY[tag][0], 'Boundary F1(%)': ACCURACY[tag][1], 'clDice(%)': ACCURACY[tag][2]})
    return result


def write_comparison(out, records, historical, meta):
    rows = comparison_rows(records, historical)
    with (out / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    cell = lambda v: 'not measured' if v is None else f'{v:.3f}' if isinstance(v, float) else str(v)
    columns = ['Model', 'Structure', 'Source', 'Params(M)', 'Estimated FLOPs(G)', 'FPS',
               'Median Latency(ms)', 'Memory(MiB)', 'F1(%)', 'Boundary F1(%)', 'clDice(%)']
    lines = ['# Stage4 pruning efficiency', '',
        f"Run status: {meta['status']}. No training or segmentation evaluation was performed.", '',
        'Historical timing uses median FPS and median of three 300-forward mean latencies. '
        'Mean/std are over those three round means, not individual-frame percentiles. '
        'Memory is maximum post-warmup allocated bytes / 2^20, including resident model/input. '
        'Reserved memory is NOT substituted for allocated memory.', '',
        '| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for row in rows:
        lines.append('| ' + ' | '.join(cell(row[k]) for k in columns) + ' |')
    lines += ['', 'Accuracy values are user-supplied three-seed summaries. '
              'Historical-reference timing rows are not evidence of equivalence with current measurements.', '',
              '## Interpretation', '']
    if set(records) == set(VARIANTS) and meta['status'] == 'complete':
        for left, right in (('T2', 'T1'), ('T2', 'T3')):
            a, b = records[left], records[right]
            ratios = {key: 100 * (a[key] / b[key] - 1) for key in
                      ('params', 'corrected_flops_estimated', 'fps', 'latency_p50_round_ms')}
            lines.append(f"{left} vs {right}: Params {ratios['params']:+.2f}%, estimated FLOPs "
                f"{ratios['corrected_flops_estimated']:+.2f}%, FPS {ratios['fps']:+.2f}%, "
                f"median round latency {ratios['latency_p50_round_ms']:+.2f}%.")
        pareto = [tag for tag in records if not any(
            ACCURACY[other][0] >= ACCURACY[tag][0]
            and records[other]['latency_p50_round_ms'] <= records[tag]['latency_p50_round_ms']
            and (ACCURACY[other][0] > ACCURACY[tag][0]
                 or records[other]['latency_p50_round_ms'] < records[tag]['latency_p50_round_ms'])
            for other in records if other != tag)]
        lines += ['', 'Observed F1–median-latency non-dominated models: ' + ', '.join(pareto) + '.',
            'These are point-estimate comparisons, not statistically established dominance.']
    else:
        lines += ['T0/T2 formal speed, full cost reductions and the four-model Pareto set cannot yet '
                  'be established from one complete uniform-environment suite. Pending values remain empty in CSV.']
    lines += ['', 'T2 vs T1 accuracy: F1 −0.14 percentage points, Boundary F1 +0.02, clDice −0.64. '
        'T2 vs T3: F1 −0.23, Boundary F1 −0.27, clDice −0.77. '
        'T0 vs T3: F1 −0.71, Boundary F1 −0.78, clDice −1.31.', '',
        'There is no unique best lightweight trade-off without a latency/memory budget and an acceptable '
        'accuracy loss. Deleting Transformer is an efficiency option to test, not currently evidence '
        'of equal segmentation quality. An EAAI architecture decision also needs per-seed dispersion '
        '(preferably paired seeds), repeated device timing, and the measured Boundary F1/clDice trade-off. '
        'Keep T3 as the accuracy reference; do not infer significance from rounded means.', '',
        'FLOPs = fvcore-counted selected graph + verified missing fused-Mamba components; one FMA = one FLOP. '
        'Per-variant JSON retains omitted elementwise/reduction/rearrangement ops. '
        'Timing uses eval + inference_mode + FP16 autocast, while Params are FP32 and historical FLOP tracing is FP32.', '',
        'See metadata.json for commands, hashes, environment/settings and history-equivalence limitations.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument('--output-root', type=Path, default=ROOT / 'runs/paper/stage4_pruning_efficiency/crack500')
    parser.add_argument('--historical-dir', type=Path, default=HISTORY)
    args = parser.parse_args()
    if len(args.variants) != len(set(args.variants)):
        parser.error('variants must be unique')
    out = args.output_root.resolve() / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    env = measurement_environment(device)
    smi = shutil.which('nvidia-smi')
    if not smi and sys.platform == 'win32':
        candidate = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'System32/nvidia-smi.exe'
        smi = str(candidate) if candidate.is_file() else None
    hardware = None
    if smi:
        query = subprocess.run([smi, '--query-gpu=name,driver_version,memory.total,pstate', '--format=csv'],
                               capture_output=True, text=True, check=False)
        hardware = query.stdout.strip() or query.stderr.strip()
    hist_meta = json.loads((args.historical_dir / 'metadata.json').read_text(encoding='utf-8'))
    historical = {'T1': one_record(args.historical_dir / 'L0.json'),
                  'T3': one_record(args.historical_dir / 'L3.json')}
    for filename in ('metadata.json', 'L0.json', 'L3.json', 'summary.json'):
        source = args.historical_dir / filename
        if source.exists():
            target = out / 'historical_reference' / filename
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
    commands = {t: command(t, out, args.device) for t in args.variants}
    meta = {'status': 'preflight', 'created': datetime.now().astimezone().isoformat(), 'environment': env,
        'nvidia_smi': hardware, 'historical_directory': str(args.historical_dir.resolve()),
        'history_comparison': history_comparison(env, hist_meta),
        'protocol': {'input': [1, 3, 512, 512], 'precision': 'fp16 autocast', 'eval': True,
                     'inference_mode': True, 'inference_path': 'mask-only', 'warmup': 100,
                     'timed_iterations': 300, 'repeats': 3, 'weights': 'random initialization',
                     'checkpoint_loaded': False, 'timing': 'unchanged historical benchmark_fps CUDA events',
                     'latency_scope': 'three round means; population std of round means'},
        'commands': commands, 'measurement_engine_sha256': sha(ENGINE),
        'benchmark_function_sha256': hashlib.sha256(inspect.getsource(benchmark_fps).encode()).hexdigest(),
        'source_and_config_hashes': provenance_snapshot(out, args.variants),
        'historical_file_hashes': {f: sha(args.historical_dir / f) for f in ('metadata.json', 'L0.json', 'L3.json')},
        'accuracy_provenance': 'user-supplied segmentation summaries; not recomputed', 'variants': {}}
    records = {}
    reason = None
    if device.type != 'cuda' or not torch.cuda.is_available():
        reason = 'Formal run requires CUDA PyTorch and the historical RTX 4090 D; current CUDA unavailable.'
    elif '4090d' not in re.sub('[^a-z0-9]', '', str(env['gpu']).lower()):
        reason = f"GPU mismatch: current={env['gpu']}; required=RTX 4090 D. No timing executed."
    if reason is None:
        try:
            from models.mamba_blocks import Mamba
            if Mamba is None:
                reason = 'Formal Mamba backend cannot be imported. Fallback timing is prohibited.'
        except ImportError as error:
            reason = repr(error)
    with (out / 'benchmark.log').open('w', encoding='utf-8') as log:
        if reason:
            meta.update(status='blocked_environment', reason=reason)
            log.write(reason + '\n'); print(reason, flush=True)
        else:
            meta['status'] = 'running'
            dump(out / 'metadata.json', meta)
            for tag in args.variants:
                print(f'Measuring {tag} using historical engine...', flush=True)
                log.write(f'COMMAND {tag}: {json.dumps(commands[tag])}\n'); log.flush()
                # Same child-process isolation as the historical orchestrator.
                with (out / f'{tag}.log').open('w', encoding='utf-8') as child_log:
                    completed = subprocess.run(commands[tag], cwd=ROOT, stdout=child_log, stderr=subprocess.STDOUT)
                log.write((out / f'{tag}.log').read_text(encoding='utf-8', errors='replace') + '\n')
                log.flush()
                meta['variants'][tag] = {'returncode': completed.returncode}
                if completed.returncode != 0:
                    meta.update(status='failed', reason=f'{tag} failed; inspect {tag}.log')
                    break
                try:
                    row = one_record(out / f'{tag}.json')
                    assert row['flops_status'] == 'estimated_under_declared_convention'
                    assert row['inference_path'] == 'mask-only' and row['mask_only_allclose']
                except (OSError, ValueError, KeyError, AssertionError) as error:
                    meta.update(status='failed', reason=f'{tag} report invalid: {error!r}')
                    log.write(meta['reason'] + '\n')
                    break
                records[tag] = row
                dump(out / 'metadata.json', meta)
            else:
                meta['status'] = 'complete'
    for tag in args.variants:
        if not (out / f'{tag}.json').exists():
            dump(out / f'{tag}.json', {'status': 'not_measured', 'reason': meta.get('reason', 'previous variant failed'),
                                      'command': commands[tag]})
    # Keep actual child settings; parent metadata is only preflight evidence.
    meta['measured_child_environments'] = {t: r['runtime_environment'] for t, r in records.items()}
    meta['all_variants_measured_in_one_suite'] = set(records) == set(VARIANTS)
    dump(out / 'metadata.json', meta)
    write_comparison(out, records, historical, meta)
    print(f'Results: {out}', flush=True)
    return 0 if meta['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
