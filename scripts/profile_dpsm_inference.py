"""Independent DPSM diagnostics; never changes production source or trains.

Full-model timing calls the historical benchmark_fps unchanged. Fine-grained
traces annotate the original Python AST (only adds record_function contexts).
--audit-only is a CPU/fallback correctness check, never performance evidence.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from datetime import datetime
import hashlib
import inspect
import json
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import textwrap
import types
import csv

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.measure_model_complexity import (load_model, benchmark_fps,
    measurement_environment, autocast_context, _MaskOnlyModel,
    validate_mask_only_equivalence)
from scripts.measure_stage4_pruning_efficiency import VARIANTS
from models.mamba_blocks import DirectionPartitionedSharedMamba, SequenceMambaCore


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def csv_write(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summary(values):
    return dict(mean_ms=statistics.mean(values), median_ms=statistics.median(values),
                std_ms=statistics.pstdev(values), samples_ms=values)


@contextmanager
def annotations(model):
    """Instrument original operations without reimplementing the forward math."""
    handles, replacements = [], []
    active = {}
    stage = model.stage4
    targets = {'Stage4': stage}
    for name in ('local', 'global_branch', 'route_mamba', 'fuse', 'refine'):
        obj = getattr(stage, name, None)
        if isinstance(obj, torch.nn.Module):
            targets['Stage4/' + name] = obj
    route = getattr(stage, 'route_mamba', None)
    if isinstance(route, DirectionPartitionedSharedMamba):
        for name in ('route_head', 'pre_mix', 'shared_mamba', 'output_mix'):
            targets['DPSM/' + name] = getattr(route, name)
        for name in ('norm', 'core', 'out_norm'):
            targets['Mamba/' + name] = getattr(route.shared_mamba, name)
        for direction, scan in route.scans.items():
            targets['scan/' + direction] = scan
            original = scan.inverse
            def inverse(*args, _original=original, _name=direction, **kwargs):
                with torch.profiler.record_function('inverse/' + _name):
                    return _original(*args, **kwargs)
            replacements.append((scan, 'inverse', 'inverse' in scan.__dict__, scan.__dict__.get('inverse')))
            scan.inverse = inverse
        # Annotate each original top-level statement. Tensor expressions and
        # control flow are retained verbatim; there is no copied forward body.
        tree = ast.parse(textwrap.dedent(inspect.getsource(type(route).forward)))
        function = tree.body[0]
        body = []
        for index, statement in enumerate(function.body):
            if isinstance(statement, ast.Assign):
                label = ast.unparse(statement.targets[0])
            elif isinstance(statement, ast.Return):
                label = 'residual_and_output'
            elif isinstance(statement, ast.If) and 'route_mode' in ast.unparse(statement.test):
                label = 'route_weights'
            else:
                label = 'statement_' + str(index)
            call = ast.Call(func=ast.Name(id='_dpsm_record_function', ctx=ast.Load()),
                            args=[ast.Constant('DPSM_op/' + label)], keywords=[])
            body.append(ast.With(items=[ast.withitem(context_expr=call)], body=[statement]))
        function.body = body
        scope = dict(type(route).forward.__globals__)
        scope['_dpsm_record_function'] = torch.profiler.record_function
        exec(compile(ast.fix_missing_locations(tree), '<dpsm-diagnostic-annotations>', 'exec'), scope)
        replacements.append((route, 'forward', 'forward' in route.__dict__, route.__dict__.get('forward')))
        route.forward = types.MethodType(scope['forward'], route)
    def before(name):
        def hook(module, inputs):
            ctx = torch.profiler.record_function(name)
            ctx.__enter__()
            active.setdefault(name, []).append(ctx)
        return hook
    def after(name):
        def hook(module, inputs, output):
            active[name].pop().__exit__(None, None, None)
        return hook
    try:
        for name, obj in targets.items():
            handles.append(obj.register_forward_pre_hook(before(name)))
            handles.append(obj.register_forward_hook(after(name)))
        yield
    finally:
        for handle in handles:
            handle.remove()
        for contexts in active.values():
            while contexts:
                contexts.pop().__exit__(None, None, None)
        for obj, name, existed, original in reversed(replacements):
            if existed:
                setattr(obj, name, original)
            else:
                delattr(obj, name)


def shape_info(value):
    if isinstance(value, torch.Tensor):
        return dict(shape=list(value.shape), dtype=str(value.dtype), stride=list(value.stride()),
                    contiguous=value.is_contiguous(), device=str(value.device))
    if isinstance(value, (list, tuple)):
        return [shape_info(v) for v in value]
    return str(type(value))


def capture_shapes(model, image, amp):
    records, handles = {}, []
    for name, obj in model.named_modules():
        if name == 'stage4' or name.startswith('stage4.'):
            def hook(module, inputs, output, _name=name):
                records[_name] = dict(input=shape_info(inputs), output=shape_info(output))
            handles.append(obj.register_forward_hook(hook))
    try:
        with torch.inference_mode(), autocast_context(image.device, amp):
            model.forward_mask_only(image)
    finally:
        for h in handles:
            h.remove()
    return records


def event_components(model, image, count=30):
    """In-context elapsed GPU intervals; diagnostic overhead is not subtracted."""
    rows = []
    stage = model.stage4
    targets = {'Stage4': stage}
    for name in ('local', 'global_branch', 'route_mamba', 'fuse', 'refine'):
        obj = getattr(stage, name, None)
        if isinstance(obj, torch.nn.Module):
            targets[name] = obj
    for repeat in range(3):
        handles, pairs, starts = [], {k: [] for k in targets}, {}
        def pre(name):
            def hook(module, args):
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                starts[name] = event
            return hook
        def post(name):
            def hook(module, args, output):
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                pairs[name].append((starts[name], end))
            return hook
        try:
            for name, obj in targets.items():
                handles.extend([obj.register_forward_pre_hook(pre(name)), obj.register_forward_hook(post(name))])
            # Includes all hooks, so before/after plain benchmark quantifies
            # total instrumentation disturbance, without assuming additivity.
            _, elapsed, _ = benchmark_fps(_MaskOnlyModel(model), image, device=image.device,
                warmup=0, iterations=count, amp='fp16')
            rows.append(dict(component='full_model_with_event_hooks', repeat=repeat,
                             metric='elapsed_gpu_interval', **summary([elapsed])))
            for name, values in pairs.items():
                rows.append(dict(component=name, repeat=repeat, metric='elapsed_gpu_interval',
                                 **summary([a.elapsed_time(b) for a, b in values])))
        finally:
            for h in handles:
                h.remove()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'runs/diagnostics/dpsm_profiling_202610')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--profile-iterations', type=int, default=10)
    args = parser.parse_args()
    if args.profile_iterations < 1:
        parser.error('--profile-iterations must be positive')
    out = args.output_root / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda')
    env = measurement_environment(device)
    smi = shutil.which('nvidia-smi')
    hardware = subprocess.run([smi, '--query-gpu=name,driver_version,memory.total,utilization.gpu,pstate',
        '--format=csv'], capture_output=True, text=True).stdout.strip() if smi else None
    source_files = [Path(__file__).resolve(), ROOT / 'scripts/measure_model_complexity.py']
    source_files += list((ROOT / 'models').glob('*.py'))
    source_files += [ROOT / v[0] for v in VARIANTS.values()]
    hashes = {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    meta = dict(status='preflight', created=datetime.now().astimezone().isoformat(),
        environment=env, nvidia_smi=hardware, source_hashes=hashes,
        protocol=dict(input=[1, 3, 512, 512], amp='fp16 autocast; parameters FP32',
                      warmup=100, iterations=300, repeats=3, inference_path='mask-only',
                      weights='random initialization; no checkpoints',
                      statistics='mean/median/population std over three round means; not per-frame quantiles'),
        profiling=dict(event_iterations=30, event_repeats=3, profiler_iterations=args.profile_iterations,
                       note='Nested inclusive intervals and attributed kernel sums are NOT additive; profiler CPU self time is not pure launch overhead.'))
    component_rows, operator_rows = [], []
    try:
        if args.audit_only:
            torch.manual_seed(42)
            _, model = load_model(ROOT / VARIANTS['T2'][0], torch.device('cpu'))
            route = model.stage4.route_mamba
            x = torch.randn(1, 320, 8, 9)
            state = {k: v.clone() for k, v in model.state_dict().items()}
            with torch.inference_mode():
                expected = route(x)
                with annotations(model), torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                    actual = route(x)
                restored = route(x)
            assert all(torch.equal(a, b) and torch.equal(a, c) for a, b, c in zip(expected, actual, restored))
            assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())
            meta['annotation_audit'] = dict(status='passed', backend=route.backends,
                input=list(x.shape), outputs_bitwise_equal=True, state_dict_unchanged=True,
                scope='CPU fallback correctness only; CUDA fused path NOT validated')
            markers = sorted(e.key for e in prof.key_averages() if
                             e.key.startswith(('DPSM', 'scan/', 'inverse/', 'Mamba/')))
            assert all('inverse/' + d in markers and 'scan/' + d in markers for d in route.directions)
            assert 'DPSM_op/packed' in markers and 'DPSM/shared_mamba' in markers
            meta['annotation_audit']['observed_markers'] = markers
            for tag in VARIANTS:
                _, check_model = load_model(ROOT / VARIANTS[tag][0], torch.device('cpu'))
                check_image = torch.randn(1, 3, 64, 96)
                with torch.inference_mode():
                    before = check_model.forward_mask_only(check_image)['mask']
                    with annotations(check_model):
                        after = check_model.forward_mask_only(check_image)['mask']
                    assert torch.equal(before, after), tag
                assert not any(m._forward_hooks or m._forward_pre_hooks for m in check_model.modules())
            meta['annotation_audit']['all_four_full_models_equal_at_64x96'] = True
            # Obtain actual full-network dimensions; no CPU time is reported.
            meta['cpu_shapes'] = capture_shapes(model, torch.randn(1, 3, 512, 512), 'off')
            meta['status'] = 'audit_only_no_gpu_measurement'
        else:
            reasons = []
            if not env['cuda_available']:
                reasons.append('CUDA PyTorch unavailable')
            if '4090d' not in re.sub('[^a-z0-9]', '', str(env['gpu']).lower()):
                reasons.append('required RTX 4090 D unavailable')
            from models.mamba_blocks import Mamba
            if Mamba is None:
                reasons.append('mamba_ssm backend unavailable; fallback timing prohibited')
            if reasons:
                meta.update(status='blocked_environment', reason='; '.join(reasons))
            else:
                for tag, (config, expected_cores, _) in VARIANTS.items():
                    torch.manual_seed(42)
                    _, model = load_model(ROOT / config, device)
                    cores = [m for m in model.stage4.modules() if isinstance(m, SequenceMambaCore)]
                    assert len(cores) == expected_cores and all(m.backend == 'mamba_ssm' for m in cores)
                    image = torch.randn(1, 3, 512, 512, device=device)
                    mask_audit = validate_mask_only_equivalence(model, image, device=device, amp='fp16')
                    dump(out / (tag + '.shapes.json'), capture_shapes(model, image, 'fp16'))
                    with torch.inference_mode(), autocast_context(device, 'fp16'):
                        expected = model.forward_mask_only(image)['mask']
                        with annotations(model):
                            actual = model.forward_mask_only(image)['mask']
                        assert torch.equal(expected, actual), 'Annotated GPU output changed'
                    del expected, actual
                    full = {}
                    for phase in ('before', 'after'):
                        times, memory, fps = [], [], []
                        for repeat in range(3):
                            rate, latency, peak = benchmark_fps(_MaskOnlyModel(model), image, device=device,
                                warmup=100, iterations=300, amp='fp16')
                            times.append(latency); memory.append(peak); fps.append(rate)
                        full[phase] = dict(**summary(times), median_fps=statistics.median(fps),
                                           peak_allocated_bytes=max(memory))
                        component_rows.append(dict(model=tag, component='full_model_' + phase,
                            repeat='three_round_means', metric='historical_benchmark_fps',
                            **summary(times)))
                        if phase == 'before':
                            for row in event_components(model, image):
                                component_rows.append(dict(model=tag, **row))
                            with annotations(model), torch.inference_mode(), autocast_context(device, 'fp16'):
                                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.CUDA], record_shapes=True,
                                    profile_memory=True, with_stack=False) as prof:
                                    for _ in range(args.profile_iterations):
                                        model.forward_mask_only(image)
                                    torch.cuda.synchronize()
                            prof.export_chrome_trace(str(out / (tag + '.trace.json')))
                            for event in prof.key_averages(group_by_input_shape=True):
                                gpu = getattr(event, 'device_time_total', getattr(event, 'cuda_time_total', 0))
                                self_gpu = getattr(event, 'self_device_time_total', getattr(event, 'self_cuda_time_total', 0))
                                operator_rows.append(dict(model=tag, operator=event.key, calls=event.count,
                                    input_shapes=str(event.input_shapes), cpu_total_us=event.cpu_time_total,
                                    self_cpu_us=event.self_cpu_time_total, device_total_us=gpu,
                                    self_device_us=self_gpu,
                                    device_us_per_forward=gpu / args.profile_iterations,
                                    device_memory_net_bytes=getattr(event, 'device_memory_usage', 0)))
                            del prof
                    dump(out / (tag + '.baseline.json'), dict(full=full, mask_audit=mask_audit,
                         environment=measurement_environment(device)))
                    del model, image
                    torch.cuda.empty_cache()
                meta['status'] = 'complete'
    except Exception as exc:
        meta.update(status='failed', reason=repr(exc))
        raise
    finally:
        csv_write(out / 'component_latency.csv', component_rows,
            ['model', 'component', 'repeat', 'metric', 'mean_ms', 'median_ms', 'std_ms', 'samples_ms'])
        csv_write(out / 'operator_summary.csv', operator_rows,
            ['model', 'operator', 'calls', 'input_shapes', 'cpu_total_us', 'self_cpu_us',
             'device_total_us', 'self_device_us', 'device_us_per_forward', 'device_memory_net_bytes'])
        dump(out / 'benchmark_metadata.json', meta)
        (out / 'profiling.log').write_text(meta['status'] + '\n' + meta.get('reason', '') + '\n', encoding='utf-8')
        print(out, meta['status'], meta.get('reason', ''), flush=True)
    return 0 if meta['status'] in ('complete', 'audit_only_no_gpu_measurement') else 2


if __name__ == '__main__':
    raise SystemExit(main())
