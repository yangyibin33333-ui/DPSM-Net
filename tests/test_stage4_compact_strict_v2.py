"""Synthetic-only T0/T2 acceptance audit. Never trains or reads test images.

Run from repository root: python tests/test_stage4_compact_strict_v2.py
--device cuda --require-mamba-ssm validates the formal backend without training.
Reports are written to a NEW directory, never to historical experiments.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import yaml
from models import build_model
from models.blocks import DWConvBlock, StripConvBlock, Stage4LocalOnly, Stage4LocalDPSM80
from models.mamba_blocks import DirectionPartitionedSharedMamba, SequenceMambaCore
from models.transformer_blocks import TransformerGlobalBlock
from utils.auto_protocol import resolve_auto_protocol_v2

NEW = ROOT / 'configs/experiments/stage4_compact_ablation/crack500'
OLD = ROOT / 'configs/experiments/paper_dependency_aware_ablation/crack500'
RUNS = ROOT / 'runs/paper/dependency_aware_ablation_strict_v2/crack500'
NAMES = {'T0': 'T0_LocalOnly_StrictV2', 'T1': 'M1_FrameworkOn_DPSMOff_StructSupOff',
         'T2': 'T2_LocalDPSM80_StrictV2', 'T3': 'M3_FrameworkOn_DPSMOn_StructSupOff'}
ARCH_KEYS = {'stage4_variant', 'transformer_stages', 'use_transformer', 'mamba_stages',
             'use_topology_matched_no_dpsm_stage4', 'use_direction_partitioned_shared_mamba'}


def load(path):
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def make(cfg):
    options = deepcopy(cfg['model'])
    return build_model(options.pop('name'), **options)


def differences(a, b, prefix=''):
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for key in sorted(a.keys() | b.keys()):
            path = f'{prefix}.{key}' if prefix else key
            if key not in a or key not in b:
                result.append([path, a.get(key, '<absent>'), b.get(key, '<absent>')])
            else:
                result.extend(differences(a[key], b[key], path))
        return result
    return [] if a == b else [[prefix, a, b]]


def resolve_from_history(cfg, history):
    """Exercise the REAL resolver with historical train metadata; no data I/O."""
    dist = Counter({tuple(map(int, k.split('x'))): v for k, v in
                    history['auto_protocol_resolved']['native_size_distribution'].items()})
    audit = deepcopy(history['auto_protocol_resolved']['foreground_audit'])
    with patch('utils.auto_protocol.scan_train_image_sizes', return_value=dist), \
         patch('utils.auto_protocol.audit_train_foreground', return_value=audit):
        resolve_auto_protocol_v2(cfg, **{f'cli_{key}': False for key in
                                ('img_size', 'batch_size', 'epochs', 'data_root', 'save_dir')})
    return cfg


def audit(args, report, out):
    torch.set_num_threads(4)
    cfgs = {tag: load((NEW if tag in ('T0', 'T2') else OLD) / f'{name}.yaml')
            for tag, name in NAMES.items()}
    histories = {tag: [load(p) for p in sorted((RUNS / NAMES[tag]).glob('*/config.yaml'))]
                 for tag in ('T1', 'T3')}
    assert all(len(v) == 3 for v in histories.values())
    for tag, items in histories.items():
        assert {h['auto_protocol_resolved']['training_seed'] for h in items} == {42, 123, 3407}
    resolved = {tag: resolve_from_history(deepcopy(cfg), histories['T1' if tag in ('T0', 'T1') else 'T3'][0])
                for tag, cfg in cfgs.items()}
    report['config_diffs'] = {}
    for a, b in itertools.combinations(resolved, 2):
        diff = differences(resolved[a], resolved[b])
        report['config_diffs'][f'{a}_vs_{b}'] = diff
        assert all(p == 'save.save_dir' or p.startswith('model.') and p.split('.')[1] in ARCH_KEYS
                   for p, _, _ in diff), diff
    # Compare ALL historical seeds' resolved settings, excluding measured run
    # state / memory / selection outcomes. Compare immutable budgets explicitly.
    budget_keys = ('selected_img_size', 'effective_batch_size', 'target_updates', 'resolved_epochs',
                   'scnp_warmup_updates', 'bpcm_start_update', 'bpcm_ramp_updates',
                   'resolved_validation_interval_updates', 'min_early_stop_update', 'patience_updates')
    report['historical_comparisons'] = []
    for tag, items in histories.items():
        for h in items:
            target = resolved[tag]
            diff = differences({k: target[k] for k in ('data', 'train', 'loss', 'evaluation')},
                               {k: h[k] for k in ('data', 'train', 'loss', 'evaluation')})
            assert not diff, diff
            budgets = {k: target['auto_protocol_resolved'][k] for k in budget_keys}
            assert budgets == {k: h['auto_protocol_resolved'][k] for k in budget_keys}
            report['historical_comparisons'].append({'tag': tag,
                'seed': h['auto_protocol_resolved']['training_seed'], 'diffs': diff, 'budgets': budgets})
    (out / 'resolved_configs.json').write_text(json.dumps(resolved, indent=2), encoding='utf-8')
    report['models'] = {}
    for tag, cfg in cfgs.items():
        torch.manual_seed(42)
        model = make(cfg).to(args.device)
        stage = model.stage4
        has_tr = any(isinstance(m, TransformerGlobalBlock) for m in stage.modules())
        has_dp = any(isinstance(m, DirectionPartitionedSharedMamba) for m in stage.modules())
        assert has_tr == (tag in ('T1', 'T3')) and has_dp == (tag in ('T2', 'T3'))
        assert isinstance(stage.local[0], DWConvBlock) and isinstance(stage.local[1], StripConvBlock)
        if tag in ('T0', 'T2'):
            assert not hasattr(stage, 'global_branch')
            assert not any(isinstance(m, torch.nn.Identity) for m in stage.modules())
        if tag == 'T0':
            assert isinstance(stage, Stage4LocalOnly) and not hasattr(stage, 'route_mamba')
        if tag == 'T2':
            assert isinstance(stage, Stage4LocalDPSM80)
        cores = [m for m in model.modules() if isinstance(m, SequenceMambaCore)]
        if args.require_mamba_ssm:
            assert all(c.backend == 'mamba_ssm' for c in cores), 'Formal Mamba backend unavailable'
        conv = next(m for m in stage.fuse.modules() if isinstance(m, torch.nn.Conv2d))
        assert conv.in_channels == {'T0': 320, 'T1': 640, 'T2': 640, 'T3': 960}[tag]
        row = {'class': type(stage).__name__, 'params_actual_backend': sum(p.numel() for p in model.parameters()),
               'backends': model.mamba_backend_info(), 'fuse': [conv.in_channels, conv.out_channels],
               'stage4_modules': {k: type(v).__name__ for k, v in stage.named_modules()}, 'forward_backward': []}
        report['models'][tag] = row
        # Real historical checkpoint schema vs current schema. Optional backend
        # differences are disclosed separately rather than silently ignored.
        if tag in histories:
            checkpoint = sorted((RUNS / NAMES[tag]).glob('*/best_structural_score.pth'))[0]
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            state = saved['model']
            actual = model.state_dict()
            backend_prefixes = tuple(n + '.core.' for n, m in model.named_modules() if isinstance(m, SequenceMambaCore))
            filtered = lambda d: {k: tuple(v.shape) for k, v in d.items() if not k.startswith(backend_prefixes)}
            assert filtered(state) == filtered(actual)
            row['checkpoint_schema'] = {'path': str(checkpoint), 'non_backend_keys_shapes': 'PASS',
                                        'full_schema_equal': {k: tuple(v.shape) for k, v in state.items()} ==
                                                            {k: tuple(v.shape) for k, v in actual.items()}}
            if row['checkpoint_schema']['full_schema_equal']:
                model.load_state_dict(state, strict=True)
            del saved, state
        for height, width in ((352, 640), (512, 512)):
            model.train()
            model.zero_grad(set_to_none=True)
            calls = []
            hooks = [c.core.register_forward_pre_hook(lambda m, inputs: calls.append(list(inputs[0].shape))) for c in cores]
            x = torch.randn(1, 3, height, width, device=args.device, requires_grad=True)
            outputs = model(x)
            for handle in hooks:
                handle.remove()
            assert all(torch.isfinite(v).all() for v in outputs.values() if isinstance(v, torch.Tensor))
            assert outputs['mask'].shape == (1, 1, height, width)
            objective = sum(v.float().square().mean() for v in outputs.values() if isinstance(v, torch.Tensor))
            objective.backward()
            assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
            missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
            assert not missing, missing
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            assert stage.local[0].state_dict()  # real local parameters were retained
            if has_dp:
                assert len(calls) == 1 and calls[0][0] == 4 and calls[0][2] == 80, calls
            row['forward_backward'].append({'input': list(x.shape), 'outputs': {k: list(v.shape) for k, v in outputs.items()},
                                            'packed_core_inputs': calls, 'all_gradients_finite': True})
            print(tag, height, width, 'forward/backward PASS', flush=True)
            del outputs, objective, x
        model.zero_grad(set_to_none=True)
        # Use the existing compatible fvcore function; fallback T2 FLOPs must
        # never be presented as formal Mamba FLOPs.
        if tag in ('T0', 'T2'):
            from scripts.measure_model_complexity import measure_flops
            model.eval()
            row['flops'] = []
            for h, w in ((352, 640), (512, 512)):
                with torch.inference_mode():
                    value, unsupported = measure_flops(model, torch.randn(1, 3, h, w, device=args.device), inference_path='mask-only')
                row['flops'].append({'input': [1, 3, h, w], 'fvcore_counted': value,
                    'unsupported': dict(unsupported), 'status': 'partial count; fallback diagnostic only' if cores and cores[0].backend != 'mamba_ssm' else 'fvcore count; unsupported ops disclosed'})
        del model
    # Exact formal parameter count from the real T3 checkpoint schema. Only
    # SequenceMambaCore.core differs between the desktop fallback and Mamba.
    # This is a tensor-count calculation, NOT a formal-backend execution claim.
    reference = make(cfgs['T3'])
    checkpoint = sorted((RUNS / NAMES['T3']).glob('*/best_structural_score.pth'))[0]
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)['model']
    prefixes = tuple(n + '.core.' for n, m in reference.named_modules() if isinstance(m, SequenceMambaCore))
    current_core_params = sum(p.numel() for n, p in reference.named_parameters() if n.startswith(prefixes))
    saved_core_params = sum(p.numel() for n, p in saved.items() if n.startswith(prefixes))
    delta = saved_core_params - current_core_params
    for tag, row in report['models'].items():
        row['formal_params_from_checkpoint_schema'] = row['params_actual_backend'] + (delta if tag in ('T2', 'T3') else 0)
    report['formal_parameter_evidence'] = {'checkpoint': str(checkpoint),
        'desktop_core_params': current_core_params, 'checkpoint_core_params': saved_core_params,
        'delta': delta, 'method': 'replace core parameter counts only; non-backend schema equality checked above'}
    del reference, saved
    report['invalid_variant_guards'] = []
    for tag, change in (('T0', {'use_direction_partitioned_shared_mamba': True}),
                        ('T2', {'use_topology_matched_no_dpsm_stage4': True}),
                        ('T2', {'transformer_stages': [4], 'use_transformer': True}),
                        ('T2', {'stage4_hybrid_order': 'dpsm_then_transformer'}),
                        ('T2', {'dpsm_route_mode': 'uniform'})):
        invalid = deepcopy(cfgs[tag]); invalid['model'].update(change)
        try:
            make(invalid)
        except ValueError:
            report['invalid_variant_guards'].append({'tag': tag, 'change': change, 'status': 'rejected'})
        else:
            raise AssertionError(f'Invalid combination accepted: {tag} {change}')
    # Same-seed retained tensors must match the corresponding unpruned model.
    report['retained_initialization'] = {}
    for new, old in (('T0', 'T1'), ('T2', 'T3')):
        torch.manual_seed(42)
        a = make(cfgs[new]).state_dict()
        torch.manual_seed(42)
        b = make(cfgs[old]).state_dict()
        keys = [k for k in a if not k.startswith('stage4.fuse.')]
        assert all(k in b and torch.equal(a[k], b[k]) for k in keys)
        report['retained_initialization'][new] = {'reference': old, 'identical_tensors': len(keys)}
    if args.baseline_source:
        spec = importlib.util.spec_from_file_location('models._stage4_before', args.baseline_source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        report['legacy_before_after'] = {}
        for tag in ('T1', 'T3'):
            options = deepcopy(cfgs[tag]['model']); options.pop('name')
            torch.manual_seed(42); old = module.DEMambaNet(**options).state_dict()
            torch.manual_seed(42); new = make(cfgs[tag]).state_dict()
            assert old.keys() == new.keys() and all(torch.equal(old[k], new[k]) for k in old)
            report['legacy_before_after'][tag] = 'all state_dict keys, shapes and initialized values identical'
    report['status'] = 'PASS'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--require-mamba-ssm', action='store_true')
    parser.add_argument('--baseline-source', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    out = args.output_dir or ROOT / 'tests/stage4_reports' / datetime.now().strftime('%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    files = [ROOT / 'train.py', ROOT / 'test.py', ROOT / 'models/mamba_blocks.py']
    files += list((ROOT / 'losses').glob('*.py')) + list((ROOT / 'metrics').glob('*.py'))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    report = {'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(),
              'device': args.device, 'status': 'RUNNING', 'untouched_hashes': hashes}
    try:
        audit(args, report, out)
        assert hashes == {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    except Exception as error:
        report.update(status='FAIL', error=repr(error))
        raise
    finally:
        (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('Report:', out, flush=True)


if __name__ == '__main__':
    main()
