"""CPU compatibility tests; never produce paper GPU-speed measurements."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scripts.measure_model_complexity import (
    load_model, audit_strict_stage4_deletion, audit_stage4_mamba,
    capture_stage4_execution, validate_mask_only_equivalence,
)
from scripts.measure_stage4_pruning_efficiency import (
    VARIANTS, command, comparison_rows, history_comparison,
)


class Stage4EfficiencyAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def build(self, tag):
        return load_model(ROOT / VARIANTS[tag][0], torch.device('cpu'))[1]

    def test_real_deleted_branches_and_mask_only_equivalence(self):
        image = torch.randn(1, 3, 64, 96)
        for tag in ('T0', 'T2'):
            model = self.build(tag)
            info = audit_strict_stage4_deletion(model)
            self.assertFalse(info['stage4_transformer_present'])
            self.assertEqual(info['stage4_fusion_in_channels'], 320 if tag == 'T0' else 640)
            with torch.inference_mode():
                outputs, captured = capture_stage4_execution(model, image)
                contract = validate_mask_only_equivalence(model, image, device=torch.device('cpu'), amp='off')
            self.assertTrue(contract['mask_only_allclose'])
            if tag == 'T0':
                audit = audit_stage4_mamba(model, variant=tag, expected_cores=0,
                    outputs=outputs, captured_inputs=captured, formal_backend=True)
                self.assertEqual(audit['mamba_forward_call_count'], 0)
            else:
                self.assertEqual(captured, {0: [(4, 96, 80)]})
                audit_stage4_mamba(model, variant=tag, expected_cores=None,
                    outputs=outputs, captured_inputs=captured, formal_backend=False)
                with self.assertRaisesRegex(RuntimeError, 'mamba_ssm'):
                    audit_stage4_mamba(model, variant=tag, expected_cores=1,
                        outputs=outputs, captured_inputs=captured, formal_backend=True)
            model.stage4.global_branch = torch.nn.Identity()
            with self.assertRaisesRegex(RuntimeError, 'physically delete'):
                audit_strict_stage4_deletion(model)

    def test_legacy_classes_still_pass_unchanged_contracts(self):
        image = torch.randn(1, 3, 64, 96)
        for tag in ('T1', 'T3'):
            model = self.build(tag)
            self.assertEqual(audit_strict_stage4_deletion(model), {})
            with torch.inference_mode():
                outputs, captured = capture_stage4_execution(model, image)
            audit_stage4_mamba(model, variant=tag, expected_cores=0 if tag == 'T1' else None,
                              outputs=outputs, captured_inputs=captured, formal_backend=False)

    def test_t1_t3_match_historical_l0_l3_model_construction(self):
        for tag, filename in (('T1', 'L0_NoMamba.yaml'), ('T3', 'L3_DPSM80.yaml')):
            torch.manual_seed(42)
            current = self.build(tag).state_dict()
            torch.manual_seed(42)
            original = load_model(ROOT / 'configs/experiments/dpsm_lightweight_ablation/crack500' / filename,
                                  torch.device('cpu'))[1].state_dict()
            self.assertEqual(current.keys(), original.keys())
            self.assertTrue(all(torch.equal(current[key], original[key]) for key in current))

    def test_protocol_command_and_unknown_historical_settings(self):
        args = command('T2', Path('new-output'), 'cuda')
        self.assertEqual(args[args.index('--warmup') + 1], '100')
        self.assertEqual(args[args.index('--iterations') + 1], '300')
        self.assertEqual(args[args.index('--benchmark_repeats') + 1], '3')
        self.assertEqual(args[args.index('--inference_path') + 1], 'mask-only')
        self.assertEqual(args[args.index('--amp') + 1], 'fp16')
        self.assertNotIn('--checkpoint', args)
        env = dict(pytorch='2.1.0+cu118', cuda='11.8', mamba_ssm='2.0.3', gpu='NVIDIA GeForce RTX 4090 D')
        comparison = history_comparison(env, env)
        self.assertFalse(comparison['fully_verified_historical_equivalence'])
        self.assertIn('tf32_matmul', comparison['unrecorded_historical_settings'])
        self.assertEqual(comparison['recorded_differences'], {})
        rows = comparison_rows({}, {})
        self.assertTrue(all(row['FPS'] is None and row['Memory(MiB)'] is None for row in rows))

    def test_t0_full_historical_engine_cli_cpu_smoke(self):
        with tempfile.TemporaryDirectory(prefix='stage4-efficiency-test-') as directory:
            target = Path(directory) / 'cpu-smoke.json'
            args = [sys.executable, str(ROOT / 'scripts/measure_model_complexity.py'),
                '--config', str(ROOT / VARIANTS['T0'][0]), '--device', 'cpu',
                '--img_size', '64', '96', '--batch_size', '1', '--amp', 'off',
                '--warmup', '0', '--iterations', '1', '--benchmark_repeats', '1',
                '--inference_path', 'mask-only', '--expected-mamba-cores', '0',
                '--report_json', str(target)]
            result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(target.read_text())[0]
            self.assertEqual(report['stage4_class'], 'Stage4LocalOnly')
            self.assertEqual(report['params'], 1681107)
            self.assertEqual(report['mamba_fused_missing_flops_estimated'], 0)
            self.assertEqual(report['inference_path'], 'mask-only')
            self.assertEqual(len(report['latency_round_means_ms']), 1)
            self.assertIsNone(report['gpu'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
