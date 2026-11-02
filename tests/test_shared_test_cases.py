"""Run from repo root: python tests/test_shared_test_cases.py. No CUDA/torch."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('shared_cases', ROOT / 'scripts/select_shared_test_cases.py')
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def rejects(function, fragment):
    try:
        function()
    except (ValueError, FileExistsError) as error:
        assert fragment in str(error), error
    else:
        raise AssertionError('Invalid registration accepted')


def fixture(root):
    dataset = root / 'data'
    for sub in ('test/images', 'test/masks'):
        (dataset / sub).mkdir(parents=True)
    for i in range(8):
        stem = f'group{i%3}_image ({i})'
        rgb = np.zeros((24, 32, 3), np.uint8)
        rgb[:, :, 0] = 20 + i * 20
        gt = np.zeros((24, 32), np.uint8)
        gt[3:21, 5+i] = 255
        Image.fromarray(rgb).save(dataset / 'test/images' / (stem + '.bmp'))
        Image.fromarray(gt).save(dataset / 'test/masks' / (stem + '.png'))
    config = {'root': str(dataset), 'image_dir': 'test/images', 'mask_dir': 'test/masks',
              'foreground': {'mode': 'binary', 'evaluation_threshold': 127},
              'test_provenance_note': 'Synthetic known test split',
              'evaluation_evidence': [{'base': 'repo', 'path': 'test.py'}]}
    return {'datasets': {'CrackMap': config}}


def test_pairing_encoding_and_pending():
    with tempfile.TemporaryDirectory() as temp:
        mapping = fixture(Path(temp))
        audit = tool.preflight(mapping)
        assert audit['datasets']['CrackMap']['test_count'] == 8
        assert any(p['dataset'] == 'CFD' for p in audit['pending_items'])
        config = mapping['datasets']['CrackMap']
        bad = copy.deepcopy(config); bad['image_dir'] = 'val/images'
        rejects(lambda: tool.audit_dataset('CrackMap', bad), 'cannot relabel')
        root = Path(config['root'])
        image = next((root / 'test/images').glob('*'))
        mask = next((root / 'test/masks').glob('*'))
        Image.fromarray((np.asarray(Image.open(mask)) > 0).astype(np.uint8)).save(mask)
        rejects(lambda: tool.decode(image, mask, config['foreground']), 'disagrees')
        _, gt, rule = tool.decode(image, mask, {'mode': 'binary', 'evaluation_threshold': 0})
        assert gt.max() == 255 and rule['encoding'].startswith('0/1')
        duplicate = image.with_suffix('.png')
        Image.open(image).save(duplicate)
        rejects(lambda: tool.audit_dataset('CrackMap', config), 'Duplicate identifier')


def test_determinism_confirmation_freeze_and_tampering():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp); mapping = fixture(root)
        a = tool.preview(mapping, root/'a', 8)
        b = tool.preview(mapping, root/'b', 8)
        assert a == b
        confirm = tool.read_json(root/'a/confirmation.template.json')
        confirmation = root/'confirm.json'
        tool.write_json(confirmation, confirm)
        rejects(lambda: tool.freeze(mapping, root/'a', confirmation), 'reviewer')
        confirm.update(reviewer='synthetic-test reviewer', reviewed_input_gt=True, test_split_verified=True)
        for decision in confirm['datasets']['CrackMap']:
            decision.update(scene_note='Synthetic vertical line', coverage_note='No genuine low-contrast/material claim')
        confirmation.unlink(); tool.write_json(confirmation, confirm)
        frozen = tool.freeze(mapping, root/'a', confirmation)
        assert frozen['status'] == 'frozen_partial'
        checksum = frozen.pop('manifest_sha256')
        assert checksum == tool.digest(tool.canonical(frozen))
        assert len(frozen['datasets']['CrackMap']['cases']) == 6
        rejects(lambda: tool.freeze(mapping, root/'a', confirmation), 'Frozen')
        image = next((Path(mapping['datasets']['CrackMap']['root'])/'test/images').glob('*'))
        pixels = np.array(Image.open(image)); pixels[0,0] = 1; Image.fromarray(pixels).save(image)
        rejects(lambda: tool.freeze(mapping, root/'b', confirmation), 'changed')
        rejects(lambda: tool.validate_roi([30,0,10,1], [32,24]), 'outside')


def test_authoritative_lists_missing_pairs_and_suffixes():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp); mapping = fixture(root)
        config = mapping['datasets']['CrackMap']; dataset = Path(config['root'])
        images = sorted((dataset/'test/images').glob('*'))
        manifest = dataset/'original_test.txt'
        # Preserve parentheses and spaces; a list selects a subset of a flat corpus.
        manifest.write_text('\n'.join(p.stem for p in images[:6]), encoding='utf-8')
        config['split_source'] = {'kind': 'stem_list', 'path': 'original_test.txt'}
        audit = tool.audit_dataset('CrackMap', config)
        assert audit['test_count'] == 6
        assert audit['split_source']['sha256'] == tool.file_hash(manifest)
        manifest.write_text('missing case (1)\n', encoding='utf-8')
        rejects(lambda: tool.audit_dataset('CrackMap', config), 'Missing test-list pairs')
        del config['split_source']
        for mask in (dataset/'test/masks').glob('*'):
            mask.rename(mask.with_stem(mask.stem+'_mask'))
        config['mask_stem_suffix'] = '_mask'
        audit = tool.audit_dataset('CrackMap', config)
        assert audit['cases'][0]['original_mask_stem'].endswith('_mask')
        mask = next((dataset/'test/masks').glob('*'))
        mask.unlink()
        rejects(lambda: tool.audit_dataset('CrackMap', config), 'Unpaired')


if __name__ == '__main__':
    for test in (test_pairing_encoding_and_pending, test_determinism_confirmation_freeze_and_tampering,
                 test_authoritative_lists_missing_pairs_and_suffixes):
        test()
        print('PASS:', test.__name__)
