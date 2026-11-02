"""Input/GT-only, immutable shared test-case registration. No torch/model imports."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

DATASETS = ('DeepCrack', 'CamCrack789', 'CrackMap', 'Crack500', 'CFD', 'TUT')
REPO = Path(__file__).resolve().parents[1]
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp'}  # Same formats as the actual test loader.
GOALS = ('thin crack', 'branch/intersection', 'curved/multidirectional',
         'low contrast (review Input)', 'background texture (review Input)', 'reference')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')


def file_hash(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else digest(handle.read())


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    # Exclusive creation, including pending artifacts: never overwrite a prior registration.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def relative(root, name):
    if Path(name).is_absolute():
        raise ValueError(f'Expected a root-relative path, not absolute: {name}')
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes root: {name}')
    return path


def index(directory, strip_suffix=''):
    if not directory.is_dir():
        raise ValueError(f'Missing directory: {directory}')
    result = {}
    for path in sorted(directory.iterdir()):  # Test loader does not recursively add subdirectories.
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            stem = path.stem
            if strip_suffix and stem.endswith(strip_suffix):
                stem = stem[:-len(strip_suffix)]
            if stem in result:
                raise ValueError(f'Duplicate identifier {stem!r}: {result[stem]} / {path}')
            result[stem] = path
    return result


def decode(image, mask, policy):
    with Image.open(image) as handle:
        rgb = np.array(handle.convert('RGB'), dtype=np.uint8)
    with Image.open(mask) as handle:
        # Match evaluation's convert(L), not a palette-index assumption.
        raw = np.array(handle.convert('L'), dtype=np.uint8)
    if rgb.shape[:2] != raw.shape:
        raise ValueError(f'Image/GT size mismatch: {image.name}')
    values = np.unique(raw).tolist()
    threshold = policy.get('evaluation_threshold')
    if threshold is None:
        raise ValueError('foreground.evaluation_threshold must come from the actual evaluation config')
    mode = policy.get('mode', 'binary')
    if mode == 'binary':
        if set(values) <= {0, 1}:
            binary = raw > 0
            encoding = '0/1 (foreground=1)'
        elif set(values) <= {0, 255}:
            binary = raw > 0
            encoding = '0/255 (foreground=255)'
        else:
            raise ValueError(f'Non-binary GT values {values[:20]}; explicitly verify and use mode=threshold')
    elif mode == 'threshold':
        binary = raw > int(threshold)
        encoding = f'L > {threshold}; explicitly verified grayscale encoding'
    else:
        raise ValueError(f'Unknown foreground mode: {mode}')
    if not np.array_equal(binary, raw > int(threshold)):
        raise ValueError(f'GT encoding disagrees with evaluation threshold={threshold}: {mask}')
    return rgb, binary.astype(np.uint8) * 255, {'mode': mode, 'encoding': encoding,
            'observed_values': values, 'evaluation_threshold': threshold, 'conversion': 'Pillow.convert(L)'}


def audit_dataset(name, spec):
    root = Path(spec['root']).resolve()
    if spec.get('split', 'test') != 'test':
        raise ValueError('Only the actual test split is admissible')
    for field in ('image_dir', 'mask_dir'):
        if any(p.lower() in {'train', 'val', 'validation'} for p in Path(spec[field]).parts):
            raise ValueError(f'{field} points to train/val; cannot relabel it test')
    evidence = []
    for item in spec.get('evaluation_evidence', []):
        if item['base'] not in ('repo', 'dataset'):
            raise ValueError('Evidence base must be repo or dataset')
        base = REPO if item['base'] == 'repo' else root
        path = relative(base, item['path'])
        evidence.append({**item, 'sha256': file_hash(path)})
    if not evidence or not spec.get('test_provenance_note'):
        raise ValueError('Provide final evaluation evidence and test_provenance_note')
    images = index(relative(root, spec['image_dir']), spec.get('image_stem_suffix', ''))
    masks = index(relative(root, spec['mask_dir']), spec.get('mask_stem_suffix', ''))
    source = spec.get('split_source')
    provenance = {'kind': 'evaluation_directory_inventory', 'evidence': evidence,
                  'note': spec['test_provenance_note']}
    if source:
        path = relative(root, source['path'])
        if source['kind'] == 'stem_list':
            ids = [s.strip() for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
        elif source['kind'] == 'source_order_tsv':
            with path.open(encoding='utf-8-sig', newline='') as handle:
                ids = [row['source_stem'] for row in csv.DictReader(handle, delimiter='\t')]
        else:
            raise ValueError('split_source kind must be stem_list or source_order_tsv')
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate IDs in test split source')
        missing = set(ids) - (images.keys() & masks.keys())
        if missing:
            raise ValueError(f'Missing test-list pairs: {sorted(missing)}')
        # A supplied list can select the original test IDs from a flat corpus.
        images = {s: images[s] for s in ids}
        masks = {s: masks[s] for s in ids}
        provenance.update({'kind': source['kind'], 'path': source['path'], 'sha256': file_hash(path)})
    if images.keys() != masks.keys() or not images:
        raise ValueError(f'Unpaired/empty test set: images-only={sorted(images.keys()-masks.keys())[:10]}, '
                         f'masks-only={sorted(masks.keys()-images.keys())[:10]}')
    for split in ('train', 'val'):
        directory = root / split / 'images'
        if directory.is_dir():
            overlap = images.keys() & index(directory).keys()
            if overlap:
                raise ValueError(f'Test/{split} identifier overlap: {sorted(overlap)[:10]}')
    rows, seen = [], {}
    for stem in sorted(images):
        image, mask = images[stem], masks[stem]
        rgb, gt, rule = decode(image, mask, spec['foreground'])
        rgb_hash = digest(rgb.tobytes())
        if rgb_hash in seen:
            raise ValueError(f'Duplicate decoded test image: {stem!r} / {seen[rgb_hash]!r}')
        seen[rgb_hash] = stem
        height, width = gt.shape
        rows.append({'dataset': name, 'split': 'test', 'pair_id': stem, 'original_stem': image.stem,
                     'original_mask_stem': mask.stem,
                     'image_relative_path': image.relative_to(root).as_posix(),
                     'mask_relative_path': mask.relative_to(root).as_posix(),
                     'original_size': [width, height], 'image_file_sha256': file_hash(image),
                     'mask_file_sha256': file_hash(mask), 'rgb_pixel_sha256': rgb_hash,
                     'gt_binary_pixel_sha256': digest(gt.tobytes()), 'foreground_rule': rule})
    provenance['inventory_sha256'] = digest(canonical(rows))
    return {'split_source': provenance, 'test_count': len(rows), 'cases': rows}


def preflight(mapping):
    result = {'schema_version': 1, 'selection_seed': 42, 'datasets': {}, 'pending_items': []}
    for name in DATASETS:
        try:
            result['datasets'][name] = audit_dataset(name, mapping['datasets'][name])
        except (ValueError, OSError, KeyError) as error:
            result['pending_items'].append({'dataset': name, 'reason': str(error)})
    return result


def measures(rgb, gt):
    binary = np.asarray(Image.fromarray(gt).resize((192, 192), Image.Resampling.NEAREST)) > 0
    gray = np.asarray(Image.fromarray(rgb).convert('L').resize((192, 192)), dtype=float)
    neighbors = sum(np.roll(np.roll(binary, y, 0), x, 1) for y, x in ((-1,0),(1,0),(0,-1),(0,1)))
    # Descriptors are candidate heuristics, NOT verified scene labels.
    edge = binary & (neighbors < 4)
    thin = float(edge.sum() / max(binary.sum(), 1))
    branch = float((binary & (neighbors >= 3)).mean())
    gy, gx = np.gradient(gray)
    multi = float(min(np.abs(gx[binary]).sum(), np.abs(gy[binary]).sum()))
    contrast = abs(float(gray[binary].mean() - gray[~binary].mean())) if binary.any() and (~binary).any() else 255.
    texture = float(np.hypot(gx, gy)[~binary].mean()) if (~binary).any() else 0.
    return [thin, branch, multi, -contrast, texture, -abs(float(binary.mean()) - .03)]


def suggested_roi(gt):
    ys, xs = np.nonzero(gt)
    if not len(xs):
        return None
    height, width = gt.shape
    w, h = max(1, width//3), max(1, height//3)
    x = max(0, min(width-w, int(np.median(xs))-w//2))
    y = max(0, min(height-h, int(np.median(ys))-h//2))
    return [x, y, w, h]


def panel(rgb, gt, title, roi):
    canvas = Image.new('RGB', (800, 330), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill='black')
    for offset, array in ((0, rgb), (400, gt)):
        picture = Image.fromarray(array).convert('RGB')
        if roi:
            x, y, w, h = roi
            ImageDraw.Draw(picture).rectangle((x, y, x+w-1, y+h-1), outline='red', width=max(1,picture.width//250))
        picture = ImageOps.contain(picture, (390, 290))
        canvas.paste(picture, (offset+5, 32))
    return canvas


def preview(mapping, output, pool_size=18):
    audit = preflight(mapping)
    write_json(output / 'preflight.json', audit)
    candidate = {'schema_version': 1, 'selection_seed': 42, 'status': 'pending_visual_confirmation',
                 'datasets': {}, 'pending_items': audit['pending_items']}
    confirmation = {'reviewer': '', 'reviewed_input_gt': False, 'test_split_verified': False, 'datasets': {}}
    for name, dataset in audit['datasets'].items():
        spec = mapping['datasets'][name]
        root = Path(spec['root']).resolve()
        scored = []
        for row in dataset['cases']:
            rgb, gt, _ = decode(relative(root, row['image_relative_path']), relative(root, row['mask_relative_path']), spec['foreground'])
            if not gt.any():
                continue
            match = re.search(spec.get('source_group_regex', r'^([^_]+)'), row['original_stem'])
            group = match.group(1) if match else row['original_stem']
            scored.append((row, measures(rgb, gt), group))
        selected, groups = [], set()
        for n in range(min(pool_size, len(scored))):
            goal = n % 6
            options = [r for r in scored if r[0]['original_stem'] not in {s[0]['original_stem'] for s in selected}]
            diverse = [r for r in options if r[2] not in groups]
            options = diverse or options
            options.sort(key=lambda r: (-r[1][goal], digest(f"42:{name}:{r[0]['original_stem']}".encode())))
            chosen = options[0]
            selected.append(chosen)
            groups.add(chosen[2])
        records, panels = [], []
        for number, (row, scores, group) in enumerate(selected, 1):
            rgb, gt, _ = decode(relative(root, row['image_relative_path']), relative(root, row['mask_relative_path']), spec['foreground'])
            record = {**row, 'candidate_number': number, 'source_group': group,
                      'candidate_goal': GOALS[(number-1)%6], 'heuristic_scores': scores,
                      'suggested_roi': suggested_roi(gt)}
            records.append(record)
            picture = panel(rgb, gt, f'{name} #{number:02d}: {row["original_stem"]}', record['suggested_roi'])
            directory = output / 'previews' / name
            directory.mkdir(parents=True, exist_ok=True)
            picture.save(directory / f'{number:02d}.png')
            panels.append(picture)
        for start in range(0, len(panels), 6):
            sheet = Image.new('RGB', (1600, 990), 'white')
            for i, picture in enumerate(panels[start:start+6]):
                sheet.paste(picture, ((i%2)*800, (i//2)*330))
            sheet.save(output / 'previews' / name / f'contact_{start//6+1:02d}.png')
        candidate['datasets'][name] = {'split_source': dataset['split_source'], 'cases': records}
        confirmation['datasets'][name] = [{'candidate_number': n, 'scene_note': '',
                  'coverage_note': '', 'accept_suggested_roi': True} for n in range(1, min(6,len(records))+1)]
    write_json(output / 'candidates.json', candidate)
    write_json(output / 'confirmation.template.json', confirmation)
    write_json(output / 'pending_items.json', audit['pending_items'])
    write_json(output / 'confirmation_pending_items.json', [
        {'dataset': name, 'reason': 'Awaiting Input/GT review and original final-test inventory verification'}
        for name in candidate['datasets']])
    return candidate


def validate_roi(roi, size):
    if roi is None:
        return
    if not isinstance(roi, list) or len(roi) != 4 or any(type(v) is not int for v in roi):
        raise ValueError('ROI must be null or integer [x,y,width,height]')
    x, y, w, h = roi
    if min(x,y) < 0 or min(w,h) <= 0 or x+w > size[0] or y+h > size[1]:
        raise ValueError(f'ROI outside original image: {roi} / {size}')


def freeze(mapping, output, confirmation_path):
    if (output / 'shared_cases.json').exists():
        raise ValueError('Frozen shared_cases.json exists: no overwriting')
    candidate = read_json(output / 'candidates.json')
    confirm = read_json(confirmation_path)
    if confirm.get('reviewed_input_gt') is not True or not confirm.get('reviewer', '').strip():
        raise ValueError('A named reviewer must confirm Input AND GT visual inspection')
    if confirm.get('test_split_verified') is not True:
        raise ValueError('Confirm test_split_verified only after checking the original final-test inventory')
    current = preflight(mapping)
    manifest = {'schema_version': 1, 'selection_seed': 42, 'status': 'frozen',
                'reviewer': confirm['reviewer'], 'candidate_manifest_sha256': digest(canonical(candidate)),
                'coordinate_system': 'original decoded pixels; [x,y,width,height]; no inference crop',
                'root_mapping_keys': list(DATASETS), 'datasets': {}, 'pending_items': list(candidate['pending_items'])}
    exports = []
    if set(confirm['datasets']) - set(candidate['datasets']):
        raise ValueError('Confirmation references a dataset without audited candidates')
    for name, data in candidate['datasets'].items():
        decisions = confirm['datasets'].get(name, [])
        if not decisions:
            manifest['pending_items'].append({'dataset': name, 'reason': 'Awaiting visual confirmation'})
            continue
        if name not in current['datasets'] or current['datasets'][name]['split_source'] != data['split_source']:
            raise ValueError(f'{name}: split, evidence, or test content changed since preview')
        if len(decisions) != 6:
            raise ValueError(f'{name}: exactly six distinct cases required; otherwise leave dataset pending')
        numbers = [d['candidate_number'] for d in decisions]
        if len(set(numbers)) != len(numbers):
            raise ValueError(f'{name}: repeated candidate numbers')
        records = []
        for i, decision in enumerate(decisions, 1):
            matches = [r for r in data['cases'] if r['candidate_number'] == decision['candidate_number']]
            if len(matches) != 1 or not decision.get('scene_note', '').strip() or not decision.get('coverage_note', '').strip():
                raise ValueError('Each selected number needs an Input/GT scene_note and coverage_note (including absent categories)')
            row = matches[0]
            original = next((r for r in current['datasets'][name]['cases']
                             if r['original_stem'] == row['original_stem']), None)
            if original is None or any(row.get(k) != v for k, v in original.items()):
                raise ValueError(f'{name}: selected candidate no longer matches audited original')
            roi = decision.get('roi', row['suggested_roi'] if decision.get('accept_suggested_roi') else None)
            validate_roi(roi, row['original_size'])
            case_id = f'{name}_{i:02d}'
            record = {k:v for k,v in row.items() if k not in ('suggested_roi', 'heuristic_scores')}
            record.update({'case_id': case_id, 'roi': roi, 'display_range': [0,0,*row['original_size']],
                           'scene_note': decision['scene_note'], 'coverage_note': decision['coverage_note'],
                           'rgb_png': f'assets/{name}/{case_id}_input.png',
                           'gt_png': f'assets/{name}/{case_id}_gt.png'})
            records.append(record)
            exports.append((name, record))
        manifest['datasets'][name] = {'split_source': data['split_source'], 'cases': records}
    if not exports:
        raise ValueError('No visually confirmed cases')
    if manifest['pending_items']:
        manifest['status'] = 'frozen_partial'
    manifest['manifest_sha256'] = digest(canonical(manifest))
    # Validate everything above before exporting. Assets and lists use exclusive creation.
    for name, record in exports:
        spec = mapping['datasets'][name]
        root = Path(spec['root']).resolve()
        rgb, gt, _ = decode(relative(root, record['image_relative_path']), relative(root, record['mask_relative_path']), spec['foreground'])
        for field, array in (('rgb_png', rgb), ('gt_png', gt)):
            destination = output / record[field]
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as handle:
                Image.fromarray(array).save(handle, format='PNG')
        picture = panel(rgb, gt, record['case_id'] + ': ' + record['original_stem'], record['roi'])
        with (output / 'previews' / name / (record['case_id'] + '_frozen.png')).open('xb') as handle:
            picture.save(handle, format='PNG')
    for name in DATASETS:
        destination = output / 'ids' / f'{name}.txt'
        destination.parent.mkdir(exist_ok=True)
        with destination.open('x', encoding='utf-8') as handle:
            for row in manifest['datasets'].get(name, {}).get('cases', []):
                handle.write(row['original_stem'] + '\n')
    write_json(output / 'shared_cases.json', manifest)
    write_json(output / 'frozen_pending_items.json', manifest['pending_items'])
    return manifest


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=('preflight', 'preview', 'freeze'))
    parser.add_argument('--roots', type=Path, required=True, help='JSON dataset root/split mapping')
    parser.add_argument('--output', type=Path, required=True, help='New shared directory, not a run directory')
    parser.add_argument('--confirmation', type=Path)
    parser.add_argument('--pool-size', type=int, default=18)
    args = parser.parse_args()
    if (args.output / 'shared_cases.json').exists():
        parser.error('Frozen shared_cases.json exists: no overwriting/reselection; use a NEW version directory')
    mapping = read_json(args.roots)
    try:
        if args.command == 'preflight':
            result = preflight(mapping)
            write_json(args.output / 'preflight_only.json', result)
        elif args.command == 'preview':
            if args.pool_size < 6:
                raise ValueError('pool-size must be at least 6')
            if (args.output / 'candidates.json').exists():
                raise ValueError('Candidates already exist; do not silently reselect')
            result = preview(mapping, args.output, args.pool_size)
        else:
            if args.confirmation is None:
                raise ValueError('freeze requires --confirmation')
            result = freeze(mapping, args.output, args.confirmation)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f'ERROR: {error}\n')
    print(json.dumps({'status': result.get('status', 'preflight'),
                      'available_datasets': list(result['datasets']), 'pending_items': result['pending_items']}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
