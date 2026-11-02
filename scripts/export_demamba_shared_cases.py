"""Export frozen qualitative cases only; never train or run full-test metrics."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import sys

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
import select_shared_test_cases as shared

DATASETS = shared.DATASETS
COLORS = {'TP': (0,158,115), 'FP': (213,94,0), 'FN': (0,114,178), 'TN': (242,242,242)}
PROTOCOL = {'precision': 'fp32', 'head': 'mask', 'activation': 'sigmoid', 'threshold': .5,
            'comparison': '>', 'tta': False, 'sliding_window': False, 'postprocess': False,
            'preprocessing': 'RGB/Pillow-bilinear-letterbox-zero/ImageNet',
            'evaluation_grid': 'entire model-input grid INCLUDING letterbox padding'}


def load(path):
    return shared.read_json(path)


def hashed(value):
    return shared.digest(shared.canonical(value))


def save_json(path, value):
    shared.write_json(path, value)


def code_hashes():
    files = [REPO/'train.py', REPO/'test.py', Path(__file__), REPO/'scripts/select_shared_test_cases.py']
    for directory in ('models', 'datasets', 'metrics', 'utils'):
        files.extend(sorted((REPO/directory).rglob('*.py')))
    return {p.relative_to(REPO).as_posix(): shared.file_hash(p) for p in files}


def read_manifest(path):
    manifest = load(path)
    if manifest.get('schema_version') != 1 or manifest.get('selection_seed') != 42:
        raise ValueError('Unsupported shared manifest schema/selection seed')
    if manifest.get('status') not in ('frozen', 'frozen_partial'):
        raise ValueError('Only a frozen shared_cases.json is accepted')
    body = {k:v for k,v in manifest.items() if k != 'manifest_sha256'}
    if hashed(body) != manifest.get('manifest_sha256'):
        raise ValueError('Frozen manifest content hash mismatch')
    ids = set()
    for name, dataset in manifest['datasets'].items():
        if name not in DATASETS or len(dataset['cases']) != 6:
            raise ValueError(f'{name}: require six registered cases')
        stems = set()
        for row in dataset['cases']:
            identifier = row['case_id']
            if not re.fullmatch(re.escape(name)+r'_[0-9]{2}', identifier) or identifier in ids:
                raise ValueError('Duplicate/unsafe case_id')
            if row['split'] != 'test' or row['original_stem'] in stems:
                raise ValueError('Non-test or duplicate sample')
            shared.validate_roi(row.get('roi'), row['original_size'])
            stems.add(row['original_stem']); ids.add(identifier)
    return manifest


def repo_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else shared.relative(REPO, value)


def audit_run(name, run, common):
    """File/config/log checks only; explicitly does not load torch/checkpoint tensors."""
    import yaml
    if common != PROTOCOL:
        raise ValueError('Unsupported inference protocol: do not silently change final-test settings')
    paths = {key: repo_path(run[key]) for key in ('config', 'checkpoint', 'log')}
    for p in paths.values():
        if not p.is_file():
            raise FileNotFoundError(p)
    cfg = yaml.safe_load(paths['config'].read_text(encoding='utf-8'))
    text = paths['log'].read_text(encoding='utf-8')
    selected = re.findall(r'Final checkpoint selection \| type=([^|]+)\| update=(\d+) \| score=([^|]+)\| path=(.+)', text)
    if not selected or 'Final test metrics (best structural-score checkpoint):' not in text:
        raise ValueError('No final-selection + executed final-test evidence in log')
    kind, update, score, selected_path = selected[-1]
    normalized = selected_path.strip().replace('\\', '/').removeprefix('./')
    specified = str(run['checkpoint']).replace('\\', '/').removeprefix('./')
    if normalized != specified and not normalized.endswith('/'+specified):
        raise ValueError('Configured checkpoint differs from final-selection log (no fallback)')
    resolved = cfg.get('auto_protocol_resolved', {})
    if resolved.get('final_checkpoint_type') != kind.strip() or int(resolved.get('final_checkpoint_update', -1)) != int(update):
        raise ValueError('Resolved config final-selection metadata disagrees with log')
    if str(resolved.get('final_checkpoint_path', '')).replace('\\','/').removeprefix('./') != normalized:
        raise ValueError('Resolved config selected checkpoint path mismatch')
    seed = int(run['training_seed'])
    if resolved.get('training_seed') != seed or f'Training seed reapplied after probe: {seed}' not in text:
        raise ValueError('Training seed lacks independent resolved-config/log evidence')
    if Path(cfg['data']['data_root']).name.lower() != name.lower():
        raise ValueError('Config dataset ownership mismatch')
    if cfg['data']['img_size'] != run['img_size']:
        raise ValueError('Input size differs from resolved config')
    if run['weight_type'] != 'ema' or not cfg['train'].get('use_model_ema') or 'Checkpoint model: EMA shadow weights' not in text:
        raise ValueError('This registered final run requires EMA saved in checkpoint[model]')
    if cfg['model'].get('name') != 'DEMambaNet' or not cfg['model'].get('use_direction_partitioned_shared_mamba'):
        raise ValueError('Not the registered final DPSM model')
    final_text = text.rsplit('Final test metrics (best structural-score checkpoint):', 1)[1]
    for metric, expected in run['paper_metrics'].items():
        match = re.search(r'\| INFO \| '+re.escape(metric)+r': ([0-9.]+)', final_text)
        if not match or abs(float(match.group(1))-float(expected)) > .00005:
            raise ValueError(f'Paper/run provenance metric mismatch: {name}/{metric}')
    reference = repo_path(run['paper_reference'])
    if not reference.is_file():
        raise FileNotFoundError(reference)
    return cfg, {'status': 'files_config_log_passed_NOT_model_loaded',
                 'config_sha256': shared.file_hash(paths['config']),
                 'checkpoint_sha256': shared.file_hash(paths['checkpoint']),
                 'log_sha256': shared.file_hash(paths['log']),
                 'paper_reference_sha256': shared.file_hash(reference),
                 'selected_checkpoint_type': kind.strip(), 'selected_update': int(update),
                 'selected_score': float(score), 'training_seed': seed,
                 'weight_type': 'EMA stored in checkpoint[model], never raw_model',
                 'img_size': run['img_size'], 'checkpoint': run['checkpoint'], 'config': run['config'],
                 'log': run['log'], 'test_code': 'train.py:run_final_test/validate + metrics/metrics.py',
                 'protocol': common}


def verify_cases(name, dataset, spec, cfg):
    root = Path(spec['root']).resolve()
    if spec.get('split', 'test') != 'test':
        raise ValueError('Dataset root mapping must identify actual test')
    for key in ('image_dir', 'mask_dir'):
        if any(p.lower() in ('val','train','validation') for p in Path(spec[key]).parts):
            raise ValueError('Refusing train/val masquerading as test')
    images = shared.index(shared.relative(root, spec['image_dir']), spec.get('image_stem_suffix',''))
    masks = shared.index(shared.relative(root, spec['mask_dir']), spec.get('mask_stem_suffix',''))
    if images.keys() != masks.keys():
        raise ValueError('Missing/extra image-GT pairs in provided test directory')
    if spec.get('split_source'):
        # Same strict split-list reader as selection; requires supplied evidence files.
        ids = {r['pair_id'] for r in shared.audit_dataset(name, spec)['cases']}
    else:
        if 'test' not in Path(spec['image_dir']).parts or 'test' not in Path(spec['mask_dir']).parts:
            raise ValueError('Nonstandard/flat test layout requires an authoritative split_source')
        ids = set(images)
    verified, hashes = [], set()
    for row in dataset['cases']:
        key = row.get('pair_id', row['original_stem'])
        if key not in ids or key not in images or key not in masks:
            raise ValueError(f'Not a member of supplied test split: {key}')
        rgb, gt, rule = shared.decode(images[key], masks[key], row['foreground_rule'])
        if rule['evaluation_threshold'] != cfg['data']['threshold']:
            raise ValueError('Shared GT policy differs from the actual evaluation threshold')
        if list(rgb.shape[1::-1]) != row['original_size'] or shared.digest(rgb.tobytes()) != row['rgb_pixel_sha256']:
            raise ValueError(f'RGB identity mismatch: {key}')
        if shared.digest(gt.tobytes()) != row['gt_binary_pixel_sha256']:
            raise ValueError(f'GT identity mismatch: {key}')
        if row['rgb_pixel_sha256'] in hashes:
            raise ValueError('Duplicate selected image content')
        hashes.add(row['rgb_pixel_sha256'])
        verified.append((row, rgb, gt, {
            'image_relative_path': images[key].relative_to(root).as_posix(),
            'mask_relative_path': masks[key].relative_to(root).as_posix(),
            'image_file_sha256': shared.file_hash(images[key]), 'mask_file_sha256': shared.file_hash(masks[key]),
            'source_encoding_changed_pixels_identical': {
                'image': shared.file_hash(images[key]) != row['image_file_sha256'],
                'gt': shared.file_hash(masks[key]) != row['mask_file_sha256']}}))
    return verified


def geometry(original_size, model_size):
    w,h = original_size; th,tw = model_size
    scale = min(th/h, tw/w)
    rh,rw = max(1,round(h*scale)),max(1,round(w*scale))
    return {'original_wh': [w,h], 'model_hw': [th,tw], 'resized_hw': [rh,rw],
            'top': (th-rh)//2, 'left': (tw-rw)//2, 'valid_rect_xywh': [(tw-rw)//2,(th-rh)//2,rw,rh],
            'color_order':'RGB','pixel_scale':'float32 / 255',
            'normalization_mean':[.485,.456,.406],'normalization_std':[.229,.224,.225],
            'input_resize': 'Pillow bilinear, round geometry, zero padding BEFORE ImageNet normalization',
            'display_inverse': 'crop probability to valid_rect, torch bilinear align_corners=False to original size, then >0.5',
            'metric_decision': 'sigmoid(logits)>0.5 on entire model grid, INCLUDING padding; no resize',
            'display_only': True}


def input_canvas(rgb, geom):
    th,tw=geom['model_hw']; rh,rw=geom['resized_hw']; top,left=geom['top'],geom['left']
    result=np.zeros((th,tw,3),dtype=np.uint8)
    result[top:top+rh,left:left+rw]=np.asarray(Image.fromarray(rgb).resize((rw,rh),Image.Resampling.BILINEAR))
    return result


def checked_array(value, probability=False):
    value=np.ascontiguousarray(value,dtype=np.float32)
    if not np.isfinite(value).all() or (probability and (value.min()<0 or value.max()>1)):
        raise ValueError('Non-finite/out-of-range model output')
    return value


def error_image(prediction, gt):
    pred=prediction>0; truth=gt>0
    error=np.empty((*truth.shape,3),np.uint8); error[:]=COLORS['TN']
    error[pred & truth]=COLORS['TP']; error[pred & ~truth]=COLORS['FP']; error[~pred & truth]=COLORS['FN']
    return np.ascontiguousarray(error)


def save_png(path, array):
    with path.open('xb') as handle:
        Image.fromarray(np.ascontiguousarray(array)).save(handle,format='PNG')


def save_array(path, array):
    with path.open('xb') as handle:
        np.save(handle,np.ascontiguousarray(array),allow_pickle=False)


def completed(directory, fingerprint=None):
    if not directory.is_dir():
        return False
    marker=load(directory/'complete.json')
    if fingerprint is not None and marker['fingerprint'] != fingerprint:
        raise ValueError(f'Old result provenance mismatch: {directory}; use a new output directory')
    for name,sha in marker['files'].items():
        path=shared.relative(directory,name)
        if not path.is_file() or shared.file_hash(path)!=sha:
            raise ValueError(f'Incomplete/corrupted result: {path}')
    metadata=load(directory/'case.json')
    required={'input.png','gt.png','probability.npy','prediction.png','error.png',
              'eval_logits.npy','eval_probability.npy','eval_prediction.png','eval_gt.png','eval_valid.png','case.json'}
    if metadata['case'].get('roi') is not None:
        required |= {'prediction_roi.png','error_roi.png'}
    if not required <= marker['files'].keys() or marker['fingerprint'] != hashed(metadata['identity']):
        raise ValueError('Incomplete completion marker')
    return True


def write_case(directory,row,rgb,gt,source,provenance,identity,eval_logits,eval_prob,display_prob,eval_gt,geom):
    directory.mkdir(parents=True,exist_ok=False)
    eval_logits=checked_array(eval_logits); eval_prob=checked_array(eval_prob,True)
    display_prob=checked_array(display_prob,True)
    if eval_prob.shape != tuple(geom['model_hw']) or display_prob.shape != gt.shape:
        raise ValueError('Probability geometry mismatch')
    pred=(display_prob>.5).astype(np.uint8)*255
    err=error_image(pred,gt)
    for name,value in [('input',rgb),('gt',gt),('prediction',pred),('error',err),
                       ('eval_prediction',(eval_prob>.5).astype(np.uint8)*255),('eval_gt',eval_gt)]:
        save_png(directory/(name+'.png'),value)
    valid=np.zeros(geom['model_hw'],np.uint8)
    x,y,w,h=geom['valid_rect_xywh']; valid[y:y+h,x:x+w]=255
    save_png(directory/'eval_valid.png',valid)
    for name,value in [('probability',display_prob),('eval_probability',eval_prob),('eval_logits',eval_logits)]:
        save_array(directory/(name+'.npy'),value)
    if row.get('roi') is not None:
        x,y,w,h=row['roi']
        save_png(directory/'prediction_roi.png',pred[y:y+h,x:x+w])
        save_png(directory/'error_roi.png',err[y:y+h,x:x+w])
    save_json(directory/'case.json',{'schema_version':1,'method':'DEMambaNet','case':row,
              'identity':identity,'source_verification':source,'run':provenance,'geometry':geom,
              'note':'No subset metrics calculated. Display prediction is NOT the padded evaluation decision grid.'})
    files={p.name:shared.file_hash(p) for p in directory.iterdir() if p.is_file()}
    save_json(directory/'complete.json',{'status':'complete','fingerprint':hashed(identity),'files':files})


def validate_checkpoint_metadata(checkpoint,cfg):
    resolved=cfg['auto_protocol_resolved']
    if checkpoint.get('checkpoint_type') != resolved['final_checkpoint_type']:
        raise ValueError('Checkpoint metadata type mismatch')
    state=checkpoint.get('auto_protocol_state',{})
    if state.get('significant_best_update')!=resolved['final_checkpoint_update'] or state.get('global_update')!=resolved['final_checkpoint_update']:
        raise ValueError('Checkpoint update metadata disagrees with final config/log')
    score=checkpoint.get('best_structural_score')
    if score is None or not np.isfinite(float(score)) or abs(float(score)-float(resolved['final_checkpoint_score']))>1e-8:
        raise ValueError('Checkpoint structural-score metadata mismatch')
    if state.get('resume_identity') is not None and state['resume_identity']!=resolved.get('resume_identity'):
        raise ValueError('Checkpoint dataset/protocol resume_identity mismatch')
    return {'checkpoint_metadata_verified':True,'checkpoint_epoch_zero_based':checkpoint.get('epoch'),
            'checkpoint_update':state['global_update'],'checkpoint_score':float(score),
            'checkpoint_resume_identity_verified':state.get('resume_identity') is not None}


def load_model(run,cfg,device,seed):
    import torch
    from models import build_model
    from models.mamba_blocks import SequenceMambaCore
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('Formal capture requires CUDA + real mamba_ssm; CPU is for check/render only')
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=bool(cfg['train'].get('cudnn_benchmark',False))
    model_cfg=dict(cfg['model']); name=model_cfg.pop('name')
    model=build_model(name,**model_cfg).to(device).float().eval()
    cores=[m for m in model.modules() if isinstance(m,SequenceMambaCore)]
    if len(cores)!=1 or cores[0].dim!=80 or any(m.backend!='mamba_ssm' for m in cores):
        raise ValueError('Real mamba_ssm required; fallback is forbidden for formal export')
    checkpoint=torch.load(repo_path(run['checkpoint']),map_location='cpu',weights_only=False)
    if 'model' not in checkpoint:
        raise ValueError('Registered EMA final checkpoint must contain model, not guessed state_dict/raw_model')
    checkpoint_metadata=validate_checkpoint_metadata(checkpoint,cfg)
    state=checkpoint['model']; stripped=False
    if state and all(k.startswith('module.') for k in state):
        state={k[len('module.'):]:v for k,v in state.items()}; stripped=True
    model.load_state_dict(state,strict=True)
    checksum=hashlib.sha256()
    for key,value in model.state_dict().items():
        checksum.update(key.encode()); checksum.update(str(value.dtype).encode())
        checksum.update(str(tuple(value.shape)).encode())
        checksum.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return model,{**checkpoint_metadata,'model_loaded_strictly':True,'backend':'mamba_ssm','core_count':len(cores),
                  'parameters':sum(p.numel() for p in model.parameters()),
                  'module_prefix_stripped':stripped,'torch_version':torch.__version__,
                  'device':str(device),'gpu':torch.cuda.get_device_name(device),'export_seed':seed,
                  'cudnn_benchmark':torch.backends.cudnn.benchmark,'loaded_state_checksum':checksum.hexdigest(),
                  'allow_tf32_matmul':torch.backends.cuda.matmul.allow_tf32,
                  'allow_tf32_cudnn':torch.backends.cudnn.allow_tf32,
                  'export_batch_size':1,'original_final_test_batch_size':cfg['train']['batch_size']}


def capture_dataset(directory,verified,run,cfg,provenance,manifest_sha,codes,device_name,seed):
    import torch
    import torch.nn.functional as F
    from datasets.crack_dataset import _letterbox_image,_letterbox_geometry,_letterbox_target,IMAGENET_MEAN,IMAGENET_STD
    model=None; results=[]
    for row,rgb,gt,source in verified:
        identity={'manifest_sha256':manifest_sha,'case_sha256':hashed(row),'run':provenance,
                  'code_hashes':codes,'export_seed':seed,'source_verification':source}
        dest=directory/row['case_id']
        if dest.exists():
            if completed(dest,hashed(identity)):
                results.append({'case_id':row['case_id'],'status':'reused'}); continue
            raise ValueError(f'Incomplete old result: {dest}')
        if model is None:
            model,backend=load_model(run,cfg,torch.device(device_name),seed)
            print('MODEL LOADED',json.dumps(backend),flush=True)
        geom=geometry(row['original_size'],run['img_size'])
        original_h,original_w=rgb.shape[:2]
        assert list(_letterbox_geometry(original_h,original_w,run['img_size']))==[*geom['resized_hw'],geom['top'],geom['left']]
        canvas=_letterbox_image(rgb,run['img_size'])
        assert np.array_equal(canvas,input_canvas(rgb,geom))
        tensor=torch.from_numpy(canvas.astype(np.float32)/255.).permute(2,0,1)
        tensor=((tensor-IMAGENET_MEAN)/IMAGENET_STD).float().unsqueeze(0).to(device_name)
        with torch.no_grad():  # Match train.py validate (FP32, no autocast).
            logits=model(tensor)['mask']
            if logits.shape!=(1,1,*run['img_size']):
                raise ValueError(f'Unexpected final mask logit shape: {tuple(logits.shape)}')
            prob=torch.sigmoid(logits)
            x,y,w,h=geom['valid_rect_xywh']
            display=F.interpolate(prob[:,:,y:y+h,x:x+w],size=gt.shape,mode='bilinear',align_corners=False)
        eval_gt=(_letterbox_target((gt>0).astype(np.float32)[None],run['img_size'],'nearest')[0]>.5).astype(np.uint8)*255
        write_case(dest,row,rgb,gt,source,{**provenance,**backend},identity,
                   logits[0,0].cpu().numpy(),prob[0,0].cpu().numpy(),display[0,0].cpu().numpy(),eval_gt,geom)
        results.append({'case_id':row['case_id'],'status':'captured','backend':backend})
        print('CAPTURED',row['case_id'],flush=True)
    if model is not None:
        del model
        torch.cuda.empty_cache()
    return results


def render_dataset(directory,dataset,manifest_sha,codes):
    """CPU-only reconstruction from valid complete bundles, no data/checkpoint/model needed."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    ratio=max(row['original_size'][1]/row['original_size'][0] for row in dataset['cases'])
    fig,axes=plt.subplots(6,4,figsize=(10,max(6.,6*2.1*ratio+1.)),squeeze=False)
    for i,row in enumerate(dataset['cases']):
        dest=directory/row['case_id']
        if not completed(dest):
            raise ValueError('Missing complete bundle')
        metadata=load(dest/'case.json'); identity=metadata['identity']
        if identity['manifest_sha256']!=manifest_sha or identity['case_sha256']!=hashed(row) or identity['code_hashes']!=codes:
            raise ValueError('Render source/manifest/code provenance mismatch; do not mix exports')
        for j,key in enumerate(('input','gt','prediction','error')):
            with Image.open(dest/(key+'.png')) as handle:
                array=np.array(handle)
            axes[i,j].imshow(array,cmap='gray' if array.ndim==2 else None,vmin=0,vmax=255,
                             interpolation='nearest',aspect='equal')
            axes[i,j].set_xticks([]); axes[i,j].set_yticks([])
            for spine in axes[i,j].spines.values(): spine.set_visible(False)
            if i==0: axes[i,j].set_title(('Input','GT','Prediction','Error')[j],fontsize=13)
        axes[i,0].set_ylabel(row['case_id'],fontsize=11)
    fig.legend(handles=[Patch(color=np.array(color)/255.,label=key) for key,color in COLORS.items()],
               loc='lower center',ncol=4,frameon=False,fontsize=11)
    fig.subplots_adjust(left=.12,right=.99,top=.94,bottom=.075,wspace=.03,hspace=.12)
    # Never overwrite an earlier preview; capture is reusable and render versions are additive.
    version=1
    while (directory/f'preview_v{version:03d}.png').exists() or (directory/f'preview_v{version:03d}.pdf').exists(): version+=1
    base=directory/f'preview_v{version:03d}'
    fig.savefig(base.with_suffix('.png'),dpi=300)
    fig.savefig(base.with_suffix('.pdf'),dpi=300)
    plt.close(fig)
    return str(base)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--cases',type=Path,required=True)
    parser.add_argument('--data-roots-json',type=Path)
    parser.add_argument('--runs-json',type=Path,default=REPO/'configs/qualitative/demamba_final_runs.json')
    parser.add_argument('--dataset',choices=('all',*DATASETS),default='all')
    parser.add_argument('--output',type=Path,default=REPO/'outputs/sota_visualizations/DEMambaNet')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--seed',type=int,default=42,help='Export RNG seed, NOT a training/checkpoint selection seed')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--check-only',action='store_true')
    group.add_argument('--capture-only',action='store_true')
    group.add_argument('--render-only',action='store_true')
    args=parser.parse_args()
    codes=code_hashes(); manifest=None; manifest_error=None
    try: manifest=read_manifest(args.cases)
    except (OSError,ValueError,KeyError) as error: manifest_error=str(error)
    runs=load(args.runs_json)
    roots=load(args.data_roots_json) if args.data_roots_json else {'datasets':{}}
    names=DATASETS if args.dataset=='all' else (args.dataset,)
    audit={'schema_version':1,'method':'DEMambaNet','mode':'check' if args.check_only else 'render' if args.render_only else 'capture',
           'manifest_error':manifest_error,'code_hashes':codes,'datasets':{},'no_training':True,'no_full_test':True}
    for name in names:
        try:
            directory=args.output/name
            if args.render_only:
                if manifest is None: raise ValueError(manifest_error)
                preview=render_dataset(directory,manifest['datasets'][name],manifest['manifest_sha256'],codes)
                audit['datasets'][name]={'status':'rendered','preview':preview,'model_loaded':False,
                                        'checkpoint_runtime_revalidated':False}; continue
            run=runs['datasets'][name]
            cfg,provenance=audit_run(name,run,runs['protocol'])
            if manifest is None:
                audit['datasets'][name]={'status':'missing_shared_cases','run':provenance,'reason':manifest_error}
                print(name,'RUN FILES/CONFIG/LOG PASS; missing frozen cases:',manifest_error,flush=True); continue
            if name not in manifest['datasets']:
                raise ValueError('Dataset absent from frozen partial manifest')
            verified=verify_cases(name,manifest['datasets'][name],roots['datasets'][name],cfg)
            if args.check_only:
                audit['datasets'][name]={'status':'precheck_passed_NOT_inference','run':provenance,
                                        'case_count':len(verified),'model_loaded':False}
                print(name,'PRECHECK PASS (not model loaded / not inference)',flush=True); continue
            cases=capture_dataset(directory,verified,run,cfg,provenance,manifest['manifest_sha256'],codes,args.device,args.seed)
            audit['datasets'][name]={'status':'captured','cases':cases,'run':provenance}
            if not args.capture_only:
                audit['datasets'][name]['preview']=render_dataset(directory,manifest['datasets'][name],manifest['manifest_sha256'],codes)
        except Exception as error:
            audit['datasets'][name]={'status':'missing' if isinstance(error,FileNotFoundError) else 'failed',
                                    'reason':f'{type(error).__name__}: {error}'}
        print(name,audit['datasets'][name]['status'],audit['datasets'][name].get('reason',''),flush=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    save_json(args.output/'audits'/f'{stamp}.json',audit)
    # audit.json is the latest summary, historical summaries remain in audits/.
    args.output.mkdir(parents=True,exist_ok=True)
    temporary=args.output/f'.audit_{stamp}.json'; save_json(temporary,audit)
    temporary.replace(args.output/'audit.json')
    success={'precheck_passed_NOT_inference','captured','rendered'}
    return 0 if all(v['status'] in success for v in audit['datasets'].values()) else 2


if __name__=='__main__':
    raise SystemExit(main())
