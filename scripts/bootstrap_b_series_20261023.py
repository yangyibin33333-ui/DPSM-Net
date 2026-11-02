"""One-time additive bootstrap. Never edits historical assets or external Slim."""
from pathlib import Path
import ast,hashlib,json,inspect,sys,subprocess,os
from copy import deepcopy
import yaml
import torch

ROOT=Path(__file__).resolve().parents[1]
EXT=Path(os.environ.get('B_SERIES_EXTERNAL_ROOT', 'external/slim_study'))
TAG='b_series_20261023';DOC=ROOT/'docs'/TAG;CFG=ROOT/'configs/experiments'/TAG
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(rel,text):
    p=ROOT/rel
    if p.exists():raise FileExistsError(p)
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding='utf-8',newline='\n')
def extract(path,names):
    s=Path(path).read_text(encoding='utf-8-sig');lines=s.splitlines();out=[]
    for n in ast.parse(s).body:
        if getattr(n,'name',None) in names:
            first=min([n.lineno]+[d.lineno for d in getattr(n,'decorator_list',[])])
            out.append('\n'.join(lines[first-1:n.end_lineno]))
    assert len(out)==len(names)
    return '\n\n'.join(out)+'\n'
def main():
    assert (ROOT/'train.py').is_file()
    assert not DOC.exists() and not CFG.exists() and not (ROOT/'models/b_series').exists()
    sys.path.insert(0,str(ROOT))
    from utils.auto_protocol import resolve_auto_protocol
    formal=ROOT/'configs/experiments/auto_protocol/C40_DPSM80_AUTO_PROTOCOL_V2.yaml'
    raw=yaml.safe_load(formal.read_text());resolved=deepcopy(raw);resolved['data']['data_root']=str(ROOT/'DeepCrack')
    plan=resolve_auto_protocol(resolved,cli_img_size=False,cli_batch_size=False,cli_epochs=False,cli_data_root=True,cli_save_dir=False)
    assert resolved['loss']['mask_pos_weight']==1.0
    c=resolved['train'];d=resolved['data'];ap=resolved['auto_protocol_resolved']
    ap['validation_interval_updates']=plan.validation_interval_updates
    # Read the formal resolver output, actual GradScaler signature and optimizer signature.
    scaler={n:p.default for n,p in inspect.signature(torch.cuda.amp.GradScaler).parameters.items() if n!='enabled'}
    optimizer_sig=inspect.signature(torch.optim.AdamW).parameters
    common={'study':TAG,'models':['B0','B1','B2','B2-Conv'],'seed':42,
      'formal_reference':str(formal.relative_to(ROOT)),'formal_reference_sha256':sha(formal),
      'data':{'root':'DeepCrack','manifest':f'configs/experiments/{TAG}/manifest.json','input_hw':d['img_size'],'counts':{'train':ap['train_samples'],'val':53,'test':106},'photometric_augment':d['photometric_augment'],'mask_threshold':d['threshold'],'normalization':'ImageNet','letterbox':True,'allow_quarter_turns':d['allow_quarter_turns']},
      'training':{'lr':c['lr'],'weight_decay':c['weight_decay'],'betas':list(optimizer_sig['betas'].default),'eps':optimizer_sig['eps'].default,'fused':c['fused_adamw'],'exclude_norm_bias_from_weight_decay':c['exclude_norm_bias_from_weight_decay'],
       'max_epochs':c['epochs'],'effective_batch':ap['effective_batch_size'],'micro_batch':ap['effective_batch_size'],'accumulation_steps':1,'steps_per_epoch':ap['optimizer_steps_per_epoch'],'target_updates':ap['target_updates'],'amp':c['amp'],'amp_dtype':c['amp_dtype'],'grad_scaler':scaler,'ema':c['model_ema_decay'],'scheduler':c['lr_scheduler'],'eta_min':c['min_lr'],'validation_interval_updates':ap['validation_interval_updates'],
       'early_stopping':{'min_update':ap['min_early_stop_update'],'patience_updates':ap['patience_updates'],'min_delta':raw['auto_protocol']['early_stopping']['min_delta']},'num_workers':c['num_workers'],'persistent_workers':c['persistent_workers'],'prefetch_factor':c['prefetch_factor'],'max_memory_fraction':raw['auto_protocol']['batch']['max_memory_fraction'],'cudnn_benchmark':c['cudnn_benchmark']},
      'loss':{'name':'original BCEDiceLoss','bce_coefficient':1,'dice_coefficient':1,'pos_weight':resolved['loss']['mask_pos_weight'],'source':'losses/loss.py::DiceLoss,BCEDiceLoss'},
      'evaluation':{'weights':{'F1':raw['evaluation']['structural_score_weights']['f1'],'clDice':raw['evaluation']['structural_score_weights']['cldice'],'BoundaryF1':raw['evaluation']['structural_score_weights']['boundary_f1']},'test_enabled':False,'threshold':.5,'comparison':'>','thin_radius_px':2.,'thin_min_points':20,'thin_min_missed':10,'thin_trigger_fraction':.15,'far_gt_distance_px':3.,'far_trigger_pixels':25,'far_trigger_prediction_fraction':.1},
      'execution':{'models':['B0','B1','B2','B2-Conv'],'report_every_epochs':1,'cpu_threads':4,'output_root':f'runs/pilot/{TAG}','all_models_same_micro_batch':True},'resolved_formal_auto_protocol':ap}
    tracked=[ROOT/'train.py',formal,ROOT/'models/blocks.py',ROOT/'models/edge_direction.py',ROOT/'models/de_mambanet.py',ROOT/'losses/loss.py',ROOT/'datasets/crack_dataset.py',ROOT/'metrics/metrics.py',ROOT/'utils/auto_protocol.py',ROOT/'utils/model_ema.py',ROOT/'utils/seed.py']
    external=[EXT/'slim/models.py',EXT/'slim/reference_blocks.py',EXT/'configs/deepcrack_seed42.json',EXT/'server_train.py']
    provenance={'workspace':str(ROOT),'historical_sources':{str(p.relative_to(ROOT)):sha(p) for p in tracked},'external_read_only_sources':{str(p):sha(p) for p in external},'recovery':'Full externally stored Slim classes and forward recovered; never inferred from DEMambaNet switches','git':None}
    result=subprocess.run(['git','-C',str(ROOT),'rev-parse','HEAD'],capture_output=True,text=True)
    provenance['git']=result.stdout.strip() if result.returncode==0 else None
    provenance['git_issue']=result.stderr.strip() if result.returncode else None
    put(f'docs/{TAG}/source_provenance.json',json.dumps(provenance,ensure_ascii=False,indent=2))
    put(f'configs/experiments/{TAG}/common.yaml',yaml.safe_dump(common,allow_unicode=True,sort_keys=False))
    for name in common['models']:put(f'configs/experiments/{TAG}/{name.replace("-","_")}.yaml',yaml.safe_dump({'model':name,'common':'common.yaml','seed':42},sort_keys=False))
    put(f'docs/{TAG}/resolved_original_protocol.json',json.dumps(resolved,ensure_ascii=False,indent=2))
    header='from __future__ import annotations\nimport torch\nfrom torch import nn\nfrom torch.nn import functional as F\n'
    put('models/b_series/reference_blocks.py',header+extract(ROOT/'models/edge_direction.py',['_to_gray','sobel_edge'])+extract(ROOT/'models/blocks.py',['ConvBNAct','RIFTStructuralDirectionalModelingBlock','DetailPreservingHalfStem','SubpixelDetailReconstruction']))
    network=(EXT/'slim/models.py').read_text();network=network.replace('from .unet_reference import UNetBasic\n','')
    network=network[:network.index('\ndef build_model(name):')]
    network=network.replace("('B0','B1','B2')","('B0','B1','B2','B2-Conv')")
    network=network.replace("if version=='B2': self.rsdm=RIFTStructuralDirectionalModelingBlock(40)","if version=='B2': self.rsdm=RIFTStructuralDirectionalModelingBlock(40)\n        if version=='B2-Conv': self.rsdm=PlainConvResidual(40,23)")
    network=network.replace("if self.version=='B2': early=self.rsdm(early)","if self.version in ('B2','B2-Conv'): early=self.rsdm(early)")
    control='''
class PlainConvResidual(nn.Module):
    """Budget control: only 1x1, ordinary 3x3, 1x1; no directional gate."""
    def __init__(self,channels=40,hidden=23):
        super().__init__()
        if channels!=40 or hidden!=23:raise ValueError('This frozen control was verified for C40/hidden23 only')
        self.pre_norm=nn.GroupNorm(8,channels)
        self.reduce=nn.Conv2d(channels,hidden,1,bias=False)
        self.spatial=nn.Conv2d(hidden,hidden,3,padding=1,bias=False)
        self.mix_norm=nn.GroupNorm(1,hidden)
        self.act=nn.GELU()
        self.expand=nn.Conv2d(hidden,channels,1,bias=False)
    def forward(self,x):
        z=self.spatial(self.reduce(self.pre_norm(x)))
        return x+self.expand(self.act(self.mix_norm(z)))

def build_model(name):return SlimCNN(name)
'''
    put('models/b_series/network.py',network+control)
    put('models/b_series/__init__.py','from .network import SlimCNN,PlainConvResidual,build_model\n')
    put('scripts/b_series_runtime/__init__.py','')
    put('scripts/b_series_runtime/loss.py',header+extract(ROOT/'losses/loss.py',['DiceLoss','BCEDiceLoss']))
    put('scripts/b_series_runtime/ema.py',(ROOT/'utils/model_ema.py').read_text())
    put('scripts/b_series_runtime/metrics.py',(ROOT/'metrics/metrics.py').read_text())
    prep='from __future__ import annotations\nimport random\nimport numpy as np\nimport torch\nfrom torch.nn import functional as F\nfrom PIL import Image\n'
    put('scripts/b_series_runtime/preprocessing.py',prep+extract(ROOT/'datasets/crack_dataset.py',['_letterbox_geometry','_letterbox_image','_letterbox_target','_photometric_augment']))
    # Pair-only geometry parity is checked against the original five-target transform.
    put('scripts/b_series_runtime/data.py',(EXT/'slim/data.py').read_text())
    runtime=(EXT/'slim/runtime.py').read_text()
    runtime=runtime.replace("lr=c['lr'],weight_decay=c['weight_decay'],betas=(.9,.999),eps=1e-8,fused=True","lr=c['lr'],weight_decay=c['weight_decay'],betas=tuple(c['betas']),eps=c['eps'],fused=c['fused']")
    runtime=runtime.replace("T_max=c['target_updates'],eta_min=0.","T_max=c['target_updates'],eta_min=c['eta_min']")
    runtime=runtime.replace("init_scale=65536.,growth_factor=2.,backoff_factor=.5,growth_interval=2000","**c['grad_scaler']")
    runtime=runtime.replace("decay=.999","decay=c['ema']")
    put('scripts/b_series_runtime/runtime.py',runtime)
    put('scripts/b_series_runtime/selection.py','from __future__ import annotations\nfrom dataclasses import dataclass\nfrom typing import Any\nfrom pathlib import Path\n'+extract(ROOT/'utils/auto_protocol.py',['UpdateEarlyStopping','is_new_best_checkpoint','AutoProtocolCheckpointState','select_auto_protocol_final_checkpoint'])+extract(ROOT/'train.py',['structural_score']))
    put('scripts/b_series_runtime/checkpointing.py',(EXT/'slim/checkpointing.py').read_text())
    helpers=extract(EXT/'server_train.py',['append_row','save_atomic','write_json','run_epochs'])
    helpers=helpers.replace("cfg['server_execution']['report_every_epochs']","cfg['execution']['report_every_epochs']")
    put('scripts/b_series_runtime/engine.py','import os,json,time\nfrom pathlib import Path\nimport torch\nfrom .checkpointing import checkpoint_payload\n'+helpers)
    put(f'docs/{TAG}/external_network_snapshot.py',(EXT/'slim/models.py').read_text())
    put(f'docs/{TAG}/original_geometry_oracle.py','import random\nimport numpy as np\n'+extract(ROOT/'datasets/crack_dataset.py',['_swap_diagonals','_rotate_route_bins','_augment']))
    print('Additive B-series bootstrap created inside '+str(ROOT))
if __name__=='__main__':main()
