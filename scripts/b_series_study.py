"""B0/B1/B2/adapted RIFT-SDM budget control. Train/Val only."""
from pathlib import Path
import argparse,ast,csv,gc,hashlib,importlib.util,json,os,platform,subprocess,sys,time
from collections import defaultdict
import numpy as np
import torch
import yaml
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'models'),str(ROOT/'scripts'),str(ROOT)]
from b_series import build_model,PlainConvResidual
from b_series_runtime.runtime import StepRuntime,set_seed
from b_series_runtime.data import verify_train_val,build_loader,augment_pair
from b_series_runtime.checkpointing import ValidationController
from b_series_runtime.engine import run_epochs,write_json
from b_series_runtime.metrics import SegmentationMetric
from torch.utils.flop_counter import FlopCounterMode
TAG='b_series_20261023';CFG=ROOT/'configs/experiments'/TAG;DOC=ROOT/'docs'/TAG
NAMES=['B0','B1','B2','B2-Conv']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def loadcfg():return yaml.safe_load((CFG/'common.yaml').read_text(encoding='utf-8'))
def env():return {'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'gpu_total_bytes':torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,'platform':platform.platform(),'executable':sys.executable}
def configure(c):
    torch.set_num_threads(c['execution']['cpu_threads'])
    set_seed(c['seed']);torch.backends.cudnn.benchmark=c['training']['cudnn_benchmark']
def clean():
    gc.collect()
    if torch.cuda.is_available():torch.cuda.synchronize();torch.cuda.empty_cache()
def csvout(path,rows):
    if not rows:return
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def data_audit(root,freeze=False):
    root=Path(root);manifest={'dataset':'DeepCrack','source_root':str(root),'splits':{},'test_inference':False};pixels={}
    for split in ['train','val']:
        ims={p.stem:p for p in (root/split/'images').iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg','.bmp')}
        masks={p.stem:p for p in (root/split/'masks').iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg','.bmp')}
        assert set(ims)==set(masks),'Image/mask stem mismatch'
        samples=[];pixels[split]=set()
        for stem,p in sorted(ims.items()):
            with Image.open(p) as im:
                rgb=im.convert('RGB');key=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
            with Image.open(masks[stem]) as im: assert np.isin(np.unique(np.asarray(im)),[0,1,255]).all()
            samples.append({'id':stem,'image':p.relative_to(root).as_posix(),'mask':masks[stem].relative_to(root).as_posix(),'image_sha256':sha(p),'mask_sha256':sha(masks[stem]),'rgb_identity':key})
            pixels[split].add(key)
        manifest['splits'][split]={'count':len(samples),'samples':samples}
    assert manifest['splits']['train']['count']==368 and manifest['splits']['val']['count']==53
    assert not(pixels['train']&pixels['val']),'Train/Val duplicate image leakage'
    assert not({s['image_sha256'] for s in manifest['splits']['train']['samples']}&{s['image_sha256'] for s in manifest['splits']['val']['samples']})
    # Test counts are directory metadata only; never decode or hash Test images/masks.
    manifest['splits']['test']={'count':sum(p.suffix.lower() in ('.png','.jpg','.jpeg','.bmp') for p in (root/'test/images').iterdir()),'contents_opened':False}
    assert manifest['splits']['test']['count']==106
    manifest['train_val_disjoint_file_and_rgb']=True
    if freeze:
        assert not (CFG/'manifest.json').exists()
        write_json(CFG/'manifest.json',manifest)
    else:
        frozen=json.loads((CFG/'manifest.json').read_text(encoding='utf-8'))
        for split in ['train','val']:assert frozen['splits'][split]['samples']==manifest['splits'][split]['samples'],'Frozen dataset differs'
    return manifest
def freeze_lock():
    paths=[*list((ROOT/'models/b_series').glob('*.py')),*list((ROOT/'scripts/b_series_runtime').glob('*.py')),ROOT/'scripts/b_series_study.py',ROOT/'tests/test_b_series.py',*list(CFG.glob('*.yaml')),CFG/'manifest.json']
    provenance=json.loads((DOC/'source_provenance.json').read_text())
    paths += [ROOT/p for p in provenance['historical_sources']]
    assert not (CFG/'protocol_lock.json').exists()
    write_json(CFG/'protocol_lock.json',{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)})
def verify_lock():
    for name,h in json.loads((CFG/'protocol_lock.json').read_text()).items():assert sha(ROOT/name)==h,('Source/config lock differs',name)
def identity(c):return {'lock_sha256':sha(CFG/'protocol_lock.json'),'config_sha256':sha(CFG/'common.yaml'),'manifest_sha256':sha(CFG/'manifest.json'),'environment':env()}

def parity_and_structure(c):
    text=(DOC/'external_network_snapshot.py').read_text()
    text=text.replace('from .reference_blocks import','from b_series.reference_blocks import').replace('from .unet_reference import UNetBasic','')
    scope={};exec(compile(text,'verified_external_network_snapshot','exec'),scope)
    common={};decoder=None
    for name in NAMES:
        set_seed(42);m=build_model(name).eval()
        if name!='B2-Conv':
            set_seed(42);old=scope['SlimCNN'](name).eval()
            assert all(torch.equal(v,old.state_dict()[k]) for k,v in m.state_dict().items())
            x=torch.randn(1,3,32,64)
            with torch.no_grad():assert torch.equal(m(x),old(x)),name
            del old
        dec={k:v for k,v in m.state_dict().items() if k.startswith(('decode','mask.'))}
        if decoder is None:decoder=dec
        else:assert all(torch.equal(v,decoder[k]) for k,v in dec.items())
        forbidden=[type(mod).__name__ for mod in m.modules() if any(x in type(mod).__name__.lower() for x in ['mamba','transformer','toporoute','lsscf','auxhead'])]
        assert not forbidden
        common[name]={'params':sum(p.numel() for p in m.parameters()),'decoder_head_identical':True,'structure':str(m),'forbidden_modules':forbidden}
    # Verify geometry and Python RNG against the actual historical augmentation.
    path=DOC/'original_geometry_oracle.py';spec=importlib.util.spec_from_file_location('b_geometry_oracle',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    import random
    im=np.arange(11*17*3,dtype=np.uint8).reshape(11,17,3);mask=np.ones((11,17),dtype=np.float32)
    for seed in range(100):
        random.seed(seed);a,b=augment_pair(im,mask);state=random.getstate()
        random.seed(seed);oa,ob,*_=module._augment(im,mask,mask,np.zeros((4,11,17)),np.zeros((1,11,17)),False)
        assert np.array_equal(a,oa) and np.array_equal(b,ob) and state==random.getstate()
    write_json(DOC/'model_structure.json',common)
    return {'external_forward_and_state_exact':True,'four_decoder_heads_identical':True,'geometry_rng_parity_seeds':100,'no_hidden_modules':True}

def count_flops(c):
    rows=[];adapter=[]
    for name in NAMES:
        set_seed(42);m=build_model(name).eval().cuda()
        for batch in [1,8]:
            x=torch.zeros(batch,3,*c['data']['input_hw'],device='cuda')
            with torch.no_grad(),FlopCounterMode(display=False) as count:y=m(x)
            assert list(y.shape)==[batch,1,*c['data']['input_hw']]
            flops=count.get_total_flops()
            rows.append({'model':name,'batch':batch,'params':sum(p.numel() for p in m.parameters()),'conv_macs':flops//2,'conv_flops':flops,'range':'registered convolution FLOPs; non-convolution ops omitted'})
            del x,y
        if name in ['B2','B2-Conv']:
            seen=[]
            handle=m.rsdm.register_forward_hook(lambda mod,args,out:seen.append({'input':list(args[0].shape),'output':list(out.shape)}))
            x=torch.zeros(1,3,384,544,device='cuda')
            with torch.no_grad():y=m(x)
            handle.remove();assert seen==[{'input':[1,40,192,272],'output':[1,40,192,272]}]
            z=torch.zeros(1,40,192,272,device='cuda')
            with torch.no_grad(),FlopCounterMode(display=False) as count:q=m.rsdm(z)
            adapter.append({'model':name,'params':sum(p.numel() for p in m.rsdm.parameters()),'conv_weights':sum(mod.weight.numel() for mod in m.rsdm.modules() if isinstance(mod,torch.nn.Conv2d)),'conv_macs':count.get_total_flops()//2,'conv_flops':count.get_total_flops(),'placement':'stage1 -> rsdm -> both reconstruct and down1','calls':seen})
            if name=='B2-Conv':assert all(mod.groups==1 and mod.dilation==(1,1) and mod.kernel_size in [(1,1),(3,3)] for mod in m.rsdm.modules() if isinstance(mod,torch.nn.Conv2d))
            del x,y,z,q
        del m;clean()
    for name in NAMES:
        a,b=[r for r in rows if r['model']==name];assert b['conv_flops']==8*a['conv_flops']
    rsdm,control=adapter
    assert rsdm['conv_flops']==685178880 and rsdm['conv_weights']==6560
    assert control['conv_weights']==6601
    gap=abs(control['conv_macs']/rsdm['conv_macs']-1);assert gap<=.05
    csvout(DOC/'complexity.csv',rows);write_json(DOC/'adapter_budget.json',{'modules':adapter,'conv_macs_relative_gap':gap})
    return {'batch8_linear':True,'adapter_macs_gap':gap,'historical_rsdm_delta_matches':True,'rows':rows}

def gpu_step_preflight(c):
    report=[]
    for name in NAMES:
        clean();torch.cuda.reset_peak_memory_stats();set_seed(42)
        model=build_model(name).cuda();runtime=StepRuntime(model,c)
        x=torch.randn(8,3,*c['data']['input_hw'],device='cuda');y=(torch.rand(8,1,*c['data']['input_hw'],device='cuda')>.965).float()
        initial=model.mask.weight.detach().clone();r=runtime.step(x,y);torch.cuda.synchronize()
        assert r['successful'] and r['finite_gradients'] and not torch.equal(initial,model.mask.weight)
        assert runtime.scheduler.last_epoch==1 and runtime.global_update==1
        peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
        assert peak<=torch.cuda.get_device_properties(0).total_memory*c['training']['max_memory_fraction'],(name,'GPU memory budget exceeded')
        report.append({'model':name,**r,'peak_allocated_bytes':peak,'peak_reserved_bytes':reserved,'capacity_test':'allocated <= formal max_memory_fraction * total','batch':8})
        del model,runtime,x,y,initial;clean()
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=['prepare','preflight','train','all'],required=True)
    parser.add_argument('--output-root',type=Path,help='New training output root inside this project; existing runs are never overwritten')
    parser.add_argument('--data-root',type=Path,default=ROOT/'DeepCrack');parser.add_argument('--model',choices=NAMES);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();c=loadcfg();configure(c);args.data_root=args.data_root.resolve()
    if args.mode=='prepare':data_audit(args.data_root,True);freeze_lock();print('Protocol and dataset manifest frozen');return
    verify_lock();manifest=data_audit(args.data_root)
    if args.mode=='preflight':
        report={'status':'running','identity':identity(c),'dataset_disjoint':True,'test_inference':False,'formal_training':False}
        try:
            assert torch.cuda.is_available(),'No CUDA GPU'
            report['structure_tests']=parity_and_structure(c);report['complexity']=count_flops(c);report['gpu']=gpu_step_preflight(c)
            result=subprocess.run([sys.executable,'-B',str(ROOT/'tests/test_b_series.py')],cwd=ROOT,capture_output=True,text=True)
            report['tests']={'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr};assert result.returncode==0,result.stderr
            verify_lock();report['status']='passed'
        except BaseException as e:
            report.update(status='failed',error=repr(e));write_json(DOC/'preflight.json',report);raise
        write_json(DOC/'preflight.json',report);print(json.dumps({'status':'passed','gpu':report['gpu']}));return
    if not args.execute:parser.error('Formal training requires --execute')
    pre=json.loads((DOC/'preflight.json').read_text());assert pre['status']=='passed' and pre['identity']==identity(c),'Matching preflight required'
    selected=NAMES if args.mode=='all' else [args.model]
    if None in selected:parser.error('--model required for train')
    # Refuse all selected existing outputs before starting the first model.
    output_root=ROOT/c['execution']['output_root']
    if args.output_root is not None:
        output_root=(ROOT/args.output_root).resolve()
        assert output_root.is_relative_to(ROOT.resolve()) and output_root!=ROOT.resolve(),'Output root must be inside the project'
        c=dict(c);c['execution']=dict(c['execution'],output_root=output_root.relative_to(ROOT.resolve()).as_posix())
    assert all(not(output_root/name/'seed42').exists() for name in selected),'Existing experiment outputs; refusing overwrite'
    for name in selected:
        verify_lock();run_one(name,c,args.data_root,manifest,output_root)

def run_one(name,c,data_root,manifest,output_root):
    clean();set_seed(c['seed']);torch.cuda.reset_peak_memory_stats()
    out=output_root/name/'seed42';out.mkdir(parents=True,exist_ok=False)
    model=build_model(name).cuda();runtime=StepRuntime(model,c);controller=ValidationController(c)
    generator=torch.Generator().manual_seed(c['seed'])
    train=build_loader(data_root,'train',manifest,c,generator);val=build_loader(data_root,'val',manifest,c)
    assert len(train)==c['training']['steps_per_epoch']
    provenance=json.loads((DOC/'source_provenance.json').read_text())
    write_json(out/'source_identity.json',{'identity':identity(c),'git':provenance['git'],'git_issue':provenance['git_issue'],'source_hashes':json.loads((CFG/'protocol_lock.json').read_text())})
    resolved=dict(c);resolved['data']=dict(c['data'],root=str(data_root));write_json(out/'config.json',resolved);write_json(out/'environment.json',env())
    print(json.dumps({'status':'training_started','model':name,'output':str(out)}),flush=True)
    try:
        summary=run_epochs(runtime,controller,train,val,resolved,out,name,generator)
        summary.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write_json(out/'run_summary.json',summary)
        # All reported metrics must come from the same significant-best EMA checkpoint.
        del train,val,runtime,model;clean()
        best=evaluate_best(name,c,data_root,manifest,out)
        update_results(name,c,summary,best)
    except BaseException as error:
        write_json(out/'failure.json',{'status':'failed','error':repr(error),'test_inference':False});raise

@torch.no_grad()
def evaluate_best(name,c,data_root,manifest,out):
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize
    path=out/'best_structural_score.pth';ck=torch.load(path,map_location='cpu',weights_only=False)
    model=build_model(name).cuda().eval();model.load_state_dict(ck['model'],strict=True)
    loader=build_loader(data_root,'val',manifest,c);meter=SegmentationMetric();records=[]
    for batch in loader:
        target=batch['mask'].cuda();logits=model(batch['image'].cuda());meter.update(logits,target)
        pred=logits.sigmoid().cpu().numpy()[:,0]>.5;gt=target.cpu().numpy()[:,0]>.5
        for stem,p,y in zip(batch['id'],pred,gt):
            thin=skeletonize(y)&(ndi.distance_transform_edt(y)<=c['evaluation']['thin_radius_px']);miss=thin&~p
            far=p&~y&(ndi.distance_transform_edt(~y)>c['evaluation']['far_gt_distance_px'])
            nt,nm,nf,npred=int(thin.sum()),int(miss.sum()),int(far.sum()),int(p.sum())
            records.append({'image':stem,'thin_points':nt,'thin_missed':nm,'thin_trigger':nt>=20 and nm>=10 and nm/max(nt,1)>=.15,'far_fp_pixels':nf,'far_fp_trigger':nf>=25 and nf/max(npred,1)>=.1})
    result={'checkpoint':str(path.relative_to(ROOT)),'checkpoint_sha256':sha(path),'epoch':ck['epoch'],'metrics':meter.compute(),
        'thin_skeleton_pixel_weighted_miss_rate':sum(r['thin_missed'] for r in records)/max(sum(r['thin_points'] for r in records),1),
        'thin_trigger_images':sum(r['thin_trigger'] for r in records),'far_gt_fp_pixels':sum(r['far_fp_pixels'] for r in records),'far_gt_fp_trigger_images':sum(r['far_fp_trigger'] for r in records),'test_inference':False,'validation_images':len(records),'latency_ms':None,'fps':None,'latency_note':'RTX4090D condition not met'}
    if '4090D' in torch.cuda.get_device_name().replace(' ','').upper():
        x=torch.randn(1,3,*c['data']['input_hw'],device='cuda');timings=[]
        with torch.autocast('cuda',dtype=torch.float16):
            for _ in range(100):model(x)
            for _ in range(3):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
                for _ in range(300):model(x)
                end.record();torch.cuda.synchronize();timings.append(start.elapsed_time(end)/300)
        result.update(latency_ms=float(np.median(timings)),fps=1000/float(np.median(timings)),latency_note='4090D, B1 FP16 autocast, 100 warmup, 300x3, CUDA events, model-only')
    csvout(out/'best_validation_per_image.csv',records);write_json(out/'best_validation.json',result)
    del loader,model,ck;clean();return result

def update_results(name,c,summary,best):
    path=DOC/'b_series_results.csv';rows=list(csv.DictReader(path.open(encoding='utf-8-sig'))) if path.exists() else []
    complexity=list(csv.DictReader((DOC/'complexity.csv').open(encoding='utf-8-sig')));comp=next(r for r in complexity if r['model']==name and r['batch']=='1')
    row={'model':name,'status':'completed','seed':42,'params':comp['params'],'conv_macs':comp['conv_macs'],'conv_flops':comp['conv_flops'],'best_epoch':summary['best_epoch'],'actual_updates':summary['actual_updates'],**best['metrics'],
        'thin_miss_rate':best['thin_skeleton_pixel_weighted_miss_rate'],'thin_trigger_images':best['thin_trigger_images'],'far_gt_fp_pixels':best['far_gt_fp_pixels'],'far_gt_fp_trigger_images':best['far_gt_fp_trigger_images'],'latency_ms':best['latency_ms'],'fps':best['fps'],'checkpoint_sha256':best['checkpoint_sha256']}
    rows=[r for r in rows if r['model']!=name];rows.append(row);csvout(path,rows)
if __name__=='__main__':main()
