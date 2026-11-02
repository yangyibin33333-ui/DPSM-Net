import os,json,time
from pathlib import Path
import torch
from .checkpointing import checkpoint_payload
def write_json(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(temp,path)

def append_row(path,row):
    with Path(path).open('a',encoding='utf-8') as f:
        f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush()

def save_atomic(path,payload):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    torch.save(payload,temporary);os.replace(temporary,path)

def run_epochs(runtime,controller,train_loader,val_loader,cfg,out,model_name,generator=None,save=save_atomic):
    """Injectable loop permits CPU synthetic orchestration checks."""
    c=cfg['training'];next_val=c['validation_interval_updates'];best_epoch=None;best_metrics=None
    device=next(runtime.model.parameters()).device;start=time.perf_counter();reason='max_epochs';overflow_skips=0
    for epoch in range(1,c['max_epochs']+1):
        epoch_start=time.perf_counter();loss_total=0.;batches=0;successes=0;crossed=False
        for batch in train_loader:
            r=runtime.step(batch['image'].to(device,non_blocking=True),batch['mask'].to(device,non_blocking=True))
            loss_total+=r['loss'];batches+=1;successes+=int(r['successful']);overflow_skips+=int(not r['successful'])
            if runtime.global_update>=next_val:
                crossed=True
                while next_val<=runtime.global_update: next_val+=c['validation_interval_updates']
            if runtime.global_update>=c['target_updates']: break
        if not batches or not successes: raise RuntimeError('Epoch has no successful optimizer updates; training stopped')
        due=crossed or epoch==c['max_epochs'] or runtime.global_update>=c['target_updates']
        metrics=runtime.validate(val_loader) if due else None
        decision=controller.consider(metrics,runtime.global_update) if due else None
        if decision is not None and decision['save_significant']: best_epoch=epoch;best_metrics=metrics
        row={'epoch':epoch,'global_update':runtime.global_update,'train_loss':loss_total/batches,'optimizer_updates_this_epoch':successes,
            'amp_skipped_updates_total':overflow_skips,'lr':runtime.scheduler.get_last_lr()[0],'validation':metrics,
            'structural_score':decision['score'] if decision else None,'seconds':time.perf_counter()-epoch_start}
        append_row(out/'train.jsonl',row)
        payload=checkpoint_payload(runtime,controller,epoch,cfg,generator)
        payload.update(model_name=model_name,best_epoch=best_epoch,best_validation=best_metrics,metrics=metrics,next_validation_update=next_val)
        if decision:
            if decision['save_absolute']: save(out/'best_absolute_score.pth',payload)
            if decision['save_significant']:
                save(out/'best_structural_score.pth',payload);save(out/'best.pth',payload)
        save(out/'latest.pth',payload);del payload
        stop=decision is not None and decision['stop']
        if epoch%cfg['execution']['report_every_epochs']==0 or stop or epoch==c['max_epochs'] or runtime.global_update>=c['target_updates']:
            print(json.dumps({'model':model_name,**row,'best_epoch':best_epoch,'best_score':controller.selector.significant_best_score}),flush=True)
        if stop: reason='early_stopping';break
        if runtime.global_update>=c['target_updates']: reason='target_updates';break
    if reason=='max_epochs' and runtime.global_update<c['target_updates']: reason='epoch_cap_below_update_target'
    summary={'status':'completed','model':model_name,'seed':cfg['seed'],'epochs_completed':epoch,'actual_updates':runtime.global_update,
        'target_updates':c['target_updates'],'stop_reason':reason,'best_epoch':best_epoch,'best_validation':best_metrics,
        'best_structural_score':controller.selector.significant_best_score,'best_update':controller.selector.significant_best_update,
        'amp_skipped_updates':overflow_skips,'seconds':time.perf_counter()-start,'test_inference':False}
    write_json(out/'run_summary.json',summary);return summary
