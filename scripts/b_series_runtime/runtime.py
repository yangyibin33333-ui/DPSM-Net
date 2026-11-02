import os,random
import numpy as np
import torch
from .ema import ModelEMA
from .loss import BCEDiceLoss
from .metrics import SegmentationMetric

def set_seed(seed=42):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    torch.backends.cudnn.benchmark=True

def make_grad_scaler(settings):
    # PyTorch 2.1 exposes the CUDA scaler under torch.cuda.amp.
    modern = getattr(getattr(torch, 'amp', None), 'GradScaler', None)
    if modern is not None:
        return modern('cuda', **settings)
    return torch.cuda.amp.GradScaler(**settings)

class StepRuntime:
    def __init__(self,model,config):
        c=config['training'];self.model=model
        self.criterion=BCEDiceLoss(pos_weight=1.)
        self.optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'],betas=tuple(c['betas']),eps=c['eps'],fused=c['fused'])
        self.scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=c['target_updates'],eta_min=c['eta_min'])
        self.scaler=make_grad_scaler(c['grad_scaler'])
        self.ema=ModelEMA(model,decay=c['ema']);self.global_update=0
        self.accumulation=c['accumulation_steps']

    def step(self,image,mask):
        if image.shape[0]%self.accumulation: raise ValueError('Incomplete effective batch')
        self.model.train();self.optimizer.zero_grad(set_to_none=True);losses=[]
        for x,y in zip(image.chunk(self.accumulation),mask.chunk(self.accumulation)):
            with torch.autocast('cuda',dtype=torch.float16):
                logits=self.model(x);loss=self.criterion(logits,y)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss): raise FloatingPointError('Non-finite logits/loss')
            self.scaler.scale(loss/self.accumulation).backward();losses.append(float(loss.detach()))
        before=self.scaler.get_scale()
        self.scaler.unscale_(self.optimizer)
        finite_grad=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in self.model.parameters())
        self.scaler.step(self.optimizer);self.scaler.update()
        success=self.scaler.get_scale()>=before
        if success:
            self.ema.update(self.model);self.scheduler.step();self.global_update+=1
        return {'loss':sum(losses)/len(losses),'successful':success,'finite_gradients':finite_grad,'scale_before':before,'scale_after':self.scaler.get_scale(),'output_shape':list(logits.shape),'optimizer_updates':self.global_update}

    @torch.no_grad()
    def validate(self,loader):
        # Original train.validate has no autocast: FP32 EMA validation.
        self.ema.module.eval();metric=SegmentationMetric();loss_sum=0.;n=0
        for batch in loader:
            x=batch['image'].to('cuda');y=batch['mask'].to('cuda')
            logits=self.ema.module(x);metric.update(logits,y)
            loss_sum+=float(self.criterion(logits,y))*len(x);n+=len(x)
        return metric.compute()|{'Loss':loss_sum/max(n,1)}
