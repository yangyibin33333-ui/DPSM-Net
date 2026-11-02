"""Meaningful contract checks; synthetic tensors only, no Test data."""
import sys,json,copy
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'models'),str(ROOT/'scripts'),str(ROOT)]
from b_series import build_model,PlainConvResidual
from b_series_runtime.loss import BCEDiceLoss
from b_series_runtime.selection import AutoProtocolCheckpointState,UpdateEarlyStopping
from b_series_runtime.checkpointing import ValidationController

def main():
    torch.set_num_threads(4)
    for name in ['B0','B1','B2','B2-Conv']:
        torch.manual_seed(42);m=build_model(name)
        x=torch.randn(2,3,32,64);target=(torch.rand(2,1,32,64)>.96).float();out=m(x)
        assert out.shape==target.shape
        loss=BCEDiceLoss()(out,target);loss.backward()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    # Empty/full/sparse-mask Dice/BCE exact reduction and gradient contract.
    for probability in [0.,.05,1.]:
        z=torch.randn(2,1,16,24,requires_grad=True);y=(torch.rand_like(z)<probability).float();p=z.sigmoid()
        manual=torch.nn.functional.binary_cross_entropy_with_logits(z,y)+1-((2*(p*y).sum((1,2,3))+1e-6)/(p.sum((1,2,3))+y.sum((1,2,3))+1e-6)).mean()
        actual=BCEDiceLoss()(z,y);assert torch.allclose(actual,manual)
        assert torch.equal(torch.autograd.grad(actual,z,retain_graph=True)[0],torch.autograd.grad(manual,z)[0])
    selector=AutoProtocolCheckpointState(.0005)
    assert selector.consider(.8,138)[:2]==(True,True)
    assert selector.consider(.8001,276)[:2]==(True,False)
    early=UpdateEarlyStopping(5060,1380,.0005);early.last_significant_update=4000
    assert not early.should_stop(5379) and early.should_stop(5380)
    print(json.dumps({'status':'passed','four_forward_backward':True,'finite_loss_gradients':True,'original_dice_formula_gradient':True,'selection_early_stopping':True,'formal_training':False,'test_inference':False}))
if __name__=='__main__':main()
