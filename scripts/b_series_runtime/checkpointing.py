"""State construction and selection only; no training loop or automatic saving."""
from copy import deepcopy
import random
import numpy as np
import torch
from .selection import AutoProtocolCheckpointState,UpdateEarlyStopping,structural_score

class ValidationController:
    def __init__(self,config):
        self.config=config
        c=config['training']['early_stopping']
        self.selector=AutoProtocolCheckpointState(c['min_delta'])
        self.early=UpdateEarlyStopping(c['min_update'],c['patience_updates'],c['min_delta'])

    def consider(self,metrics,global_update):
        weights=self.config['evaluation']['weights']
        score=structural_score(metrics,{'structural_score_weights':{'f1':weights['F1'],'cldice':weights['clDice'],'boundary_f1':weights['BoundaryF1']}})
        absolute,significant,delta=self.selector.consider(score,global_update)
        self.early.best_significant_score=self.selector.significant_best_score
        self.early.last_significant_update=self.selector.last_significant_update
        return {'score':score,'save_absolute':absolute,'save_significant':significant,'stop':self.early.should_stop(global_update),'delta_from_significant':delta}

def checkpoint_payload(runtime,controller,epoch,config,generator=None):
    # Snapshot deepcopy prevents subsequent optimizer/EMA updates mutating payloads.
    return deepcopy({
        'epoch':epoch,'model':runtime.ema.state_dict(),'raw_model':runtime.model.state_dict(),
        'optimizer':runtime.optimizer.state_dict(),'scheduler':runtime.scheduler.state_dict(),
        'scaler':runtime.scaler.state_dict(),'global_update':runtime.global_update,
        'selection':controller.selector.state_dict(),'early_stopping':vars(controller.early),
        'config':config,'rng':{'python':random.getstate(),'numpy':np.random.get_state(),
            'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},
        'formal_train_generator_state':generator.get_state() if generator is not None else None,
    })
