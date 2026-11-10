"""Explicit controls sharing the real evidence cache, never fabricated experts."""
import torch
from torch import nn
from reference.core import FusionOutput

class BestSingleExpert(nn.Module):
    def __init__(self,index=0):
        super().__init__();self.register_buffer('chosen_expert',torch.tensor(index,dtype=torch.long))
    def forward(self,hidden,scores,valid,probe_features,quality):
        index=int(self.chosen_expert);return FusionOutput(scores[:,index,0],valid[:,index,0])

class ProbeSubset(nn.Module):
    def __init__(self,model,indices=(0,2,4)):
        super().__init__();self.model=model;self.indices=tuple(indices)
    def forward(self,hidden,scores,valid,probe_features,quality):
        mask=valid.clone();keep=torch.zeros(valid.shape[-1],dtype=torch.bool,device=valid.device);keep[0]=True
        for i in self.indices:keep[i+1]=True
        mask &= keep[None,None,:]
        return self.model(hidden,scores,mask,probe_features,quality)

class InformationControl(nn.Module):
    """Controlled token-input ablation; original observations/caches stay intact."""
    def __init__(self,model,mode):super().__init__();self.model=model;self.mode=mode
    def forward(self,hidden,scores,valid,probe_features,quality):
        valid=valid & valid[:,:,:1]  # copied responses require a valid original
        if self.mode in ('no_embedding_response','identity_copy_views'):
            hidden=[torch.cat([h[:,:1],h[:,:1].expand(-1,h.shape[1]-1,-1)],1) for h in hidden]
        if self.mode in ('no_score_response','identity_copy_views'):
            scores=torch.cat([scores[:,:,:1],scores[:,:,:1].expand(-1,-1,scores.shape[-1]-1)],-1)
        return self.model(hidden,scores,valid,probe_features,quality)
