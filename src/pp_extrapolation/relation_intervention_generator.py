"""Sparse relation-restricted transition generator, not causal identification."""
import torch
from torch import nn


def stretched_gate(logits):
    """Deterministic hard-sigmoid relaxation with exact zeros and ones."""
    return (1.5*logits.sigmoid()-.25).clamp(0,1)


class RelationInterventionGenerator(nn.Module):
    """Generate level and acceleration transition coefficient deltas."""
    def __init__(self,features=15,mode='sparse'):
        super().__init__()
        if mode not in ('sparse','all','gaussian'):
            raise ValueError('unknown generator mode')
        self.features,self.mode=features,mode
        self.edge_logits=nn.Parameter(torch.full((2,features,3),-1.,dtype=torch.float64))
        self.effect=nn.Parameter(torch.zeros((2,features,3),dtype=torch.float64))
        self.bias=nn.Parameter(torch.zeros((2,3),dtype=torch.float64))
        self.log_scale=nn.Parameter(torch.full((2,3),-2.,dtype=torch.float64))
        # level shift and acceleration shift. Small cross-effects are allowed.
        self.register_buffer('physical_mask',torch.tensor([[1.,.25,0.],[.15,1.,1.]],dtype=torch.float64))

    def edge_gate(self):
        if self.mode=='all':return torch.ones_like(self.edge_logits)
        if self.mode=='gaussian':return torch.zeros_like(self.edge_logits)
        return stretched_gate(self.edge_logits)

    def parameters_for(self,relation_features):
        """relation_features [batch, transition_type=2, feature]."""
        if relation_features.ndim!=3 or relation_features.shape[1:]!=(2,self.features):
            raise ValueError('batch x 2 x relation feature tensor required')
        gate=self.edge_gate()
        if self.mode=='gaussian':location=self.bias[None].expand(len(relation_features),-1,-1)
        else:location=torch.einsum('bkf,kfj,kfj->bkj',relation_features,gate,self.effect)+self.bias
        location=location*self.physical_mask
        scale=torch.nn.functional.softplus(self.log_scale)*self.physical_mask.clamp_min(.1)
        return location,scale,gate

    def draw(self,theta_before,relation_features,noise):
        """Return post-transition theta [batch, type, sample, 3]."""
        location,scale,gate=self.parameters_for(relation_features)
        delta=location[:,:,None,:]+scale[None,:,None,:]*noise[None,None,:,:]
        return theta_before[:,:,None,:]+delta,gate
