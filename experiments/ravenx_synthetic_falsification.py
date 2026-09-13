"""Synthetic structural test only; never evidence of real-data causality/performance."""
import json
import numpy as np
import torch
from ravenx_train_screen import ROOT
from relation_local_transport_screen import write
from pp_extrapolation.relation_intervention_generator import RelationInterventionGenerator


def main():
    torch.set_num_threads(2);out=ROOT/'results/ravenx_train_screen_v1'
    rng=np.random.default_rng(20260913);n=4096;features=15
    x=rng.normal(size=(n,2,features));truth=np.zeros((2,features,3));
    truth[0,0,0]=.6;truth[0,1,0]=-.4;truth[0,2,1]=.25
    truth[1,3,1]=.45;truth[1,4,1]=-.35;truth[1,5,2]=.55;truth[1,6,2]=-.3;truth[1,7,0]=.15
    mask=truth!=0;physical=np.array([[1,.25,0],[.15,1,1]])
    y=np.einsum('nkf,kfj->nkj',x,truth)*physical[None]+rng.normal(scale=.02,size=(n,2,3))*physical[None]
    train=np.arange(3072);test=np.arange(3072,n);model=RelationInterventionGenerator(features,'sparse')
    opt=torch.optim.Adam(model.parameters(),lr=.03)
    tx=torch.tensor(x[train]);ty=torch.tensor(y[train])
    for step in range(800):
        location,_,gate=model.parameters_for(tx)
        loss=(location-ty).square().mean()+.002*gate.mean()
        opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():
        pred,_,gate=model.parameters_for(torch.tensor(x[test]));mse=float((pred-torch.tensor(y[test])).square().mean())
        g=gate.numpy();allowed=np.broadcast_to(physical[:,None,:]>0,g.shape)
        count=int(mask.sum());rank=np.argsort(g[allowed])[-count:];selected=np.flatnonzero(allowed)[rank]
        top=np.zeros(g.size,dtype=bool);top[selected]=True;top=top.reshape(g.shape)
        precision=float((top&mask).sum()/count);chance=float(mask[allowed].mean())
        # Perturb only protected features 10:15; learned output should remain nearly invariant.
        base=model.parameters_for(torch.zeros((256,2,features),dtype=torch.float64))[0]
        intervention=torch.zeros((256,2,features),dtype=torch.float64);intervention[:,:,10:]=3
        shifted=model.parameters_for(intervention)[0]
        protected_change=float((shifted-base).abs().max())
    passed=precision>=.75 and precision>=2*chance and protected_change<=.05 and mse<=.01
    result=dict(complete=True,seed=20260913,true_edges=count,top_edge_precision=precision,chance_precision=chance,
        test_delta_mse=mse,protected_feature_max_change=protected_change,gate_passed=passed,
        limits=['known synthetic graph only','not evidence of observational causal identification','same generator class, not full RAVEN trajectory recovery'])
    torch.save(model,out/'synthetic_generator.pt');write(out/'synthetic_falsification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
