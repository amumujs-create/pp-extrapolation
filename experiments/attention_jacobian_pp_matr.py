#!/usr/bin/env python3
"""FT attention with a label-free affine-Jacobian prior on future rays."""
import copy
import json
from pathlib import Path

import numpy as np
import torch

from extended_nn_benchmark import Model
from pp_extrapolation import regression_metrics, select_affine_initialization
from pp_extrapolation.model import equal_group_weights, transform_features

SEEDS=(42,43,44,45,46)
WEIGHTS=(0.0,0.01,0.1,1.0,10.0)

def load_split(folder):
    d=np.load(folder/'split.npz');return [{k:d[f'{label}_{k}'] for k in ('x','y','groups')} for label in ('train','validation','test')]

def train(parts,aff,cfg,seed,prior_weight):
    tr,va,te=parts;cap=float(aff['target_scale']);z=transform_features(tr['x'],aff['center'],aff['scale']);vz=transform_features(va['x'],aff['center'],aff['scale']);tz=transform_features(te['x'],aff['center'],aff['scale'])
    x=torch.tensor(z);v=torch.tensor(vz);t=torch.tensor(tz);y=torch.tensor(tr['y']/cap);weights=torch.tensor(equal_group_weights(tr['groups']),dtype=torch.float32)
    direction=1.0 if vz[:,0].mean()>z[:,0].mean() else -1.0;rng=np.random.default_rng(seed);n=min(2048,len(z));ray=z[rng.choice(len(z),n,replace=len(z)<n)].copy();q=direction*z[:,0];vq=direction*vz[:,0];gap=max(float(vq.max()-q.max()),1e-3);ray[:,0]=direction*rng.uniform(float(q.max()),float(vq.max()+gap),n);ray=torch.tensor(ray)
    torch.manual_seed(seed);model=Model('ft_transformer',z.shape[1],cfg,aff,z,vz);opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['wd']);order_rng=np.random.default_rng(seed)
    target_gradient=torch.tensor(float(aff['initialization'].weight[0]))
    def val():
        model.eval()
        with torch.no_grad():p=torch.cat([model(b) for b in v.split(512)]).numpy()*cap
        return float(np.mean((np.clip(p,0,cap)-va['y'])**2))
    best=val();be=0;state=copy.deepcopy(model.state_dict())
    for epoch in range(1,201):
        model.train();order=order_rng.permutation(len(x))
        for start in range(0,len(x),512):
            ix=torch.tensor(order[start:start+512]);loss=torch.mean(weights[ix]*(model(x[ix])-y[ix])**2)
            if prior_weight>0 and (start//512)%4==0:
                ri=torch.tensor(order_rng.integers(0,len(ray),size=min(128,len(ray))));future=ray[ri].detach().requires_grad_(True);fp=model(future);gradient=torch.autograd.grad(fp.sum(),future,create_graph=True)[0][:,0];loss=loss+float(prior_weight)*torch.mean((gradient-target_gradient)**2)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.0);opt.step()
        score=val()
        if score<best-1e-10:best=score;be=epoch;state=copy.deepcopy(model.state_dict())
        if epoch-be>40:break
    model.load_state_dict(state);model.eval()
    with torch.no_grad():raw=torch.cat([model(b) for b in t.split(512)]).numpy()*cap
    return np.clip(raw,0,cap),{'validation_mse':best,'selected_epoch':be,'epochs':epoch}

def main():
    torch.set_num_threads(2);folder=Path('results/ft_original_budget_v1/matr2019');parts=load_split(folder);aff=select_affine_initialization(parts[0],parts[1]);runs=[];pred=[]
    cfg42=torch.load(folder/'seed42.pt',map_location='cpu',weights_only=False)['config'];search=[]
    for weight in WEIGHTS:
        _,info=train(parts,aff,cfg42,42,weight);search.append({'weight':weight,**info});print('SEARCH',weight,round(info['validation_mse'],3),flush=True)
    chosen=min(search,key=lambda row:(row['validation_mse'],row['weight']))
    for seed in SEEDS:
        cfg=torch.load(folder/f'seed{seed}.pt',map_location='cpu',weights_only=False)['config'];p,info=train(parts,aff,cfg,seed,chosen['weight']);m=regression_metrics(parts[2]['y'],p,parts[2]['groups']);pred.append(p);runs.append({'seed':seed,'selected_weight':chosen['weight'],**info,'metrics':m});print('SEED',seed,chosen['weight'],m['pooled']['r2'],flush=True)
    r=np.asarray([x['metrics']['pooled']['r2'] for x in runs]);result={'status':'post-hoc development on inspected MATR2019','architecture':'FT attention predictor + affine-Jacobian prior on label-free counterfactual future rays','weights':list(WEIGHTS),'search_seed':42,'search':search,'selected':chosen,'runs':runs,'mean_r2':float(r.mean()),'sample_sd_r2':float(r.std(ddof=1)),'ensemble':regression_metrics(parts[2]['y'],np.mean(pred,0),parts[2]['groups'])}
    out=Path('results/attention_jacobian_pp_matr_v1');out.mkdir(parents=True,exist_ok=True);(out/'results.json').write_text(json.dumps(result,indent=2));print('DONE',result['ensemble']['pooled']['r2'])
if __name__=='__main__':main()
