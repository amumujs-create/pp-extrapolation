#!/usr/bin/env python3
"""Temporal latent PP on the fixed MATR2019 extrapolation split."""
import copy
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from extended_nn_benchmark import data
from pp_extrapolation import regression_metrics, select_affine_initialization
from pp_extrapolation.model import equal_group_weights, transform_features

SEEDS=(42,43,44,45,46)
GRID=[
    {"width":width,"lr":lr,"weight_decay":wd,"regularizer":reg}
    for width in (16,32,64)
    for lr,wd,reg in ((2e-4,.1,0.0),(5e-4,2.0,.001),(1e-3,0.0,.01))
]
RESIDUAL_GRID=[
    {"width":width,"lr":lr,"weight_decay":wd,"regularizer":0.0,
     "mode":"residual","residual_scale":scale}
    for width,scale in ((16,.25),(32,.5),(64,1.0))
    for lr,wd in ((2e-4,.1),(5e-4,2.0),(1e-3,0.0))
]

class TemporalLatentPP(nn.Module):
    """Frozen affine prior with a causal GRU-conditioned latent residual."""
    def __init__(self,d,sequence_features,direction,knot,width):
        super().__init__();self.direction=float(direction);self.knot=float(knot)
        self.affine=nn.Linear(d,1);self.encoder=nn.GRU(sequence_features,width,batch_first=True)
        context_dim=d+width
        self.gate_q=nn.Parameter(torch.tensor(-2.0));self.gate_bias=nn.Parameter(torch.tensor(-2.0))
        self.gate_context=nn.Sequential(nn.Linear(context_dim-1,width),nn.Tanh(),nn.Linear(width,1))
        self.experts=nn.ModuleList([nn.Sequential(nn.Linear(context_dim,width),nn.Tanh(),nn.Linear(width,2)) for _ in range(2)])
        for block in [self.gate_context,*self.experts]:
            nn.init.zeros_(block[-1].weight);nn.init.zeros_(block[-1].bias)
    def components(self,x,sequence):
        encoded=self.encoder(sequence)[0][:,-1];joint=torch.cat([x,encoded],dim=1)
        q=self.direction*x[:,:1];gate_input=torch.cat([x[:,1:],encoded],dim=1)
        gate=torch.sigmoid(torch.nn.functional.softplus(self.gate_q)*q+self.gate_bias+self.gate_context(gate_input))
        hinge=torch.relu(q-self.knot);tails=[]
        for expert in self.experts:
            raw=expert(joint);tails.append(.25*torch.tanh(raw[:,:1])+.25*torch.tanh(raw[:,1:])*hinge)
        correction=(1-gate)*tails[0]+gate*tails[1]
        return self.affine(x).squeeze(1)+correction.squeeze(1),gate,tails
    def forward(self,x,sequence):return self.components(x,sequence)[0]

class TemporalResidualPP(nn.Module):
    """Frozen affine prior plus one causal GRU residual path."""
    def __init__(self,d,sequence_features,width,residual_scale):
        super().__init__();self.affine=nn.Linear(d,1);self.encoder=nn.GRU(sequence_features,width,batch_first=True);self.residual_scale=float(residual_scale)
        self.head=nn.Sequential(nn.Linear(d+width,width),nn.Tanh(),nn.Linear(width,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def components(self,x,sequence):
        encoded=self.encoder(sequence)[0][:,-1];correction=self.residual_scale*torch.tanh(self.head(torch.cat([x,encoded],1)))
        dummy=torch.zeros_like(correction);return self.affine(x).squeeze(1)+correction.squeeze(1),dummy,[correction,correction]
    def forward(self,x,sequence):return self.components(x,sequence)[0]

def arrays(parts,aff):
    tab=[torch.tensor(transform_features(p['x'],aff['center'],aff['scale'])) for p in parts]
    center=parts[0]['sequence'].mean((0,1));scale=np.maximum(parts[0]['sequence'].std((0,1)),1e-6)
    seq=[torch.tensor((p['sequence']-center)/scale,dtype=torch.float32) for p in parts]
    return tab,seq,center,scale

def fit(parts,aff,cfg,seed):
    tr,va,_=parts;(tabs,seqs,sc,ss)=arrays(parts,aff);x,v,_=tabs;sx,sv,_=seqs
    cap=float(aff['target_scale']);direction=1. if v[:,0].mean()>x[:,0].mean() else -1.;knot=float(np.quantile(direction*x[:,0].numpy(),.8))
    torch.manual_seed(seed)
    if cfg.get('mode')=='residual':model=TemporalResidualPP(x.shape[1],sx.shape[2],cfg['width'],cfg['residual_scale'])
    else:model=TemporalLatentPP(x.shape[1],sx.shape[2],direction,knot,cfg['width'])
    with torch.no_grad():model.affine.weight.copy_(torch.tensor(aff['initialization'].weight)[None,:]);model.affine.bias.fill_(aff['initialization'].bias)
    model.affine.requires_grad_(False);y=torch.tensor(tr['y']/cap);w=torch.tensor(equal_group_weights(tr['groups']),dtype=torch.float32)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['lr'],weight_decay=cfg['weight_decay']);rng=np.random.default_rng(seed)
    def validation():
        model.eval()
        with torch.no_grad():pred=torch.cat([model(a,b) for a,b in zip(v.split(512),sv.split(512))]).numpy()*cap
        return float(np.mean((np.clip(pred,0,cap)-va['y'])**2))
    best=validation();be=0;state=copy.deepcopy(model.state_dict());started=time.monotonic()
    for epoch in range(1,301):
        model.train();order=rng.permutation(len(x))
        for start in range(0,len(x),512):
            ix=torch.tensor(order[start:start+512]);pred,gate,tails=model.components(x[ix],sx[ix]);loss=torch.mean(w[ix]*(pred-y[ix])**2)
            separation=torch.mean((tails[0]-tails[1])**2);gate_reg=(gate.mean()-.5).square()+torch.mean(gate*(1-gate));reg=cfg['regularizer']
            loss=loss-reg*torch.clamp(separation,max=.05)+reg*gate_reg
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
        score=validation()
        if score<best-1e-10:best=score;be=epoch;state=copy.deepcopy(model.state_dict())
        if epoch-be>70:break
    model.load_state_dict(state)
    return model,tabs[2],seqs[2],{"validation_mse":best,"selected_epoch":be,"epochs":epoch,"seconds":time.monotonic()-started,"sequence_center":sc.tolist(),"sequence_scale":ss.tolist()}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--architecture',choices=('latent','residual'),default='latent');parser.add_argument('--output',type=Path);args=parser.parse_args()
    grid=RESIDUAL_GRID if args.architecture=='residual' else GRID
    torch.set_num_threads(2);parts=data();aff=select_affine_initialization(parts[0],parts[1]);out=args.output or Path(f'results/temporal_{args.architecture}_pp_matr_v1');out.mkdir(parents=True,exist_ok=True)
    search=[]
    for index,cfg in enumerate(grid):
        model,_,_,info=fit(parts,aff,cfg,42);search.append({"index":index,"config":cfg,**info});print('SEARCH',index,round(info['validation_mse'],3),flush=True)
    chosen=min(search,key=lambda row:(row['validation_mse'],row['index']));cfg=chosen['config'];runs=[];pred=[]
    for seed in SEEDS:
        model,test_x,test_seq,info=fit(parts,aff,cfg,seed);model.eval()
        with torch.no_grad():raw=torch.cat([model(a,b) for a,b in zip(test_x.split(512),test_seq.split(512))]).numpy()*aff['target_scale']
        prediction=np.clip(raw,0,aff['target_scale']);metrics=regression_metrics(parts[2]['y'],prediction,parts[2]['groups']);pred.append(prediction);runs.append({"seed":seed,**info,"metrics":metrics});print('REFIT',seed,metrics['pooled']['r2'],flush=True)
        np.savez_compressed(out/f'seed{seed}.npz',prediction=prediction,raw=raw,y=parts[2]['y'],groups=parts[2]['groups'])
        torch.save({'state':model.state_dict(),'config':cfg,'center':aff['center'],'scale':aff['scale'],'target_scale':aff['target_scale']},out/f'seed{seed}.pt')
    label='frozen affine + causal GRU residual' if args.architecture=='residual' else 'frozen affine + causal GRU encoder + latent gated residual experts'
    r=np.asarray([x['metrics']['pooled']['r2'] for x in runs]);result={'status':'post-hoc model development on inspected MATR2019','architecture':label,'search':search,'selected_config':cfg,'runs':runs,'mean_r2':float(r.mean()),'sample_sd_r2':float(r.std(ddof=1)),'ensemble':regression_metrics(parts[2]['y'],np.mean(pred,axis=0),parts[2]['groups'])}
    (out/'results.json').write_text(json.dumps(result,indent=2));print('DONE',result['ensemble']['pooled']['r2'])
if __name__=='__main__':main()
