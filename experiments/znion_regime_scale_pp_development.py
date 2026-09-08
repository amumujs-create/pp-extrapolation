#!/usr/bin/env python3
"""Post-confirmation development of a regime-conditioned latent-scale PP."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
import znion_bq_confirmatory as data
from naion_prefix_gate_eval import prefix_descriptor
from pp_extrapolation import fit_boundary_quotient_pp,predict_boundary_quotient,regression_metrics

OUT=ROOT/'results/znion_regime_scale_pp_development_v1';SEEDS=(42,43,44,45,46)

class LifetimeScaleHead(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(6,8),nn.Tanh(),nn.Linear(8,1))
 def forward(self,x):return self.net(x).squeeze(1)

def main():
 torch.set_num_threads(2);train_cells,_=data.load_split('train');val_cells,_=data.load_split('validation');test_cells,_=data.load_split('test');dev={**train_cells,**val_cells}
 life=[c['cycle'][-1]-c['cycle'][0] for c in train_cells.values()];boundary=float(np.floor(.6*np.median(life)));scale=float(max(life))
 train=data.make_rows(train_cells,boundary,scale,'prefix');validation=data.make_rows(val_cells,boundary,scale,'tail');test=data.make_rows(test_cells,boundary,scale,'tail')
 descriptors=np.asarray([prefix_descriptor(c['cycle'],c['capacity'],min(boundary,c['cycle'][-1]-c['cycle'][0])) for c in dev.values()],np.float32);target=np.log(np.asarray([c['cycle'][-1] for c in dev.values()],np.float32));center=descriptors.mean(0);spread=descriptors.std(0);spread[spread<1e-6]=1
 x=torch.tensor((descriptors-center)/spread);y=torch.tensor(target);bq_predictions=[];scale_predictions=[];runs=[]
 for seed in SEEDS:
  bq=fit_boundary_quotient_pp(train,validation,seed=seed,residual_bound=.5,max_epochs=300,patience=50);bq_pred=predict_boundary_quotient(bq,test);bq_predictions.append(bq_pred)
  torch.manual_seed(seed);head=LifetimeScaleHead();optimizer=torch.optim.AdamW(head.parameters(),lr=.01,weight_decay=.01)
  for _ in range(2000):
   loss=(head(x)-y).square().mean();optimizer.zero_grad();loss.backward();optimizer.step()
  pred=[]
  for uid,c in sorted(test_cells.items()):
   if c['cycle'][-1]-c['cycle'][0]<=boundary:continue
   d=prefix_descriptor(c['cycle'],c['capacity'],boundary);estimated_life=float(np.exp(head(torch.tensor(((d-center)/spread)[None],dtype=torch.float32)).item()));idx=np.flatnonzero((c['cycle']-c['cycle'][0])>boundary);pred.extend(np.maximum(estimated_life-c['cycle'][idx],0))
  scale_predictions.append(pred)
 bq_predictions=np.asarray(bq_predictions);scale_predictions=np.asarray(scale_predictions);mixed=[];gates={};offset=0
 for uid,c in sorted(test_cells.items()):
  if c['cycle'][-1]-c['cycle'][0]<=boundary:continue
  idx=np.flatnonzero((c['cycle']-c['cycle'][0])>boundary);slope=float(np.polyfit(c['cycle'][:int(boundary)+1],c['capacity'][:int(boundary)+1],1)[0]);gate=float(1/(1+np.exp(-slope/1e-4)));gates[uid]={'prefix_slope':slope,'latent_scale_gate':gate};n=len(idx);mixed.append((1-gate)*bq_predictions[:,offset:offset+n]+gate*scale_predictions[:,offset:offset+n]);offset+=n
 mixed=np.concatenate(mixed,axis=1)
 for k,seed in enumerate(SEEDS):runs.append({'seed':seed,'metrics':regression_metrics(test['y'],mixed[k],test['groups'])})
 result={'status':'post-untouched developmental result; not independent confirmation','model':'single regime-conditioned PP executor with boundary-quotient path and neural latent-lifetime head','boundary':boundary,'gate':'sigmoid(prefix_slope / 1e-4)','scale_head':{'layers':[6,8,1],'activation':'tanh','target':'log EOL','epochs':2000,'lr':.01,'weight_decay':.01},'gates':gates,'runs':runs,'mean_ensemble':regression_metrics(test['y'],mixed.mean(0),test['groups']),'median_ensemble':regression_metrics(test['y'],np.median(mixed,axis=0),test['groups']),'single_seed_pooled_r2_range':[float(min(r['metrics']['pooled']['r2'] for r in runs)),float(max(r['metrics']['pooled']['r2'] for r in runs))],'references':{'frozen_bq_pp_pooled_r2':-.37892141303524873,'plain_mlp_pooled_r2':-.20365513152787207}}
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(OUT/'predictions.npz',y=test['y'],groups=test['groups'],prediction=mixed,bq=bq_predictions,latent_scale=scale_predictions);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
