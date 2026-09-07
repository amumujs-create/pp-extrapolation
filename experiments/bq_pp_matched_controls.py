#!/usr/bin/env python3
"""Matched six-arm novelty controls for Boundary-Quotient PP."""
from __future__ import annotations

import copy, json, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import wilcoxon

ROOT=Path(__file__).resolve().parents[1];PAE=ROOT.parent/'ca-css-ncmapss'
sys.path[:0]=[str(ROOT/'src'),str(PAE),str(ROOT/'experiments')]
from pae_boundary_realdata import DATASETS,prepare_dataset
from pae_shared_battery_nn import BatteryRepresentationScale,concatenate_rows
from boundary_quotient_pp_batteries import build_rows,full_part,score_by_dataset
from pp_extrapolation.boundary_quotient import (
 BoundaryQuotientPPNet,_softplus_inverse,equal_dataset_unit_weights,
 fit_boundary_quotient_pp,predict_boundary_affine,predict_boundary_quotient,
)

OUT=ROOT/'results/bq_pp_matched_controls_v1';SEEDS=(42,43,44,45,46)
WIDTH=64;ALPHA=1000.;LR=1e-3;WD=.01;BOUND=2.;SOFT_WEIGHTS=(.1,1.,10.)

def direct_affine(rows,center,scale):
 x=(rows['x']-center)/scale;y=_softplus_inverse(rows['y']);w=equal_dataset_unit_weights(rows['dataset'],rows['units']).astype(float)
 total=w.sum();xc=np.sum(w[:,None]*x,0)/total;yc=np.sum(w*y)/total;xx=x-xc;yy=y-yc
 coefficient=np.linalg.solve(xx.T@(w[:,None]*xx)+ALPHA*np.eye(x.shape[1]),xx.T@(w*yy));return coefficient.astype('float32'),float(yc-xc@coefficient)

def fit_direct(train,validation,seed,soft_weight,max_epochs,patience,restore_best=True):
 center=train['x'].mean(0);scale=train['x'].std(0);scale[scale<1e-6]=1.;coef,bias=direct_affine(train,center,scale)
 torch.manual_seed(seed);model=BoundaryQuotientPPNet(train['x'].shape[1],WIDTH,None)
 with torch.no_grad():model.affine.weight.copy_(torch.tensor(coef)[None,:]);model.affine.bias.copy_(torch.tensor([bias]))
 opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD);x=torch.tensor((train['x']-center)/scale,dtype=torch.float32);y=torch.tensor(train['y'],dtype=torch.float32);w=torch.tensor(equal_dataset_unit_weights(train['dataset'],train['units']));vx=torch.tensor((validation['x']-center)/scale,dtype=torch.float32);vy=validation['y'];vd=validation['dataset'];rng=np.random.default_rng(seed);boundary_col=float((0-center[0])/scale[0])
 def forward(value):return model.components(value)[2].squeeze(1)
 def vl():
  model.eval()
  with torch.no_grad():p=forward(vx).numpy()
  return float(np.mean([np.mean((p[vd==d]-vy[vd==d])**2) for d in np.unique(vd)]))
 best=vl();be=0;state=copy.deepcopy(model.state_dict())
 for epoch in range(1,max_epochs+1):
  model.train();order=rng.permutation(len(x))
  for start in range(0,len(x),512):
   ix=torch.tensor(order[start:start+512]);pred=forward(x[ix]);loss=torch.mean(w[ix]*(pred-y[ix]).square())
   if soft_weight>0:
    bx=x[ix].clone();bx[:,0]=boundary_col;loss=loss+soft_weight*forward(bx).square().mean()
   opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
  cur=vl()
  if cur<best-1e-8:best=cur;be=epoch;state=copy.deepcopy(model.state_dict())
  if epoch-be>patience:break
 if restore_best:model.load_state_dict(state)
 else:be=max_epochs
 return {'model':model,'center':center,'scale':scale,'epoch':be,'validation_mse':best}

def predict_direct(fit,rows):
 x=torch.tensor((rows['x']-fit['center'])/fit['scale'],dtype=torch.float32);fit['model'].eval()
 with torch.no_grad():return fit['model'].components(x)[2].squeeze(1).numpy()

def bq_arm(train,validation,full,source,seed,*,trainable=False,bound=BOUND,affine_only=False):
 if affine_only:
  fit=fit_boundary_quotient_pp(train,validation,seed=seed,width=WIDTH,alpha=ALPHA,learning_rate=LR,weight_decay=WD,residual_bound=BOUND,max_epochs=0,patience=0)
  return predict_boundary_affine(fit,source),0,fit
 selected=fit_boundary_quotient_pp(train,validation,seed=seed,width=WIDTH,alpha=ALPHA,learning_rate=LR,weight_decay=WD,residual_bound=bound,trainable_affine=trainable,max_epochs=500,patience=70)
 epochs=max(selected.selection['selected_epoch'],1)
 fit=fit_boundary_quotient_pp(full,full,seed=seed,width=WIDTH,alpha=ALPHA,learning_rate=LR,weight_decay=WD,residual_bound=bound,trainable_affine=trainable,max_epochs=epochs,patience=10000,restore_best=False)
 return predict_boundary_quotient(fit,source),epochs,fit

def bootstrap_units(reference,competitor,reps=50000):
 values=[]
 for d in DATASETS:
  a=reference[d]['per_unit'];b=competitor[d]['per_unit']
  for x,y in zip(a,b):
   assert x['unit']==y['unit'];values.append((x['rmse']-y['rmse'])/max(y['rmse'],1e-12))
 values=np.asarray(values);rng=np.random.default_rng(20260907);sample=np.array([rng.choice(values,len(values),replace=True).mean() for _ in range(reps)])
 return {'n_units':len(values),'reference_wins':int(np.sum(values<0)),'mean_relative_rmse_delta':float(values.mean()),'bootstrap_95_ci':np.quantile(sample,[.025,.975]).tolist(),'wilcoxon_two_sided_p':float(wilcoxon(values).pvalue)}

def shell_metrics(prediction,rows,scales):
 out={}
 for i,name in enumerate(DATASETS):
  mask=rows['dataset']==i;m=rows['margin'][mask];y=rows['y'][mask]*scales[name].time_scale;p=prediction[mask]*scales[name].time_scale
  edges=np.quantile(m,[0,.25,.5,.75,1]);shell=[]
  for j in range(4):
   q=(m>=edges[j])&(m<=edges[j+1]);rmse=float(np.sqrt(np.mean((p[q]-y[q])**2)));shell.append({'margin_min':float(edges[j]),'margin_max':float(edges[j+1]),'n':int(q.sum()),'rmse':rmse,'normalized_rmse':rmse/max(float(np.std(y[q])),1e-12)})
  out[name]=shell
 return out

def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);started=time.perf_counter();scales={};parts={k:[] for k in ('train','validation','full','source')}
 for i,name in enumerate(DATASETS):
  split,audit=prepare_dataset(name);scales[name]=BatteryRepresentationScale.fit(split['train'],audit['boundary'])
  for key,part in [('train',split['train']),('validation',split['val']),('full',full_part(split)),('source',split['source'])]:parts[key].append(build_rows(part,scales[name],i))
 rows={k:concatenate_rows(v) for k,v in parts.items()}
 # Tune only the soft-constraint weight; all remaining settings are shared.
 soft_search=[]
 for value in SOFT_WEIGHTS:
  fit=fit_direct(rows['train'],rows['validation'],42,value,350,55);soft_search.append({'weight':value,'validation_dataset_macro_mse':fit['validation_mse']})
 soft_weight=min(soft_search,key=lambda x:x['validation_dataset_macro_mse'])['weight'];arms={k:[] for k in ('direct_nn','soft_boundary_nn','hard_trainable_nn','affine_quotient_only','frozen_unbounded_pp','bq_pp')};epochs={k:[] for k in arms}
 for seed in SEEDS:
  for arm,sw in [('direct_nn',0.),('soft_boundary_nn',soft_weight)]:
   sel=fit_direct(rows['train'],rows['validation'],seed,sw,500,70);n=max(sel['epoch'],1);fit=fit_direct(rows['full'],rows['full'],seed,sw,n,10000,False);arms[arm].append(predict_direct(fit,rows['source']));epochs[arm].append(n)
  specifications={'hard_trainable_nn':dict(trainable=True,bound=BOUND),'affine_quotient_only':dict(affine_only=True),'frozen_unbounded_pp':dict(bound=None),'bq_pp':dict(bound=BOUND)}
  for arm,kw in specifications.items():
   p,n,_=bq_arm(rows['train'],rows['validation'],rows['full'],rows['source'],seed,**kw);arms[arm].append(p);epochs[arm].append(n)
  print('seed',seed,'done',flush=True)
 result={'status':'retrospective matched six-arm novelty experiment','shared_budget':{'features':11,'width':WIDTH,'alpha':ALPHA,'learning_rate':LR,'weight_decay':WD,'seeds':SEEDS},'soft_weight_search':soft_search,'selected_soft_weight':soft_weight,'arms':{}}
 ensembles={}
 for arm,matrix in arms.items():
  ensemble=np.mean(matrix,0);ensembles[arm]=ensemble;metrics=score_by_dataset(ensemble,rows['source'],scales)
  result['arms'][arm]={'selected_epochs':epochs[arm],'single_seed_metrics':[score_by_dataset(p,rows['source'],scales) for p in matrix],'ensemble':metrics,'health_margin_shells':shell_metrics(ensemble,rows['source'],scales),'dataset_mean_pooled_r2':float(np.mean([metrics[d]['pooled_r2'] for d in DATASETS])),'dataset_macro_unit_r2':float(np.mean([metrics[d]['macro_unit_r2'] for d in DATASETS]))}
  print(arm,{d:round(metrics[d]['pooled_r2'],3) for d in DATASETS},flush=True)
 result['bq_vs_controls']={arm:bootstrap_units(result['arms']['bq_pp']['ensemble'],result['arms'][arm]['ensemble']) for arm in arms if arm!='bq_pp'}
 # Deterministic theorem audit on every source prediction and every seed.
 ratios=[];violations=0
 for seed in SEEDS:
  _,_,fit=bq_arm(rows['train'],rows['validation'],rows['full'],rows['source'],seed,bound=BOUND)
  pred=predict_boundary_quotient(fit,rows['source']);base=predict_boundary_affine(fit,rows['source']);envelope=rows['source']['margin']*BOUND
  violations+=int(np.sum(np.abs(pred-base)>envelope+1e-5));ratios.extend((np.abs(pred-base)/np.maximum(envelope,1e-12)).tolist())
 result['boundary_contraction_audit']={'theorem':'abs(prediction-affine)<=margin*B','n_predictions':len(ratios),'violations':violations,'max_envelope_utilization':float(np.max(ratios)),'p95_envelope_utilization':float(np.quantile(ratios,.95))}
 result['runtime_seconds']=time.perf_counter()-started;(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(OUT/'predictions.npz',y=rows['source']['y'],units=rows['source']['units'],dataset=rows['source']['dataset'],margin=rows['source']['margin'],**{k:np.asarray(v) for k,v in arms.items()})
 print('audit',result['boundary_contraction_audit'])
if __name__=='__main__':main()
