#!/usr/bin/env python3
"""Ensemble-free-at-inference PP with a nested-OOF uncertainty head."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from pp_extrapolation import (combine_uncertainty_gated,fit_pp,fit_uncertainty_head,
 nested_oof_disagreement,predict,predict_uncertainty,regression_metrics,select_affine_initialization,support_distance)
from pp_extrapolation.support_gate import predict_components
from distance_uncertainty_pp import prepare_battery

SEEDS=(42,43,44,45,46);BETAS=(0.,.05,.1,.25,.5,1.,2.,4.,8.);GAMMAS=(0.,1.,2.,4.,8.,16.,32.,64.)
def met(y,p,g):return regression_metrics(y,p,g)
def summ(y,m,g):
 r=np.array([met(y,p,g)['pooled']['r2'] for p in m]);return {'pooled_r2_mean':float(r.mean()),'pooled_r2_sd':float(r.std()),'ensemble':met(y,m.mean(0),g),'per_seed_r2':r.tolist()}
def components(fits,x):
 c=[predict_components(f,x) for f in fits];return np.asarray([z[0] for z in c]),np.asarray([z[1] for z in c]),min(f.target_scale for f in fits)
def choose(fits,x,y,d,u):
 a,r,cap=components(fits,x);rows=[]
 for b in BETAS:
  for ga in GAMMAS:
   p=combine_uncertainty_gated(a,r,d,u,beta=b,gamma=ga,output_cap=cap)
   rows.append({'beta':b,'gamma':ga,'mean_seed_validation_mse':float(np.mean((p-y[None,:])**2))})
 return min(rows,key=lambda z:(z['mean_seed_validation_mse'],z['beta'],z['gamma'])),rows
def run(name,tr,va,te,epochs):
 oof=nested_oof_disagreement(tr,max_epochs=min(epochs,200));head=fit_uncertainty_head(tr,oof)
 affine=select_affine_initialization(tr,va);fits=[fit_pp(tr,va,seed=s,affine_selection=affine,max_epochs=epochs) for s in SEEDS]
 vd,_=support_distance(tr['x'][:,[0]],va['x'][:,[0]]);td,_=support_distance(tr['x'][:,[0]],te['x'][:,[0]])
 vu=predict_uncertainty(head,va['x']);tu=predict_uncertainty(head,te['x']);sel,rows=choose(fits,va['x'],va['y'],vd,vu)
 a,r,cap=components(fits,te['x']);gated=combine_uncertainty_gated(a,r,td,tu,beta=sel['beta'],gamma=sel['gamma'],output_cap=cap);pp=np.asarray([predict(f,te['x']) for f in fits])
 print(name,sel,'PP',summ(te['y'],pp,te['groups'])['pooled_r2_mean'],'OOF-UQ',summ(te['y'],gated,te['groups'])['pooled_r2_mean'],flush=True)
 return {'selection':sel,'candidates':rows,'oof_uncertainty':{'mean':head.oof_target_mean,'sd':head.oof_target_sd},'test_predicted_uncertainty':{'mean':float(tu.mean()),'sd':float(tu.std())},'pp':summ(te['y'],pp,te['groups']),'oof_uq_pp':summ(te['y'],gated,te['groups']),'_a':(te['y'],te['groups'],pp,gated)}
def nasa(folds,epochs):
 parts=[]
 for f in folds:parts.append(run('NASA-'+f['test_cell'],f['train'],f['validation'],f['test'],epochs))
 y=np.concatenate([d['_a'][0] for d in parts]);g=np.concatenate([d['_a'][1] for d in parts]);pp=np.asarray([np.concatenate([d['_a'][2][i] for d in parts]) for i in range(5)]);uq=np.asarray([np.concatenate([d['_a'][3][i] for d in parts]) for i in range(5)])
 for d in parts:d.pop('_a')
 return {'fold_results':parts,'pp':summ(y,pp,g),'oof_uq_pp':summ(y,uq,g)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--legacy-root',default=str(Path(__file__).resolve().parents[2]/'ca-css-ncmapss'));p.add_argument('--max-epochs',type=int,default=300);p.add_argument('--output',default='results/oof_uncertainty_pp_v1');a=p.parse_args();sys.path.insert(0,str(Path(a.legacy_root).resolve()))
 from run_affine_tail_external_three import prepare_hust,prepare_virkler
 from pae_boundary_realdata import prepare_dataset
 torch.set_num_threads(2);start=time.time();ds={}
 for n,pd in [('hust',prepare_hust()),('virkler',prepare_virkler())]:ds[n]=run(n,*pd[:3],a.max_epochs);ds[n].pop('_a')
 for n in ('rwth','mich'):
  raw,_=prepare_dataset(n);ds[n]=run(n,*prepare_battery(raw),a.max_epochs);ds[n].pop('_a')
 ds['nasa']={'status':'not_estimable','reason':'each fold has fewer than six independent training cells; nested unit-OOF uncertainty would be pseudo-replication'}
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);(out/'results.json').write_text(json.dumps({'experiment':'nested_oof_uncertainty_head_v1','test_inference_pp_models':1,'oof_outer_folds':3,'oof_teacher_seeds':[101,102,103],'datasets':ds,'runtime_seconds':time.time()-start},indent=2)+'\n')
if __name__=='__main__':main()
