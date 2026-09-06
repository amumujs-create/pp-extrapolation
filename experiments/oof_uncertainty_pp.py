#!/usr/bin/env python3
"""Ensemble-free-at-inference PP with a nested-OOF uncertainty head."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
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
def audit_uncertainty(y,g,pp,gated,u):
 error=np.mean(np.abs(pp-y[None,:]),axis=0);rho,pvalue=spearmanr(u,error)
 order=np.argsort(u);risk={}
 for coverage in (.25,.5,.75,1.):
  ix=order[:max(1,int(np.ceil(len(order)*coverage)))];risk[str(coverage)]={'rmse':float(np.sqrt(np.mean((pp.mean(0)[ix]-y[ix])**2))),'n':len(ix)}
 units=np.unique(g);rng=np.random.default_rng(20260906);delta=[]
 for _ in range(5000):
  draw=rng.choice(units,len(units),replace=True);ix=np.concatenate([np.flatnonzero(g==z) for z in draw])
  delta.append(np.sqrt(np.mean((gated.mean(0)[ix]-y[ix])**2))-np.sqrt(np.mean((pp.mean(0)[ix]-y[ix])**2)))
 return {'spearman_u_vs_absolute_pp_error':float(rho),'spearman_p':float(pvalue),'risk_coverage':risk,
 'unit_bootstrap_ensemble_rmse_delta':{'mean':float(np.mean(delta)),'ci95':[float(np.quantile(delta,.025)),float(np.quantile(delta,.975))]},
 'improved_seed_count':int(np.sum([met(y,b,g)['pooled']['r2']>met(y,a,g)['pooled']['r2'] for a,b in zip(pp,gated)]))}
def run(name,tr,va,te,epochs):
 oof=nested_oof_disagreement(tr,max_epochs=min(epochs,200));head=fit_uncertainty_head(tr,oof)
 affine=select_affine_initialization(tr,va);fits=[fit_pp(tr,va,seed=s,affine_selection=affine,max_epochs=epochs) for s in SEEDS]
 vd,_=support_distance(tr['x'][:,[0]],va['x'][:,[0]]);td,_=support_distance(tr['x'][:,[0]],te['x'][:,[0]])
 vu=predict_uncertainty(head,va['x']);tu=predict_uncertainty(head,te['x']);sel,rows=choose(fits,va['x'],va['y'],vd,vu)
 constant_vu=np.full_like(vu,float(vu.mean()));constant_tu=np.full_like(tu,float(vu.mean()));constant_sel,constant_rows=choose(fits,va['x'],va['y'],vd,constant_vu)
 a,r,cap=components(fits,te['x']);gated=combine_uncertainty_gated(a,r,td,tu,beta=sel['beta'],gamma=sel['gamma'],output_cap=cap);pp=np.asarray([predict(f,te['x']) for f in fits])
 constant=combine_uncertainty_gated(a,r,td,constant_tu,beta=constant_sel['beta'],gamma=constant_sel['gamma'],output_cap=cap)
 localization_gain=(constant_sel['mean_seed_validation_mse']-sel['mean_seed_validation_mse'])/max(constant_sel['mean_seed_validation_mse'],1e-12)
 certificate={'accepted':bool(sel['gamma']>0 and localization_gain>=.001),'minimum_relative_localization_gain':.001,'relative_localization_gain':float(localization_gain)}
 certified=gated if certificate['accepted'] else pp
 print(name,sel,'PP',summ(te['y'],pp,te['groups'])['pooled_r2_mean'],'OOF-UQ',summ(te['y'],gated,te['groups'])['pooled_r2_mean'],flush=True)
 return {'selection':sel,'candidates':rows,'constant_selection':constant_sel,'constant_candidates':constant_rows,
 'oof_uncertainty':{'mean':head.oof_target_mean,'sd':head.oof_target_sd},'test_predicted_uncertainty':{'mean':float(tu.mean()),'sd':float(tu.std())},
 'pp':summ(te['y'],pp,te['groups']),'constant_u_pp':summ(te['y'],constant,te['groups']),'oof_uq_pp':summ(te['y'],gated,te['groups']),
 'localization_certificate':certificate,'certified_oof_uq_pp':summ(te['y'],certified,te['groups']),
 'uncertainty_audit':audit_uncertainty(te['y'],te['groups'],pp,gated,tu),'_a':(te['y'],te['groups'],pp,gated)}
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
 for n in ('sunwoda','rwth','mich'):
  raw,_=prepare_dataset(n);ds[n]=run(n,*prepare_battery(raw),a.max_epochs);ds[n].pop('_a')
 ds['nasa']={'status':'not_estimable','reason':'each fold has fewer than six independent training cells; nested unit-OOF uncertainty would be pseudo-replication'}
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);(out/'results.json').write_text(json.dumps({'experiment':'nested_oof_uncertainty_head_v1','test_inference_pp_models':1,'oof_outer_folds':3,'oof_teacher_seeds':[101,102,103],'datasets':ds,'runtime_seconds':time.time()-start},indent=2)+'\n')
if __name__=='__main__':main()
