#!/usr/bin/env python3
"""Develop boundary-factorized PP after the prospective generic-PP failure."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
import naion_80eol_prospective_gate as data
from naion_prefix_gate_eval import make_rows
from pp_extrapolation import fit_boundary_quotient_pp,predict_boundary_affine,predict_boundary_quotient,regression_metrics
OUT=ROOT/'results/naion_boundary_quotient_pp_v1';SEEDS=(42,43,44,45,46)

def quotient_rows(series,ids,boundary,time_scale,side):
 r=make_rows(series,ids,boundary,time_scale,side);margin=[]
 for uid in ids:
  z=series[uid];t,c=z['cycle'],z['capacity'];mask=(t-t[0]<=boundary) if side=='prefix' else (t-t[0]>boundary)
  for i in np.flatnonzero(mask&(np.arange(len(t))>=3)):margin.append(max(float(c[i])-.8,0.0))
 r.update(margin=np.asarray(margin,np.float32),units=r['groups'],dataset=np.asarray(['naion']*len(margin)))
 return r

def main():
 torch.set_num_threads(2);train,val=data.load_development();raw={n:data.read_file(ROOT/'data/naion_80eol_test'/n) for n in data.TEST_NAMES};test={u:data.eol_truncate(z) for u,z in raw.items()};test={u:z for u,z in test.items() if z is not None};life=[z['cycle'][-1]-z['cycle'][0] for z in train.values()];boundary=float(np.floor(.6*np.median(life)));scale=float(max(life));tr=quotient_rows(train,sorted(train),boundary,scale,'prefix');va=quotient_rows(val,sorted(val),boundary,scale,'tail');te=quotient_rows(test,sorted(test),boundary,scale,'tail')
 pred_v=[];pred_t=[];aff_t=[];runs=[]
 for seed in SEEDS:
  fit=fit_boundary_quotient_pp(tr,va,seed=seed,residual_bound=.5,max_epochs=300,patience=50);pv=predict_boundary_quotient(fit,va);pt=predict_boundary_quotient(fit,te);pa=predict_boundary_affine(fit,te);pred_v.append(pv);pred_t.append(pt);aff_t.append(pa);runs.append({'seed':seed,'validation':regression_metrics(va['y'],pv,va['groups']),'test':regression_metrics(te['y'],pt,te['groups']),'selected_epoch':fit.selection['selected_epoch']})
 pred_v,pred_t,aff_t=map(np.asarray,(pred_v,pred_t,aff_t));result={'status':'post-prospective model development; test cohort was already observed by generic PP','model':'single boundary-quotient PP: RUL=max(capacity-0.8,0)*positive learned quotient','fixed_eol_ah':.8,'residual_bound':.5,'boundary':boundary,'n':{'train_units':len(train),'validation_units':len(val),'test_units':len(test),'test_rows':len(te['y'])},'runs':runs,'validation_ensemble':regression_metrics(va['y'],pred_v.mean(0),va['groups']),'test_affine_ensemble':regression_metrics(te['y'],aff_t.mean(0),te['groups']),'test_bq_pp_ensemble':regression_metrics(te['y'],pred_t.mean(0),te['groups']),'reference_from_frozen_prospective':{'plain_mlp_pooled_r2':.46980484376937115,'generic_pp_pooled_r2':.45708068789441625}}
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(OUT/'predictions.npz',y=te['y'],groups=te['groups'],prediction=pred_t,affine=aff_t);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
