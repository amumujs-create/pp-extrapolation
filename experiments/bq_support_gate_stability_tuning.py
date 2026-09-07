#!/usr/bin/env python3
"""Retrospective stability tuning for the unified support-gated PP."""
from __future__ import annotations
import json,sys,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];ADAPTERS=ROOT.parent/'ca-css-ncmapss';sys.path[:0]=[str(ROOT/'src'),str(ADAPTERS),str(ROOT/'experiments')]
from pae_boundary_realdata import DATASETS,prepare_dataset
from pae_shared_battery_nn import BatteryRepresentationScale,concatenate_rows
from boundary_quotient_pp_batteries import build_rows,full_part,score_by_dataset
from pp_extrapolation.boundary_quotient import fit_boundary_quotient_pp,predict_boundary_quotient
OUT=ROOT/'results/bq_support_gate_stability_tuning_v1';SCREEN_SEEDS=(42,43,44);FINAL_SEEDS=(42,43,44,45,46)
CONFIGS=[{'width':w,'learning_rate':lr,'weight_decay':wd} for w in (32,64) for lr in (3e-4,1e-3) for wd in (.03,.1)]
FIXED=dict(alpha=1000.,residual_bound=2.,late_bound_growth=3.,late_bound_power=1.,support_gate_feature=2,support_gate_threshold=.5,support_gate_temperature=.5)
def fit_one(c,seed,rows,scales):
 kw={**FIXED,**c};sel=fit_boundary_quotient_pp(rows['train'],rows['validation'],seed=seed,max_epochs=650,patience=90,**kw);ep=max(sel.selection['selected_epoch'],1);fit=fit_boundary_quotient_pp(rows['full'],rows['full'],seed=seed,max_epochs=ep,patience=10000,restore_best=False,**kw);p=predict_boundary_quotient(fit,rows['source']);return p,score_by_dataset(p,rows['source'],scales),ep,sel.selection['validation_dataset_macro_mse']
def summarize(runs):
 values=np.array([[r['metrics'][d]['pooled_r2'] for d in DATASETS] for r in runs]);return {'worst_seed_dataset_r2':float(values.min()),'mean_seed_dataset_r2':float(values.mean()),'dataset_seed_mean':{d:float(values[:,i].mean()) for i,d in enumerate(DATASETS)},'dataset_seed_sd':{d:float(values[:,i].std()) for i,d in enumerate(DATASETS)}}
def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);start=time.perf_counter();scales={};audits={};parts={k:[] for k in ('train','validation','full','source')}
 for i,n in enumerate(DATASETS):
  s,a=prepare_dataset(n);audits[n]=a;scales[n]=BatteryRepresentationScale.fit(s['train'],a['boundary'])
  for k,p in [('train',s['train']),('validation',s['val']),('full',full_part(s)),('source',s['source'])]:parts[k].append(build_rows(p,scales[n],i))
 rows={k:concatenate_rows(v) for k,v in parts.items()};screen=[]
 for c in CONFIGS:
  runs=[]
  for seed in SCREEN_SEEDS:
   p,m,e,v=fit_one(c,seed,rows,scales);runs.append({'seed':seed,'selected_epoch':e,'validation_mse':v,'metrics':m})
  row={**c,'runs':runs,**summarize(runs)};screen.append(row);print('SCREEN',c,'worst',round(row['worst_seed_dataset_r2'],3),'means',{d:round(x,3) for d,x in row['dataset_seed_mean'].items()},flush=True)
 finalists=sorted(screen,key=lambda x:(x['worst_seed_dataset_r2'],x['mean_seed_dataset_r2']),reverse=True)[:2];final=[]
 for x in finalists:
  c={k:x[k] for k in ('width','learning_rate','weight_decay')};runs=[];preds=[]
  for seed in FINAL_SEEDS:
   p,m,e,v=fit_one(c,seed,rows,scales);preds.append(p);runs.append({'seed':seed,'selected_epoch':e,'validation_mse':v,'metrics':m})
  en=score_by_dataset(np.mean(preds,0),rows['source'],scales);row={**c,'runs':runs,**summarize(runs),'ensemble':en,'ensemble_min_dataset_r2':float(min(en[d]['pooled_r2'] for d in DATASETS)),'ensemble_mean_dataset_r2':float(np.mean([en[d]['pooled_r2'] for d in DATASETS]))};final.append(row);print('FINAL',c,'worst',round(row['worst_seed_dataset_r2'],3),'ensemble',{d:round(en[d]['pooled_r2'],3) for d in DATASETS},flush=True)
 chosen=max(final,key=lambda x:(x['worst_seed_dataset_r2'],x['mean_seed_dataset_r2']));result={'status':'retrospective source-informed stability development; freeze before external confirmation','objective':'maximize worst single-seed dataset pooled R2, then mean','model':'one support-heterogeneity-gated BQ-PP','fixed':FIXED,'screen':screen,'finalists':final,'selected':chosen,'data_audits':audits,'runtime_seconds':time.perf_counter()-start};(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
