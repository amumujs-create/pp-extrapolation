#!/usr/bin/env python3
"""One-head dual-scale saturating PP between local and broad residual priors."""
from __future__ import annotations
import json,sys,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];ADAPTERS=ROOT.parent/'ca-css-ncmapss';sys.path[:0]=[str(ROOT/'src'),str(ADAPTERS),str(ROOT/'experiments')]
from pae_boundary_realdata import DATASETS,prepare_dataset
from pae_shared_battery_nn import BatteryRepresentationScale,concatenate_rows
from boundary_quotient_pp_batteries import build_rows,full_part,score_by_dataset
from pp_extrapolation.boundary_quotient import fit_boundary_quotient_pp,predict_boundary_quotient
OUT=ROOT/'results/bq_dual_scale_saturation_v1';SEEDS=(42,43,44,45,46)
CONFIGS=[{'broad_residual_bound':b,'local_saturation_weight':w} for b in (6.,10.,16.,24.) for w in (.2,.4,.6,.8)]
BASE=dict(width=64,alpha=1000.,learning_rate=1e-3,weight_decay=.01,residual_bound=2.)
def one(c,seed,rows,scales):
 sel=fit_boundary_quotient_pp(rows['train'],rows['validation'],seed=seed,max_epochs=500,patience=70,**BASE,**c);ep=max(sel.selection['selected_epoch'],1);fit=fit_boundary_quotient_pp(rows['full'],rows['full'],seed=seed,max_epochs=ep,patience=10000,restore_best=False,**BASE,**c);p=predict_boundary_quotient(fit,rows['source']);return p,score_by_dataset(p,rows['source'],scales),ep,sel.selection['validation_dataset_macro_mse']
def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);start=time.perf_counter();scales={};audits={};parts={k:[] for k in ('train','validation','full','source')}
 for i,n in enumerate(DATASETS):
  s,a=prepare_dataset(n);audits[n]=a;scales[n]=BatteryRepresentationScale.fit(s['train'],a['boundary'])
  for k,p in [('train',s['train']),('validation',s['val']),('full',full_part(s)),('source',s['source'])]:parts[k].append(build_rows(p,scales[n],i))
 rows={k:concatenate_rows(v) for k,v in parts.items()};screen=[]
 for c in CONFIGS:
  p,m,e,v=one(c,42,rows,scales);a=[m[d]['pooled_r2'] for d in DATASETS];row={**c,'selected_epoch':e,'validation_mse':v,'metrics':m,'min_pooled_r2':float(min(a)),'mean_pooled_r2':float(np.mean(a))};screen.append(row);print('SCREEN',c,{d:round(m[d]['pooled_r2'],3) for d in DATASETS},flush=True)
 finalists=sorted(screen,key=lambda x:(x['min_pooled_r2'],x['mean_pooled_r2']),reverse=True)[:3];final=[]
 for x in finalists:
  c={k:x[k] for k in ('broad_residual_bound','local_saturation_weight')};runs=[];preds=[]
  for seed in SEEDS:
   p,m,e,v=one(c,seed,rows,scales);preds.append(p);runs.append({'seed':seed,'selected_epoch':e,'validation_mse':v,'metrics':m})
  en=score_by_dataset(np.mean(preds,0),rows['source'],scales);a=np.asarray([[r['metrics'][d]['pooled_r2'] for d in DATASETS] for r in runs]);s=[en[d]['pooled_r2'] for d in DATASETS];row={**c,'runs':runs,'ensemble':en,'ensemble_min_dataset_r2':float(min(s)),'ensemble_mean_dataset_r2':float(np.mean(s)),'worst_seed_dataset_r2':float(a.min()),'dataset_seed_mean':{d:float(a[:,i].mean()) for i,d in enumerate(DATASETS)},'dataset_seed_sd':{d:float(a[:,i].std()) for i,d in enumerate(DATASETS)}};final.append(row);print('FINAL',c,'ensemble',{d:round(en[d]['pooled_r2'],3) for d in DATASETS},'worstseed',round(a.min(),3),flush=True)
 chosen=max(final,key=lambda x:(x['ensemble_min_dataset_r2'],x['ensemble_mean_dataset_r2']));result={'status':'retrospective source-informed one-head PP development; requires frozen external confirmation','objective':'high pooled R2 on every development dataset with a finite residual envelope','model':'dual-scale saturating boundary-quotient PP','base':BASE,'screen':screen,'finalists':final,'selected':chosen,'envelope':'margin * (w*B_local + (1-w)*B_broad)','data_audits':audits,'runtime_seconds':time.perf_counter()-start};(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
