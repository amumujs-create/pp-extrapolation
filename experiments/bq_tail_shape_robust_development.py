#!/usr/bin/env python3
"""Retrospective maximin development of one tail-adaptive PP."""
from __future__ import annotations
import json,sys,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];ADAPTERS=ROOT.parent/'ca-css-ncmapss';sys.path[:0]=[str(ROOT/'src'),str(ADAPTERS),str(ROOT/'experiments')]
from pae_boundary_realdata import DATASETS,prepare_dataset
from pae_shared_battery_nn import BatteryRepresentationScale,concatenate_rows
from boundary_quotient_pp_batteries import build_rows,full_part,score_by_dataset
from pp_extrapolation.boundary_quotient import fit_boundary_quotient_pp,predict_boundary_quotient
OUT=ROOT/'results/bq_tail_shape_robust_development_v1';SEEDS=(42,43,44,45,46)
CONFIGS=[{'late_bound_growth':g,'late_bound_power':p} for g in (1.5,2.,2.5,3.) for p in (1.,2.,3.)]
BASE=dict(width=64,alpha=1000.,learning_rate=1e-3,weight_decay=.01,residual_bound=2.)
def fit_config(config,seed,rows,scales):
 sel=fit_boundary_quotient_pp(rows['train'],rows['validation'],seed=seed,max_epochs=500,patience=70,**BASE,**config);epochs=max(sel.selection['selected_epoch'],1)
 fit=fit_boundary_quotient_pp(rows['full'],rows['full'],seed=seed,max_epochs=epochs,patience=10000,restore_best=False,**BASE,**config);pred=predict_boundary_quotient(fit,rows['source']);return pred,score_by_dataset(pred,rows['source'],scales),epochs,sel.selection['validation_dataset_macro_mse']
def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);started=time.perf_counter();scales={};audits={};parts={k:[] for k in ('train','validation','full','source')}
 for i,name in enumerate(DATASETS):
  split,audit=prepare_dataset(name);audits[name]=audit;scales[name]=BatteryRepresentationScale.fit(split['train'],audit['boundary'])
  for key,part in [('train',split['train']),('validation',split['val']),('full',full_part(split)),('source',split['source'])]:parts[key].append(build_rows(part,scales[name],i))
 rows={k:concatenate_rows(v) for k,v in parts.items()};screen=[]
 for config in CONFIGS:
  pred,metrics,epochs,vmse=fit_config(config,42,rows,scales);scores=[metrics[d]['pooled_r2'] for d in DATASETS];item={**config,'selected_epoch':epochs,'validation_mse':vmse,'metrics':metrics,'min_pooled_r2':float(min(scores)),'mean_pooled_r2':float(np.mean(scores))};screen.append(item);print('SCREEN',config,{d:round(metrics[d]['pooled_r2'],3) for d in DATASETS},flush=True)
 finalists=sorted(screen,key=lambda x:(x['min_pooled_r2'],x['mean_pooled_r2']),reverse=True)[:3];final=[]
 for item in finalists:
  config={k:item[k] for k in ('late_bound_growth','late_bound_power')};preds=[];runs=[]
  for seed in SEEDS:
   pred,metrics,epochs,vmse=fit_config(config,seed,rows,scales);preds.append(pred);runs.append({'seed':seed,'selected_epoch':epochs,'validation_mse':vmse,'metrics':metrics})
  ensemble=score_by_dataset(np.mean(preds,0),rows['source'],scales);scores=[ensemble[d]['pooled_r2'] for d in DATASETS];row={**config,'runs':runs,'ensemble':ensemble,'min_pooled_r2':float(min(scores)),'mean_pooled_r2':float(np.mean(scores))};final.append(row);print('FINAL',config,{d:round(ensemble[d]['pooled_r2'],3) for d in DATASETS},flush=True)
 chosen=max(final,key=lambda x:(x['min_pooled_r2'],x['mean_pooled_r2']));result={'status':'retrospective source-informed architecture development; requires frozen external confirmation','objective':'maximize minimum pooled R2 across Sunwoda, RWTH, and MICH, then mean pooled R2','model':'one frozen-affine BQ-PP with power-shaped late residual envelope','base_config':BASE,'screen':screen,'finalists':final,'selected':chosen,'global_envelope':'abs(yhat-y_affine) <= margin * B * (1 + growth * (1-clip(margin,0,1))^power)','data_audits':audits,'runtime_seconds':time.perf_counter()-started};(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
