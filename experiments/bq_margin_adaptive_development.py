#!/usr/bin/env python3
"""Post-hoc development screen for margin-adaptive PP contraction."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];PAE=ROOT.parent/'ca-css-ncmapss';sys.path[:0]=[str(ROOT/'src'),str(PAE),str(ROOT/'experiments')]
from pae_boundary_realdata import DATASETS,prepare_dataset
from pae_shared_battery_nn import BatteryRepresentationScale,concatenate_rows
from boundary_quotient_pp_batteries import build_rows,full_part,score_by_dataset
from pp_extrapolation.boundary_quotient import fit_boundary_quotient_pp,predict_boundary_quotient
OUT=ROOT/'results/bq_margin_adaptive_development_v1';SEEDS=(42,43,44,45,46);GROWTH=(0.,.5,1.,2.,4.)
CFG={'width':64,'alpha':1000.,'learning_rate':1e-3,'weight_decay':.01,'residual_bound':2.}
def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);scales={};parts={k:[] for k in ('train','validation','full','source')}
 for i,name in enumerate(DATASETS):
  split,audit=prepare_dataset(name);scales[name]=BatteryRepresentationScale.fit(split['train'],audit['boundary'])
  for key,part in [('train',split['train']),('validation',split['val']),('full',full_part(split)),('source',split['source'])]:parts[key].append(build_rows(part,scales[name],i))
 rows={k:concatenate_rows(v) for k,v in parts.items()};result={'status':'post-hoc architecture development; test not confirmatory','config':CFG,'growth':{}}
 for growth in GROWTH:
  preds=[];validation_preds=[];runs=[]
  for seed in SEEDS:
   sel=fit_boundary_quotient_pp(rows['train'],rows['validation'],seed=seed,max_epochs=500,patience=70,late_bound_growth=growth,**CFG);validation_preds.append(predict_boundary_quotient(sel,rows['validation']));epochs=max(sel.selection['selected_epoch'],1)
   fit=fit_boundary_quotient_pp(rows['full'],rows['full'],seed=seed,max_epochs=epochs,patience=10000,restore_best=False,late_bound_growth=growth,**CFG);p=predict_boundary_quotient(fit,rows['source']);preds.append(p);runs.append({'seed':seed,'selected_epoch':epochs,'validation_mse':sel.selection['validation_dataset_macro_mse']})
  vp=np.mean(validation_preds,0);validation_mse={name:float(np.mean((vp[rows['validation']['dataset']==i]-rows['validation']['y'][rows['validation']['dataset']==i])**2)) for i,name in enumerate(DATASETS)}
  metrics=score_by_dataset(np.mean(preds,0),rows['source'],scales);result['growth'][str(growth)]={'runs':runs,'mean_validation_mse':float(np.mean(list(validation_mse.values()))),'worst_validation_mse':float(max(validation_mse.values())),'validation_mse_by_dataset':validation_mse,'ensemble':metrics,'dataset_mean_pooled_r2':float(np.mean([metrics[d]['pooled_r2'] for d in DATASETS])),'dataset_macro_unit_r2':float(np.mean([metrics[d]['macro_unit_r2'] for d in DATASETS]))}
  print(growth,{d:round(metrics[d]['pooled_r2'],3) for d in DATASETS},'mean',round(result['growth'][str(growth)]['dataset_mean_pooled_r2'],3),flush=True)
 result['validation_selected_growth']=min(GROWTH,key=lambda g:result['growth'][str(g)]['mean_validation_mse'])
 result['minimax_validation_selected_growth']=min(GROWTH,key=lambda g:result['growth'][str(g)]['worst_validation_mse'])
 (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
