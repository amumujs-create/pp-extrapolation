#!/usr/bin/env python3
"""Unified BQ-PP with validation-selected structured rate-history dropout."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, torch

ROOT=Path(__file__).resolve().parents[1]; ADAPTERS=ROOT.parent/'ca-css-ncmapss'
sys.path[:0]=[str(ROOT/'src'),str(ADAPTERS),str(ROOT/'experiments')]
from pae_boundary_realdata import DATASETS,prepare_dataset
from pae_shared_battery_nn import BatteryRepresentationScale,concatenate_rows
from boundary_quotient_pp_batteries import build_rows,full_part,score_by_dataset
from pp_extrapolation.boundary_quotient import fit_boundary_quotient_pp,predict_boundary_quotient

OUT=ROOT/'results/bq_rate_dropout_pp_v1'; SEEDS=(42,43,44,45,46)
RATES=(0.,.05,.1,.2,.3,.4,.5,.6)
BASE=dict(width=64,alpha=1000.,learning_rate=1e-3,weight_decay=.01,residual_bound=2.)

def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);started=time.perf_counter();scales={};audits={};parts={k:[] for k in ('train','validation','full','source')}
 for i,name in enumerate(DATASETS):
  split,audit=prepare_dataset(name);audits[name]=audit;scales[name]=BatteryRepresentationScale.fit(split['train'],audit['boundary'])
  for key,part in [('train',split['train']),('validation',split['val']),('full',full_part(split)),('source',split['source'])]:parts[key].append(build_rows(part,scales[name],i))
 rows={k:concatenate_rows(v) for k,v in parts.items()};search=[]
 for rate in RATES:
  fit=fit_boundary_quotient_pp(rows['train'],rows['validation'],seed=42,max_epochs=500,patience=70,rate_feature_dropout=rate,**BASE)
  search.append({'rate_feature_dropout':rate,**fit.selection});print('SEARCH',rate,fit.selection['validation_dataset_macro_mse'],flush=True)
 chosen=min(search,key=lambda r:r['validation_dataset_macro_mse']);rate=chosen['rate_feature_dropout'];preds=[];runs=[]
 for seed in SEEDS:
  sel=fit_boundary_quotient_pp(rows['train'],rows['validation'],seed=seed,max_epochs=600,patience=80,rate_feature_dropout=rate,**BASE);epochs=max(sel.selection['selected_epoch'],1)
  fit=fit_boundary_quotient_pp(rows['full'],rows['full'],seed=seed,max_epochs=epochs,patience=10000,restore_best=False,rate_feature_dropout=rate,**BASE);p=predict_boundary_quotient(fit,rows['source']);preds.append(p);metrics=score_by_dataset(p,rows['source'],scales);runs.append({'seed':seed,'selected_epoch':epochs,'validation_mse':sel.selection['validation_dataset_macro_mse'],'metrics':metrics});print('SEED',seed,{d:round(metrics[d]['pooled_r2'],3) for d in DATASETS},flush=True)
 ensemble=score_by_dataset(np.mean(preds,0),rows['source'],scales);result={'status':'retrospective unified PP development; source labels excluded from selection','model':'frozen affine bounded BQ-PP with structured rate-history dropout','selection':'one shared dropout selected by validation dataset-macro normalized MSE','base_config':BASE,'selected_rate_feature_dropout':rate,'search':search,'runs':runs,'ensemble':ensemble,'dataset_mean_pooled_r2':float(np.mean([ensemble[d]['pooled_r2'] for d in DATASETS])),'dataset_macro_unit_r2':float(np.mean([ensemble[d]['macro_unit_r2'] for d in DATASETS])),'data_audits':audits,'runtime_seconds':time.perf_counter()-started}
 (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(OUT/'predictions.npz',prediction=np.asarray(preds),y=rows['source']['y'],units=rows['source']['units'],dataset=rows['source']['dataset']);print('ENSEMBLE',{d:ensemble[d]['pooled_r2'] for d in DATASETS},flush=True)
if __name__=='__main__':main()
