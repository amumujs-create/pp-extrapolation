#!/usr/bin/env python3
"""Frozen replay and robust aggregation of the unified support-gated PP."""
from __future__ import annotations
import json,sys,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];ADAPTERS=ROOT.parent/'ca-css-ncmapss';sys.path[:0]=[str(ROOT/'src'),str(ADAPTERS),str(ROOT/'experiments')]
from pae_boundary_realdata import DATASETS,prepare_dataset
from pae_shared_battery_nn import BatteryRepresentationScale,concatenate_rows
from boundary_quotient_pp_batteries import build_rows,full_part,score_by_dataset
from pp_extrapolation.boundary_quotient import fit_boundary_quotient_pp,predict_boundary_quotient
OUT=ROOT/'results/bq_support_gate_final_replay_v1';SEEDS=(42,43,44,45,46)
CONFIG=dict(width=64,alpha=1000.,learning_rate=1e-3,weight_decay=.01,residual_bound=2.,late_bound_growth=3.,late_bound_power=1.,support_gate_feature=2,support_gate_threshold=.5,support_gate_temperature=.5)
def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);start=time.perf_counter();scales={};audits={};parts={k:[] for k in ('train','validation','full','source')}
 for i,n in enumerate(DATASETS):
  s,a=prepare_dataset(n);audits[n]=a;scales[n]=BatteryRepresentationScale.fit(s['train'],a['boundary'])
  for k,p in [('train',s['train']),('validation',s['val']),('full',full_part(s)),('source',s['source'])]:parts[k].append(build_rows(p,scales[n],i))
 rows={k:concatenate_rows(v) for k,v in parts.items()};preds=[];runs=[]
 for seed in SEEDS:
  sel=fit_boundary_quotient_pp(rows['train'],rows['validation'],seed=seed,max_epochs=500,patience=70,**CONFIG);ep=max(sel.selection['selected_epoch'],1);fit=fit_boundary_quotient_pp(rows['full'],rows['full'],seed=seed,max_epochs=ep,patience=10000,restore_best=False,**CONFIG);p=predict_boundary_quotient(fit,rows['source']);preds.append(p);runs.append({'seed':seed,'selected_epoch':ep,'validation_mse':sel.selection['validation_dataset_macro_mse'],'metrics':score_by_dataset(p,rows['source'],scales)})
 matrix=np.asarray(preds);ordered=np.sort(matrix,axis=0);aggregates={'mean':matrix.mean(0),'median':np.median(matrix,axis=0),'trimmed_mean':ordered[1:-1].mean(0)};metrics={k:score_by_dataset(v,rows['source'],scales) for k,v in aggregates.items()};result={'status':'frozen replay of retrospectively developed PP; requires external confirmation','model':'one support-heterogeneity-gated BQ-PP architecture','config':CONFIG,'runs':runs,'aggregates':metrics,'aggregation_note':'all aggregations use five fits of the same PP architecture','data_audits':audits,'runtime_seconds':time.perf_counter()-start};(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(OUT/'predictions.npz',prediction=matrix,mean=aggregates['mean'],median=aggregates['median'],trimmed_mean=aggregates['trimmed_mean'],y=rows['source']['y'],units=rows['source']['units'],dataset=rows['source']['dataset']);print({a:{d:round(metrics[a][d]['pooled_r2'],3) for d in DATASETS} for a in metrics},flush=True)
if __name__=='__main__':main()
