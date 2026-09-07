#!/usr/bin/env python3
"""Seed-count scaling audit for the frozen support-gated PP architecture."""
from __future__ import annotations
import json,sys,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];ADAPTERS=ROOT.parent/'ca-css-ncmapss';sys.path[:0]=[str(ROOT/'src'),str(ADAPTERS),str(ROOT/'experiments')]
from pae_boundary_realdata import DATASETS,prepare_dataset
from pae_shared_battery_nn import BatteryRepresentationScale,concatenate_rows
from boundary_quotient_pp_batteries import build_rows,full_part,score_by_dataset
from pp_extrapolation.boundary_quotient import fit_boundary_quotient_pp,predict_boundary_quotient
OUT=ROOT/'results/bq_support_gate_ensemble_scaling_v1';NEW_SEEDS=tuple(range(47,57))
CFG=dict(width=64,alpha=1000.,learning_rate=1e-3,weight_decay=.01,residual_bound=2.,late_bound_growth=3.,late_bound_power=1.,support_gate_feature=2,support_gate_threshold=.5,support_gate_temperature=.5)
def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);start=time.perf_counter();scales={};audits={};parts={k:[] for k in ('train','validation','full','source')}
 for i,n in enumerate(DATASETS):
  s,a=prepare_dataset(n);audits[n]=a;scales[n]=BatteryRepresentationScale.fit(s['train'],a['boundary'])
  for k,p in [('train',s['train']),('validation',s['val']),('full',full_part(s)),('source',s['source'])]:parts[k].append(build_rows(p,scales[n],i))
 rows={k:concatenate_rows(v) for k,v in parts.items()};old=np.load(ROOT/'results/bq_support_gate_final_replay_v1/predictions.npz')['prediction'];new=[];runs=[]
 for seed in NEW_SEEDS:
  sel=fit_boundary_quotient_pp(rows['train'],rows['validation'],seed=seed,max_epochs=500,patience=70,**CFG);ep=max(sel.selection['selected_epoch'],1);fit=fit_boundary_quotient_pp(rows['full'],rows['full'],seed=seed,max_epochs=ep,patience=10000,restore_best=False,**CFG);p=predict_boundary_quotient(fit,rows['source']);new.append(p);m=score_by_dataset(p,rows['source'],scales);runs.append({'seed':seed,'selected_epoch':ep,'validation_mse':sel.selection['validation_dataset_macro_mse'],'metrics':m});print(seed,{d:round(m[d]['pooled_r2'],3) for d in DATASETS},flush=True)
 matrix=np.concatenate([old,np.asarray(new)],axis=0);scaling={}
 for count in (5,10,15):
  m=score_by_dataset(matrix[:count].mean(0),rows['source'],scales);scaling[str(count)]={'metrics':m,'minimum_dataset_r2':float(min(m[d]['pooled_r2'] for d in DATASETS)),'mean_dataset_r2':float(np.mean([m[d]['pooled_r2'] for d in DATASETS]))};print('COUNT',count,{d:round(m[d]['pooled_r2'],3) for d in DATASETS},flush=True)
 result={'status':'frozen architecture seed-count scaling audit','config':CFG,'seed_order':list(range(42,57)),'new_runs':runs,'scaling':scaling,'data_audits':audits,'runtime_seconds':time.perf_counter()-start};(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(OUT/'predictions.npz',prediction=matrix,y=rows['source']['y'],units=rows['source']['units'],dataset=rows['source']['dataset'])
if __name__=='__main__':main()
