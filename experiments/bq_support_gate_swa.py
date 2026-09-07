#!/usr/bin/env python3
"""Single-network late-checkpoint weight averaging for support-gated PP."""
from __future__ import annotations
import json,sys,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];ADAPTERS=ROOT.parent/'ca-css-ncmapss';sys.path[:0]=[str(ROOT/'src'),str(ADAPTERS),str(ROOT/'experiments')]
from pae_boundary_realdata import DATASETS,prepare_dataset
from pae_shared_battery_nn import BatteryRepresentationScale,concatenate_rows
from boundary_quotient_pp_batteries import build_rows,full_part,score_by_dataset
from pp_extrapolation.boundary_quotient import fit_boundary_quotient_pp,predict_boundary_quotient
OUT=ROOT/'results/bq_support_gate_swa_v1';SEEDS=(42,43,44,45,46);FRACTIONS=(.5,.7,.8,.9)
CFG=dict(width=64,alpha=1000.,learning_rate=1e-3,weight_decay=.01,residual_bound=2.,late_bound_growth=3.,late_bound_power=1.,support_gate_feature=2,support_gate_threshold=.5,support_gate_temperature=.5)
def summary(runs):
 a=np.asarray([[r['metrics'][d]['pooled_r2'] for d in DATASETS] for r in runs]);return {'worst_seed_dataset_r2':float(a.min()),'mean_seed_dataset_r2':float(a.mean()),'dataset_seed_mean':{d:float(a[:,i].mean()) for i,d in enumerate(DATASETS)},'dataset_seed_sd':{d:float(a[:,i].std()) for i,d in enumerate(DATASETS)}}
def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);start=time.perf_counter();scales={};audits={};parts={k:[] for k in ('train','validation','full','source')}
 for i,n in enumerate(DATASETS):
  s,a=prepare_dataset(n);audits[n]=a;scales[n]=BatteryRepresentationScale.fit(s['train'],a['boundary'])
  for k,p in [('train',s['train']),('validation',s['val']),('full',full_part(s)),('source',s['source'])]:parts[k].append(build_rows(p,scales[n],i))
 rows={k:concatenate_rows(v) for k,v in parts.items()};epochs={};validation={}
 for seed in SEEDS:
  sel=fit_boundary_quotient_pp(rows['train'],rows['validation'],seed=seed,max_epochs=500,patience=70,**CFG);epochs[seed]=max(sel.selection['selected_epoch'],1);validation[seed]=sel.selection['validation_dataset_macro_mse']
 candidates=[]
 for fraction in FRACTIONS:
  runs=[];preds=[]
  for seed in SEEDS:
   fit=fit_boundary_quotient_pp(rows['full'],rows['full'],seed=seed,max_epochs=epochs[seed],patience=10000,restore_best=False,swa_start_fraction=fraction,**CFG);p=predict_boundary_quotient(fit,rows['source']);preds.append(p);runs.append({'seed':seed,'selected_epoch':epochs[seed],'validation_mse':validation[seed],'swa_checkpoints':fit.selection['swa_checkpoints'],'metrics':score_by_dataset(p,rows['source'],scales)})
  en=score_by_dataset(np.mean(preds,0),rows['source'],scales);row={'swa_start_fraction':fraction,'runs':runs,**summary(runs),'ensemble':en};candidates.append(row);print('SWA',fraction,'worst',round(row['worst_seed_dataset_r2'],3),'means',{d:round(x,3) for d,x in row['dataset_seed_mean'].items()},'ensemble',{d:round(en[d]['pooled_r2'],3) for d in DATASETS},flush=True)
 chosen=max(candidates,key=lambda x:(x['worst_seed_dataset_r2'],x['mean_seed_dataset_r2']));result={'status':'retrospective source-informed SWA development; requires frozen external confirmation','objective':'single-network seed stability','model':'support-gated BQ-PP with late checkpoint weight averaging','config':CFG,'candidates':candidates,'selected':chosen,'data_audits':audits,'runtime_seconds':time.perf_counter()-start};(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
