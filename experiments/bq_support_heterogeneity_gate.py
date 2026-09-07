#!/usr/bin/env python3
"""One PP with a support-calibrated history-heterogeneity envelope gate."""
from __future__ import annotations
import json,sys,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];ADAPTERS=ROOT.parent/'ca-css-ncmapss';sys.path[:0]=[str(ROOT/'src'),str(ADAPTERS),str(ROOT/'experiments')]
from pae_boundary_realdata import DATASETS,prepare_dataset
from pae_shared_battery_nn import BatteryRepresentationScale,concatenate_rows
from boundary_quotient_pp_batteries import build_rows,full_part,score_by_dataset
from pp_extrapolation.boundary_quotient import fit_boundary_quotient_pp,predict_boundary_quotient
OUT=ROOT/'results/bq_support_heterogeneity_gate_v1';SEEDS=(42,43,44,45,46)
CONFIGS=[{'late_bound_growth':g,'support_gate_threshold':t,'support_gate_temperature':temp} for g in (2.,3.,4.) for t in (.5,1.,1.5) for temp in (.25,.5)]
BASE=dict(width=64,alpha=1000.,learning_rate=1e-3,weight_decay=.01,residual_bound=2.,late_bound_power=1.,support_gate_feature=2)
def fit_config(c,seed,rows,scales):
 sel=fit_boundary_quotient_pp(rows['train'],rows['validation'],seed=seed,max_epochs=500,patience=70,**BASE,**c);ep=max(sel.selection['selected_epoch'],1);fit=fit_boundary_quotient_pp(rows['full'],rows['full'],seed=seed,max_epochs=ep,patience=10000,restore_best=False,**BASE,**c);p=predict_boundary_quotient(fit,rows['source']);return p,score_by_dataset(p,rows['source'],scales),ep,sel.selection['validation_dataset_macro_mse']
def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);started=time.perf_counter();scales={};audits={};parts={k:[] for k in ('train','validation','full','source')}
 for i,n in enumerate(DATASETS):
  s,a=prepare_dataset(n);audits[n]=a;scales[n]=BatteryRepresentationScale.fit(s['train'],a['boundary'])
  for k,p in [('train',s['train']),('validation',s['val']),('full',full_part(s)),('source',s['source'])]:parts[k].append(build_rows(p,scales[n],i))
 rows={k:concatenate_rows(v) for k,v in parts.items()};screen=[]
 for c in CONFIGS:
  p,m,e,v=fit_config(c,42,rows,scales);s=[m[d]['pooled_r2'] for d in DATASETS];row={**c,'selected_epoch':e,'validation_mse':v,'metrics':m,'min_pooled_r2':float(min(s)),'mean_pooled_r2':float(np.mean(s))};screen.append(row);print('SCREEN',c,{d:round(m[d]['pooled_r2'],3) for d in DATASETS},flush=True)
 finalists=sorted(screen,key=lambda x:(x['min_pooled_r2'],x['mean_pooled_r2']),reverse=True)[:3];final=[]
 for x in finalists:
  c={k:x[k] for k in ('late_bound_growth','support_gate_threshold','support_gate_temperature')};ps=[];runs=[]
  for seed in SEEDS:
   p,m,e,v=fit_config(c,seed,rows,scales);ps.append(p);runs.append({'seed':seed,'selected_epoch':e,'validation_mse':v,'metrics':m})
  en=score_by_dataset(np.mean(ps,0),rows['source'],scales);s=[en[d]['pooled_r2'] for d in DATASETS];row={**c,'runs':runs,'ensemble':en,'min_pooled_r2':float(min(s)),'mean_pooled_r2':float(np.mean(s))};final.append(row);print('FINAL',c,{d:round(en[d]['pooled_r2'],3) for d in DATASETS},flush=True)
 chosen=max(final,key=lambda x:(x['min_pooled_r2'],x['mean_pooled_r2']));result={'status':'retrospective source-informed PP development; freeze before external confirmation','objective':'one PP maximizing minimum dataset pooled R2','model':'support-heterogeneity-gated boundary-quotient PP','mechanism':'expand residual envelope only when standardized causal-window margin variability exceeds training support','base_config':BASE,'screen':screen,'finalists':final,'selected':chosen,'global_envelope':'abs(yhat-y_affine) <= margin * B * (1 + growth)','data_audits':audits,'runtime_seconds':time.perf_counter()-started};(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
