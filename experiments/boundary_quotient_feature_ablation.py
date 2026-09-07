#!/usr/bin/env python3
"""Feature ablation for boundary-quotient PP relationship-shift mechanism."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch

ROOT=Path(__file__).resolve().parents[1]; PAE=ROOT.parent/'ca-css-ncmapss'
sys.path[:0]=[str(ROOT/'src'),str(PAE),str(ROOT/'experiments')]
from pae_boundary_realdata import DATASETS,prepare_dataset
from pae_shared_battery_nn import BatteryRepresentationScale,rows_from_part,concatenate_rows
from boundary_quotient_pp_batteries import build_rows,full_part,score_by_dataset
from pp_extrapolation.boundary_quotient import fit_boundary_quotient_pp,predict_boundary_quotient

OUT=ROOT/'results/boundary_quotient_feature_ablation_v1';SEEDS=(42,43,44,45,46)
FEATURES={'margin_current':(0,), 'margin_history':tuple(range(5)),
          'margin_history_cycle':(0,1,2,3,4,10), 'full_rate_history':tuple(range(11))}
CONFIG={'width':64,'alpha':1000.,'weight_decay':.01,'residual_bound':2.}

def subset(rows,columns):
 return {**rows,'x':rows['x'][:,columns]}

def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);scales={};parts={k:[] for k in ('train','validation','full','source')}
 for i,name in enumerate(DATASETS):
  split,audit=prepare_dataset(name);scales[name]=BatteryRepresentationScale.fit(split['train'],audit['boundary'])
  parts['train'].append(build_rows(split['train'],scales[name],i));parts['validation'].append(build_rows(split['val'],scales[name],i));parts['full'].append(build_rows(full_part(split),scales[name],i));parts['source'].append(build_rows(split['source'],scales[name],i))
 rows={k:concatenate_rows(v) for k,v in parts.items()};result={'status':'retrospective fixed-config mechanism ablation','config':CONFIG,'features':{}}
 source_ensembles={}
 for label,columns in FEATURES.items():
  predictions=[];validation_predictions=[];runs=[]
  for seed in SEEDS:
   selected=fit_boundary_quotient_pp(subset(rows['train'],columns),subset(rows['validation'],columns),seed=seed,max_epochs=500,patience=70,**CONFIG)
   validation_predictions.append(predict_boundary_quotient(selected,subset(rows['validation'],columns)))
   epochs=max(selected.selection['selected_epoch'],1)
   fit=fit_boundary_quotient_pp(subset(rows['full'],columns),subset(rows['full'],columns),seed=seed,max_epochs=epochs,patience=10000,restore_best=False,**CONFIG)
   p=predict_boundary_quotient(fit,subset(rows['source'],columns));predictions.append(p);runs.append({'seed':seed,'selected_epoch':selected.selection['selected_epoch']})
  source_ensembles[label]=np.mean(predictions,0);validation_ensemble=np.mean(validation_predictions,0)
  validation_mse={name:float(np.mean((validation_ensemble[rows['validation']['dataset']==i]-rows['validation']['y'][rows['validation']['dataset']==i])**2)) for i,name in enumerate(DATASETS)}
  metrics=score_by_dataset(source_ensembles[label],rows['source'],scales);result['features'][label]={'columns':columns,'runs':runs,'validation_normalized_mse':validation_mse,'ensemble':metrics}
  print(label,{d:round(metrics[d]['pooled_r2'],3) for d in DATASETS},flush=True)
 choices={name:min(FEATURES,key=lambda label:result['features'][label]['validation_normalized_mse'][name]) for name in DATASETS}
 routed=np.zeros(len(rows['source']['y']))
 for i,name in enumerate(DATASETS):
  mask=rows['source']['dataset']==i;routed[mask]=source_ensembles[choices[name]][mask]
 result['validation_selected_route']=choices
 result['routed_ensemble']=score_by_dataset(routed,rows['source'],scales)
 print('routed',choices,{d:round(result['routed_ensemble'][d]['pooled_r2'],3) for d in DATASETS})
 (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
