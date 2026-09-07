#!/usr/bin/env python3
"""Matched frozen-vs-trainable affine ablation for boundary-quotient PP."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];PAE=ROOT.parent/'ca-css-ncmapss';sys.path[:0]=[str(ROOT/'src'),str(PAE),str(ROOT/'experiments')]
from pae_boundary_realdata import DATASETS,prepare_dataset
from pae_shared_battery_nn import BatteryRepresentationScale,concatenate_rows
from boundary_quotient_pp_batteries import build_rows,full_part,score_by_dataset
from pp_extrapolation.boundary_quotient import fit_boundary_quotient_pp,predict_boundary_quotient
OUT=ROOT/'results/boundary_quotient_affine_ablation_v1';SEEDS=(42,43,44,45,46);CFG={'width':64,'alpha':1000.,'weight_decay':.01,'residual_bound':2.}
def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);scales={};parts={k:[] for k in ('train','validation','full','source')}
 for i,name in enumerate(DATASETS):
  split,audit=prepare_dataset(name);scales[name]=BatteryRepresentationScale.fit(split['train'],audit['boundary'])
  for key,part in [('train',split['train']),('validation',split['val']),('full',full_part(split)),('source',split['source'])]:parts[key].append(build_rows(part,scales[name],i))
 rows={k:concatenate_rows(v) for k,v in parts.items()};result={'status':'retrospective matched affine-path ablation','config':CFG,'arms':{}}
 for arm,trainable in [('frozen_affine',False),('trainable_affine',True)]:
  preds=[];runs=[]
  for seed in SEEDS:
   sel=fit_boundary_quotient_pp(rows['train'],rows['validation'],seed=seed,max_epochs=500,patience=70,trainable_affine=trainable,**CFG);epochs=max(sel.selection['selected_epoch'],1)
   fit=fit_boundary_quotient_pp(rows['full'],rows['full'],seed=seed,max_epochs=epochs,patience=10000,restore_best=False,trainable_affine=trainable,**CFG)
   p=predict_boundary_quotient(fit,rows['source']);preds.append(p);runs.append({'seed':seed,'selected_epoch':sel.selection['selected_epoch'],'metrics':score_by_dataset(p,rows['source'],scales)})
  ensemble=score_by_dataset(np.mean(preds,0),rows['source'],scales);result['arms'][arm]={'runs':runs,'ensemble':ensemble}
  print(arm,{d:round(ensemble[d]['pooled_r2'],3) for d in DATASETS},flush=True)
 (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
