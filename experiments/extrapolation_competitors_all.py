#!/usr/bin/env python3
"""Run extrapolation-oriented competitors across every generic PP dataset."""
from __future__ import annotations
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments'),str(ROOT/'.benchmark_deps'),str(ROOT.parent/'ca-css-ncmapss')]
import torch
import extrapolation_competitors_matr as bench
from matr_2019_latent_confirmatory import load_cells,make_rows
from pp_extrapolation import select_affine_initialization
from pae_boundary_realdata import prepare_dataset
from run_affine_tail_external_three import prepare_hust,prepare_virkler
from distance_uncertainty_pp import prepare_battery

OUT=ROOT/'results/extrapolation_competitors_all_v1';bench.OUT=OUT

def matr():
 audit=json.load(open(ROOT/'results/matr_2019_latent_confirmatory/results.json'))['pretest'];cells={c['index']:c for c in load_cells()[0]}
 return tuple(make_rows([cells[i] for i in audit['split_indices'][n]],audit['actual_boundary'],train=n=='train') for n in ('train','validation','test'))

def datasets():
 out={'hust':prepare_hust()[:3],'virkler':prepare_virkler()[:3],'matr':matr()}
 for name in ('sunwoda','rwth','mich'):
  raw,_=prepare_dataset(name);out[name]=prepare_battery(raw)
 from matr_batch2_confirmatory import endpoints,load_cells as load_batch2,make_rows as batch2_rows
 import numpy as np
 cells=load_batch2();cut=float(np.quantile(endpoints(cells,range(30)),.25));tr0=batch2_rows(cells,range(30),cut,train=True);boundary=float(tr0['coordinate'].min())
 out['matr_batch2']=(batch2_rows(cells,range(30),boundary,train=True),batch2_rows(cells,range(30,39),boundary),batch2_rows(cells,range(39,48),boundary))
 from xjtu_untouched import build_cache,windows
 raw=build_cache();out['xjtu']=(windows(raw,'37.5Hz11kN'),windows(raw,'35Hz12kN'),windows(raw,'40Hz10kN'))
 from femto_pp_prospective import split as femto_split
 from femto_bearing_loader import load_femto_phm2012,FEATURE_COLS
 frame,groups=load_femto_phm2012(root=ROOT/'data/femto/raw',file_stride=5);end=frame[frame.unit.isin(groups['test'])].sort_values('cycle').groupby('unit',as_index=False).tail(1)
 out['femto']=(femto_split(frame,groups['train']),femto_split(frame,groups['val']),{'x':end[FEATURE_COLS].to_numpy(np.float32),'y':end.RUL.to_numpy(np.float32),'groups':end.bearing.to_numpy()})
 from milling_locked_transfer import subset
 from nasa_milling_causal import prepare_causal_milling
 raw,_=prepare_causal_milling();cut=float(np.quantile(raw['train']['health'],.60));out['milling']=(subset(raw['train'],raw['train']['health']<=cut),subset(raw['validation'],raw['validation']['health']>cut),subset(raw['source'],raw['source']['health']>cut))
 return out

def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);result={'protocol':'fixed existing PP splits; validation-only selection; 5 seeds','datasets':{}};started=time.time()
 existing={}
 partial=OUT/'results.partial.json'
 if partial.exists():existing=json.load(open(partial)).get('datasets',{})
 result['datasets'].update(existing)
 for name,parts in datasets().items():
  if name in result['datasets']:continue
  aff=select_affine_initialization(parts[0],parts[1]);row={}
  for kind in ('vrex','groupdro','monotone'):
   row[kind]=bench.neural_model(kind,parts,aff,save_prefix=name+'_')
  row['linear_rff']=bench.linear_rff(parts,aff,save_prefix=name+'_');result['datasets'][name]=row
  (OUT/'results.partial.json').write_text(json.dumps(result,indent=2)+'\n');print('DATASET_DONE',name,{k:round(v['ensemble']['pooled']['r2'],4) for k,v in row.items()},flush=True)
 result['runtime_seconds']=time.time()-started;(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
