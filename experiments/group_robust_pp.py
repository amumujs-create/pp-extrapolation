#!/usr/bin/env python3
"""Group-robust loss inside PP; no prediction mixture with GroupDRO."""
import json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments'),str(ROOT.parent/'ca-css-ncmapss')]
from pp_extrapolation import fit_latent_regime_pp,predict_latent_regime,regression_metrics,select_affine_initialization
from run_affine_tail_external_three import prepare_hust
from matr_batch2_confirmatory import endpoints,load_cells,make_rows

OUT=ROOT/'results/group_robust_pp_v1';SEEDS=range(42,47)
CONFIGS=[{'width':w,'eta':e,'trainable':False,'anchor':0.} for w in (24,64) for e in (0.,.01,.1,1.)]+[{'width':24,'eta':e,'trainable':True,'anchor':a} for e in (.1,1.) for a in (1.,10.)]

def batch2():
 c=load_cells();cut=float(np.quantile(endpoints(c,range(30)),.25));t0=make_rows(c,range(30),cut,train=True);b=float(t0['coordinate'].min());return make_rows(c,range(30),b,train=True),make_rows(c,range(30,39),b),make_rows(c,range(39,48),b)

def fit(parts,cfg,seed):
 a=select_affine_initialization(parts[0],parts[1]);return fit_latent_regime_pp(parts[0],parts[1],seed=seed,affine_selection=a,max_epochs=300,patience=70,width=cfg['width'],group_dro_eta=cfg['eta'],trainable_affine=cfg['trainable'],affine_anchor_weight=cfg['anchor'])

def run(name,parts):
 search=[]
 for cfg in CONFIGS:
  m=fit(parts,cfg,42);search.append({'config':cfg,'validation_mse':m.selection['validation_mse'],'epoch':m.selection['selected_epoch']});print(name,cfg,search[-1]['validation_mse'],flush=True)
 cfg=min(search,key=lambda x:x['validation_mse'])['config'];runs=[];pred=[]
 for seed in SEEDS:
  m=fit(parts,cfg,seed);p=predict_latent_regime(m,parts[2]['x']);pred.append(p);runs.append({'seed':seed,'selection':m.selection,'metrics':regression_metrics(parts[2]['y'],p,parts[2]['groups'])})
 return {'selected_config':cfg,'search':search,'runs':runs,'ensemble':regression_metrics(parts[2]['y'],np.mean(pred,0),parts[2]['groups'])}

def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);r={}
 for name,parts in [('hust',prepare_hust()[:3]),('matr_batch2',batch2())]:r[name]=run(name,parts);(OUT/'results.partial.json').write_text(json.dumps(r,indent=2)+'\n')
 (OUT/'results.json').write_text(json.dumps(r,indent=2)+'\n');print({k:v['ensemble']['pooled']['r2'] for k,v in r.items()})
if __name__=='__main__':main()
