#!/usr/bin/env python3
"""GroupDRO applied only to PP's bounded neural residual training."""
import json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments'),str(ROOT.parent/'ca-css-ncmapss')]
from pp_extrapolation import fit_pp,predict,regression_metrics,select_affine_initialization,select_group_loo_calibrator
from run_affine_tail_external_three import prepare_hust
from group_robust_pp import batch2
OUT=ROOT/'results/group_robust_residual_pp_v1';SEEDS=range(42,47)
GRID=[{'width':w,'eta':e,'decay':d} for w in (32,64) for e in (0.,.001,.01,.1,1.) for d in (0.,.1)]
def train(parts,cfg,seed):
 a=select_affine_initialization(parts[0],parts[1]);return fit_pp(parts[0],parts[1],seed=seed,affine_selection=a,max_epochs=300,patience=70,width=cfg['width'],group_dro_eta=cfg['eta'],residual_decay=cfg['decay'])
def run(name,parts):
 search=[]
 for c in GRID:
  f=train(parts,c,42);search.append({'config':c,'validation_mse':f.selection['best_validation_mse'],'epoch':f.selection['selected_epoch']});print(name,c,search[-1]['validation_mse'],flush=True)
 cfg=min(search,key=lambda x:x['validation_mse'])['config'];raw=[];cal=[];runs=[]
 for s in SEEDS:
  f=train(parts,cfg,s);vp=predict(f,parts[1]['x']);tp=predict(f,parts[2]['x']);c,_=select_group_loo_calibrator(parts[1]['y'],vp,parts[1]['groups'],upper=f.target_scale);raw.append(tp);cal.append(c.predict(tp,upper=f.target_scale));runs.append({'seed':s,'selection':f.selection,'calibrator':c.kind})
 return {'selected':cfg,'search':search,'runs':runs,'raw':regression_metrics(parts[2]['y'],np.mean(raw,0),parts[2]['groups']),'calibrated':regression_metrics(parts[2]['y'],np.mean(cal,0),parts[2]['groups'])}
def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);r={}
 for n,p in [('hust',prepare_hust()[:3]),('matr_batch2',batch2())]:r[n]=run(n,p);(OUT/'results.partial.json').write_text(json.dumps(r,indent=2)+'\n')
 (OUT/'results.json').write_text(json.dumps(r,indent=2)+'\n');print({n:(x['raw']['pooled']['r2'],x['calibrated']['pooled']['r2']) for n,x in r.items()})
if __name__=='__main__':main()
