#!/usr/bin/env python3
"""HUST: softly train PP affine path together with group-robust residual."""
import json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments'),str(ROOT.parent/'ca-css-ncmapss')]
from pp_extrapolation import fit_pp,predict,regression_metrics,select_affine_initialization,select_group_loo_calibrator
from run_affine_tail_external_three import prepare_hust
OUT=ROOT/'results/hust_trainable_affine_group_pp_v1';SEEDS=range(42,47);GRID=[{'eta':e,'anchor':a} for e in (0.,.001,.01,.1) for a in (.1,1.,10.)]
def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);parts=prepare_hust()[:3];aff=select_affine_initialization(parts[0],parts[1]);search=[]
 def fit(c,s):return fit_pp(parts[0],parts[1],seed=s,affine_selection=aff,max_epochs=300,patience=70,width=64,group_dro_eta=c['eta'],affine_anchor_weight=c['anchor'])
 for c in GRID:
  f=fit(c,42);search.append({'config':c,'validation_mse':f.selection['best_validation_mse'],'epoch':f.selection['selected_epoch']});print(c,search[-1],flush=True)
 cfg=min(search,key=lambda x:x['validation_mse'])['config'];raw=[];cal=[];runs=[]
 for s in SEEDS:
  f=fit(cfg,s);vp=predict(f,parts[1]['x']);tp=predict(f,parts[2]['x']);c,_=select_group_loo_calibrator(parts[1]['y'],vp,parts[1]['groups'],upper=f.target_scale);raw.append(tp);cal.append(c.predict(tp,upper=f.target_scale));runs.append({'seed':s,'selection':f.selection,'calibrator':c.kind})
 r={'selected':cfg,'search':search,'runs':runs,'raw':regression_metrics(parts[2]['y'],np.mean(raw,0),parts[2]['groups']),'calibrated':regression_metrics(parts[2]['y'],np.mean(cal,0),parts[2]['groups'])};(OUT/'results.json').write_text(json.dumps(r,indent=2)+'\n');print(r['raw']['pooled']['r2'],r['calibrated']['pooled']['r2'])
if __name__=='__main__':main()
