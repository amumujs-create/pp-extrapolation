#!/usr/bin/env python3
"""Train-label noise robustness of the frozen NASA causal multiscale PP."""
from pathlib import Path
import json,sys
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments'),str(ROOT.parent/'ca-css-ncmapss')]
from pp_extrapolation import fit_latent_regime_pp,predict_latent_regime,regression_metrics,select_affine_initialization
from nasa_causal_multiscale_pp import folds
OUT=ROOT/'results/nasa_causal_robustness_v1';SEEDS=range(42,47);NOISE=(0.,.05,.10,.20)
def main():
 torch.set_num_threads(2);OUT.mkdir(exist_ok=True);arch=json.load(open(ROOT/'results/nasa_causal_multiscale_pp_v1/results.json'));result={'status':'post-hoc robustness; selected architecture frozen from clean validation experiment','noise':'Gaussian train-target noise as fraction of train target SD; validation and test remain clean','levels':{}}
 for level in NOISE:
  pred=[[] for _ in SEEDS];ys=[];gs=[]
  for fi,row in enumerate(arch['folds']):
   cfg=row['selected'];f=folds(cfg['preset'])[fi]
   for j,seed in enumerate(SEEDS):
    tr={k:(v.copy() if hasattr(v,'copy') else v) for k,v in f['train'].items()};rng=np.random.default_rng(seed+int(level*10000));tr['y']=np.clip(tr['y']+rng.normal(0,level*np.std(tr['y']),len(tr['y'])),0,None).astype(np.float32);a=select_affine_initialization(tr,f['validation']);q=fit_latent_regime_pp(tr,f['validation'],seed=seed,affine_selection=a,max_epochs=400,patience=80,separation_weight=cfg['separation'],gate_weight=0.,width=32);pred[j].append(predict_latent_regime(q,f['test']['x']))
   ys.append(f['test']['y']);gs.append(f['test']['groups'])
  y=np.concatenate(ys);g=np.concatenate(gs);P=np.asarray([np.concatenate(x) for x in pred]);result['levels'][str(level)]={'ensemble':regression_metrics(y,P.mean(0),g),'per_seed':[regression_metrics(y,p,g) for p in P]};print(level,result['levels'][str(level)]['ensemble']['pooled']['r2'],flush=True)
 (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
