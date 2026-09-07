#!/usr/bin/env python3
"""Validation-selected dual-view PP: scalar regime tail + causal-history regime tail."""
from pathlib import Path
import json,sys
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments'),str(ROOT.parent/'ca-css-ncmapss')]
from pp_extrapolation import fit_regime_spline_pp,predict_regime_spline,fit_latent_regime_pp,predict_latent_regime,regression_metrics,select_affine_initialization
from run_affine_tail_external_nasa_health_v2 import prepare_folds
from nasa_causal_multiscale_pp import folds as temporal_folds
OUT=ROOT/'results/nasa_dual_view_pp_v1';SEEDS=range(42,47);WEIGHTS=np.linspace(0,1,11)
def main():
 torch.set_num_threads(2);OUT.mkdir(exist_ok=True);scalar=prepare_folds()[0];sa=json.load(open(ROOT/'results/nasa_regime_spline_tuning_v1/results.json'))['folds'];ta=json.load(open(ROOT/'results/nasa_causal_multiscale_pp_v1/results.json'))['folds'];tf={p:temporal_folds(p) for p in ('short','multiscale','moments')};pred=[[] for _ in SEEDS];ys=[];gs=[];audit=[]
 for i,sf in enumerate(scalar):
  cfg=sa[i]['selected'];preset=ta[i]['selected']['preset'];sep=ta[i]['selected']['separation'];hf=tf[preset][i];asf=select_affine_initialization(sf['train'],sf['validation']);ah=select_affine_initialization(hf['train'],hf['validation']);sv=[];st=[];hv=[];ht=[]
  for seed in SEEDS:
   a=fit_regime_spline_pp(sf['train'],sf['validation'],seed=seed,affine_selection=asf,max_epochs=400,patience=80,monotone=cfg[0],jacobian_weight=cfg[1],jacobian_ray_multiplier=cfg[2]);b=fit_latent_regime_pp(hf['train'],hf['validation'],seed=seed,affine_selection=ah,max_epochs=400,patience=80,separation_weight=sep,gate_weight=0.,width=32)
   sv.append(predict_regime_spline(a,sf['validation']['x']));st.append(predict_regime_spline(a,sf['test']['x']));hv.append(predict_latent_regime(b,hf['validation']['x']));ht.append(predict_latent_regime(b,hf['test']['x']))
  sv=np.asarray(sv);st=np.asarray(st);hv=np.asarray(hv);ht=np.asarray(ht);scores=[float(np.mean(((1-w)*sv.mean(0)+w*hv.mean(0)-sf['validation']['y'])**2)) for w in WEIGHTS];w=float(WEIGHTS[int(np.argmin(scores))])
  for j in range(len(SEEDS)):pred[j].append((1-w)*st[j]+w*ht[j])
  ys.append(sf['test']['y']);gs.append(sf['test']['groups']);audit.append({'test_cell':sf['test_cell'],'history_preset':preset,'history_separation':sep,'selected_history_weight':w,'validation_mse_by_weight':dict(zip(map(str,WEIGHTS),scores))});print(sf['test_cell'],w,flush=True)
 y=np.concatenate(ys);g=np.concatenate(gs);P=np.asarray([np.concatenate(x) for x in pred]);r={'status':'post-hoc NASA development; internal dual-view PP weight selected only on each outer fold validation cell','weights':WEIGHTS.tolist(),'folds':audit,'ensemble':regression_metrics(y,P.mean(0),g),'per_seed':[regression_metrics(y,p,g) for p in P]};(OUT/'results.json').write_text(json.dumps(r,indent=2)+'\n');np.savez_compressed(OUT/'predictions.npz',prediction=P,y=y,groups=g);print('FINAL',r['ensemble']['pooled']['r2'],r['ensemble']['unit_macro_r2'])
if __name__=='__main__':main()
