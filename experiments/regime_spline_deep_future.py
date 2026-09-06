#!/usr/bin/env python3
"""Development experiment for the doctoral deep-future regime extension."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from pp_extrapolation import fit_regime_spline_pp,predict_regime_spline,regression_metrics,select_affine_initialization
ROOT=Path(__file__).resolve().parents[2]/'ca-css-ncmapss';sys.path.insert(0,str(ROOT))
from run_affine_tail_external_three import causal_hust_frame,reconstruct_virkler,frame_to_arrays
SEEDS=(42,43,44,45,46)
def splits():
 f,_=causal_hust_frame();fe=['capacity_ah','cycle','prefix_rate','recent_rate'];f=f[f.protocol<=6].copy();early=f[f.capacity_ah>=1.05];tail=f[f.capacity_ah<1.05];cut=float(tail.capacity_ah.median());h=(frame_to_arrays(early,fe),frame_to_arrays(tail[tail.capacity_ah>=cut],fe),frame_to_arrays(tail[tail.capacity_ah<cut],fe))
 v,_,_=reconstruct_virkler();fe=['crack_length_mm','elapsed_kcycles','prefix_rate_mm_per_kcycle','recent_rate_mm_per_kcycle'];vs=(frame_to_arrays(v[v.crack_length_mm<=26],fe),frame_to_arrays(v[v.crack_length_mm==33],fe),frame_to_arrays(v[v.crack_length_mm>33],fe));return {'hust':h,'virkler':vs}
def run(tr,va,te,monotone=False):
 affine=select_affine_initialization(tr,va);rows=[];pred=[]
 for s in SEEDS:
  fit=fit_regime_spline_pp(tr,va,seed=s,affine_selection=affine,max_epochs=300,monotone=monotone);p=predict_regime_spline(fit,te['x']);m=regression_metrics(te['y'],p,te['groups']);pred.append(p);rows.append({'seed':s,'selection':fit.selection,'metrics':m});print('mono' if monotone else 'free',s,fit.selection['selected_epoch'],m['pooled']['r2'],flush=True)
 pred=np.asarray(pred);r=np.asarray([x['metrics']['pooled']['r2'] for x in rows]);return {'pooled_r2_mean':float(r.mean()),'pooled_r2_sd':float(r.std()),'ensemble':regression_metrics(te['y'],pred.mean(0),te['groups']),'runs':rows}
def main():
 torch.set_num_threads(2);base=json.load(open('results/extrapolation_type_split_v1/results.json'))['datasets'];outcomes={}
 for n,s in splits().items():outcomes[n]={'baseline':{'ridge':base[n]['ridge']['pooled'],'pp':base[n]['pp'],'plain_nn':base[n]['plain_nn']},'regime_spline_pp':run(*s),'monotone_regime_spline_pp':run(*s,monotone=True)}
 payload={'experiment':'regime_spline_pp_deep_future_v1','status':'post-hoc doctoral extension development; not confirmatory','architecture':'frozen affine + context-conditioned oriented ReLU hinge slopes + bounded local residual','datasets':outcomes}
 out=Path('results/regime_spline_deep_future_v1');out.mkdir(parents=True,exist_ok=True);(out/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
if __name__=='__main__':main()
