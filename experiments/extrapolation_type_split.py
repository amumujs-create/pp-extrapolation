#!/usr/bin/env python3
"""Seen-unit future tail versus prior unseen-unit late-tail PP experiments."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from pp_extrapolation import fit_pp,predict,regression_metrics,select_affine_initialization,support_distance
from pp_extrapolation.model import _affine_prediction
from plain_mlp_ablation import fit_plain,predict_plain
ROOT=Path(__file__).resolve().parents[2]/'ca-css-ncmapss';sys.path.insert(0,str(ROOT))
from run_affine_tail_external_three import causal_hust_frame,reconstruct_virkler,frame_to_arrays
SEEDS=(42,43,44,45,46)
def evaluate(name,tr,va,te):
 affine=select_affine_initialization(tr,va);ridge=_affine_prediction(affine['initialization'],te['x'],affine['center'],affine['scale'],affine['target_scale']);pp=[];nn=[]
 for s in SEEDS:
  pp.append(predict(fit_pp(tr,va,seed=s,affine_selection=affine,max_epochs=300),te['x']));nn.append(predict_plain(fit_plain(tr,va,seed=s,max_epochs=300),te['x']))
 pp=np.asarray(pp);nn=np.asarray(nn)
 def sm(m):
  r=np.array([regression_metrics(te['y'],p,te['groups'])['pooled']['r2'] for p in m]);return {'mean_r2':float(r.mean()),'sd_r2':float(r.std()),'ensemble':regression_metrics(te['y'],m.mean(0),te['groups'])}
 vd,vaudit=support_distance(tr['x'][:,[0]],va['x'][:,[0]]);td,taudit=support_distance(tr['x'][:,[0]],te['x'][:,[0]])
 return {'n':{'train':len(tr['y']),'validation':len(va['y']),'test':len(te['y'])},'units':{k:len(np.unique(v['groups'])) for k,v in [('train',tr),('validation',va),('test',te)]},'hull':{'validation':vaudit.summary(),'test':taudit.summary()},'ridge':regression_metrics(te['y'],ridge,te['groups']),'pp':sm(pp),'plain_nn':sm(nn)}
def hust():
 f,_=causal_hust_frame();features=['capacity_ah','cycle','prefix_rate','recent_rate'];f=f[f.protocol<=6].copy();early=f[f.capacity_ah>=1.05];tail=f[f.capacity_ah<1.05];cut=float(tail.capacity_ah.median());va=tail[tail.capacity_ah>=cut];te=tail[tail.capacity_ah<cut]
 return evaluate('hust_seen_future',frame_to_arrays(early,features),frame_to_arrays(va,features),frame_to_arrays(te,features))|{'tail_cutoff':cut}
def virkler():
 f,_,_=reconstruct_virkler();features=['crack_length_mm','elapsed_kcycles','prefix_rate_mm_per_kcycle','recent_rate_mm_per_kcycle'];tr=f[f.crack_length_mm<=26];va=f[f.crack_length_mm==33];te=f[f.crack_length_mm>33]
 return evaluate('virkler_seen_future',frame_to_arrays(tr,features),frame_to_arrays(va,features),frame_to_arrays(te,features))
def main():
 torch.set_num_threads(2);payload={'experiment':'extrapolation_type_seen_unit_future_v1','status':'retrospective development','datasets':{'hust':hust(),'virkler':virkler()}}
 out=Path('results/extrapolation_type_split_v1');out.mkdir(parents=True,exist_ok=True);(out/'results.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
