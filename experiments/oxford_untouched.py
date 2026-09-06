#!/usr/bin/env python3
"""Locked Oxford Battery Degradation Dataset 1 untouched evaluation."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch

from pp_extrapolation import (apply_shell_calibration, calibrate_distance_shells,
    fit_pp, predict, regression_metrics, select_affine_initialization, support_distance)
from pp_extrapolation.model import _affine_prediction
from pp_extrapolation.support_gate import predict_support_gated, select_support_gate
from plain_mlp_ablation import fit_plain, predict_plain

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/oxford_untouched_v1'
MAT=ROOT/'data/oxford/Oxford_Battery_Degradation_Dataset_1.mat'; SEEDS=(42,43,44,45,46)

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def load_capacity():
    mat=sio.loadmat(MAT,squeeze_me=True,struct_as_record=False); cells={}
    for number in range(1,9):
        cell=mat[f'Cell{number}']; capacity=[]
        for cycle in cell._fieldnames:
            q=np.asarray(getattr(getattr(cell,cycle),'C1dc').q,dtype=float).reshape(-1)
            capacity.append(float(np.ptp(q)))
        cells[number]=np.asarray(capacity,dtype=np.float32)
    return cells

def make_rows(cells,numbers,cutoff,train=False,targets=True):
    xs=[]; ys=[]; groups=[]; coords=[]
    for number in numbers:
        cap=cells[number]; rate=np.r_[0.,np.maximum(cap[:-1]-cap[1:],0.)]
        for end in range(7,len(cap)):
            if (train and not cap[end]>cutoff) or (not train and not cap[end]<cutoff): continue
            h=cap[end-7:end+1]; r=rate[end-7:end+1]
            xs.append([h[-1],r[-1],h.mean(),r.mean(),h[-1]-h[0],r[-1]-r[0]])
            coords.append(h[-1]); groups.append(f'Cell{number}')
            if targets: ys.append(len(cap)-1-end)
    out={'x':np.asarray(xs,dtype=np.float32),'groups':np.asarray(groups),'coordinate':np.asarray(coords,dtype=np.float32)}
    if targets: out['y']=np.asarray(ys,dtype=np.float32)
    return out

def main():
    OUT.mkdir(parents=True,exist_ok=True); cells=load_capacity(); all_train=np.concatenate([cells[i][7:] for i in range(1,6)])
    cutoff=float(np.quantile(all_train,.25)); train=make_rows(cells,range(1,6),cutoff,train=True)
    val=make_rows(cells,[6],cutoff); source=make_rows(cells,[7,8],cutoff,targets=False)
    if len(val['x'])<8 or len(source['x'])<8: raise RuntimeError('locked Oxford tail is infeasible')
    vd,va=support_distance(train['coordinate'][:,None],val['coordinate'][:,None]); sd,sa=support_distance(train['coordinate'][:,None],source['coordinate'][:,None])
    if va.outside_fraction<1 or sa.outside_fraction<1: raise RuntimeError('strict hull audit failed')
    affine=select_affine_initialization(train,val); val_pp=[]; source_pp=[]; source_plain=[]; source_gated=[]
    for seed in SEEDS:
        pp=fit_pp(train,val,seed=seed,affine_selection=affine,max_epochs=300)
        val_pp.append(predict(pp,val['x'])); source_pp.append(predict(pp,source['x']))
        beta=select_support_gate(pp,val['x'],val['y'],vd).beta
        source_gated.append(predict_support_gated(pp,source['x'],sd,beta=beta))
        plain=fit_plain(train,val,seed=seed,max_epochs=300); source_plain.append(predict_plain(plain,source['x']))
    base_val=_affine_prediction(affine['initialization'],val['x'],affine['center'],affine['scale'],affine['target_scale'])
    calibration=calibrate_distance_shells(vd,val['y'],np.asarray(val_pp),base_val)
    accepted=apply_shell_calibration(calibration,sd)
    pretest={'protocol_commit':'c92622a','cutoff':cutoff,'n_train':len(train['x']),'n_validation':len(val['x']),'n_source':len(source['x']),
      'hull':{'validation':va.summary(),'source':sa.summary()},'accepted_count':int(accepted.sum()),'coverage':float(accepted.mean()),
      'shells':[s.__dict__ for s in calibration.shells]}
    (OUT/'gate_decision_pretest.json').write_text(json.dumps(pretest,indent=2)+'\n'); print('PRETEST GATE',pretest['coverage'],flush=True)
    test=make_rows(cells,[7,8],cutoff,targets=True); pp=np.asarray(source_pp); plain=np.asarray(source_plain); gated=np.asarray(source_gated)
    ridge=_affine_prediction(affine['initialization'],test['x'],affine['center'],affine['scale'],affine['target_scale'])
    result={'pretest_gate':pretest,'archive_sha256':sha256(MAT),
      'ridge':regression_metrics(test['y'],ridge,test['groups']),
      'plain_ensemble':regression_metrics(test['y'],plain.mean(0),test['groups']),
      'pp_ensemble':regression_metrics(test['y'],pp.mean(0),test['groups']),
      'support_pp_ensemble':regression_metrics(test['y'],gated.mean(0),test['groups']),
      'pp_seeds':[regression_metrics(test['y'],p,test['groups']) for p in pp]}
    if accepted.sum()>=2:
        result['selective_pp']=regression_metrics(test['y'][accepted],pp.mean(0)[accepted],test['groups'][accepted])
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT',result['ridge']['pooled']['r2'],result['plain_ensemble']['pooled']['r2'],result['pp_ensemble']['pooled']['r2'])
if __name__=='__main__': main()
