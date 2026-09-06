#!/usr/bin/env python3
"""One-shot sealed MATR batch2 confirmatory evaluation."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import h5py
import numpy as np
import torch

from pp_extrapolation import (apply_shell_calibration, calibrate_distance_shells,
    fit_pp, predict, regression_metrics, select_affine_initialization, support_distance)
from pp_extrapolation.model import _affine_prediction
from pp_extrapolation.support_gate import predict_support_gated, select_support_gate
from plain_mlp_ablation import fit_plain, predict_plain

ROOT=Path(__file__).resolve().parents[1]; MAT=ROOT/'data/matr/2017-06-30_batchdata_updated_struct_errorcorrect.mat'
OUT=ROOT/'results/matr_batch2_confirmatory'; SEEDS=(42,43,44,45,46)

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def load_cells():
    cells=[]
    with h5py.File(MAT,'r') as f:
        refs=f['batch/summary']
        if refs.shape[0]!=48: raise RuntimeError(f'locked protocol requires 48 cells, found {refs.shape[0]}')
        for i in range(48):
            summary=f[refs[i,0]]
            q=np.asarray(summary['QDischarge']).reshape(-1).astype(np.float32)
            cycle=np.asarray(summary['cycle']).reshape(-1).astype(np.float32)
            cells.append((q,cycle))
    return cells

def endpoints(cells,ids): return np.concatenate([cells[i][0][7:] for i in ids])

def make_rows(cells,ids,boundary,train=False,targets=True):
    xs=[]; ys=[]; groups=[]; coordinates=[]
    for i in ids:
        q,cycle=cells[i]; rate=np.r_[0.,np.maximum(q[:-1]-q[1:],0.)]
        for end in range(7,len(q)):
            keep=q[end]>boundary if train else q[end]<boundary
            if not keep: continue
            h=q[end-7:end+1]; r=rate[end-7:end+1]
            xs.append([h[-1],r[-1],h.mean(),r.mean(),h[-1]-h[0],r[-1]-r[0]])
            coordinates.append(h[-1]); groups.append(f'b2c{i}')
            if targets: ys.append(float(cycle[-1]-cycle[end]))
    out={'x':np.asarray(xs,dtype=np.float32),'groups':np.asarray(groups),'coordinate':np.asarray(coordinates,dtype=np.float32)}
    if targets: out['y']=np.asarray(ys,dtype=np.float32)
    return out

def main():
    OUT.mkdir(parents=True,exist_ok=True); cells=load_cells(); cutoff=float(np.quantile(endpoints(cells,range(30)),.25))
    train=make_rows(cells,range(30),cutoff,train=True); boundary=float(train['coordinate'].min())
    val=make_rows(cells,range(30,39),boundary); source=make_rows(cells,range(39,48),boundary,targets=False)
    if len(val['x'])<20 or len(source['x'])<20: raise RuntimeError('locked batch2 tail infeasible')
    vd,va=support_distance(train['coordinate'][:,None],val['coordinate'][:,None]); sd,sa=support_distance(train['coordinate'][:,None],source['coordinate'][:,None])
    if va.outside_fraction<1 or sa.outside_fraction<1: raise RuntimeError('strict hull audit failed')
    affine=select_affine_initialization(train,val); val_pp=[]; source_pp=[]; source_plain=[]; source_gated=[]
    for seed in SEEDS:
        pp=fit_pp(train,val,seed=seed,affine_selection=affine,max_epochs=300)
        val_pp.append(predict(pp,val['x'])); source_pp.append(predict(pp,source['x']))
        beta=select_support_gate(pp,val['x'],val['y'],vd).beta
        source_gated.append(predict_support_gated(pp,source['x'],sd,beta=beta))
        plain=fit_plain(train,val,seed=seed,max_epochs=300); source_plain.append(predict_plain(plain,source['x']))
    val_base=_affine_prediction(affine['initialization'],val['x'],affine['center'],affine['scale'],affine['target_scale'])
    calibration=calibrate_distance_shells(vd,val['y'],np.asarray(val_pp),val_base)
    accepted=apply_shell_calibration(calibration,sd)
    pretest={'protocol_commit':'7b3eeea','cutoff_quantile':cutoff,'actual_train_boundary':boundary,
      'n_train':len(train['x']),'n_validation':len(val['x']),'n_source':len(source['x']),
      'hull':{'validation':va.summary(),'source':sa.summary()},'accepted_count':int(accepted.sum()),
      'coverage':float(accepted.mean()),'shells':[s.__dict__ for s in calibration.shells]}
    (OUT/'gate_decision_pretest.json').write_text(json.dumps(pretest,indent=2)+'\n'); print('PRETEST GATE',pretest['coverage'],flush=True)
    test=make_rows(cells,range(39,48),boundary,targets=True); pp=np.asarray(source_pp); plain=np.asarray(source_plain); gated=np.asarray(source_gated)
    ridge=_affine_prediction(affine['initialization'],test['x'],affine['center'],affine['scale'],affine['target_scale'])
    result={'pretest_gate':pretest,'archive_sha256':sha256(MAT),
      'ridge':regression_metrics(test['y'],ridge,test['groups']),
      'plain_ensemble':regression_metrics(test['y'],plain.mean(0),test['groups']),
      'pp_ensemble':regression_metrics(test['y'],pp.mean(0),test['groups']),
      'support_pp_ensemble':regression_metrics(test['y'],gated.mean(0),test['groups']),
      'pp_seeds':[regression_metrics(test['y'],p,test['groups']) for p in pp],
      'plain_seeds':[regression_metrics(test['y'],p,test['groups']) for p in plain],
      'support_pp_seeds':[regression_metrics(test['y'],p,test['groups']) for p in gated]}
    if accepted.sum()>=2:
        result['selective_pp']=regression_metrics(test['y'][accepted],pp.mean(0)[accepted],test['groups'][accepted])
        result['selective_ridge']=regression_metrics(test['y'][accepted],ridge[accepted],test['groups'][accepted])
    criteria={'nonzero_coverage':pretest['coverage']>0,'selective_positive_r2':result.get('selective_pp',{}).get('pooled',{}).get('r2',-np.inf)>0,
      'selective_beats_ridge_rmse':result.get('selective_pp',{}).get('pooled',{}).get('rmse',np.inf)<result.get('selective_ridge',{}).get('pooled',{}).get('rmse',-np.inf),
      'full_pp_beats_ridge_r2':result['pp_ensemble']['pooled']['r2']>result['ridge']['pooled']['r2']}
    result['success_criteria']=criteria; result['confirmatory_success']=all(criteria.values())
    rng=np.random.default_rng(42); comparisons={}
    for name in ('plain_ensemble','pp_ensemble','support_pp_ensemble'):
        delta=np.asarray([result[name]['per_unit'][u]['rmse']-result['ridge']['per_unit'][u]['rmse'] for u in result['ridge']['per_unit']])
        boot=np.mean(rng.choice(delta,(20000,len(delta)),replace=True),axis=1)
        comparisons[name]={'mean_unit_rmse_delta_vs_ridge':float(delta.mean()),'units_better':int(np.sum(delta<0)),
          'units_total':len(delta),'bootstrap_95_ci':[float(x) for x in np.quantile(boot,[.025,.975])]}
    result['paired_unit_comparisons']=comparisons
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT',result['confirmatory_success'],result['ridge']['pooled']['r2'],result['plain_ensemble']['pooled']['r2'],result['pp_ensemble']['pooled']['r2'])
if __name__=='__main__': main()
