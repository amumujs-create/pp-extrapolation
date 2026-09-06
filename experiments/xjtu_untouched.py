#!/usr/bin/env python3
"""One-shot XJTU-SY evaluation locked by XJTU_UNTOUCHED_PROTOCOL.md."""
from __future__ import annotations
import json, hashlib, sys, time
from pathlib import Path
import numpy as np
import torch

from pp_extrapolation import (apply_shell_calibration, calibrate_distance_shells,
    fit_pp, predict, regression_metrics, select_affine_initialization)
from pp_extrapolation.model import _affine_prediction
from pp_extrapolation.support_gate import predict_support_gated, select_support_gate
from plain_mlp_ablation import fit_plain, predict_plain

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/xjtu_sy/extracted/XJTU-SY_Bearing_Datasets'
OUT=ROOT/'results/xjtu_untouched_v1'
CONDITIONS={'35Hz12kN':(2100.,12.),'37.5Hz11kN':(2250.,11.),'40Hz10kN':(2400.,10.)}
SEEDS=(42,43,44,45,46)

def recording_stats(path):
    x=np.loadtxt(path,delimiter=',',skiprows=1,dtype=np.float64)
    cols=[]
    for a in (x[:,0],x[:,1]):
        mean=a.mean(); centered=a-mean; sd=max(float(np.sqrt(np.mean(centered**2))),1e-12)
        rms=float(np.sqrt(np.mean(a*a))); peak=float(np.max(np.abs(a)))
        cols += [mean,sd,rms,peak,float(np.mean((centered/sd)**3)),
                 float(np.mean((centered/sd)**4)-3.),peak/max(rms,1e-12)]
    return cols

def build_cache():
    cache=ROOT/'data/xjtu_sy/features_locked.npz'
    if cache.exists(): return dict(np.load(cache,allow_pickle=False))
    rows=[]; units=[]; conditions=[]; positions=[]; lives=[]
    for cname in CONDITIONS:
        for folder in sorted((DATA/cname).glob('Bearing*')):
            files=sorted(folder.glob('*.csv'),key=lambda p:int(p.stem)); life=len(files)
            for pos,path in enumerate(files,1):
                rows.append(recording_stats(path)); units.append(folder.name)
                conditions.append(cname); positions.append(pos); lives.append(life)
            print('parsed',cname,folder.name,life,flush=True)
    payload={'stats':np.asarray(rows,dtype=np.float32),'units':np.asarray(units),
             'conditions':np.asarray(conditions),'positions':np.asarray(positions),
             'lives':np.asarray(lives)}
    np.savez_compressed(cache,**payload); return payload

def windows(raw,cname,include_targets=True):
    xs=[]; ys=[]; groups=[]
    for unit in np.unique(raw['units'][raw['conditions']==cname]):
        idx=np.flatnonzero((raw['conditions']==cname)&(raw['units']==unit))
        stat=raw['stats'][idx]; pos=raw['positions'][idx]; life=int(raw['lives'][idx][0])
        for end in range(7,len(idx)):
            block=stat[end-7:end+1]
            feat=np.r_[block[-1],block.mean(0),block[-1]-block[0],CONDITIONS[cname]]
            xs.append(feat); groups.append(unit)
            if include_targets: ys.append(life-int(pos[end]))
    out={'x':np.asarray(xs,dtype=np.float32),'groups':np.asarray(groups)}
    if include_targets: out['y']=np.asarray(ys,dtype=np.float32)
    return out

def op_distance(cname):
    speed,load=CONDITIONS[cname]; center=CONDITIONS['37.5Hz11kN']
    return float(np.linalg.norm([(speed-center[0])/150.,(load-center[1])/1.]))

def file_sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True); raw=build_cache(); torch.set_num_threads(2)
    train=windows(raw,'37.5Hz11kN'); val=windows(raw,'35Hz12kN')
    source_x=windows(raw,'40Hz10kN',include_targets=False)
    affine=select_affine_initialization(train,val); fits=[]; val_predictions=[]; source_predictions=[]; gated_predictions=[]; plain_predictions=[]
    vd=np.full(len(val['y']),op_distance('35Hz12kN')); sd=np.full(len(source_x['x']),op_distance('40Hz10kN'))
    for seed in SEEDS:
        fit=fit_pp(train,val,seed=seed,affine_selection=affine,max_epochs=300); fits.append(fit)
        val_predictions.append(predict(fit,val['x'])); source_predictions.append(predict(fit,source_x['x']))
        beta=select_support_gate(fit,val['x'],val['y'],vd).beta
        gated_predictions.append(predict_support_gated(fit,source_x['x'],sd,beta=beta))
        plain=fit_plain(train,val,seed=seed,max_epochs=300); plain_predictions.append(predict_plain(plain,source_x['x']))
    val_base=_affine_prediction(affine['initialization'],val['x'],affine['center'],affine['scale'],affine['target_scale'])
    calibration=calibrate_distance_shells(vd,val['y'],np.asarray(val_predictions),val_base)
    mask=apply_shell_calibration(calibration,sd)
    pretest={'protocol_commit':'a3e8999','shell_commit':'f6eb4c3','validation_distance':vd[0],
      'source_distance':sd[0],'source_count':len(sd),'accepted_count':int(mask.sum()),
      'coverage':float(mask.mean()),'shells':[s.__dict__ for s in calibration.shells]}
    (OUT/'gate_decision_pretest.json').write_text(json.dumps(pretest,indent=2)+'\n')
    print('PRETEST GATE',pretest['coverage'],flush=True)
    # Test lifetimes/labels are materialized only after the frozen decision is saved.
    test=windows(raw,'40Hz10kN',include_targets=True)
    pp=np.asarray(source_predictions); gated=np.asarray(gated_predictions); plain=np.asarray(plain_predictions)
    ridge=_affine_prediction(affine['initialization'],test['x'],affine['center'],affine['scale'],affine['target_scale'])
    result={'pretest_gate':pretest,'n':{k:len(v['x']) for k,v in [('train',train),('validation',val),('test',test)]},
      'units':{k:int(len(np.unique(v['groups']))) for k,v in [('train',train),('validation',val),('test',test)]},
      'ridge':regression_metrics(test['y'],ridge,test['groups']),
      'plain_ensemble':regression_metrics(test['y'],plain.mean(0),test['groups']),
      'pp_ensemble':regression_metrics(test['y'],pp.mean(0),test['groups']),
      'support_pp_ensemble':regression_metrics(test['y'],gated.mean(0),test['groups']),
      'pp_seeds':[regression_metrics(test['y'],p,test['groups']) for p in pp],
      'archive_sha256':file_sha256(ROOT/'data/xjtu_sy/XJTU-SY_Bearing_Datasets.zip')}
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT',result['ridge']['pooled']['r2'],result['plain_ensemble']['pooled']['r2'],result['pp_ensemble']['pooled']['r2'])
if __name__=='__main__': main()
