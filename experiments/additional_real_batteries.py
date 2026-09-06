#!/usr/bin/env python3
"""Frozen PP comparison on Sunwoda, RWTH, and MICH strict late tails."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

from pp_extrapolation.hull import audit_convex_hull_support
from pp_extrapolation.metrics import regression_metrics
from pp_extrapolation.model import fit_pp, predict, select_affine_initialization, _affine_prediction
from pp_extrapolation.support_gate import predict_support_gated, select_support_gate, support_distance
from plain_mlp_ablation import fit_plain, predict_plain

SEEDS=(42,43,44,45,46)
BETAS=(0.,.05,.1,.25,.5,1.,2.,4.,8.)

def features(part, cycle_scale):
    z=np.asarray(part['x'],dtype=np.float32); h=z[:,:,0]; r=z[:,:,1]
    return {'x':np.column_stack((h[:,-1],r[:,-1],h.mean(1),r.mean(1),h[:,-1]-h[:,0],r[:,-1]-r[:,0],part['cycles']/cycle_scale)).astype(np.float32),
            'y':np.asarray(part['y'],dtype=np.float32),'groups':np.asarray(part['units'])}

def metric(y,p,g): return regression_metrics(y,p,g)
def stats(rows,key,path):
    a=np.array([r[key][path]['r2'] if path=='pooled' else r[key][path] for r in rows])
    return {'mean':float(a.mean()),'sd':float(a.std())}

def run_dataset(name, raw, epochs):
    endpoint=raw['train']['x'][:,-1,0]; cutoff=float(np.quantile(endpoint,.25))
    trmask=endpoint>cutoff; vamask=raw['val']['x'][:,-1,0]<cutoff
    if not vamask.any(): raise RuntimeError(f'{name}: empty pseudo-tail validation')
    cycle_scale=max(float(raw['train']['cycles'].max()),1.)
    train=features({k:v[trmask] for k,v in raw['train'].items()},cycle_scale)
    val=features({k:v[vamask] for k,v in raw['val'].items()},cycle_scale)
    test=features(raw['source'],cycle_scale)
    va=audit_convex_hull_support(train['x'][:,[0]],val['x'][:,[0]])
    te=audit_convex_hull_support(train['x'][:,[0]],test['x'][:,[0]])
    if va.outside_fraction<1 or te.outside_fraction<1: raise RuntimeError(f'{name}: hull check failed')
    affine=select_affine_initialization(train,val); rows=[]
    for seed in SEEDS:
        plain=fit_plain(train,val,seed=seed,max_epochs=epochs)
        pp=fit_pp(train,val,seed=seed,affine_selection=affine,max_epochs=epochs)
        vp_dist,_=support_distance(train['x'][:,[0]],val['x'][:,[0]])
        tp_dist,_=support_distance(train['x'][:,[0]],test['x'][:,[0]])
        gate=select_support_gate(pp,val['x'],val['y'],vp_dist,betas=BETAS)
        ridge_pred=_affine_prediction(affine['initialization'],test['x'],affine['center'],affine['scale'],affine['target_scale'])
        row={'seed':seed,'beta':gate.beta,
             'ridge':metric(test['y'],ridge_pred,test['groups']),
             'plain':metric(test['y'],predict_plain(plain,test['x']),test['groups']),
             'pp':metric(test['y'],predict(pp,test['x']),test['groups']),
             'support_pp':metric(test['y'],predict_support_gated(pp,test['x'],tp_dist,beta=gate.beta),test['groups'])}
        rows.append(row); print(name,seed,gate.beta,*(f'{k}={row[k]["pooled"]["r2"]:.3f}' for k in ('ridge','plain','pp','support_pp')),flush=True)
    return {'cutoff':cutoff,'n':{'train':len(train['y']),'validation':len(val['y']),'test':len(test['y'])},
            'units':{k:int(len(np.unique(v['groups']))) for k,v in [('train',train),('validation',val),('test',test)]},
            'hull':{'validation':va.summary(),'test':te.summary()},'summary':{k:{'pooled_r2':stats(rows,k,'pooled'),'unit_macro_r2':stats(rows,k,'unit_macro_r2')} for k in ('ridge','plain','pp','support_pp')},'runs':rows}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--legacy-root',default=str(Path(__file__).resolve().parents[2]/'ca-css-ncmapss')); p.add_argument('--max-epochs',type=int,default=300); p.add_argument('--output',default='results/additional_real_batteries'); a=p.parse_args()
    sys.path.insert(0,str(Path(a.legacy_root).resolve())); from pae_boundary_realdata import prepare_dataset
    torch.set_num_threads(2); start=time.time(); datasets={}
    for name in ('sunwoda','rwth','mich'):
        split,audit=prepare_dataset(name); datasets[name]=run_dataset(name,split,a.max_epochs); datasets[name]['loader_audit']=audit
    payload={'experiment':'additional_real_batteries_v1','protocol_commit':'567cb2f','seeds':list(SEEDS),'datasets':datasets,'runtime_seconds':time.time()-start}
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True); (out/'results.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
if __name__=='__main__': main()
