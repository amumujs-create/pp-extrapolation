"""Run fixed NASA health-v2 ablations from a locally prepared fold archive.

Archive keys: f{0..3}_{train|validation|test}_{x|y|groups}.
Use the original health-v2 split, not a new random split.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from pp_extrapolation import PriorPairs, TransportTriples, fit_pp, predict, regression_metrics
from pp_extrapolation.hull import audit_convex_hull_support
from pp_extrapolation.priors import prepare_pairs, prepare_transport

ARMS = ('PP', 'direction', 'transport', 'both')
SEEDS = (42,43,44,45,46)

def priors(train):
    lo=float(train['x'].min()); hi=float(train['x'].max())
    # Design grid uses train minimum and known health zero, never test X or y.
    q=np.linspace(0.,lo,64)[:,None]
    pair=PriorPairs(q+0.05,q,np.full(64,-np.inf),np.zeros(64),np.ones(64))
    # Equal inward/outward steps cover near and distant extrapolation.
    step=np.linspace(.02,min(lo,hi-lo),64)[:,None]
    boundary=np.full_like(step,lo)
    triple=TransportTriples(boundary+step,boundary,boundary-step,np.ones(64),np.full(64,2.),np.ones(64))
    return pair,triple

def diagnostic(fit,p,t):
    a,b,lower,upper,c=prepare_pairs(p,fit.center,fit.scale,fit.target_scale)
    i,j,k,r,tol,c=prepare_transport(t,fit.center,fit.scale,fit.target_scale)
    with torch.no_grad():
        delta=fit.model(b)-fit.model(a)
        violation=torch.relu(delta-upper)+torch.relu(lower-delta)
        error=torch.abs((fit.model(k)-fit.model(j))-r*(fit.model(j)-fit.model(i)))
    return dict(direction_violation_fraction=float((violation>1e-7).float().mean()),
                direction_violation_cycles=float(violation.mean()*fit.target_scale),
                transport_active_fraction=float((error>tol+1e-7).float().mean()),
                transport_error_cycles=float(error.mean()*fit.target_scale))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--folds',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args(); out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    archive=np.load(args.folds,allow_pickle=False)
    folds=[{part:{key:archive[f'f{i}_{part}_{key}'] for key in ('x','y','groups')} for part in ('train','validation','test')} for i in range(4)]
    protocol=dict(status='post_hoc_development_on_reused_NASA_health_v2',seeds=SEEDS,arms=ARMS,
                  weights=10.,transport_tolerance_cycles=2.,max_epochs=300,
                  archive_sha256=hashlib.sha256(Path(args.folds).read_bytes()).hexdigest(),
                  selection='validation MSE only; all arms reported; no test-based weight search')
    (out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2)+'\n')
    audits=[]
    for f in folds:
        groups=[set(f[p]['groups']) for p in ('train','validation','test')]
        assert all(groups[i].isdisjoint(groups[j]) for i,j in ((0,1),(0,2),(1,2)))
        audit={p:audit_convex_hull_support(f['train']['x'],f[p]['x']).summary() for p in ('validation','test')}
        assert all(v['outside_fraction']==1. for v in audit.values())
        audits.append(audit)
    truth=np.concatenate([f['test']['y'] for f in folds]); groups=np.concatenate([f['test']['groups'] for f in folds])
    saved=dict(truth=truth,groups=groups); runs=[]
    for arm in ARMS:
        for seed in SEEDS:
            parts=[]; details=[]
            for fi,f in enumerate(folds):
                p,t=priors(f['train'])
                kw={}
                if arm in ('direction','both'):kw.update(prior_pairs=p,prior_weight=10.)
                if arm in ('transport','both'):kw.update(transport_triples=t,transport_weight=10.)
                fit=fit_pp(f['train'],f['validation'],seed=seed,**kw)
                parts.append(predict(fit,f['test']['x']))
                details.append(dict(fold=fi,selection=fit.selection,diagnostics=diagnostic(fit,p,t)))
            prediction=np.concatenate(parts);saved[f'{arm}_{seed}']=prediction
            metric=regression_metrics(truth,prediction,groups)
            runs.append(dict(arm=arm,seed=seed,metrics=metric,folds=details))
            print(arm,seed,metric['pooled']['r2'],flush=True)
    payload=dict(protocol=protocol,audits=audits,runs=runs)
    (out/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
    np.savez_compressed(out/'predictions.npz',**saved)
if __name__=='__main__':main()
