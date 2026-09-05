"""Synthetic development ablation; not real-data/confirmatory evidence.

All variants and priors are fixed before scoring. No prior uses test labels.
"""
import json
from pathlib import Path
import numpy as np
import torch
from pp_extrapolation import PriorPairs, TransportTriples, fit_pp, predict, regression_metrics

def main():
    torch.set_num_threads(2)
    rng = np.random.default_rng(20260906)
    records=[]
    for scenario in ('smooth','turning'):
        def truth(q):
            if scenario == 'smooth': return 100*q+12*np.sin(3*q)
            return 40+70*(q-.35)**2
        def split(units,low,high):
            q=np.tile(np.linspace(low,high,24),len(units))
            return dict(x=q[:,None].astype('float32'),y=(truth(q)+rng.normal(0,1,len(q))).astype('float32'),groups=np.repeat(units,24))
        train=split(['a','b','c','d'],.55,1)
        val=split(['v1','v2'],.35,.5)
        test=split(['t1','t2'],.05,.3)
        # Declared 1D health interventions; all anchors are train-derived.
        anchor=train['x']; changed=anchor-.30
        p=PriorPairs(anchor,changed,np.full(len(anchor),-np.inf),np.zeros(len(anchor)),np.ones(len(anchor)))
        wrong=PriorPairs(anchor,changed,np.zeros(len(anchor)),np.full(len(anchor),np.inf),np.ones(len(anchor)))
        b=anchor[anchor[:,0]<=.7]
        t=TransportTriples(b+.1,b,b-.2,np.full(len(b),2.),np.full(len(b),2.),np.ones(len(b)))
        for name,kw in [('PP',{}),('direction',dict(prior_pairs=p,prior_weight=10)),('transport',dict(transport_triples=t,transport_weight=10)),('both',dict(prior_pairs=p,prior_weight=10,transport_triples=t,transport_weight=10)),('wrong_direction',dict(prior_pairs=wrong,prior_weight=10))]:
            scores=[]
            for seed in (42,43,44):
                fit=fit_pp(train,val,seed=seed,max_epochs=150,**kw)
                scores.append(regression_metrics(test['y'],predict(fit,test['x']),test['groups'])['pooled']['r2'])
            row=dict(scenario=scenario,variant=name,pooled_r2_mean=float(np.mean(scores)),pooled_r2_sd=float(np.std(scores)),seeds=scores)
            records.append(row)
            print(f'{scenario:8s} {name:16s} {np.mean(scores): .4f} +/- {np.std(scores):.4f}',flush=True)
    output=Path('results/prior_ablation'); output.mkdir(parents=True,exist_ok=True)
    (output/'results.json').write_text(json.dumps(records,indent=2)+'\n')
if __name__=='__main__': main()
