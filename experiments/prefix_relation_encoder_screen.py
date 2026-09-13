"""Fixed TRAIN-only strict support/unit screen. No validation/test access."""
import json
import hashlib
import numpy as np
import torch
from function_prior_train_diagnosis import ROOT,score
from relation_local_transport_screen import subset,write
from component_transfer_screen import source_episode_indices
from pp_extrapolation.function_prior_sharing import fit_bank,SharingRule,SharingFit,predict_sharing
from pp_extrapolation.prefix_relation_encoder import fit_prefix,predict_prefix


def main():
    torch.set_num_threads(2)
    out=ROOT/'results/prefix_relation_encoder_train_v1';out.mkdir(exist_ok=False)
    path=ROOT/'results/relation_local_transport_mich_v1/train_rows.npz'
    write(out/'protocol.json',dict(primary='prefix_distribution',seeds=[42,43,44],
        scope='TRAIN only; strict donor early support; held-unit labels never used for fitting',
        arms=['uniform','fixed_similarity','ridge_distribution','no_consistency','prefix_distribution'],
        training='160 fixed steps, energy loss, residual MLP16, adjacent-prefix coefficient penalty .1; no selection on query',
        acceptance='primary >=2% lower episode-average unit MSE than every control; worst unit MSE no worse than uniform; interval coverage >=.9',
        limits=['small overlapping source episodes','empirical coefficient covariance not calibrated',
                'no PP-X/Engression victory or novelty claim','prefix history starts at first available supplied row'],
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                (path,__file__ and ROOT/'experiments/prefix_relation_encoder_screen.py',ROOT/'src/pp_extrapolation/prefix_relation_encoder.py')}))
    rows=dict(np.load(path));episodes=[];saved={}
    for fold in range(3):
        for frac in (.6,.8):
            donor,query,cut,_=source_episode_indices(rows,'mich',fold,frac)
            d,q=subset(rows,donor),subset(rows,query)
            assert not set(d['groups'])&set(q['groups'])
            assert d['x'][:,0].min()>q['x'][:,0].max()
            bank=fit_bank(d);pred={};intervals={}
            for name,mode in [('uniform','fixed'),('fixed_similarity','component')]:
                dist=predict_sharing(SharingFit(SharingRule(mode),bank,{}),q)
                pred[name]=dist['mean'];intervals[name]=[dist['q05'],dist['q95']]
            for name,neural,consistency in [('ridge_distribution',False,False),('no_consistency',True,False),('prefix_distribution',True,True)]:
                distributions=[]
                for seed in ([42,43,44] if neural else [42]):
                    fit=fit_prefix(d,bank,neural=neural,consistency=consistency,seed=seed)
                    dist=predict_prefix(fit,q);distributions.append(dist)
                    torch.save(fit,out/f'f{fold}_q{frac}_{name}_{seed}.pt')
                # Per-seed interval coverage, not quantiles of an ensemble mixture.
                pred[name]=np.mean([v['mean'] for v in distributions],axis=0)
                intervals[name]=[np.stack([v['q05'] for v in distributions]),np.stack([v['q95'] for v in distributions])]
            report=dict(fold=fold,fraction=frac,scores={k:score(q['y'],p,q['groups']) for k,p in pred.items()},
                interval_coverage={k:float(np.mean((q['y']>=v[0])&(q['y']<=v[1]))) for k,v in intervals.items()})
            episodes.append(report);prefix=f'f{fold}_q{frac}'
            saved[prefix+'_y']=q['y'];saved[prefix+'_groups']=q['groups']
            for k,v in pred.items():saved[prefix+'_'+k]=v
            for k,(lo,hi) in intervals.items():saved[prefix+'_'+k+'_lo']=lo;saved[prefix+'_'+k+'_hi']=hi
            print('EPISODE',fold,frac,{k:round(v['unit_mse'],4) for k,v in report['scores'].items()},flush=True)
    mse={k:float(np.mean([e['scores'][k]['unit_mse'] for e in episodes])) for k in pred}
    coverage={k:float(np.mean([e['interval_coverage'][k] for e in episodes])) for k in pred}
    worst={k:max(max(e['scores'][k]['unit_mse_values']) for e in episodes) for k in pred}
    accepted=all(mse['prefix_distribution']<.98*mse[k] for k in pred if k!='prefix_distribution') and worst['prefix_distribution']<=worst['uniform'] and coverage['prefix_distribution']>=.9
    np.savez_compressed(out/'predictions.npz',**saved)
    write(out/'results.json',dict(complete=True,episodes=episodes,unit_mse=mse,coverage=coverage,worst_unit_mse=worst,
        source_gate_passed=accepted,successor_promoted=False))
    print('COMPLETE',mse,coverage,'gate',accepted,flush=True)


if __name__=='__main__':main()
