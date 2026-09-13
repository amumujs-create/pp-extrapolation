"""Nested TRAIN-only error distribution screen; outer queries never fit correction."""
import hashlib
import numpy as np
import torch
from dataclasses import replace
from function_prior_train_diagnosis import ROOT,score
from relation_local_transport_screen import subset,write
from component_transfer_screen import source_episode_indices
from pp_extrapolation.function_prior_sharing import fit_bank
from pp_extrapolation.prefix_relation_encoder import fit_prefix,predict_prefix
from pp_extrapolation.transfer_error_distribution import distance_features,fit_error,factors


def summarize(y,g,p,lo,hi):
    if not all(np.isfinite(v).all() for v in (p,lo,hi)):raise ValueError('nonfinite prediction')
    s=score(y,p,g)
    s.update(coverage=float(np.mean((y>=lo)&(y<=hi))),width=float(np.mean(hi-lo)),
        interval_score=float(np.mean(hi-lo+20*np.maximum(lo-y,0)+20*np.maximum(y-hi,0))))
    return s


def main():
    torch.set_num_threads(2)
    out=ROOT/'results/transfer_error_distribution_train_v1';out.mkdir(exist_ok=False)
    path=ROOT/'results/relation_local_transport_mich_v1/train_rows.npz'
    write(out/'protocol.json',dict(primary='conditional',arms=['ridge','spread_only','global','conditional'],
        scope='TRAIN only; six outer unit/support splits, six nested splits per outer donor',
        calibration='log(y/base mean), unit-balanced empirical errors; LOOU linear distance residuals',
        prediction='base mean times empirical factors; replaces rather than stacks coefficient uncertainty',
        point='arithmetic mean of empirical predictive draws',
        acceptance='conditional MSE <= ridge; coverage >= .9; interval score < ridge and spread_only; worst unit MSE <= ridge',
        fixed='ridge penalty .1; residual and location clipped separately to +-log4; 256 deterministic quantiles',
        limits=['nested errors do not guarantee further-support coverage','same outer development episodes as previous rounds',
                'multiplicative error law is not new prior-component sharing or novelty proof'],
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (path,ROOT/'experiments/transfer_error_distribution_screen.py',ROOT/'src/pp_extrapolation/transfer_error_distribution.py',
             ROOT/'src/pp_extrapolation/prefix_relation_encoder.py',ROOT/'src/pp_extrapolation/function_prior_sharing.py')}))
    rows=dict(np.load(path));episodes=[];saved={};audit=[]
    for fold in range(3):
        for frac in (.6,.8):
            donor,query,_,_=source_episode_indices(rows,'mich',fold,frac)
            d,q=subset(rows,donor),subset(rows,query)
            assert not set(d['groups'])&set(q['groups'])
            assert d['x'][:,0].min()>q['x'][:,0].max()
            features=[];errors=[];groups=[];inner_audit=[]
            for f in range(3):
                for t in (.6,.8):
                    a,b,_,_=source_episode_indices(d,'mich',f,t);src,cal=subset(d,a),subset(d,b)
                    assert not set(src['groups'])&set(cal['groups'])
                    assert src['x'][:,0].min()>cal['x'][:,0].max()
                    fit=fit_prefix(src,fit_bank(src),neural=False)
                    p=predict_prefix(fit,cal)['mean']
                    features.append(distance_features(cal['x'],src['x'][:,0].min(),np.ptp(src['x'][:,0])))
                    errors.append(np.log(cal['y']/p));groups.append(cal['groups'])
                    inner_audit.append(dict(fold=f,fraction=t,source_units=np.unique(src['groups']).tolist(),
                        calibration_units=np.unique(cal['groups']).tolist(),source_min=float(src['x'][:,0].min()),
                        calibration_max=float(cal['x'][:,0].max())))
            x=np.concatenate(features);err=np.concatenate(errors);g=np.concatenate(groups)
            model=fit_error(x,err,g)
            base=fit_prefix(d,fit_bank(d),neural=False);dist=predict_prefix(base,q)
            z=distance_features(q['x'],d['x'][:,0].min(),np.ptp(d['x'][:,0]))
            tag=f'f{fold}_q{frac}';torch.save(dict(base=base,error=model,features=z,raw_error=err),out/f'{tag}.pt')
            scores={'ridge':summarize(q['y'],q['groups'],dist['mean'],dist['q05'],dist['q95'])}
            saved[tag+'_y']=q['y'];saved[tag+'_groups']=q['groups']
            for key,value in [('mean',dist['mean']),('lo',dist['q05']),('hi',dist['q95'])]:saved[tag+'_ridge_'+key]=value
            for mode in ('spread_only','global','conditional'):
                fm=replace(model,residual=err) if mode in ('spread_only','global') else model
                draws=dist['mean'][:,None]*factors(fm,z,mode)
                p=draws.mean(1);lo,hi=np.quantile(draws,[.05,.95],axis=1)
                scores[mode]=summarize(q['y'],q['groups'],p,lo,hi)
                for k,v in [('mean',p),('lo',lo),('hi',hi)]:saved[tag+'_'+mode+'_'+k]=v
            episodes.append(dict(fold=fold,fraction=frac,scores=scores))
            audit.append(dict(fold=fold,fraction=frac,outer_query_units=np.unique(q['groups']).tolist(),inner=inner_audit,
                error_model=model.audit,coef=model.coef.tolist()))
            print(tag,{k:round(v['unit_mse'],4) for k,v in scores.items()},flush=True)
    aggregate={k:{v:float(np.mean([e['scores'][k][v] for e in episodes])) for v in ('unit_mse','coverage','width','interval_score')} for k in scores}
    worst={k:max(max(e['scores'][k]['unit_mse_values']) for e in episodes) for k in scores}
    c,b,s=aggregate['conditional'],aggregate['ridge'],aggregate['spread_only']
    accepted=c['unit_mse']<=b['unit_mse'] and c['coverage']>=.9 and c['interval_score']<min(b['interval_score'],s['interval_score']) and worst['conditional']<=worst['ridge']
    np.savez_compressed(out/'predictions.npz',**saved)
    write(out/'source_audit.json',audit)
    write(out/'results.json',dict(complete=True,episodes=episodes,aggregate=aggregate,worst_unit_mse=worst,
        source_gate_passed=accepted,successor_promoted=False))
    print('COMPLETE',aggregate,'gate',accepted,flush=True)


if __name__=='__main__':main()
