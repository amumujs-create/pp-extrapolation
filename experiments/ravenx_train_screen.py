"""Locked TRAIN-only RAVEN-X screen. Never imports validation/test artifacts."""
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pp_extrapolation.function_prior_sharing import fit_bank
from pp_extrapolation.prefix_relation_encoder import fit_prefix
from pp_extrapolation.regime_validity_attention import prepare_windows,fit_raven,predictive_summary
from pp_extrapolation.piecewise_health_path import no_switch_rul
from relation_local_transport_screen import subset,write
from component_transfer_screen import source_episode_indices
from function_prior_train_diagnosis import score
from transfer_error_distribution_screen import summarize

SEEDS=(42,43,44,45,46)
ARMS={
    'ordinary_attention':dict(config=dict(attention='softmax',transitions=False),counterfactual=False),
    'regime_no_validity':dict(config=dict(attention='sparse',validity=False),counterfactual=True),
    'unrestricted_transition':dict(config=dict(generator='all'),counterfactual=True),
    'no_window_counterfactual':dict(config=dict(),counterfactual=False),
    'independent_gaussian':dict(config=dict(generator='gaussian'),counterfactual=True),
    'point_transition':dict(config=dict(point_transition=True),counterfactual=True),
    'independent_horizon':dict(config=dict(coherent=False),counterfactual=True),
    'ravenx':dict(config=dict(),counterfactual=True),
}


def sample_summary(draws,y):
    mean=draws.mean(1);lo,hi=np.quantile(draws,[.05,.95],axis=1);crps=[]
    for values,target in zip(draws,y):
        v=np.sort(values);n=len(v);half_pair=np.sum(v*(2*np.arange(1,n+1)-n-1))/n**2
        crps.append(np.mean(np.abs(v-target))-half_pair)
    return mean,lo,hi,np.asarray(crps)


def ridge_summary(p,y,samples=1024):
    u=torch.quasirandom.SobolEngine(3,scramble=True,seed=1729).draw(samples).double().clamp(1e-9,1-1e-9)
    noise=torch.erfinv(2*u-1)*np.sqrt(2);chol=torch.linalg.cholesky(p.base_cov);chunks=[]
    with torch.no_grad():
        for start in range(0,len(p.margin),16):
            theta=p.theta[start:start+16,3,None]+torch.einsum('ij,sj->si',chol,noise)[None]
            chunks.append(no_switch_rul(theta,p.margin[start:start+16]).numpy())
    draws=np.concatenate(chunks);mean,lo,hi,crps=sample_summary(draws,y)
    return dict(mean=mean,q05=lo,q95=hi,crps=crps)


def mixture_summary(distributions,y):
    support=np.concatenate([v['support'] for v in distributions],1)
    weight=np.concatenate([v['support_weight']/len(distributions) for v in distributions],1)
    mean=(support*weight).sum(1);lo=[];hi=[];crps=[]
    for values,w,target in zip(support,weight,y):
        order=np.argsort(values);v=values[order];sw=w[order];cw=np.cumsum(sw);cw[-1]=1
        lo.append(v[np.searchsorted(cw,.05)]);hi.append(v[np.searchsorted(cw,.95)])
        first=np.sum(sw*np.abs(v-target));half_pair=np.sum(sw*v*(2*(cw-sw)+sw-1));crps.append(first-half_pair)
    return dict(mean=mean,q05=np.asarray(lo),q95=np.asarray(hi),crps=np.asarray(crps))


def full_score(y,g,d):
    answer=summarize(y,g,d['mean'],d['q05'],d['q95']);answer['crps']=float(np.mean(d['crps']))
    positive=0
    for unit in np.unique(g):
        ix=g==unit;den=np.sum((y[ix]-y[ix].mean())**2)
        positive+=int(den>0 and 1-np.sum((y[ix]-d['mean'][ix])**2)/den>0)
    answer['positive_units']=positive
    return answer


def main():
    torch.set_num_threads(2);out=ROOT/'results/ravenx_train_screen_v1';out.mkdir(exist_ok=False)
    source_path=ROOT/'results/relation_local_transport_mich_v1/train_rows.npz'
    files=(Path(__file__),ROOT/'src/pp_extrapolation/regime_validity_attention.py',
        ROOT/'src/pp_extrapolation/relation_intervention_generator.py',ROOT/'src/pp_extrapolation/piecewise_health_path.py',
        ROOT/'protocols/RAVENX_TRAIN_SCREEN_PROTOCOL.md')
    write(out/'protocol.json',dict(primary='ravenx',seeds=SEEDS,arms=ARMS,steps=80,
        evaluation_samples='ridge 1024 Sobol; learned arms 256 Sobol per seed, 1280-draw equal seed mixture',
        data='MICH TRAIN only; outer and inner 3 unit folds x .6/.8 strict lower-health queries',
        context='held-unit observed X prefix allowed causally; held-unit RUL excluded from fitting',
        gate='protocols/RAVENX_TRAIN_SCREEN_PROTOCOL.md',
        deviations=['transition-neighborhood metrics use an observed-feature proxy because true regime labels are unavailable',
                    'synthetic falsification is a separate required artifact and cannot be replaced by real-data scores'],
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source_path,)+files}))
    rows=dict(np.load(source_path));episodes=[];audit=[];saved={}
    for outer_fold in range(3):
        for outer_fraction in (.6,.8):
            donor,query,cut,held=source_episode_indices(rows,'mich',outer_fold,outer_fraction)
            d,q=subset(rows,donor),subset(rows,query)
            assert not set(d['groups'])&set(q['groups']) and d['x'][:,0].min()>q['x'][:,0].max()
            inner=[];inner_audit=[]
            for fold in range(3):
                for fraction in (.6,.8):
                    a,b,icut,iheld=source_episode_indices(d,'mich',fold,fraction);src,iq=subset(d,a),subset(d,b)
                    assert not set(src['groups'])&set(iq['groups']) and src['x'][:,0].min()>iq['x'][:,0].max()
                    bank=fit_bank(src);base=fit_prefix(src,bank,neural=False)
                    prepared=prepare_windows(base,bank,src,d,iq)
                    inner.append(dict(prepared=prepared,y=iq['y'],groups=iq['groups']))
                    inner_audit.append(dict(fold=fold,fraction=fraction,source_units=np.unique(src['groups']).tolist(),
                        query_units=np.unique(iq['groups']).tolist(),source_min=float(src['x'][:,0].min()),
                        query_max=float(iq['x'][:,0].max()),query_labels_fit=False))
            bank=fit_bank(d);base=fit_prefix(d,bank,neural=False);prepared=prepare_windows(base,bank,d,rows,q)
            tag=f'f{outer_fold}_q{outer_fraction}';torch.save(dict(inner=inner,outer=prepared),out/f'{tag}_prepared.pt')
            distributions={'ridge':ridge_summary(prepared,q['y'])};logs={};attention={}
            for arm,configuration in ARMS.items():
                predictions=[];logs[arm]=[];attention[arm]=[]
                for seed in SEEDS:
                    start=time.monotonic();model,log=fit_raven(inner,configuration['config'],seed,80,configuration['counterfactual'])
                    dist=predictive_summary(model,prepared,256,1729,q['y'],return_support=True);predictions.append(dist);logs[arm].append(log)
                    attention[arm].append(dict(mean_path_probability=dist['path_probability'].mean(0).tolist(),
                        mean_attention=dist['attention'].mean(0).tolist(),edge_gate=dist['edge_gate'].tolist()))
                    torch.save(model,out/f'{tag}_{arm}_seed{seed}.pt')
                    print('FIT',tag,arm,seed,'seconds',round(time.monotonic()-start,1),flush=True)
                distributions[arm]=mixture_summary(predictions,q['y'])
            scores={k:full_score(q['y'],q['groups'],v) for k,v in distributions.items()}
            transition_score=np.abs(q['x'][:,4])+np.abs(q['x'][:,9]);threshold=float(np.quantile(np.abs(d['x'][:,4])+np.abs(d['x'][:,9]),.75))
            strata={}
            for k,v in distributions.items():
                strata[k]={}
                for name,mask in [('transition_proxy',transition_score>=threshold),('stable_proxy',transition_score<threshold)]:
                    strata[k][name]=None if not mask.any() else float(np.sqrt(np.mean((q['y'][mask]-v['mean'][mask])**2)))
            episodes.append(dict(fold=outer_fold,fraction=outer_fraction,coordinate_cutoff=cut,held_units=held.tolist(),
                scores=scores,strata_rmse=strata,training=logs,diagnostics=attention,
                interval_note='exact weighted empirical quantiles of equal-seed, path-weighted support'))
            audit.append(dict(fold=outer_fold,fraction=outer_fraction,outer_source_units=np.unique(d['groups']).tolist(),
                outer_query_units=np.unique(q['groups']).tolist(),source_min=float(d['x'][:,0].min()),
                query_max=float(q['x'][:,0].max()),inner=inner_audit))
            saved[tag+'_y']=q['y'];saved[tag+'_groups']=q['groups']
            for k,v in distributions.items():
                for field in ('mean','q05','q95','crps'):saved[tag+'_'+k+'_'+field]=v[field]
            write(out/'partial.json',dict(episodes=episodes))
            print('SCORE',tag,{k:round(v['unit_mse'],4) for k,v in scores.items()},flush=True)
    names=list(episodes[0]['scores']);metrics=('unit_mse','coverage','width','interval_score','crps','positive_units')
    aggregate={k:{m:float(np.mean([e['scores'][k][m] for e in episodes])) for m in metrics} for k in names}
    worst={k:max(max(e['scores'][k]['unit_mse_values']) for e in episodes) for k in names}
    r=aggregate['ravenx'];controls=[k for k in names if k!='ravenx']
    seed_wins=0
    # Seed-specific win criterion is computed from stored checkpoint means over all episodes.
    for si,seed in enumerate(SEEDS):
        rm=[];bm=[]
        for e in episodes:
            tag=f"f{e['fold']}_q{e['fraction']}";p=torch.load(out/f'{tag}_prepared.pt',weights_only=False)['outer']
            model=torch.load(out/f'{tag}_ravenx_seed{seed}.pt',weights_only=False)
            rm.append(score(saved[tag+'_y'],predictive_summary(model,p,256,1729)['mean'],saved[tag+'_groups'])['unit_mse'])
            bm.append(e['scores']['ridge']['unit_mse'])
        seed_wins+=int(np.mean(rm)<.98*np.mean(bm))
    accepted=(r['unit_mse']<.98*aggregate['ridge']['unit_mse'] and
        all(r['unit_mse']<.95*aggregate[k]['unit_mse'] for k in controls if k!='ridge') and
        all(r['crps']<aggregate[k]['crps'] for k in controls) and worst['ravenx']<=worst['ridge'] and
        .85<=r['coverage']<=.95 and r['interval_score']<aggregate['ridge']['interval_score'] and
        r['positive_units']>=aggregate['ridge']['positive_units'] and seed_wins>=4)
    np.savez_compressed(out/'predictions.npz',**saved);write(out/'source_audit.json',audit)
    write(out/'results.json',dict(complete=True,aggregate=aggregate,worst_unit_mse=worst,episodes=episodes,
        seed_wins_over_ridge=seed_wins,real_data_gate_passed=accepted,synthetic_gate_passed=None,
        source_gate_passed=False,successor_promoted=False,
        decision='source gate cannot pass until real-data and synthetic conjunctive gates pass'))
    print('COMPLETE',json.dumps(aggregate,indent=2),'real gate',accepted,flush=True)


if __name__=='__main__':main()
