"""MICH-first fixed screen of learned function-prior sharing, not residual gates."""
from pathlib import Path
import json
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pp_extrapolation.function_prior_sharing import fit_bank,fit_sharing,predict_sharing,SharingFit
from pp_extrapolation.boundary_quotient import predict_boundary_quotient
from pp_extrapolation.ds03_prospective import sha256_file
from pp_extrapolation.relation_local_transport import validation_acceptance
from relation_local_transport_screen import subset,metrics,write
from component_transfer_screen import source_episode_indices

SEEDS=(42,43,44)
ARMS=dict(no_unit_borrowing=dict(mode='none'),fixed_sharing=dict(mode='fixed'),
    tied_sharing=dict(mode='tied'),component_sharing=dict(mode='component'),
    no_observation_update=dict(mode='component',condition=False),
    collapsed_distribution=dict(mode='component',distribution=False))


def main():
    torch.set_num_threads(2)
    out=ROOT/'results/function_prior_sharing_mich_v1';out.mkdir(exist_ok=False)
    paths=[Path(__file__),ROOT/'src/pp_extrapolation/function_prior_sharing.py']
    write(out/'protocol.json',dict(primary='component_sharing',arms=ARMS,seeds=SEEDS,
        hypothesis='learn per-coefficient donor sharing by hidden-unit lower-health transfer, preserve Gaussian uncertainty in unobserved coefficient directions',
        function='RUL(m)=integral_0^m exp(a+b*h+c*h^2)dh; log slowness clipped to [-8,8] for numerical safety',
        prior='per-unit log-RUL MAP; effective-sample Gauss-Newton Gaussian; not calibrated Bayesian posterior',
        observation='TRAIN-fitted rate proxy calibration with reliability-weighted rank-one coefficient update',
        no_unit_borrowing='generic population level and fixed slope covariance; no donor-specific coefficients',
        source='3 unit folds x .6/.8 lower-health cuts; unit priors see donor earlier-health rows only',
        training='energy loss, independent 16+16 coefficient samples, max 150 epochs, patience 35, 3 source unit batches per epoch',
        validation='64-draw mean unit MSE checkpoint; 256-draw mean external gate; no test selection',
        refit='all unit priors and proxy calibration refit on TRAIN+validation; sharing parameters frozen',
        gate='unchanged >2% unit MSE improvement, >=60% unit wins, max unit RMSE ratio <=1.05 versus full PP-X',
        limits=['already-opened MICH endpoint, retrospective development only',
                'known boundary and positive rate proxy required; not a DS03-compatible fallback',
                'function coefficients are not directly measured physical acceleration',
                'static arms repeat identically across seeds; covariance and intervals not calibrated',
                'no full benchmark coverage or equal baseline search-budget claim'],
        hashes={str(p.relative_to(ROOT)):sha256_file(p) for p in paths}))
    old=ROOT/'results/relation_local_transport_mich_v1'
    train,val=[dict(np.load(old/f'{p}_rows.npz')) for p in ('train','validation')]
    full={k:np.concatenate((train[k],val[k])) for k in train}
    assert not set(train['groups'])&set(val['groups'])
    bank=fit_bank(train);final_bank=fit_bank(full)
    torch.save(dict(train=bank,full=final_bank),out/'banks.pt')
    episodes=[];audit=[]
    for fold in range(3):
        for frac in (.6,.8):
            donor,query,cutoff,held=source_episode_indices(train,'mich',fold,frac)
            a,q=subset(train,donor),subset(train,query)
            assert a['x'][:,0].min()>q['x'][:,0].max()
            assert not set(a['groups'])&set(q['groups'])
            eb=fit_bank(a);episodes.append(dict(bank=eb,query=q))
            audit.append(dict(fold=fold,fraction=frac,cutoff=cutoff,held_units=held.tolist(),
                donor_indices=np.flatnonzero(donor).tolist(),query_indices=np.flatnonzero(query).tolist(),bank=eb.audit))
            print('BANK',fold,frac,'proxy reliability',round(eb.evidence_reliability,4),flush=True)
    torch.save(episodes,out/'source_episodes.pt')
    write(out/'source_audit.json',dict(episodes=audit,train=bank.audit,full=final_bank.audit))
    bases=torch.load(old/'baseline_models.pt',weights_only=False)['selected']
    baseline_val=np.array([predict_boundary_quotient(f,val) for f in bases])
    selected={};models={};vp={};vl={}
    for arm,config in ARMS.items():
        models[arm]=[];vp[arm]=[];vl[arm]=[]
        for seed in SEEDS:
            start=time.monotonic();fit=fit_sharing(bank,episodes,val,seed=seed,**config)
            dist=predict_sharing(fit,val)
            vp[arm].append(dist['mean']);vl[arm].append(fit.selection)
            models[arm].append(SharingFit(fit.model,final_bank,fit.selection))
            torch.save(fit,out/f'{arm}_seed{seed}_selected.pt')
            torch.save(models[arm][-1],out/f'{arm}_seed{seed}_refit.pt')
            print('FIT',arm,seed,'epoch',fit.selection['selected_epoch'],'seconds',round(time.monotonic()-start,1),flush=True)
        vp[arm]=np.array(vp[arm])
        selected[arm]=validation_acceptance(val['y'],val['groups'],baseline_val.mean(0),vp[arm].mean(0))
        write(out/'selection.partial.json',dict(choices=selected,logs=vl))
    write(out/'selection.json',dict(choices=selected,logs=vl))
    np.savez_compressed(out/'validation_predictions.npz',baseline=baseline_val,**vp)
    # Reopen historical test only after every selection is frozen.
    test=dict(np.load(old/'test_rows.npz'));previous=dict(np.load(old/'test_predictions.npz'))
    assert not set(test['groups'])&(set(train['groups'])|set(val['groups']))
    assert test['x'][:,0].max()<min(train['x'][:,0].min(),val['x'][:,0].min())
    scale=json.loads((ROOT/'results/prefix_speed_integral_screen_v1/scales.json').read_text())['mich']['time_scale']
    predictions={k:previous[k] for k in ('ppx','engression')};distribution={}
    for arm in ARMS:
        d=[predict_sharing(f,test) for f in models[arm]]
        predictions[arm]=np.array([x['mean'] for x in d])
        predictions[arm+'_guarded']=predictions[arm] if selected[arm]['accepted'] else predictions['ppx'].copy()
        samples=np.concatenate([x['samples'] for x in d],axis=1)
        lo,hi=np.quantile(samples,[.05,.95],axis=1)
        distribution[arm]=dict(interval90_empirical_coverage=float(np.mean((test['y']>=lo)&(test['y']<=hi))),
            mean_width=float(np.mean(hi-lo)*scale),calibration_guarantee=False)
        np.savez_compressed(out/f'{arm}_test_distribution.npz',samples=samples,q05=lo,q95=hi)
    np.savez_compressed(out/'test_predictions.npz',**predictions)
    scores={name:metrics(test['y']*scale,test['groups'],p*scale) for name,p in predictions.items()}
    primary=scores['component_sharing_guarded'];gates={}
    for name in ('ppx','engression'):
        b=scores[name];gates[name]=dict(lower_rmse=primary['ensemble']['rmse']<b['ensemble']['rmse'],
            no_worst_unit_loss=primary['ensemble']['worst_unit_rmse']<=b['ensemble']['worst_unit_rmse'],
            no_unit_coverage_loss=primary['ensemble']['positive_units']>=b['ensemble']['positive_units'],
            no_seed_coverage_loss=primary['positive_seeds']>=b['positive_seeds'])
    write(out/'results.json',dict(complete=True,scores=scores,choices=selected,gates=gates,
        screen_passed=all(all(v.values()) for v in gates.values()),successor_promoted=False,
        predictive_intervals=distribution,physical_unit_scale=scale,strict_lower_health_test_fraction=1.))
    print('COMPLETE',selected['component_sharing'],flush=True)


if __name__=='__main__':main()
