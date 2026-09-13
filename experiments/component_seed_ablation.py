"""5 base seeds x 5 gate initializations; fixed primary, no best-seed selection."""
from pathlib import Path
import sys
import json
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pp_extrapolation.component_ablation import fit_ablation
from pp_extrapolation.component_transfer import make_bank,component_features,predict_components
from pp_extrapolation.ds03_prospective import _predict_direct,sha256_file
from pp_extrapolation.relation_local_transport import validation_acceptance
from component_transfer_screen import crossfit_episodes,source_episode_indices,envelope
from relation_local_transport_screen import metrics,write,subset

SEEDS=(42,43,44,45,46)
MODES=('component_cf','scalar_cf','component_insample')


def paired_units(y,groups,candidate,baseline):
    """Exploratory paired unit bootstrap, not independent-cohort confirmation."""
    c=np.array([np.mean((candidate[groups==g]-y[groups==g])**2) for g in np.unique(groups)])
    b=np.array([np.mean((baseline[groups==g]-y[groups==g])**2) for g in np.unique(groups)])
    lr=.5*np.log(np.maximum(c,1e-12)/np.maximum(b,1e-12))
    rng=np.random.default_rng(20260913)
    boot=lr[rng.integers(0,len(lr),(10000,len(lr)))].mean(1)
    return dict(mean_unit_log_rmse_ratio=float(lr.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),
                winning_units=int(np.sum(c<b)),unit_count=len(lr),conditional_on_fitted_models=True,
                independent_cohort_confirmation=False)


def main():
    torch.set_num_threads(2)
    out=ROOT/'results/component_seed_ablation_ds03_v1';out.mkdir(exist_ok=False)
    old=ROOT/'results/component_transfer_ds03_v1'
    rows=ROOT/'results/relation_local_transport_ds03_v1'
    paths=[Path(__file__),ROOT/'src/pp_extrapolation/component_ablation.py',ROOT/'src/pp_extrapolation/component_transfer.py',
           ROOT/'experiments/component_transfer_screen.py',ROOT/'src/pp_extrapolation/ds03_prospective.py']
    write(out/'protocol.json',dict(base_seeds=SEEDS,gate_initializations=SEEDS,modes=MODES,
        primary='component_cf diagonal: base seed equals gate seed, ensemble all five',
        initializer_slices='five ensembles of five base models, one per fixed gate seed; report all, do not select',
        test_use='historical DS03 development only; no cohort or cycle-support extrapolation confirmation',
        matching='same source episode rows, optimizer, epochs, validation; scalar ties logits with same allocated 147 parameters but lower effective dimension',
        insample_control='same donor/query rows but base trained on full TRAIN including query units and late support; not independent source evaluation',
        guard='unchanged validation ensemble unit MSE >2% better, >=60% unit wins, max unit RMSE ratio <=1.05',
        uncertainty='10000 paired unit bootstrap draws; six units in one cohort, conditional on fitted models',
        hashes={str(p.relative_to(ROOT)):sha256_file(p) for p in paths}))
    train,val=[dict(np.load(rows/f'{p}_rows.npz')) for p in ('train','validation')]
    basefits=torch.load(ROOT/'results/ncmapss_ds03_ppx_v1/selection.models.pt',weights_only=False)['fits']['direct_fallback']
    assert len(basefits)==5
    matrices={m:np.empty((5,5,len(val['y']))) for m in MODES}
    logs={};banks=[];models={}
    baseline_val=[]
    for bi,seed in enumerate(SEEDS):
        pa=_predict_direct(basefits[bi],train['x']);pv=_predict_direct(basefits[bi],val['x'])
        bank=make_bank(train,pa);banks.append(bank);validation=envelope(val,pv,bank);baseline_val.append(pv)
        np.savez_compressed(out/f'validation_base{seed}.npz',**validation)
        if seed<=44:
            episodes=[dict(np.load(old/f'seed{seed}_fold{f}_cut{cut}.npz')) for f in range(3) for cut in (60,80)]
        else:
            episodes,records=crossfit_episodes('ds03',train,None,seed,out)
            write(out/f'episodes_seed{seed}.json',records)
        insample=[]
        for f in range(3):
            for frac in (.6,.8):
                donor,query,_,_=source_episode_indices(train,'ds03',f,frac)
                local=make_bank(subset(train,donor),pa[donor])
                e=envelope(subset(train,query),pa[query],local,tag=f'{f}:{frac}:')
                insample.append(e)
        np.savez_compressed(out/f'insample_base{seed}.npz',**{k:np.concatenate([e[k] for e in insample]) for k in insample[0]})
        for mode in MODES:
            for gi,init in enumerate(SEEDS):
                fit=fit_ablation(insample if mode=='component_insample' else episodes,validation,
                    seed=init,shared=mode=='scalar_cf',cap=float(basefits[bi]['cap']))
                key=f'{mode}_base{seed}_init{init}'
                matrices[mode][bi,gi]=predict_components(fit,validation['c'],validation['z'],pv)
                models[key]=fit;logs[key]=fit.selection
                torch.save(fit,out/f'{key}.pt')
            print('FIT',mode,'base',seed,'epochs',[logs[f'{mode}_base{seed}_init{s}']['selected_epoch'] for s in SEEDS],flush=True)
        # Existing first three primary runs must replay; this is an extension, not a replacement.
        if seed<=44:
            reference=dict(np.load(old/'validation_predictions.npz'))['robust_components'][bi]
            np.testing.assert_allclose(matrices['component_cf'][bi,bi],reference,rtol=1e-7,atol=1e-7)
    baseline_val=np.array(baseline_val)
    choices={}
    for mode,mat in matrices.items():
        slices={'diagonal':np.array([mat[i,i] for i in range(5)])}
        slices.update({f'init{s}':mat[:,j] for j,s in enumerate(SEEDS)})
        for label,vp in slices.items():
            choices[f'{mode}_{label}']=validation_acceptance(val['y'],val['groups'],baseline_val.mean(0),vp.mean(0))
    write(out/'selection.json',dict(choices=choices,logs=logs))
    np.savez_compressed(out/'validation_predictions.npz',baseline=baseline_val,**matrices)
    torch.save(banks,out/'banks.pt')
    # No test array has been loaded above this line in this runner.
    test=dict(np.load(rows/'test_rows.npz'))
    baseline=np.array([_predict_direct(f,test['x']) for f in basefits])
    eng=dict(np.load(ROOT/'results/ncmapss_ds03_equal_budget_v1/engression/predictions.npz'))
    for k in ('y','groups'):np.testing.assert_array_equal(test[k],eng[k])
    predictions={m:np.empty((5,5,len(test['y']))) for m in MODES}
    for bi,seed in enumerate(SEEDS):
        c,z=component_features(banks[bi],test['x'],baseline[bi])
        np.savez_compressed(out/f'test_components_base{seed}.npz',c=c,z=z,base=baseline[bi])
        for mode in MODES:
            for gi,init in enumerate(SEEDS):
                predictions[mode][bi,gi]=predict_components(models[f'{mode}_base{seed}_init{init}'],c,z,baseline[bi])
    np.savez_compressed(out/'factorial_predictions.npz',baseline=baseline,engression=eng['prediction'],**predictions)
    scores={'ppx':metrics(test['y'],test['groups'],baseline),'engression':metrics(test['y'],test['groups'],eng['prediction'])}
    guarded={};raw={}
    for mode,mat in predictions.items():
        slices={'diagonal':np.array([mat[i,i] for i in range(5)])}
        slices.update({f'init{s}':mat[:,j] for j,s in enumerate(SEEDS)})
        for label,tp in slices.items():
            key=f'{mode}_{label}';raw[key]=tp
            guarded[key]=tp if choices[key]['accepted'] else baseline.copy()
            scores[key]=metrics(test['y'],test['groups'],guarded[key])
    primary=guarded['component_cf_diagonal']
    uncertainty={name:paired_units(test['y'],test['groups'],primary.mean(0),b.mean(0)) for name,b in [('ppx',baseline),('engression',eng['prediction'])]}
    uncertainty['scalar_ablation']=paired_units(test['y'],test['groups'],primary.mean(0),guarded['scalar_cf_diagonal'].mean(0))
    uncertainty['insample_ablation']=paired_units(test['y'],test['groups'],primary.mean(0),guarded['component_insample_diagonal'].mean(0))
    result=dict(complete=True,scores=scores,raw_scores={k:metrics(test['y'],test['groups'],p) for k,p in raw.items()},
        choices=choices,uncertainty=uncertainty,successor_promoted=False,
        activation={m:[[logs[f'{m}_base{b}_init{g}']['enabled'] for g in SEEDS] for b in SEEDS] for m in MODES})
    np.savez_compressed(out/'guarded_predictions.npz',**guarded)
    write(out/'results.json',result)
    print('COMPLETE',scores['component_cf_diagonal']['ensemble'],flush=True)


if __name__=='__main__':main()
