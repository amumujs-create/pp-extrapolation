"""Replay unit banks, exclusion, full distributions, scores and policy."""
import json
import numpy as np
import torch
from function_prior_sharing_screen import ROOT,ARMS,SEEDS
from pp_extrapolation.function_prior_sharing import fit_bank,predict_sharing,integral
from pp_extrapolation.ds03_prospective import sha256_file
from pp_extrapolation.relation_local_transport import validation_acceptance
from component_transfer_screen import source_episode_indices
from relation_local_transport_screen import metrics,write,subset
from verify_relation_local_transport_screen import close_tree


def main():
    torch.set_num_threads(2)
    out=ROOT/'results/function_prior_sharing_mich_v1';old=ROOT/'results/relation_local_transport_mich_v1'
    protocol=json.loads((out/'protocol.json').read_text())
    for p,h in protocol['hashes'].items():assert sha256_file(ROOT/p)==h
    train,val,test=[dict(np.load(old/f'{p}_rows.npz')) for p in ('train','validation','test')]
    banks=torch.load(out/'banks.pt',weights_only=False)
    refitted=fit_bank(train)
    np.testing.assert_allclose(refitted.means,banks['train'].means,rtol=1e-9,atol=1e-9)
    np.testing.assert_allclose(refitted.covariances,banks['train'].covariances,rtol=1e-9,atol=1e-9)
    episodes=torch.load(out/'source_episodes.pt',weights_only=False)
    for i,(fold,frac) in enumerate((f,q) for f in range(3) for q in (.6,.8)):
        a,q,_,_=source_episode_indices(train,'mich',fold,frac)
        eb=episodes[i]['bank'];query=episodes[i]['query']
        assert not set(eb.groups)&set(query['groups'])
        assert train['x'][a,0].min()>query['x'][:,0].max()
        np.testing.assert_array_equal(query['x'],train['x'][q])
        regenerated=fit_bank(subset(train,a))
        np.testing.assert_allclose(regenerated.means,eb.means,rtol=1e-9,atol=1e-9)
        np.testing.assert_allclose(regenerated.covariances,eb.covariances,rtol=1e-9,atol=1e-9)
    result=json.loads((out/'results.json').read_text());selection=json.loads((out/'selection.json').read_text())
    vp=dict(np.load(out/'validation_predictions.npz'));tp=dict(np.load(out/'test_predictions.npz'))
    for arm in ARMS:
        samples=[]
        for i,seed in enumerate(SEEDS):
            selected=torch.load(out/f'{arm}_seed{seed}_selected.pt',weights_only=False)
            final=torch.load(out/f'{arm}_seed{seed}_refit.pt',weights_only=False)
            np.testing.assert_array_equal(predict_sharing(selected,val)['mean'],vp[arm][i])
            p=predict_sharing(final,test)
            np.testing.assert_array_equal(p['mean'],tp[arm][i]);samples.append(p['samples'])
            x=test['x'][:1].copy();x[:,0]=0
            m,c,w=final.model.coefficient_distribution(final.bank,x,test['rate'][:1])
            assert torch.linalg.eigvalsh(c).min()>0
            zero=final.model.draws(final.bank,x,test['rate'][:1],torch.zeros((3,3),dtype=torch.float64))
            np.testing.assert_array_equal(zero.detach().numpy(),0.)
        samples=np.concatenate(samples,axis=1)
        saved=dict(np.load(out/f'{arm}_test_distribution.npz'))
        np.testing.assert_array_equal(samples,saved['samples'])
        np.testing.assert_array_equal(np.quantile(samples,.05,axis=1),saved['q05'])
        gate=validation_acceptance(val['y'],val['groups'],vp['baseline'].mean(0),vp[arm].mean(0))
        close_tree(gate,selection['choices'][arm])
        np.testing.assert_array_equal(tp[arm+'_guarded'],tp[arm] if gate['accepted'] else tp['ppx'])
        interval=result['predictive_intervals'][arm]
        coverage=float(np.mean((test['y']>=saved['q05'])&(test['y']<=saved['q95'])))
        np.testing.assert_allclose(coverage,interval['interval90_empirical_coverage'])
    for name,p in tp.items():
        scale=result['physical_unit_scale']
        close_tree(metrics(test['y']*scale,test['groups'],p*scale),result['scores'][name])
    assert not result['successor_promoted']
    report=dict(verified=True,unit_bank_refits=7,source_exclusion_checks=6,
        selected_and_final_checkpoint_replays=36,distribution_matrices=6,metric_matrices=len(tp),
        boundary_checked=True,covariance_positive_definite_checked=True,interval_guarantee=False)
    write(out/'verification.json',report);print(report,flush=True)


if __name__=='__main__':main()
