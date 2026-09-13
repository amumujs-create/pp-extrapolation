"""Replay source exclusion, component construction, gates, and test metrics."""
import json
import numpy as np
import torch
from component_transfer_screen import ROOT, ARMS, SEEDS, source_episode_indices
from pp_extrapolation.component_transfer import component_features, predict_components
from pp_extrapolation.boundary_quotient import predict_boundary_quotient
from pp_extrapolation.ds03_prospective import _predict_direct, sha256_file
from pp_extrapolation.relation_local_transport import validation_acceptance
from relation_local_transport_screen import metrics, write
from verify_relation_local_transport_screen import close_tree


def verify(case):
    out=ROOT/f'results/component_transfer_{case}_v1'
    old=ROOT/f'results/relation_local_transport_{case}_v1'
    protocol=json.loads((out/'protocol.json').read_text())
    for name, expected in protocol['hashes'].items():
        actual=sha256_file(ROOT/name)
        if actual!=expected:
            fix=json.loads((out/'execution_fix.json').read_text())
            assert name==fix['path'] and expected==fix['previous_sha256'] and actual==fix['current_sha256']
            assert not fix['retrained'] and not fix['selection_changed']
    train=dict(np.load(old/'train_rows.npz')); val=dict(np.load(old/'validation_rows.npz')); test=dict(np.load(old/'test_rows.npz'))
    selection=json.loads((out/'selection.json').read_text())
    result=json.loads((out/'results.json').read_text())
    vpred=dict(np.load(out/'validation_predictions.npz')); tpred=dict(np.load(out/'test_predictions.npz'))
    banks=torch.load(out/'banks.pt',weights_only=False)
    episode_count=0
    for seed in SEEDS:
        for fold in range(3):
            for fraction in (.6,.8):
                identifier=f'seed{seed}_fold{fold}_cut{int(100*fraction)}'
                saved=dict(np.load(out/f'{identifier}.npz'))
                bundle=torch.load(out/f'{identifier}.pt',weights_only=False)
                a,q=bundle['anchors'],bundle['queries']
                dmask,qmask,cutoff,held=source_episode_indices(train,case,fold,fraction)
                np.testing.assert_array_equal(a['x'],train['x'][dmask]);np.testing.assert_array_equal(q['x'],train['x'][qmask])
                assert not set(a['groups']) & set(q['groups'])
                if case=='mich':
                    assert a['x'][:,0].min()>q['x'][:,0].max()
                    base=predict_boundary_quotient(bundle['base_model'],q)
                    record=next(r for r in selection['episodes'] if r['id']==identifier)
                    assert not {f'2:{u}' for u in held} & set(record['base_train_units'])
                else:
                    assert a['x'][:,0].max()<q['x'][:,0].min()
                    base=_predict_direct(bundle['base_model'],q['x'])
                np.testing.assert_array_equal(base,saved['base'])
                c,z=component_features(bundle['bank'],q['x'],base)
                np.testing.assert_allclose(c,saved['c'],atol=1e-10,rtol=1e-10)
                np.testing.assert_allclose(z,saved['z'],atol=1e-10,rtol=1e-10)
                np.testing.assert_array_equal(q['y'],saved['y'])
                episode_count+=1
    for i,seed in enumerate(SEEDS):
        v=dict(np.load(out/f'validation_seed{seed}.npz'))
        saved=dict(np.load(out/f'test_components_seed{seed}.npz'))
        c,z=component_features(banks[i],test['x'],saved['base'])
        np.testing.assert_allclose(c,saved['c'],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(z,saved['z'],rtol=1e-10,atol=1e-10)
        for arm in ARMS:
            fit=torch.load(out/f'{arm}_seed{seed}.pt',weights_only=False)
            np.testing.assert_array_equal(predict_components(fit,c,z,saved['base']),tpred[arm][i])
            np.testing.assert_array_equal(predict_components(fit,v['c'],v['z'],v['base']),vpred[arm][i])
            if case=='mich':
                zero=test['x'][:2].copy();zero[:,0]=0
                cz,zz=component_features(banks[i],zero,np.zeros(2))
                np.testing.assert_array_equal(predict_components(fit,cz,zz,np.zeros(2)),0.)
    for arm in (*ARMS,'unshrunk_residual'):
        gate=validation_acceptance(val['y'],val['groups'],vpred['baseline'].mean(0),vpred[arm].mean(0))
        close_tree(gate,selection['decisions'][arm])
        np.testing.assert_array_equal(tpred[arm+'_guarded'],tpred[arm] if gate['accepted'] else tpred['ppx'])
    for name,mat in tpred.items():
        s=result['physical_unit_scale']
        close_tree(metrics(test['y']*s,test['groups'],mat*s),result['scores'][name])
    report=dict(verified=True,source_episode_replays=episode_count,gate_model_replays=9,
                metric_matrices=len(tpred),unit_support_exclusion=True,source_hashes_checked=True)
    write(out/'verification.json',report)
    print(case,report,flush=True)


def verify_clock():
    from pp_extrapolation.clock_factor_fallback import predict_clock, audit_clock
    out=ROOT/'results/clock_factor_fallback_ds03_v1'
    old=ROOT/'results/relation_local_transport_ds03_v1'
    protocol=json.loads((out/'protocol.json').read_text())
    for name,expected in protocol['hashes'].items():
        assert sha256_file(ROOT/name)==expected
    train,val,test=[dict(np.load(old/f'{p}_rows.npz')) for p in ('train','validation','test')]
    audit=audit_clock(train)
    result=json.loads((out/'results.json').read_text())
    selection=json.loads((out/'selection.json').read_text())
    vp=dict(np.load(out/'validation_predictions.npz'));tp=dict(np.load(out/'test_predictions.npz'))
    base=dict(np.load(old/'validation_predictions.npz'))['baseline']
    for arm in ('clock','clock_consistent'):
        fits=torch.load(out/f'{arm}.pt',weights_only=False)
        for i,fit in enumerate(fits):
            close_tree(audit,fit['selection']['clock_audit'])
            np.testing.assert_array_equal(predict_clock(fit,val['x']),vp[arm][i])
            np.testing.assert_array_equal(predict_clock(fit,test['x']),tp[arm][i])
        gate=validation_acceptance(val['y'],val['groups'],base.mean(0),vp[arm].mean(0))
        close_tree(gate,selection['decisions'][arm])
        np.testing.assert_array_equal(tp[arm+'_guarded'],tp[arm] if gate['accepted'] else tp['ppx'])
    for name,mat in tp.items():
        close_tree(metrics(test['y'],test['groups'],mat),result['scores'][name])
    report=dict(verified=True,checkpoint_replays=6,metric_matrices=6,clock_train_audit=audit)
    write(out/'verification.json',report)
    print('clock control',report,flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    for case in ('mich','ds03'):
        verify(case)
    verify_clock()
