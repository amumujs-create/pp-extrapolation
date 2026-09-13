"""Replay every factorial model, policies, baseline alignment, and uncertainty."""
import json
import numpy as np
import torch
from component_seed_ablation import ROOT,SEEDS,MODES,paired_units
from pp_extrapolation.component_transfer import component_features,predict_components
from pp_extrapolation.ds03_prospective import _predict_direct,sha256_file
from pp_extrapolation.relation_local_transport import validation_acceptance
from relation_local_transport_screen import metrics,write
from verify_relation_local_transport_screen import close_tree


def main():
    torch.set_num_threads(2)
    out=ROOT/'results/component_seed_ablation_ds03_v1'
    old=ROOT/'results/relation_local_transport_ds03_v1'
    protocol=json.loads((out/'protocol.json').read_text())
    for path,expected in protocol['hashes'].items():assert sha256_file(ROOT/path)==expected
    vp=dict(np.load(out/'validation_predictions.npz'));tp=dict(np.load(out/'factorial_predictions.npz'))
    gp=dict(np.load(out/'guarded_predictions.npz'))
    val,test=[dict(np.load(old/f'{p}_rows.npz')) for p in ('validation','test')]
    sel=json.loads((out/'selection.json').read_text());result=json.loads((out/'results.json').read_text())
    banks=torch.load(out/'banks.pt',weights_only=False)
    bases=torch.load(ROOT/'results/ncmapss_ds03_ppx_v1/selection.models.pt',weights_only=False)['fits']['direct_fallback']
    eng=dict(np.load(ROOT/'results/ncmapss_ds03_equal_budget_v1/engression/predictions.npz'))
    for key in ('y','groups'):np.testing.assert_array_equal(test[key],eng[key])
    np.testing.assert_array_equal(tp['engression'],eng['prediction'])
    for bi,seed in enumerate(SEEDS):
        np.testing.assert_array_equal(tp['baseline'][bi],_predict_direct(bases[bi],test['x']))
        np.testing.assert_array_equal(vp['baseline'][bi],_predict_direct(bases[bi],val['x']))
        c,z=component_features(banks[bi],test['x'],tp['baseline'][bi])
        v=dict(np.load(out/f'validation_base{seed}.npz'))
        for mode in MODES:
            for gi,init in enumerate(SEEDS):
                fit=torch.load(out/f'{mode}_base{seed}_init{init}.pt',weights_only=False)
                np.testing.assert_array_equal(predict_components(fit,c,z,tp['baseline'][bi]),tp[mode][bi,gi])
                np.testing.assert_array_equal(predict_components(fit,v['c'],v['z'],v['base']),vp[mode][bi,gi])
                assert fit.enabled==result['activation'][mode][bi][gi]
    for mode in MODES:
        for label,idx in [('diagonal',None)]+[(f'init{s}',j) for j,s in enumerate(SEEDS)]:
            v=np.array([vp[mode][i,i] for i in range(5)]) if idx is None else vp[mode][:,idx]
            t=np.array([tp[mode][i,i] for i in range(5)]) if idx is None else tp[mode][:,idx]
            key=f'{mode}_{label}'
            choice=validation_acceptance(val['y'],val['groups'],vp['baseline'].mean(0),v.mean(0))
            close_tree(choice,sel['choices'][key])
            np.testing.assert_array_equal(gp[key],t if choice['accepted'] else tp['baseline'])
            close_tree(metrics(test['y'],test['groups'],gp[key]),result['scores'][key])
            close_tree(metrics(test['y'],test['groups'],t),result['raw_scores'][key])
    primary=gp['component_cf_diagonal'].mean(0)
    comparisons={'ppx':tp['baseline'].mean(0),'engression':tp['engression'].mean(0),
        'scalar_ablation':gp['scalar_cf_diagonal'].mean(0),'insample_ablation':gp['component_insample_diagonal'].mean(0)}
    for key,b in comparisons.items():close_tree(paired_units(test['y'],test['groups'],primary,b),result['uncertainty'][key])
    report=dict(verified=True,models_replayed=75,validation_and_test_replayed=True,
        baseline_models=5,policy_slices_recomputed=18,uncertainty_comparisons_recomputed=4,
        snapshot_phase='post-run integrity snapshot, not pre-training registration',
        saved_file_hashes={p.name:sha256_file(p) for p in sorted(out.iterdir()) if p.suffix in ('.pt','.npz','.json') and p.name!='verification.json'})
    write(out/'verification.json',report)
    print({k:v for k,v in report.items() if k!='saved_file_hashes'},flush=True)


if __name__=='__main__':main()
