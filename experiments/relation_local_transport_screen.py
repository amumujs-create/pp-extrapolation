"""Retrospective two-route screen. Fixed arms; never select using test scores."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pp_extrapolation.relation_local_transport import (
    fit_relation_transport, predict_relation_transport, balanced_bank,
    group_weights, validation_acceptance,
)
from pp_extrapolation.boundary_quotient import fit_boundary_quotient_pp, predict_boundary_quotient
from pp_extrapolation.presets import battery_dual_scale_pp_config
from pp_extrapolation.ds03_prospective import (
    load_development, load_revealed_test, causal_features, _predict_direct,
    EXPECTED_SIZE, EXPECTED_SHA256, sha256_file,
)

SEEDS = (42, 43, 44)
ARMS = {
    'kernel_mean': dict(learn=False, linear=False, relation_weight=0.),
    'local_linear_fixed': dict(learn=False, linear=True, relation_weight=0.),
    'learned_point': dict(learn=True, linear=True, relation_weight=0.),
    'learned_relation': dict(learn=True, linear=True, relation_weight=.1),
}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def subset(rows, mask):
    return {k: v[mask] for k, v in rows.items()}


def metrics(y, groups, matrix):
    y, matrix = np.asarray(y, float), np.asarray(matrix, float)
    unique = np.unique(groups)

    def r2(a, p):
        total = np.sum((a-a.mean())**2)
        return float(1-np.sum((a-p)**2)/total) if total > 0 else None

    def one(p):
        rmse = [float(np.sqrt(np.mean((p[groups == g]-y[groups == g])**2))) for g in unique]
        unit_r2 = [r2(y[groups == g], p[groups == g]) for g in unique]
        return dict(r2=r2(y, p), rmse=float(np.sqrt(np.mean((p-y)**2))),
                    unit_rmse=float(np.mean(rmse)), worst_unit_rmse=max(rmse),
                    positive_units=sum(v is not None and v > 0 for v in unit_r2),
                    units={str(g): dict(rmse=a, r2=b) for g, a, b in zip(unique, rmse, unit_r2)})
    per_seed = [one(p) for p in matrix]
    return dict(ensemble=one(matrix.mean(0)), seeds=per_seed,
                positive_seeds=sum(v['r2'] is not None and v['r2'] > 0 for v in per_seed),
                finite_fraction=float(np.isfinite(matrix).mean()), rows=len(y), units=len(unique))


def memory_refit(fit, rows, prior):
    """Expand memory after validation; freeze TRAIN normalization and geometry."""
    final = copy.deepcopy(fit)
    ix = balanced_bank(rows['groups'])
    values = rows['y'].astype(float)-prior
    if final.boundary_index is not None:
        factor = rows['x'][:, final.boundary_index]
        if np.any(factor <= 0):
            raise ValueError('positive refit boundary factor required')
        values /= factor
    final.bank_indices = ix
    final.bank_x = ((rows['x']-final.center)/final.scale)[ix]
    final.bank_values = values[ix]/final.value_scale
    final.bank_weights = group_weights(rows['groups'][ix])
    return final


def prepare_mich(out):
    source = ROOT/'results/prefix_speed_integral_screen_v1'
    joint = {part: dict(np.load(source/f'{part}_rows.npz')) for part in ('train', 'validation', 'full')}
    masks = {part: rows['dataset'] == 2 for part, rows in joint.items()}
    data = {part: {**subset(rows, masks[part]), 'groups': rows['units'][masks[part]].astype(str)}
            for part, rows in joint.items()}
    selected, refitted, prior_train, prior_val, prior_full = [], [], [], [], []
    for seed in SEEDS:
        start = time.monotonic()
        fit = fit_boundary_quotient_pp(joint['train'], joint['validation'], seed=seed,
                                      **battery_dual_scale_pp_config())
        prior_train.append(predict_boundary_quotient(fit, joint['train'])[masks['train']])
        prior_val.append(predict_boundary_quotient(fit, joint['validation'])[masks['validation']])
        final = fit_boundary_quotient_pp(joint['full'], joint['full'], seed=seed,
            max_epochs=max(1, fit.selection['selected_epoch']), patience=10000,
            restore_best=False, **battery_dual_scale_pp_config())
        prior_full.append(predict_boundary_quotient(final, joint['full'])[masks['full']])
        selected.append(fit)
        refitted.append(final)
        print('MICH baseline', seed, 'seconds', round(time.monotonic()-start, 1), flush=True)
    torch.save(dict(selected=selected, refitted=refitted), out/'baseline_models.pt')

    def reveal():
        test_joint = dict(np.load(source/'test_rows.npz'))
        mask = test_joint['dataset'] == 2
        test = {**subset(test_joint, mask), 'groups': test_joint['units'][mask].astype(str)}
        baseline = np.array([predict_boundary_quotient(f, test_joint)[mask] for f in refitted])
        archive = ROOT/'results/mich_pool_boundary_refit_factorial_v1'
        saved = dict(np.load(archive/'joint_all_bq_dual_refit1.npz'))
        eng = dict(np.load(archive/'joint_all_engression_refit1.npz'))
        for a in (saved, eng):
            for k in ('y', 'dataset', 'units'):
                np.testing.assert_array_equal(test_joint[k], a[k])
        np.testing.assert_allclose(baseline, saved['prediction'][:, mask], atol=1e-6, rtol=1e-6)
        return test, baseline, eng['prediction'][:, mask]

    scale = json.loads((source/'scales.json').read_text())['mich']['time_scale']
    return data, np.array(prior_train), np.array(prior_val), np.array(prior_full), reveal, float(scale), None


def prepare_ds03(out):
    raw = ROOT/'data/N-CMAPSS_DS03-012.h5'
    if raw.stat().st_size != EXPECTED_SIZE or sha256_file(raw) != EXPECTED_SHA256:
        raise ValueError('DS03 raw data differs from frozen source')
    rows = causal_features(load_development(raw), 'direct')
    data = dict(train=subset(rows, np.isin(rows['groups'], np.asarray(range(1, 7), str))),
                validation=subset(rows, np.isin(rows['groups'], np.asarray(range(7, 10), str))))
    path = ROOT/'results/ncmapss_ds03_ppx_v1'
    selection = json.loads((path/'selection.json').read_text())
    assert selection['selected_route'] == 'direct_fallback'
    # Trusted local artifact generated by the existing frozen experiment.
    fits = torch.load(path/'selection.models.pt', map_location='cpu', weights_only=False)['fits']['direct_fallback'][:3]
    baseline_val = np.array([_predict_direct(f, data['validation']['x']) for f in fits])

    def reveal():
        test = causal_features(load_revealed_test(raw), 'direct')
        baseline = np.array([_predict_direct(f, test['x']) for f in fits])
        eng = dict(np.load(ROOT/'results/ncmapss_ds03_equal_budget_v1/engression/predictions.npz'))
        for k in ('y', 'groups'):
            np.testing.assert_array_equal(test[k], eng[k])
        return test, baseline, eng['prediction'][:3]
    return data, None, baseline_val, None, reveal, 1., float(fits[0]['cap'])


def run(case, out):
    out.mkdir(parents=True, exist_ok=False)
    write(out/'protocol.json', dict(case=case, seeds=SEEDS, arms=ARMS, primary='learned_relation',
        status='retrospective development screen; historically revealed test, not prospective',
        validation_gate='ensemble unit MSE improves >2%; >=60% unit wins; max unit RMSE ratio <=1.05',
        test_gate='lower pooled RMSE vs both baselines; no worst-unit or positive-R2 coverage loss',
        selection='TRAIN episodes fit geometry; disjoint validation selects epoch and global acceptance; no test selection',
        limits=['local linear metric learning is not established novelty',
                'two settings cannot establish full benchmark coverage',
                'no interval coverage, causal identification or counterfactual guarantee',
                'fixed arms are ablations, not a test-selected portfolio',
                'baseline search budgets differ; not an equal-compute claim',
                'static arms repeat identically, not independent stochastic evidence'],
        refit='MICH full PP-X canonical refit + expanded memory, frozen geometry; DS03 train memory only',
        hashes={str(p.relative_to(ROOT)): sha256_file(p) for p in (
            Path(__file__), ROOT/'src/pp_extrapolation/relation_local_transport.py',
            ROOT/'src/pp_extrapolation/boundary_quotient.py', ROOT/'src/pp_extrapolation/ds03_prospective.py')}))
    data, pt, pv, pf, reveal, unit_scale, cap = (prepare_mich if case == 'mich' else prepare_ds03)(out)
    prior_on = case == 'mich'
    all_fits, choices, validation = {}, {}, {}
    for part in ('train', 'validation'):
        np.savez_compressed(out/f'{part}_rows.npz', **data[part])
    for arm, config in ARMS.items():
        fits, predictions, logs = [], [], []
        for i, seed in enumerate(SEEDS):
            start = time.monotonic()
            fit = fit_relation_transport(data['train'], data['validation'], progress_index=0,
                increasing=not prior_on, seed=seed, prior_approved=prior_on,
                prior_train=pt[i] if prior_on else None, prior_validation=pv[i] if prior_on else None,
                boundary_index=0 if prior_on else None, cap=cap, **config)
            predictions.append(predict_relation_transport(fit, data['validation']['x'],
                prior_prediction=pv[i] if prior_on else None))
            logs.append(fit.selection)
            if prior_on:
                fit = memory_refit(fit, data['full'], pf[i])
            fits.append(fit)
            print(case, arm, seed, 'epoch', logs[-1]['selected_epoch'],
                  'seconds', round(time.monotonic()-start, 1), flush=True)
        validation[arm] = np.array(predictions)
        choices[arm] = dict(gate=validation_acceptance(data['validation']['y'], data['validation']['groups'],
            pv.mean(0), validation[arm].mean(0)), runs=logs)
        all_fits[arm] = fits
        torch.save(fits, out/f'{arm}_models.pt')
        write(out/'selection.partial.json', choices)
    # Immutable selection is persisted before this run loads any test rows.
    write(out/'selection.json', choices)
    np.savez_compressed(out/'validation_predictions.npz', baseline=pv, **validation)
    test, baseline, engression = reveal()
    assert not set(test['groups']) & (set(data['train']['groups']) | set(data['validation']['groups']))
    train_coordinate = data['train']['x'][:, 0]
    t = test['x'][:, 0]
    write(out/'support_audit.json', dict(coordinate=0, strict_extrapolation_fraction=float(
        np.mean((t < train_coordinate.min()) | (t > train_coordinate.max()))),
        train_range=[float(train_coordinate.min()), float(train_coordinate.max())],
        test_range=[float(t.min()), float(t.max())], unit_disjoint=True,
        note='DS03 new-unit generalization is not uniformly coordinate extrapolation'))
    np.savez_compressed(out/'test_rows.npz', **test)
    predictions = dict(ppx=baseline, engression=engression)
    for arm, fits in all_fits.items():
        mat = np.array([predict_relation_transport(f, test['x'],
            prior_prediction=baseline[i] if prior_on else None) for i, f in enumerate(fits)])
        predictions[arm] = mat
        predictions[arm+'_guarded'] = mat if choices[arm]['gate']['accepted'] else baseline.copy()
    np.savez_compressed(out/'test_predictions.npz', **predictions)
    scores = {name: metrics(test['y']*unit_scale, test['groups'], mat*unit_scale)
              for name, mat in predictions.items()}
    candidate = scores['learned_relation_guarded']
    gates = {}
    for name in ('ppx', 'engression'):
        b = scores[name]
        gates[name] = dict(lower_rmse=candidate['ensemble']['rmse'] < b['ensemble']['rmse'],
            no_worst_unit_loss=candidate['ensemble']['worst_unit_rmse'] <= b['ensemble']['worst_unit_rmse'],
            no_unit_coverage_loss=candidate['ensemble']['positive_units'] >= b['ensemble']['positive_units'],
            no_seed_coverage_loss=candidate['positive_seeds'] >= b['positive_seeds'])
    passed = all(all(v.values()) for v in gates.values())
    result = dict(complete=True, scores=scores, gates=gates, screen_passed=passed,
                  successor_promoted=False, validation=choices, physical_unit_scale=unit_scale)
    write(out/'results.json', result)
    print(case, 'complete', 'screen_passed', passed, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=('mich', 'ds03'), required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args.case, ROOT/f'results/relation_local_transport_{args.case}_v1')
