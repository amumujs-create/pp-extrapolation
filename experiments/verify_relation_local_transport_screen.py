"""Replay saved checkpoints and recompute selection/metrics for both routes."""
import json
from pathlib import Path

import numpy as np
import torch

from relation_local_transport_screen import (
    ROOT, ARMS, metrics, write, predict_relation_transport,
    validation_acceptance, group_weights, sha256_file,
    predict_boundary_quotient, _predict_direct,
)


def close_tree(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            close_tree(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            close_tree(x, y)
    elif a is None or isinstance(a, (bool, str)):
        assert a == b
    else:
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)


def verify(case):
    out = ROOT/f'results/relation_local_transport_{case}_v1'
    protocol = json.loads((out/'protocol.json').read_text())
    for path, expected in protocol['hashes'].items():
        assert sha256_file(ROOT/path) == expected, path
    result = json.loads((out/'results.json').read_text())
    selection = json.loads((out/'selection.json').read_text())
    train, val, test = [dict(np.load(out/f'{p}_rows.npz')) for p in ('train', 'validation', 'test')]
    vpred = dict(np.load(out/'validation_predictions.npz'))
    tpred = dict(np.load(out/'test_predictions.npz'))
    if case == 'mich':
        bundle = torch.load(out/'baseline_models.pt', weights_only=False)
        joint = dict(np.load(ROOT/'results/prefix_speed_integral_screen_v1/test_rows.npz'))
        baseline = np.array([predict_boundary_quotient(f, joint)[joint['dataset'] == 2] for f in bundle['refitted']])
    else:
        bundle = torch.load(ROOT/'results/ncmapss_ds03_ppx_v1/selection.models.pt', weights_only=False)
        baseline = np.array([_predict_direct(f, test['x']) for f in bundle['fits']['direct_fallback'][:3]])
    np.testing.assert_array_equal(baseline, tpred['ppx'])
    replayed = 0
    for arm in ARMS:
        fits = torch.load(out/f'{arm}_models.pt', weights_only=False)
        gate = validation_acceptance(val['y'], val['groups'], vpred['baseline'].mean(0), vpred[arm].mean(0))
        close_tree(gate, selection[arm]['gate'])
        expected_guarded = tpred[arm] if gate['accepted'] else tpred['ppx']
        np.testing.assert_array_equal(expected_guarded, tpred[arm+'_guarded'])
        for i, fit in enumerate(fits):
            weights = group_weights(train['groups'])
            np.testing.assert_allclose(fit.center, (weights[:, None]*train['x']).sum(0))
            p = predict_relation_transport(fit, test['x'],
                prior_prediction=baseline[i] if case == 'mich' else 'ignored rejected prior')
            np.testing.assert_array_equal(p, tpred[arm][i])
            if case == 'ds03':
                # No validation memory has entered the fallback.
                np.testing.assert_allclose(fit.bank_x, ((train['x']-fit.center)/fit.scale)[fit.bank_indices])
                assert not fit.prior_approved and fit.boundary_index is None
            else:
                zero = test['x'][:2].copy()
                zero[:, 0] = 0
                np.testing.assert_array_equal(predict_relation_transport(fit, zero, prior_prediction=np.zeros(2)), 0.)
            replayed += 1
    for name, mat in tpred.items():
        scale = result['physical_unit_scale']
        close_tree(metrics(test['y']*scale, test['groups'], mat*scale), result['scores'][name])
    assert result['screen_passed'] == all(all(v.values()) for v in result['gates'].values())
    assert not result['successor_promoted']
    report = dict(verified=True, checkpoint_replays=replayed, metric_matrices=len(tpred),
                  source_hashes_checked=len(protocol['hashes']), baseline_replayed=True,
                  validation_gates_recomputed=True, guarded_routes_replayed=True)
    write(out/'verification.json', report)
    print(case, report, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    for case in ('mich', 'ds03'):
        verify(case)
