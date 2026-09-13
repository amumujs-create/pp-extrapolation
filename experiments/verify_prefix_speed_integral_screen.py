"""Independent score, checkpoint, split and source-pair audit from saved arrays."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from pp_extrapolation.prefix_speed_integral import PrefixSpeedIntegral, predict_prefix_speed

NAMES = ('sunwoda', 'rwth', 'mich')


def main():
    torch.set_num_threads(2)
    out = ROOT/'results/prefix_speed_integral_screen_v1'
    result = json.loads((out/'results.json').read_text())
    scales = json.loads((out/'scales.json').read_text())
    assert result['complete'] and len(result['records']) == 10
    for path, expected in result['protocol']['hashes'].items():
        p = Path(path) if Path(path).is_absolute() else ROOT/path
        assert hashlib.sha256(p.read_bytes()).hexdigest() == expected, p
    data = {p: dict(np.load(out/f'{p}_rows.npz')) for p in ('train', 'validation', 'full', 'test')}
    def groups(part):
        return set(zip(part['dataset'], part['units']))
    assert not groups(data['train']) & groups(data['validation'])
    assert not groups(data['full']) & groups(data['test'])
    assert groups(data['full']) == groups(data['train']) | groups(data['validation'])
    for p in ('train', 'full'):
        pairs = np.load(out/f'{p}_pairs.npz')
        rows = data[p]
        a, b = pairs['anchor'], pairs['future']
        assert np.all(rows['dataset'][a] == rows['dataset'][b])
        assert np.all(rows['units'][a] == rows['units'][b])
        assert np.all(rows['cycles'][b]-rows['cycles'][a] >= 8)
        assert np.all(rows['time'][b] > rows['time'][a])
        assert np.all(rows['margin'][b] < rows['margin'][a])
        np.testing.assert_array_equal(pairs['target_margin'], rows['margin'][b])
        np.testing.assert_array_equal(pairs['elapsed'], rows['time'][b]-rows['time'][a])
    rows = data['test']
    checks, checkpoints = 0, 0
    for record in result['records']:
        arm, refit = record['model'], int(record['refit'])
        path = Path(record['source']) if 'source' in record else out/f'{arm}_refit{refit}.npz'
        arr = np.load(path)
        for k in ('y', 'units', 'dataset'):
            np.testing.assert_array_equal(arr[k], rows[k])
        matrix = arr['prediction'].astype(float)
        assert matrix.shape == (3, len(rows['y'])) and np.isfinite(matrix).all()
        if 'source' not in record:
            for i, seed in enumerate((42, 43, 44)):
                state = torch.load(out/f'{arm}_seed{seed}_refit{refit}.pt', weights_only=True)
                net = PrefixSpeedIntegral(state['state_dict']['center'].numpy(),
                    state['state_dict']['scale'].numpy(), width=state['width'],
                    variable_speed=state['variable_speed'], quadrature=state['quadrature'])
                net.load_state_dict(state['state_dict'])
                np.testing.assert_allclose(predict_prefix_speed(net, rows), matrix[i], rtol=1e-6, atol=1e-6)
                checkpoints += 1
        for di, name in enumerate(NAMES):
            mask = rows['dataset'] == di
            y = rows['y'][mask].astype(float)*scales[name]['time_scale']
            p = matrix[:, mask]*scales[name]['time_scale']
            ids = rows['units'][mask]
            saved = record['metrics'][name]
            err = (p.mean(0)-y)**2
            denom = np.sum((y-y.mean())**2)
            unit_err = [err[ids == u] for u in np.unique(ids)]
            values = [1-err.sum()/denom, np.sqrt(err.mean()),
                      np.mean([np.sqrt(e.mean()) for e in unit_err]),
                      max(np.sqrt(e.mean()) for e in unit_err)]
            np.testing.assert_allclose(values, [saved['ensemble'][k] for k in
                ('r2', 'rmse', 'unit_rmse', 'worst_unit_rmse')], rtol=1e-9, atol=1e-9)
            seed_r2 = 1-np.sum((p-y)**2, axis=1)/denom
            np.testing.assert_allclose(seed_r2, [m['r2'] for m in saved['seeds']], atol=1e-10)
            assert int(np.sum(seed_r2 > 0)) == saved['positive_seed_r2']
            positive = sum(np.sum((p.mean(0)[ids == u]-y[ids == u])**2)
                           < np.sum((y[ids == u]-y[ids == u].mean())**2) for u in np.unique(ids))
            assert positive == saved['positive_units']
            checks += 1
    audit = dict(complete=True, dataset_metric_checks=checks, checkpoint_replays=checkpoints,
                 source_only_ordered_pairs=True, row_identity=True, source_hashes_unchanged=True)
    (out/'verification.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
