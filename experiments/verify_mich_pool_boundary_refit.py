"""Independent saved-prediction and source-support audit; does not train models."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'ca-css-ncmapss'))
from pae_boundary_realdata import DATASETS, prepare_dataset


def main():
    out = ROOT / 'results/mich_pool_boundary_refit_factorial_v1'
    result = json.loads((out / 'results.json').read_text())
    scales = json.loads((out / 'scales.json').read_text())
    assert result['complete'] and len(result['records']) == 24
    support = {}
    for name in DATASETS:
        splits, _ = prepare_dataset(name)
        lower = min(float(splits[p]['x'][:, -1, 0].min()) for p in ('train', 'val'))
        upper = float(splits['source']['x'][:, -1, 0].max())
        assert upper < lower, (name, upper, lower)
        support[name] = dict(train_validation_min_endpoint_health=lower,
                             test_max_endpoint_health=upper, strict_lower_support=True)
    ref = np.load(out / 'mich_all_bq_dual_refit1.npz')
    checked = 0
    for row in result['records']:
        arr = np.load(out / f"{row['pool']}_{row['model']}_refit{int(row['refit'])}.npz")
        mask = arr['dataset'] == DATASETS.index('mich')
        np.testing.assert_array_equal(arr['y'][mask], ref['y'])
        np.testing.assert_array_equal(arr['units'][mask], ref['units'])
        assert arr['prediction'].shape == (3, len(arr['y']))
        for di in np.unique(arr['dataset']):
            m = arr['dataset'] == di
            y = arr['y'][m].astype(float)
            pred = arr['prediction'][:, m].astype(float).mean(0)
            r2 = 1 - np.sum((pred-y)**2) / np.sum((y-y.mean())**2)
            name = DATASETS[int(di)]
            saved = row['metrics'][name]
            np.testing.assert_allclose(r2, saved['ensemble']['r2'], atol=1e-10)
            scale = scales[name]['time_scale']
            err = ((pred-y) * scale)**2
            unit_errors = [np.sqrt(err[arr['units'][m] == u].mean()) for u in np.unique(arr['units'][m])]
            np.testing.assert_allclose(
                [np.sqrt(err.mean()), np.mean(unit_errors), max(unit_errors)],
                [saved['ensemble'][k] for k in ('rmse', 'unit_rmse', 'worst_unit_rmse')], atol=1e-9)
            seed_r2 = 1 - np.sum((arr['prediction'][:, m].astype(float)-y)**2, axis=1) / np.sum((y-y.mean())**2)
            np.testing.assert_allclose(seed_r2, [v['r2'] for v in saved['seeds']], atol=1e-10)
            assert int(np.sum(seed_r2 > 0)) == saved['positive_seed_r2']
            checked += 1
    archive = np.load(ROOT / 'results/bq_dual_scale_final_replay_v1/predictions.npz')
    replay = np.load(out / 'joint_all_bq_dual_refit1.npz')
    np.testing.assert_allclose(replay['prediction'], archive['prediction'][:3], rtol=1e-5, atol=1e-5)
    audit = dict(complete=True, conditions=24, dataset_metric_checks=checked,
                 same_mich_test_rows=202, archived_three_seed_replay=True, support=support)
    (out / 'verification.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
