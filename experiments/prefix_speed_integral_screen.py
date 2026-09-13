"""Fixed three-battery RUL screen; no test-selected switching or PP-X mutation."""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'experiments'), str(ROOT.parent/'ca-css-ncmapss')]
from pae_boundary_realdata import DATASETS, prepare_dataset
from pae_shared_battery_nn import BatteryRepresentationScale, concatenate_rows
from boundary_quotient_pp_batteries import build_rows, full_part
from mich_pool_boundary_refit_factorial import score
from pp_extrapolation.prefix_speed_integral import fit_prefix_speed, predict_prefix_speed, make_prefix_pairs

SEEDS = (42, 43, 44)
ARMS = dict(constant_rul=dict(variable_speed=False, pair_weight=0.),
            variable_rul=dict(variable_speed=True, pair_weight=0.),
            variable_prefix=dict(variable_speed=True, pair_weight=.1))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pack(raw, scale, di):
    rows = build_rows(raw, scale, di)
    rows.update(rate=(raw['x'][:, -1, 1]*scale.time_scale/scale.margin_scale).astype(np.float32),
                time=(raw['cycles']/scale.time_scale).astype(np.float32),
                cycles=raw['cycles'])
    return rows


def main():
    torch.set_num_threads(2)
    out = ROOT/'results/prefix_speed_integral_screen_v1'
    out.mkdir(exist_ok=False)
    base = ROOT/'results/mich_pool_boundary_refit_factorial_v1'
    paths = [Path(__file__), ROOT/'src/pp_extrapolation/prefix_speed_integral.py',
             ROOT.parent/'ca-css-ncmapss/pae_boundary_realdata.py',
             ROOT.parent/'ca-css-ncmapss/pae_shared_battery_nn.py',
             base/'results.json', base/'protocol.json', base/'resume.json']
    for name in ('bq_dual', 'engression'):
        paths.extend(base/f'joint_all_{name}_refit{i}.npz' for i in (0, 1))
    protocol = dict(status='retrospective implementation screen, not novel successor adoption',
        seeds=SEEDS, arms=ARMS, primary='variable_prefix, train+validation refit',
        epochs=500, patience=70, width=64, quadrature=24, optimizer='AdamW lr .001 wd .01',
        selection='dataset-macro normalized validation RUL MSE; no test model selection',
        pair_loss='.1 * dataset/unit weighted ((pred_elapsed-elapsed)/(elapsed+.1))^2',
        pair_offsets=(1, 4, 16), min_cycle_gap=8,
        source='original early-health partitions; lower-health source test; unit disjoint',
        primary_gate='strict RMSE improvement on all three datasets versus both frozen baselines; no coverage loss',
        limitations=['known margin and positive causal rate required', 'only three battery settings, not original full coverage',
                     'extra source-transition supervision; not equal objective or compute to baselines',
                     'no prospective novelty or statistical superiority claim'],
        hashes={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): digest(p) for p in paths})
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    gathered = {p: [] for p in ('train', 'validation', 'full', 'test')}
    scales, audit = {}, {}
    for di, name in enumerate(DATASETS):
        raw, source_audit = prepare_dataset(name)
        scale = BatteryRepresentationScale.fit(raw['train'], source_audit['boundary'])
        scales[name] = scale
        lo = min(float(raw[p]['x'][:, -1, 0].min()) for p in ('train', 'val'))
        hi = float(raw['source']['x'][:, -1, 0].max())
        assert hi < lo
        groups = [set(raw[p]['units']) for p in ('train', 'val', 'source')]
        assert all(not groups[i] & groups[j] for i in range(3) for j in range(i))
        for p, part in (('train', raw['train']), ('validation', raw['val']),
                        ('full', full_part(raw)), ('test', raw['source'])):
            gathered[p].append(pack(part, scale, di))
        audit[name] = dict(train_validation_min_health=lo, test_max_health=hi,
                           strict_lower_support=True, unit_disjoint=True,
                           zero_test_margin=int(np.sum(gathered['test'][-1]['margin'] == 0)))
    data = {p: concatenate_rows(parts) for p, parts in gathered.items()}
    for p in ('train', 'full'):
        pairs = make_prefix_pairs(data[p])
        np.savez_compressed(out/f'{p}_pairs.npz', **pairs,
                            dataset=data[p]['dataset'][pairs['anchor']], units=data[p]['units'][pairs['anchor']])
        audit[p+'_pairs'] = len(pairs['anchor'])
    for p, rows in data.items():
        np.savez_compressed(out/f'{p}_rows.npz', **rows)
    (out/'data_audit.json').write_text(json.dumps(audit, indent=2))
    (out/'scales.json').write_text(json.dumps({n:s.metadata() for n,s in scales.items()}, indent=2))
    baseline_matrices = {}
    for name in ('bq_dual', 'engression'):
        for refit in (False, True):
            arr = np.load(base/f'joint_all_{name}_refit{int(refit)}.npz')
            for key in ('y', 'dataset', 'units'):
                np.testing.assert_array_equal(data['test'][key], arr[key])
            assert arr['prediction'].shape == (3, len(data['test']['y']))
            baseline_matrices[name, refit] = arr['prediction']
    records = []
    for arm, config in ARMS.items():
        matrices = {False: [], True: []}
        runs = []
        for seed in SEEDS:
            start = time.monotonic()
            fit = fit_prefix_speed(data['train'], data['validation'], seed=seed, **config)
            before = predict_prefix_speed(fit, data['test'])
            epochs = max(1, fit.selection['selected_epoch'])
            final = fit_prefix_speed(data['full'], data['full'], seed=seed, max_epochs=epochs,
                                     restore_best=False, **config)
            after = predict_prefix_speed(final, data['test'])
            for refit, model in ((False, fit.model), (True, final.model)):
                torch.save(dict(state_dict=model.state_dict(), width=64, quadrature=24,
                                variable_speed=config['variable_speed']),
                           out/f'{arm}_seed{seed}_refit{int(refit)}.pt')
            matrices[False].append(before)
            matrices[True].append(after)
            runs.append(dict(seed=seed, selection=fit.selection, refit=final.selection,
                             seconds=time.monotonic()-start))
            print('FIT', arm, seed, 'selected', epochs, 'seconds', round(time.monotonic()-start, 1), flush=True)
        for refit, predictions in matrices.items():
            mat = np.asarray(predictions)
            np.savez_compressed(out/f'{arm}_refit{int(refit)}.npz', prediction=mat,
                **{k:data['test'][k] for k in ('y', 'dataset', 'units', 'cycles')})
            records.append(dict(model=arm, refit=refit, metrics=score(data['test'], mat, scales), runs=runs))
        (out/'partial.json').write_text(json.dumps(dict(records=records, complete=False), indent=2))
    for (name, refit), mat in baseline_matrices.items():
        records.append(dict(model=name, refit=refit, metrics=score(data['test'], mat, scales),
                            source=str(base/f'joint_all_{name}_refit{int(refit)}.npz')))
    chosen = next(r for r in records if r['model'] == 'variable_prefix' and r['refit'])
    gates = {}
    for name in ('bq_dual', 'engression'):
        b = next(r for r in records if r['model'] == name and r['refit'])
        gates[name] = {d: dict(
            lower_rmse=chosen['metrics'][d]['ensemble']['rmse'] < b['metrics'][d]['ensemble']['rmse'],
            no_seed_coverage_loss=chosen['metrics'][d]['positive_seed_r2'] >= b['metrics'][d]['positive_seed_r2'],
            no_unit_coverage_loss=chosen['metrics'][d]['positive_units'] >= b['metrics'][d]['positive_units'],
            no_worst_unit_loss=chosen['metrics'][d]['ensemble']['worst_unit_rmse'] <= b['metrics'][d]['ensemble']['worst_unit_rmse'])
            for d in DATASETS}
    passed = all(all(v.values()) for comparator in gates.values() for v in comparator.values())
    result = dict(complete=True, protocol=protocol, records=records, gates=gates,
                  screen_passed=passed, successor_promoted=False,
                  decision='additional novelty/full-coverage/locked-cohort gates remain' if passed else 'reject this fixed candidate; preserve all failures')
    (out/'results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(screen_passed=passed, gates=gates), indent=2), flush=True)


if __name__ == '__main__':
    main()
