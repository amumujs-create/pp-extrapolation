"""Fixed follow-up to the rejected local-transport prototype; no test selection."""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from pp_extrapolation.component_transfer import make_bank, component_features, fit_components, predict_components
from pp_extrapolation.ds03_prospective import _fit_direct, _predict_direct, sha256_file
from pp_extrapolation.boundary_quotient import fit_boundary_quotient_pp, predict_boundary_quotient
from pp_extrapolation.presets import battery_dual_scale_pp_config
from pp_extrapolation.relation_local_transport import validation_acceptance
from relation_local_transport_screen import metrics, write, subset

SEEDS = (42, 43, 44)
ARMS = dict(global_gate=dict(conditional=False, robust_weight=0.),
            context_gate=dict(conditional=True, robust_weight=0.),
            robust_components=dict(conditional=True, robust_weight=.25))


def envelope(rows, base, bank, *, tag=''):
    c, z = component_features(bank, rows['x'], base)
    return dict(c=c, z=z, base=base, y=rows['y'], groups=np.array([tag+str(g) for g in rows['groups']]))


def source_episode_indices(rows, case, fold, fraction):
    units = np.unique(rows['groups'])
    held = units[fold::3]
    direction = -1 if case == 'mich' else 1
    coordinate = direction*rows['x'][:, 0]
    cutoff = float(np.quantile(coordinate, fraction))
    donor = (~np.isin(rows['groups'], held)) & (coordinate <= cutoff)
    query = np.isin(rows['groups'], held) & (coordinate > cutoff)
    return donor, query, cutoff, held


def crossfit_episodes(case, train, joint, seed, out):
    episodes, records = [], []
    for fold in range(3):
        for fraction in (.6, .8):
            start = time.monotonic()
            donor, query, cutoff, held = source_episode_indices(train, case, fold, fraction)
            anchors, queries = subset(train, donor), subset(train, query)
            assert not set(anchors['groups']) & set(queries['groups'])
            if case == 'mich':
                allowed = (joint['x'][:, 0] >= -cutoff) & ~((joint['dataset'] == 2) & np.isin(joint['units'].astype(str), held))
                source = subset(joint, allowed)
                # Fixed source-only fit: query units/outcomes and external VAL absent.
                model = fit_boundary_quotient_pp(source, source, seed=seed, max_epochs=200,
                    patience=10000, restore_best=False, **battery_dual_scale_pp_config())
                pa = predict_boundary_quotient(model, anchors)
                pq = predict_boundary_quotient(model, queries)
                ids = [f'{d}:{u}' for d, u in zip(source['dataset'], source['units'])]
            else:
                model = _fit_direct(anchors, anchors, seed)
                pa, pq = _predict_direct(model, anchors['x']), _predict_direct(model, queries['x'])
                ids = anchors['groups'].tolist()
            bank = make_bank(anchors, pa, boundary_index=0 if case == 'mich' else None)
            tag = f'{fold}:{fraction}:'
            episode = envelope(queries, pq, bank, tag=tag)
            episodes.append(episode)
            identifier = f'seed{seed}_fold{fold}_cut{int(100*fraction)}'
            np.savez_compressed(out/f'{identifier}.npz', **episode)
            torch.save(dict(base_model=model, bank=bank, anchors=anchors, queries=queries), out/f'{identifier}.pt')
            records.append(dict(id=identifier, held_units=held.tolist(), base_train_units=sorted(set(ids)),
                donor_indices=np.flatnonzero(donor).tolist(), query_indices=np.flatnonzero(query).tolist(),
                cutoff=cutoff, fraction=fraction, source_unit_disjoint=True, strict_ordered_support=True))
            print(case, identifier, 'rows', len(queries['y']), 'seconds', round(time.monotonic()-start, 1), flush=True)
    return episodes, records


def run(case):
    out = ROOT/f'results/component_transfer_{case}_v1'
    out.mkdir(parents=True, exist_ok=False)
    write(out/'protocol.json', dict(seeds=SEEDS, arms=ARMS, primary='robust_components',
        hypothesis='learn residual component transfer on held-out units/support; preserve the full base predictor',
        source_episode='3 deterministic unit folds x 2 ordered cutoffs (.6,.8); base fitted without query units or later support',
        base_source_fit='MICH fixed 200 epochs; DS03 native 300-epoch fit with source in-sample checkpoint, never query selection',
        outer_gate='unchanged: >2% ensemble unit MSE gain, >=60% unit wins, no unit RMSE increase >5%',
        architecture='fixed local ridge residual level/progress-slope/other-slope; context-conditioned 0..1 component shrinkage',
        novelty='unverified; cross-fitting, residual correction and local linear transfer already have prior art',
        limitations=['historically revealed development endpoints, not prospective confirmation',
                    'source episodes have smaller training support than deployment',
                    'DS03 test is new-unit generalization, not cycle-support extrapolation',
                    'two-case prototype, not full coverage or equal compute',
                    'MICH selected correction geometry transferred to canonical train+validation memory refit'],
        hashes={str(p.relative_to(ROOT)): sha256_file(p) for p in (Path(__file__),
            ROOT/'src/pp_extrapolation/component_transfer.py', ROOT/'src/pp_extrapolation/relation_local_transport.py',
            ROOT/'src/pp_extrapolation/boundary_quotient.py', ROOT/'src/pp_extrapolation/ds03_prospective.py',
            ROOT/'experiments/relation_local_transport_screen.py')}))
    old = ROOT/f'results/relation_local_transport_{case}_v1'
    train, val = [dict(np.load(old/f'{p}_rows.npz')) for p in ('train', 'validation')]
    if case == 'mich':
        src = ROOT/'results/prefix_speed_integral_screen_v1'
        joint = dict(np.load(src/'train_rows.npz'))
        full_joint = dict(np.load(src/'full_rows.npz'))
        full = subset(full_joint, full_joint['dataset'] == 2)
        full['groups'] = full['units'].astype(str)
        bundle = torch.load(old/'baseline_models.pt', weights_only=False)
        selected, final = bundle['selected'], bundle['refitted']
        pred = predict_boundary_quotient
        physical_scale = json.loads((src/'scales.json').read_text())['mich']['time_scale']
        cap = None
    else:
        joint = None
        bundle = torch.load(ROOT/'results/ncmapss_ds03_ppx_v1/selection.models.pt', weights_only=False)
        selected = final = bundle['fits']['direct_fallback'][:3]
        pred = lambda f, r: _predict_direct(f, r['x'])
        physical_scale, cap = 1., float(selected[0]['cap'])
    all_models, all_banks, val_matrices, logs, episode_records = {}, [], {a: [] for a in ARMS}, {a: [] for a in ARMS}, []
    unshrunk_val, val_base = [], []
    for i, seed in enumerate(SEEDS):
        pa, pv = pred(selected[i], train), pred(selected[i], val)
        bank = make_bank(train, pa, boundary_index=0 if case == 'mich' else None)
        validation = envelope(val, pv, bank)
        np.savez_compressed(out/f'validation_seed{seed}.npz', **validation)
        val_base.append(pv)
        unshrunk_val.append(np.clip(pv+validation['c'].sum(1), 0, cap) if cap is not None else np.maximum(pv+validation['c'].sum(1), 0))
        episodes, records = crossfit_episodes(case, train, joint, seed, out)
        episode_records.extend(records)
        for arm, config in ARMS.items():
            fit = fit_components(episodes, validation, seed=seed, cap=cap, **config)
            val_matrices[arm].append(predict_components(fit, validation['c'], validation['z'], pv))
            logs[arm].append(fit.selection)
            all_models.setdefault(arm, []).append(fit)
            torch.save(fit, out/f'{arm}_seed{seed}.pt')
            print(case, arm, seed, 'epoch', fit.selection['selected_epoch'], flush=True)
        final_bank = make_bank(full, pred(final[i], full), boundary_index=0) if case == 'mich' else bank
        all_banks.append(final_bank)
    val_base = np.array(val_base)
    val_matrices = {k: np.array(v) for k, v in val_matrices.items()}
    val_matrices['unshrunk_residual'] = np.array(unshrunk_val)
    decisions = {a: validation_acceptance(val['y'], val['groups'], val_base.mean(0), mat.mean(0)) for a, mat in val_matrices.items()}
    write(out/'selection.json', dict(decisions=decisions, logs=logs, episodes=episode_records))
    np.savez_compressed(out/'validation_predictions.npz', baseline=val_base, **val_matrices)
    torch.save(all_banks, out/'banks.pt')
    score_saved(case, out)


def score_saved(case, out):
    """Resume reveal without retraining or changing the saved validation choice."""
    if (out/'results.json').exists():
        raise FileExistsError('completed results are immutable')
    old = ROOT/f'results/relation_local_transport_{case}_v1'
    decisions = json.loads((out/'selection.json').read_text())['decisions']
    all_banks = torch.load(out/'banks.pt', weights_only=False)
    all_models = {a: [torch.load(out/f'{a}_seed{s}.pt', weights_only=False) for s in SEEDS] for a in ARMS}
    if case == 'mich':
        final = torch.load(old/'baseline_models.pt', weights_only=False)['refitted']
        canonical_rows = dict(np.load(ROOT/'results/prefix_speed_integral_screen_v1/test_rows.npz'))
        pred = lambda f, r: predict_boundary_quotient(f, canonical_rows)[canonical_rows['dataset'] == 2]
        physical_scale = json.loads((ROOT/'results/prefix_speed_integral_screen_v1/scales.json').read_text())['mich']['time_scale']
        cap = None
    else:
        final = torch.load(ROOT/'results/ncmapss_ds03_ppx_v1/selection.models.pt', weights_only=False)['fits']['direct_fallback'][:3]
        pred = lambda f, r: _predict_direct(f, r['x'])
        physical_scale, cap = 1., float(final[0]['cap'])
    # Only now reopen historical development test observations/outcomes.
    test = dict(np.load(old/'test_rows.npz'))
    previous = dict(np.load(old/'test_predictions.npz'))
    predictions = {k: previous[k] for k in ('ppx', 'engression')}
    predictions.update({a: [] for a in (*ARMS, 'unshrunk_residual')})
    gates_summary = {a: [] for a in ARMS}
    for i, seed in enumerate(SEEDS):
        base = pred(final[i], test)
        np.testing.assert_array_equal(base, previous['ppx'][i])
        c, z = component_features(all_banks[i], test['x'], base)
        np.savez_compressed(out/f'test_components_seed{seed}.npz', c=c, z=z, base=base)
        predictions['unshrunk_residual'].append(np.clip(base+c.sum(1), 0, cap) if cap is not None else np.maximum(base+c.sum(1), 0))
        for arm in ARMS:
            p, gate = predict_components(all_models[arm][i], c, z, base, return_gates=True)
            predictions[arm].append(p)
            gates_summary[arm].append(dict(seed=seed, mean=gate.mean(0).tolist(), min=gate.min(0).tolist(), max=gate.max(0).tolist()))
    predictions = {k: np.array(v) for k, v in predictions.items()}
    for arm in (*ARMS, 'unshrunk_residual'):
        predictions[arm+'_guarded'] = predictions[arm] if decisions[arm]['accepted'] else predictions['ppx'].copy()
    np.savez_compressed(out/'test_predictions.npz', **predictions)
    scores = {name: metrics(test['y']*physical_scale, test['groups'], mat*physical_scale) for name, mat in predictions.items()}
    gates = {}
    c = scores['robust_components_guarded']
    for name in ('ppx', 'engression'):
        b = scores[name]
        gates[name] = dict(lower_rmse=c['ensemble']['rmse'] < b['ensemble']['rmse'],
            no_worst_unit_loss=c['ensemble']['worst_unit_rmse'] <= b['ensemble']['worst_unit_rmse'],
            no_unit_coverage_loss=c['ensemble']['positive_units'] >= b['ensemble']['positive_units'],
            no_seed_coverage_loss=c['positive_seeds'] >= b['positive_seeds'])
    write(out/'results.json', dict(complete=True, scores=scores, gates=gates,
        screen_passed=all(all(v.values()) for v in gates.values()), successor_promoted=False,
        physical_unit_scale=physical_scale, gate_summaries=gates_summary, decisions=decisions))
    print(case, 'COMPLETE', decisions['robust_components'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--case', choices=('mich', 'ds03'), required=True)
    parser.add_argument('--reveal-only', action='store_true')
    torch.set_num_threads(2)
    args = parser.parse_args()
    if args.reveal_only:
        score_saved(args.case, ROOT/f'results/component_transfer_{args.case}_v1')
    else:
        run(args.case)
