"""Frozen residual protocol transferred to the nine historical main settings."""
from pathlib import Path
import sys
import json
import traceback
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'experiments'), str(ROOT.parent / 'ca-css-ncmapss')]
from pp_extrapolation.innovation_slope_transport import fit_innovation_slope_transport as fit
from pp_extrapolation.innovation_slope_transport import predict_innovation_slope_transport as predict
from pp_extrapolation.anchored_residual_transport import fit_anchored_residual, predict_anchored_residual
from cist_mean_alignment_source_audit import metric

OUT = ROOT / 'results/main9_anchored_residual_audit_v1'
SEEDS = (42, 43, 44)


def datasets():
    from run_affine_tail_external_three import prepare_hust, prepare_virkler
    from run_affine_tail_external_nasa_health_v2 import prepare_folds
    from pae_boundary_realdata import prepare_dataset
    from distance_uncertainty_pp import prepare_battery
    from extrapolation_competitors_all import matr
    import matr_batch2_confirmatory as b2
    yield 'HUST', [prepare_hust()[:3]], 'capacity_ah'
    yield 'Virkler', [prepare_virkler()[:3]], 'crack_length_mm'
    folds, _ = prepare_folds()
    yield 'NASA', [(f['train'], f['validation'], f['test']) for f in folds], 'health_phi'
    for name in ('sunwoda', 'rwth', 'mich'):
        raw, _ = prepare_dataset(name)
        yield name.upper(), [prepare_battery(raw)], 'endpoint_health'
    yield 'MATR2019', [matr()], 'endpoint_capacity'
    cells = b2.load_cells()
    cut = np.quantile(b2.endpoints(cells, range(30)), .25)
    train = b2.make_rows(cells, range(30), cut, train=True)
    boundary = train['coordinate'].min()
    yield 'MATR-b2', [(train, b2.make_rows(cells, range(30, 39), boundary),
                        b2.make_rows(cells, range(39, 48), boundary))], 'endpoint_capacity'
    from apps.ncmapss_data_utils import FEATURE_COLS
    from ncmapss_tra_quantile_split import make_tra_hard_split
    from ncmapss_pp_benchmark import rows
    split = make_tra_hard_split(ROOT / 'data/N-CMAPSS_DS02-006.h5',
                                max_windows_per_unit=1500, random_seed=42)
    yield 'N-CMAPSS', [(rows(split.train, list(FEATURE_COLS)),
                        rows(split.val, list(FEATURE_COLS)),
                        rows(split.test, list(FEATURE_COLS)))], 'TRA_operating_coordinate_not_age'


def score(y, groups, predictions):
    runs = [dict(seed=s, **metric(y, p, groups)) for s, p in zip(SEEDS, predictions)]
    return dict(ensemble=metric(y, predictions.mean(0), groups), runs=runs,
                seed_sd=float(np.std([r['r2'] for r in runs], ddof=1)),
                minimum_seed_r2=float(min(r['r2'] for r in runs)))


def report(records, protocol, failures):
    names = ('direct', 'integral_residual', 'ppx_reference')
    summary = {}
    for name in names:
        values = [r['models'][name] for r in records if name in r['models']]
        if values:
            summary[name] = dict(settings=len(values),
                mean_setting_r2=float(np.mean([v['ensemble']['r2'] for v in values])),
                minimum_setting_r2=float(min(v['ensemble']['r2'] for v in values)),
                mean_seed_sd=float(np.mean([v['seed_sd'] for v in values])),
                minimum_seed_r2=float(min(v['minimum_seed_r2'] for v in values)),
                positive_settings=sum(v['ensemble']['r2'] > 0 for v in values))
    payload = dict(protocol=protocol, records=records, summary=summary, failures=failures,
                   complete=len(records) == 9 and not failures)
    (OUT / 'results.json').write_text(json.dumps(payload, indent=2))
    lines = ['# Main-nine frozen integral residual transfer', '',
        'Retrospective audit, three paired seeds (42-44). No test-driven model selection.',
        'PP-X is an archived reference with different features and training budgets, not a matched architecture control.',
        'The nine-setting arithmetic mean is NOT pooled R2. No cross-unit/target-scale pooled score is reported.', '',
        '| Setting | Direct R2 | Integral R2 | Archived PP-X R2 | Integral seed SD |',
        '|---|---:|---:|---:|---:|']
    for r in records:
        m = r['models']
        lines.append('| ' + r['setting'] + ' | ' + ' | '.join(
            f"{m[n]['ensemble']['r2']:.5f}" if n in m else 'unmatched' for n in names)
            + f" | {m['integral_residual']['seed_sd']:.5f} |")
    lines += ['', '```json', json.dumps(summary, indent=2), '```', '',
              'Integral coordinate is column zero of the existing causal adapter; TRA is an operating coordinate, not physical degradation time.',
              'No claim of global monotonicity, established novelty, or untouched external confirmation.',
              'A rejected residual remains zero; there is no fallback to archived PP-X predictions.', '',
              'Failures: ' + json.dumps(failures)]
    (OUT / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    return summary


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=False)
    protocol = dict(seeds=SEEDS, settings=9, progress_index=0,
        status='retrospective original main-setting transfer; not new external confirmation',
        base=dict(route='direct_only', width=32, max_epochs=60, patience=20,
                  learning_rate=.001, weight_decay=.1, beta=1., mean_weight=.5),
        residual=dict(route='integral', width=16, steps=4, max_epochs=60, patience=20,
                      penalty=.1, min_validation_gain=.01, max_worst_group_ratio=1.05),
        comparison='First three archived PP-X seeds; exact target/order matching required.',
        limitations=['Native PP-X feature/training budgets differ.',
                     'Original splits are reused, not repeated source holdout splits.',
                     'No untouched external evaluation; no hyperparameter search.'])
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    print('Loading archived reference rows', flush=True)
    from final_modular_pp_evidence import build_datasets
    references = build_datasets()
    print('Reference settings', list(references), flush=True)
    records, failures = [], []
    for setting, folds, coordinate in datasets():
        print('START', setting, [(len(t['y']), len(v['y']), len(e['y'])) for t, v, e in folds], flush=True)
        try:
            y = np.concatenate([f[2]['y'] for f in folds])
            groups = np.concatenate([f[2]['groups'] for f in folds]).astype(str)
            reference = references[setting]
            ry, rg, rp = reference[:3]
            aligned = (np.shape(ry) == y.shape and np.allclose(ry, y, rtol=1e-4, atol=1e-4)
                       and np.array_equal(np.asarray(rg).astype(str), groups))
            if not aligned:
                raise ValueError(f'Archived target/group rows do not align: adapter={y.shape}, reference={np.shape(ry)}')
            predictions = {'direct': [], 'integral_residual': []}
            diagnostics = []
            for seed in SEEDS:
                direct_parts, residual_parts = [], []
                for fold_index, (train, val, test) in enumerate(folds):
                    gs = [set(np.asarray(p['groups']).astype(str)) for p in (train, val, test)]
                    assert all(not gs[i].intersection(gs[j]) for i in range(3) for j in range(i)), 'Unit overlap'
                    base = fit(train, val, progress_index=0, seed=seed, route='direct_only', width=32,
                        max_epochs=60, patience=20, learning_rate=.001, weight_decay=.1,
                        steps=4, beta=1., mean_weight=.5, mean_objective='zero_noise',
                        validation_samples=128, validation_clip_to_train_range=True)
                    bp = np.clip(predict(base, test['x'], samples=128), 0, max(float(train['y'].max()), 1))
                    residual = fit_anchored_residual(base, train, val, seed=seed, route='integral')
                    p = predict_anchored_residual(residual, test['x'])
                    direct_parts.append(bp)
                    residual_parts.append(p)
                    diagnostics.append(dict(seed=seed, fold=fold_index, accepted=residual.accepted,
                        selected_epoch=residual.selected_epoch, diagnostics=residual.diagnostics))
                predictions['direct'].append(np.concatenate(direct_parts))
                predictions['integral_residual'].append(np.concatenate(residual_parts))
                print('FIT', setting, seed, flush=True)
            arrays = {k: np.asarray(v) for k, v in predictions.items()}
            arrays['ppx_reference'] = np.asarray(rp)[:len(SEEDS)]
            rec = dict(setting=setting, coordinate=coordinate, rows=len(y), reference_rows_aligned=aligned,
                       diagnostics=diagnostics, models={k: score(y, groups, p) for k, p in arrays.items()})
            np.savez_compressed(OUT / f'{setting}.npz', truth=y, groups=groups, **arrays)
            records.append(rec)
            print('SCORED', setting, {k: v['ensemble']['r2'] for k, v in rec['models'].items()}, flush=True)
        except Exception as exc:
            traceback.print_exc()
            failures.append(dict(setting=setting, error=str(exc)))
        report(records, protocol, failures)
    print('SUMMARY', json.dumps(report(records, protocol, failures)), flush=True)


if __name__ == '__main__':
    main()
