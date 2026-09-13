"""Fixed post-test development ablations; no full PP-X superiority claim."""
from pathlib import Path
import copy
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'experiments'), str(ROOT.parent/'ca-css-ncmapss')]
from eto_nasa_horizon_screen import CELLS, SEEDS, TRAIN_H, TEST_H, build, flat, scores, EPOCHS
from nasa_battery_loader import _discharge_rows
from pp_extrapolation.trajectory_operator import LatentTrajectoryOperator
from pp_extrapolation.profiled_coordinate_flow import ProfiledCoordinateFlow

VARIANTS = ('anchored_eto', 'profiled_fixed', 'profiled_learned')


def fit_variant(tr, va, seed, cap, variant):
    torch.manual_seed(seed)
    center, scale = float(tr['x'].mean()), max(float(tr['x'].std()), 1e-6)
    if variant == 'anchored_eto':
        model = LatentTrajectoryOperator(1, 1, width=32, latent_dim=12)
    else:
        model = ProfiledCoordinateFlow(learn_dictionary=variant == 'profiled_learned')
    def forward(data):
        x, t, h = torch.tensor((data['x']-center)/scale), torch.tensor(data['time']), torch.tensor(data['horizon'])
        mask = torch.ones(x.shape[:2], dtype=torch.bool)
        if variant == 'anchored_eto':
            z = model.encode(x, t, mask)
            start = model.decoder(z)
            pred = torch.stack([x[:, -1]+model.decoder(model.advance(z, h[:, j], steps=16))-start
                                for j in range(h.shape[1])], dim=1)
        else:
            pred = model.forecast(x, t, mask, h)
        return pred.squeeze(-1)*scale+center
    parameters = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(parameters, lr=.001, weight_decay=.001) if parameters else None
    best, best_epoch, state = float('inf'), 0, None
    for epoch in range(EPOCHS+1 if opt else 1):
        if epoch:
            model.train(); opt.zero_grad()
            loss = ((forward(tr)-torch.tensor(tr['y']))/scale).square().mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(parameters, 2.); opt.step()
        model.eval()
        with torch.no_grad(): score = float((forward(va).clamp(0, cap)-torch.tensor(va['y'])).square().mean())
        if score < best:
            best, best_epoch, state = score, epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    info = dict(epoch=best_epoch, validation_mse=best,
                trainable_parameters=sum(p.numel() for p in parameters))
    if isinstance(model, ProfiledCoordinateFlow):
        info.update(tau=model.log_tau.exp().detach().tolist(), ridge=model.log_ridge.exp().detach().tolist(),
                    coefficient_prior=model.prior.detach().tolist())
    return model, forward, info


def main():
    torch.set_num_threads(2)
    out = ROOT/'results/eto_profiled_improvement_screen_v1'
    out.mkdir(parents=True, exist_ok=False)
    old_path = ROOT/'results/eto_nasa_horizon_screen_v1'
    old = json.loads((old_path/'results.json').read_text())
    # Unchanged archived references on the identical task; integrity is checked.
    for path, digest in old['protocol']['code_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
    previous = np.load(old_path/'predictions.npz')
    data_root = ROOT.parent/'ca-css-ncmapss/data/affine_tail_external/nasa_battery/extracted'
    for unit, digest in old['protocol']['data_sha256'].items():
        assert hashlib.sha256((data_root/f'{unit}.mat').read_bytes()).hexdigest() == digest
    protocol = dict(status='post-test hypothesis development on already opened NASA endpoints',
        variants=VARIANTS, seeds=SEEDS, epochs=EPOCHS, train_horizons=TRAIN_H, test_horizons=TEST_H,
        acceptance='all cells and horizons must beat both archived PP core and Engression; plus novelty and original coverage, plus full PP-X comparison',
        selection='validation short-horizon MSE; no selection by test; all ablations reported',
        limitations=['one cohort, four cells; not full PP-X RUL comparison',
                     'fixed profile deterministic; repeating seeds does not create independent fitted models',
                     'relative coordinate API does not establish unordered operating-covariate coverage',
                     'no novelty established over variable projection or optimized DMD'],
        code_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
            'experiments/eto_profiled_improvement_screen.py', 'src/pp_extrapolation/profiled_coordinate_flow.py')},
        archive_sha256=hashlib.sha256((old_path/'predictions.npz').read_bytes()).hexdigest())
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    cells = {}
    for unit in CELLS:
        frame = pd.DataFrame(_discharge_rows(data_root/f'{unit}.mat', unit)).sort_values('cycle')
        cells[unit] = (frame.cycle.to_numpy(), frame.capacity.to_numpy())
    records, arrays = [], {}
    for i, unit in enumerate(CELLS):
        valid = CELLS[(i+1)%len(CELLS)]
        train_ids = [u for u in CELLS if u not in (unit, valid)]
        tr, va, te = build(cells, train_ids, TRAIN_H), build(cells, [valid], TRAIN_H), build(cells, [unit], TEST_H)
        assert np.array_equal(te['keys'].ravel(), previous[unit+'_keys'])
        assert np.array_equal(te['y'].ravel(), previous[unit+'_truth'])
        cap = max(float(tr['y'].max()), 1.)
        record = dict(test_unit=unit, validation_unit=valid, train_units=train_ids, models={})
        arrays[unit+'_truth'] = te['y'].ravel(); arrays[unit+'_keys'] = te['keys'].ravel()
        for variant in VARIANTS:
            matrix, fits = [], []
            for seed in SEEDS:
                start = time.monotonic()
                net, forward, info = fit_variant(tr, va, seed, cap, variant)
                with torch.no_grad(): pred = forward(te).clamp(0, cap).numpy().ravel()
                assert pred.shape == te['y'].ravel().shape and np.isfinite(pred).all()
                matrix.append(pred); fits.append(dict(seed=seed, seconds=time.monotonic()-start, **info))
            matrix = np.asarray(matrix); pred = matrix.mean(0)
            arrays[unit+'_'+variant] = matrix
            record['models'][variant] = dict(fits=fits, ensemble=scores(te['y'].ravel(), pred),
                seeds=[scores(te['y'].ravel(), p) for p in matrix],
                by_horizon={str(h): scores(te['y'][:, j], pred.reshape(te['y'].shape)[:, j]) for j,h in enumerate(TEST_H)})
            print(unit, variant, record['models'][variant]['ensemble'], flush=True)
        records.append(record)
    names = (*VARIANTS, 'eto', 'ppx_core', 'engression', 'persistence', 'local_linear')
    # Merge immutable reference predictions and scores, without retraining or filtering.
    for record, reference in zip(records, old['records']):
        assert record['test_unit'] == reference['test_unit']
        record['models'].update(reference['models'])
        for name in reference['models']:
            arrays[record['test_unit']+'_'+name] = previous[record['test_unit']+'_'+name]
    summary = {}
    for name in names:
        values = [r['models'][name] for r in records]
        summary[name] = dict(mean_cell_rmse=float(np.mean([v['ensemble']['rmse'] for v in values])),
            worst_cell_rmse=max(v['ensemble']['rmse'] for v in values),
            horizon32_rmse=float(np.mean([v['by_horizon']['32']['rmse'] for v in values])),
            positive_cell_r2=sum(v['ensemble']['r2']>0 for v in values),
            positive_cell_horizon_r2=sum(h['r2']>0 for v in values for h in v['by_horizon'].values()),
            cell_wins={b: sum(r['models'][name]['ensemble']['rmse'] < r['models'][b]['ensemble']['rmse'] for r in records)
                       for b in ('ppx_core','engression')},
            strict_screen_pass=all(r['models'][name]['by_horizon'][str(h)]['rmse'] < r['models'][b]['by_horizon'][str(h)]['rmse']
                for r in records for h in TEST_H for b in ('ppx_core','engression')))
    result = dict(protocol=protocol, records=records, summary=summary, promotion=False,
                  reason='Full PP-X, broad coverage and modeling novelty are not established.')
    (out/'results.json').write_text(json.dumps(result, indent=2))
    np.savez_compressed(out/'predictions.npz', **arrays)
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
