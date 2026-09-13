"""One frozen rejection screen: held-out NASA cells and out-of-range horizons.

Capacity forecasting, NOT the historical PP-X RUL benchmark. PP-X core is an
adapted baseline, not the complete frozen PP-X policy. No promotion can follow
from this screen alone. Output directories are exclusive and never overwritten.
"""
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
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / '.benchmark_deps'),
               str(ROOT.parent / 'ca-css-ncmapss')]
from nasa_battery_loader import _discharge_rows
from pp_extrapolation.trajectory_operator import LatentTrajectoryOperator
from pp_extrapolation.model import fit_pp, predict, select_affine_initialization
from engression.engression import Engressor

CELLS = ('B0005', 'B0006', 'B0007', 'B0018')
SEEDS = (42, 43, 44)
EPOCHS = 150
ENDS = (15, 23, 31, 39)  # index of the LAST observed discharge
TRAIN_H = (1, 2, 4)
TEST_H = (8, 16, 32)
NAMES = ('eto', 'ppx_core', 'engression', 'persistence', 'local_linear')


def build(cells, ids, horizons):
    x, times, target, groups, keys = [], [], [], [], []
    for unit in ids:
        t, y = cells[unit]
        for end in ENDS:
            indices = np.arange(end - 15, end + 1)
            # Inclusive prefix ending at end; horizon 1 means index end+1.
            future = end + np.asarray(horizons)
            assert np.all(t[future] > t[end])
            assert np.allclose(t[future] - t[end], horizons)
            x.append(y[indices, None]); times.append(t[indices] - t[end - 15])
            target.append(y[future]); groups.append(unit)
            keys.append([f'{unit}:{end}:{h}' for h in horizons])
    return dict(x=np.asarray(x, np.float32), time=np.asarray(times, np.float32) / 32,
                y=np.asarray(target, np.float32), groups=np.asarray(groups),
                horizon=np.tile(np.asarray(horizons, np.float32), (len(x), 1)) / 32,
                keys=np.asarray(keys))


def flat(data):
    n, h = data['y'].shape
    features = np.concatenate((data['x'].reshape(n, -1), data['time']), axis=1)
    return dict(x=np.column_stack((np.repeat(features, h, axis=0), data['horizon'].ravel())),
                y=data['y'].ravel(), groups=np.repeat(data['groups'], h))


def eng_mean(model, x, seed):
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(seed)
        return model.predict_onebatch(torch.tensor(x), target='mean', sample_size=128).numpy().ravel()


def fit_eto(tr, va, seed, cap):
    torch.manual_seed(seed)
    mean, scale = float(tr['x'].mean()), max(float(tr['x'].std()), 1e-6)
    model = LatentTrajectoryOperator(1, 1, width=32, latent_dim=12)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)

    def forward(data):
        x = torch.tensor((data['x'] - mean) / scale)
        return model.forecast(x, torch.tensor(data['time']), torch.ones(x.shape[:2], dtype=torch.bool),
                              torch.tensor(data['horizon']), steps=16).squeeze(-1) * scale + mean

    best, state, epoch_best = float('inf'), None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train(); opt.zero_grad()
        loss = ((forward(tr) - torch.tensor(tr['y'])) / scale).square().mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2); opt.step()
        model.eval()
        with torch.no_grad(): score = float((forward(va).clamp(0, cap) - torch.tensor(va['y'])).square().mean())
        if score < best:
            best, state, epoch_best = score, copy.deepcopy(model.state_dict()), epoch
    model.load_state_dict(state)
    return model, forward, dict(epoch=epoch_best, validation_mse=best,
                                parameters=sum(p.numel() for p in model.parameters()))


def scores(y, prediction):
    mse = float(np.mean((prediction - y)**2))
    return dict(rmse=mse**.5, r2=float(1 - mse / np.var(y)))


def main():
    torch.set_num_threads(2)
    out = ROOT / 'results/eto_nasa_horizon_screen_v1'
    out.mkdir(parents=True, exist_ok=False)
    data_root = ROOT.parent / 'ca-css-ncmapss/data/affine_tail_external/nasa_battery/extracted'
    protocol = dict(status='retrospective rejection screen, no promotion authority',
        train_horizons=TRAIN_H, validation_horizons=TRAIN_H, test_horizons=TEST_H,
        last_observed_indices=ENDS, seeds=SEEDS, epochs=EPOCHS,
        target='future measured capacity, NOT historical RUL',
        information='all learned models receive the same 16 observations, relative times and query horizon',
        support='test query horizon strictly exceeds train maximum: sufficient for convex hull exclusion in augmented query input',
        comparison='fixed recipes; 150 full-batch epochs; same seeds and clip; native scaling, optimizers, parameter counts differ',
        ppx_reference='adapted PP core, ridge alpha=10; NOT full frozen PP-X policy',
        predeclared_rejection='no further ETO development unless it beats both baselines on every cell and every horizon; novelty and original-domain coverage must also pass',
        code_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
            'experiments/eto_nasa_horizon_screen.py', 'src/pp_extrapolation/trajectory_operator.py',
            'src/pp_extrapolation/model.py', '.benchmark_deps/engression/engression.py')},
        data_sha256={u: hashlib.sha256((data_root/f'{u}.mat').read_bytes()).hexdigest() for u in CELLS})
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    cells = {}
    for unit in CELLS:
        frame = pd.DataFrame(_discharge_rows(data_root/f'{unit}.mat', unit)).sort_values('cycle')
        cells[unit] = (frame.cycle.to_numpy(), frame.capacity.to_numpy())
    records, archive = [], {}
    for i, test_unit in enumerate(CELLS):
        validation_unit = CELLS[(i+1) % len(CELLS)]
        train_units = [u for u in CELLS if u not in (test_unit, validation_unit)]
        tr, va, te = build(cells, train_units, TRAIN_H), build(cells, [validation_unit], TRAIN_H), build(cells, [test_unit], TEST_H)
        assert tr['horizon'].max() < te['horizon'].min()
        a, b, c = map(flat, (tr, va, te))
        cap = max(float(a['y'].max()), 1.)
        affine = select_affine_initialization(a, b, alphas=(10.,))
        matrices = {n: [] for n in NAMES}
        fits = []
        for seed in SEEDS:
            start = time.monotonic()
            model, forward, info = fit_eto(tr, va, seed, cap)
            with torch.no_grad(): p = forward(te).clamp(0, cap).numpy().ravel()
            matrices['eto'].append(p)
            pp = fit_pp(a, b, seed=seed, affine_selection=affine, max_epochs=EPOCHS, patience=EPOCHS,
                        width=32, learning_rate=.001, weight_decay=.001)
            matrices['ppx_core'].append(predict(pp, c['x']))
            torch.manual_seed(seed)
            eng = Engressor(a['x'].shape[1], 1, num_layer=2, hidden_dim=32, noise_dim=32,
                            beta=1., lr=.001, standardize=True, device='cpu', check_device=False, verbose=False)
            best, state, ep = float('inf'), None, 0
            for epoch in range(1, EPOCHS+1):
                eng.train(torch.tensor(a['x']), torch.tensor(a['y'][:, None]), num_epoches=1,
                          batch_size=len(a['y']), verbose=False)
                v = np.clip(eng_mean(eng, b['x'], 10000+seed), 0, cap)
                loss = float(np.mean((v-b['y'])**2))
                if loss < best: best, state, ep = loss, copy.deepcopy(eng.model.state_dict()), epoch
            eng.model.load_state_dict(state)
            matrices['engression'].append(np.clip(eng_mean(eng, c['x'], 20000+seed), 0, cap))
            last = te['x'][:, -1, 0]
            matrices['persistence'].append(np.clip(np.repeat(last, 3), 0, cap))
            centered_t = np.arange(16)-7.5
            slope = np.sum(te['x'][:, :, 0]*centered_t, axis=1)/np.sum(centered_t**2)
            linear = last[:, None] + slope[:, None]*te['horizon']*32
            matrices['local_linear'].append(np.clip(linear.ravel(), 0, cap))
            fits.append(dict(seed=seed, eto=info, engression=dict(epoch=ep, validation_mse=best,
                parameters=sum(p.numel() for p in eng.model.parameters())),
                ppx=dict(epoch=pp.selection['selected_epoch'], parameters=sum(p.numel() for p in pp.model.parameters())),
                seconds=time.monotonic()-start))
            print('FIT', test_unit, seed, round(time.monotonic()-start, 1), flush=True)
        archive[test_unit+'_truth'] = c['y']; archive[test_unit+'_keys'] = te['keys'].ravel()
        rec = dict(test_unit=test_unit, train_units=train_units, validation_unit=validation_unit, fits=fits, models={})
        for name, values in matrices.items():
            mat = np.asarray(values); p = mat.mean(0)
            archive[test_unit+'_'+name] = mat
            rec['models'][name] = dict(ensemble=scores(c['y'], p),
                seeds=[scores(c['y'], v) for v in mat],
                by_horizon={str(h): scores(te['y'][:, j], p.reshape(te['y'].shape)[:, j]) for j, h in enumerate(TEST_H)})
        records.append(rec)
        (out/'progress.json').write_text(json.dumps(records, indent=2))
        print('RESULT', test_unit, {n: r['ensemble'] for n, r in rec['models'].items()}, flush=True)
    summary = {}
    for name in NAMES:
        values = [r['models'][name] for r in records]
        summary[name] = dict(mean_cell_rmse=float(np.mean([v['ensemble']['rmse'] for v in values])),
                            worst_cell_rmse=max(v['ensemble']['rmse'] for v in values),
                            positive_cell_r2=sum(v['ensemble']['r2']>0 for v in values),
                            positive_cell_horizon_r2=sum(h['r2']>0 for v in values for h in v['by_horizon'].values()))
    wins = {baseline: sum(r['models']['eto']['ensemble']['rmse'] < r['models'][baseline]['ensemble']['rmse'] for r in records)
            for baseline in ('ppx_core', 'engression')}
    strict_pass = all(r['models']['eto']['by_horizon'][str(h)]['rmse'] < r['models'][b]['by_horizon'][str(h)]['rmse']
                      for r in records for h in TEST_H for b in ('ppx_core', 'engression'))
    result = dict(protocol=protocol, records=records, summary=summary, wins=wins,
                  strict_performance_screen_pass=strict_pass, promotion=False,
                  reason='Novelty and full original-domain coverage unestablished; full PP-X comparison missing.')
    np.savez_compressed(out/'predictions.npz', **archive)
    (out/'results.json').write_text(json.dumps(result, indent=2))
    print('SUMMARY', json.dumps(dict(summary=summary, wins=wins, strict_pass=strict_pass)), flush=True)


if __name__ == '__main__':
    main()
