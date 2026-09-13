"""Resume failed transfer: fixed feature/budget controls, not a new model search.

NASA: scalar vs causal short history on IDENTICAL rows, crossed with 60/400
epochs. MICH: same seven features crossed with budget. Both compare direct CIST,
its integral residual, latent PP and official Engression. Native loss/scaling
remain different; this is not a full policy/tuning comparison.
"""
from pathlib import Path
import copy
import hashlib
import json
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'.benchmark_deps'), str(ROOT/'src'), str(ROOT/'experiments'),
               str(ROOT.parent/'ca-css-ncmapss')]
from pp_extrapolation.innovation_slope_transport import fit_innovation_slope_transport, predict_innovation_slope_transport
from pp_extrapolation.anchored_residual_transport import fit_anchored_residual, predict_anchored_residual
from pp_extrapolation.regime_mixture import fit_latent_regime_pp, predict_latent_regime
from pp_extrapolation.model import select_affine_initialization
from engression.engression import Engressor

SEEDS = (42, 43, 44)
BUDGETS = (60, 400)
NAMES = ('direct', 'integral', 'latent_pp', 'engression')


def metric(y, p, groups):
    error = (p-y)**2
    per_unit = [float(np.mean(error[groups == u])) for u in np.unique(groups)]
    return dict(r2=float(1-error.sum()/np.sum((y-y.mean())**2)), rmse=float(np.sqrt(error.mean())),
                unit_rmse=float(np.mean(np.sqrt(per_unit))), worst_unit_rmse=float(np.sqrt(max(per_unit))))


def nasa_inputs():
    from run_affine_tail_external_nasa_health_v2 import prepare_folds
    from run_affine_tail_external_three import nasa_cells, NASA_BOUNDARY
    original, _ = prepare_folds()
    cells = nasa_cells()
    rich = []
    for f in original:
        out = {}
        cutoff = float(f['train']['x'][:, 0].min())
        for part, names in [('train', f['train_cells']), ('validation', [f['validation_cell']]), ('test', [f['test_cell']])]:
            xs, ys, gs, keys = [], [], [], []
            for unit in names:
                frame = cells[unit]
                health = (frame.capacity.to_numpy()-NASA_BOUNDARY)/(float(frame.iloc[0].capacity)-NASA_BOUNDARY)
                # Selection follows original float64 frame mask, with the
                # original threshold expressed at float32 precision.
                threshold = min(float(((cells[u].capacity-NASA_BOUNDARY)/(float(cells[u].iloc[0].capacity)-NASA_BOUNDARY))[
                    ((cells[u].capacity-NASA_BOUNDARY)/(float(cells[u].iloc[0].capacity)-NASA_BOUNDARY)) >= .5].min()) for u in f['train_cells'])
                keep = health >= .5 if part == 'train' else health < threshold
                t = frame.cycle.to_numpy()
                for i in np.flatnonzero(keep):
                    a = max(0, i-2)
                    z = health[a:i+1]
                    slope = (z[-1]-z[0])/max(float(t[i]-t[a]), 1.)
                    xs.append([health[i], z.mean(), slope])
                    ys.append(frame.RUL.iloc[i]); gs.append(unit); keys.append(f'{unit}:{t[i]}')
            rows = dict(x=np.asarray(xs, np.float32), y=np.asarray(ys, np.float32), groups=np.asarray(gs), keys=np.asarray(keys))
            assert np.array_equal(rows['y'], f[part]['y'])
            assert np.array_equal(rows['groups'], f[part]['groups'])
            np.testing.assert_allclose(rows['x'][:, :1], f[part]['x'], rtol=1e-6)
            out[part] = rows
        rich.append(out)
    scalar = [{p: {**r, 'x': r['x'][:, :1]} for p, r in f.items()} for f in rich]
    return [('NASA', 'scalar', scalar), ('NASA', 'causal_short', rich)]


def mich_inputs():
    from pae_boundary_realdata import prepare_dataset
    from distance_uncertainty_pp import prepare_battery
    raw, _ = prepare_dataset('mich')
    tr, va, te = prepare_battery(raw)
    return ('MICH', 'seven_features', [dict(train=tr, validation=va, test=te)])


def eng_predict(model, x, seed):
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return model.predict_onebatch(torch.tensor(x), target='mean', sample_size=128).numpy().ravel()


def eng_fit(tr, va, seed, epochs, patience):
    torch.manual_seed(seed)
    net = Engressor(tr['x'].shape[1], 1, num_layer=2, hidden_dim=32, noise_dim=32,
                    beta=1., lr=.001, standardize=True, device='cpu', check_device=False, verbose=False)
    cap = max(float(tr['y'].max()), 1.)
    best, ep, state = float('inf'), 0, None
    for e in range(1, epochs+1):
        net.train(torch.tensor(tr['x']), torch.tensor(tr['y'][:, None]),
                  num_epoches=1, batch_size=min(512, len(tr['y'])), verbose=False)
        pred = np.clip(eng_predict(net, va['x'], seed+10000), 0, cap)
        score = float(np.mean((pred-va['y'])**2))
        if score < best: best, ep, state = score, e, copy.deepcopy(net.model.state_dict())
        if e-ep >= patience: break
    net.model.load_state_dict(state)
    return net, dict(selected_epoch=ep, validation_mse=best, executed_epochs=e)


def audit_ncmapss(out):
    from ncmapss_tra_quantile_split import make_tra_hard_split
    from ncmapss_config import UNIFIED_VAL_UNITS
    from ncmapss_pp_benchmark import rows
    from apps.ncmapss_data_utils import FEATURE_COLS
    s = make_tra_hard_split(ROOT/'data/N-CMAPSS_DS02-006.h5', max_windows_per_unit=1500, random_seed=42)
    tr, va, te = [rows(p, list(FEATURE_COLS)) for p in (s.train, s.val, s.test)]
    gs = [set(p['groups']) for p in (tr, va, te)]
    keep = np.isin(va['groups'], np.asarray(UNIFIED_VAL_UNITS, str))
    strict = {k: v[keep] for k, v in va.items()}
    assert len(strict['y']) > 0
    assert not gs[0] & gs[2] and not gs[1] & gs[2]
    assert not set(strict['groups']) & (gs[0] | gs[2])
    report = dict(historical_groups=[sorted(g) for g in gs], n=[len(p['y']) for p in (tr,va,te)],
        train_validation_overlap=sorted(gs[0]&gs[1]), test_overlap=sorted(gs[2]&(gs[0]|gs[1])),
        strict_validation_groups=sorted(set(strict['groups'])), strict_validation_n=len(strict['y']),
        cause='loader intentionally sets val_units=train_unit_ids+val_unit_ids; transfer runner asserted all three disjoint',
        decision='strict unit20 validation exported as a NEW protocol; original test rows/train preserved; all models must be refit/reselected before ranking')
    (out/'ncmapss_split_audit.json').write_text(json.dumps(report, indent=2))
    np.savez_compressed(out/'ncmapss_strict_parts.npz', **{f'{p}_{k}':v for p,r in [('train',tr),('validation',strict),('test',te)] for k,v in r.items()})
    print('NCMAPSS', report, flush=True)


def main():
    torch.set_num_threads(2)
    out = ROOT/'results/main9_transfer_diagnosis_v1'
    resume = '--resume' in sys.argv
    out.mkdir(parents=True, exist_ok=resume)
    protocol = dict(status='retrospective diagnosis, no successor promotion', seeds=SEEDS,
        budgets=BUDGETS, patience={60:20, 400:80},
        design='NASA identical rows: scalar vs causal short, crossed with budget; MICH fixed seven features, budget control',
        unchanged='original RUL labels, unit splits, test rows; all current models see identical features within each arm',
        native_differences='loss, scaling, optimizer and parameter count differ; equal epoch cap not equal FLOPs',
        integral='same zero-start residual, additional 60 epoch cap; performance not an equal-total-budget baseline',
        ppx='latent PP executor with fixed separation0; not complete frozen cross-domain PP-X policy',
        limitation='short history control includes original early rows; archived NASA dropped earliest two rows; no exact final-pipeline replay claim',
        hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
            'experiments/main9_transfer_diagnosis.py','src/pp_extrapolation/innovation_slope_transport.py',
            'src/pp_extrapolation/anchored_residual_transport.py','src/pp_extrapolation/regime_mixture.py')})
    if resume:
        protocol = json.loads((out/'protocol.json').read_text())
        records = json.loads((out/'results.json').read_text())['records']
        (out/'resume.json').write_text(json.dumps(dict(
            reason='serialize numpy unit IDs as strings; resume already completed arms without rerunning',
            current_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    else:
        (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
        audit_ncmapss(out)
        records = []
    for dataset, features, folds in [*nasa_inputs(), mich_inputs()]:
        y = np.concatenate([f['test']['y'] for f in folds])
        groups = np.concatenate([f['test']['groups'] for f in folds])
        for epochs in BUDGETS:
            if any(r['dataset']==dataset and r['features']==features and r['epochs']==epochs for r in records):
                continue
            matrix = {name: [[] for _ in SEEDS] for name in NAMES}
            diagnostics = []
            for fi, f in enumerate(folds):
                tr, va, te = [f[p] for p in ('train','validation','test')]
                gs=[set(p['groups']) for p in (tr,va,te)]
                assert all(not gs[i]&gs[j] for i in range(3) for j in range(i))
                assert te['x'][:,0].max() < tr['x'][:,0].min(), 'health-support extrapolation lost'
                cap=max(float(tr['y'].max()),1.)
                affine=select_affine_initialization(tr,va)
                patience=20 if epochs==60 else 80
                for j,seed in enumerate(SEEDS):
                    start=time.monotonic()
                    base=fit_innovation_slope_transport(tr,va,progress_index=0,seed=seed,route='direct_only',
                        width=32,max_epochs=epochs,patience=patience,learning_rate=.001,weight_decay=.1,
                        steps=4,beta=1.,mean_weight=.5,mean_objective='zero_noise',validation_samples=128,
                        validation_clip_to_train_range=True)
                    matrix['direct'][j].append(np.clip(predict_innovation_slope_transport(base,te['x'],samples=128),0,cap))
                    residual=fit_anchored_residual(base,tr,va,seed=seed,route='integral')
                    matrix['integral'][j].append(predict_anchored_residual(residual,te['x']))
                    pp=fit_latent_regime_pp(tr,va,seed=seed,affine_selection=affine,width=32,
                        max_epochs=epochs,patience=patience,separation_weight=0.,gate_weight=0.)
                    matrix['latent_pp'][j].append(predict_latent_regime(pp,te['x']))
                    eng, eng_info=eng_fit(tr,va,seed,epochs,patience)
                    matrix['engression'][j].append(np.clip(eng_predict(eng,te['x'],seed+20000),0,cap))
                    diagnostics.append(dict(fold=fi,seed=seed,n=[len(p['y']) for p in (tr,va,te)],
                        dimensions=tr['x'].shape[1],unit_ids=[sorted(map(str,g)) for g in gs],
                        direct_epoch=base.selected_epoch,direct_validation_mse=base.validation_mse,
                        integral_accepted=residual.accepted,integral=residual.diagnostics,
                        latent_pp=pp.selection,engression=eng_info,seconds=time.monotonic()-start))
                    print('FIT',dataset,features,epochs,fi,seed,round(time.monotonic()-start,1),flush=True)
            matrices={k:np.asarray([np.concatenate(parts) for parts in v]) for k,v in matrix.items()}
            record=dict(dataset=dataset,features=features,epochs=epochs,diagnostics=diagnostics,
                models={k:dict(ensemble=metric(y,p.mean(0),groups),seeds=[metric(y,v,groups) for v in p],
                              positive_units=sum(metric(y[groups==g],p.mean(0)[groups==g],groups[groups==g])['r2']>0 for g in np.unique(groups)))
                        for k,p in matrices.items()})
            for p in matrices.values(): assert p.shape==(3,len(y)) and np.isfinite(p).all()
            records.append(record)
            prediction_path=out/f'{dataset}_{features}_{epochs}.npz'
            if prediction_path.exists():
                prior=np.load(prediction_path)
                for k,p in matrices.items(): np.testing.assert_allclose(prior[k],p,rtol=1e-6,atol=1e-6)
            else:
                np.savez_compressed(prediction_path,truth=y,groups=groups,**matrices)
            (out/'results.json').write_text(json.dumps(dict(protocol=protocol,records=records,complete=len(records)==6),indent=2))
            print('SCORED',dataset,features,epochs,{k:v['ensemble']['r2'] for k,v in record['models'].items()},flush=True)


if __name__=='__main__':
    main()
