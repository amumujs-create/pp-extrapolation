#!/usr/bin/env python3
"""Statistical audit of the frozen seed-consensus output calibration results."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pp_extrapolation import consensus_tail_probability


def paired_unit_bootstrap(base, final, *, draws=20000, seed=20260907):
    units=sorted(set(base) & set(final))
    delta=np.asarray([final[u]['rmse']-base[u]['rmse'] for u in units],dtype=float)
    rng=np.random.default_rng(seed)
    sampled=delta[rng.integers(0,len(delta),size=(draws,len(delta)))].mean(axis=1)
    return {
        'units':len(units),
        'units_improved':int(np.sum(delta<0)),
        'mean_unit_rmse_delta':float(delta.mean()),
        'bootstrap_95_ci':[float(x) for x in np.quantile(sampled,[.025,.975])],
        'probability_mean_delta_below_zero':float(np.mean(sampled<0)),
    }


def main():
    cross=json.load(open('results/output_calibration_cross_dataset_v1/results.json'))['datasets']
    matr=json.load(open('results/matr_pp_validation_calibration_v1/results.json'))
    rows={}
    for name,value in cross.items():
        votes=sum(c['kind']=='affine' for c in value['choices'])
        row={
            'affine_votes':votes,
            'seed_sign_test_p':consensus_tail_probability(votes,len(value['choices'])),
            'approved':bool(value['consensus_approved']),
        }
        if row['approved']:
            row['paired_unit_bootstrap']=paired_unit_bootstrap(
                value['base']['ensemble']['per_unit'],value['final']['ensemble']['per_unit'])
        rows[name]=row
    rows['matr2019']={
        'affine_votes':5,
        'seed_sign_test_p':consensus_tail_probability(5,5),
        'approved':True,
        'paired_unit_bootstrap':paired_unit_bootstrap(
            matr['base_ensemble']['per_unit'],matr['calibrated_ensemble']['per_unit']),
    }
    approved=[v for v in rows.values() if v['approved']]
    improved=sum(v['paired_unit_bootstrap']['mean_unit_rmse_delta']<0 for v in approved)
    result={
        'method':'exact one-sided binomial seed-vote test plus paired physical-unit bootstrap',
        'null_seed_selection_probability':.5,
        'alpha':.05,
        'bootstrap_draws':20000,
        'datasets':rows,
        'approved_dataset_direction':{
            'improved':improved,'total':len(approved),
            'exact_one_sided_sign_p':consensus_tail_probability(improved,len(approved)),
        },
        'interpretation':(
            'seed p-values measure optimization-selection stability, not independent scientific '
            'replication; physical-unit bootstrap is the within-dataset inferential analysis'
        ),
    }
    out=Path('results/consensus_statistical_audit_v1')
    out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
