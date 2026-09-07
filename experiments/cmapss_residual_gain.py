#!/usr/bin/env python3
"""Validation-ensemble residual-gain audit for C-MAPSS support PP."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from cmapss_support_adaptive import append_exact_hull_distance, pp_split
from pp_extrapolation import combine_residual_gain, select_residual_gain
from pp_extrapolation.model import fit_pp, select_affine_initialization
from pp_extrapolation.support_gate import predict_components

SEEDS=(42,43,44,45,46)

def main():
    legacy=Path(__file__).resolve().parents[2]/"ca-css-ncmapss";sys.path.insert(0,str(legacy))
    from run_pae_machine_strict_hull import FDS,aggregate,filter_table,make_condition_rows,split_units
    from run_pae_machine_strict_hull_v4 import prepare_hull_validation
    from cmapss_prior_off import load_fd
    from cmapss_fleet_nn import OperatingConditionNormalizer
    from pae_extrapolation_audit import audit_convex_hull_support
    torch.set_num_threads(2)
    train0,val0,tests0,_,_=prepare_hull_validation()
    append_exact_hull_distance(train0,val0,tests0,(FDS,load_fd,split_units,filter_table,OperatingConditionNormalizer,make_condition_rows,audit_convex_hull_support))
    train,val=pp_split(train0),pp_split(val0);aff=select_affine_initialization(train,val)
    va=[];vr=[];test_components=[];base=[]
    for seed in SEEDS:
        fit=fit_pp(train,val,seed=seed,affine_selection=aff,affine_anchor_weight=100.0,residual_decay=0.0)
        a,r=predict_components(fit,val['x']);va.append(a);vr.append(r);by_fd={};bp={}
        for fd in FDS:
            a,r=predict_components(fit,tests0[fd]['x']);by_fd[fd]=(a,r);bp[fd]=combine_residual_gain(a,r,gain=1.0,output_cap=fit.target_scale)
        test_components.append(by_fd);base.append(aggregate(tests0,bp))
    choice=select_residual_gain(np.mean(va,0),np.mean(vr,0),val['y'],output_cap=fit.target_scale)
    gained=[]
    for by_fd in test_components:
        pred={fd:combine_residual_gain(*by_fd[fd],gain=choice.gain,output_cap=fit.target_scale) for fd in FDS}
        gained.append(aggregate(tests0,pred))
    def pack(rows):
        r=np.asarray([x['pooled']['r2'] for x in rows]);return {'mean_r2':float(r.mean()),'sample_sd_r2':float(r.std(ddof=1)),'runs':rows}
    candidates=list(choice.candidates);b=next(x['validation_mse'] for x in candidates if x['gain']==1.0)
    result={'gain':choice.gain,'relative_validation_mse_gain':(b-min(x['validation_mse'] for x in candidates))/b,'candidates':candidates,'pp':pack(base),'gain_pp':pack(gained)}
    out=Path('results/cmapss_residual_gain_v1');out.mkdir(parents=True,exist_ok=True);(out/'results.json').write_text(json.dumps(result,indent=2))
    print(choice.gain,result['pp']['mean_r2'],result['gain_pp']['mean_r2'])
if __name__=='__main__':main()
