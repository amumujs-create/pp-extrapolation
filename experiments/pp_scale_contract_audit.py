#!/usr/bin/env python3
"""Audit the physical correction capacity of the Zn-ion BQ residual."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
import znion_bq_confirmatory as data
from pp_extrapolation import fit_boundary_quotient_pp, predict_boundary_affine, predict_boundary_quotient

OUT = ROOT / "results" / "pp_scale_contract_audit_v1"

def summary(value):
    value=np.asarray(value,float)
    return {"min":float(value.min()),"median":float(np.median(value)),
            "p95":float(np.quantile(value,.95)),"max":float(value.max())}

def main():
    torch.set_num_threads(2)
    tr,_=data.load_split("train"); va,_=data.load_split("validation")
    life=[c["cycle"][-1]-c["cycle"][0] for c in tr.values()]
    boundary=float(np.floor(.6*np.median(life))); time_scale=float(max(life))
    train=data.make_rows(tr,boundary,time_scale,"prefix")
    validation=data.make_rows(va,boundary,time_scale,"tail")
    fit=fit_boundary_quotient_pp(train,validation,seed=42,alpha=1000.,residual_bound=.5,
                                 max_epochs=300,patience=50)
    affine=predict_boundary_affine(fit,validation); pp=predict_boundary_quotient(fit,validation)
    possible=validation["margin"]*.5
    needed=np.abs(validation["y"]-affine); actual=np.abs(pp-affine)
    result={
        "status":"implementation-derived scale audit on development validation",
        "target_unit":"cycle", "output_target_normalized":False,
        "feature_time_scale":time_scale, "boundary_age":boundary,
        "health_margin":"SOH - 0.8", "residual_bound_score_units":.5,
        "proof":"softplus is 1-Lipschitz, hence abs(BQ-affine) <= margin*0.5",
        "margin":summary(validation["margin"]),
        "theoretical_max_cycle_correction":summary(possible),
        "affine_absolute_error_cycle":summary(needed),
        "actual_nn_correction_cycle":summary(actual),
        "fraction_needed_error_reachable_by_bound":float(np.mean(needed<=possible+1e-8)),
        "selected_epoch":fit.selection["selected_epoch"],
        "limitation":"The bound diagnoses representational scale; it is not a performance comparison."
    }
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
