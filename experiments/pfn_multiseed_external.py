#!/usr/bin/env python3
"""Official cloud TabPFN five-seed audit on the external frozen PP splits."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

LEGACY = Path(__file__).resolve().parents[2] / "ca-css-ncmapss"
sys.path.insert(0, str(LEGACY))
from run_pp_vs_tabpfn_external_three import (
    MAX_TRAIN, VENDOR, equal_unit_subsample, validate_token,
)
from run_affine_tail_external_three import aggregate_metrics, prepare_hust, prepare_virkler
from run_affine_tail_external_nasa_health_v2 import prepare_folds
from general_affine_tail_nn import fit_feature_scale, transform_features

SEEDS=(42,43,44,45,46)

def fit_predict(train, test_x, token, seed):
    sys.path.append(str(VENDOR))
    import tabpfn_client
    from tabpfn_client import TabPFNRegressor, set_access_token
    center, scale=fit_feature_scale(train["x"])
    xtr=transform_features(train["x"],center,scale); xte=transform_features(test_x,center,scale)
    ys=max(float(np.max(train["y"])),1.0); set_access_token(token)
    model=TabPFNRegressor(model_path="auto",random_state=seed,thinking_mode=False,
                          ignore_pretraining_limits=True,fit_mode="fit_with_cache")
    model.fit(xtr,train["y"]/ys)
    pred=np.concatenate([np.asarray(model.predict(xte[i:i+512])).reshape(-1) for i in range(0,len(xte),512)])
    return np.clip(pred*ys,0,ys), getattr(tabpfn_client,"__version__","unknown")

def run_single(prepared, token):
    train, _, test, _=prepared; train,sampling=equal_unit_subsample(train,MAX_TRAIN)
    rows=[]
    for seed in SEEDS:
        pred,version=fit_predict(train,test["x"],token,seed)
        m=aggregate_metrics(test["y"],pred,test["groups"])
        rows.append({"seed":seed,"metrics":m}); print(seed,m["pooled"]["r2"],flush=True)
    v=np.array([r["metrics"]["pooled"]["r2"] for r in rows])
    return {"sampling":sampling,"client_version":version,"pooled_r2_mean":float(v.mean()),"pooled_r2_sd":float(v.std()),"runs":rows}

def run_nasa(folds,token):
    rows=[]
    truth=np.concatenate([f["test"]["y"] for f in folds]); groups=np.concatenate([f["test"]["groups"] for f in folds])
    for seed in SEEDS:
        parts=[]
        for f in folds:
            train,_=equal_unit_subsample(f["train"],MAX_TRAIN)
            pred,version=fit_predict(train,f["test"]["x"],token,seed); parts.append(pred)
        m=aggregate_metrics(truth,np.concatenate(parts),groups)
        rows.append({"seed":seed,"metrics":m}); print("nasa",seed,m["pooled"]["r2"],flush=True)
    v=np.array([r["metrics"]["pooled"]["r2"] for r in rows])
    return {"client_version":version,"pooled_r2_mean":float(v.mean()),"pooled_r2_sd":float(v.std()),"runs":rows}

def main():
    token=validate_token(); folds,_=prepare_folds()
    payload={"protocol":"same inputs, frozen split, max 3000 equal-unit rows, seeds 42-46",
             "hust":run_single(prepare_hust(),token),"virkler":run_single(prepare_virkler(),token),
             "nasa":run_nasa(folds,token)}
    out=Path("results/pfn_multiseed_fair");out.mkdir(parents=True,exist_ok=True)
    (out/"external.json").write_text(json.dumps(payload,indent=2)+"\n")
if __name__=="__main__":main()
