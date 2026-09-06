#!/usr/bin/env python3
"""Locked PP replay: material-1 early wear to material-2 late wear."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
LEGACY=Path(__file__).resolve().parents[2]/"ca-css-ncmapss";sys.path.insert(0,str(LEGACY))
from nasa_milling_causal import prepare_causal_milling
from pp_extrapolation.metrics import regression_metrics
from pp_extrapolation.model import fit_pp,predict,select_affine_initialization
from plain_mlp_ablation import fit_plain,predict_plain

FEATURES=("health","rate","elapsed","previous_interval");SEEDS=(42,43,44,45,46)
def subset(raw,mask):
    return {"x":np.column_stack([raw[k] for k in FEATURES])[mask].astype("float32"),
            "y":raw["y"][mask].astype("float32"),"groups":raw["units"][mask].astype(str)}
def main():
    raw,audit=prepare_causal_milling();cut=float(np.quantile(raw["train"]["health"],.60))
    train=subset(raw["train"],raw["train"]["health"]<=cut)
    val=subset(raw["validation"],raw["validation"]["health"]>cut)
    test=subset(raw["source"],raw["source"]["health"]>cut)
    if not (train["x"][:,0].max()<val["x"][:,0].min() and train["x"][:,0].max()<test["x"][:,0].min()):
        raise RuntimeError("locked strict wear-tail split failed")
    affine=select_affine_initialization(train,val);rows=[]
    for seed in SEEDS:
        pp=fit_pp(train,val,seed=seed,affine_selection=affine,affine_anchor_weight=.1,residual_decay=.3)
        plain=fit_plain(train,val,seed=seed)
        pm=regression_metrics(test["y"],predict(pp,test["x"]),test["groups"])
        nm=regression_metrics(test["y"],predict_plain(plain,test["x"]),test["groups"])
        rows.append({"seed":seed,"pp":pm,"plain":nm});print(seed,pm["pooled"]["r2"],nm["pooled"]["r2"],flush=True)
    def summary(k):
        v=np.array([r[k]["pooled"]["r2"] for r in rows]);return {"mean":float(v.mean()),"sd":float(v.std())}
    payload={"status":"locked_replay","globally_untouched":False,"cut_train_q60":cut,
      "features":FEATURES,"n":{"train":len(train["y"]),"validation":len(val["y"]),"test":len(test["y"])},
      "strict_tail":True,"pp":summary("pp"),"plain":summary("plain"),"runs":rows,"source_audit":audit}
    out=Path("results/milling_locked_transfer");out.mkdir(parents=True,exist_ok=True);(out/"results.json").write_text(json.dumps(payload,indent=2)+"\n")
if __name__=="__main__":main()
