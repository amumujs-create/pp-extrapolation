#!/usr/bin/env python3
"""Train-only counterfactual residual-trust learning on HUST."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
LEGACY=Path(__file__).resolve().parents[2]/"ca-css-ncmapss";sys.path.insert(0,str(LEGACY))
from run_affine_tail_external_three import aggregate_metrics,prepare_hust
from pp_extrapolation.model import fit_pp,predict,select_affine_initialization
from pp_extrapolation.priors import CounterfactualRays

def rays(train):
    boundary=[];outer=[];decay=[]
    for group in np.unique(train["groups"]):
        rows=train["x"][train["groups"]==group];rows=rows[np.argsort(rows[:,1])]
        if len(rows)<12: continue
        inner,b=rows[-11],rows[-1];delta=b-inner
        for ratio in (.5,1.,2.,4.):
            boundary.append(b);outer.append(b+ratio*delta);decay.append(np.exp(-ratio))
    return CounterfactualRays(np.asarray(boundary),np.asarray(outer),np.asarray(decay),np.ones(len(decay)))

def main():
    train,val,test,_=prepare_hust();prior=rays(train);affine=select_affine_initialization(train,val);rows=[]
    support_choices={42:(0.,.3),43:(.1,1.),44:(.1,1.),45:(0.,1.),46:(.1,.3)}
    for seed in (42,43,44,45,46):
        candidates=[]
        anchor,decay=support_choices[seed]
        for weight in (0.,.01,.1,1.):
            fit=fit_pp(train,val,seed=seed,affine_selection=affine,affine_anchor_weight=anchor,
                       residual_decay=decay,counterfactual_rays=prior,counterfactual_weight=weight)
            candidates.append((fit.selection["best_validation_mse"],weight,fit))
        _,weight,fit=min(candidates,key=lambda z:(z[0],z[1]))
        m=aggregate_metrics(test["y"],predict(fit,test["x"]),test["groups"])
        rows.append({"seed":seed,"anchor":anchor,"decay":decay,"counterfactual_weight":weight,"metrics":m})
        print(seed,anchor,decay,weight,m["pooled"]["r2"],flush=True)
    v=np.array([r["metrics"]["pooled"]["r2"] for r in rows]);payload={"n_train_only_rays":len(prior.decay),
      "pooled_r2_mean":float(v.mean()),"pooled_r2_sd":float(v.std()),"runs":rows}
    out=Path("results/counterfactual_trust_hust");out.mkdir(parents=True,exist_ok=True);(out/"results.json").write_text(json.dumps(payload,indent=2)+"\n")
if __name__=="__main__":main()
