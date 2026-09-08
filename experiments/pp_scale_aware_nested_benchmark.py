#!/usr/bin/env python3
"""Nested unit-held-out development benchmark for scale-aware Zn-ion BQ."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"experiments")]
import znion_bq_confirmatory as data
from pp_extrapolation import fit_boundary_quotient_pp,predict_boundary_affine,predict_boundary_quotient,regression_metrics

OUT=ROOT/"results"/"pp_scale_aware_nested_benchmark_v1"
SEEDS=(42,43); ADD_BOUNDS=(.5,2.,6.); MULT_BOUNDS=(.5,1.,2.)

def scaled(rows, factor):
    return {**rows,"y":np.asarray(rows["y"],np.float32)/float(factor)}

def score(y,p,g): return regression_metrics(np.asarray(y),np.asarray(p),np.asarray(g))

def fit_predict(train,val,test,*,scale,mode,bound,seed,max_epochs=220):
    tr=scaled(train,scale); va=scaled(val,scale); te=scaled(test,scale)
    fit=fit_boundary_quotient_pp(tr,va,seed=seed,alpha=1000.,residual_bound=bound,
                                 correction_mode=mode,max_epochs=max_epochs,patience=40)
    pred=predict_boundary_quotient(fit,te)*scale
    affine=predict_boundary_affine(fit,te)*scale
    return pred,affine,fit.selection

def main():
    torch.set_num_threads(2)
    tr,_=data.load_split("train"); va,_=data.load_split("validation"); dev={**tr,**va}
    boundary=152.; eligible=[u for u,c in sorted(dev.items()) if c["cycle"][-1]-c["cycle"][0]-boundary>=50]
    names=("A0_raw_additive","A1_normalized_affine","A2_normalized_additive","A3_multiplicative","A4_support_shrunk")
    folds=[]; pooled={a:[] for a in names}
    for oi,outer in enumerate(eligible):
        inner_candidates=[u for u in eligible if u!=outer]
        inner_val=inner_candidates[oi%len(inner_candidates)]
        train_cells={u:c for u,c in dev.items() if u not in (outer,inner_val)}
        val_cells={inner_val:dev[inner_val]}; test_cells={outer:dev[outer]}
        time_scale=float(max(c["cycle"][-1]-c["cycle"][0] for c in train_cells.values()))
        train=data.make_rows(train_cells,boundary,time_scale,"prefix")
        val=data.make_rows(val_cells,boundary,time_scale,"tail")
        test=data.make_rows(test_cells,boundary,time_scale,"tail")
        selections={}
        for arm,mode,candidates in (("A2_normalized_additive","additive",ADD_BOUNDS),
                                    ("A3_multiplicative","multiplicative",MULT_BOUNDS)):
            screen=[]
            for bound in candidates:
                p,_,s=fit_predict(train,val,val,scale=time_scale,mode=mode,bound=bound,seed=42)
                screen.append({"bound":bound,"pooled_r2":score(val["y"],p,val["groups"])["pooled"]["r2"],
                               "epoch":s["selected_epoch"]})
            selections[arm]={"selected_bound":max(screen,key=lambda x:x["pooled_r2"])["bound"],"screen":screen}
        center=np.asarray(train["x"]).mean(0); spread=np.asarray(train["x"]).std(0);spread[spread<1e-6]=1
        ztr=(np.asarray(train["x"])-center)/spread; zv=(np.asarray(val["x"])-center)/spread
        distance=np.linalg.norm(np.maximum(ztr.min(0)-zv,0)+np.maximum(zv-ztr.max(0),0),axis=1)/np.sqrt(ztr.shape[1])
        support_screen=[]
        for source,mode in (("A2_normalized_additive","additive"),("A3_multiplicative","multiplicative")):
            bound=selections[source]["selected_bound"]
            p,a,_=fit_predict(train,val,val,scale=time_scale,mode=mode,bound=bound,seed=42)
            for beta in (0.,.1,.5,1.,2.,5.):
                q=a+np.exp(-beta*distance)*(p-a)
                support_screen.append({"source":source,"mode":mode,"bound":bound,"beta":beta,
                                       "pooled_r2":score(val["y"],q,val["groups"])["pooled"]["r2"]})
        support_choice=max(support_screen,key=lambda x:x["pooled_r2"])
        selections["A4_support_shrunk"]={"selected":support_choice,"screen":support_screen}
        fold={"outer_unit":outer,"inner_validation_unit":inner_val,"output_scale":time_scale,
              "selection":selections,"arms":{}}
        configs={
            "A0_raw_additive":(1.,"additive",.5),
            "A1_normalized_affine":(time_scale,"additive",.5),
            "A2_normalized_additive":(time_scale,"additive",selections["A2_normalized_additive"]["selected_bound"]),
            "A3_multiplicative":(time_scale,"multiplicative",selections["A3_multiplicative"]["selected_bound"]),
        }
        for arm,(scale,mode,bound) in configs.items():
            preds=[]; aff=[]; runs=[]
            for seed in SEEDS:
                p,a,s=fit_predict(train,val,test,scale=scale,mode=mode,bound=bound,seed=seed,
                                  max_epochs=0 if arm=="A1_normalized_affine" else 220)
                preds.append(p); aff.append(a); runs.append({"seed":seed,"epoch":s["selected_epoch"]})
            prediction=np.mean(preds,axis=0) if arm!="A1_normalized_affine" else np.mean(aff,axis=0)
            metrics=score(test["y"],prediction,test["groups"])
            fold["arms"][arm]={"metrics":metrics,"runs":runs,"bound":bound,"mode":mode}
            pooled[arm].append((test["y"],prediction,test["groups"]))
        choice=support_choice; preds=[]; aff=[]; runs=[]
        zq=(np.asarray(test["x"])-center)/spread
        distance=np.linalg.norm(np.maximum(ztr.min(0)-zq,0)+np.maximum(zq-ztr.max(0),0),axis=1)/np.sqrt(ztr.shape[1])
        for seed in SEEDS:
            p,a,s=fit_predict(train,val,test,scale=time_scale,mode=choice["mode"],bound=choice["bound"],seed=seed)
            preds.append(a+np.exp(-choice["beta"]*distance)*(p-a));aff.append(a);runs.append({"seed":seed,"epoch":s["selected_epoch"]})
        prediction=np.mean(preds,axis=0);metrics=score(test["y"],prediction,test["groups"])
        fold["arms"]["A4_support_shrunk"]={"metrics":metrics,"runs":runs,**choice,
                                             "distance":{"median":float(np.median(distance)),"max":float(distance.max())}}
        pooled["A4_support_shrunk"].append((test["y"],prediction,test["groups"]))
        folds.append(fold); print("fold",outer,{a:round(fold["arms"][a]["metrics"]["pooled"]["r2"],3) for a in pooled},flush=True)
    aggregate={}
    for arm,parts in pooled.items():
        y=np.concatenate([v[0] for v in parts]); p=np.concatenate([v[1] for v in parts]); g=np.concatenate([v[2] for v in parts])
        aggregate[arm]=score(y,p,g)
    result={"status":"nested unit-held-out development; not untouched confirmation",
            "boundary":boundary,"seeds":SEEDS,"eligible_outer_units":eligible,
            "protocol":"outer and inner-validation units excluded from training; scale fit on inner train; bounds selected on inner validation pooled R2",
            "folds":folds,"aggregate":aggregate,
            "limitations":["fixed boundary and candidate families were motivated after earlier Zn-ion tests","small and imbalanced development cohort","A0 uses raw-cycle target while A1-A3 use train-only output normalization"]}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({a:{"pooled_r2":m["pooled"]["r2"],"macro_r2":m["unit_macro_r2"]} for a,m in aggregate.items()},indent=2))

if __name__=="__main__": main()
