#!/usr/bin/env python3
"""Structural PP experiment on FEMTO, XJTU and NASA milling."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"experiments"),str(ROOT.parent/"ca-css-ncmapss")]
from pp_extrapolation import fit_lifetime_scale_pp,predict_lifetime_scale
from femto_sensor_adapter_v2 import load as load_femto
from femto_corrected_benchmark_v2 import causal_features,rows,TRAIN,VAL,TEST

SEEDS=(42,43,44,45,46); OUT=ROOT/"results/lifetime_scale_pp_three_domains_v1"
GRID=({"width":w,"learning_rate":lr,"weight_decay":wd,"residual_bound":rb,"support_decay":sd}
      for w in (16,32) for lr in (3e-4,1e-3) for wd in (.01,.1) for rb in (.75,1.5) for sd in (.1,.5))
GRID=tuple(GRID)

def metric(y,p): return {"r2":float(r2_score(y,p)),"rmse":float(mean_squared_error(y,p)**.5)}

def tune(name,tr,va,te,etr,eva,ete):
    search=[]
    for cfg in GRID:
        f=fit_lifetime_scale_pp(tr,va,etr,eva,seed=42,max_epochs=220,patience=35,**cfg)
        search.append({**cfg,"validation_rmse":metric(va["y"],predict_lifetime_scale(f,va["x"],eva))["rmse"]})
    chosen=min(search,key=lambda r:r["validation_rmse"]); pred=[]; runs=[]
    for seed in SEEDS:
        f=fit_lifetime_scale_pp(tr,va,etr,eva,seed=seed,max_epochs=350,patience=60,**{k:chosen[k] for k in ("width","learning_rate","weight_decay","residual_bound","support_decay")})
        p=predict_lifetime_scale(f,te["x"],ete); pred.append(p); runs.append({"seed":seed,"metrics":metric(te["y"],p),"selection":f.selection})
    pred=np.asarray(pred); result={"selected":chosen,"runs":runs,"ensemble":metric(te["y"],pred.mean(0)),"n":{"train":len(etr),"validation":len(eva),"test":len(ete)}}
    np.savez_compressed(OUT/f"{name}_predictions.npz",y=te["y"],groups=te["groups"],predictions=pred)
    print(name,result["ensemble"],flush=True); return result

def femto():
    d=load_femto(); x=causal_features(d); tr,tri=rows(d,x,TRAIN); va,vai=rows(d,x,VAL); te,tei=rows(d,x,TEST,True)
    return tune("femto",tr,va,te,d["elapsed_s"][tri]+10,d["elapsed_s"][vai]+10,d["elapsed_s"][tei]+10)

def xjtu():
    from xjtu_untouched import build_cache,windows
    raw=build_cache(); names=("37.5Hz11kN","35Hz12kN","40Hz10kN"); parts=[windows(raw,n) for n in names]
    elapsed=[]
    for n,p in zip(names,parts):
        vals=[]
        for unit in p["groups"]:
            # windows start at the eighth recording; group order is chronological.
            vals.append(0)
        for unit in np.unique(p["groups"]):
            ix=np.flatnonzero(p["groups"]==unit); vals_arr=np.arange(8,8+len(ix),dtype=float); np.asarray(vals)[ix] if False else None
        e=np.zeros(len(p["y"]),float)
        for unit in np.unique(p["groups"]):
            ix=np.flatnonzero(p["groups"]==unit); e[ix]=np.arange(8,8+len(ix))
        elapsed.append(e)
    return tune("xjtu",*parts,*elapsed)

def milling():
    from nasa_milling_causal import prepare_causal_milling
    from milling_locked_transfer import subset
    raw,_=prepare_causal_milling(); cut=float(np.quantile(raw["train"]["health"],.60))
    source=(subset(raw["train"],raw["train"]["health"]<=cut),subset(raw["validation"],raw["validation"]["health"]>cut),subset(raw["source"],raw["source"]["health"]>cut))
    elapsed=[np.maximum(p["x"][:,2],1e-3) for p in source]
    return tune("milling",*source,*elapsed)

def main():
    import torch; torch.set_num_threads(2); OUT.mkdir(parents=True,exist_ok=True)
    result={"model":"lifetime-scale PP: log(total-life)=frozen affine+support-decayed bounded NN; RUL=max(total-life-elapsed,0)","femto":femto(),"xjtu":xjtu(),"milling":milling()}
    (OUT/"results.json").write_text(json.dumps(result,indent=2)+"\n")
if __name__=="__main__": main()
