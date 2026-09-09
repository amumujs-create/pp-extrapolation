#!/usr/bin/env python3
"""Train/validation-only model selection on the corrected FEMTO adapter."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"experiments")]
from femto_sensor_adapter_v2 import load, OFFICIAL
from plain_mlp_ablation import fit_plain, predict_plain
from pp_extrapolation import fit_pp, predict, select_affine_initialization

OUT=ROOT/"results/femto_corrected_v2"
TRAIN={11,12,21,22,31}; VAL={32}; TEST={13,14,15,16,17,23,24,25,26,27,33}

def causal_features(d):
    out=np.zeros((len(d["y"]), 2+14*5), np.float32)
    for bearing in np.unique(d["bearing"]):
        ix=np.flatnonzero(d["bearing"]==bearing); ix=ix[np.argsort(d["recording_index"][ix])]
        z=d["sensor"][ix].astype(float); t=d["elapsed_s"][ix].astype(float)
        for j,k in enumerate(ix):
            base=np.median(z[:min(10,j+1)],axis=0); scale=np.maximum(np.abs(base),1e-5)
            def slope(w):
                q=max(0,j-w+1); dt=max(t[j]-t[q],10.); return (z[j]-z[q])/dt*1000/scale
            recent=z[max(0,j-15):j+1]
            out[k]=np.r_[d["condition"][k],np.log1p(t[j])/10,z[j],(z[j]-base)/scale,
                         slope(8),slope(32),np.std(recent,axis=0)/scale]
    return out

def rows(d,x,units,endpoint=False):
    ix=np.flatnonzero(np.isin(d["unit"],list(units)))
    if endpoint:
        ix=np.asarray([g[np.argmax(d["recording_index"][g])] for u in sorted(units)
                       if (g:=ix[d["unit"][ix]==u]).size])
    return {"x":x[ix],"y":d["y"][ix],"groups":d["bearing"][ix]},ix

def metrics(y,p):
    return {"r2":float(r2_score(y,p)),"rmse":float(mean_squared_error(y,p)**.5)}

def main():
    d=load(); x=causal_features(d)
    tr,_=rows(d,x,TRAIN); va,_=rows(d,x,VAL); te,ti=rows(d,x,TEST,True)
    # Guard the label protocol before any model fitting.
    assert all(abs(te["y"][i]-OFFICIAL[b])<1e-6 for i,b in enumerate(te["groups"]))
    candidates=[]
    for width in (8,16,32):
      for lr in (3e-4,1e-3):
       for wd in (.01,.1,1.):
        for kind in ("mlp","pp"):
         preds=[]
         for seed in (42,43,44):
          if kind=="mlp":
           f=fit_plain(tr,va,seed=seed,width=width,learning_rate=lr,weight_decay=wd)
           preds.append(predict_plain(f,va["x"]))
          else:
           a=select_affine_initialization(tr,va)
           f=fit_pp(tr,va,seed=seed,width=width,learning_rate=lr,weight_decay=wd,
                    affine_selection=a,direct_residual_mixture=True,fixed_affine_trust=0.)
           preds.append(predict(f,va["x"]))
         candidates.append({"kind":kind,"width":width,"lr":lr,"wd":wd,
                            "val_rmse":float(mean_squared_error(va["y"],np.mean(preds,0))**.5)})
    selected={k:min((r for r in candidates if r["kind"]==k),key=lambda r:r["val_rmse"]) for k in ("mlp","pp")}
    OUT.mkdir(parents=True,exist_ok=True)
    # Freeze selection before test labels enter evaluation.
    (OUT/"selection_manifest.json").write_text(json.dumps({"split":{"train":sorted(TRAIN),"val":sorted(VAL),"test":sorted(TEST)},"selected":selected,"grid":candidates},indent=2)+"\n")
    results={}
    for kind,c in selected.items():
      pred=[]
      for seed in (42,43,44,45,46):
       if kind=="mlp":
        f=fit_plain(tr,va,seed=seed,width=c["width"],learning_rate=c["lr"],weight_decay=c["wd"]); pred.append(predict_plain(f,te["x"]))
       else:
        f=fit_pp(tr,va,seed=seed,width=c["width"],learning_rate=c["lr"],weight_decay=c["wd"],affine_selection=select_affine_initialization(tr,va),direct_residual_mixture=True,fixed_affine_trust=0.); pred.append(predict(f,te["x"]))
      pred=np.asarray(pred); results[kind]={"ensemble":metrics(te["y"],pred.mean(0)),"seed_r2":[metrics(te["y"],p)["r2"] for p in pred]}
      np.savez_compressed(OUT/f"{kind}_predictions.npz",y=te["y"],groups=te["groups"],predictions=pred)
    for alpha in (.01,.1,1,10,100):
      m=Ridge(alpha=alpha).fit(tr["x"],tr["y"]); score=mean_squared_error(va["y"],m.predict(va["x"]))
      if 'best' not in locals() or score<best[0]: best=(score,alpha,m)
    results["ridge"]={"alpha":best[1],"test":metrics(te["y"],best[2].predict(te["x"]))}
    (OUT/"results.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(results,indent=2))
if __name__=="__main__": main()
