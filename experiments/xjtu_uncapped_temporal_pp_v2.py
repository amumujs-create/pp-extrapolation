#!/usr/bin/env python3
"""Uncapped summary/spectrum temporal PP on the locked XJTU condition split."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/"src"),str(ROOT/"experiments")]
from pp_extrapolation import regression_metrics
from pp_extrapolation.temporal_residual_pp import fit_temporal_residual_pp,predict_temporal_residual_pp
from xjtu_spectrum_adapter_v2 import load
from xjtu_untouched import CONDITIONS
OUT=ROOT/"results/xjtu_uncapped_temporal_pp_v2";SEEDS=(42,43,44,45,46);WINDOW=16
SPLIT={"train":"37.5Hz11kN","validation":"35Hz12kN","test":"40Hz10kN"}

def rows(raw,condition,representation):
 xs=[];masks=[];ys=[];groups=[];positions=[]
 for unit in np.unique(raw["units"][raw["conditions"]==condition]):
  ix=np.flatnonzero((raw["conditions"]==condition)&(raw["units"]==unit));ix=ix[np.argsort(raw["positions"][ix])]
  stat=raw["stats"][ix];spec=raw["spectrum"][ix];pos=raw["positions"][ix];life=int(raw["lives"][ix][0])
  rpm,load_value=CONDITIONS[condition]
  common=np.c_[stat,pos/500.,np.log1p(pos)/8.,
               np.full(len(pos),rpm/2400.),np.full(len(pos),load_value/12.)].astype(np.float32)
  for end in range(7,len(ix)):
   baseline=np.median(spec[:min(10,end+1)],axis=0)
   feature=common if representation=="summary" else np.c_[common,spec,spec-baseline].astype(np.float32)
   start=max(0,end-WINDOW+1);observed=feature[start:end+1];seq=np.zeros((WINDOW,feature.shape[1]),np.float32);mask=np.zeros(WINDOW,bool)
   seq[-len(observed):]=observed;mask[-len(observed):]=True;xs.append(seq);masks.append(mask);ys.append(life-int(pos[end]));groups.append(unit);positions.append(pos[end])
 return {"x":np.asarray(xs),"mask":np.asarray(masks),"y":np.asarray(ys,np.float32),"groups":np.asarray(groups),"positions":np.asarray(positions,np.float32)}

def main():
 import torch;torch.set_num_threads(2);raw=load();search=[]
 for representation in ("summary","spectrum"):
  tr=rows(raw,SPLIT["train"],representation);va=rows(raw,SPLIT["validation"],representation)
  for mode in ("direct","pp"):
   for bound in ((.25,.5) if mode=="pp" else (.5,)):
    fit=fit_temporal_residual_pp(tr,va,seed=42,mode=mode,width=16,learning_rate=5e-4,weight_decay=.05,residual_bound=bound,support_decay=0,max_epochs=180,patience=35)
    search.append({"representation":representation,"mode":mode,"bound":bound,"validation_rmse":fit.selection["validation_rmse"],"epoch":fit.selection["selected_epoch"]});print(search[-1],flush=True)
 selected={mode:min((r for r in search if r["mode"]==mode),key=lambda r:r["validation_rmse"]) for mode in ("direct","pp")}
 OUT.mkdir(parents=True,exist_ok=True);(OUT/"selection_manifest.json").write_text(json.dumps({"split":SPLIT,"selection":selected,"search":search},indent=2)+"\n")
 result={}
 for mode,choice in selected.items():
  tr=rows(raw,SPLIT["train"],choice["representation"]);va=rows(raw,SPLIT["validation"],choice["representation"]);te=rows(raw,SPLIT["test"],choice["representation"]);pred=[];runs=[]
  for seed in SEEDS:
   fit=fit_temporal_residual_pp(tr,va,seed=seed,mode=mode,width=16,learning_rate=5e-4,weight_decay=.05,residual_bound=choice["bound"],support_decay=0,max_epochs=260,patience=50)
   p=predict_temporal_residual_pp(fit,te);pred.append(p);runs.append({"seed":seed,"selection":fit.selection,"metrics":regression_metrics(te["y"],p,te["groups"])});print(mode,seed,runs[-1]["metrics"]["pooled"]["r2"],flush=True)
  pred=np.asarray(pred);result[mode]={"selected":choice,"runs":runs,"ensemble":regression_metrics(te["y"],pred.mean(0),te["groups"]),"seed_mean":float(np.mean([r["metrics"]["pooled"]["r2"] for r in runs])),"seed_sd":float(np.std([r["metrics"]["pooled"]["r2"] for r in runs],ddof=1))};np.savez_compressed(OUT/f"{mode}.npz",y=te["y"],groups=te["groups"],predictions=pred)
 (OUT/"results.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({k:{"ensemble":v["ensemble"]["pooled"],"seed_mean":v["seed_mean"],"seed_sd":v["seed_sd"]} for k,v in result.items()},indent=2))
if __name__=="__main__":main()
