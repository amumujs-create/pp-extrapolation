#!/usr/bin/env python3
"""Development protocol using two observed conditions before condition-3 extrapolation."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/"src"),str(ROOT/"experiments")]
from pp_extrapolation import regression_metrics
from pp_extrapolation.temporal_residual_pp import fit_temporal_residual_pp,predict_temporal_residual_pp
from xjtu_spectrum_adapter_v2 import load
from xjtu_uncapped_temporal_pp_v2 import rows
OUT=ROOT/"results/xjtu_two_condition_refit_v2";SEEDS=(42,43,44,45,46)

def combine(*parts):
 return {key:np.concatenate([part[key] for part in parts]) for key in parts[0]}
def select_units(part,excluded):
 mask=~np.isin(part["groups"],list(excluded));return {key:value[mask] for key,value in part.items()}
def only_units(part,included):
 mask=np.isin(part["groups"],list(included));return {key:value[mask] for key,value in part.items()}

def main():
 import torch;torch.set_num_threads(2);raw=load();results={}
 # One representative unit from each observed condition is held out. This is a
 # separate post-test development protocol, not the original untouched split.
 held={"Bearing1_3","Bearing2_2"}
 for representation in ("summary","spectrum"):
  c1=rows(raw,"35Hz12kN",representation);c2=rows(raw,"37.5Hz11kN",representation);development=combine(c1,c2)
  train=select_units(development,held);validation=only_units(development,held);test=rows(raw,"40Hz10kN",representation)
  for mode in ("direct","pp"):
   predictions=[];runs=[]
   for seed in SEEDS:
    fit=fit_temporal_residual_pp(train,validation,seed=seed,mode=mode,width=16,learning_rate=5e-4,weight_decay=.05,residual_bound=.5,support_decay=0,max_epochs=260,patience=50)
    prediction=predict_temporal_residual_pp(fit,test);predictions.append(prediction);runs.append({"seed":seed,"selection":fit.selection,"metrics":regression_metrics(test["y"],prediction,test["groups"])});print(representation,mode,seed,runs[-1]["metrics"]["pooled"]["r2"],flush=True)
   predictions=np.asarray(predictions);key=f"{representation}_{mode}";results[key]={"heldout_development_units":sorted(held),"runs":runs,"ensemble":regression_metrics(test["y"],predictions.mean(0),test["groups"]),"seed_mean":float(np.mean([r["metrics"]["pooled"]["r2"] for r in runs])),"seed_sd":float(np.std([r["metrics"]["pooled"]["r2"] for r in runs],ddof=1))};OUT.mkdir(parents=True,exist_ok=True);np.savez_compressed(OUT/f"{key}.npz",y=test["y"],groups=test["groups"],predictions=predictions)
 (OUT/"results.json").write_text(json.dumps({"status":"post-test two-condition development protocol","results":results},indent=2)+"\n");print(json.dumps({k:v["ensemble"]["pooled"] for k,v in results.items()},indent=2))
if __name__=="__main__":main()
