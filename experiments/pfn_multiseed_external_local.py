#!/usr/bin/env python3
"""Downloaded local TabPFN v3 on the frozen external PP splits."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np

os.environ["TABPFN_FORCE_LOCAL"]="1"
LEGACY=Path(__file__).resolve().parents[2]/"ca-css-ncmapss";sys.path.insert(0,str(LEGACY))
from baselines import TabPFNPredictor
from run_pp_vs_tabpfn_external_three import MAX_TRAIN,equal_unit_subsample
from run_affine_tail_external_three import aggregate_metrics,prepare_hust,prepare_virkler
from run_affine_tail_external_nasa_health_v2 import prepare_folds
SEEDS=(42,43,44,45,46)

def predict(train,test_x,seed):
    model=TabPFNPredictor(max_train=MAX_TRAIN,random_state=seed,
                          ignore_pretraining_limits=True,model_version="v3")
    model.fit(train["x"],train["y"])
    return np.clip(model.predict(test_x,batch_size=512),0,max(float(np.max(train["y"])),1.0)),model._backend

def summarize(rows):
    v=np.array([r["metrics"]["pooled"]["r2"] for r in rows])
    return {"pooled_r2_mean":float(v.mean()),"pooled_r2_sd":float(v.std()),"runs":rows}

def single(prepared,name):
    train,_,test,_=prepared;train,sampling=equal_unit_subsample(train,MAX_TRAIN);rows=[]
    for seed in SEEDS:
        pred,backend=predict(train,test["x"],seed);m=aggregate_metrics(test["y"],pred,test["groups"])
        rows.append({"seed":seed,"backend":backend,"metrics":m});print(name,seed,m["pooled"]["r2"],flush=True)
    return {"sampling":sampling,**summarize(rows)}

def nasa(folds):
    truth=np.concatenate([f["test"]["y"] for f in folds]);groups=np.concatenate([f["test"]["groups"] for f in folds]);rows=[]
    for seed in SEEDS:
        parts=[];backends=[]
        for f in folds:
            train,_=equal_unit_subsample(f["train"],MAX_TRAIN);p,b=predict(train,f["test"]["x"],seed);parts.append(p);backends.append(b)
        m=aggregate_metrics(truth,np.concatenate(parts),groups);rows.append({"seed":seed,"backend":backends,"metrics":m});print("nasa",seed,m["pooled"]["r2"],flush=True)
    return summarize(rows)

def main():
    folds,_=prepare_folds();payload={"model":"downloaded local TabPFN v3","max_train":MAX_TRAIN,"seeds":list(SEEDS),
      "hust":single(prepare_hust(),"hust"),"virkler":single(prepare_virkler(),"virkler"),"nasa":nasa(folds)}
    out=Path("results/pfn_multiseed_fair");out.mkdir(parents=True,exist_ok=True);(out/"external_local.json").write_text(json.dumps(payload,indent=2)+"\n")
if __name__=="__main__":main()
