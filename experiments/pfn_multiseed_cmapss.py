#!/usr/bin/env python3
"""Local TabPFN five-seed audit on the frozen C-MAPSS strict OP-hull split."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np

os.environ.setdefault("TABPFN_FORCE_LOCAL","1")
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET","1")
LEGACY=Path(__file__).resolve().parents[2]/"ca-css-ncmapss";sys.path.insert(0,str(LEGACY))
from baselines import TabPFNPredictor
from cmapss_prior_off import RUL_CAP
from run_pae_machine_strict_hull import FDS,aggregate
from run_pae_machine_strict_hull_v2 import prepare,robust_transform

def main():
    train,validation,test,audit=prepare();train,validation,test,transform=robust_transform(train,validation,test)
    rows=[]
    seeds=tuple(int(s) for s in os.environ.get("PFN_SEEDS","42,43,44,45,46").split(","))
    for seed in seeds:
        model=TabPFNPredictor(max_train=3000,random_state=seed,ignore_pretraining_limits=True,model_version="v3")
        model.fit(train["x"],train["y"])
        pred={fd:np.clip(model.predict(test[fd]["x"]),0,RUL_CAP) for fd in FDS}
        metrics=aggregate(test,pred);rows.append({"seed":seed,"metrics":metrics,"backend":model._backend})
        print(seed,metrics["pooled"]["r2"],flush=True)
    v=np.array([r["metrics"]["pooled"]["r2"] for r in rows])
    payload={"protocol":"same robust 115 features, same 2616 test rows, max_train=3000",
             "seeds":list(seeds),
             "pooled_r2_mean":float(v.mean()),"pooled_r2_sd":float(v.std()),"runs":rows,
             "split_audit":audit,"transform":transform}
    out=Path("results/pfn_multiseed_fair");out.mkdir(parents=True,exist_ok=True)
    (out/"cmapss.json").write_text(json.dumps(payload,indent=2)+"\n")
if __name__=="__main__":main()
