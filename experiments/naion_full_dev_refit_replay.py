#!/usr/bin/env python3
"""Matched PP/MLP full-development refit replay on the observed Na-ion cohort."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"experiments")]
import naion_80eol_prospective_gate as data
from naion_boundary_quotient_pp import quotient_rows
from plain_mlp_ablation import fit_plain,predict_plain
from pp_extrapolation import fit_boundary_quotient_pp,predict_boundary_quotient,regression_metrics

OUT=ROOT/"results"/"naion_full_dev_refit_replay_v1";SEEDS=(42,43,44,45,46)

def main():
 torch.set_num_threads(2);train,val=data.load_development();dev={**train,**val};raw={n:data.read_file(ROOT/"data/naion_80eol_test"/n) for n in data.TEST_NAMES};test={u:data.eol_truncate(z) for u,z in raw.items()};test={u:z for u,z in test.items() if z is not None};life=[z["cycle"][-1]-z["cycle"][0] for z in train.values()];boundary=float(np.floor(.6*np.median(life)));scale=float(max(life));tr=quotient_rows(train,sorted(train),boundary,scale,"prefix");va=quotient_rows(val,sorted(val),boundary,scale,"tail");full=quotient_rows(dev,sorted(dev),boundary,scale,"prefix");te=quotient_rows(test,sorted(test),boundary,scale,"tail")
 pp=[];plain=[];runs=[]
 for seed in SEEDS:
  ps=fit_boundary_quotient_pp(tr,va,seed=seed,residual_bound=.5,max_epochs=300,patience=50);pe=max(ps.selection["selected_epoch"],1);pf=fit_boundary_quotient_pp(full,full,seed=seed,residual_bound=.5,max_epochs=pe,patience=10000,restore_best=False);yp=predict_boundary_quotient(pf,te)
  ns=fit_plain(tr,va,seed=seed,max_epochs=300,patience=50);ne=max(ns["selected_epoch"],1);nf=fit_plain(full,full,seed=seed,max_epochs=ne,patience=10000,restore_best=False);yn=predict_plain(nf,te["x"],clip_to_train_max=False)
  pp.append(yp);plain.append(yn);runs.append({"seed":seed,"pp_epoch":pe,"plain_epoch":ne,"pp":regression_metrics(te["y"],yp,te["groups"]),"plain":regression_metrics(te["y"],yn,te["groups"])});print(seed,flush=True)
 pp=np.asarray(pp);plain=np.asarray(plain);result={"status":"retrospective matched full-development refit; test previously observed","boundary":boundary,"seeds":SEEDS,"pp":regression_metrics(te["y"],pp.mean(0),te["groups"]),"plain_mlp":regression_metrics(te["y"],plain.mean(0),te["groups"]),"runs":runs,"reference_no_refit":{"bq_pp_pooled_r2":.78085,"plain_mlp_pooled_r2":.46980484376937115}}
 OUT.mkdir(parents=True,exist_ok=True);(OUT/"results.json").write_text(json.dumps(result,indent=2)+"\n");np.savez_compressed(OUT/"predictions.npz",y=te["y"],groups=te["groups"],pp=pp,plain=plain);print(json.dumps({"pp":result["pp"],"plain":result["plain_mlp"]},indent=2))

if __name__=="__main__":main()
