#!/usr/bin/env python3
"""HUST support-adaptive PP with the exact 3000-row PFN training budget."""
from __future__ import annotations
import json,sys
from pathlib import Path
LEGACY=Path(__file__).resolve().parents[2]/"ca-css-ncmapss";sys.path.insert(0,str(LEGACY))
from run_affine_tail_external_three import prepare_hust
from run_pp_vs_tabpfn_external_three import equal_unit_subsample
from soft_anchor_ablation import evaluate

def main():
    train,val,test,_=prepare_hust();train,sampling=equal_unit_subsample(train,3000)
    result=evaluate(train,val,test,(42,43,44,45,46),(0,.01,.1,1,10,100),(0,.1,.3,1))
    out=Path("results/pfn_multiseed_fair");out.mkdir(parents=True,exist_ok=True)
    (out/"hust_pp_same_budget.json").write_text(json.dumps({"sampling":sampling,"result":result},indent=2)+"\n")
if __name__=="__main__":main()
