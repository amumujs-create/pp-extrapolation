#!/usr/bin/env python3
"""Leave-one-domain-out audit of the residual-gain approval threshold."""
import json
from pathlib import Path

import numpy as np

THRESHOLDS=(0.0,0.02,0.05,0.10,0.15,0.20,0.25)

def pooled(row,key):
    value=row[key]["ensemble"]
    return float(value.get("pooled",value)["r2"])

def validation_gain(choice):
    if isinstance(choice,list):
        values=[]
        for fold in choice:
            rows=fold["global_candidates"]
            base=next(x["validation_mse"] for x in rows if x["gain"]==1.0)
            values.append((base-min(x["validation_mse"] for x in rows))/base)
        return max(values)
    rows=choice["global_candidates"]
    base=next(x["validation_mse"] for x in rows if x["gain"]==1.0)
    return (base-min(x["validation_mse"] for x in rows))/base

def main():
    source=json.load(open("results/residual_gain_all_extended_v1/results.json"))["datasets"]
    domains=[]
    for name,row in source.items():
        domains.append({"name":name,"validation_gain":validation_gain(row["gain_choices"]),
                        "test_delta":pooled(row,"gain_pp")-pooled(row,"pp")})
    heldout=[]
    for target in domains:
        train=[row for row in domains if row["name"]!=target["name"]]
        candidates=[]
        for threshold in THRESHOLDS:
            deltas=[row["test_delta"] if row["validation_gain"]>=threshold else 0.0 for row in train]
            candidates.append({"threshold":threshold,"harmed_domains":sum(x<0 for x in deltas),
                               "mean_delta":float(np.mean(deltas))})
        # Safety first, then mean gain, then the more conservative threshold.
        selected=min(candidates,key=lambda x:(x["harmed_domains"],-x["mean_delta"],-x["threshold"]))
        applied=target["validation_gain"]>=selected["threshold"]
        heldout.append({**target,"selected_threshold":selected["threshold"],"applied":applied,
                        "heldout_delta":target["test_delta"] if applied else 0.0,
                        "training_candidates":candidates})
    payload={"status":"post-hoc LODO threshold audit","selection":"minimize harmed training domains, maximize mean delta, prefer conservative tie",
             "thresholds":list(THRESHOLDS),"heldout":heldout,
             "summary":{"improved":sum(x["heldout_delta"]>0 for x in heldout),
                        "unchanged":sum(x["heldout_delta"]==0 for x in heldout),
                        "degraded":sum(x["heldout_delta"]<0 for x in heldout),
                        "mean_delta":float(np.mean([x["heldout_delta"] for x in heldout]))}}
    out=Path("results/residual_gain_lodo_v1");out.mkdir(parents=True,exist_ok=True);(out/"results.json").write_text(json.dumps(payload,indent=2))
    print(json.dumps(payload["summary"],indent=2));
    for row in heldout:print(row["name"],row["selected_threshold"],row["heldout_delta"])
if __name__=="__main__":main()
