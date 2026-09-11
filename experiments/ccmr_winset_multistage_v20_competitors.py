#!/usr/bin/env python3
"""Score CCMR v2.0 against sealed competitors on the MultiStage win-set.

Loads already-aligned artifacts only. Does not retune models or open Stage-2.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from pp_extrapolation.metrics import regression_metrics
from pp_extrapolation.stability_first import raw_unit_regret, regret_summary

OUT = ROOT / "results/ccmr_winset_multistage_v20_competitors"
V20 = ROOT / "results/ccmr_v20_frozen_holdout_replay/predictions.npz"
COMP = ROOT / (
    "results/two_success_cohorts_extrapolation_competitors_v2_nonnegative/"
    "multistage_rpt_ensemble_predictions.npz"
)
MODELS = (
    "CCMR_v2.0",
    "PP_latest_successful",
    "Persistence",
    "V-REx",
    "GroupDRO",
    "Monotone_NN",
    "Linear_tail_RBF",
    "Engression",
    "Linear_mean_GP",
    "TabPFN_v3",
)


def score(y, groups, anchor, prediction):
    prediction = np.asarray(prediction, dtype=np.float64)
    model = regression_metrics(y, prediction, groups)
    persistence = regression_metrics(y, anchor, groups)
    model_macro = float(np.mean([
        value["rmse"] for value in model["per_unit"].values()
    ]))
    persistence_macro = float(np.mean([
        value["rmse"] for value in persistence["per_unit"].values()
    ]))
    regret = regret_summary(raw_unit_regret(y, groups, anchor, prediction))
    return {
        "pooled_r2": model["pooled"]["r2"],
        "pooled_rmse": model["pooled"]["rmse"],
        "pooled_mae": model["pooled"]["mae"],
        "macro_rmse": model_macro,
        "pooled_improvement": float(
            (persistence["pooled"]["rmse"] - model["pooled"]["rmse"])
            / persistence["pooled"]["rmse"]
        ),
        "macro_improvement": float(
            (persistence_macro - model_macro) / persistence_macro
        ),
        "raw_regret_mean": regret[0],
        "raw_regret_cvar20": regret[1],
        "raw_regret_max": regret[2],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "results.json"
    if result_path.exists():
        raise RuntimeError("refusing to overwrite win-set competitor scores")
    v20 = np.load(V20)
    comp = np.load(COMP)
    truth = v20["MultiStage_RPT_truth"]
    groups = v20["MultiStage_RPT_groups"].astype(str)
    anchor = v20["MultiStage_RPT_persistence"]
    if not np.allclose(truth, comp["truth"]):
        raise RuntimeError("truth alignment failure")
    if not np.allclose(anchor, comp["persistence"]):
        raise RuntimeError("persistence alignment failure")
    predictions = {
        "CCMR_v2.0": v20["MultiStage_RPT_CCMR_v20"],
        "Persistence": anchor,
        "PP_latest_successful": comp["PP_latest_successful"],
        "V-REx": comp["V-REx"],
        "GroupDRO": comp["GroupDRO"],
        "Monotone_NN": comp["Monotone_NN"],
        "Linear_tail_RBF": comp["Linear_tail_RBF"],
        "Engression": comp["Engression"],
        "Linear_mean_GP": comp["Linear_mean_GP"],
        "TabPFN_v3": comp["TabPFN_v3"],
    }
    scores = {
        name: score(truth, groups, anchor, predictions[name])
        for name in MODELS
    }
    ranked = sorted(
        scores.items(),
        key=lambda item: (
            -item[1]["pooled_r2"],
            item[1]["raw_regret_max"],
            item[1]["pooled_rmse"],
        ),
    )
    winner = ranked[0][0]
    engression = scores["Engression"]
    ccmr = scores["CCMR_v2.0"]
    beats_engression = bool(
        ccmr["pooled_rmse"] < engression["pooled_rmse"]
        and ccmr["raw_regret_max"] <= engression["raw_regret_max"]
        and ccmr["macro_improvement"] >= engression["macro_improvement"]
    )
    payload = {
        "status": "retrospective MultiStage win-set scored",
        "confirmatory": False,
        "cohort": "MultiStage_RPT_Stage1",
        "n_test_origins": int(len(truth)),
        "n_test_units": int(len(np.unique(groups))),
        "scores": scores,
        "rank_by_pooled_r2_then_max_regret": [
            {"model": name, **metrics} for name, metrics in ranked
        ],
        "accuracy_leader": winner,
        "ccmr_v20_beats_engression_on_rmse_and_regret": beats_engression,
        "claim_template": (
            "MultiStage형 unit-disjoint 시간 외삽에서 CCMR v2.0은 "
            "Engression 대비 평균 RMSE를 개선하고 max unit regret 0%를 "
            "유지했다."
            if beats_engression else
            "이 산출물만으로는 Engression 우위 문장을 쓰지 않는다."
        ),
        "artifacts": {
            "ccmr_v20": str(V20.relative_to(ROOT)),
            "competitors": str(COMP.relative_to(ROOT)),
        },
    }
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "accuracy_leader": winner,
        "beats_engression": beats_engression,
        "CCMR_v2.0": scores["CCMR_v2.0"],
        "Engression": scores["Engression"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
