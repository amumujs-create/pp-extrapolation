#!/usr/bin/env python3
"""Post-reveal audit; never used for DS03 route or configuration selection."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pp_extrapolation.ds03_prospective import (
    _predict_direct,
    causal_features,
    load_revealed_test,
)
from pp_extrapolation.regime_mixture import predict_latent_regime

PPX = ROOT / "results/ncmapss_ds03_ppx_v1"
BASELINES = ROOT / "results/ncmapss_ds03_equal_budget_v1"


def r2(y, prediction):
    return float(1 - np.sum((y - prediction) ** 2) / np.sum((y - y.mean()) ** 2))


def main():
    selection = json.loads((PPX / "selection.json").read_text())
    reveal = json.loads((PPX / "reveal.json").read_text())
    bundle = torch.load(
        PPX / selection["model_artifact"], map_location="cpu", weights_only=False
    )
    test = load_revealed_test(ROOT / "data/N-CMAPSS_DS03-012.h5")
    route_scores = {}
    route_predictions = {}
    for route in ("direct_fallback", "basic", "multiscale"):
        rows = causal_features(test, "direct" if route == "direct_fallback" else route)
        predictions = [
            _predict_direct(fit, rows["x"])
            if route == "direct_fallback"
            else predict_latent_regime(fit, rows["x"])
            for fit in bundle["fits"][route]
        ]
        prediction = np.mean(predictions, axis=0)
        route_predictions[route] = prediction
        route_scores[route] = {
            "pooled_r2": r2(rows["y"], prediction),
            "pooled_rmse": math.sqrt(float(np.mean((rows["y"] - prediction) ** 2))),
        }
    baseline_result = json.loads((BASELINES / "results.json").read_text())
    baseline_scores = {
        model: values["ensemble"]["pooled"]["r2"]
        for model, values in baseline_result["models"].items()
    }
    strongest = max(baseline_scores, key=baseline_scores.get)
    z = np.load(BASELINES / strongest / "predictions.npz", allow_pickle=True)
    baseline = z["prediction"].mean(axis=0)
    direct_rows = causal_features(test, "direct")
    selected_prediction = route_predictions[selection["selected_route"]]
    ratios, logs = {}, []
    for unit in np.unique(direct_rows["groups"]):
        mask = direct_rows["groups"] == unit
        selected_rmse = math.sqrt(
            float(np.mean((direct_rows["y"][mask] - selected_prediction[mask]) ** 2))
        )
        baseline_rmse = math.sqrt(
            float(np.mean((direct_rows["y"][mask] - baseline[mask]) ** 2))
        )
        ratios[str(unit)] = selected_rmse / max(baseline_rmse, 1e-12)
        logs.append(math.log(max(baseline_rmse, 1e-12) / max(selected_rmse, 1e-12)))
    payload = {
        "status": "prospective route-selection success; predictive superiority not confirmed",
        "selection_sha256": reveal["selection_sha256"],
        "test_outcome_used_for_selection": False,
        "selected_route": selection["selected_route"],
        "selected_route_is_test_best_ppx_route": (
            selection["selected_route"]
            == max(route_scores, key=lambda key: route_scores[key]["pooled_r2"])
        ),
        "ppx_route_scores_post_reveal": route_scores,
        "equal_budget_baseline_scores": baseline_scores,
        "strongest_equal_budget_baseline": strongest,
        "strongest_equal_budget_r2": baseline_scores[strongest],
        "selected_ppx_r2": reveal["pooled_r2"],
        "pooled_r2_gap": reveal["pooled_r2"] - baseline_scores[strongest],
        "unit_rmse_ratio_ppx_over_strongest": ratios,
        "mean_unit_log_rmse_ratio": float(np.mean(logs)),
        "units_won": int(np.sum(np.asarray(logs) > 0)),
        "n_units": len(logs),
        "worst_unit_rmse_ratio": float(max(ratios.values())),
        "preregistered_criteria": {
            "information_contract": True,
            "nonnegative_pooled_r2": reveal["pooled_r2"] >= 0,
            "positive_mean_unit_log_rmse_ratio": float(np.mean(logs)) > 0,
            "no_unit_rmse_ratio_above_2": max(ratios.values()) <= 2,
        },
    }
    payload["all_predictive_success_criteria"] = all(
        payload["preregistered_criteria"].values()
    )
    path = PPX / "post_reveal_audit.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
