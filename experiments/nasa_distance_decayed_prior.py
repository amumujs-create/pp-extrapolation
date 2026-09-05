"""Evaluate one predeclared distance-decayed boundary transport prior."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from pp_extrapolation import TransportTriples, fit_pp, predict, regression_metrics
from pp_extrapolation.priors import prepare_transport
from nasa_prior_reliability_gate import load_folds


SEEDS = (42, 43, 44, 45, 46)
WEIGHT = 10.0
TOLERANCE_CYCLES = 2.0
TAU_FRACTION_OF_TRAIN_RANGE = 0.10


def decayed_transport(train):
    low = float(train["x"].min())
    high = float(train["x"].max())
    maximum_step = min(low, high - low)
    step = np.linspace(0.02, maximum_step, 64)[:, None]
    boundary = np.full_like(step, low)
    tau = max(TAU_FRACTION_OF_TRAIN_RANGE * (high - low), 1e-6)
    confidence = np.exp(-step[:, 0] / tau)
    return TransportTriples(
        inner=boundary + step,
        boundary=boundary,
        outer=boundary - step,
        ratio=np.ones(len(step)),
        tolerance=np.full(len(step), TOLERANCE_CYCLES),
        confidence=confidence,
    ), tau


def diagnostics(fit, prior):
    inner, boundary, outer, ratio, tolerance, confidence = prepare_transport(
        prior, fit.center, fit.scale, fit.target_scale
    )
    with torch.no_grad():
        error = torch.abs(
            (fit.model(outer) - fit.model(boundary))
            - ratio * (fit.model(boundary) - fit.model(inner))
        )
        violation = torch.relu(error - tolerance)
    return {
        "mean_confidence": float(confidence.mean()),
        "active_fraction": float((violation > 1e-7).float().mean()),
        "unweighted_error_cycles": float(error.mean() * fit.target_scale),
        "weighted_violation_squared": float(
            (confidence * violation.square()).mean()
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", required=True)
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    protocol = {
        "status": "post_hoc_method_development_on_reused_NASA_health_v2",
        "hypothesis": "decay counterfactual trust with distance beyond the train boundary",
        "seeds": list(SEEDS),
        "weight": WEIGHT,
        "tolerance_cycles": TOLERANCE_CYCLES,
        "tau_fraction_of_train_range": TAU_FRACTION_OF_TRAIN_RANGE,
        "test_based_hyperparameter_selection": False,
        "fold_archive_sha256": hashlib.sha256(Path(args.folds).read_bytes()).hexdigest(),
        "baseline_prediction_sha256": hashlib.sha256(
            Path(args.baseline_predictions).read_bytes()
        ).hexdigest(),
    }
    (output / "PROTOCOL.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    folds = load_folds(args.folds)
    truth = np.concatenate([fold["test"]["y"] for fold in folds])
    groups = np.concatenate([fold["test"]["groups"] for fold in folds])
    saved = {"truth": truth, "groups": groups}
    runs = []
    for seed in SEEDS:
        parts, details = [], []
        for index, fold in enumerate(folds):
            prior, tau = decayed_transport(fold["train"])
            fit = fit_pp(
                fold["train"],
                fold["validation"],
                seed=seed,
                transport_triples=prior,
                transport_weight=WEIGHT,
            )
            parts.append(predict(fit, fold["test"]["x"]))
            details.append(
                {
                    "fold": index,
                    "tau": tau,
                    "selection": fit.selection,
                    "diagnostics": diagnostics(fit, prior),
                }
            )
        prediction = np.concatenate(parts)
        metrics = regression_metrics(truth, prediction, groups)
        runs.append({"seed": seed, "metrics": metrics, "folds": details})
        saved[f"prediction_seed{seed}"] = prediction
        print(seed, metrics["pooled"]["r2"], flush=True)
    baseline = np.load(args.baseline_predictions, allow_pickle=False)
    baseline_scores = [
        regression_metrics(truth, baseline[f"PP_{seed}"], groups)["pooled"]["r2"]
        for seed in SEEDS
    ]
    scores = [run["metrics"]["pooled"]["r2"] for run in runs]
    payload = {
        "protocol": protocol,
        "baseline_pp": {
            "pooled_r2_mean": float(np.mean(baseline_scores)),
            "pooled_r2_sd": float(np.std(baseline_scores)),
        },
        "distance_decayed_transport": {
            "pooled_r2_mean": float(np.mean(scores)),
            "pooled_r2_sd": float(np.std(scores)),
            "delta_vs_pp": float(np.mean(scores) - np.mean(baseline_scores)),
        },
        "runs": runs,
    }
    (output / "results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(output / "predictions.npz", **saved)
    print(payload["distance_decayed_transport"])


if __name__ == "__main__":
    main()
