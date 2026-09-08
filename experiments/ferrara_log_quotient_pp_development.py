#!/usr/bin/env python3
"""Post-lock Ferrara development: scale-stable log-quotient PP.

The original E5/E6 one-shot result is immutable.  This script is explicitly a
posthoc development analysis prompted by that failure.  It changes the task to
the standard unseen-unit RUL setting: complete run-to-failure trajectories are
available for development bearings, while only causal history is used for the
held-out bearing tails.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from ferrara_bearing_external_eval import LOADS, SEEDS, TEST, TRAIN, VALIDATION, causal_unit, rows
from pp_extrapolation import (
    fit_log_boundary_quotient_pp,
    predict_log_boundary_affine,
    predict_log_boundary_quotient,
    regression_metrics,
)

OUT = ROOT / "results" / "ferrara_log_quotient_pp_posthoc_v1"


def complete_rows(cells: dict, units: tuple[str, ...]) -> dict:
    prefix, tail = rows(cells, units, "prefix"), rows(cells, units, "tail")
    return {key: np.concatenate([prefix[key], tail[key]]) for key in prefix}


def portable_features(split: dict) -> dict:
    # Published load is deliberately excluded.  Four development units exhibit
    # a reversed load-life association, so treating load as a transportable
    # scale coordinate produced the catastrophic frozen PP extrapolation.
    # Speed is constant and is excluded with it.  Elapsed time remains causal.
    result = dict(split)
    result["x"] = np.asarray(split["x"])[:, :-2]
    return result


def causal_regime_score(cells: dict, units: tuple[str, ...]) -> np.ndarray:
    """Robust causal vibration-onset score aligned with each unit's tail."""
    scores = []
    for unit in units:
        wave = np.load(ROOT / "data" / "ferrara_bearing" / "features" / f"{unit}.npz")["values"]
        wave = wave[:cells[unit]["event"] + 1, [1, 7, 8, 9]]  # RMS and three high bands
        smooth = np.empty_like(wave)
        for column in range(wave.shape[1]):
            cumulative = np.r_[0.0, np.cumsum(wave[:, column])]
            smooth[:, column] = [
                (cumulative[i + 1] - cumulative[max(0, i - 23)]) / min(24, i + 1)
                for i in range(len(wave))
            ]
        baseline = smooth[:min(100, len(smooth))]
        center = np.median(baseline, axis=0)
        scale = 1.4826 * np.median(np.abs(baseline - center), axis=0) + 1e-6
        score = np.max((smooth - center) / scale, axis=1)
        scores.append(score[int(np.floor(0.70 * len(score))):])
    return np.concatenate(scores)


def main() -> None:
    torch.set_num_threads(2)
    cells = {unit: causal_unit(unit) for unit in LOADS}

    # Hyperparameters use E1-E3 only: early prefixes fit, their late shells
    # select alpha and stopping time.  E4 is added only during the final refit.
    selection_train = portable_features(rows(cells, TRAIN, "prefix"))
    selection_validation = portable_features(rows(cells, TRAIN, "tail"))
    refit = portable_features(complete_rows(cells, TRAIN + VALIDATION))
    test = portable_features(rows(cells, TEST, "tail"))

    pp_predictions, affine_predictions, runs = [], [], []
    for seed in SEEDS:
        selected = fit_log_boundary_quotient_pp(
            selection_train, selection_validation, seed=seed,
            max_epochs=300, patience=50,
        )
        alpha = float(selected.latent_fit.selection["affine_alpha"])
        epoch = max(int(selected.latent_fit.selection["selected_epoch"]), 1)
        final = fit_log_boundary_quotient_pp(
            refit, refit, seed=seed, selected_alpha=alpha,
            max_epochs=epoch, patience=epoch + 1,
        )
        pp = predict_log_boundary_quotient(final, test)
        affine = predict_log_boundary_affine(final, test)
        pp_predictions.append(pp)
        affine_predictions.append(affine)
        runs.append({
            "seed": int(seed), "selected_alpha": alpha, "selected_epoch": epoch,
            "pp": regression_metrics(test["y"], pp, test["groups"]),
            "affine": regression_metrics(test["y"], affine, test["groups"]),
        })
        print(seed, alpha, epoch, runs[-1]["pp"]["pooled"]["r2"], flush=True)

    pp = np.asarray(pp_predictions)
    affine = np.asarray(affine_predictions)
    locked = json.loads((ROOT / "results" / "ferrara_bearing_external_locked_v1" / "results.json").read_text())
    locked_prediction = np.load(
        ROOT / "results" / "ferrara_bearing_external_locked_v1" / "predictions.npz",
        allow_pickle=True,
    )
    boundary_rate = locked_prediction["boundary_rate"]
    regime_score = causal_regime_score(cells, TEST)
    # Five robust standard deviations is a conventional high-specificity onset
    # threshold.  The sharp sigmoid prevents normal baseline fluctuation from
    # routing observations away from the scale-stable quotient path.
    regime_gate = 1.0 / (1.0 + np.exp(np.clip(-(regime_score - 5.0) / 0.5, -50, 50)))
    regime_pp = (1.0 - regime_gate[None, :]) * pp + regime_gate[None, :] * boundary_rate[None, :]
    result = {
        "status": "posthoc development after review of the locked E5/E6 result",
        "confirmation_claim_allowed": False,
        "reason": "E5 and E6 labels had already been inspected before this structure was developed",
        "revised_estimand": "unseen-bearing late-tail RUL with complete development run-to-failure trajectories",
        "split": {"selection_train": TRAIN, "selection_validation": TRAIN,
                  "refit": TRAIN + VALIDATION, "posthoc_test": TEST},
        "selection_shell": {"train_fraction": [0.0, 0.7], "validation_fraction": [0.7, 1.0]},
        "features": "causal vibration/history plus elapsed time; published load and constant speed excluded",
        "model": "regime gate chooses between scale-stable log-quotient and local boundary-rate PP paths",
        "equation": "(1-g) m expm1(a(x)+r_NN(x)) + g m/max(local_rate, floor)",
        "regime_gate": {"inputs": "causal RMS and high-band robust anomaly",
                        "threshold_mad": 5.0, "temperature": 0.5,
                        "mean_by_test_unit": {
                            unit: float(np.mean(regime_gate[test["groups"] == unit])) for unit in TEST
                        }},
        "runs": runs,
        "ensemble": {
            "regime_log_quotient_pp": regression_metrics(test["y"], regime_pp.mean(0), test["groups"]),
            "log_quotient_pp": regression_metrics(test["y"], pp.mean(0), test["groups"]),
            "log_quotient_affine": regression_metrics(test["y"], affine.mean(0), test["groups"]),
        },
        "locked_comparators": locked["ensemble"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(OUT / "predictions.npz", y=test["y"], groups=test["groups"],
                        pp=pp, affine=affine, regime_pp=regime_pp,
                        boundary_rate=boundary_rate, regime_score=regime_score,
                        regime_gate=regime_gate)
    print(json.dumps(result["ensemble"], indent=2), flush=True)


if __name__ == "__main__":
    main()
