#!/usr/bin/env python3
"""Retrospective PP-X residual-transport experiment on MultiStage RPT Stage 1.

The test split is already opened.  Persistence is the only structurally
aligned prior available on train, validation, and test; the archived
PP_latest_successful prediction is test-only and is used only as a comparator.
All transport parameters, including rho and the envelope radius, are fitted
from train/validation labels.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

import multistage_rpt_ccmr_v19 as multistage
from pp_extrapolation.metrics import regression_metrics
from pp_extrapolation.residual_transport import (
    ContractEnvelope,
    fit_residual_transport,
    predict_residual_mean,
)
from pp_extrapolation.stability_first import raw_unit_regret, regret_summary

COMPARATORS = ROOT / (
    "results/two_success_cohorts_extrapolation_competitors_v2_nonnegative/"
    "multistage_rpt_ensemble_predictions.npz"
)
CCMR = ROOT / "results/ccmr_v20_frozen_holdout_replay/predictions.npz"
OUT = ROOT / "results/multistage_rpt_ppx_residual_transport_v1"
SAMPLE_COUNT = 1024
SEED = 20260911


def score(y, groups, persistence, prediction):
    model = regression_metrics(y, prediction, groups)
    per_unit_rmse = [
        item["rmse"] for item in model["per_unit"].values()
    ]
    regret = regret_summary(
        raw_unit_regret(y, groups, persistence, prediction)
    )
    return {
        "pooled_r2": model["pooled"]["r2"],
        "pooled_rmse": model["pooled"]["rmse"],
        "macro_rmse": float(np.mean(per_unit_rmse)),
        "worst_unit_regret": regret[2],
        "unit_regret_mean": regret[0],
        "unit_regret_cvar20": regret[1],
    }


def envelope(rows, radius):
    return ContractEnvelope(
        rows["x"][:, 0], np.full(len(rows["y"]), radius)
    )


def repeated_prior(rows):
    return np.repeat(rows["x"][:, :1], SAMPLE_COUNT, axis=1)


def adapt_rows(rows):
    return {
        "x": np.asarray(rows["context"], dtype=float),
        "y": np.asarray(rows["y"], dtype=float),
        "groups": np.asarray(rows["groups"]),
    }


def load_frozen_cohort():
    trajectories = multistage.load_trajectories()
    split = multistage.split_units(trajectories)
    return {
        "train": adapt_rows(multistage.make_rows(
            trajectories, split["train"], (0.0, 0.30)
        )),
        "validation": adapt_rows(multistage.make_rows(
            trajectories, split["validation"], (0.40, 0.60)
        )),
        "test": adapt_rows(multistage.make_rows(
            trajectories, split["test"], (0.75, 0.90)
        )),
    }


def main():
    result_path = OUT / "results.json"
    prediction_path = OUT / "predictions.npz"
    if result_path.exists() or prediction_path.exists():
        raise RuntimeError("refusing to overwrite retrospective PP-X run")

    cohort = load_frozen_cohort()
    train, validation, test = (
        cohort["train"], cohort["validation"], cohort["test"]
    )
    comparators = np.load(COMPARATORS)
    ccmr = np.load(CCMR)
    alignments = {
        "competitor_truth": np.allclose(comparators["truth"], test["y"]),
        "competitor_groups": np.array_equal(
            comparators["groups"].astype(str),
            test["groups"].astype(str),
        ),
        "competitor_persistence": np.allclose(
            comparators["persistence"], test["x"][:, 0]
        ),
        "ccmr_truth": np.allclose(
            ccmr["MultiStage_RPT_truth"], test["y"]
        ),
        "ccmr_groups": np.array_equal(
            ccmr["MultiStage_RPT_groups"].astype(str),
            test["groups"].astype(str),
        ),
    }
    if not all(alignments.values()):
        failed = [key for key, value in alignments.items() if not value]
        raise RuntimeError(f"frozen adapter alignment failure: {failed}")

    # Train-label-only scale; validation labels remain reserved for rho.
    train_residual = train["y"] - train["x"][:, 0]
    radius = max(
        float(np.max(np.abs(train_residual))) * 1.05,
        np.finfo(float).eps,
    )
    model = fit_residual_transport(
        train["x"],
        train["y"],
        train["groups"],
        envelope(train, radius),
        validation["x"],
        validation["y"],
        validation["groups"],
        envelope(validation, radius),
        repeated_prior(validation),
    )
    prediction = predict_residual_mean(
        model,
        test["x"],
        envelope(test, radius),
        repeated_prior(test),
        seed=SEED,
    )

    candidates = {
        "PP-X_residual_transport": prediction,
        "PP_latest_successful": comparators["PP_latest_successful"],
        "Engression": comparators["Engression"],
        "CCMR_v2.0": ccmr["MultiStage_RPT_CCMR_v20"],
        "Persistence": test["x"][:, 0],
    }
    scores = {
        name: score(
            test["y"], test["groups"], test["x"][:, 0], values
        )
        for name, values in candidates.items()
    }
    reference_r2 = {
        "PP_latest_successful": 0.979397,
        "Engression": 0.979404,
        "CCMR_v2.0": 0.988847,
    }
    payload = {
        "status": "retrospective exploratory matched experiment complete",
        "confirmatory": False,
        "test_status": "previously opened",
        "cohort": "MultiStage_RPT_Stage1",
        "adapter": (
            "multistage_rpt_ccmr_v19 frozen unit split/make_rows; identical "
            "intervals to cohort_ml_benchmark.load_cohorts"
        ),
        "fit_label_policy": (
            "train labels fit residual map/radius; validation labels select "
            "rho; test labels score only"
        ),
        "prior": {
            "name": "Persistence",
            "reason": (
                "structurally aligned on train/validation/test; archived "
                "PP_latest_successful predictions are test-only"
            ),
            "sample_count": SAMPLE_COUNT,
            "envelope_radius": radius,
        },
        "rows": {key: len(value["y"]) for key, value in cohort.items()},
        "units": {
            key: int(len(np.unique(value["groups"])))
            for key, value in cohort.items()
        },
        "alignment_checks": alignments,
        "transport": {
            "rho": model.rho,
            "risk_decision": asdict(model.risk_decision),
            "seed": SEED,
        },
        "scores": scores,
        "reference_r2_rounded": reference_r2,
        "r2_delta_vs_reference": {
            name: scores["PP-X_residual_transport"]["pooled_r2"] - value
            for name, value in reference_r2.items()
        },
        "artifacts": {
            "comparators": str(COMPARATORS.relative_to(ROOT)),
            "ccmr": str(CCMR.relative_to(ROOT)),
        },
        "source_sha256": {
            "comparators": hashlib.sha256(
                COMPARATORS.read_bytes()
            ).hexdigest(),
            "ccmr": hashlib.sha256(CCMR.read_bytes()).hexdigest(),
        },
    }
    OUT.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        prediction_path,
        truth=test["y"],
        groups=test["groups"].astype(str),
        persistence=test["x"][:, 0],
        ppx_residual_transport=prediction,
        **{
            "PP_latest_successful": candidates["PP_latest_successful"],
            "Engression": candidates["Engression"],
            "CCMR_v2.0": candidates["CCMR_v2.0"],
        },
    )
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "rho": model.rho,
        "PP-X": scores["PP-X_residual_transport"],
        "r2_delta_vs_reference": payload["r2_delta_vs_reference"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
