#!/usr/bin/env python3
"""Retrospective v2 grade gate for the opened MultiStage Stage-1 CRT test.

The gate uses leave-one-validation-unit-out transport predictions.  Each held
unit is excluded from rho selection as well as residual-map fitting.  Frozen
shell routes are then applied label-free to the already-opened test split.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

import multistage_rpt_ccmr_v19 as multistage
from multistage_rpt_ppx_residual_transport import score
from pp_extrapolation.extrapolation_grade_gate import (
    fit_extrapolation_grade_gate,
    predict_with_grade_gate,
    select_grade_route,
)
from pp_extrapolation.residual_transport import (
    ContractEnvelope,
    fit_residual_transport,
    predict_residual_mean,
)

V1 = ROOT / "results/multistage_rpt_ppx_residual_transport_v1/predictions.npz"
COMPARATORS = ROOT / (
    "results/two_success_cohorts_extrapolation_competitors_v2_nonnegative/"
    "multistage_rpt_ensemble_predictions.npz"
)
CCMR = ROOT / "results/ccmr_v20_frozen_holdout_replay/predictions.npz"
OUT = ROOT / "results/multistage_rpt_ppx_grade_gate_v2"
SHELL_EDGES = (0.0, 0.5, 0.75, 1.0)
OOF_SAMPLE_COUNT = 256
SEED = 20260911


def adapt(rows):
    return {
        "x": np.asarray(rows["context"], dtype=float),
        "y": np.asarray(rows["y"], dtype=float),
        "groups": np.asarray(rows["groups"]),
        "progress": np.asarray(rows["progress"], dtype=float),
    }


def load_cohort():
    trajectories = multistage.load_trajectories()
    split = multistage.split_units(trajectories)
    return {
        "train": adapt(multistage.make_rows(
            trajectories, split["train"], (0.0, 0.30)
        )),
        "validation": adapt(multistage.make_rows(
            trajectories, split["validation"], (0.40, 0.60)
        )),
        "test": adapt(multistage.make_rows(
            trajectories, split["test"], (0.75, 0.90)
        )),
    }


def take(rows, indices):
    return {key: value[indices] for key, value in rows.items()}


def envelope(rows, radius):
    return ContractEnvelope(
        rows["x"][:, 0], np.full(len(rows["y"]), radius)
    )


def prior(rows):
    return np.repeat(rows["x"][:, :1], OOF_SAMPLE_COUNT, axis=1)


def within_unit_grades(rows):
    grades = np.empty(len(rows["y"]), dtype=float)
    for unit in dict.fromkeys(rows["groups"].tolist()):
        indices = np.flatnonzero(rows["groups"] == unit)
        ranked = indices[np.argsort(rows["progress"][indices], kind="stable")]
        grades[ranked] = (np.arange(len(ranked)) + 0.5) / len(ranked)
    return grades


def main():
    result_path = OUT / "results.json"
    prediction_path = OUT / "predictions.npz"
    if result_path.exists() or prediction_path.exists():
        raise RuntimeError("refusing to overwrite MultiStage grade-gate v2")

    cohort = load_cohort()
    train, validation, test = (
        cohort["train"], cohort["validation"], cohort["test"]
    )
    v1 = np.load(V1)
    comparators = np.load(COMPARATORS)
    ccmr = np.load(CCMR)
    alignments = {
        "v1_truth": np.allclose(v1["truth"], test["y"]),
        "v1_groups": np.array_equal(
            v1["groups"].astype(str), test["groups"].astype(str)
        ),
        "comparators_truth": np.allclose(comparators["truth"], test["y"]),
        "ccmr_truth": np.allclose(
            ccmr["MultiStage_RPT_truth"], test["y"]
        ),
    }
    if not all(alignments.values()):
        raise RuntimeError(
            "artifact alignment failure: "
            + repr([key for key, value in alignments.items() if not value])
        )

    radius = max(
        float(np.max(np.abs(train["y"] - train["x"][:, 0]))) * 1.05,
        np.finfo(float).eps,
    )

    def outer_oof_predict(route, train_indices, heldout_indices):
        if route == "persistence":
            return validation["x"][heldout_indices, 0]
        outer_validation = take(validation, train_indices)
        heldout = take(validation, heldout_indices)
        model = fit_residual_transport(
            train["x"],
            train["y"],
            train["groups"],
            envelope(train, radius),
            outer_validation["x"],
            outer_validation["y"],
            outer_validation["groups"],
            envelope(outer_validation, radius),
            prior(outer_validation),
        )
        return predict_residual_mean(
            model,
            heldout["x"],
            envelope(heldout, radius),
            prior(heldout),
            seed=SEED,
        )

    gate = fit_extrapolation_grade_gate(
        validation["groups"],
        validation["progress"],
        validation["y"],
        candidate_routes=("persistence", "crt"),
        fallback_route="persistence",
        prediction_callback=outer_oof_predict,
        shell_edges=SHELL_EDGES,
        minimum_units=10,
        maximum_worst_unit_regret=0.0,
    )
    test_grades = within_unit_grades(test)
    gated = predict_with_grade_gate(
        gate,
        test_grades,
        candidate_predictions={
            "persistence": test["x"][:, 0],
            "crt": v1["ppx_residual_transport"],
        },
    )
    candidates = {
        "PP-X_grade_gate_v2": gated,
        "PP-X_CRT_v1": v1["ppx_residual_transport"],
        "PP_latest_successful": comparators["PP_latest_successful"],
        "Engression": comparators["Engression"],
        "CCMR_v2.0": ccmr["MultiStage_RPT_CCMR_v20"],
        "Persistence": test["x"][:, 0],
    }
    scores = {
        name: score(
            test["y"], test["groups"], test["x"][:, 0], prediction
        )
        for name, prediction in candidates.items()
    }
    decisions = [select_grade_route(gate, grade) for grade in test_grades]
    route_counts = {
        route: sum(decision.route == route for decision in decisions)
        for route in gate.routes
    }
    payload = {
        "status": "retrospective grade-gated CRT v2 complete",
        "confirmatory": False,
        "test_status": "previously opened",
        "cohort": "MultiStage_RPT_Stage1",
        "evidence_grade": "unit-disjoint outer OOF on validation units",
        "evidence_limitation": (
            "outer OOF is confined to the frozen validation cohort; test was "
            "already opened before this exploratory gate was designed"
        ),
        "label_policy": (
            "validation labels only construct outer-OOF shell evidence; "
            "frozen test routing uses progress-derived grades without labels"
        ),
        "shell_definition": (
            "within-unit progress order mid-rank; edges "
            + repr(SHELL_EDGES)
        ),
        "gate": {
            "selected_routes": list(gate.selected_routes),
            "minimum_units": gate.minimum_units,
            "maximum_worst_unit_regret": gate.maximum_worst_unit_regret,
            "evidence": [asdict(item) for item in gate.evidence],
            "outer_oof_episode_count": len(gate.episodes),
            "oof_sample_count": OOF_SAMPLE_COUNT,
        },
        "test_route_counts": route_counts,
        "scores": scores,
        "goal_check": {
            "pooled_rmse_improvement_retained_vs_persistence": (
                scores["PP-X_grade_gate_v2"]["pooled_rmse"]
                < scores["Persistence"]["pooled_rmse"]
            ),
            "worst_regret_reduced_vs_crt_v1": (
                scores["PP-X_grade_gate_v2"]["worst_unit_regret"]
                < scores["PP-X_CRT_v1"]["worst_unit_regret"]
            ),
        },
        "alignment_checks": alignments,
    }
    OUT.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        prediction_path,
        truth=test["y"],
        groups=test["groups"].astype(str),
        progress=test["progress"],
        grade=test_grades,
        route=np.asarray([decision.route for decision in decisions]),
        ppx_grade_gate_v2=gated,
        ppx_crt_v1=candidates["PP-X_CRT_v1"],
        persistence=candidates["Persistence"],
    )
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "selected_routes": list(gate.selected_routes),
        "test_route_counts": route_counts,
        "scores": scores,
        "goal_check": payload["goal_check"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
