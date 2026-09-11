#!/usr/bin/env python3
"""Depth-shell gated DS03 CRT retrospective experiment (immutable v2)."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from ncmapss_ds03_crt_experiment import (
    ENGRESSION,
    SELECTION,
    clipped_baseline,
    held_out_oof_anchor,
    load_engression,
    prediction_matrix,
    score,
    subset,
)
from pp_extrapolation.ds03_prospective import (
    SEEDS,
    TEST_UNITS,
    TRAIN_UNITS,
    VALIDATION_UNITS,
    causal_features,
    load_development,
    load_revealed_test,
)
from pp_extrapolation.extrapolation_grade_gate import (
    fit_extrapolation_grade_gate,
    predict_with_grade_gate,
    select_grade_route,
)
from pp_extrapolation.residual_transport import (
    ContractEnvelope,
    fit_residual_transport,
    sample_residual_transport,
)

DEFAULT_OUT = ROOT / "results/ncmapss_ds03_crt_grade_gate_v2"
SHELL_EDGES = (0.5, 0.65, 0.8, 0.9, 1.0)


def within_unit_grades(groups: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
    """Mid-rank depth grades, exactly matching gate construction."""
    grades = np.empty(len(groups), dtype=np.float64)
    for unit in dict.fromkeys(groups.tolist()):
        indices = np.flatnonzero(groups == unit)
        ranked = indices[np.argsort(coordinate[indices], kind="stable")]
        grades[ranked] = (np.arange(len(ranked), dtype=float) + 0.5) / len(ranked)
    return grades


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, default=ROOT / "data/N-CMAPSS_DS03-012.h5")
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"immutable v2 output already exists: {args.out}")
    torch.set_num_threads(2)

    selection = json.loads(args.selection.read_text())
    if selection["selected_route"] != "direct_fallback" or selection["approved"]:
        raise ValueError("v2 requires the frozen rejected-prior direct fallback")
    if tuple(selection["configuration"]["seeds"]) != SEEDS:
        raise ValueError("selection seeds are not frozen at 42--46")
    bundle = torch.load(
        args.selection.with_name(selection["model_artifact"]),
        map_location="cpu",
        weights_only=False,
    )
    frozen_fits = bundle["fits"]["direct_fallback"]

    development = causal_features(load_development(args.h5), "direct")
    train = subset(development, TRAIN_UNITS)
    validation = subset(development, VALIDATION_UNITS)
    train_matrix, oof_audit = held_out_oof_anchor(train)
    validation_matrix = prediction_matrix(validation, frozen_fits)
    train_location = train_matrix.mean(axis=0)
    validation_location = validation_matrix.mean(axis=0)
    radius = float(np.max(np.abs(train["y"] - train_location)) + 1e-9)
    train_envelope = ContractEnvelope(train_location, np.full(len(train["y"]), radius))
    validation_envelope = ContractEnvelope(
        validation_location, np.full(len(validation["y"]), radius)
    )
    validation_baseline = clipped_baseline(
        validation_matrix, validation_location, radius
    )
    model = fit_residual_transport(
        train["x"], train["y"], train["groups"], train_envelope,
        validation["x"], validation["y"], validation["groups"],
        validation_envelope, validation_baseline,
        train_disagreement=train_matrix.std(axis=0),
        validation_disagreement=validation_matrix.std(axis=0),
    )
    validation_crt = sample_residual_transport(
        model, validation["x"], validation_envelope, validation_baseline,
        disagreement=validation_matrix.std(axis=0), seed=42,
    )

    # All target-bearing gate work ends here, before the test file is opened.
    gate = fit_extrapolation_grade_gate(
        validation["groups"],
        validation["cycles"],
        validation["y"],
        candidate_routes=("direct_fallback", "crt"),
        fallback_route="direct_fallback",
        candidate_predictions={
            "direct_fallback": validation_baseline.mean(axis=1),
            "crt": validation_crt.mean(axis=1),
        },
        shell_edges=SHELL_EDGES,
        minimum_units=len(VALIDATION_UNITS),
        maximum_worst_unit_regret=0.0,
    )
    frozen_gate = {
        "shell_edges": list(gate.shell_edges),
        "selected_routes": list(gate.selected_routes),
        "evidence": [asdict(row) for row in gate.evidence],
        "minimum_units": gate.minimum_units,
        "maximum_worst_unit_regret": gate.maximum_worst_unit_regret,
    }

    test = causal_features(load_revealed_test(args.h5), "direct")
    if tuple(map(int, np.unique(test["groups"]))) != TEST_UNITS:
        raise ValueError("unexpected test units")
    test_matrix = prediction_matrix(test, frozen_fits)
    test_location = test_matrix.mean(axis=0)
    test_envelope = ContractEnvelope(test_location, np.full(len(test["y"]), radius))
    test_baseline = clipped_baseline(test_matrix, test_location, radius)
    test_crt = sample_residual_transport(
        model, test["x"], test_envelope, test_baseline,
        disagreement=test_matrix.std(axis=0), seed=42,
    )
    test_grades = within_unit_grades(test["groups"], test["cycles"])

    gated_columns = [
        predict_with_grade_gate(
            gate,
            test_grades,
            candidate_predictions={
                "direct_fallback": test_baseline[:, column],
                "crt": test_crt[:, column],
            },
        )
        for column in range(test_baseline.shape[1])
    ]
    gated_samples = np.column_stack(gated_columns)
    decisions = [select_grade_route(gate, value) for value in test_grades]
    selected_counts = {
        route: int(sum(item.route == route for item in decisions))
        for route in gate.routes
    }
    assert all(item.reason != "safe_oof_shell" or item.route == "crt" for item in decisions)
    assert np.array_equal(
        gated_samples[
            np.asarray([item.route == "direct_fallback" for item in decisions])
        ],
        test_baseline[
            np.asarray([item.route == "direct_fallback" for item in decisions])
        ],
    )

    anchor_mean = test_baseline.mean(axis=1)
    gated_score = score(
        test["y"], test["groups"], gated_samples, anchor_mean
    )
    direct_score = score(
        test["y"], test["groups"], test_baseline, anchor_mean
    )
    crt_score = score(test["y"], test["groups"], test_crt, anchor_mean)
    engression = load_engression(test)
    result = {
        "schema": "ncmapss-ds03-crt-depth-shell-gate-v2",
        "status": "retrospective validation-evidence gate; test-label-blind selection",
        "split": {
            "train_units": list(TRAIN_UNITS),
            "validation_units": list(VALIDATION_UNITS),
            "test_units": list(TEST_UNITS),
        },
        "gate_training": {
            "source": "independent validation physical units 7--9",
            "test_labels_used": False,
            "candidate_routes": ["direct_fallback", "crt"],
            "engression_candidate": False,
            "engression_exclusion_reason": "no unit-OOF prediction artifact",
            "frozen_gate": frozen_gate,
        },
        "selected_shell_routes": [
            {
                "lower_grade": lower,
                "upper_grade": upper,
                "route": route,
            }
            for lower, upper, route in zip(
                gate.shell_edges, gate.shell_edges[1:], gate.selected_routes
            )
        ],
        "selected_test_row_counts": selected_counts,
        "rho": model.rho,
        "envelope_radius_value": radius,
        "test": gated_score,
        "ungated_direct_fallback": direct_score,
        "ungated_crt": crt_score,
        "equal_budget_engression_external_only": engression,
        "comparison": {
            "gated_minus_direct_pooled_r2": (
                gated_score["pooled_r2"] - direct_score["pooled_r2"]
            ),
            "gated_minus_direct_macro_rmse": (
                gated_score["macro_rmse"] - direct_score["macro_rmse"]
            ),
            "gated_minus_engression_pooled_r2": (
                gated_score["pooled_r2"] - engression["pooled_r2"]
            ),
        },
        "invariance_checks": {
            "held_train_unit_absent_from_oof_fit": all(
                str(unit) not in fitted for unit, fitted in oof_audit.items()
            ),
            "gate_frozen_before_test_load": True,
            "fallback_rows_bitwise_exact": True,
        },
        "external_artifacts": {
            "v1_results_unchanged": "results/ncmapss_ds03_crt_retrospective_v1",
            "engression": str(ENGRESSION.relative_to(ROOT)),
        },
    }
    args.out.mkdir(parents=True)
    np.savez_compressed(
        args.out / "predictions.npz",
        y=test["y"],
        groups=test["groups"],
        grades=test_grades,
        direct_samples=test_baseline,
        crt_samples=test_crt,
        gated_samples=gated_samples,
        gated_prediction=gated_samples.mean(axis=1),
    )
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
