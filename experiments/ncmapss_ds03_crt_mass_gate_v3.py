#!/usr/bin/env python3
"""Continuous depth-shell transport-mass gate for DS03 (immutable v3)."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from ncmapss_ds03_crt_experiment import (
    ENGRESSION,
    SELECTION,
    clipped_baseline,
    held_out_oof_anchor,
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
from pp_extrapolation.residual_transport import (
    ContractEnvelope,
    fit_residual_transport,
    sample_residual_transport,
)

DEFAULT_OUT = ROOT / "results/ncmapss_ds03_crt_mass_gate_v3"
SHELL_EDGES = (0.5, 0.65, 0.8, 0.9, 1.0)
ALPHA_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0)
MINIMUM_UNITS = 3
MAXIMUM_WORST_UNIT_RELATIVE_RMSE_REGRET = 0.02


def within_unit_grades(groups: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
    """Return deterministic within-unit mid-rank depth grades."""
    groups = np.asarray(groups)
    coordinate = np.asarray(coordinate, dtype=np.float64)
    if groups.ndim != 1 or coordinate.shape != groups.shape:
        raise ValueError("groups and coordinate must be aligned vectors")
    if not np.isfinite(coordinate).all():
        raise ValueError("coordinate must be finite")
    grades = np.empty(len(groups), dtype=np.float64)
    for unit in dict.fromkeys(groups.tolist()):
        indices = np.flatnonzero(groups == unit)
        ranked = indices[np.argsort(coordinate[indices], kind="stable")]
        grades[ranked] = (np.arange(len(ranked), dtype=float) + 0.5) / len(ranked)
    return grades


def _shell_mask(
    grades: np.ndarray,
    lower: float,
    upper: float,
    *,
    final_shell: bool,
) -> np.ndarray:
    return (grades >= lower) & ((grades <= upper) if final_shell else (grades < upper))


def fit_continuous_mass_gate(
    groups: Sequence[object],
    coordinate: Sequence[float],
    targets: Sequence[float],
    direct_prediction: Sequence[float],
    crt_prediction: Sequence[float],
    *,
    shell_edges: Sequence[float] = SHELL_EDGES,
    alpha_grid: Sequence[float] = ALPHA_GRID,
    minimum_units: int = MINIMUM_UNITS,
    maximum_worst_unit_relative_rmse_regret: float = (
        MAXIMUM_WORST_UNIT_RELATIVE_RMSE_REGRET
    ),
) -> dict:
    """Select one safe continuous transport mass independently per shell."""
    units = np.asarray(groups)
    y = np.asarray(targets, dtype=np.float64)
    p0 = np.asarray(direct_prediction, dtype=np.float64)
    p1 = np.asarray(crt_prediction, dtype=np.float64)
    coordinate_array = np.asarray(coordinate, dtype=np.float64)
    n = len(units)
    if n == 0 or any(value.shape != (n,) for value in (y, p0, p1, coordinate_array)):
        raise ValueError("gate inputs must be non-empty aligned vectors")
    if not all(np.isfinite(value).all() for value in (y, p0, p1, coordinate_array)):
        raise ValueError("gate inputs must be finite")

    edges = tuple(float(value) for value in shell_edges)
    alphas = tuple(float(value) for value in alpha_grid)
    if (
        len(edges) < 2
        or not np.isfinite(edges).all()
        or any(right <= left for left, right in zip(edges, edges[1:]))
    ):
        raise ValueError("shell_edges must be finite and strictly increasing")
    if (
        not alphas
        or alphas[0] != 0.0
        or any(not np.isfinite(alpha) or alpha < 0.0 or alpha > 1.0 for alpha in alphas)
        or any(right <= left for left, right in zip(alphas, alphas[1:]))
    ):
        raise ValueError("alpha_grid must be strictly increasing from exact zero in [0, 1]")
    if minimum_units < 1 or not np.isfinite(maximum_worst_unit_relative_rmse_regret):
        raise ValueError("invalid safety constraint")

    grades = within_unit_grades(units, coordinate_array)
    shell_results = []
    for shell_index, (lower, upper) in enumerate(zip(edges, edges[1:])):
        shell = _shell_mask(
            grades, lower, upper, final_shell=shell_index == len(edges) - 2
        )
        shell_units = tuple(dict.fromkeys(units[shell].tolist()))
        baseline_unit_rmse = {}
        for unit in shell_units:
            mask = shell & (units == unit)
            baseline_unit_rmse[str(unit)] = math.sqrt(
                float(np.mean((y[mask] - p0[mask]) ** 2))
            )

        evidence = []
        for alpha in alphas:
            prediction = p0 + alpha * (p1 - p0)
            unit_rmse = {}
            unit_regret = {}
            for unit in shell_units:
                mask = shell & (units == unit)
                rmse = math.sqrt(float(np.mean((y[mask] - prediction[mask]) ** 2)))
                baseline_rmse = baseline_unit_rmse[str(unit)]
                unit_rmse[str(unit)] = rmse
                unit_regret[str(unit)] = (
                    rmse - baseline_rmse
                ) / max(baseline_rmse, 1e-12)
            unit_count = len(unit_rmse)
            mean_rmse = float(np.mean(list(unit_rmse.values()))) if unit_rmse else None
            worst_regret = (
                float(np.max(list(unit_regret.values()))) if unit_regret else None
            )
            enough = unit_count >= minimum_units
            safe = (
                enough
                and worst_regret is not None
                and worst_regret
                <= maximum_worst_unit_relative_rmse_regret + 1e-12
            )
            evidence.append(
                {
                    "alpha": alpha,
                    "safe": safe,
                    "unit_count": unit_count,
                    "mean_unit_rmse": mean_rmse,
                    "worst_unit_relative_rmse_regret": worst_regret,
                    "physical_unit_rmse": unit_rmse,
                    "unit_relative_rmse_regret": unit_regret,
                    "reason": (
                        "approved"
                        if safe
                        else (
                            "insufficient_unit_evidence"
                            if not enough
                            else "relative_regret_budget_exceeded"
                        )
                    ),
                }
            )

        direct_evidence = evidence[0]
        safe_candidates = [row for row in evidence if row["safe"]]
        if safe_candidates:
            best = min(
                safe_candidates,
                key=lambda row: (float(row["mean_unit_rmse"]), float(row["alpha"])),
            )
            improvement = float(direct_evidence["mean_unit_rmse"]) - float(
                best["mean_unit_rmse"]
            )
            selected_alpha = float(best["alpha"]) if improvement > 0.0 else 0.0
            selection_reason = (
                "safe_positive_mean_rmse_improvement"
                if selected_alpha > 0.0
                else "nonpositive_improvement_exact_fallback"
            )
        else:
            improvement = 0.0
            selected_alpha = 0.0
            selection_reason = "insufficient_unit_evidence_exact_fallback"
        shell_results.append(
            {
                "shell_index": shell_index,
                "lower_grade": lower,
                "upper_grade": upper,
                "selected_alpha": selected_alpha,
                "mean_unit_rmse_improvement_vs_direct": max(improvement, 0.0),
                "selection_reason": selection_reason,
                "validation_evidence": evidence,
            }
        )

    return {
        "shell_edges": list(edges),
        "alpha_grid": list(alphas),
        "minimum_units": minimum_units,
        "maximum_worst_unit_relative_rmse_regret": (
            maximum_worst_unit_relative_rmse_regret
        ),
        "shells": shell_results,
    }


def alpha_by_grade(grades: Sequence[float], frozen_gate: dict) -> np.ndarray:
    """Apply a frozen gate without labels; out-of-shell rows use exact alpha zero."""
    grade_array = np.asarray(grades, dtype=np.float64)
    if grade_array.ndim != 1 or not np.isfinite(grade_array).all():
        raise ValueError("grades must be a finite vector")
    result = np.zeros(len(grade_array), dtype=np.float64)
    shells = frozen_gate["shells"]
    for shell_index, shell in enumerate(shells):
        mask = _shell_mask(
            grade_array,
            float(shell["lower_grade"]),
            float(shell["upper_grade"]),
            final_shell=shell_index == len(shells) - 1,
        )
        result[mask] = float(shell["selected_alpha"])
    return result


def mix_exact_fallback(
    direct: np.ndarray, crt: np.ndarray, alpha: Sequence[float]
) -> np.ndarray:
    """Mix aligned predictions while preserving alpha-zero rows bitwise."""
    direct_array = np.asarray(direct)
    crt_array = np.asarray(crt)
    mass = np.asarray(alpha, dtype=np.float64)
    if direct_array.shape != crt_array.shape or direct_array.shape[0] != len(mass):
        raise ValueError("direct, CRT, and alpha must align")
    if direct_array.ndim not in (1, 2):
        raise ValueError("predictions must be a vector or matrix")
    output = direct_array.copy()
    transported = mass != 0.0
    if direct_array.ndim == 1:
        output[transported] = direct_array[transported] + mass[transported] * (
            crt_array[transported] - direct_array[transported]
        )
    else:
        output[transported] = direct_array[transported] + mass[transported, None] * (
            crt_array[transported] - direct_array[transported]
        )
    return output


def load_engression_external(test: dict, anchor: np.ndarray) -> dict:
    """Load the frozen equal-budget Engression artifact and score it externally."""
    published = json.loads((ENGRESSION / "results.json").read_text())
    artifact_path = ENGRESSION / "engression" / "predictions.npz"
    artifact = np.load(artifact_path)
    if not np.array_equal(artifact["y"], test["y"]) or not np.array_equal(
        artifact["groups"], test["groups"]
    ):
        raise ValueError("Engression artifact is not row-aligned to DS03 test")
    samples = np.asarray(artifact["prediction"], dtype=np.float64)
    if samples.ndim != 2:
        raise ValueError("Engression prediction artifact must be a sample matrix")
    if samples.shape[1] == len(test["y"]):
        samples = samples.T
    elif samples.shape[0] != len(test["y"]):
        raise ValueError("Engression predictions do not align with DS03 test rows")
    external_score = score(test["y"], test["groups"], samples, anchor)
    published_metrics = published["models"]["engression"]["ensemble"]
    return {
        **external_score,
        "artifact": str(artifact_path.relative_to(ROOT)),
        "candidate_budget": published["candidate_budget"],
        "refit_seeds": published["refit_seeds"],
        "published_pooled_r2": float(published_metrics["pooled"]["r2"]),
        "published_pooled_rmse": float(published_metrics["pooled"]["rmse"]),
        "published_metrics_match": bool(
            np.isclose(
                external_score["pooled_r2"],
                float(published_metrics["pooled"]["r2"]),
                rtol=0.0,
                atol=1e-7,
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, default=ROOT / "data/N-CMAPSS_DS03-012.h5")
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"immutable v3 output already exists: {args.out}")
    torch.set_num_threads(2)

    selection = json.loads(args.selection.read_text())
    if selection["selected_route"] != "direct_fallback" or selection["approved"]:
        raise ValueError("v3 requires the frozen rejected-prior direct fallback")
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
    validation_direct_samples = clipped_baseline(
        validation_matrix, validation_location, radius
    )
    model = fit_residual_transport(
        train["x"],
        train["y"],
        train["groups"],
        train_envelope,
        validation["x"],
        validation["y"],
        validation["groups"],
        validation_envelope,
        validation_direct_samples,
        train_disagreement=train_matrix.std(axis=0),
        validation_disagreement=validation_matrix.std(axis=0),
    )
    validation_crt_samples = sample_residual_transport(
        model,
        validation["x"],
        validation_envelope,
        validation_direct_samples,
        disagreement=validation_matrix.std(axis=0),
        seed=42,
    )
    frozen_gate = fit_continuous_mass_gate(
        validation["groups"],
        validation["cycles"],
        validation["y"],
        validation_direct_samples.mean(axis=1),
        validation_crt_samples.mean(axis=1),
    )
    # Gate fitting and all validation-label access end before this line.
    gate_frozen_before_test_load = True

    test = causal_features(load_revealed_test(args.h5), "direct")
    if tuple(map(int, np.unique(test["groups"]))) != TEST_UNITS:
        raise ValueError("unexpected test units")
    test_matrix = prediction_matrix(test, frozen_fits)
    test_location = test_matrix.mean(axis=0)
    test_envelope = ContractEnvelope(test_location, np.full(len(test["y"]), radius))
    test_direct_samples = clipped_baseline(test_matrix, test_location, radius)
    test_crt_samples = sample_residual_transport(
        model,
        test["x"],
        test_envelope,
        test_direct_samples,
        disagreement=test_matrix.std(axis=0),
        seed=42,
    )
    test_grades = within_unit_grades(test["groups"], test["cycles"])
    test_alpha = alpha_by_grade(test_grades, frozen_gate)
    gated_samples = mix_exact_fallback(
        test_direct_samples, test_crt_samples, test_alpha
    )
    direct_point = test_direct_samples.mean(axis=1)
    crt_point = test_crt_samples.mean(axis=1)
    gated_point = mix_exact_fallback(direct_point, crt_point, test_alpha)
    fallback_rows = test_alpha == 0.0
    fallback_samples_exact = np.array_equal(
        gated_samples[fallback_rows], test_direct_samples[fallback_rows]
    )
    fallback_point_exact = np.array_equal(
        gated_point[fallback_rows], direct_point[fallback_rows]
    )
    if not fallback_samples_exact or not fallback_point_exact:
        raise AssertionError("alpha-zero fallback lost bitwise identity")

    gated_score = score(test["y"], test["groups"], gated_samples, direct_point)
    direct_score = score(test["y"], test["groups"], test_direct_samples, direct_point)
    crt_score = score(test["y"], test["groups"], test_crt_samples, direct_point)
    engression_score = load_engression_external(test, direct_point)

    beats_direct = (
        gated_score["pooled_r2"] > direct_score["pooled_r2"]
        and gated_score["macro_rmse"] < direct_score["macro_rmse"]
    )
    beats_engression = (
        gated_score["pooled_r2"] > engression_score["pooled_r2"]
        and gated_score["macro_rmse"] < engression_score["macro_rmse"]
    )
    accepted = beats_direct and beats_engression
    rejection_reasons = []
    if not beats_direct:
        rejection_reasons.append(
            "continuous gate failed to strictly improve both pooled R2 and macro RMSE over direct fallback"
        )
    if not beats_engression:
        rejection_reasons.append(
            "continuous gate failed to strictly improve both pooled R2 and macro RMSE over equal-budget Engression"
        )

    result = {
        "schema": "ncmapss-ds03-crt-continuous-transport-mass-gate-v3",
        "status": "REJECTED" if not accepted else "ACCEPTED",
        "verdict": {
            "accepted": accepted,
            "rule": "must strictly beat both direct fallback and Engression on pooled R2 and macro RMSE",
            "beats_direct_fallback": beats_direct,
            "beats_engression": beats_engression,
            "rejection_reasons": rejection_reasons,
        },
        "split": {
            "train_units": list(TRAIN_UNITS),
            "validation_units": list(VALIDATION_UNITS),
            "test_units": list(TEST_UNITS),
        },
        "gate_training": {
            "source": "independent validation physical units 7--9 only",
            "test_labels_used": False,
            "point_prediction_primary": True,
            "frozen_gate": frozen_gate,
        },
        "selected_shell_alphas": [
            {
                "lower_grade": shell["lower_grade"],
                "upper_grade": shell["upper_grade"],
                "alpha": shell["selected_alpha"],
            }
            for shell in frozen_gate["shells"]
        ],
        "selected_test_row_counts_by_alpha": {
            str(alpha): int(np.sum(test_alpha == alpha))
            for alpha in sorted(set(test_alpha.tolist()))
        },
        "rho": model.rho,
        "envelope_radius_value": radius,
        "test": {
            "continuous_mass_gate": gated_score,
            "direct_fallback": direct_score,
            "crt": crt_score,
            "engression_external": engression_score,
        },
        "comparison": {
            "gate_minus_direct_pooled_r2": (
                gated_score["pooled_r2"] - direct_score["pooled_r2"]
            ),
            "gate_minus_direct_macro_rmse": (
                gated_score["macro_rmse"] - direct_score["macro_rmse"]
            ),
            "gate_minus_engression_pooled_r2": (
                gated_score["pooled_r2"] - engression_score["pooled_r2"]
            ),
            "gate_minus_engression_macro_rmse": (
                gated_score["macro_rmse"] - engression_score["macro_rmse"]
            ),
        },
        "invariance_checks": {
            "held_train_unit_absent_from_oof_fit": all(
                str(unit) not in fitted for unit, fitted in oof_audit.items()
            ),
            "gate_frozen_before_test_load": gate_frozen_before_test_load,
            "fallback_alpha_zero_point_bitwise_exact": fallback_point_exact,
            "fallback_alpha_zero_samples_bitwise_exact": fallback_samples_exact,
        },
        "external_artifacts": {
            "v1_results_unchanged": "results/ncmapss_ds03_crt_retrospective_v1",
            "v2_results_unchanged": "results/ncmapss_ds03_crt_grade_gate_v2",
            "engression": str(ENGRESSION.relative_to(ROOT)),
        },
    }
    args.out.mkdir(parents=True)
    np.savez_compressed(
        args.out / "predictions.npz",
        y=test["y"],
        groups=test["groups"],
        grades=test_grades,
        alpha=test_alpha,
        direct_samples=test_direct_samples,
        crt_samples=test_crt_samples,
        gated_samples=gated_samples,
        gated_prediction=gated_point,
    )
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
