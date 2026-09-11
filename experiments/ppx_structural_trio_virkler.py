#!/usr/bin/env python3
"""Frozen Virkler screen of three PP-X-nested structural candidates."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "experiments"),
    str(ROOT.parent / "ca-css-ncmapss"),
]

from pp_extrapolation import fit_pp, regression_metrics, select_affine_initialization
from pp_extrapolation.support_gate import (
    predict_components,
    select_support_gate,
    support_distance,
)
from run_affine_tail_external_three import prepare_virkler

OUT = ROOT / "results" / "ppx_structural_trio_virkler"
SEEDS = (42, 43, 44, 45, 46)
BETAS = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
LAMBDAS = (0.5, 0.75, 0.9, 1.0)
TUBE_FRACTIONS = (0.1, 0.25, 0.5, 1.0, math.inf)
ETAS = (0.5, 0.75, 0.9, 0.95)
GAMMAS = (0.0, 0.25, 0.5, 0.75, 1.0)
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


def group_order(groups: np.ndarray, coordinate: np.ndarray):
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        yield group, indices[np.argsort(coordinate[indices], kind="stable")]


def support_gated_components(fit, x, distance, beta):
    affine, residual = predict_components(fit, x)
    gated_residual = np.exp(-float(beta) * np.maximum(distance, 0.0)) * residual
    prediction = np.clip(affine + gated_residual, 0.0, fit.target_scale)
    return affine, gated_residual, prediction


def projected_residual_state(
    affine, residual, x, groups, cap, *, persistence, tube_fraction
):
    output = np.empty(len(affine), dtype=float)
    tube = math.inf if math.isinf(tube_fraction) else tube_fraction * cap
    for _, order in group_order(groups, x[:, 0]):
        previous_raw = float(residual[order[0]])
        state = previous_raw
        if np.isfinite(tube):
            state = float(np.clip(state, -tube, tube))
        output[order[0]] = affine[order[0]] + state
        for index in order[1:]:
            raw = float(residual[index])
            innovation = raw - previous_raw
            state = persistence * state + innovation
            if np.isfinite(tube):
                state = float(np.clip(state, -tube, tube))
            output[index] = affine[index] + state
            previous_raw = raw
    return np.clip(output, 0.0, cap)


def temporal_projection(prediction, x, groups, cap, *, eta, strength):
    output = np.asarray(prediction, dtype=float).copy()
    for _, order in group_order(groups, x[:, 0]):
        elapsed = x[order, 1].astype(float)
        lifetime_observation = prediction[order].astype(float) + elapsed
        running = float(lifetime_observation[0])
        projected = np.empty(len(order), dtype=float)
        projected[0] = running - elapsed[0]
        for position in range(1, len(order)):
            running = eta * running + (1.0 - eta) * float(
                lifetime_observation[position]
            )
            projected[position] = running - elapsed[position]
        output[order] = (
            (1.0 - strength) * prediction[order] + strength * projected
        )
    return np.clip(output, 0.0, cap)


def backward_prior_geometry(affine, x, groups, train_max, train_sd):
    slope = np.zeros(len(affine), dtype=float)
    curvature = np.zeros(len(affine), dtype=float)
    for _, order in group_order(groups, x[:, 0]):
        coordinate = x[order, 0].astype(float)
        prior = affine[order].astype(float)
        local_slope = np.zeros(len(order), dtype=float)
        if len(order) > 1:
            dx = np.maximum(np.diff(coordinate), 1e-8)
            local_slope[1:] = np.diff(prior) / dx
            local_slope[0] = local_slope[1]
        local_curvature = np.zeros(len(order), dtype=float)
        if len(order) > 2:
            dx = np.maximum(np.diff(coordinate), 1e-8)
            local_curvature[2:] = np.diff(local_slope[1:]) / dx[1:]
            local_curvature[:2] = local_curvature[2]
        slope[order] = local_slope
        curvature[order] = local_curvature
    crack = np.asarray(x[:, 0], dtype=float)
    distance = np.maximum(crack - train_max, 0.0) / max(train_sd, 1e-8)
    cosine = np.maximum(np.cos(np.pi * crack / 152.4), 1e-8)
    paris_geometry = np.sqrt(np.maximum(crack, 0.0) / cosine)
    return np.column_stack((distance, slope, curvature, paris_geometry))


def standardize(values, center=None, scale=None):
    if center is None:
        center = np.mean(values, axis=0)
        scale = np.std(values, axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
    return (values - center) / scale, center, scale


def geometry_design(residual, geometry):
    return np.asarray(residual)[:, None] * np.column_stack(
        (np.ones(len(residual)), geometry)
    )


def geometry_loo(affine, residual, geometry, y, groups, alpha, cap):
    prediction = np.empty(len(y), dtype=float)
    for group in np.unique(groups):
        train_mask = groups != group
        held_mask = ~train_mask
        train_z, center, scale = standardize(geometry[train_mask])
        held_z, _, _ = standardize(geometry[held_mask], center, scale)
        model = Ridge(alpha=alpha, fit_intercept=False).fit(
            geometry_design(residual[train_mask], train_z),
            y[train_mask] - affine[train_mask],
        )
        correction = model.predict(
            geometry_design(residual[held_mask], held_z)
        )
        prediction[held_mask] = np.clip(
            affine[held_mask] + correction, 0.0, cap
        )
    return prediction


def geometry_fit_predict(
    validation_affine,
    validation_residual,
    validation_geometry,
    validation_y,
    test_affine,
    test_residual,
    test_geometry,
    alpha,
    cap,
):
    validation_z, center, scale = standardize(validation_geometry)
    test_z, _, _ = standardize(test_geometry, center, scale)
    model = Ridge(alpha=alpha, fit_intercept=False).fit(
        geometry_design(validation_residual, validation_z),
        validation_y - validation_affine,
    )
    return np.clip(
        test_affine + model.predict(geometry_design(test_residual, test_z)),
        0.0,
        cap,
    )


def rmse(y, prediction):
    return float(np.sqrt(np.mean((np.asarray(prediction) - np.asarray(y)) ** 2)))


def unit_summary(y, baseline, candidate, groups):
    ratios, log_improvements = [], []
    wins = 0
    for group in np.unique(groups):
        mask = groups == group
        base_rmse = rmse(y[mask], baseline[mask])
        candidate_rmse = rmse(y[mask], candidate[mask])
        ratio = candidate_rmse / max(base_rmse, 1e-12)
        ratios.append(ratio)
        log_improvements.append(-math.log(max(ratio, 1e-12)))
        wins += int(candidate_rmse < base_rmse)
    values = np.asarray(log_improvements)
    rng = np.random.default_rng(20260912)
    draws = rng.choice(values, size=(30_000, len(values)), replace=True).mean(1)
    return {
        "units_won": wins,
        "total_units": len(ratios),
        "win_fraction": float(wins / len(ratios)),
        "worst_unit_rmse_ratio": float(max(ratios)),
        "mean_unit_log_rmse_improvement": float(values.mean()),
        "unit_log_improvement_bootstrap_ci95": [
            float(value) for value in np.quantile(draws, (0.025, 0.975))
        ],
    }


def summarize_arm(name, y, groups, baseline, prediction, off_replay_error, approved):
    base_rmse = rmse(y, baseline)
    candidate_rmse = rmse(y, prediction)
    risk = unit_summary(y, baseline, prediction, groups)
    return {
        "name": name,
        "approved_on_validation": bool(approved),
        "metrics": regression_metrics(y, prediction, groups),
        "relative_test_rmse_change": candidate_rmse / base_rmse - 1.0,
        "full_row_coverage": float(np.mean(np.isfinite(prediction))),
        "finite": bool(np.isfinite(prediction).all()),
        "range_violations": int(
            np.count_nonzero((prediction < 0.0) | (prediction > np.max(y) * 10))
        ),
        "exact_off_replay_max_abs_error": float(off_replay_error),
        "unit_risk": risk,
        "promotion_candidate": bool(
            off_replay_error <= 1e-6
            and candidate_rmse <= 1.02 * base_rmse
            and risk["worst_unit_rmse_ratio"] <= 1.10
            and (
                candidate_rmse < base_rmse
                or risk["unit_log_improvement_bootstrap_ci95"][0] > 0
            )
        ),
        "rejected_over_5pct_harm": bool(candidate_rmse > 1.05 * base_rmse),
    }


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "results.json"
    if result_path.exists():
        raise RuntimeError("refusing to overwrite frozen structural-trio result")
    train, validation, test, audit = prepare_virkler()
    validation_distance, _ = support_distance(
        train["x"][:, [0]], validation["x"][:, [0]]
    )
    test_distance, _ = support_distance(
        train["x"][:, [0]], test["x"][:, [0]]
    )
    affine_selection = select_affine_initialization(train, validation)
    cap = float(affine_selection["target_scale"])
    validation_affines, validation_residuals, validation_ppx = [], [], []
    test_affines, test_residuals, test_ppx, base_runs = [], [], [], []
    for seed in SEEDS:
        fitted = fit_pp(
            train,
            validation,
            seed=seed,
            affine_selection=affine_selection,
            max_epochs=300,
        )
        gate = select_support_gate(
            fitted,
            validation["x"],
            validation["y"],
            validation_distance,
            betas=BETAS,
        )
        va, vr, vp = support_gated_components(
            fitted, validation["x"], validation_distance, gate.beta
        )
        ta, tr, tp = support_gated_components(
            fitted, test["x"], test_distance, gate.beta
        )
        validation_affines.append(va)
        validation_residuals.append(vr)
        validation_ppx.append(vp)
        test_affines.append(ta)
        test_residuals.append(tr)
        test_ppx.append(tp)
        base_runs.append(
            {
                "seed": seed,
                "selected_beta": gate.beta,
                "selected_epoch": fitted.selection["selected_epoch"],
            }
        )
    validation_affine = np.mean(validation_affines, axis=0)
    validation_residual = np.mean(validation_residuals, axis=0)
    validation_base = np.mean(validation_ppx, axis=0)
    test_affine = np.mean(test_affines, axis=0)
    test_residual = np.mean(test_residuals, axis=0)
    test_base = np.mean(test_ppx, axis=0)
    validation_base_rmse = rmse(validation["y"], validation_base)

    state_search = []
    for persistence in LAMBDAS:
        for tube_fraction in TUBE_FRACTIONS:
            prediction = projected_residual_state(
                validation_affine,
                validation_residual,
                validation["x"],
                validation["groups"],
                cap,
                persistence=persistence,
                tube_fraction=tube_fraction,
            )
            state_search.append(
                {
                    "persistence": persistence,
                    "tube_fraction": (
                        "infinity" if math.isinf(tube_fraction) else tube_fraction
                    ),
                    "validation_rmse": rmse(validation["y"], prediction),
                }
            )
    state_selected = min(
        state_search,
        key=lambda row: (
            row["validation_rmse"],
            row["persistence"] != 1.0
            or row["tube_fraction"] != "infinity",
        ),
    )
    state_approved = (
        state_selected["validation_rmse"] <= 0.995 * validation_base_rmse
    )
    state_persistence = (
        float(state_selected["persistence"]) if state_approved else 1.0
    )
    state_tube = (
        math.inf
        if not state_approved or state_selected["tube_fraction"] == "infinity"
        else float(state_selected["tube_fraction"])
    )
    state_test = projected_residual_state(
        test_affine,
        test_residual,
        test["x"],
        test["groups"],
        cap,
        persistence=state_persistence,
        tube_fraction=state_tube,
    )
    state_off = projected_residual_state(
        test_affine,
        test_residual,
        test["x"],
        test["groups"],
        cap,
        persistence=1.0,
        tube_fraction=math.inf,
    )

    temporal_search = []
    for eta in ETAS:
        for strength in GAMMAS:
            prediction = temporal_projection(
                validation_base,
                validation["x"],
                validation["groups"],
                cap,
                eta=eta,
                strength=strength,
            )
            temporal_search.append(
                {
                    "eta": eta,
                    "strength": strength,
                    "validation_rmse": rmse(validation["y"], prediction),
                }
            )
    temporal_selected = min(
        temporal_search,
        key=lambda row: (
            row["validation_rmse"],
            row["strength"] != 0.0,
            row["strength"],
        ),
    )
    temporal_approved = (
        temporal_selected["strength"] > 0
        and temporal_selected["validation_rmse"] <= 0.995 * validation_base_rmse
    )
    temporal_eta = float(temporal_selected["eta"])
    temporal_strength = (
        float(temporal_selected["strength"]) if temporal_approved else 0.0
    )
    temporal_test = temporal_projection(
        test_base,
        test["x"],
        test["groups"],
        cap,
        eta=temporal_eta,
        strength=temporal_strength,
    )
    temporal_off = temporal_projection(
        test_base,
        test["x"],
        test["groups"],
        cap,
        eta=temporal_eta,
        strength=0.0,
    )

    train_max = float(np.max(train["x"][:, 0]))
    train_sd = float(np.std(train["x"][:, 0]))
    validation_geometry = backward_prior_geometry(
        validation_affine,
        validation["x"],
        validation["groups"],
        train_max,
        train_sd,
    )
    test_geometry = backward_prior_geometry(
        test_affine,
        test["x"],
        test["groups"],
        train_max,
        train_sd,
    )
    geometry_search = []
    for alpha in RIDGE_ALPHAS:
        prediction = geometry_loo(
            validation_affine,
            validation_residual,
            validation_geometry,
            validation["y"],
            validation["groups"],
            alpha,
            cap,
        )
        geometry_search.append(
            {
                "alpha": alpha,
                "validation_rmse": rmse(validation["y"], prediction),
            }
        )
    geometry_selected = min(
        geometry_search, key=lambda row: (row["validation_rmse"], -row["alpha"])
    )
    geometry_approved = (
        geometry_selected["validation_rmse"] <= 0.995 * validation_base_rmse
    )
    geometry_test = (
        geometry_fit_predict(
            validation_affine,
            validation_residual,
            validation_geometry,
            validation["y"],
            test_affine,
            test_residual,
            test_geometry,
            float(geometry_selected["alpha"]),
            cap,
        )
        if geometry_approved
        else test_base.copy()
    )

    base_metrics = regression_metrics(test["y"], test_base, test["groups"])
    arms = {
        "projected_residual_state": summarize_arm(
            "projected_residual_state",
            test["y"],
            test["groups"],
            test_base,
            state_test,
            np.max(np.abs(state_off - test_base)),
            state_approved,
        ),
        "temporal_self_consistency": summarize_arm(
            "temporal_self_consistency",
            test["y"],
            test["groups"],
            test_base,
            temporal_test,
            np.max(np.abs(temporal_off - test_base)),
            temporal_approved,
        ),
        "prior_geometry_residual": summarize_arm(
            "prior_geometry_residual",
            test["y"],
            test["groups"],
            test_base,
            geometry_test,
            0.0,
            geometry_approved,
        ),
    }
    payload = {
        "status": "complete frozen Virkler PP-X structural trio screen",
        "protocol": "PPX_STRUCTURAL_TRIO_VIRKLER_PROTOCOL",
        "development_only": True,
        "audit": audit["audit"],
        "base_runs": base_runs,
        "validation_ppx_rmse": validation_base_rmse,
        "test_ppx": base_metrics,
        "selection": {
            "projected_residual_state": {
                "selected": state_selected,
                "approved": state_approved,
                "executed": {
                    "persistence": state_persistence,
                    "tube_fraction": (
                        "infinity" if math.isinf(state_tube) else state_tube
                    ),
                },
                "search": state_search,
            },
            "temporal_self_consistency": {
                "selected": temporal_selected,
                "approved": temporal_approved,
                "executed": {
                    "eta": temporal_eta,
                    "strength": temporal_strength,
                },
                "search": temporal_search,
            },
            "prior_geometry_residual": {
                "selected": geometry_selected,
                "approved": geometry_approved,
                "search": geometry_search,
            },
        },
        "arms": arms,
    }
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(
        OUT / "predictions.npz",
        y=test["y"],
        groups=test["groups"],
        ppx=test_base,
        projected_residual_state=state_test,
        temporal_self_consistency=temporal_test,
        prior_geometry_residual=geometry_test,
    )
    print(
        json.dumps(
            {
                "ppx": base_metrics["pooled"],
                "selection": {
                    key: {
                        "selected": value["selected"],
                        "approved": value["approved"],
                    }
                    for key, value in payload["selection"].items()
                },
                "arms": {
                    key: {
                        "pooled": value["metrics"]["pooled"],
                        "relative_test_rmse_change": value[
                            "relative_test_rmse_change"
                        ],
                        "worst_unit_rmse_ratio": value["unit_risk"][
                            "worst_unit_rmse_ratio"
                        ],
                        "promotion_candidate": value["promotion_candidate"],
                    }
                    for key, value in arms.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
