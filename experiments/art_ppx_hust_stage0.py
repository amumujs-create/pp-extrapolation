#!/usr/bin/env python3
"""HUST Stage-0 screen for anchored residual-derivative transport."""
from __future__ import annotations

import json
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

from ctbf_stanford_isu_development import bootstrap, unit_wins
from hust_regime_transport_pp import ALPHAS, calibrate, loo
from pp_extrapolation import (
    fit_pp,
    predict,
    regression_metrics,
    select_affine_initialization,
)
from run_affine_tail_external_three import prepare_hust

OUT = ROOT / "results" / "art_ppx_hust_stage0"
SEEDS = (42, 43, 44, 45, 46)
CONTEXTS = {"state": (0, 1), "rate": (2, 3), "all": (0, 1, 2, 3)}
BOUND_FRACTIONS = (0.02, 0.05, 0.10, 0.20)
AUTHORITIES = (0.0, 0.25, 0.50, 0.75, 1.0)


def regime_loo_prediction(kind, alpha, y, prediction, x, groups, cap):
    output = np.empty(len(y), dtype=float)
    for unit in np.unique(groups):
        fit_mask = groups != unit
        test_mask = ~fit_mask
        output[test_mask] = calibrate(
            kind,
            alpha,
            y[fit_mask],
            prediction[fit_mask],
            x[fit_mask],
            prediction[test_mask],
            x[test_mask],
            cap,
        )
    return output


def integrated_design(distance, context, center=None, scale=None):
    values = np.asarray(context, dtype=float)
    if center is None:
        center = values.mean(axis=0)
        scale = np.maximum(values.std(axis=0), 1e-8)
    z = (values - center) / scale
    d = np.asarray(distance, dtype=float)[:, None]
    return np.column_stack((d, 0.5 * d**2, d * z, 0.5 * d**2 * z)), center, scale


def correction_loo(base, y, x, groups, distance, context_indices, alpha):
    output = np.zeros(len(y), dtype=float)
    for unit in np.unique(groups):
        fit_mask = groups != unit
        test_mask = ~fit_mask
        design, center, scale = integrated_design(
            distance[fit_mask], x[fit_mask][:, context_indices]
        )
        test_design, _, _ = integrated_design(
            distance[test_mask],
            x[test_mask][:, context_indices],
            center,
            scale,
        )
        residual = y[fit_mask] - base[fit_mask]
        model = Ridge(alpha=alpha, fit_intercept=False).fit(design, residual)
        output[test_mask] = model.predict(test_design)
    return output


def bounded_prediction(base, correction, bound, authority, cap):
    bounded = bound * np.tanh(np.asarray(correction) / max(bound, 1e-8))
    return np.clip(base + authority * bounded, 0.0, cap)


def unit_risk(y, fallback, candidate, groups):
    ratios = []
    wins = 0
    for unit in np.unique(groups):
        mask = groups == unit
        fallback_rmse = float(np.sqrt(np.mean((fallback[mask] - y[mask]) ** 2)))
        candidate_rmse = float(np.sqrt(np.mean((candidate[mask] - y[mask]) ** 2)))
        ratios.append(candidate_rmse / max(fallback_rmse, 1e-12))
        wins += int(candidate_rmse < fallback_rmse)
    return {
        "wins": wins,
        "total": len(ratios),
        "win_fraction": float(wins / len(ratios)),
        "worst_rmse_ratio": float(max(ratios)),
    }


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "results.json"
    if result_path.exists():
        raise RuntimeError("refusing to overwrite frozen HUST ART-PPX result")
    train, validation, test = prepare_hust()[:3]
    affine = select_affine_initialization(train, validation)
    validation_base_seeds, test_base_seeds, base_runs = [], [], []
    cap = float(affine["target_scale"])
    for seed in SEEDS:
        fitted = fit_pp(
            train,
            validation,
            seed=seed,
            affine_selection=affine,
            max_epochs=300,
            patience=70,
            width=64,
            affine_anchor_weight=0.1,
        )
        validation_raw = predict(fitted, validation["x"])
        test_raw = predict(fitted, test["x"])
        candidates = []
        for kind in ("affine", "state", "rate", "all"):
            for alpha in ALPHAS:
                candidates.append(
                    {
                        "kind": kind,
                        "alpha": alpha,
                        "loo_mse": loo(
                            kind,
                            alpha,
                            validation["y"],
                            validation_raw,
                            validation["x"],
                            validation["groups"],
                            cap,
                        ),
                    }
                )
        selected = min(candidates, key=lambda row: row["loo_mse"])
        validation_base_seeds.append(
            regime_loo_prediction(
                selected["kind"],
                selected["alpha"],
                validation["y"],
                validation_raw,
                validation["x"],
                validation["groups"],
                cap,
            )
        )
        test_base_seeds.append(
            calibrate(
                selected["kind"],
                selected["alpha"],
                validation["y"],
                validation_raw,
                validation["x"],
                test_raw,
                test["x"],
                cap,
            )
        )
        base_runs.append({"seed": seed, "selected_transport": selected})
    validation_base = np.mean(validation_base_seeds, axis=0)
    test_base = np.mean(test_base_seeds, axis=0)
    train_boundary = float(np.min(train["x"][:, 0]))
    validation_distance = np.maximum(train_boundary - validation["x"][:, 0], 0.0)
    test_distance = np.maximum(train_boundary - test["x"][:, 0], 0.0)

    fallback_rmse = float(
        np.sqrt(np.mean((validation_base - validation["y"]) ** 2))
    )
    search = []
    correction_cache = {}
    for kind, indices in CONTEXTS.items():
        for alpha in ALPHAS:
            correction = correction_loo(
                validation_base,
                validation["y"],
                validation["x"],
                validation["groups"],
                validation_distance,
                indices,
                alpha,
            )
            correction_cache[(kind, alpha)] = correction
            for fraction in BOUND_FRACTIONS:
                bound = fraction * cap
                for authority in AUTHORITIES:
                    prediction = bounded_prediction(
                        validation_base, correction, bound, authority, cap
                    )
                    rmse = float(
                        np.sqrt(np.mean((prediction - validation["y"]) ** 2))
                    )
                    risk = unit_risk(
                        validation["y"],
                        validation_base,
                        prediction,
                        validation["groups"],
                    )
                    search.append(
                        {
                            "kind": kind,
                            "alpha": alpha,
                            "bound_fraction": fraction,
                            "authority": authority,
                            "validation_rmse": rmse,
                            "relative_gain": (fallback_rmse - rmse) / fallback_rmse,
                            **risk,
                        }
                    )
    selected = min(
        search,
        key=lambda row: (
            row["validation_rmse"],
            row["authority"],
            row["bound_fraction"],
            len(CONTEXTS[row["kind"]]),
            -row["alpha"],
        ),
    )
    selected_correction = correction_cache[
        (selected["kind"], selected["alpha"])
    ]
    selected_validation_prediction = bounded_prediction(
        validation_base,
        selected_correction,
        selected["bound_fraction"] * cap,
        selected["authority"],
        cap,
    )
    selected["bootstrap"] = bootstrap(
        validation["y"],
        validation_base,
        selected_validation_prediction,
        validation["groups"],
    )
    approved = bool(
        selected["authority"] > 0
        and selected["relative_gain"] >= 0.02
        and selected["win_fraction"] >= 0.60
        and selected["worst_rmse_ratio"] <= 1.05
        and selected["bootstrap"]["ci95"][0] > 0
    )
    authority = selected["authority"] if approved else 0.0
    indices = CONTEXTS[selected["kind"]]
    design, center, scale = integrated_design(
        validation_distance, validation["x"][:, indices]
    )
    model = Ridge(alpha=selected["alpha"], fit_intercept=False).fit(
        design, validation["y"] - validation_base
    )
    test_design, _, _ = integrated_design(
        test_distance, test["x"][:, indices], center, scale
    )
    test_correction = model.predict(test_design)
    test_prediction = bounded_prediction(
        test_base,
        test_correction,
        selected["bound_fraction"] * cap,
        authority,
        cap,
    )
    fallback_replay_error = float(
        np.max(
            np.abs(
                bounded_prediction(
                    test_base,
                    test_correction,
                    selected["bound_fraction"] * cap,
                    0.0,
                    cap,
                )
                - test_base
            )
        )
    )
    test_risk = unit_risk(
        test["y"], test_base, test_prediction, test["groups"]
    )
    base_metrics = regression_metrics(test["y"], test_base, test["groups"])
    art_metrics = regression_metrics(test["y"], test_prediction, test["groups"])
    relative_test_harm = (
        art_metrics["pooled"]["rmse"] / base_metrics["pooled"]["rmse"] - 1.0
    )
    continue_family = bool(
        approved
        and fallback_replay_error <= 1e-6
        and relative_test_harm <= 0.05
        and test_risk["worst_rmse_ratio"] <= 1.10
    )
    payload = {
        "status": "complete frozen HUST ART-PPX Stage 0",
        "protocol": "ART_PPX_HUST_STAGE0_PROTOCOL",
        "base_runs": base_runs,
        "train_support_boundary": train_boundary,
        "validation_fallback_rmse": fallback_rmse,
        "search": search,
        "selected": selected,
        "approved": approved,
        "executed_authority": authority,
        "fallback_replay_max_abs_error": fallback_replay_error,
        "test": {
            "ppx": base_metrics,
            "art_ppx": art_metrics,
            "relative_rmse_harm": float(relative_test_harm),
            "unit_risk": test_risk,
            "bootstrap": bootstrap(
                test["y"], test_base, test_prediction, test["groups"]
            ),
            "unit_wins": unit_wins(
                test["y"], test_base, test_prediction, test["groups"]
            ),
        },
        "continue_model_family": continue_family,
    }
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(
        OUT / "predictions.npz",
        y=test["y"],
        groups=test["groups"],
        ppx=test_base,
        art_ppx=test_prediction,
        correction=test_correction,
    )
    print(
        json.dumps(
            {
                "selected": selected,
                "approved": approved,
                "ppx": base_metrics["pooled"],
                "art_ppx": art_metrics["pooled"],
                "test_unit_risk": test_risk,
                "fallback_error": fallback_replay_error,
                "continue_model_family": continue_family,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
