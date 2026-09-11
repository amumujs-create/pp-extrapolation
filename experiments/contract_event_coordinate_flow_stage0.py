#!/usr/bin/env python3
"""Nine-setting Stage-0 screen of contract-conditioned event flow."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "experiments"),
    str(ROOT.parent / "ca-css-ncmapss"),
]

from ctbf_main_contract_benchmark import transformed_copy
from final_modular_pp_evidence import build_datasets
from full_equal_candidate_budget import load_settings
from pp_extrapolation import regression_metrics
from pp_extrapolation.event_flow import (
    fit_event_flow,
    latent_margin_violation_rate,
    predict_event_flow,
)
from summarize_full_equal_candidate_budget import baseline

OUT = ROOT / "results" / "contract_event_coordinate_flow_stage0"
SEEDS = (42, 43, 44)
CONFIGS = tuple(
    {"width": width, "learning_rate": learning_rate}
    for width in (16, 32)
    for learning_rate in (5e-4, 1e-3)
)
SETTINGS = (
    "hust",
    "virkler",
    "nasa",
    "sunwoda",
    "rwth",
    "mich",
    "matr",
    "matr_batch2",
    "ncmapss",
)
PAPER_NAMES = {
    "hust": "HUST",
    "virkler": "Virkler",
    "nasa": "NASA",
    "sunwoda": "SUNWODA",
    "rwth": "RWTH",
    "mich": "MICH",
    "matr": "MATR2019",
    "matr_batch2": "MATR-b2",
    "ncmapss": "N-CMAPSS",
}
OBSERVED = {"virkler", "nasa", "sunwoda", "rwth"}


def adapt_parts(name, parts):
    if name == "sunwoda":
        return tuple(
            transformed_copy(
                part,
                boundary=880.0,
                upper_boundary=False,
                rate_indices=(1, 3),
                mean_indices=(2,),
            )
            for part in parts
        )
    if name == "rwth":
        return tuple(
            transformed_copy(
                part,
                boundary=0.8,
                upper_boundary=False,
                rate_indices=(1, 3),
                mean_indices=(2,),
            )
            for part in parts
        )
    if name == "virkler":
        return tuple(
            transformed_copy(
                part,
                boundary=49.8,
                upper_boundary=True,
                rate_indices=(2, 3),
            )
            for part in parts
        )
    return tuple(
        {
            "x": np.asarray(part["x"], dtype=np.float32),
            "y": np.asarray(part["y"], dtype=np.float32),
            "groups": np.asarray(part["groups"]),
        }
        for part in parts
    )


def fit_candidate(parts, observed, config, seed, order_weight):
    return fit_event_flow(
        parts[0],
        parts[1],
        observed_coordinate=observed,
        seed=seed,
        width=config["width"],
        learning_rate=config["learning_rate"],
        weight_decay=0.1,
        quadrature_points=24,
        order_weight=order_weight,
        gauge_weight=0.01,
        max_epochs=300,
        patience=50,
    )


def run_parts(parts, observed):
    rows, predictions, fits = [], {}, {}
    for config in CONFIGS:
        key = f"w{config['width']}_lr{config['learning_rate']}"
        validation, test, selections, saved_fits = [], [], [], []
        for seed in SEEDS:
            fitted = fit_candidate(parts, observed, config, seed, 0.1)
            validation.append(predict_event_flow(fitted, parts[1]["x"]))
            test.append(predict_event_flow(fitted, parts[2]["x"]))
            selections.append(
                {
                    name: value
                    for name, value in fitted.selection.items()
                    if name != "history"
                }
            )
            saved_fits.append(fitted)
        validation_ensemble = np.mean(validation, axis=0)
        rows.append(
            {
                "key": key,
                **config,
                "validation_rmse": float(
                    np.sqrt(
                        np.mean(
                            (validation_ensemble - parts[1]["y"]) ** 2
                        )
                    )
                ),
                "selections": selections,
            }
        )
        predictions[key] = np.asarray(test)
        fits[key] = saved_fits
    selected = min(
        rows,
        key=lambda row: (
            row["validation_rmse"],
            row["width"],
            row["learning_rate"],
        ),
    )
    selected_prediction = predictions[selected["key"]]
    selected_fits = fits[selected["key"]]
    ablation = None
    if not observed:
        config = {
            "width": selected["width"],
            "learning_rate": selected["learning_rate"],
        }
        ablation = np.asarray(
            [
                predict_event_flow(
                    fit_candidate(parts, False, config, seed, 0.0),
                    parts[2]["x"],
                )
                for seed in SEEDS
            ]
        )
    violations = {
        "train": float(
            np.nanmean(
                [
                    latent_margin_violation_rate(fit, parts[0])
                    for fit in selected_fits
                ]
            )
        ),
        "validation": float(
            np.nanmean(
                [
                    latent_margin_violation_rate(fit, parts[1])
                    for fit in selected_fits
                ]
            )
        ),
    } if not observed else None
    return selected, rows, selected_prediction, ablation, violations


def load_reference(name):
    frozen = build_datasets()[PAPER_NAMES[name]]
    y = np.asarray(frozen[0])
    groups = np.asarray(frozen[1]).astype(str)
    ppx = np.asarray(frozen[2]).mean(axis=0)
    by, bg, direct = baseline(name, "plain_mlp")
    if not np.allclose(y, by) or not np.array_equal(groups, bg.astype(str)):
        raise RuntimeError(f"{name}: reference rows mismatch")
    return y, groups, ppx, np.asarray(direct).mean(axis=0)


def unit_summary(y, baseline_prediction, candidate, groups):
    ratios = []
    wins = 0
    for unit in np.unique(groups):
        mask = groups == unit
        baseline_rmse = math.sqrt(
            float(np.mean((baseline_prediction[mask] - y[mask]) ** 2))
        )
        candidate_rmse = math.sqrt(
            float(np.mean((candidate[mask] - y[mask]) ** 2))
        )
        ratios.append(candidate_rmse / max(baseline_rmse, 1e-12))
        wins += int(candidate_rmse < baseline_rmse)
    return {
        "wins": wins,
        "total": len(ratios),
        "win_fraction": float(wins / len(ratios)),
        "worst_rmse_ratio": float(np.max(ratios)),
        "mean_log_rmse_improvement": float(
            np.mean(-np.log(np.maximum(ratios, 1e-12)))
        ),
    }


def summarize_setting(name, parts, prediction, ablation, selected, search, violations):
    y = np.asarray(parts[2]["y"])
    groups = np.asarray(parts[2]["groups"]).astype(str)
    ref_y, ref_groups, ppx, direct = load_reference(name)
    if not np.allclose(y, ref_y) or not np.array_equal(groups, ref_groups):
        raise RuntimeError(f"{name}: test rows mismatch after adaptation")
    ensemble = prediction.mean(axis=0)
    metrics = {
        "event_flow": regression_metrics(y, ensemble, groups),
        "ppx": regression_metrics(y, ppx, groups),
        "direct_mlp_30c": regression_metrics(y, direct, groups),
    }
    arrays = {
        "y": y,
        "groups": groups,
        "event_flow_seeds": prediction,
        "event_flow": ensemble,
        "ppx": ppx,
        "direct_mlp_30c": direct,
    }
    if ablation is not None:
        ablation_ensemble = ablation.mean(axis=0)
        metrics["without_order_loss"] = regression_metrics(
            y, ablation_ensemble, groups
        )
        arrays["without_order_loss_seeds"] = ablation
        arrays["without_order_loss"] = ablation_ensemble
    return {
        "setting": name,
        "coordinate_mode": "observed" if name in OBSERVED else "latent",
        "selected": selected,
        "search": search,
        "metrics": metrics,
        "vs_ppx": unit_summary(y, ppx, ensemble, groups),
        "vs_direct": unit_summary(y, direct, ensemble, groups),
        "latent_margin_violation_rate": violations,
        "finite_predictions": bool(np.isfinite(prediction).all()),
    }, arrays


def evaluate_regular(name, raw):
    parts = adapt_parts(name, raw)
    selected, search, prediction, ablation, violations = run_parts(
        parts, name in OBSERVED
    )
    return summarize_setting(
        name, parts, prediction, ablation, selected, search, violations
    )


def evaluate_nasa(folds):
    predictions, ys, groups, selections, searches = [], [], [], [], []
    for fold_index, fold in enumerate(folds):
        parts = adapt_parts("nasa", fold)
        selected, search, prediction, _, _ = run_parts(parts, True)
        predictions.append(prediction)
        ys.append(parts[2]["y"])
        groups.append(parts[2]["groups"])
        selections.append({"fold": fold_index, **selected})
        searches.append({"fold": fold_index, "rows": search})
    prediction = np.concatenate(predictions, axis=1)
    synthetic_parts = (
        {"x": np.zeros((1, 1)), "y": np.zeros(1), "groups": np.asarray(["x"])},
        {"x": np.zeros((1, 1)), "y": np.zeros(1), "groups": np.asarray(["x"])},
        {
            "x": np.zeros((prediction.shape[1], 1)),
            "y": np.concatenate(ys),
            "groups": np.concatenate(groups),
        },
    )
    return summarize_setting(
        "nasa",
        synthetic_parts,
        prediction,
        None,
        {"by_fold": selections},
        {"by_fold": searches},
        None,
    )


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "results.json"
    if result_path.exists():
        raise RuntimeError("refusing to overwrite frozen event-flow Stage 0")
    raw = load_settings(SETTINGS)
    results = {}
    for name in SETTINGS:
        if name == "nasa":
            result, arrays = evaluate_nasa(raw[name])
        else:
            result, arrays = evaluate_regular(name, raw[name])
        results[name] = result
        np.savez_compressed(OUT / f"{name}_predictions.npz", **arrays)
        print(
            name,
            {
                model: round(values["pooled"]["r2"], 4)
                for model, values in result["metrics"].items()
            },
            flush=True,
        )
        (OUT / "results.partial.json").write_text(
            json.dumps({"status": "incomplete", "datasets": results}, indent=2)
            + "\n"
        )
    coverage = sum(row["finite_predictions"] for row in results.values())
    positive = sum(
        row["metrics"]["event_flow"]["pooled"]["r2"] > 0 for row in results.values()
    )
    direct_wins = sum(
        row["metrics"]["event_flow"]["pooled"]["rmse"]
        < row["metrics"]["direct_mlp_30c"]["pooled"]["rmse"]
        for row in results.values()
    )
    ppx_wins = sum(
        row["metrics"]["event_flow"]["pooled"]["rmse"]
        < row["metrics"]["ppx"]["pooled"]["rmse"]
        for row in results.values()
    )
    harm = max(
        row["metrics"]["event_flow"]["pooled"]["rmse"]
        / row["metrics"]["ppx"]["pooled"]["rmse"]
        - 1
        for row in results.values()
    )
    latent = [row for row in results.values() if row["coordinate_mode"] == "latent"]
    order_improvements = sum(
        row["metrics"]["event_flow"]["pooled"]["rmse"]
        < row["metrics"]["without_order_loss"]["pooled"]["rmse"]
        for row in latent
    )
    order_harms = sum(
        row["metrics"]["event_flow"]["pooled"]["rmse"]
        > 1.05 * row["metrics"]["without_order_loss"]["pooled"]["rmse"]
        for row in latent
    )
    passed = bool(
        coverage == 9
        and positive >= 7
        and direct_wins >= 6
        and ppx_wins >= 3
        and harm <= 0.50
        and order_improvements >= 2
        and order_harms <= 1
    )
    payload = {
        "status": "complete frozen contract event-coordinate flow Stage 0",
        "protocol": "CONTRACT_EVENT_COORDINATE_FLOW_STAGE0_PROTOCOL",
        "seeds": SEEDS,
        "candidate_count": len(CONFIGS),
        "datasets": results,
        "aggregate": {
            "coverage": coverage,
            "positive_r2": positive,
            "wins_vs_direct_mlp_30c": direct_wins,
            "wins_vs_ppx": ppx_wins,
            "maximum_rmse_harm_vs_ppx": float(harm),
            "latent_order_improvements": order_improvements,
            "latent_order_material_harms": order_harms,
            "stage0_pass": passed,
        },
    }
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    (OUT / "results.partial.json").unlink(missing_ok=True)
    print("aggregate", payload["aggregate"], flush=True)


if __name__ == "__main__":
    main()
