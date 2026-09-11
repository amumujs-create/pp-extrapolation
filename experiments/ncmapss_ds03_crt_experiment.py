#!/usr/bin/env python3
"""Rigorous retrospective matched CRT experiment on N-CMAPSS DS03.

The generic prior route was rejected prospectively.  Consequently this audit
uses the frozen direct-fallback distribution as its contract location and
freezes one outcome-space envelope radius from train-unit-held-out OOF
residuals only.  Validation physical units select rho; test outcomes are used
only once for frozen scoring.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pp_extrapolation.ds03_prospective import (
    SEEDS,
    TEST_UNITS,
    TRAIN_UNITS,
    VALIDATION_UNITS,
    _fit_direct,
    _predict_direct,
    causal_features,
    load_development,
    load_revealed_test,
)
from pp_extrapolation.residual_transport import (
    ContractEnvelope,
    fit_residual_transport,
    group_balanced_energy_score,
    sample_residual_transport,
)

DEFAULT_OUT = ROOT / "results/ncmapss_ds03_crt_retrospective_v1"
SELECTION = ROOT / "results/ncmapss_ds03_ppx_v1/selection.json"
ENGRESSION = ROOT / "results/ncmapss_ds03_equal_budget_v1"


def subset(rows: dict, units: tuple[int, ...]) -> dict:
    mask = np.isin(rows["groups"], np.asarray(units, dtype=str))
    return {key: value[mask] for key, value in rows.items()}


def prediction_matrix(rows: dict, fits: list[dict]) -> np.ndarray:
    return np.asarray([_predict_direct(fit, rows["x"]) for fit in fits])


def held_out_oof_anchor(train: dict) -> tuple[np.ndarray, dict[str, list[str]]]:
    matrix = np.empty((len(SEEDS), len(train["y"])), dtype=np.float64)
    audit: dict[str, list[str]] = {}
    for unit in TRAIN_UNITS:
        held = subset(train, (unit,))
        fit_rows = subset(train, tuple(u for u in TRAIN_UNITS if u != unit))
        indices = np.flatnonzero(train["groups"] == str(unit))
        audit[str(unit)] = sorted(set(fit_rows["groups"].tolist()))
        assert str(unit) not in audit[str(unit)]
        for seed_index, seed in enumerate(SEEDS):
            fit = _fit_direct(fit_rows, held, seed)
            matrix[seed_index, indices] = _predict_direct(fit, held["x"])
    assert np.all(np.isfinite(matrix))
    return matrix, audit


def clipped_baseline(matrix: np.ndarray, location: np.ndarray, radius: float) -> np.ndarray:
    # residual_transport expects rows x samples.
    samples = np.asarray(matrix, dtype=np.float64).T
    return np.clip(samples, location[:, None] - radius, location[:, None] + radius)


def score(y: np.ndarray, groups: np.ndarray, samples: np.ndarray, anchor: np.ndarray) -> dict:
    prediction = samples.mean(axis=1)
    unit_rmse, anchor_rmse, regret = {}, {}, {}
    for unit in np.unique(groups):
        mask = groups == unit
        model_value = math.sqrt(float(np.mean((y[mask] - prediction[mask]) ** 2)))
        anchor_value = math.sqrt(float(np.mean((y[mask] - anchor[mask]) ** 2)))
        unit_rmse[str(unit)] = model_value
        anchor_rmse[str(unit)] = anchor_value
        regret[str(unit)] = (model_value - anchor_value) / max(anchor_value, 1e-12)
    pooled_sse = float(np.sum((y - prediction) ** 2))
    return {
        "pooled_r2": float(1.0 - pooled_sse / np.sum((y - np.mean(y)) ** 2)),
        "pooled_rmse": math.sqrt(float(np.mean((y - prediction) ** 2))),
        "macro_rmse": float(np.mean(list(unit_rmse.values()))),
        "energy_score": group_balanced_energy_score(y, samples, groups),
        "physical_unit_rmse": unit_rmse,
        "anchor_physical_unit_rmse": anchor_rmse,
        "unit_regret_relative_rmse": regret,
        "unit_regret_summary": {
            "mean": float(np.mean(list(regret.values()))),
            "maximum": float(np.max(list(regret.values()))),
            "units_improved": int(np.sum(np.asarray(list(regret.values())) < 0)),
        },
    }


def load_engression(test: dict) -> dict:
    result = json.loads((ENGRESSION / "results.json").read_text())
    artifact = np.load(ENGRESSION / "engression" / "predictions.npz")
    if not np.array_equal(artifact["y"], test["y"]) or not np.array_equal(
        artifact["groups"], test["groups"]
    ):
        raise ValueError("Engression artifact is not row-aligned to DS03 test")
    prediction = np.asarray(artifact["prediction"]).mean(axis=0)
    metrics = result["models"]["engression"]["ensemble"]
    macro = float(np.mean([v["rmse"] for v in metrics["per_unit"].values()]))
    return {
        "artifact": "results/ncmapss_ds03_equal_budget_v1/engression/predictions.npz",
        "candidate_budget": result["candidate_budget"],
        "refit_seeds": result["refit_seeds"],
        "pooled_r2": float(metrics["pooled"]["r2"]),
        "pooled_rmse": float(metrics["pooled"]["rmse"]),
        "macro_rmse": macro,
        "prediction_checked_r2": float(
            1 - np.sum((test["y"] - prediction) ** 2)
            / np.sum((test["y"] - np.mean(test["y"])) ** 2)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, default=ROOT / "data/N-CMAPSS_DS03-012.h5")
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    torch.set_num_threads(2)

    selection = json.loads(args.selection.read_text())
    if selection["selected_route"] != "direct_fallback" or selection["approved"]:
        raise ValueError("experiment requires the prospectively rejected generic prior/direct fallback")
    bundle = torch.load(
        args.selection.with_name(selection["model_artifact"]),
        map_location="cpu",
        weights_only=False,
    )
    frozen_fits = bundle["fits"]["direct_fallback"]
    if tuple(selection["configuration"]["seeds"]) != SEEDS:
        raise ValueError("frozen selection seeds differ from 42--46")

    development = causal_features(load_development(args.h5), "direct")
    train = subset(development, TRAIN_UNITS)
    validation = subset(development, VALIDATION_UNITS)
    test = causal_features(load_revealed_test(args.h5), "direct")
    if tuple(map(int, np.unique(test["groups"]))) != TEST_UNITS:
        raise ValueError("unexpected test units")

    train_matrix, oof_audit = held_out_oof_anchor(train)
    validation_matrix = prediction_matrix(validation, frozen_fits)
    test_matrix = prediction_matrix(test, frozen_fits)
    train_location = train_matrix.mean(axis=0)
    validation_location = validation_matrix.mean(axis=0)
    test_location = test_matrix.mean(axis=0)

    # This scalar is deliberately fixed before validation/test outcomes enter
    # any transport choice. max(abs(.)) guarantees all train targets satisfy it.
    train_oof_residual = train["y"] - train_location
    radius = float(np.max(np.abs(train_oof_residual)) + 1e-9)
    train_envelope = ContractEnvelope(train_location, np.full(len(train["y"]), radius))
    validation_envelope = ContractEnvelope(
        validation_location, np.full(len(validation["y"]), radius)
    )
    test_envelope = ContractEnvelope(test_location, np.full(len(test["y"]), radius))
    validation_baseline = clipped_baseline(validation_matrix, validation_location, radius)
    test_baseline = clipped_baseline(test_matrix, test_location, radius)

    model = fit_residual_transport(
        train["x"], train["y"], train["groups"], train_envelope,
        validation["x"], validation["y"], validation["groups"],
        validation_envelope, validation_baseline,
        train_disagreement=train_matrix.std(axis=0),
        validation_disagreement=validation_matrix.std(axis=0),
    )
    test_samples = sample_residual_transport(
        model, test["x"], test_envelope, test_baseline,
        disagreement=test_matrix.std(axis=0), seed=42,
    )
    rho_zero = replace(model, rho=0.0)
    rho_zero_samples = sample_residual_transport(
        rho_zero, test["x"], test_envelope, test_baseline,
        disagreement=test_matrix.std(axis=0), seed=42,
    )
    assert np.array_equal(rho_zero_samples, test_baseline)
    assert np.all(test_samples >= test_location[:, None] - radius)
    assert np.all(test_samples <= test_location[:, None] + radius)

    frozen_anchor_mean = test_baseline.mean(axis=1)
    crt = score(test["y"], test["groups"], test_samples, frozen_anchor_mean)
    ablation = score(
        test["y"], test["groups"], rho_zero_samples, frozen_anchor_mean
    )
    engression = load_engression(test)
    result = {
        "schema": "ncmapss-ds03-crt-rigorous-retrospective-v1",
        "status": "retrospective matched; test outcome never used for selection",
        "split": {
            "train_units": list(TRAIN_UNITS),
            "validation_units": list(VALIDATION_UNITS),
            "test_units": list(TEST_UNITS),
        },
        "information_contract": {
            "prior_route": "generic prior rejected; direct fallback",
            "train_anchor": "unit-held-out OOF direct fits, seeds 42--46",
            "validation_test_anchor": "frozen prospective selection direct fits, seeds 42--46",
            "envelope_radius": "single max-absolute train OOF residual; frozen before validation/test",
            "test_used_for_selection": False,
            "oof_fit_units_by_held_unit": oof_audit,
        },
        "n_rows": {"train": len(train["y"]), "validation": len(validation["y"]), "test": len(test["y"])},
        "envelope_radius_value": radius,
        "rho": model.rho,
        "validation_risk_decision": asdict(model.risk_decision),
        "test": crt,
        "ablation_rho_0": ablation,
        "equal_budget_engression": engression,
        "comparison_to_engression": {
            "pooled_r2_difference": crt["pooled_r2"] - engression["pooled_r2"],
            "macro_rmse_difference": crt["macro_rmse"] - engression["macro_rmse"],
        },
        "invariance_checks": {
            "rho_zero_bitwise_anchor_distribution": True,
            "test_samples_inside_frozen_train_oof_envelope": True,
            "held_unit_absent_from_each_oof_fit": True,
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / "predictions.npz", y=test["y"], groups=test["groups"],
        anchor_samples=test_baseline, crt_samples=test_samples,
        prediction=test_samples.mean(axis=1),
    )
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
