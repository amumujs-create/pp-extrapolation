#!/usr/bin/env python3
"""DS03 development experiment for Grade-CVaR Implicit Expert (GCIE) v4.

Twelve configurations are selected on units 1--6 / 7--9.  The architecture,
checkpoint policy, and fallback mass are serialized before the test loader is
called.  Units 10--15 are then loaded and scored once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pp_extrapolation.ds03_prospective import (
    SEEDS,
    TEST_UNITS,
    TRAIN_UNITS,
    VALIDATION_UNITS,
    _predict_direct,
    causal_features,
    load_development,
    load_revealed_test,
)
from pp_extrapolation.grade_cvar_implicit_expert import (
    GCIEConfig,
    fit_gcie,
    predict_samples,
)
from pp_extrapolation.residual_transport import group_balanced_energy_score


SEARCH_SEED = 42
DEFAULT_OUT = ROOT / "results/ncmapss_ds03_gcie_v4_dev"
DEFAULT_SELECTION = ROOT / "results/ncmapss_ds03_ppx_v1/selection.json"
ENGRESSION = ROOT / "results/ncmapss_ds03_equal_budget_v1"
MAXIMUM_UNIT_REGRET = 0.02
RHO_GRID = tuple(float(value) for value in np.linspace(0.0, 1.0, 21))


def _configs() -> tuple[GCIEConfig, ...]:
    common = {
        "noise_dim": 4,
        "weight_decay": 1e-3,
        "max_epochs": 600,
        "patience": 90,
        "sample_count": 8,
        "nearest_centroids": 1,
        "minimum_scale": 1e-3,
        "seed": SEARCH_SEED,
    }
    settings = (
        ((32, 32), 1e-3, 0.00, 0.00, 0.34),
        ((32, 32), 2e-3, 0.50, 0.25, 0.34),
        ((32, 32), 3e-3, 1.00, 1.00, 0.34),
        ((64, 64), 5e-4, 0.00, 0.50, 0.34),
        ((64, 64), 1e-3, 0.50, 1.00, 0.34),
        ((64, 64), 2e-3, 1.00, 2.00, 0.34),
        ((64, 64, 32), 5e-4, 0.50, 0.00, 0.50),
        ((64, 64, 32), 1e-3, 1.00, 0.50, 0.50),
        ((64, 64, 32), 2e-3, 2.00, 2.00, 0.50),
        ((128, 64), 5e-4, 0.00, 1.00, 0.34),
        ((128, 64), 1e-3, 1.00, 3.00, 0.34),
        ((128, 64), 2e-3, 2.00, 5.00, 0.50),
    )
    result = tuple(
        GCIEConfig(
            hidden_dims=hidden,
            learning_rate=lr,
            cvar_weight=cvar,
            monotonic_weight=monotonic,
            cvar_fraction=fraction,
            **common,
        )
        for hidden, lr, cvar, monotonic, fraction in settings
    )
    serialized = {json.dumps(asdict(config), sort_keys=True) for config in result}
    if len(result) != 12 or len(serialized) != 12:
        raise AssertionError("development contract requires 12 distinct configurations")
    return result


CONFIGS = _configs()


def subset(rows: dict[str, np.ndarray], units: tuple[int, ...]) -> dict[str, np.ndarray]:
    mask = np.isin(rows["groups"], np.asarray(units, dtype=str))
    return {key: value[mask] for key, value in rows.items()}


def _hash(value: Any) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _split_audit(rows: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "rows": int(len(rows["y"])),
        "units": [int(value) for value in np.unique(rows["groups"]).tolist()],
        "x_sha256": _hash(rows["x"]),
        "y_sha256": _hash(rows["y"]),
        "groups_sha256": _hash(rows["groups"].astype("U")),
        "cycles_sha256": _hash(rows["cycles"]),
    }


def _metrics(
    y: np.ndarray,
    groups: np.ndarray,
    samples: np.ndarray,
    direct_prediction: np.ndarray | None = None,
) -> dict[str, Any]:
    samples = np.asarray(samples, dtype=np.float64)
    prediction = samples.mean(axis=1)
    unit_rmse = {}
    unit_r2 = {}
    unit_regret = {}
    for unit in np.unique(groups):
        mask = groups == unit
        error = y[mask] - prediction[mask]
        rmse = math.sqrt(float(np.mean(error**2)))
        denominator = float(np.sum((y[mask] - np.mean(y[mask])) ** 2))
        unit_rmse[str(unit)] = rmse
        unit_r2[str(unit)] = (
            None if denominator <= 0 else float(1.0 - np.sum(error**2) / denominator)
        )
        if direct_prediction is not None:
            direct_rmse = math.sqrt(
                float(np.mean((y[mask] - direct_prediction[mask]) ** 2))
            )
            unit_regret[str(unit)] = (
                rmse - direct_rmse
            ) / max(direct_rmse, 1e-12)
    pooled_error = y - prediction
    pooled_denominator = float(np.sum((y - np.mean(y)) ** 2))
    result = {
        "pooled_r2": float(
            1.0 - np.sum(pooled_error**2) / pooled_denominator
        ),
        "pooled_rmse": math.sqrt(float(np.mean(pooled_error**2))),
        "macro_rmse": float(np.mean(list(unit_rmse.values()))),
        "energy_score": group_balanced_energy_score(y, samples, groups),
        "physical_unit_rmse": unit_rmse,
        "physical_unit_r2": unit_r2,
    }
    if unit_regret:
        result["unit_regret_relative_rmse_vs_direct"] = unit_regret
        result["worst_unit_regret_relative_rmse_vs_direct"] = float(
            max(unit_regret.values())
        )
    return result


def _fit(
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    config: GCIEConfig,
):
    return fit_gcie(
        train["x"],
        train["y"],
        train["groups"],
        train["cycles"],
        validation["x"],
        validation["y"],
        validation["groups"],
        validation["cycles"],
        config=config,
    )


def _direct_matrix(rows: dict[str, np.ndarray], fits: list[dict]) -> np.ndarray:
    return np.asarray([_predict_direct(fit, rows["x"]) for fit in fits]).T


def _repeat_direct(matrix: np.ndarray, samples_per_seed: int) -> np.ndarray:
    return np.concatenate(
        [
            np.repeat(matrix[:, index : index + 1], samples_per_seed, axis=1)
            for index in range(matrix.shape[1])
        ],
        axis=1,
    )


def _ensemble_samples(
    fits: list,
    rows: dict[str, np.ndarray],
    direct_matrix: np.ndarray,
    rho: float,
) -> np.ndarray:
    parts = []
    for index, fit in enumerate(fits):
        fallback = np.repeat(
            direct_matrix[:, index : index + 1], fit.config.sample_count, axis=1
        )
        parts.append(
            predict_samples(
                fit,
                rows["x"],
                sample_count=fit.config.sample_count,
                seed=int(fit.config.seed) + 30_000,
                rho=rho,
                fallback=fallback,
            )
        )
    return np.concatenate(parts, axis=1)


def _select_rho(
    y: np.ndarray,
    groups: np.ndarray,
    gcie_samples: np.ndarray,
    direct_samples: np.ndarray,
) -> tuple[float, list[dict[str, Any]]]:
    direct_point = direct_samples.mean(axis=1)
    table = []
    for rho in RHO_GRID:
        mixed = (
            direct_samples.copy()
            if rho == 0.0
            else direct_samples + rho * (gcie_samples - direct_samples)
        )
        metric = _metrics(y, groups, mixed, direct_point)
        safe = (
            metric["worst_unit_regret_relative_rmse_vs_direct"]
            <= MAXIMUM_UNIT_REGRET + 1e-12
        )
        table.append(
            {
                "rho": rho,
                "safe": safe,
                "pooled_r2": metric["pooled_r2"],
                "macro_rmse": metric["macro_rmse"],
                "worst_unit_regret_relative_rmse_vs_direct": metric[
                    "worst_unit_regret_relative_rmse_vs_direct"
                ],
            }
        )
    safe = [row for row in table if row["safe"]]
    chosen = min(safe, key=lambda row: (row["macro_rmse"], -row["pooled_r2"], row["rho"]))
    return float(chosen["rho"]), table


def _engression_metrics(
    test: dict[str, np.ndarray], direct_prediction: np.ndarray
) -> tuple[dict[str, Any], dict[str, Any]]:
    published = json.loads((ENGRESSION / "results.json").read_text())
    artifact_path = ENGRESSION / "engression/predictions.npz"
    artifact = np.load(artifact_path)
    if not np.array_equal(artifact["y"], test["y"]) or not np.array_equal(
        artifact["groups"], test["groups"]
    ):
        raise ValueError("Engression external artifact does not align with test rows")
    samples = np.asarray(artifact["prediction"], dtype=np.float64)
    if samples.shape[1] == len(test["y"]):
        samples = samples.T
    elif samples.shape[0] != len(test["y"]):
        raise ValueError("Engression external samples have the wrong shape")
    metrics = _metrics(test["y"], test["groups"], samples, direct_prediction)
    external = published["models"]["engression"]["ensemble"]
    audit = {
        "artifact": str(artifact_path.relative_to(ROOT)),
        "published_pooled_r2": float(external["pooled"]["r2"]),
        "published_pooled_rmse": float(external["pooled"]["rmse"]),
        "published_r2_matches_artifact": bool(
            np.isclose(
                metrics["pooled_r2"],
                float(external["pooled"]["r2"]),
                rtol=0.0,
                atol=1e-7,
            )
        ),
        "package_called_by_gcie": False,
        "used_for_tuning": False,
    }
    return metrics, audit


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, default=ROOT / "data/N-CMAPSS_DS03-012.h5")
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"immutable GCIE v4 output already exists: {args.out}")
    started = time.monotonic()
    torch.set_num_threads(2)

    prospective = json.loads(args.selection.read_text())
    if prospective["selected_route"] != "direct_fallback" or prospective["approved"]:
        raise ValueError("GCIE v4 requires the prior-rejected generic branch")
    if tuple(prospective["configuration"]["seeds"]) != SEEDS:
        raise ValueError("prospective direct fallback seeds are not frozen at 42--46")
    bundle = torch.load(
        args.selection.with_name(prospective["model_artifact"]),
        map_location="cpu",
        weights_only=False,
    )
    direct_fits = bundle["fits"]["direct_fallback"]

    development = causal_features(load_development(args.h5), "direct")
    train = subset(development, TRAIN_UNITS)
    validation = subset(development, VALIDATION_UNITS)
    validation_direct_matrix = _direct_matrix(validation, direct_fits)
    validation_direct_samples = _repeat_direct(
        validation_direct_matrix, CONFIGS[0].sample_count
    )
    validation_direct_point = validation_direct_samples.mean(axis=1)

    search_started = time.monotonic()
    search = []
    selected_search_fit = None
    for index, config in enumerate(CONFIGS):
        candidate_started = time.monotonic()
        fit = _fit(train, validation, config)
        samples = predict_samples(
            fit,
            validation["x"],
            sample_count=config.sample_count,
            seed=SEARCH_SEED + 10_000,
        )
        metric = _metrics(
            validation["y"],
            validation["groups"],
            samples,
            validation_direct_point,
        )
        search.append(
            {
                "candidate": index,
                "config": asdict(config),
                "selected_epoch": fit.selection["selected_epoch"],
                "epochs_executed": fit.selection["epochs_executed"],
                "best_validation_objective": fit.selection[
                    "best_validation_objective"
                ],
                "validation": metric,
                "seconds": time.monotonic() - candidate_started,
            }
        )
        if selected_search_fit is None or (
            metric["macro_rmse"],
            -metric["pooled_r2"],
            index,
        ) < selected_search_fit[0]:
            selected_search_fit = (
                (metric["macro_rmse"], -metric["pooled_r2"], index),
                fit,
            )
        print(
            "SEARCH",
            index,
            f"macro={metric['macro_rmse']:.6f}",
            f"r2={metric['pooled_r2']:.6f}",
            flush=True,
        )
    selected_row = min(
        search,
        key=lambda row: (
            row["validation"]["macro_rmse"],
            -row["validation"]["pooled_r2"],
            row["candidate"],
        ),
    )
    selected_index = int(selected_row["candidate"])
    selected_config = CONFIGS[selected_index]

    refit_started = time.monotonic()
    fits = []
    refits = []
    for seed in SEEDS:
        run_started = time.monotonic()
        if seed == SEARCH_SEED and selected_search_fit is not None:
            fit = selected_search_fit[1]
        else:
            fit = _fit(train, validation, replace(selected_config, seed=seed))
        fits.append(fit)
        validation_samples = predict_samples(
            fit,
            validation["x"],
            sample_count=fit.config.sample_count,
            seed=seed + 30_000,
        )
        refits.append(
            {
                "seed": seed,
                "selected_epoch": fit.selection["selected_epoch"],
                "epochs_executed": fit.selection["epochs_executed"],
                "validation": _metrics(
                    validation["y"],
                    validation["groups"],
                    validation_samples,
                    validation_direct_point,
                ),
                "seconds": time.monotonic() - run_started,
            }
        )
    validation_gcie_samples = _ensemble_samples(
        fits, validation, validation_direct_matrix, 1.0
    )
    rho, rho_table = _select_rho(
        validation["y"],
        validation["groups"],
        validation_gcie_samples,
        validation_direct_samples,
    )
    validation_selected_samples = (
        validation_direct_samples.copy()
        if rho == 0.0
        else validation_direct_samples
        + rho * (validation_gcie_samples - validation_direct_samples)
    )
    validation_selected = _metrics(
        validation["y"],
        validation["groups"],
        validation_selected_samples,
        validation_direct_point,
    )

    # Freeze the complete label-dependent policy before the first test loader call.
    args.out.mkdir(parents=True)
    selection_payload = {
        "schema": "ncmapss-ds03-gcie-v4-development-selection",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": "Grade-CVaR Implicit Expert (GCIE)",
        "prior_branch": "generic prior rejected; direct fallback retained as rho=0",
        "candidate_count": len(CONFIGS),
        "search_seed": SEARCH_SEED,
        "refit_seeds": list(SEEDS),
        "selected_candidate": selected_index,
        "selected_config": asdict(selected_config),
        "selected_rho": rho,
        "rho_selection_rule": (
            "minimum validation macro RMSE among rho grid values with maximum "
            "validation physical-unit relative RMSE regret <= 2%"
        ),
        "validation_config_table": search,
        "validation_refits": refits,
        "validation_rho_table": rho_table,
        "validation_selected": validation_selected,
        "split_audit": {
            "train": _split_audit(train),
            "validation": _split_audit(validation),
            "train_units_expected": list(TRAIN_UNITS),
            "validation_units_expected": list(VALIDATION_UNITS),
            "test_units_expected": list(TEST_UNITS),
        },
        "information_contract": {
            "engression_package_called": False,
            "engression_used_for_tuning": False,
            "test_loader_called": False,
            "test_labels_used_for_configuration_or_policy": False,
            "support_grade": (
                "robust-standardized distance to source physical-unit centroid; "
                "source rows LOO, application rows train centroids only"
            ),
        },
        "timing_seconds": {
            "search": time.monotonic() - search_started,
            "refit_and_policy": time.monotonic() - refit_started,
        },
    }
    selection_path = args.out / "selection.json"
    _write_json(selection_path, selection_payload)
    selection_sha256 = hashlib.sha256(selection_path.read_bytes()).hexdigest()

    test_started = time.monotonic()
    test = causal_features(load_revealed_test(args.h5), "direct")
    if tuple(map(int, np.unique(test["groups"]))) != TEST_UNITS:
        raise ValueError("unexpected test physical units")
    test_direct_matrix = _direct_matrix(test, direct_fits)
    test_direct_samples = _repeat_direct(
        test_direct_matrix, selected_config.sample_count
    )
    test_direct_point = test_direct_samples.mean(axis=1)
    test_gcie_samples = _ensemble_samples(fits, test, test_direct_matrix, rho)
    test_gcie = _metrics(
        test["y"], test["groups"], test_gcie_samples, test_direct_point
    )
    test_direct = _metrics(
        test["y"], test["groups"], test_direct_samples, test_direct_point
    )
    engression, engression_audit = _engression_metrics(test, test_direct_point)

    beats_engression = (
        test_gcie["pooled_r2"] > engression["pooled_r2"]
        and test_gcie["macro_rmse"] < engression["macro_rmse"]
    )
    regret_safe = (
        test_gcie["worst_unit_regret_relative_rmse_vs_direct"]
        <= MAXIMUM_UNIT_REGRET + 1e-12
    )
    accepted = beats_engression and regret_safe
    failures = []
    if test_gcie["pooled_r2"] <= engression["pooled_r2"]:
        failures.append("pooled R2 did not exceed Engression")
    if test_gcie["macro_rmse"] >= engression["macro_rmse"]:
        failures.append("macro RMSE did not improve on Engression")
    if not regret_safe:
        failures.append("maximum physical-unit regret exceeded 2%")
    total_seconds = time.monotonic() - started
    result = {
        "schema": "ncmapss-ds03-gcie-v4-development-result",
        "status": "ACCEPTED" if accepted else "REJECTED",
        "model_name": "Grade-CVaR Implicit Expert (GCIE)",
        "verdict": {
            "accepted": accepted,
            "criteria": {
                "pooled_r2_strictly_above_engression": True,
                "macro_rmse_strictly_below_engression": True,
                "maximum_unit_regret_vs_direct_at_most": MAXIMUM_UNIT_REGRET,
            },
            "beats_engression_both_aggregate_metrics": beats_engression,
            "maximum_unit_regret_safe": regret_safe,
            "failure_reasons": failures,
        },
        "selection_sha256": selection_sha256,
        "selection": selection_payload,
        "split_audit": {
            **selection_payload["split_audit"],
            "test": _split_audit(test),
        },
        "information_contract": {
            "config_and_policy_frozen_before_test_load": True,
            "test_evaluations": 1,
            "test_labels_used_for_configuration_or_policy": False,
            "engression_package_called": False,
            "engression_used_for_tuning": False,
            "engression_external_artifact_scored_after_freeze": True,
        },
        "test": {
            "gcie": test_gcie,
            "direct_fallback": test_direct,
            "engression_external": engression,
        },
        "engression_external_audit": engression_audit,
        "comparison": {
            "gcie_minus_engression_pooled_r2": (
                test_gcie["pooled_r2"] - engression["pooled_r2"]
            ),
            "gcie_minus_engression_macro_rmse": (
                test_gcie["macro_rmse"] - engression["macro_rmse"]
            ),
            "gcie_minus_direct_pooled_r2": (
                test_gcie["pooled_r2"] - test_direct["pooled_r2"]
            ),
            "gcie_minus_direct_macro_rmse": (
                test_gcie["macro_rmse"] - test_direct["macro_rmse"]
            ),
        },
        "timing_seconds": {
            **selection_payload["timing_seconds"],
            "test_load_predict_score": time.monotonic() - test_started,
            "total": total_seconds,
        },
    }
    np.savez_compressed(
        args.out / "predictions.npz",
        y=test["y"],
        groups=test["groups"],
        direct_samples=test_direct_samples,
        gcie_samples=test_gcie_samples,
        prediction=test_gcie_samples.mean(axis=1),
        selected_rho=np.asarray(rho),
    )
    _write_json(args.out / "results.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
