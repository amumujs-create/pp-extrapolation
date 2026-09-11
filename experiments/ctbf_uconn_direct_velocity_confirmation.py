#!/usr/bin/env python3
"""Frozen UConn confirmation of contract-normalized direct-velocity CTBF."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

import uconn_ilcc_ppx_v12 as uconn
from ctbf_stanford_isu_development import bootstrap, unit_wins
from plain_mlp_ablation import fit_plain, predict_plain
from pp_extrapolation import regression_metrics
from pp_extrapolation.ctbf import (
    contract_normalize_features,
    fit_ctbf,
    predict_ctbf,
)

OUT = ROOT / "results" / "ctbf_uconn_direct_velocity_confirmation"
WIDTHS = (16, 32)
SELECTION_SEEDS = (42, 43, 44)
FINAL_SEEDS = (42, 43, 44, 45, 46)


def prepare():
    cells = uconn.load_cells()
    audit = [uconn.eligibility(cell, cells[cell]) for cell in sorted(cells)]
    eligible = [row["cell"] for row in audit if row["eligible"]]
    if len(eligible) < uconn.MIN_ELIGIBLE:
        raise RuntimeError(
            f"infeasible: {len(eligible)} eligible cells is below "
            f"the inherited minimum {uconn.MIN_ELIGIBLE}"
        )
    crossings = {
        row["cell"]: row["crossing_index"] for row in audit if row["eligible"]
    }
    ids = uconn.split_ids(eligible)
    train_all = uconn.make_rows(cells, ids["train"], crossings)
    cutoff = float(np.quantile(train_all["health"], 0.25))
    train = uconn.make_rows(
        cells, ids["train"], crossings, boundary=cutoff, train=True
    )
    tail_boundary = float(np.min(train["health"]))
    validation = uconn.make_rows(
        cells, ids["validation"], crossings, boundary=tail_boundary
    )
    test = uconn.make_rows(
        cells, ids["test"], crossings, boundary=tail_boundary
    )
    if (
        len(validation["y"]) < uconn.MIN_OUTER_ROWS
        or len(test["y"]) < uconn.MIN_OUTER_ROWS
    ):
        raise RuntimeError(
            "infeasible: strict-tail validation/test rows are below the "
            f"inherited minimum {uconn.MIN_OUTER_ROWS}"
        )
    boundaries = {
        split: np.full(len(rows["y"]), 0.8, dtype=float)
        for split, rows in (
            ("train", train),
            ("validation", validation),
            ("test", test),
        )
    }
    transformed = {}
    for split, rows in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        transformed[split] = {
            **rows,
            "x": contract_normalize_features(
                rows["x"],
                boundaries[split],
                rate_indices=(1, 2, 3),
                mean_indices=(4,),
                std_indices=(5,),
            ),
        }
    return transformed["train"], transformed["validation"], transformed["test"], {
        "eligibility": audit,
        "eligible_ids": eligible,
        "split_ids": ids,
        "train_health_q25": cutoff,
        "actual_train_min": tail_boundary,
        "rows": {
            "train": len(train["y"]),
            "validation": len(validation["y"]),
            "test": len(test["y"]),
        },
    }


def direct_fit(train, validation, width, seed):
    return fit_plain(
        train,
        validation,
        seed=seed,
        width=width,
        learning_rate=1e-3,
        weight_decay=0.1,
        max_epochs=350,
        patience=60,
    )


def flow_fit(train, validation, width, seed):
    return fit_ctbf(
        train,
        validation,
        seed=seed,
        width=width,
        boundary=0.0,
        quadrature_points=24,
        weak_rate_prior=False,
        residual_bound=1.0,
        learning_rate=1e-3,
        weight_decay=0.1,
        max_epochs=350,
        patience=60,
    )


def select_width(train, validation, *, model):
    rows = []
    for width in WIDTHS:
        predictions, epochs = [], []
        for seed in SELECTION_SEEDS:
            if model == "direct_mlp":
                fitted = direct_fit(train, validation, width, seed)
                predictions.append(predict_plain(fitted, validation["x"]))
                epochs.append(int(fitted["selected_epoch"]))
            else:
                fitted = flow_fit(train, validation, width, seed)
                predictions.append(predict_ctbf(fitted, validation["x"]))
                epochs.append(int(fitted.selection["selected_epoch"]))
        ensemble = np.mean(predictions, axis=0)
        rows.append(
            {
                "width": width,
                "validation_rmse": float(
                    np.sqrt(np.mean((ensemble - validation["y"]) ** 2))
                ),
                "epochs": epochs,
            }
        )
    selected = min(rows, key=lambda row: (row["validation_rmse"], row["width"]))
    return selected, rows


def final_predictions(train, validation, test, *, model, width):
    validation_predictions, test_predictions, epochs = [], [], []
    for seed in FINAL_SEEDS:
        if model == "direct_mlp":
            fitted = direct_fit(train, validation, width, seed)
            validation_predictions.append(predict_plain(fitted, validation["x"]))
            test_predictions.append(predict_plain(fitted, test["x"]))
            epochs.append(int(fitted["selected_epoch"]))
        else:
            fitted = flow_fit(train, validation, width, seed)
            validation_predictions.append(predict_ctbf(fitted, validation["x"]))
            test_predictions.append(predict_ctbf(fitted, test["x"]))
            epochs.append(int(fitted.selection["selected_epoch"]))
    return (
        np.mean(validation_predictions, axis=0),
        np.mean(test_predictions, axis=0),
        epochs,
    )


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "results.json"
    if result_path.exists():
        raise RuntimeError("refusing to overwrite frozen UConn CTBF confirmation")

    train, validation, test, data_audit = prepare()
    selected, search = {}, {}
    predictions = {}
    for model in ("direct_mlp", "direct_velocity_ctbf"):
        selected[model], search[model] = select_width(
            train, validation, model=model
        )
        val, tst, epochs = final_predictions(
            train,
            validation,
            test,
            model=model,
            width=selected[model]["width"],
        )
        predictions[model] = {"validation": val, "test": tst, "epochs": epochs}

    direct = predictions["direct_mlp"]["test"]
    flow = predictions["direct_velocity_ctbf"]["test"]
    wins = unit_wins(test["y"], direct, flow, test["groups"])
    paired = bootstrap(test["y"], direct, flow, test["groups"])
    probe = np.asarray(test["x"][: min(32, len(test["x"]))]).copy()
    probe[:, 0] = 0.0
    probe_fit = flow_fit(
        train,
        validation,
        selected["direct_velocity_ctbf"]["width"],
        FINAL_SEEDS[0],
    )
    boundary_error = float(np.max(np.abs(predict_ctbf(probe_fit, probe))))
    flow_metrics = regression_metrics(test["y"], flow, test["groups"])
    direct_metrics = regression_metrics(test["y"], direct, test["groups"])
    success = bool(
        flow_metrics["pooled"]["r2"] is not None
        and flow_metrics["pooled"]["r2"] > 0
        and flow_metrics["pooled"]["rmse"] < direct_metrics["pooled"]["rmse"]
        and wins["fraction"] >= 0.60
        and paired["ci95"][0] > 0
        and boundary_error == 0.0
    )
    stored_ppx = json.loads(
        (ROOT / "results/uconn_ilcc_ppx_v12/results.json").read_text()
    )
    payload = {
        "status": "frozen UConn external retrospective CTBF confirmation",
        "protocol": "CTBF_UCONN_DIRECT_VELOCITY_CONFIRMATION_PROTOCOL",
        "data_audit": data_audit,
        "selection_seeds": SELECTION_SEEDS,
        "final_seeds": FINAL_SEEDS,
        "search": search,
        "selected": selected,
        "epochs": {
            model: values["epochs"] for model, values in predictions.items()
        },
        "validation": {
            model: regression_metrics(
                validation["y"], values["validation"], validation["groups"]
            )
            for model, values in predictions.items()
        },
        "test": {
            "direct_mlp": direct_metrics,
            "direct_velocity_ctbf": flow_metrics,
        },
        "paired_cell_bootstrap": paired,
        "unit_wins": wins,
        "boundary_max_abs_prediction": boundary_error,
        "stored_ppx_v12_context": {
            "selected_test": stored_ppx.get("selected_test"),
            "matched_mlp_test": stored_ppx.get("matched_mlp_test"),
        },
        "confirmatory_success": success,
    }
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(
        OUT / "predictions.npz",
        y=test["y"],
        groups=test["groups"],
        direct_mlp=direct,
        direct_velocity_ctbf=flow,
    )
    print(
        json.dumps(
            {
                "direct_mlp": direct_metrics["pooled"],
                "direct_velocity_ctbf": flow_metrics["pooled"],
                "unit_wins": {
                    key: wins[key] for key in ("wins", "total", "fraction")
                },
                "bootstrap": paired,
                "boundary_error": boundary_error,
                "confirmatory_success": success,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
