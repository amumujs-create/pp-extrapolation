#!/usr/bin/env python3
"""DS03 prior-off fallback selection under unit-risk validation rules.

Prior-on routes stay rejected. Only the fallback arm is redesigned:
validation unit statistics choose among {PP-X direct, Engression, FT, ...}.

Test labels are used only after a rule freezes its choice.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "experiments"),
    str(ROOT / ".benchmark_deps"),
]

from full_equal_candidate_budget import REFIT_SEEDS, executor  # noqa: E402
from ncmapss_ds03_equal_budget import parts as ds03_parts  # noqa: E402
from pp_extrapolation.ds03_prospective import _predict_direct  # noqa: E402

EQUAL_DIR = ROOT / "results/ncmapss_ds03_equal_budget_v1"
PPX_DIR = ROOT / "results/ncmapss_ds03_ppx_v1"
OUT = ROOT / "results/ds03_prior_off_unit_risk_fallback_v1"
H5 = ROOT / "data/N-CMAPSS_DS03-012.h5"

PORTFOLIO = (
    "plain_mlp",
    "engression",
    "ft_transformer",
    "groupdro",
    "linear_tail_rbf",
    "vrex",
    "monotone",
    "svgp",
)


def unit_stats(y: np.ndarray, pred: np.ndarray, groups: np.ndarray) -> dict:
    unit_rmse = []
    for unit in np.unique(groups):
        mask = groups == unit
        unit_rmse.append(float(np.sqrt(np.mean((y[mask] - pred[mask]) ** 2))))
    unit_rmse = np.asarray(unit_rmse, dtype=float)
    return {
        "validation_mse": float(np.mean((y - pred) ** 2)),
        "unit_macro_rmse": float(np.mean(unit_rmse)),
        "worst_unit_rmse": float(np.max(unit_rmse)),
        "unit_rmse": {str(u): float(v) for u, v in zip(np.unique(groups), unit_rmse)},
    }


def relative_unit_risk(
    y: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    groups: np.ndarray,
) -> dict:
    ratios = []
    gains = []
    wins = 0
    for unit in np.unique(groups):
        mask = groups == unit
        base = float(np.sqrt(np.mean((y[mask] - baseline[mask]) ** 2)))
        cand = float(np.sqrt(np.mean((y[mask] - candidate[mask]) ** 2)))
        ratios.append(cand / max(base, 1e-12))
        gains.append(base - cand)
        wins += int(cand < base)
    ratios = np.asarray(ratios)
    gains = np.asarray(gains)
    if len(gains) <= 1:
        ci_low = float("-inf")
    else:
        idx = np.random.default_rng(20260913).integers(
            0, len(gains), size=(20_000, len(gains))
        )
        ci_low = float(np.quantile(gains[idx].mean(axis=1), 0.025))
    return {
        "unit_win_fraction_vs_baseline": float(wins / len(ratios)),
        "worst_unit_rmse_ratio_vs_baseline": float(ratios.max()),
        "mean_unit_gain_vs_baseline": float(gains.mean()),
        "unit_gain_ci_low": ci_low,
    }


def load_ppx_direct_validation(train: dict, validation: dict) -> np.ndarray:
    artifact = torch.load(
        PPX_DIR / "selection.models.pt",
        map_location="cpu",
        weights_only=False,
    )
    fits = artifact["fits"]["direct_fallback"]
    preds = [_predict_direct(fit, validation["x"]) for fit in fits]
    return np.mean(np.asarray(preds), axis=0)


def refit_validation_ensemble(model: str, split, config: dict) -> np.ndarray:
    """Refit selected config; return seed-mean validation prediction."""
    fit = executor(model)
    # Predict on validation by placing validation in the test slot.
    val_parts = (split[0], split[1], split[1])
    preds = []
    for seed in REFIT_SEEDS:
        info, pred = fit(val_parts, config, seed, True)
        pred = np.asarray(pred, dtype=float)
        if pred.shape != np.asarray(split[1]["y"]).shape:
            raise ValueError(f"{model} seed {seed}: validation shape mismatch")
        preds.append(pred)
        print(
            f"  {model} seed={seed} val_mse={info.get('validation_mse')}",
            flush=True,
        )
    return np.mean(np.asarray(preds), axis=0)


def apply_rules(rows: list[dict], baseline_name: str) -> dict:
    by_name = {row["model"]: row for row in rows}
    baseline = by_name[baseline_name]

    def pick(name: str) -> dict:
        row = by_name[name]
        return {
            "model": name,
            "validation_mse": row["validation_mse"],
            "unit_macro_rmse": row["unit_macro_rmse"],
            "worst_unit_rmse": row["worst_unit_rmse"],
            "test_pooled_r2": row["test_pooled_r2"],
        }

    # R1: pooled MSE (previous audit)
    r1 = min(rows, key=lambda r: (r["validation_mse"], -r["test_pooled_r2"]))

    # R2: unit-macro RMSE
    r2 = min(rows, key=lambda r: (r["unit_macro_rmse"], r["validation_mse"]))

    # R3: maximin / worst-unit RMSE
    r3 = min(rows, key=lambda r: (r["worst_unit_rmse"], r["unit_macro_rmse"], r["validation_mse"]))

    # R4: paper-style gate vs baseline; else keep baseline
    admitted = [baseline]
    threshold = baseline["validation_mse"] * 0.98
    for row in rows:
        if row["model"] == baseline_name:
            continue
        if (
            row["validation_mse"] < threshold
            and row["unit_win_fraction_vs_baseline"] >= 0.60
            and row["worst_unit_rmse_ratio_vs_baseline"] <= 1.10
        ):
            admitted.append(row)
    r4 = min(admitted, key=lambda r: (r["validation_mse"], r["model"]))

    # R5: lexicographic unit-risk then mse
    r5 = min(
        rows,
        key=lambda r: (
            r["worst_unit_rmse"],
            -r["unit_win_fraction_vs_baseline"],
            r["unit_macro_rmse"],
            r["validation_mse"],
        ),
    )

    # R6: require nonnegative unit-gain CI vs baseline, then lowest mse;
    #     if none, keep baseline.
    positive = [
        row
        for row in rows
        if row["model"] != baseline_name and row["unit_gain_ci_low"] >= 0.0
    ]
    r6 = (
        min(positive, key=lambda r: (r["validation_mse"], r["worst_unit_rmse"]))
        if positive
        else baseline
    )

    return {
        "pooled_mse": pick(r1["model"]),
        "unit_macro_rmse": pick(r2["model"]),
        "worst_unit_rmse": pick(r3["model"]),
        "paper_unit_gate_vs_direct": {
            **pick(r4["model"]),
            "admitted": [row["model"] for row in admitted],
        },
        "lexicographic_unit_risk": pick(r5["model"]),
        "unit_gain_ci_then_mse": pick(r6["model"]),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    equal = json.loads((EQUAL_DIR / "results.json").read_text())["models"]
    reveal = json.loads((PPX_DIR / "reveal.json").read_text())
    selection = json.loads((PPX_DIR / "selection.json").read_text())

    print("loading DS03 folds...", flush=True)
    split = ds03_parts(H5)
    train, validation, test = split
    yv, gv = validation["y"], validation["groups"]

    print("PP-X direct validation ensemble...", flush=True)
    ppx_val = load_ppx_direct_validation(train, validation)
    ppx_stats = unit_stats(yv, ppx_val, gv)
    ppx_rel = relative_unit_risk(yv, ppx_val, ppx_val, gv)
    ppx_row = {
        "model": "ppx_direct_fallback_frozen",
        **ppx_stats,
        **ppx_rel,
        "test_pooled_r2": float(reveal["pooled_r2"]),
        "source": "selection.models.pt",
    }

    cache_path = OUT / "validation_predictions.npz"
    cached = {}
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        for name in PORTFOLIO:
            key = f"{name}_val"
            if key in z.files:
                cached[name] = z[key]

    rows = [ppx_row]
    arrays = {"ppx_direct_fallback_frozen_val": ppx_val, "y": yv, "groups": gv}
    started = time.monotonic()
    for name in PORTFOLIO:
        print(f"refit validation ensemble: {name}", flush=True)
        if name in cached:
            pred = np.asarray(cached[name], dtype=float)
        else:
            pred = refit_validation_ensemble(name, split, equal[name]["selected_config"])
        arrays[f"{name}_val"] = pred
        stats = unit_stats(yv, pred, gv)
        rel = relative_unit_risk(yv, ppx_val, pred, gv)
        test_r2 = float(equal[name]["ensemble"]["pooled"]["r2"])
        rows.append(
            {
                "model": name,
                **stats,
                **rel,
                "test_pooled_r2": test_r2,
                "source": "selected_config_refit_on_validation",
            }
        )
        # Persist incrementally so long Engression runs are not lost.
        np.savez_compressed(cache_path, **arrays)

    rules = apply_rules(rows, "ppx_direct_fallback_frozen")
    oracle = max(rows, key=lambda r: r["test_pooled_r2"])

    payload = {
        "scope": (
            "DS03 prior-off fallback only; prior-on PP-X core unchanged. "
            "Selection uses validation unit-risk rules; test R2 is post-hoc."
        ),
        "protocol": {
            "baseline": "ppx_direct_fallback_frozen",
            "paper_gate": {
                "min_relative_mse_improvement": 0.02,
                "min_unit_win_fraction": 0.60,
                "max_worst_unit_rmse_ratio": 1.10,
            },
            "refit_seeds": list(REFIT_SEEDS),
            "frozen_ppx_reason": selection.get("reason"),
        },
        "candidates": rows,
        "rules": rules,
        "test_oracle": {
            "model": oracle["model"],
            "test_pooled_r2": oracle["test_pooled_r2"],
        },
        "findings": {
            "any_rule_picks_engression": any(
                value["model"] == "engression" for value in rules.values()
            ),
            "rule_models": {key: value["model"] for key, value in rules.items()},
            "rule_test_r2": {
                key: value["test_pooled_r2"] for key, value in rules.items()
            },
            "best_rule_test_r2": max(value["test_pooled_r2"] for value in rules.values()),
            "ppx_direct_test_r2": ppx_row["test_pooled_r2"],
            "seconds": time.monotonic() - started,
        },
    }
    (OUT / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(payload["findings"], indent=2, ensure_ascii=False))
    print("rules", json.dumps(rules, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
