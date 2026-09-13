#!/usr/bin/env python3
"""Develop independent distributional PP-X against Engression on opened cohorts."""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from axial_fan_tail_gate_pp_development import tail as axial_tail
from axial_fan_untouched_confirmation import CONFIGS as AXIAL_CONFIGS
from axial_fan_untouched_confirmation import DATA as AXIAL_DATA
from axial_fan_untouched_confirmation import load_features as axial_features
from matwi_tool_life_untouched import DATA as MATWI_DATA
from matwi_tool_life_untouched import clean_tools, folds, make_rows
from misata_machine_untouched import DATA as MISATA_DATA
from misata_machine_untouched import prepare as misata_prepare
from misata_machine_untouched import rows as misata_rows
from pp_extrapolation.distributional_pp import (
    fit_distributional_pp,
    predict_distributional_pp,
)

OUT = ROOT / "results" / "dspr_ppx_external_development_v1"
PROTOCOL = "protocols/DSPR_PPX_EXTERNAL_DEVELOPMENT_PROTOCOL.md"
SEEDS = tuple(range(42, 47))
CONFIGS = tuple({
    "width": width,
    "residual_bound": bound,
    "mean_weight": mean_weight,
} for width, bound, mean_weight in itertools.product(
    (32, 64), (0.5, 1.0), (0.1, 0.5)
))
ENGRESSION_R2 = {
    "Axial-fan 1P_8F": 0.665970,
    "Axial-fan 4P_1F": 0.547276,
    "Axial-fan 4P_8F": 0.633560,
    "MATWI tool-life": -0.305695,
    "Misata machine": 0.829752,
}


def metric(y, prediction):
    y = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    residual = y - prediction
    return {
        "r2": float(1.0 - np.sum(residual**2) / np.sum((y - np.mean(y))**2)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
    }


def axial():
    result = []
    for tag, regimes in AXIAL_CONFIGS.items():
        train, validation, development, test, *_ = axial_features(tag, regimes)
        train, validation, development = map(axial_tail, (train, validation, development))
        test = {**test, "y": np.loadtxt(
            AXIAL_DATA / "sealed" / f"RUL_FAN_{tag}.txt"
        ).reshape(-1)}
        result.append({
            "name": f"Axial-fan {tag}", "train": train,
            "validation": validation, "development": development, "test": test,
        })
    return result


def matwi():
    frame = pd.read_csv(MATWI_DATA)
    tools, _ = clean_tools(frame)
    identifiers = sorted(str(int(value)) for value in frame["Set"].unique())
    permutation = np.asarray(identifiers)[np.random.default_rng(42).permutation(len(identifiers))]
    development_units = [unit for unit in permutation[:12] if unit in tools]
    test_units = [unit for unit in permutation[12:] if unit in tools]
    max_length = float(max(len(tools[unit]) for unit in development_units))
    fraction = 0.30
    development = make_rows(
        tools, development_units, tail_fraction=fraction,
        max_development_length=max_length,
    )
    test = make_rows(
        tools, test_units, tail_fraction=0.30,
        max_development_length=max_length,
    )
    fold_parts = []
    for held_out in folds(development_units):
        fit_units = [unit for unit in development_units if unit not in held_out]
        fold_parts.append((
            make_rows(tools, fit_units, tail_fraction=fraction,
                      max_development_length=max_length),
            make_rows(tools, held_out, tail_fraction=0.30,
                      max_development_length=max_length),
        ))
    return {
        "name": "MATWI tool-life", "train": fold_parts[0][0],
        "validation": fold_parts[0][1], "development": development,
        "test": test, "fold_parts": fold_parts,
    }


def misata():
    frame = pd.read_csv(MISATA_DATA)
    units, modes, controls, max_length = misata_prepare(frame)
    train_ids = sorted(str(value) for value in frame.loc[
        frame.split == "train", "unit_id"
    ].unique())
    test_ids = sorted(str(value) for value in frame.loc[
        frame.split == "test", "unit_id"
    ].unique())
    order = np.asarray(train_ids)[np.random.default_rng(42).permutation(len(train_ids))]
    fit_ids, validation_ids = list(order[:64]), list(order[64:])
    return {
        "name": "Misata machine",
        "train": misata_rows(units, fit_ids, modes, controls, max_length, 0.30, stride=3),
        "validation": misata_rows(units, validation_ids, modes, controls, max_length, 0.30),
        "development": misata_rows(units, train_ids, modes, controls, max_length, 0.30, stride=3),
        "test": misata_rows(units, test_ids, modes, controls, max_length, 0.30),
    }


def select(item):
    rows = []
    folds_to_use = item.get("fold_parts")
    for config in CONFIGS:
        losses, epochs = [], []
        parts = folds_to_use or [(item["train"], item["validation"])]
        for train, validation in parts:
            fit = fit_distributional_pp(
                train, validation, seed=42, max_epochs=300, patience=50, **config
            )
            losses.append(fit.validation_mse)
            epochs.append(fit.selected_epoch)
        row = {
            "config": config,
            "validation_mse": float(np.mean(losses)),
            "selected_epoch": int(max(1, round(np.median(epochs)))),
        }
        rows.append(row)
        print("SEARCH", item["name"], row, flush=True)
    return min(rows, key=lambda row: row["validation_mse"]), rows


def run(item):
    selected, search = select(item)
    predictions, runs = [], []
    for seed in SEEDS:
        fit = fit_distributional_pp(
            item["development"], item["development"], seed=seed,
            max_epochs=selected["selected_epoch"],
            patience=selected["selected_epoch"] + 1,
            restore_best=False, **selected["config"],
        )
        prediction = predict_distributional_pp(fit, item["test"]["x"])
        predictions.append(prediction)
        runs.append({"seed": seed, **metric(item["test"]["y"], prediction)})
        print("REFIT", item["name"], seed, runs[-1]["r2"], flush=True)
    matrix = np.asarray(predictions)
    ensemble = metric(item["test"]["y"], matrix.mean(axis=0))
    seed_r2 = np.asarray([row["r2"] for row in runs])
    return {
        "name": item["name"], "selected": selected, "search": search,
        "ensemble": ensemble, "runs": runs,
        "seed_r2_mean": float(np.mean(seed_r2)),
        "seed_r2_min": float(np.min(seed_r2)),
        "seed_r2_sd": float(np.std(seed_r2, ddof=1)),
        "engression_r2": ENGRESSION_R2[item["name"]],
        "delta_r2_vs_engression": ensemble["r2"] - ENGRESSION_R2[item["name"]],
        "beats_engression": bool(ensemble["r2"] > ENGRESSION_R2[item["name"]]),
    }, matrix


def render(payload):
    lines = [
        "# DSPR-PPX 외부 Engression 격파 개발 결과\n\n",
        "> 외부 outcome 개봉 후 개발 결과이며 새 외부 확증이 아니다.\n\n",
        "| 설정 | Engression R2 | DSPR-PPX R2 | delta | 판정 |\n",
        "|---|---:|---:|---:|---|\n",
    ]
    for row in payload["settings"]:
        verdict = "승" if row["beats_engression"] else "패"
        lines.append(
            f"| {row['name']} | {row['engression_r2']:.4f} | "
            f"{row['ensemble']['r2']:.4f} | {row['delta_r2_vs_engression']:+.4f} | "
            f"**{verdict}** |\n"
        )
    lines.extend([
        "\n## 요약\n\n",
        f"- Engression 상대 승리: {payload['summary']['wins']}/5\n",
        f"- DSPR-PPX 평균 R2: {payload['summary']['mean_r2']:.4f}\n",
        f"- DSPR-PPX 최저 R2: {payload['summary']['minimum_r2']:.4f}\n",
        f"- 평균 seed SD: {payload['summary']['mean_seed_sd']:.4f}\n",
        f"- 엄격 성공: {payload['strict_success']}\n\n",
        "DSPR-PPX는 Engression을 입력이나 teacher로 사용하지 않는다. 다만 현재 "
        "코호트는 모두 열린 뒤이므로 성공하더라도 untouched 확증은 별도로 필요하다.\n",
    ])
    return "".join(lines)


def main():
    torch.set_num_threads(2)
    target = OUT / "results.json"
    if target.exists():
        raise RuntimeError(f"refusing to overwrite {target}")
    items = axial() + [matwi(), misata()]
    settings, arrays = [], {}
    for item in items:
        row, predictions = run(item)
        settings.append(row)
        key = item["name"].lower().replace(" ", "_").replace("-", "_")
        arrays[f"{key}_truth"] = item["test"]["y"]
        arrays[f"{key}_dspr_ppx"] = predictions
        print("SCORED", item["name"], row["ensemble"], flush=True)
    values = np.asarray([row["ensemble"]["r2"] for row in settings])
    summary = {
        "wins": int(sum(row["beats_engression"] for row in settings)),
        "mean_r2": float(np.mean(values)),
        "minimum_r2": float(np.min(values)),
        "positive_settings": int(np.sum(values > 0)),
        "mean_seed_r2": float(np.mean([row["seed_r2_mean"] for row in settings])),
        "minimum_seed_r2": float(np.min([row["seed_r2_min"] for row in settings])),
        "mean_seed_sd": float(np.mean([row["seed_r2_sd"] for row in settings])),
    }
    payload = {
        "model": "Distributional Structural Prior-Residual PP-X (DSPR-PPX)",
        "status": "retrospective external-cohort development",
        "protocol": PROTOCOL,
        "settings": settings,
        "summary": summary,
        "strict_success": summary["wins"] == 5,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(OUT / "predictions.npz", **arrays)
    (OUT / "RESULTS_KO.md").write_text(render(payload))
    print(json.dumps({"summary": summary, "strict_success": payload["strict_success"]}, indent=2))


if __name__ == "__main__":
    main()
