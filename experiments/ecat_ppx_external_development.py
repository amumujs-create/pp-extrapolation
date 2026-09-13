#!/usr/bin/env python3
"""Endpoint-conditioned analog transport challenge against Engression."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
from pp_extrapolation.endpoint_analog_transport import (
    predict_transport_policy,
    select_transport_policy,
)

OUT = ROOT / "results" / "ecat_ppx_external_development_v1"
PROTOCOL = "protocols/ECAT_PPX_EXTERNAL_DEVELOPMENT_PROTOCOL.md"
SEEDS = tuple(range(42, 47))
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
    error = y - prediction
    return {
        "r2": float(1.0 - np.sum(error**2) / np.sum((y - np.mean(y))**2)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
    }


def load_axial():
    items = []
    for tag, regimes in AXIAL_CONFIGS.items():
        train, validation, development, test, *_ = axial_features(tag, regimes)
        train, validation, development = map(axial_tail, (train, validation, development))
        items.append({
            "name": f"Axial-fan {tag}",
            "folds": [(train, validation)],
            "development": development,
            "test": {**test, "y": np.loadtxt(
                AXIAL_DATA / "sealed" / f"RUL_FAN_{tag}.txt"
            ).reshape(-1)},
        })
    return items


def load_matwi():
    frame = pd.read_csv(MATWI_DATA)
    tools, _ = clean_tools(frame)
    identifiers = sorted(str(int(value)) for value in frame["Set"].unique())
    permutation = np.asarray(identifiers)[np.random.default_rng(42).permutation(len(identifiers))]
    development_units = [unit for unit in permutation[:12] if unit in tools]
    test_units = [unit for unit in permutation[12:] if unit in tools]
    max_length = float(max(len(tools[unit]) for unit in development_units))
    fraction = 0.30
    development = make_rows(tools, development_units, tail_fraction=fraction,
                            max_development_length=max_length)
    test = make_rows(tools, test_units, tail_fraction=0.30,
                     max_development_length=max_length)
    split_folds = []
    for held_out in folds(development_units):
        fit_units = [unit for unit in development_units if unit not in held_out]
        split_folds.append((
            make_rows(tools, fit_units, tail_fraction=fraction,
                      max_development_length=max_length),
            make_rows(tools, held_out, tail_fraction=0.30,
                      max_development_length=max_length),
        ))
    return {"name": "MATWI tool-life", "folds": split_folds,
            "development": development, "test": test}


def load_misata():
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
    train = misata_rows(units, fit_ids, modes, controls, max_length, 0.30, stride=3)
    validation = misata_rows(units, validation_ids, modes, controls, max_length, 0.30)
    development = misata_rows(units, train_ids, modes, controls, max_length, 0.30, stride=3)
    test = misata_rows(units, test_ids, modes, controls, max_length, 0.30)
    return {"name": "Misata machine", "folds": [(train, validation)],
            "development": development, "test": test}


def policy_dict(policy):
    return {
        "experts": [
            {"name": candidate.name, "kind": candidate.kind,
             "parameters": candidate.parameters, "weight": weight}
            for candidate, weight in zip(policy.candidates, policy.weights)
        ],
        "validation_macro_mse": policy.validation_macro_mse,
        "validation_cvar20_mse": policy.validation_cvar20_mse,
        "validation_max_mse": policy.validation_max_mse,
        "affine_rejected": not any(
            candidate.kind == "ridge" for candidate in policy.candidates
        ),
    }


def evaluate(item):
    policy, audit = select_transport_policy(item["folds"])
    print("POLICY", item["name"], policy_dict(policy), flush=True)
    predictions, runs = [], []
    for seed in SEEDS:
        prediction = predict_transport_policy(
            policy, item["development"], item["test"], seed=seed
        )
        score = metric(item["test"]["y"], prediction)
        predictions.append(prediction)
        runs.append({"seed": seed, **score})
        print("SEED", item["name"], seed, score["r2"], flush=True)
    matrix = np.asarray(predictions)
    ensemble = metric(item["test"]["y"], matrix.mean(axis=0))
    seed_r2 = np.asarray([row["r2"] for row in runs])
    engression = ENGRESSION_R2[item["name"]]
    row = {
        "name": item["name"], "policy": policy_dict(policy), "audit": audit,
        "ensemble": ensemble, "runs": runs,
        "seed_r2_mean": float(np.mean(seed_r2)),
        "seed_r2_min": float(np.min(seed_r2)),
        "seed_r2_sd": float(np.std(seed_r2, ddof=1)),
        "engression_r2": engression,
        "delta_r2_vs_engression": ensemble["r2"] - engression,
        "beats_engression": bool(ensemble["r2"] > engression),
    }
    return row, matrix


def render(payload):
    lines = [
        "# ECAT-PPX 외부 Engression challenge 결과\n\n",
        "> 이미 열린 외부 코호트에서 수행한 retrospective development다.\n\n",
        "| 설정 | Engression R2 | ECAT-PPX R2 | delta | affine rejection | 판정 |\n",
        "|---|---:|---:|---:|---:|---|\n",
    ]
    for row in payload["settings"]:
        lines.append(
            f"| {row['name']} | {row['engression_r2']:.4f} | "
            f"{row['ensemble']['r2']:.4f} | {row['delta_r2_vs_engression']:+.4f} | "
            f"{row['policy']['affine_rejected']} | "
            f"**{'승' if row['beats_engression'] else '패'}** |\n"
        )
    summary = payload["summary"]
    lines.extend([
        "\n## 요약\n\n",
        f"- Engression 상대 승리: {summary['wins']}/5\n",
        f"- mean R2: Engression {summary['engression_mean_r2']:.4f}, "
        f"ECAT-PPX {summary['ecat_mean_r2']:.4f}\n",
        f"- minimum R2: {summary['minimum_r2']:.4f}\n",
        f"- positive settings: {summary['positive_settings']}/5\n",
        f"- mean seed SD: {summary['mean_seed_sd']:.4f}\n",
        f"- strict 5/5 success: {payload['strict_success']}\n\n",
        "Engression은 모델 입력이나 teacher로 사용하지 않았다. 5/5를 달성해도 "
        "새 untouched 외부 코호트 확증 전에는 일반적 우월성을 주장하지 않는다.\n",
    ])
    return "".join(lines)


def main():
    target = OUT / "results.json"
    if target.exists():
        raise RuntimeError(f"refusing to overwrite {target}")
    items = load_axial() + [load_matwi(), load_misata()]
    settings, arrays = [], {}
    for item in items:
        row, predictions = evaluate(item)
        settings.append(row)
        key = item["name"].lower().replace(" ", "_").replace("-", "_")
        arrays[f"{key}_truth"] = item["test"]["y"]
        arrays[f"{key}_ecat_ppx"] = predictions
        print("SCORED", item["name"], row["ensemble"], flush=True)
    values = np.asarray([row["ensemble"]["r2"] for row in settings])
    engression = np.asarray([row["engression_r2"] for row in settings])
    summary = {
        "wins": int(sum(row["beats_engression"] for row in settings)),
        "ecat_mean_r2": float(np.mean(values)),
        "engression_mean_r2": float(np.mean(engression)),
        "minimum_r2": float(np.min(values)),
        "positive_settings": int(np.sum(values > 0)),
        "mean_seed_r2": float(np.mean([row["seed_r2_mean"] for row in settings])),
        "minimum_seed_r2": float(np.min([row["seed_r2_min"] for row in settings])),
        "mean_seed_sd": float(np.mean([row["seed_r2_sd"] for row in settings])),
    }
    payload = {
        "model": "Endpoint-Conditioned Analog Transport PP-X (ECAT-PPX)",
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
