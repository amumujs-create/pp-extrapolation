#!/usr/bin/env python3
"""Validation-gated assembly of plain, affine, and PP extrapolation heads."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from plain_mlp_ablation import fit_plain, predict_plain
from pp_extrapolation.metrics import regression_metrics
from pp_extrapolation.model import fit_pp, predict, select_affine_initialization
from pp_extrapolation.support_gate import predict_components


ARMS = ("plain_nn", "affine_tail", "pp")
MIN_RELATIVE_MSE_GAIN = 0.02
MIN_SEED_WINS = 4


def mse(truth: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.mean((np.asarray(estimate) - np.asarray(truth)) ** 2))


def fit_arm_predictions(
    train: dict,
    validation: dict,
    test: dict,
    seeds: tuple[int, ...],
) -> list[dict]:
    affine_selection = select_affine_initialization(train, validation)
    rows = []
    for seed in seeds:
        plain_fit = fit_plain(train, validation, seed=seed)
        pp_fit = fit_pp(
            train,
            validation,
            seed=seed,
            affine_selection=affine_selection,
        )
        validation_affine, _ = predict_components(pp_fit, validation["x"])
        test_affine, _ = predict_components(pp_fit, test["x"])
        validation_prediction = {
            "plain_nn": predict_plain(plain_fit, validation["x"]),
            "affine_tail": np.clip(
                validation_affine, 0.0, pp_fit.target_scale
            ),
            "pp": predict(pp_fit, validation["x"]),
        }
        test_prediction = {
            "plain_nn": predict_plain(plain_fit, test["x"]),
            "affine_tail": np.clip(test_affine, 0.0, pp_fit.target_scale),
            "pp": predict(pp_fit, test["x"]),
        }
        rows.append({
            "seed": seed,
            "validation_mse": {
                arm: mse(validation["y"], validation_prediction[arm]) for arm in ARMS
            },
            "test_prediction": test_prediction,
        })
    return rows


def select_arm(rows: list[dict]) -> dict:
    mean_mse = {
        arm: float(np.mean([row["validation_mse"][arm] for row in rows]))
        for arm in ARMS
    }
    plain = mean_mse["plain_nn"]
    audits = {}
    accepted = ["plain_nn"]
    for arm in ("affine_tail", "pp"):
        gain = (plain - mean_mse[arm]) / max(plain, 1e-12)
        wins = sum(
            row["validation_mse"][arm] < row["validation_mse"]["plain_nn"]
            for row in rows
        )
        passed = gain >= MIN_RELATIVE_MSE_GAIN and wins >= MIN_SEED_WINS
        audits[arm] = {
            "relative_validation_mse_gain_vs_plain": float(gain),
            "seed_wins_vs_plain": int(wins),
            "passed": bool(passed),
        }
        if passed:
            accepted.append(arm)
    selected = min(accepted, key=lambda arm: (mean_mse[arm], ARMS.index(arm)))
    return {
        "selected_arm": selected,
        "mean_validation_mse": mean_mse,
        "candidate_audits": audits,
        "accepted_arms": accepted,
        "thresholds": {
            "minimum_relative_mse_gain": MIN_RELATIVE_MSE_GAIN,
            "minimum_seed_wins": MIN_SEED_WINS,
        },
    }


def summarize_metrics(metrics: list[dict]) -> dict:
    values = np.asarray([row["pooled"]["r2"] for row in metrics])
    macros = np.asarray([row["unit_macro_r2"] for row in metrics])
    return {
        "pooled_r2_mean": float(values.mean()),
        "pooled_r2_sd": float(values.std(ddof=0)),
        "unit_macro_r2_mean": float(macros.mean()),
        "unit_macro_r2_sd": float(macros.std(ddof=0)),
    }


def evaluate_single(train, validation, test, seeds) -> dict:
    rows = fit_arm_predictions(train, validation, test, seeds)
    decision = select_arm(rows)
    arm_metrics = {
        arm: [
            regression_metrics(
                test["y"], row["test_prediction"][arm], test["groups"]
            )
            for row in rows
        ]
        for arm in ARMS
    }
    selected = decision["selected_arm"]
    return {
        "decision": decision,
        "arms": {arm: summarize_metrics(arm_metrics[arm]) for arm in ARMS},
        "selected": summarize_metrics(arm_metrics[selected]),
        "seed_validation_mse": [
            {"seed": row["seed"], **row["validation_mse"]} for row in rows
        ],
    }


def evaluate_nasa(folds, seeds) -> dict:
    fold_rows, fold_decisions = [], []
    for fold in folds:
        rows = fit_arm_predictions(
            fold["train"], fold["validation"], fold["test"], seeds
        )
        decision = select_arm(rows)
        fold_rows.append(rows)
        fold_decisions.append({
            "test_cell": fold["test_cell"],
            "validation_cell": fold["validation_cell"],
            **decision,
        })
    truth = np.concatenate([fold["test"]["y"] for fold in folds])
    groups = np.concatenate([fold["test"]["groups"] for fold in folds])
    per_arm_metrics = {arm: [] for arm in ARMS}
    selected_metrics = []
    for seed_index, seed in enumerate(seeds):
        arm_predictions = {
            arm: np.concatenate([
                fold_rows[index][seed_index]["test_prediction"][arm]
                for index in range(len(folds))
            ])
            for arm in ARMS
        }
        selected_prediction = np.concatenate([
            fold_rows[index][seed_index]["test_prediction"][
                fold_decisions[index]["selected_arm"]
            ]
            for index in range(len(folds))
        ])
        for arm in ARMS:
            per_arm_metrics[arm].append(
                regression_metrics(truth, arm_predictions[arm], groups)
            )
        selected_metrics.append(
            regression_metrics(truth, selected_prediction, groups)
        )
        print(
            f"NASA seed={seed} routes="
            f"{[row['selected_arm'] for row in fold_decisions]} "
            f"selected_R2={selected_metrics[-1]['pooled']['r2']:.4f}",
            flush=True,
        )
    return {
        "fold_decisions": fold_decisions,
        "arms": {
            arm: summarize_metrics(per_arm_metrics[arm]) for arm in ARMS
        },
        "selected": summarize_metrics(selected_metrics),
    }


def write_report(payload: dict, path: Path) -> None:
    lines = [
        "# Evidence-gated modular extrapolation heads", "",
        "Candidate heads were frozen as plain NN, affine tail, and PP. A non-plain "
        "head is eligible only when its mean validation MSE improves by at least 2% "
        "and it beats plain NN in at least four of five seeds. Final-test labels are "
        "not used for routing.", "",
        "| dataset | validation-selected route | plain NN R² | affine R² | PP R² | selected R² |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key, label in (("hust", "HUST"), ("virkler", "Virkler")):
        row = payload["datasets"][key]
        lines.append(
            f"| {label} | **{row['decision']['selected_arm']}** | "
            f"{row['arms']['plain_nn']['pooled_r2_mean']:.3f}±"
            f"{row['arms']['plain_nn']['pooled_r2_sd']:.3f} | "
            f"{row['arms']['affine_tail']['pooled_r2_mean']:.3f}±"
            f"{row['arms']['affine_tail']['pooled_r2_sd']:.3f} | "
            f"{row['arms']['pp']['pooled_r2_mean']:.3f}±"
            f"{row['arms']['pp']['pooled_r2_sd']:.3f} | "
            f"**{row['selected']['pooled_r2_mean']:.3f}±"
            f"{row['selected']['pooled_r2_sd']:.3f}** |"
        )
    nasa = payload["datasets"]["nasa"]
    routes = ", ".join(
        f"{row['test_cell']}:{row['selected_arm']}" for row in nasa["fold_decisions"]
    )
    lines.append(
        f"| NASA | **{routes}** | "
        f"{nasa['arms']['plain_nn']['pooled_r2_mean']:.3f}±"
        f"{nasa['arms']['plain_nn']['pooled_r2_sd']:.3f} | "
        f"{nasa['arms']['affine_tail']['pooled_r2_mean']:.3f}±"
        f"{nasa['arms']['affine_tail']['pooled_r2_sd']:.3f} | "
        f"{nasa['arms']['pp']['pooled_r2_mean']:.3f}±"
        f"{nasa['arms']['pp']['pooled_r2_sd']:.3f} | "
        f"**{nasa['selected']['pooled_r2_mean']:.3f}±"
        f"{nasa['selected']['pooled_r2_sd']:.3f}** |"
    )
    lines += [
        "", "This is a retrospective development test of the routing algorithm. "
        "The selected route must be frozen before evaluation on a new cohort to support "
        "a confirmatory coverage claim.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--legacy-root",
        default=str(Path(__file__).resolve().parents[2] / "ca-css-ncmapss"),
    )
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--output", default="results/evidence_gated_modules_v1")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.legacy_root).expanduser().resolve()))
    from run_affine_tail_external_nasa_health_v2 import prepare_folds
    from run_affine_tail_external_three import prepare_hust, prepare_virkler

    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if len(seeds) < MIN_SEED_WINS:
        raise ValueError(f"at least {MIN_SEED_WINS} seeds are required")
    torch.set_num_threads(2)
    hust = prepare_hust()
    virkler = prepare_virkler()
    nasa_folds, _ = prepare_folds()
    payload = {
        "experiment": "evidence_gated_modules_v1",
        "status": "retrospective route-development experiment",
        "primary_metric": "raw pooled R2 on final-test rows",
        "route_selection": "validation MSE only",
        "seeds": list(seeds),
        "thresholds": {
            "minimum_relative_validation_mse_gain": MIN_RELATIVE_MSE_GAIN,
            "minimum_seed_wins": MIN_SEED_WINS,
        },
        "datasets": {
            "hust": evaluate_single(*hust[:3], seeds),
            "virkler": evaluate_single(*virkler[:3], seeds),
            "nasa": evaluate_nasa(nasa_folds, seeds),
        },
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_report(payload, output / "RESULTS.md")
    print(f"saved {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
