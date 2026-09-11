#!/usr/bin/env python3
"""Validation-only mass gate over MultiStage Stage-1 safe predictors.

Selection uses leave-one-validation-unit-out predictions for every candidate.
The already-opened test rows and prediction artifacts are loaded only after
the shell routes and masses have been frozen.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

import multistage_rpt_ccmr_v19 as multistage
from multistage_rpt_ppx_residual_transport import score
from pp_extrapolation.causal_dynamics_bank import (
    fit_causal_dynamics_bank,
    predict_causal_dynamics_bank,
)
from pp_extrapolation.consensus_residual import (
    fit_consensus_residual,
    predict_consensus_residual,
)
from pp_extrapolation.residual_transport import (
    ContractEnvelope,
    fit_residual_transport,
    predict_residual_mean,
)

V1 = ROOT / "results/multistage_rpt_ppx_residual_transport_v1/predictions.npz"
COMPARATORS = ROOT / (
    "results/two_success_cohorts_extrapolation_competitors_v2_nonnegative/"
    "multistage_rpt_ensemble_predictions.npz"
)
CCMR = ROOT / "results/ccmr_v20_frozen_holdout_replay/predictions.npz"
OUT = ROOT / "results/multistage_rpt_ppx_mass_gate_v3"
SHELL_EDGES = (0.0, 0.5, 0.75, 1.0)
ALPHAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0)
MAX_RELATIVE_RMSE_REGRET = 0.02
OOF_SAMPLE_COUNT = 256
SEED = 20260911


def load_development():
    trajectories = multistage.load_trajectories()
    split = multistage.split_units(trajectories)
    train = multistage.make_rows(
        trajectories, split["train"], (0.0, 0.30)
    )
    validation = multistage.make_rows(
        trajectories, split["validation"], (0.40, 0.60)
    )
    return trajectories, split, train, validation


def subset(rows, indices):
    return {key: value[indices] for key, value in rows.items()}


def envelope(rows, radius):
    return ContractEnvelope(
        rows["context"][:, 0], np.full(len(rows["y"]), radius)
    )


def prior(rows):
    return np.repeat(
        rows["context"][:, :1], OOF_SAMPLE_COUNT, axis=1
    )


def grades(rows):
    result = np.empty(len(rows["y"]), dtype=float)
    for unit in dict.fromkeys(rows["groups"].tolist()):
        indices = np.flatnonzero(rows["groups"] == unit)
        ranked = indices[np.argsort(rows["progress"][indices], kind="stable")]
        result[ranked] = (np.arange(len(ranked)) + 0.5) / len(ranked)
    return result


def fit_oof_predictions(train, validation, radius):
    predictions = {
        name: np.empty(len(validation["y"]), dtype=float)
        for name in ("persistence", "PP_latest_successful", "CRT", "CCMR")
    }
    rho_by_unit = {}
    for unit in dict.fromkeys(validation["groups"].tolist()):
        heldout_idx = np.flatnonzero(validation["groups"] == unit)
        outer_idx = np.flatnonzero(validation["groups"] != unit)
        outer = subset(validation, outer_idx)
        heldout = subset(validation, heldout_idx)
        predictions["persistence"][heldout_idx] = heldout["context"][:, 0]

        pp = fit_consensus_residual(
            train["correction"],
            train["context"],
            train["y"],
            train["groups"],
            train["context"][:, 0],
            outer["correction"],
            outer["context"],
            outer["y"],
            outer["groups"],
            outer["context"][:, 0],
            max_ensemble_folds=20,
        )
        predictions["PP_latest_successful"][heldout_idx] = (
            predict_consensus_residual(
                pp,
                heldout["correction"],
                heldout["context"],
                heldout["context"][:, 0],
            )[0]
        )

        ccmr = fit_causal_dynamics_bank(
            train["correction"],
            train["context"],
            train["y"],
            train["groups"],
            train["context"][:, 0],
            outer["correction"],
            outer["context"],
            outer["y"],
            outer["groups"],
            outer["context"][:, 0],
        )
        predictions["CCMR"][heldout_idx] = predict_causal_dynamics_bank(
            ccmr,
            heldout["correction"],
            heldout["context"],
            heldout["context"][:, 0],
        )[0]

        crt = fit_residual_transport(
            train["context"],
            train["y"],
            train["groups"],
            envelope(train, radius),
            outer["context"],
            outer["y"],
            outer["groups"],
            envelope(outer, radius),
            prior(outer),
        )
        rho_by_unit[str(unit)] = crt.rho
        predictions["CRT"][heldout_idx] = predict_residual_mean(
            crt,
            heldout["context"],
            envelope(heldout, radius),
            prior(heldout),
            seed=SEED,
        )
    return predictions, rho_by_unit


def shell_metrics(y, units, shell, persistence, prediction):
    rmse, relative_regret = [], []
    for unit in dict.fromkeys(units.tolist()):
        chosen = shell & (units == unit)
        if not np.any(chosen):
            continue
        base = float(np.sqrt(np.mean(
            (persistence[chosen] - y[chosen]) ** 2
        )))
        candidate = float(np.sqrt(np.mean(
            (prediction[chosen] - y[chosen]) ** 2
        )))
        rmse.append(candidate)
        relative_regret.append(
            (candidate - base) / max(base, 1e-12)
        )
    return {
        "unit_count": len(rmse),
        "mean_unit_rmse": float(np.mean(rmse)),
        "worst_unit_relative_rmse_regret": float(
            np.max(relative_regret)
        ),
        "safe": bool(
            np.max(relative_regret) <= MAX_RELATIVE_RMSE_REGRET + 1e-12
        ),
    }


def freeze_policy(validation, predictions):
    validation_grades = grades(validation)
    shells = []
    for lower, upper in zip(SHELL_EDGES, SHELL_EDGES[1:]):
        shell = (validation_grades >= lower) & (
            (validation_grades <= upper)
            if upper == SHELL_EDGES[-1]
            else (validation_grades < upper)
        )
        discrete = {
            name: shell_metrics(
                validation["y"],
                validation["groups"],
                shell,
                predictions["persistence"],
                prediction,
            )
            for name, prediction in predictions.items()
        }
        safe_discrete = [
            (metrics["mean_unit_rmse"], name)
            for name, metrics in discrete.items()
            if metrics["safe"]
        ]
        _, anchor = min(safe_discrete, key=lambda item: (item[0], item[1]))

        mixtures = {}
        for alpha in ALPHAS:
            mixed = (
                (1.0 - alpha) * predictions["CCMR"]
                + alpha * predictions["CRT"]
            )
            mixtures[str(alpha)] = shell_metrics(
                validation["y"],
                validation["groups"],
                shell,
                predictions["persistence"],
                mixed,
            )
        safe_mixtures = [
            (metrics["mean_unit_rmse"], alpha)
            for alpha, metrics in zip(ALPHAS, mixtures.values())
            if metrics["safe"]
        ]
        best_mix = (
            min(safe_mixtures, key=lambda item: (item[0], item[1]))
            if safe_mixtures else None
        )
        selected_route, selected_alpha = anchor, None
        anchor_rmse = discrete[anchor]["mean_unit_rmse"]
        if best_mix is not None and best_mix[0] < anchor_rmse - 1e-15:
            selected_route = "CCMR_CRT_mix"
            selected_alpha = best_mix[1]
        shells.append({
            "lower_grade": lower,
            "upper_grade": upper,
            "discrete": discrete,
            "anchor": anchor,
            "alpha_grid": mixtures,
            "selected_route": selected_route,
            "selected_alpha": selected_alpha,
            "selection_reason": (
                "safe_alpha_improves_anchor"
                if selected_alpha is not None
                else "anchor_exact_fallback"
            ),
        })
    return tuple(shells)


def apply_policy(policy, test_grades, test_predictions):
    output = np.empty(len(test_grades), dtype=float)
    route = np.empty(len(test_grades), dtype=object)
    alpha_used = np.full(len(test_grades), np.nan)
    for shell in policy:
        lower, upper = shell["lower_grade"], shell["upper_grade"]
        chosen = (test_grades >= lower) & (
            (test_grades <= upper)
            if upper == SHELL_EDGES[-1]
            else (test_grades < upper)
        )
        selected = shell["selected_route"]
        if selected == "CCMR_CRT_mix":
            alpha = shell["selected_alpha"]
            output[chosen] = (
                (1.0 - alpha) * test_predictions["CCMR"][chosen]
                + alpha * test_predictions["CRT"][chosen]
            )
            alpha_used[chosen] = alpha
        else:
            output[chosen] = test_predictions[selected][chosen]
        route[chosen] = selected
    return output, route, alpha_used


def main():
    if OUT.exists():
        raise RuntimeError("refusing to overwrite MultiStage mass-gate v3")
    trajectories, split, train, validation = load_development()
    radius = max(
        float(np.max(np.abs(
            train["y"] - train["context"][:, 0]
        ))) * 1.05,
        np.finfo(float).eps,
    )
    oof_predictions, rho_by_unit = fit_oof_predictions(
        train, validation, radius
    )
    policy = freeze_policy(validation, oof_predictions)

    # The test split and all test prediction artifacts are opened only now.
    test = multistage.make_rows(
        trajectories, split["test"], (0.75, 0.90)
    )
    v1 = np.load(V1)
    comparators = np.load(COMPARATORS)
    ccmr_artifact = np.load(CCMR)
    alignments = {
        "v1_truth": np.allclose(v1["truth"], test["y"]),
        "comparator_truth": np.allclose(comparators["truth"], test["y"]),
        "ccmr_truth": np.allclose(
            ccmr_artifact["MultiStage_RPT_truth"], test["y"]
        ),
        "groups": np.array_equal(
            v1["groups"].astype(str), test["groups"].astype(str)
        ),
    }
    if not all(alignments.values()):
        raise RuntimeError(f"test artifact alignment failure: {alignments}")
    test_predictions = {
        "persistence": test["context"][:, 0],
        "PP_latest_successful": comparators["PP_latest_successful"],
        "CRT": v1["ppx_residual_transport"],
        "CCMR": ccmr_artifact["MultiStage_RPT_CCMR_v20"],
    }
    test_grades = grades(test)
    combined, route, alpha_used = apply_policy(
        policy, test_grades, test_predictions
    )
    candidates = {
        "PP-X_mass_gate_v3": combined,
        "CCMR_v2.0": test_predictions["CCMR"],
        "PP-X_CRT_v1": test_predictions["CRT"],
        "PP_latest_successful": test_predictions["PP_latest_successful"],
        "Engression": comparators["Engression"],
        "Persistence": test_predictions["persistence"],
    }
    scores = {
        name: score(
            test["y"],
            test["groups"],
            test_predictions["persistence"],
            prediction,
        )
        for name, prediction in candidates.items()
    }
    promoted = bool(
        scores["PP-X_mass_gate_v3"]["pooled_rmse"]
        < scores["CCMR_v2.0"]["pooled_rmse"]
        and scores["PP-X_mass_gate_v3"]["macro_rmse"]
        <= scores["CCMR_v2.0"]["macro_rmse"]
        and scores["PP-X_mass_gate_v3"]["worst_unit_regret"]
        <= scores["CCMR_v2.0"]["worst_unit_regret"]
    )
    payload = {
        "status": "retrospective validation-only mass gate v3 complete",
        "confirmatory": False,
        "test_status": "previously opened; loaded after policy freeze",
        "evidence_grade": "validation 14-unit leave-one-unit-out",
        "validation_reconstruction": {
            "CCMR": (
                "causal_dynamics_bank refit per held unit; held unit excluded "
                "from fit and risk selection"
            ),
            "PP_latest_successful": (
                "consensus_residual refit per held unit; held unit excluded "
                "from fit and risk selection"
            ),
            "CRT": (
                "residual_transport refit per held unit; held unit excluded "
                "from rho selection"
            ),
        },
        "safety_constraint": {
            "metric": "worst per-unit relative RMSE regret vs persistence",
            "maximum": MAX_RELATIVE_RMSE_REGRET,
        },
        "alpha_grid": list(ALPHAS),
        "shells": list(policy),
        "oof_crt_rho_by_held_unit": rho_by_unit,
        "test_route_counts": {
            name: int(np.sum(route == name)) for name in np.unique(route)
        },
        "test_alpha_counts": {
            str(alpha): int(np.sum(alpha_used == alpha))
            for alpha in ALPHAS
            if np.any(alpha_used == alpha)
        },
        "scores": scores,
        "promotion": {
            "promoted_over_ccmr": promoted,
            "decision": (
                "promotion_success"
                if promoted else "promotion_failed_no_CCMR_improvement"
            ),
        },
        "alignment_checks": alignments,
    }
    OUT.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        OUT / "predictions.npz",
        truth=test["y"],
        groups=test["groups"].astype(str),
        progress=test["progress"],
        grade=test_grades,
        route=route.astype(str),
        alpha=alpha_used,
        mass_gate_v3=combined,
        ccmr=test_predictions["CCMR"],
        crt=test_predictions["CRT"],
        pp_latest=test_predictions["PP_latest_successful"],
        persistence=test_predictions["persistence"],
    )
    (OUT / "results.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps({
        "shell_policy": [
            {
                "shell": [item["lower_grade"], item["upper_grade"]],
                "route": item["selected_route"],
                "alpha": item["selected_alpha"],
            }
            for item in policy
        ],
        "scores": scores,
        "promotion": payload["promotion"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
