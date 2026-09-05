"""Select a prior on inner train-only pseudo-extrapolation folds.

This is post-hoc method development on previously inspected NASA health-v2 data.
The outer test labels are read only after all four fold decisions are saved.
"""
import argparse
import hashlib
import itertools
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from pp_extrapolation import fit_pp, predict, regression_metrics, select_prior_from_scores
from pp_extrapolation.hull import audit_convex_hull_support
from nasa_prior_ablation import ARMS, SEEDS, priors


SELECTOR_SEED = 20260906
PSEUDO_CUTOFF = 0.75
MINIMUM_GAIN = 0.01
PRIOR_WEIGHT = 10.0


def load_folds(path):
    archive = np.load(path, allow_pickle=False)
    return [
        {
            part: {
                key: archive[f"f{index}_{part}_{key}"]
                for key in ("x", "y", "groups")
            }
            for part in ("train", "validation", "test")
        }
        for index in range(4)
    ]


def subset(data, mask):
    return {key: value[mask] for key, value in data.items()}


def pseudo_folds(outer_train):
    result = []
    groups = np.unique(outer_train["groups"])
    if len(groups) != 2:
        raise ValueError("this predeclared NASA gate requires two outer-train cells")
    for held_out in groups:
        train_mask = (outer_train["groups"] != held_out) & (
            outer_train["x"][:, 0] >= PSEUDO_CUTOFF
        )
        pseudo_train = subset(outer_train, train_mask)
        boundary = float(pseudo_train["x"][:, 0].min())
        validation_mask = (outer_train["groups"] == held_out) & (
            outer_train["x"][:, 0] < boundary
        )
        pseudo_validation = subset(outer_train, validation_mask)
        if min(len(pseudo_train["y"]), len(pseudo_validation["y"])) == 0:
            raise ValueError("empty pseudo-extrapolation fold")
        audit = audit_convex_hull_support(
            pseudo_train["x"], pseudo_validation["x"]
        ).summary()
        if audit["outside_fraction"] != 1.0:
            raise RuntimeError("pseudo-validation is not strictly hull-out")
        result.append(
            {
                "held_out": str(held_out),
                "train": pseudo_train,
                "validation": pseudo_validation,
                "boundary": boundary,
                "audit": audit,
            }
        )
    return result


def arm_kwargs(arm, train):
    pair, triple = priors(train)
    kwargs = {}
    if arm in ("direction", "both"):
        kwargs.update(prior_pairs=pair, prior_weight=PRIOR_WEIGHT)
    if arm in ("transport", "both"):
        kwargs.update(transport_triples=triple, transport_weight=PRIOR_WEIGHT)
    return kwargs


def score_arm(arm, inner_folds):
    truth, prediction, groups, selections = [], [], [], []
    for fold in inner_folds:
        fit = fit_pp(
            fold["train"],
            fold["validation"],
            seed=SELECTOR_SEED,
            **arm_kwargs(arm, fold["train"]),
        )
        truth.append(fold["validation"]["y"])
        groups.append(fold["validation"]["groups"])
        prediction.append(predict(fit, fold["validation"]["x"]))
        selections.append(fit.selection)
    metric = regression_metrics(
        np.concatenate(truth), np.concatenate(prediction), np.concatenate(groups)
    )
    return metric, selections


def compose_predictions(candidate_archive, decisions):
    truth = candidate_archive["truth"]
    groups = candidate_archive["groups"].astype(str)
    cells = ["B0005", "B0006", "B0007", "B0018"]
    runs = []
    for seed in SEEDS:
        prediction = np.empty(len(truth), dtype=np.float64)
        for cell, decision in zip(cells, decisions):
            mask = groups == cell
            prediction[mask] = candidate_archive[
                f"{decision.selected}_{seed}"
            ][mask]
        runs.append(
            {
                "seed": seed,
                "metrics": regression_metrics(truth, prediction, groups),
                "prediction": prediction,
            }
        )
    return runs


def shuffled_control(candidate_archive, fold_scores, repeats=10000):
    rng = np.random.default_rng(20260906)
    values = []
    arms = list(ARMS)
    for _ in range(repeats):
        decisions = []
        for scores in fold_scores:
            shuffled = dict(zip(arms, rng.permutation([scores[a] for a in arms])))
            decisions.append(
                select_prior_from_scores(
                    shuffled, minimum_gain=MINIMUM_GAIN
                )
            )
        runs = compose_predictions(candidate_archive, decisions)
        values.append(np.mean([run["metrics"]["pooled"]["r2"] for run in runs]))
    return {
        "repeats": repeats,
        "seed": 20260906,
        "mean": float(np.mean(values)),
        "sd": float(np.std(values)),
        "q025": float(np.quantile(values, 0.025)),
        "q975": float(np.quantile(values, 0.975)),
        "values": values,
    }


def exhaustive_policy_distribution(candidate_archive):
    values = []
    for choice in itertools.product(ARMS, repeat=4):
        decisions = [
            select_prior_from_scores(
                {"PP": 0.0, arm: 1.0} if arm != "PP" else {"PP": 1.0, "direction": 0.0},
                minimum_gain=0.01,
            )
            for arm in choice
        ]
        runs = compose_predictions(candidate_archive, decisions)
        values.append(
            {
                "choice": list(choice),
                "mean_seed_pooled_r2": float(
                    np.mean([run["metrics"]["pooled"]["r2"] for run in runs])
                ),
            }
        )
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", required=True)
    parser.add_argument("--candidate-predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)

    protocol = {
        "status": "post_hoc_method_development_on_reused_NASA_health_v2",
        "selector_seed": SELECTOR_SEED,
        "pseudo_cutoff": PSEUDO_CUTOFF,
        "minimum_pooled_r2_gain_to_enable_prior": MINIMUM_GAIN,
        "candidate_arms": list(ARMS),
        "prior_weight": PRIOR_WEIGHT,
        "selection_data": "outer-train cells only",
        "test_labels_used_for_gate": False,
        "fold_archive_sha256": hashlib.sha256(Path(args.folds).read_bytes()).hexdigest(),
        "candidate_prediction_sha256": hashlib.sha256(
            Path(args.candidate_predictions).read_bytes()
        ).hexdigest(),
    }
    (output / "PROTOCOL.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )

    folds = load_folds(args.folds)
    fold_scores, decisions, inner_audits = [], [], []
    for outer_index, fold in enumerate(folds):
        inner = pseudo_folds(fold["train"])
        scores, details = {}, {}
        for arm in ARMS:
            metric, selections = score_arm(arm, inner)
            scores[arm] = metric["pooled"]["r2"]
            details[arm] = {"metrics": metric, "selections": selections}
        decision = select_prior_from_scores(
            scores, minimum_gain=MINIMUM_GAIN
        )
        fold_scores.append(scores)
        decisions.append(decision)
        inner_audits.append(
            {
                "outer_fold": outer_index,
                "pseudo_folds": [
                    {
                        "held_out": row["held_out"],
                        "boundary": row["boundary"],
                        "n_train": len(row["train"]["y"]),
                        "n_validation": len(row["validation"]["y"]),
                        "audit": row["audit"],
                    }
                    for row in inner
                ],
                "scores": scores,
                "details": details,
                "decision": asdict(decision),
            }
        )
        print(outer_index, scores, "selected", decision.selected, flush=True)

    candidates = np.load(args.candidate_predictions, allow_pickle=False)
    gated_runs = compose_predictions(candidates, decisions)
    shuffled = shuffled_control(candidates, fold_scores)
    policies = exhaustive_policy_distribution(candidates)
    observed = float(
        np.mean([run["metrics"]["pooled"]["r2"] for run in gated_runs])
    )
    shuffled_values = np.asarray(shuffled.pop("values"))
    payload = {
        "protocol": protocol,
        "inner_audits": inner_audits,
        "selected_arms": [decision.selected for decision in decisions],
        "gated_runs": [
            {"seed": run["seed"], "metrics": run["metrics"]}
            for run in gated_runs
        ],
        "gated_mean_seed_pooled_r2": observed,
        "gated_seed_sd": float(
            np.std([run["metrics"]["pooled"]["r2"] for run in gated_runs])
        ),
        "shuffled_score_control": {
            **shuffled,
            "fraction_at_or_below_gated": float(np.mean(shuffled_values <= observed)),
        },
        "exhaustive_256_arm_policies": policies,
    }
    (output / "results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        output / "gated_predictions.npz",
        truth=candidates["truth"],
        groups=candidates["groups"],
        **{f"prediction_seed{run['seed']}": run["prediction"] for run in gated_runs},
    )
    print("gated mean", observed, "selected", payload["selected_arms"])


if __name__ == "__main__":
    main()
