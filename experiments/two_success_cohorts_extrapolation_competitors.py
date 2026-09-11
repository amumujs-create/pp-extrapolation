#!/usr/bin/env python3
"""Figure-matched competitors on the two successful CCMR cohorts.

This is a retrospective benchmark on already-opened tests. Hyperparameters are
selected on validation only; sealed CCMR predictions are loaded unchanged.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
import types
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from engression import engression
from torch import nn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    DotProduct,
    RBF,
    WhiteKernel,
)
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "experiments"),
    str(ROOT / ".benchmark_deps"),
]

from cohort_ml_benchmark import load_cohorts
from pp_extrapolation import regression_metrics, select_affine_initialization
from pp_extrapolation.model import equal_group_weights, transform_features
from pp_extrapolation.stability_first import raw_unit_regret, regret_summary

OUT = ROOT / (
    "results/two_success_cohorts_extrapolation_competitors_v2_nonnegative"
)
SEEDS = tuple(range(42, 47))
NEURAL_ARCH = [
    (width, depth, learning_rate, weight_decay, 0.01)
    for width, depth in ((16, 1), (32, 1), (32, 2), (64, 2))
    for learning_rate in (2e-4, 5e-4, 1e-3)
    for weight_decay in (0.1, 2.0)
]
PENALTIES = (0.001, 0.01, 0.1, 1.0, 10.0)
ENGRESSION_GRID = [
    (hidden, learning_rate, beta, layers, epochs)
    for hidden in (32, 64)
    for learning_rate in (1e-3, 5e-3)
    for beta in (0.5, 1.0)
    for layers, epochs in ((2, 250), (3, 500))
]
GP_GRID = [
    (length_scale, noise)
    for length_scale in (0.3, 1.0, 3.0)
    for noise in (0.01, 0.1, 1.0)
]


SEALED = {
    "Alloy_A": (
        ROOT / "results/alloya_unopened_ccmr_v16/sealed_predictions.npz",
        "prediction",
        "CCMR_v1.6",
    ),
    "MultiStage_RPT": (
        ROOT / "results/multistage_rpt_ccmr_v19_amended2/sealed_predictions.npz",
        "deployed",
        "CCMR_v1.9",
    ),
}


def score(rows, prediction):
    prediction = np.asarray(prediction, dtype=np.float64)
    anchor = rows["x"][:, 0]
    model = regression_metrics(rows["y"], prediction, rows["groups"])
    persistence = regression_metrics(rows["y"], anchor, rows["groups"])
    model_macro = float(np.mean([
        value["rmse"] for value in model["per_unit"].values()
    ]))
    persistence_macro = float(np.mean([
        value["rmse"] for value in persistence["per_unit"].values()
    ]))
    regret = regret_summary(raw_unit_regret(
        rows["y"], rows["groups"], anchor, prediction
    ))
    return {
        "pooled": model["pooled"],
        "unit_macro_r2": model["unit_macro_r2"],
        "macro_rmse": model_macro,
        "pooled_rmse_improvement": float(
            (persistence["pooled"]["rmse"] - model["pooled"]["rmse"])
            / persistence["pooled"]["rmse"]
        ),
        "macro_rmse_improvement": float(
            (persistence_macro - model_macro) / persistence_macro
        ),
        "raw_regret": {
            "mean": regret[0],
            "cvar20": regret[1],
            "maximum": regret[2],
        },
        "coverage": 1.0,
    }


def summarize_runs(test, predictions, run_info):
    predictions = np.asarray(predictions, dtype=np.float64)
    run_scores = [score(test, prediction) for prediction in predictions]
    r2_values = np.asarray([
        item["pooled"]["r2"] for item in run_scores
    ])
    ensemble = np.mean(predictions, axis=0)
    return {
        "runs": [
            {"seed": seed, **info, "scores": scores}
            for seed, info, scores in zip(SEEDS, run_info, run_scores)
        ],
        "seed_r2_mean": float(np.mean(r2_values)),
        "seed_r2_sd": float(np.std(r2_values, ddof=1)),
        "ensemble": score(test, ensemble),
    }, ensemble


class CompetitorMLP(nn.Module):
    def __init__(self, input_dim, width, depth):
        super().__init__()
        layers = []
        for index in range(depth):
            layers.extend([
                nn.Linear(input_dim if index == 0 else width, width),
                nn.Tanh(),
            ])
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, values):
        return self.network(values).squeeze(-1)


def group_risks(error, groups):
    return torch.stack([
        error[groups == unit].mean() for unit in torch.unique(groups)
    ])


def fit_neural(kind, config, seed, parts, affine, return_test=False):
    train, validation, test = parts
    width, depth, learning_rate, weight_decay, penalty = config
    target_scale = float(affine["target_scale"])
    values = [
        torch.tensor(
            transform_features(
                part["x"], affine["center"], affine["scale"]
            ),
            dtype=torch.float32,
        )
        for part in parts
    ]
    train_x, validation_x, test_x = values
    train_y = torch.tensor(
        train["y"] / target_scale, dtype=torch.float32
    )
    validation_y = torch.tensor(
        validation["y"] / target_scale, dtype=torch.float32
    )
    labels, inverse = np.unique(train["groups"], return_inverse=True)
    group_index = torch.tensor(inverse)
    weights = torch.tensor(
        equal_group_weights(train["groups"]), dtype=torch.float32
    )
    correlation = np.corrcoef(train["x"][:, 0], train["y"])[0, 1]
    monotone_sign = 1.0 if correlation >= 0 else -1.0
    torch.manual_seed(seed)
    model = CompetitorMLP(train_x.shape[1], width, depth)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    group_weight = torch.ones(len(labels)) / len(labels)
    best = np.inf
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    rng = np.random.default_rng(seed)
    started = time.monotonic()
    for epoch in range(1, 201):
        model.train()
        order = rng.permutation(len(train_x))
        for start in range(0, len(train_x), 1024):
            batch = torch.tensor(order[start:start + 1024])
            batch_x = train_x[batch]
            if kind == "monotone":
                batch_x = batch_x.detach().requires_grad_(True)
            error = (model(batch_x) - train_y[batch]).square()
            if kind == "vrex":
                risks = group_risks(error, group_index[batch])
                loss = risks.mean() + penalty * risks.var(unbiased=False)
            elif kind == "groupdro":
                risks = group_risks(error, group_index[batch])
                present = torch.unique(group_index[batch])
                with torch.no_grad():
                    group_weight[present] *= torch.exp(
                        penalty * risks.detach()
                    )
                    group_weight /= group_weight.sum()
                loss = (
                    group_weight[present] / group_weight[present].sum()
                    * risks
                ).sum()
            else:
                estimate = model(batch_x)
                gradient = torch.autograd.grad(
                    estimate.sum(), batch_x, create_graph=True
                )[0][:, 0]
                loss = (
                    weights[batch]
                    * (estimate - train_y[batch]).square()
                ).mean() + penalty * torch.relu(
                    -monotone_sign * gradient
                ).square().mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_prediction = torch.clamp(
                model(validation_x), min=0.0
            )
            validation_mse = float(
                ((validation_prediction - validation_y) ** 2).mean()
            )
        if validation_mse < best - 1e-10:
            best = validation_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= 30:
            break
    model.load_state_dict(best_state)
    info = {
        "validation_mse": best * target_scale * target_scale,
        "selected_epoch": best_epoch,
        "epochs": epoch,
        "seconds": time.monotonic() - started,
    }
    if not return_test:
        return info, None
    model.eval()
    with torch.no_grad():
        prediction = (
            torch.clamp(model(test_x), min=0.0).numpy() * target_scale
        )
    return info, prediction


def tune_neural(kind, parts):
    affine = select_affine_initialization(parts[0], parts[1])
    search = []
    for config in NEURAL_ARCH:
        info, _ = fit_neural(kind, config, 42, parts, affine)
        search.append({"config": config, **info})
    base = min(search, key=lambda item: item["validation_mse"])["config"]
    for penalty in PENALTIES:
        config = (*base[:4], penalty)
        info, _ = fit_neural(kind, config, 42, parts, affine)
        search.append({"config": config, **info})
    selected = min(
        search, key=lambda item: item["validation_mse"]
    )["config"]
    predictions, info = [], []
    for seed in SEEDS:
        run, prediction = fit_neural(
            kind, selected, seed, parts, affine, True
        )
        predictions.append(prediction)
        info.append(run)
    result, ensemble = summarize_runs(parts[2], predictions, info)
    result.update({
        "selected_config": selected,
        "validation_search": search,
        "search_budget": len(search),
    })
    return result, ensemble


def linear_rff(parts):
    train, validation, test = parts
    affine = select_affine_initialization(train, validation)
    transformed = [
        transform_features(part["x"], affine["center"], affine["scale"])
        for part in parts
    ]

    def design(values, rff):
        return np.column_stack([
            np.ones(len(values)), values, rff.transform(values)
        ])

    search = []
    for gamma in (0.03, 0.1, 0.3, 1.0):
        for alpha in (0.1, 1.0, 10.0):
            rff = RBFSampler(
                gamma=gamma, n_components=384, random_state=42
            ).fit(transformed[0])
            model = Ridge(alpha=alpha, fit_intercept=False).fit(
                design(transformed[0], rff), train["y"]
            )
            estimate = np.maximum(
                model.predict(design(transformed[1], rff)), 0.0
            )
            search.append({
                "gamma": gamma,
                "alpha": alpha,
                "validation_mse": float(np.mean(
                    (estimate - validation["y"]) ** 2
                )),
            })
    selected = min(search, key=lambda item: item["validation_mse"])
    predictions = []
    for seed in SEEDS:
        rff = RBFSampler(
            gamma=selected["gamma"], n_components=384, random_state=seed
        ).fit(transformed[0])
        model = Ridge(
            alpha=selected["alpha"], fit_intercept=False
        ).fit(design(transformed[0], rff), train["y"])
        predictions.append(np.maximum(
            model.predict(design(transformed[2], rff)), 0.0
        ))
    result, ensemble = summarize_runs(
        test, predictions, [{} for _ in SEEDS]
    )
    result.update({
        "selected_config": {
            "gamma": selected["gamma"],
            "alpha": selected["alpha"],
            "rff_components": 384,
        },
        "validation_search": search,
    })
    return result, ensemble


def engression_fit_predict(train, validation, test, config, seed):
    x = torch.tensor(train["x"], dtype=torch.float32)
    y = torch.tensor(train["y"][:, None], dtype=torch.float32)
    validation_x = torch.tensor(
        validation["x"], dtype=torch.float32
    )
    test_x = torch.tensor(test["x"], dtype=torch.float32)
    hidden, learning_rate, beta, layers, epochs = config
    torch.manual_seed(seed)
    model = engression(
        x,
        y,
        num_layer=layers,
        hidden_dim=hidden,
        noise_dim=32,
        beta=beta,
        lr=learning_rate,
        num_epoches=epochs,
        batch_size=min(512, len(x)),
        device="cpu",
        standardize=True,
        verbose=False,
    )
    with torch.no_grad():
        validation_prediction = model.predict(
            validation_x, target="mean", sample_size=50
        ).squeeze().cpu().numpy()
        test_prediction = model.predict(
            test_x, target="mean", sample_size=100
        ).squeeze().cpu().numpy()
    validation_prediction = np.maximum(validation_prediction, 0.0)
    test_prediction = np.maximum(test_prediction, 0.0)
    mse = float(np.mean(
        (validation_prediction - validation["y"]) ** 2
    ))
    return mse, test_prediction


def run_engression(parts):
    train, validation, test = parts
    search = []
    for config in ENGRESSION_GRID:
        mse, _ = engression_fit_predict(
            train, validation, test, config, 42
        )
        search.append({"config": config, "validation_mse": mse})
    selected = min(
        search, key=lambda item: item["validation_mse"]
    )["config"]
    predictions, run_info = [], []
    for seed in SEEDS:
        started = time.monotonic()
        mse, prediction = engression_fit_predict(
            train, validation, test, selected, seed
        )
        predictions.append(prediction)
        run_info.append({
            "validation_mse": mse,
            "seconds": time.monotonic() - started,
        })
    result, ensemble = summarize_runs(test, predictions, run_info)
    result.update({
        "package_version": version("engression"),
        "selected_config": selected,
        "validation_search": search,
    })
    return result, ensemble


def gp_fit_predict(train, validation, test, config, seed):
    rng = np.random.default_rng(seed)
    index = np.arange(len(train["y"]))
    if len(index) > 750:
        index = np.sort(rng.choice(index, 750, replace=False))
    x = train["x"][index].astype(float)
    y = train["y"][index].astype(float)
    center = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-8)
    target_center = float(y.mean())
    target_scale = max(float(y.std()), 1e-8)
    kernel = (
        DotProduct(sigma_0=1.0)
        + ConstantKernel(1.0) * RBF(length_scale=config[0])
        + WhiteKernel(noise_level=config[1])
    )
    model = GaussianProcessRegressor(
        kernel=kernel, optimizer=None, normalize_y=False
    ).fit((x - center) / scale, (y - target_center) / target_scale)

    def predict(values):
        estimate = (
            model.predict((values - center) / scale) * target_scale
            + target_center
        )
        return np.maximum(estimate, 0.0)

    validation_prediction = predict(validation["x"])
    return (
        float(np.mean(
            (validation_prediction - validation["y"]) ** 2
        )),
        predict(test["x"]),
    )


def run_gp(parts):
    train, validation, test = parts
    search = []
    for config in GP_GRID:
        mse, _ = gp_fit_predict(
            train, validation, test, config, 42
        )
        search.append({"config": config, "validation_mse": mse})
    selected = min(
        search, key=lambda item: item["validation_mse"]
    )["config"]
    predictions, info = [], []
    for seed in SEEDS:
        mse, prediction = gp_fit_predict(
            train, validation, test, selected, seed
        )
        predictions.append(prediction)
        info.append({"validation_mse": mse})
    result, ensemble = summarize_runs(test, predictions, info)
    result.update({
        "selected_config": selected,
        "validation_search": search,
        "max_train_rows": 750,
    })
    return result, ensemble


def import_tabpfn():
    fake_mlx = types.ModuleType("mlx")
    fake_mlx_core = types.ModuleType("mlx.core")
    fake_mlx.core = fake_mlx_core
    sys.modules.setdefault("mlx", fake_mlx)
    sys.modules.setdefault("mlx.core", fake_mlx_core)
    os.environ.setdefault("TABPFN_DEVICE", "cpu")
    os.environ.setdefault("TORCH_DEVICE", "cpu")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    return TabPFNRegressor, ModelVersion


def run_tabpfn(parts):
    TabPFNRegressor, ModelVersion = import_tabpfn()
    train, validation, test = parts
    predictions, info = [], []
    for seed in SEEDS:
        started = time.monotonic()
        model = TabPFNRegressor.create_default_for_version(
            ModelVersion("v3"),
            device="cpu",
            n_estimators=1,
            random_state=seed,
            ignore_pretraining_limits=True,
        )
        model.fit(train["x"].astype(np.float32), train["y"])
        validation_prediction = np.maximum(
            model.predict(validation["x"].astype(np.float32)),
            0.0,
        )
        chunks = [
            model.predict(test["x"][start:start + 256].astype(np.float32))
            for start in range(0, len(test["x"]), 256)
        ]
        predictions.append(np.maximum(np.concatenate(chunks), 0.0))
        info.append({
            "validation_mse": float(np.mean(
                (validation_prediction - validation["y"]) ** 2
            )),
            "seconds": time.monotonic() - started,
        })
    result, ensemble = summarize_runs(test, predictions, info)
    result.update({
        "model_version": "v3",
        "package_version": version("tabpfn"),
        "n_estimators": 1,
        "device": "cpu",
        "train_rows": len(train["y"]),
    })
    return result, ensemble


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "results.json"
    if result_path.exists():
        raise RuntimeError("refusing to overwrite competitor benchmark")
    all_cohorts = load_cohorts()
    cohorts = {
        name: all_cohorts[name]
        for name in ("Alloy_A", "MultiStage_RPT")
    }
    result = {
        "status": "retrospective opened-test benchmark",
        "confirmatory": False,
        "selection": "validation only",
        "output_contract": (
            "nonnegative predictions; no train-range upper clipping"
        ),
        "seeds": list(SEEDS),
        "models": [
            "successful latest CCMR",
            "V-REx",
            "GroupDRO",
            "Monotone NN",
            "Linear-tail RBF",
            "Engression",
            "Linear-mean GP",
            "TabPFN v3",
        ],
        "cohorts": {},
    }
    for cohort_name, mapping in cohorts.items():
        parts = (
            mapping["train"], mapping["validation"], mapping["test"]
        )
        sealed_path, prediction_key, version = SEALED[cohort_name]
        sealed = np.load(sealed_path)
        if not np.allclose(sealed["truth"], mapping["test"]["y"]):
            raise RuntimeError(f"{cohort_name}: sealed truth misalignment")
        if not np.array_equal(
            sealed["groups"].astype(str),
            mapping["test"]["groups"].astype(str),
        ):
            raise RuntimeError(f"{cohort_name}: sealed groups misalignment")
        cohort_result = {
            "rows": {
                key: len(value["y"]) for key, value in mapping.items()
            },
            "units": {
                key: len(np.unique(value["groups"]))
                for key, value in mapping.items()
            },
            "models": {
                "persistence": {
                    "ensemble": score(
                        mapping["test"], mapping["test"]["x"][:, 0]
                    )
                },
                "PP_latest_successful": {
                    "version": version,
                    "source": str(sealed_path.relative_to(ROOT)),
                    "ensemble": score(
                        mapping["test"], sealed[prediction_key]
                    ),
                },
            },
        }
        prediction_artifacts = {
            "truth": mapping["test"]["y"],
            "groups": mapping["test"]["groups"].astype(str),
            "persistence": mapping["test"]["x"][:, 0],
            "PP_latest_successful": sealed[prediction_key],
        }
        for kind, label in (
            ("vrex", "V-REx"),
            ("groupdro", "GroupDRO"),
            ("monotone", "Monotone_NN"),
        ):
            started = time.monotonic()
            model_result, prediction = tune_neural(kind, parts)
            model_result["total_seconds"] = time.monotonic() - started
            cohort_result["models"][label] = model_result
            prediction_artifacts[label] = prediction
            print(cohort_name, label, flush=True)
        for label, runner in (
            ("Linear_tail_RBF", linear_rff),
            ("Engression", run_engression),
            ("Linear_mean_GP", run_gp),
            ("TabPFN_v3", run_tabpfn),
        ):
            started = time.monotonic()
            model_result, prediction = runner(parts)
            model_result["total_seconds"] = time.monotonic() - started
            cohort_result["models"][label] = model_result
            prediction_artifacts[label] = prediction
            print(cohort_name, label, flush=True)
        result["cohorts"][cohort_name] = cohort_result
        np.savez_compressed(
            OUT / f"{cohort_name.lower()}_ensemble_predictions.npz",
            **prediction_artifacts,
        )
        result_path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
