#!/usr/bin/env python3
"""Full-train sparse variational GP baseline on the fixed MICH PP split."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import gpytorch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments"), str(ROOT.parent / "ca-css-ncmapss")]

from extrapolation_competitors_all import datasets
from pp_extrapolation import regression_metrics, select_affine_initialization
from pp_extrapolation.model import transform_features

OUT = ROOT / "results/final_svgp_extension_v1"
SEEDS = (42, 43, 44, 45, 46)
GRID = [(lr, length, noise) for lr in (0.003, 0.01, 0.03)
        for length in (0.3, 1.0, 3.0, 10.0, 30.0) for noise in (0.01, 0.1)]


class SVGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing: torch.Tensor, length: float):
        distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing.size(0))
        strategy = gpytorch.variational.VariationalStrategy(self, inducing, distribution, learn_inducing_locations=True)
        super().__init__(strategy)
        self.mean_module = gpytorch.means.LinearMean(inducing.size(1))
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module.base_kernel.lengthscale = float(length)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


def prepare(parts):
    affine = select_affine_initialization(parts[0], parts[1])
    x = [transform_features(p["x"], affine["center"], affine["scale"]).astype(np.float32) for p in parts]
    y = [np.asarray(p["y"], np.float32) for p in parts]
    return affine, x, y


def fit_predict(parts, cfg, seed: int):
    affine, arrays, targets = prepare(parts)
    train_x, valid_x, test_x = (torch.tensor(x) for x in arrays)
    train_y, valid_y = (torch.tensor(y) for y in targets[:2])
    y_mean, y_scale = train_y.mean(), train_y.std().clamp_min(1e-6)
    scaled_y = (train_y - y_mean) / y_scale
    generator = torch.Generator().manual_seed(seed)
    n_inducing = min(128, len(train_x))
    inducing = train_x[torch.randperm(len(train_x), generator=generator)[:n_inducing]].clone()
    lr, length, noise = cfg
    model = SVGP(inducing, length)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = float(noise)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=float(lr))
    objective = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(train_x))
    best_loss, best_state, stale = float("inf"), None, 0
    torch.manual_seed(seed)
    for epoch in range(180):
        model.train(); likelihood.train()
        optimizer.zero_grad()
        loss = -objective(model(train_x), scaled_y)
        loss.backward(); optimizer.step()
        model.eval(); likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            val = likelihood(model(valid_x)).mean * y_scale + y_mean
            score = float(torch.mean((val - valid_y).square()))
        if score < best_loss - 1e-8:
            best_loss, best_state, stale = score, copy.deepcopy((model.state_dict(), likelihood.state_dict())), 0
        else:
            stale += 1
        if stale >= 30:
            break
    model.load_state_dict(best_state[0]); likelihood.load_state_dict(best_state[1]); model.eval(); likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        prediction = likelihood(model(test_x)).mean.numpy() * float(y_scale) + float(y_mean)
    prediction = np.clip(prediction, 0.0, float(affine["target_scale"]))
    return best_loss, prediction, int(epoch + 1), n_inducing


def main() -> None:
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    parts = datasets()["mich"]
    candidates = []
    for cfg in GRID:
        loss, _, epochs, inducing = fit_predict(parts, cfg, 42)
        candidates.append({"config": cfg, "validation_mse": loss, "epochs": epochs, "inducing_points": inducing})
        print(cfg, loss, flush=True)
    chosen = min(candidates, key=lambda row: row["validation_mse"])["config"]
    predictions, runs = [], []
    for seed in SEEDS:
        loss, prediction, epochs, inducing = fit_predict(parts, chosen, seed)
        predictions.append(prediction)
        runs.append({"seed": seed, "validation_mse": loss, "epochs": epochs, "inducing_points": inducing,
                     "metrics": regression_metrics(parts[2]["y"], prediction, parts[2]["groups"])})
    ensemble = regression_metrics(parts[2]["y"], np.mean(predictions, axis=0), parts[2]["groups"])
    payload = {"protocol": "full-train SVGP; 30 validation candidates; 5 refits; 128 inducing-point maximum",
               "selected_config": chosen, "search": candidates, "runs": runs, "ensemble": ensemble}
    (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(OUT / "predictions.npz", prediction=np.asarray(predictions), y=parts[2]["y"], groups=parts[2]["groups"])
    print(ensemble["pooled"]["r2"], flush=True)


if __name__ == "__main__":
    main()
