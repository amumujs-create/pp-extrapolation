"""Strong no-prior anchor portfolio for CCMR v2.3 AC-CRPE."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from torch import nn

from .model import equal_group_weights
from .stability_first import raw_unit_regret, regret_summary


class _Predictor(Protocol):
    def predict(self, x: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class PredictorPortfolioFit:
    names: tuple[str, ...]
    predictors: tuple[_Predictor, ...]
    weights: np.ndarray
    validation_objective: float
    validation_macro_rmse: float
    validation_worst_rmse: float
    unavailable: tuple[str, ...]
    anchor_index: int
    support_center: np.ndarray
    support_scale: np.ndarray
    support_prototypes: np.ndarray
    support_threshold: float


@dataclass(frozen=True)
class _Persistence:
    anchor_index: int

    def predict(self, x):
        return np.asarray(x, dtype=float)[:, self.anchor_index].copy()


@dataclass(frozen=True)
class _Linear:
    center: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float

    def predict(self, x):
        z = (np.asarray(x, dtype=float) - self.center) / self.scale
        return z @ self.coefficient + self.intercept


@dataclass(frozen=True)
class _RFF:
    center: np.ndarray
    scale: np.ndarray
    frequencies: np.ndarray
    phases: np.ndarray
    coefficient: np.ndarray
    intercept: float

    def _features(self, x):
        z = (np.asarray(x, dtype=float) - self.center) / self.scale
        rff = np.sqrt(2.0 / self.frequencies.shape[1]) * np.cos(
            z @ self.frequencies + self.phases
        )
        return np.column_stack([z, rff])

    def predict(self, x):
        return self._features(x) @ self.coefficient + self.intercept


class _TorchMLP(nn.Module):
    def __init__(self, dimension, width=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dimension, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


@dataclass
class _MLPEnsemble:
    center: np.ndarray
    scale: np.ndarray
    target_center: float
    target_scale: float
    models: tuple[_TorchMLP, ...]

    def predict(self, x):
        values = torch.as_tensor(
            (np.asarray(x, dtype=float) - self.center) / self.scale,
            dtype=torch.float32,
        )
        predictions = []
        with torch.no_grad():
            for model in self.models:
                predictions.append(
                    model(values).cpu().numpy() * self.target_scale
                    + self.target_center
                )
        return np.mean(predictions, axis=0)


@dataclass
class _EngressionEnsemble:
    models: tuple[object, ...]
    seeds: tuple[int, ...]
    sample_size: int = 30

    def predict(self, x):
        predictions = []
        for model, seed in zip(self.models, self.seeds):
            predictions.append(_engression_predict(
                model, x, self.sample_size, seed + 991
            ))
        return np.mean(predictions, axis=0)


def _scale(x):
    center = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    return center, np.where(scale > 1e-8, scale, 1.0)


def _support_columns(dimension, anchor_index):
    return np.asarray([
        index for index in range(dimension) if index != anchor_index
    ])


def _support_fit(train_x, validation_x, anchor_index):
    columns = _support_columns(train_x.shape[1], anchor_index)
    train = train_x[:, columns]
    validation = validation_x[:, columns]
    center = np.median(train, axis=0)
    scale = np.quantile(train, 0.75, axis=0) - np.quantile(
        train, 0.25, axis=0
    )
    scale = np.where(scale > 1e-8, scale, np.std(train, axis=0))
    scale = np.where(scale > 1e-8, scale, 1.0)
    standardized = (train - center) / scale
    if len(standardized) > 5000:
        index = np.round(np.linspace(
            0, len(standardized) - 1, 5000
        )).astype(int)
        standardized = standardized[index]
    validation_distance = cKDTree(standardized).query(
        (validation - center) / scale, k=1
    )[0] / np.sqrt(len(columns))
    threshold = float(np.quantile(validation_distance, 0.99))
    return center, scale, standardized, max(threshold, 1e-12)


def _balanced_cap(x, y, groups, maximum, seed=20260911):
    if len(y) <= maximum:
        return x, y, groups
    rng = np.random.default_rng(seed)
    labels = np.unique(groups)
    per_group = max(1, maximum // len(labels))
    chosen = []
    for label in labels:
        rows = np.flatnonzero(groups == label)
        count = min(len(rows), per_group)
        chosen.extend(rng.choice(rows, count, replace=False))
    if len(chosen) < maximum:
        remaining = np.setdiff1d(np.arange(len(y)), chosen)
        count = min(maximum - len(chosen), len(remaining))
        chosen.extend(rng.choice(remaining, count, replace=False))
    chosen = np.sort(np.asarray(chosen[:maximum], dtype=int))
    return x[chosen], y[chosen], groups[chosen]


def _weighted_ridge(x, y, groups, alpha):
    weights = equal_group_weights(groups)
    total = float(np.sum(weights))
    x_mean = np.sum(weights[:, None] * x, axis=0) / total
    y_mean = float(np.sum(weights * y) / total)
    centered = x - x_mean
    coefficient = np.linalg.solve(
        centered.T @ (weights[:, None] * centered)
        + float(alpha) * np.eye(x.shape[1]),
        centered.T @ (weights * (y - y_mean)),
    )
    return coefficient, y_mean - float(x_mean @ coefficient)


def _fit_linear(train_x, train_y, train_groups, validation_x, validation_y):
    center, scale = _scale(train_x)
    z = (train_x - center) / scale
    candidates = []
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
        coefficient, intercept = _weighted_ridge(
            z, train_y, train_groups, alpha
        )
        prediction = (
            (validation_x - center) / scale
        ) @ coefficient + intercept
        candidates.append((
            float(np.mean((validation_y - prediction) ** 2)),
            alpha,
            coefficient,
            intercept,
        ))
    _, _, coefficient, intercept = min(candidates, key=lambda row: row[:2])
    return _Linear(center, scale, coefficient, float(intercept))


def _fit_rff(train_x, train_y, train_groups, validation_x, validation_y):
    center, scale = _scale(train_x)
    rng = np.random.default_rng(20260911)
    frequencies = rng.normal(size=(train_x.shape[1], 128))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=128)
    shell = _RFF(
        center, scale, frequencies, phases,
        np.zeros(train_x.shape[1] + 128), 0.0,
    )
    train_features = shell._features(train_x)
    validation_features = shell._features(validation_x)
    candidates = []
    for alpha in (0.1, 1.0, 10.0, 100.0):
        coefficient, intercept = _weighted_ridge(
            train_features, train_y, train_groups, alpha
        )
        prediction = validation_features @ coefficient + intercept
        candidates.append((
            float(np.mean((validation_y - prediction) ** 2)),
            alpha,
            coefficient,
            intercept,
        ))
    _, _, coefficient, intercept = min(candidates, key=lambda row: row[:2])
    return _RFF(
        center, scale, frequencies, phases, coefficient, float(intercept)
    )


def _fit_mlp(
    train_x, train_y, train_groups, validation_x, validation_y, seeds
):
    train_x, train_y, train_groups = _balanced_cap(
        train_x, train_y, train_groups, 10_000
    )
    center, scale = _scale(train_x)
    target_center = float(np.mean(train_y))
    target_scale = max(float(np.std(train_y)), 1e-8)
    x = torch.as_tensor((train_x - center) / scale, dtype=torch.float32)
    y = torch.as_tensor(
        (train_y - target_center) / target_scale, dtype=torch.float32
    )
    validation_index = np.arange(len(validation_y))
    if len(validation_index) > 10_000:
        validation_index = np.round(np.linspace(
            0, len(validation_index) - 1, 10_000
        )).astype(int)
    vx = torch.as_tensor(
        (validation_x[validation_index] - center) / scale,
        dtype=torch.float32,
    )
    checkpoint_y = validation_y[validation_index]
    weights = torch.as_tensor(
        equal_group_weights(train_groups), dtype=torch.float32
    )
    models = []
    for seed in seeds:
        torch.manual_seed(int(seed))
        model = _TorchMLP(train_x.shape[1])
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-3, weight_decay=0.1
        )
        best, best_epoch = float("inf"), 0
        state = copy.deepcopy(model.state_dict())
        for epoch in range(1, 251):
            model.train()
            estimate = model(x)
            loss = torch.mean(weights * (estimate - y).square())
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    prediction = (
                        model(vx).numpy() * target_scale + target_center
                    )
                value = float(np.mean((checkpoint_y - prediction) ** 2))
                if value < best - 1e-10:
                    best, best_epoch = value, epoch
                    state = copy.deepcopy(model.state_dict())
                if epoch - best_epoch >= 40:
                    break
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return _MLPEnsemble(
        center, scale, target_center, target_scale, tuple(models)
    )


def _fit_engression(
    train_x, train_y, validation_x, validation_y, seeds, max_train_rows
):
    from engression import engression

    candidates = (
        (32, 5e-3, 1.0, 2, 150),
        (64, 1e-3, 0.5, 2, 200),
    )
    rng = np.random.default_rng(20260911)
    index = np.arange(len(train_y))
    if len(index) > max_train_rows:
        index = np.sort(rng.choice(index, max_train_rows, replace=False))
    x = torch.as_tensor(train_x[index], dtype=torch.float32)
    y = torch.as_tensor(train_y[index, None], dtype=torch.float32)
    search = []
    for config in candidates:
        hidden, learning_rate, beta, layers, epochs = config
        torch.manual_seed(int(seeds[0]))
        model = engression(
            x, y, num_layer=layers, hidden_dim=hidden, noise_dim=32,
            beta=beta, lr=learning_rate, num_epochs=epochs,
            batch_size=min(512, len(x)), device="cpu", standardize=True,
            verbose=False,
        )
        prediction = _engression_predict(
            model, validation_x, 20, int(seeds[0]) + 991
        )
        search.append((
            float(np.mean((validation_y - prediction) ** 2)), config
        ))
    selected = min(search, key=lambda row: row[0])[1]
    models = []
    for seed in seeds:
        hidden, learning_rate, beta, layers, epochs = selected
        torch.manual_seed(int(seed))
        models.append(engression(
            x, y, num_layer=layers, hidden_dim=hidden, noise_dim=32,
            beta=beta, lr=learning_rate, num_epochs=epochs,
            batch_size=min(512, len(x)), device="cpu", standardize=True,
            verbose=False,
        ))
    return _EngressionEnsemble(tuple(models), tuple(map(int, seeds)))


def _engression_predict(model, x, sample_size, seed, chunk_size=2048):
    x = np.asarray(x, dtype=np.float32)
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), chunk_size):
            torch.manual_seed(int(seed) + start)
            values = torch.as_tensor(
                x[start:start + chunk_size], dtype=torch.float32
            )
            value = model.predict(
                values, target="mean", sample_size=int(sample_size)
            )
            chunks.append(
                np.asarray(value.squeeze().cpu().numpy()).reshape(-1)
            )
    return np.concatenate(chunks)


def _risk_objective(y, prediction, groups):
    unit_rmse = np.asarray([
        np.sqrt(np.mean((y[groups == label] - prediction[groups == label]) ** 2))
        for label in np.unique(groups)
    ])
    macro = float(np.mean(unit_rmse))
    worst = float(np.max(unit_rmse))
    return macro + 0.20 * worst, macro, worst


def fit_predictor_portfolio(
    train_x,
    train_y,
    train_groups,
    validation_x,
    validation_y,
    validation_groups,
    *,
    anchor_index=0,
    seeds=(42, 43, 44),
    include_engression=True,
    max_engression_rows=3000,
    validation_mean_cap=0.0,
    validation_cvar_cap=0.01,
    validation_max_cap=0.02,
):
    """Fit a validation-selected strong anchor without test information."""
    train_x = np.asarray(train_x, dtype=float)
    train_y = np.asarray(train_y, dtype=float)
    train_groups = np.asarray(train_groups)
    validation_x = np.asarray(validation_x, dtype=float)
    validation_y = np.asarray(validation_y, dtype=float)
    validation_groups = np.asarray(validation_groups)
    if train_x.ndim != 2 or validation_x.shape[1] != train_x.shape[1]:
        raise ValueError("train and validation x must be aligned matrices")
    if len(np.unique(train_groups)) < 2:
        raise ValueError("at least two training groups are required")
    names = ["persistence", "ridge", "linear_tail_rff", "mlp_ensemble"]
    predictors: list[_Predictor] = [
        _Persistence(int(anchor_index)),
        _fit_linear(
            train_x, train_y, train_groups, validation_x, validation_y
        ),
        _fit_rff(
            train_x, train_y, train_groups, validation_x, validation_y
        ),
        _fit_mlp(
            train_x, train_y, train_groups, validation_x, validation_y,
            tuple(seeds),
        ),
    ]
    unavailable = []
    if include_engression:
        try:
            predictors.append(_fit_engression(
                train_x, train_y, validation_x, validation_y,
                tuple(seeds), int(max_engression_rows),
            ))
            names.append("engression_ensemble")
        except (ImportError, ModuleNotFoundError) as error:
            unavailable.append(f"engression: {error}")
    matrix = np.column_stack([
        predictor.predict(validation_x) for predictor in predictors
    ])
    count = matrix.shape[1]

    def objective(weights):
        return _risk_objective(
            validation_y, matrix @ weights, validation_groups
        )[0]

    persistence = matrix[:, 0]
    candidates = []

    def add_if_feasible(weights):
        summary = regret_summary(raw_unit_regret(
            validation_y, validation_groups, persistence,
            matrix @ weights,
        ))
        if (
            summary[0] <= validation_mean_cap + 1e-12
            and summary[1] <= validation_cvar_cap + 1e-12
            and summary[2] <= validation_max_cap + 1e-12
        ):
            candidates.append((objective(weights), weights))

    for index in range(count):
        target = np.zeros(count)
        target[index] = 1.0
        for mass in np.linspace(0.0, 1.0, 21):
            weights = np.zeros(count)
            weights[0] = 1.0 - mass
            weights += mass * target
            add_if_feasible(weights)
    result = minimize(
        objective,
        np.ones(count) / count,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints={"type": "eq", "fun": lambda value: value.sum() - 1.0},
        options={"maxiter": 200, "ftol": 1e-10},
    )
    if result.success:
        weights = np.maximum(result.x, 0.0)
        weights /= weights.sum()
        for mass in np.linspace(0.0, 1.0, 21):
            shrunk = weights * mass
            shrunk[0] += 1.0 - mass
            add_if_feasible(shrunk)
    _, selected = min(candidates, key=lambda row: row[0])
    prediction = matrix @ selected
    risk, macro, worst = _risk_objective(
        validation_y, prediction, validation_groups
    )
    support = _support_fit(
        train_x, validation_x, int(anchor_index)
    )
    return PredictorPortfolioFit(
        tuple(names), tuple(predictors), selected, risk, macro, worst,
        tuple(unavailable), int(anchor_index), *support,
    )


def predict_predictor_portfolio(model, x):
    x = np.asarray(x, dtype=float)
    matrix = np.column_stack([
        predictor.predict(x) for predictor in model.predictors
    ])
    prediction = matrix @ model.weights
    columns = _support_columns(x.shape[1], model.anchor_index)
    distance = cKDTree(model.support_prototypes).query(
        (x[:, columns] - model.support_center) / model.support_scale,
        k=1,
    )[0] / np.sqrt(len(columns))
    return np.where(
        distance <= model.support_threshold,
        prediction,
        x[:, model.anchor_index],
    )
