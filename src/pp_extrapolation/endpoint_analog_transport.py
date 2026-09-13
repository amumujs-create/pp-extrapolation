"""Endpoint-conditioned analog transport with rejectable structural experts."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class TransportCandidate:
    name: str
    kind: str
    parameters: dict


@dataclass(frozen=True)
class TransportPolicy:
    candidates: tuple[TransportCandidate, ...]
    weights: tuple[float, ...]
    validation_macro_mse: float
    validation_cvar20_mse: float
    validation_max_mse: float


def candidate_grid() -> tuple[TransportCandidate, ...]:
    candidates = []
    for alpha in (1.0, 10.0, 100.0):
        candidates.append(TransportCandidate(
            f"ridge_a{alpha:g}", "ridge", {"alpha": alpha}
        ))
    for neighbors in (8, 24, 64):
        for correlation_power in (0.0, 1.0):
            for temperature in (0.5, 1.5):
                candidates.append(TransportCandidate(
                    f"analog_k{neighbors}_p{correlation_power:g}_t{temperature:g}",
                    "analog",
                    {"neighbors": neighbors, "correlation_power": correlation_power,
                     "temperature": temperature},
                ))
    for leaf in (2, 6):
        for max_features in (0.5, 1.0):
            candidates.append(TransportCandidate(
                f"extra_leaf{leaf}_f{max_features:g}", "extra",
                {"min_samples_leaf": leaf, "max_features": max_features},
            ))
            candidates.append(TransportCandidate(
                f"ridge_residual_extra_leaf{leaf}_f{max_features:g}",
                "ridge_residual_extra",
                {"min_samples_leaf": leaf, "max_features": max_features,
                 "alpha": 10.0},
            ))
    for leaves in (15, 31):
        for minimum in (10, 30):
            for l2 in (0.0, 1.0):
                candidates.append(TransportCandidate(
                    f"hist_l{leaves}_m{minimum}_r{l2:g}", "hist",
                    {"max_leaf_nodes": leaves, "min_samples_leaf": minimum,
                     "l2_regularization": l2},
                ))
    return tuple(candidates)


def _arrays(rows: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(rows["x"], dtype=np.float64)
    y = np.asarray(rows["y"], dtype=np.float64)
    groups = np.asarray(rows["groups"])
    if x.ndim != 2 or y.shape != (len(x),) or groups.shape != y.shape:
        raise ValueError("rows must contain aligned x, y, and groups")
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("rows must be finite and nonempty")
    return x, y, groups


def _scale(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(train_x, axis=0)
    scale = np.quantile(train_x, 0.75, axis=0) - np.quantile(train_x, 0.25, axis=0)
    scale = np.where(scale > 1e-8, scale, np.std(train_x, axis=0))
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (train_x - center) / scale, (test_x - center) / scale


def _correlation_weights(x: np.ndarray, y: np.ndarray, power: float) -> np.ndarray:
    if power == 0:
        return np.ones(x.shape[1], dtype=np.float64)
    target = rankdata(y)
    weights = []
    for column in x.T:
        value = np.corrcoef(rankdata(column), target)[0, 1]
        weights.append(0.0 if not np.isfinite(value) else abs(float(value)))
    weights = np.asarray(weights)
    weights = 0.10 + weights ** float(power)
    return weights / np.mean(weights)


def predict_candidate(
    candidate: TransportCandidate,
    train: dict,
    target: dict,
    *,
    seed: int,
) -> np.ndarray:
    train_x, train_y, _ = _arrays(train)
    target_x = np.asarray(target["x"], dtype=np.float64)
    z_train, z_target = _scale(train_x, target_x)
    cap = max(float(np.max(train_y)), 1.0)
    if candidate.kind == "ridge":
        model = Ridge(alpha=float(candidate.parameters["alpha"]))
        model.fit(z_train, train_y)
        prediction = model.predict(z_target)
    elif candidate.kind == "analog":
        weights = _correlation_weights(
            z_train, train_y, float(candidate.parameters["correlation_power"])
        )
        fit_x = z_train * np.sqrt(weights)
        query_x = z_target * np.sqrt(weights)
        count = min(int(candidate.parameters["neighbors"]), len(fit_x))
        neighbors = NearestNeighbors(n_neighbors=count).fit(fit_x)
        distance, index = neighbors.kneighbors(query_x)
        positive = distance[distance > 1e-12]
        distance_scale = float(np.median(positive)) if len(positive) else 1.0
        kernel = np.exp(
            -0.5 * (distance / max(
                distance_scale * float(candidate.parameters["temperature"]), 1e-8
            )) ** 2
        )
        kernel /= np.maximum(kernel.sum(axis=1, keepdims=True), 1e-12)
        prediction = np.sum(kernel * train_y[index], axis=1)
    elif candidate.kind == "extra":
        model = ExtraTreesRegressor(
            n_estimators=160,
            min_samples_leaf=int(candidate.parameters["min_samples_leaf"]),
            max_features=float(candidate.parameters["max_features"]),
            random_state=int(seed),
            n_jobs=1,
        )
        model.fit(z_train, train_y)
        prediction = model.predict(z_target)
    elif candidate.kind == "ridge_residual_extra":
        ridge = Ridge(alpha=float(candidate.parameters["alpha"]))
        ridge.fit(z_train, train_y)
        residual = train_y - ridge.predict(z_train)
        model = ExtraTreesRegressor(
            n_estimators=160,
            min_samples_leaf=int(candidate.parameters["min_samples_leaf"]),
            max_features=float(candidate.parameters["max_features"]),
            random_state=int(seed),
            n_jobs=1,
        )
        model.fit(z_train, residual)
        prediction = ridge.predict(z_target) + model.predict(z_target)
    elif candidate.kind == "hist":
        model = HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.05,
            max_leaf_nodes=int(candidate.parameters["max_leaf_nodes"]),
            min_samples_leaf=int(candidate.parameters["min_samples_leaf"]),
            l2_regularization=float(candidate.parameters["l2_regularization"]),
            random_state=int(seed),
        )
        model.fit(z_train, train_y)
        prediction = model.predict(z_target)
    else:
        raise ValueError(f"unknown candidate kind: {candidate.kind}")
    return np.clip(np.asarray(prediction, dtype=np.float64), 0.0, cap)


def _group_risk(y: np.ndarray, prediction: np.ndarray, groups: np.ndarray) -> np.ndarray:
    return np.asarray([
        np.mean((prediction[groups == label] - y[groups == label]) ** 2)
        for label in np.unique(groups)
    ])


def _risk_summary(parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple:
    risks = np.concatenate([
        _group_risk(y, prediction, groups) for y, prediction, groups in parts
    ])
    tail_count = max(1, int(np.ceil(0.20 * len(risks))))
    return (
        float(np.mean(risks)),
        float(np.mean(np.sort(risks)[-tail_count:])),
        float(np.max(risks)),
    )


def _simplex(step: int = 4):
    for values in np.ndindex(*([step + 1] * 4)):
        if sum(values) == step:
            yield np.asarray(values, dtype=np.float64) / step


def select_transport_policy(
    folds: list[tuple[dict, dict]],
    *,
    candidates: tuple[TransportCandidate, ...] | None = None,
) -> tuple[TransportPolicy, dict]:
    candidates = candidates or candidate_grid()
    predictions = []
    individual = []
    for fold_index, (train, validation) in enumerate(folds):
        fold_predictions = {}
        y = np.asarray(validation["y"], dtype=np.float64)
        groups = np.asarray(validation["groups"])
        for candidate in candidates:
            fold_predictions[candidate.name] = predict_candidate(
                candidate, train, validation, seed=42 + fold_index
            )
        predictions.append((fold_predictions, y, groups))
    for candidate in candidates:
        risk = _risk_summary([
            (y, values[candidate.name], groups)
            for values, y, groups in predictions
        ])
        individual.append((risk, candidate))
    individual.sort(key=lambda value: value[0])
    finalists = tuple(value[1] for value in individual[:4])
    best = None
    for weights in _simplex():
        parts = []
        for values, y, groups in predictions:
            estimate = sum(
                float(weight) * values[candidate.name]
                for weight, candidate in zip(weights, finalists)
            )
            parts.append((y, estimate, groups))
        risk = _risk_summary(parts)
        value = (risk[0], risk[1], risk[2], -float(np.max(weights)))
        if best is None or value < best[0]:
            best = (value, weights, risk)
    _, weights, risk = best
    keep = weights > 0
    policy = TransportPolicy(
        tuple(candidate for candidate, selected in zip(finalists, keep) if selected),
        tuple(float(weight) for weight in weights[keep]),
        *risk,
    )
    audit = {
        "individual_candidates": [
            {"name": candidate.name, "kind": candidate.kind,
             "macro_mse": risk[0], "cvar20_mse": risk[1], "max_mse": risk[2]}
            for risk, candidate in individual
        ],
        "finalists": [candidate.name for candidate in finalists],
    }
    return policy, audit


def predict_transport_policy(
    policy: TransportPolicy,
    train: dict,
    target: dict,
    *,
    seed: int,
) -> np.ndarray:
    return sum(
        weight * predict_candidate(candidate, train, target, seed=seed)
        for candidate, weight in zip(policy.candidates, policy.weights)
    )
