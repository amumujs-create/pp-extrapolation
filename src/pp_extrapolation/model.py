"""PP: a frozen affine tail plus a learned bounded-capacity nonlinear path."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch
from torch import nn
from .priors import (CounterfactualRays, PriorPairs, TransportTriples,
                     counterfactual_residual_loss, pair_loss, prepare_counterfactual,
                     prepare_pairs, prepare_transport, transport_loss)


class PPNet(nn.Module):
    """One network containing an affine path and a tanh correction path."""

    def __init__(self, input_dim: int, width: int = 32, *, residual_decay: float = 0.0):
        super().__init__()
        if residual_decay < 0 or not np.isfinite(residual_decay):
            raise ValueError("residual_decay must be finite and nonnegative")
        self.residual_decay = float(residual_decay)
        self.register_buffer("support_min", torch.full((input_dim,), -torch.inf))
        self.register_buffer("support_max", torch.full((input_dim,), torch.inf))
        self.affine = nn.Linear(input_dim, 1)
        self.nonlinear = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1),
        )
        nn.init.zeros_(self.nonlinear[-1].weight)
        nn.init.zeros_(self.nonlinear[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        correction = self.nonlinear(value)
        if self.residual_decay > 0:
            below = torch.relu(self.support_min - value)
            above = torch.relu(value - self.support_max)
            distance = torch.linalg.vector_norm(below + above, dim=1, keepdim=True)
            correction = correction * torch.exp(-self.residual_decay * distance)
        return (self.affine(value) + correction).squeeze(1)


@dataclass(frozen=True)
class AffineInitialization:
    weight: np.ndarray
    bias: float
    alpha: float


@dataclass
class PPFit:
    model: PPNet
    center: np.ndarray
    scale: np.ndarray
    target_scale: float
    selection: dict


def _arrays(split: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(split["x"], dtype=np.float64)
    y = np.asarray(split["y"], dtype=np.float64)
    groups = np.asarray(split["groups"])
    if x.ndim != 2 or y.shape != (len(x),) or groups.shape != (len(x),):
        raise ValueError("split requires x=(n,d), y=(n,), groups=(n,)")
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("x and y must be finite and nonempty")
    return x, y, groups


def equal_group_weights(groups: np.ndarray) -> np.ndarray:
    groups = np.asarray(groups)
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(np.float64)
    return weights / weights.mean()


def fit_feature_scale(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return center.astype(np.float32), scale.astype(np.float32)


def transform_features(x: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((np.asarray(x, dtype=np.float64) - center) / scale).astype(np.float32)


def solve_weighted_affine(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, *, alpha: float
) -> AffineInitialization:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    x_center = np.sum(weights[:, None] * x, axis=0) / total
    y_center = float(np.sum(weights * y) / total)
    centered_x = x - x_center
    centered_y = y - y_center
    gram = centered_x.T @ (weights[:, None] * centered_x)
    right = centered_x.T @ (weights * centered_y)
    coefficient = np.linalg.solve(
        gram + float(alpha) * np.eye(x.shape[1], dtype=np.float64), right
    )
    bias = y_center - float(x_center @ coefficient)
    return AffineInitialization(coefficient.astype(np.float32), bias, float(alpha))


def _affine_prediction(
    initialization: AffineInitialization,
    x: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    target_scale: float,
) -> np.ndarray:
    z = transform_features(x, center, scale)
    normalized = z @ initialization.weight + initialization.bias
    return np.clip(normalized * target_scale, 0.0, target_scale)


def select_affine_initialization(
    train: dict,
    validation: dict,
    *,
    alphas: Iterable[float] = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0),
) -> dict:
    train_x, train_y, groups = _arrays(train)
    validation_x, validation_y, _ = _arrays(validation)
    center, scale = fit_feature_scale(train_x)
    z = transform_features(train_x, center, scale)
    target_scale = max(float(np.max(train_y)), 1.0)
    weights = equal_group_weights(groups)
    candidates = []
    solutions = {}
    for alpha in alphas:
        solution = solve_weighted_affine(
            z, train_y / target_scale, weights, alpha=float(alpha)
        )
        prediction = _affine_prediction(
            solution, validation_x, center, scale, target_scale
        )
        mse = float(np.mean((prediction - validation_y) ** 2))
        candidates.append({"alpha": float(alpha), "validation_mse": mse})
        solutions[float(alpha)] = solution
    selected = min(candidates, key=lambda row: (row["validation_mse"], row["alpha"]))
    return {
        "center": center,
        "scale": scale,
        "target_scale": target_scale,
        "initialization": solutions[selected["alpha"]],
        "selected_alpha": selected["alpha"],
        "candidates": candidates,
    }


def fit_pp(
    train: dict,
    validation: dict,
    *,
    seed: int,
    affine_selection: Optional[dict] = None,
    max_epochs: int = 300,
    patience: int = 70,
    batch_size: int = 512,
    prior_pairs: Optional[PriorPairs] = None,
    transport_triples: Optional[TransportTriples] = None,
    prior_weight: float = 0.0,
    transport_weight: float = 0.0,
    affine_anchor_weight: float | None = None,
    residual_decay: float = 0.0,
    counterfactual_rays: Optional[CounterfactualRays] = None,
    counterfactual_weight: float = 0.0,
    consistency_x: Optional[np.ndarray] = None,
    consistency_target: Optional[np.ndarray] = None,
    consistency_weight: float = 0.0,
) -> PPFit:
    """Fit PP and select the checkpoint using validation MSE only."""
    train_x, train_y, groups = _arrays(train)
    validation_x, validation_y, _ = _arrays(validation)
    selection = affine_selection or select_affine_initialization(train, validation)
    center = selection["center"]
    scale = selection["scale"]
    target_scale = float(selection["target_scale"])
    initialization = selection["initialization"]

    if not np.isfinite([prior_weight, transport_weight, counterfactual_weight, consistency_weight]).all() or min(prior_weight, transport_weight, counterfactual_weight, consistency_weight) < 0:
        raise ValueError("prior weights must be finite and nonnegative")
    if prior_weight > 0 and prior_pairs is None:
        raise ValueError("prior_pairs required for positive prior_weight")
    if transport_weight > 0 and transport_triples is None:
        raise ValueError("transport_triples required for positive transport_weight")
    if counterfactual_weight > 0 and counterfactual_rays is None:
        raise ValueError("counterfactual_rays required for positive counterfactual_weight")
    pair_values = prepare_pairs(prior_pairs, center, scale, target_scale) if prior_weight > 0 else None
    transport_values = prepare_transport(transport_triples, center, scale, target_scale) if transport_weight > 0 else None
    counterfactual_values = prepare_counterfactual(counterfactual_rays, center, scale) if counterfactual_weight > 0 else None
    consistency_values = None
    if consistency_weight > 0:
        if consistency_x is None or consistency_target is None:
            raise ValueError("consistency_x and consistency_target required")
        cx=np.asarray(consistency_x,dtype=np.float64); ct=np.asarray(consistency_target,dtype=np.float64)
        if cx.ndim!=2 or cx.shape[1]!=train_x.shape[1] or ct.shape!=(len(cx),):
            raise ValueError("invalid consistency reference")
        consistency_values=(torch.as_tensor(transform_features(cx,center,scale),dtype=torch.float32),
                            torch.as_tensor(ct/target_scale,dtype=torch.float32))
    prior_rng = np.random.default_rng(int(seed) + 100000)

    def sample(values):
        idx = torch.as_tensor(prior_rng.choice(len(values[0]), min(batch_size, len(values[0])), replace=False))
        return tuple(v[idx] for v in values)

    torch.manual_seed(int(seed))
    x = torch.as_tensor(transform_features(train_x, center, scale), dtype=torch.float32)
    y = torch.as_tensor(train_y / target_scale, dtype=torch.float32)
    weights = torch.as_tensor(equal_group_weights(groups), dtype=torch.float32)
    val_x = torch.as_tensor(
        transform_features(validation_x, center, scale), dtype=torch.float32
    )

    model = PPNet(input_dim=x.shape[1], width=32, residual_decay=residual_decay)
    with torch.no_grad():
        model.support_min.copy_(x.amin(dim=0))
        model.support_max.copy_(x.amax(dim=0))
        model.affine.weight.copy_(torch.as_tensor(initialization.weight)[None, :])
        model.affine.bias.copy_(torch.as_tensor([initialization.bias]))
    soft_anchor = affine_anchor_weight is not None
    if soft_anchor and (not np.isfinite(affine_anchor_weight) or affine_anchor_weight < 0):
        raise ValueError("affine_anchor_weight must be finite and nonnegative")
    model.affine.requires_grad_(soft_anchor)
    parameters = list(model.nonlinear.parameters())
    if soft_anchor:
        parameters += list(model.affine.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=5e-4, weight_decay=2.0)
    anchor_weight = torch.as_tensor(initialization.weight, dtype=torch.float32)
    anchor_bias = torch.as_tensor(initialization.bias, dtype=torch.float32)
    rng = np.random.default_rng(int(seed))

    def validation_loss() -> float:
        model.eval()
        with torch.no_grad():
            value = np.clip(model(val_x).cpu().numpy() * target_scale, 0.0, target_scale)
        return float(np.mean((value - validation_y) ** 2))

    best_loss = validation_loss()
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    last_epoch = 0
    loss_history = []
    for epoch in range(1, int(max_epochs) + 1):
        last_epoch = epoch
        model.train()
        epoch_terms = np.zeros(5, dtype=np.float64)
        batches = 0
        order = rng.permutation(len(x))
        for start in range(0, len(x), int(batch_size)):
            index = torch.as_tensor(order[start : start + int(batch_size)], dtype=torch.long)
            loss = torch.mean(weights[index] * (model(x[index]) - y[index]) ** 2)
            data_term = loss.detach().item()
            relation_term = pair_loss(model, sample(pair_values)) if pair_values is not None else loss.new_zeros(())
            transport_term = transport_loss(model, sample(transport_values)) if transport_values is not None else loss.new_zeros(())
            counterfactual_term = counterfactual_residual_loss(model, sample(counterfactual_values)) if counterfactual_values is not None else loss.new_zeros(())
            consistency_term = loss.new_zeros(())
            if consistency_values is not None:
                con_x,con_y=sample(consistency_values)
                consistency_term=torch.mean((model(con_x)-con_y)**2)
            anchor_term = loss.new_zeros(())
            if soft_anchor:
                anchor_term = torch.mean((model.affine.weight.squeeze(0) - anchor_weight) ** 2)
                anchor_term = anchor_term + (model.affine.bias.squeeze(0) - anchor_bias) ** 2
            loss = loss + prior_weight * relation_term + transport_weight * transport_term
            loss = loss + counterfactual_weight * counterfactual_term
            loss = loss + consistency_weight * consistency_term
            if soft_anchor:
                loss = loss + float(affine_anchor_weight) * anchor_term
            epoch_terms += [data_term, relation_term.detach().item(), transport_term.detach().item(), anchor_term.detach().item(), counterfactual_term.detach().item()]
            batches += 1
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite PP loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.nonlinear.parameters(), 2.0, error_if_nonfinite=True)
            optimizer.step()
        current = validation_loss()
        loss_history.append(dict(epoch=epoch, data_mse=float(epoch_terms[0]/batches),
                                 relation_loss=float(epoch_terms[1]/batches),
                                 transport_loss=float(epoch_terms[2]/batches),
                                 affine_anchor_loss=float(epoch_terms[3]/batches), validation_mse=current))
        loss_history[-1]["counterfactual_residual_loss"] = float(epoch_terms[4]/batches)
        loss_history[-1]["consistency_weight"] = float(consistency_weight)
        if current < best_loss - 1e-10:
            best_loss = current
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch - best_epoch > int(patience):
            break
    model.load_state_dict(best_state)
    return PPFit(
        model=model,
        center=center,
        scale=scale,
        target_scale=target_scale,
        selection={
            "loss_history": loss_history,
            "prior_weight": float(prior_weight),
            "transport_weight": float(transport_weight),
            "affine_anchor_weight": None if affine_anchor_weight is None else float(affine_anchor_weight),
            "residual_decay": float(residual_decay),
            "counterfactual_weight": float(counterfactual_weight),
            "counterfactual_rays": 0 if counterfactual_values is None else len(counterfactual_values[0]),
            "prior_pairs": 0 if pair_values is None else len(pair_values[0]),
            "transport_triples": 0 if transport_values is None else len(transport_values[0]),
            "seed": int(seed),
            "affine_alpha": float(selection["selected_alpha"]),
            "selected_epoch": int(best_epoch),
            "epochs_executed": int(last_epoch),
            "best_validation_mse": float(best_loss),
        },
    )


def predict(fit: PPFit, x: np.ndarray) -> np.ndarray:
    fit.model.eval()
    value = torch.as_tensor(
        transform_features(x, fit.center, fit.scale), dtype=torch.float32
    )
    with torch.no_grad():
        prediction = fit.model(value).cpu().numpy() * fit.target_scale
    return np.clip(prediction, 0.0, fit.target_scale).astype(np.float64)
