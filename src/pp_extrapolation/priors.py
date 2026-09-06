"""Optional soft relational priors. All arrays must come from training/design only.

These losses express assumptions, not identified causal effects or guarantees.
Differences and tolerance use original target units; features use original units.
"""
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class PriorPairs:
    """Require lower <= f(changed)-f(anchor) <= upper, with fixed confidence.

    Degradation: [-inf, 0]; invariance: [-epsilon, epsilon].
    confidence is a vector in [0,1], declared independently of test outcomes.
    """
    anchor: np.ndarray
    changed: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    confidence: np.ndarray


@dataclass
class TransportTriples:
    """Transport a detached inward secant to a declared outward intervention.

    inner -> boundary -> outer must follow the same feasible intervention path.
    ratio = outward step / inward step, measured in a declared path parameter.
    tolerance is allowed target-difference error, in original target units.
    """
    inner: np.ndarray
    boundary: np.ndarray
    outer: np.ndarray
    ratio: np.ndarray
    tolerance: np.ndarray
    confidence: np.ndarray


@dataclass
class CounterfactualRays:
    """Train-only outward rays for learning residual trust without test adaptation.

    boundary and outer follow a feasible outward intervention. decay is the desired
    fraction of the boundary NN correction retained at outer, in [0, 1].
    """
    boundary: np.ndarray
    outer: np.ndarray
    decay: np.ndarray
    confidence: np.ndarray


def _features(arrays, center, scale):
    values = [np.asarray(a, dtype=np.float64) for a in arrays]
    n = len(values[0])
    if not n or any(a.shape != (n, len(center)) or not np.isfinite(a).all() for a in values):
        raise ValueError('prior features must be finite, nonempty, matching (n,d) arrays')
    return [torch.tensor((a-center)/scale, dtype=torch.float32) for a in values]


def _vector(value, n, name, finite=True):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != (n,) or np.isnan(a).any() or (finite and not np.isfinite(a).all()):
        raise ValueError(f'{name} must have shape (n,) and valid values')
    return torch.tensor(a, dtype=torch.float32)


def _confidence(value, n):
    c = _vector(value, n, 'confidence')
    if torch.any((c < 0) | (c > 1)):
        raise ValueError('confidence must be in [0,1]')
    return c


def prepare_pairs(prior, center, scale, target_scale):
    a, b = _features([prior.anchor, prior.changed], center, scale)
    lower = _vector(prior.lower, len(a), 'lower', False) / target_scale
    upper = _vector(prior.upper, len(a), 'upper', False) / target_scale
    if torch.any(lower > upper) or torch.isposinf(lower).any() or torch.isneginf(upper).any():
        raise ValueError('invalid lower/upper interval')
    return a, b, lower, upper, _confidence(prior.confidence, len(a))


def prepare_transport(prior, center, scale, target_scale):
    a, b, c = _features([prior.inner, prior.boundary, prior.outer], center, scale)
    ratio = _vector(prior.ratio, len(a), 'ratio')
    tolerance = _vector(prior.tolerance, len(a), 'tolerance') / target_scale
    if torch.any(ratio <= 0) or torch.any(tolerance < 0):
        raise ValueError('ratio must be positive and tolerance nonnegative')
    return a, b, c, ratio, tolerance, _confidence(prior.confidence, len(a))


def prepare_counterfactual(prior, center, scale):
    boundary, outer = _features([prior.boundary, prior.outer], center, scale)
    decay = _vector(prior.decay, len(boundary), 'decay')
    if torch.any((decay < 0) | (decay > 1)):
        raise ValueError('decay must be in [0,1]')
    return boundary, outer, decay, _confidence(prior.confidence, len(boundary))


def pair_loss(model, values):
    a, b, lower, upper, confidence = values
    delta = model(b) - model(a)
    # Denominator is row count: low confidence genuinely reduces prior strength.
    return (confidence * (torch.relu(lower-delta).square() + torch.relu(delta-upper).square())).mean()


def transport_loss(model, values):
    inner, boundary, outer, ratio, tolerance, confidence = values
    at_boundary = model(boundary)
    inward_delta = (at_boundary - model(inner)).detach()
    outward_delta = model(outer) - at_boundary
    violation = torch.relu(torch.abs(outward_delta-ratio*inward_delta)-tolerance)
    return (confidence * violation.square()).mean()


def counterfactual_residual_loss(model, values):
    boundary, outer, decay, confidence = values
    boundary_residual = model.nonlinear(boundary).squeeze(1).detach()
    outer_residual = model.nonlinear(outer).squeeze(1)
    return (confidence * (outer_residual - decay * boundary_residual).square()).mean()
