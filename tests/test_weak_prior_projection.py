import numpy as np
import torch

from pp_extrapolation.weak_prior_projection import (
    WeakPriorProjection,
    apply_weak_prior_projection,
    fit_authority_head,
    predict_authority,
)


def test_hard_projection_is_nonnegative_and_monotone():
    raw = torch.tensor([[1.0, -1.0, 0.5, 0.2]])
    prediction, feasible = WeakPriorProjection()(
        raw, torch.ones(1)
    )
    np.testing.assert_allclose(prediction.numpy(), feasible.numpy())
    assert torch.all(feasible >= 0)
    assert torch.all(feasible[:, 1:] >= feasible[:, :-1])


def test_zero_authority_is_exact_raw_fallback():
    raw = np.asarray([[1.0, -1.0, 0.5]])
    prediction, _ = apply_weak_prior_projection(raw, np.asarray([0.0]))
    np.testing.assert_array_equal(prediction, raw)


def test_partial_authority_is_convex_blend():
    raw = np.asarray([[1.0, -1.0, 0.5]])
    hard, feasible = apply_weak_prior_projection(raw, np.asarray([1.0]))
    partial, _ = apply_weak_prior_projection(raw, np.asarray([0.25]))
    np.testing.assert_allclose(partial, raw + 0.25 * (feasible - raw))
    np.testing.assert_allclose(hard, feasible)


def test_projection_can_enforce_weak_minimum_slope():
    raw = np.asarray([[1.0, 1.0, 1.1]])
    _, feasible = apply_weak_prior_projection(
        raw, np.asarray([1.0]), np.asarray([0.2])
    )
    assert np.all(np.diff(feasible[0]) >= 0.2 - 1e-6)


def test_projection_is_differentiable_almost_everywhere():
    raw = torch.tensor([[0.1, 0.2, -0.3]], requires_grad=True)
    prediction, _ = WeakPriorProjection()(raw, torch.tensor([0.7]))
    prediction.sum().backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()


def test_authority_head_predicts_bounded_values():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(80, 4))
    target = 1.0 / (1.0 + np.exp(features[:, 0]))
    fit = fit_authority_head(
        features, target, seed=7, max_epochs=500, patience=50
    )
    prediction = predict_authority(fit, features[:10])
    assert np.all((prediction >= 0) & (prediction <= 1))
    assert np.corrcoef(prediction, target[:10])[0, 1] > 0.5
