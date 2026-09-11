import numpy as np
import torch

from pp_extrapolation.grade_cvar_implicit_expert import (
    GCIEConfig,
    fit_gcie,
    fit_support_grade,
    grade_cvar_energy_loss,
    monotonic_violation_penalty,
    predict_mean,
    predict_samples,
    support_grade,
)


def _small_problem():
    x = np.asarray(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [1.0, 0.9],
            [1.2, 1.0],
            [2.0, 1.8],
            [2.2, 2.0],
        ],
        dtype=np.float32,
    )
    groups = np.asarray(["a", "a", "b", "b", "c", "c"])
    cycles = np.tile(np.asarray([0.0, 1.0]), 3)
    y = np.asarray([3.0, 2.0, 3.0, 2.0, 3.0, 2.0], dtype=np.float32)
    validation_x = np.asarray([[0.5, 0.4], [0.7, 0.6], [1.5, 1.4]], np.float32)
    validation_y = np.asarray([3.0, 2.0, 1.0], np.float32)
    validation_groups = np.asarray(["v", "v", "v"])
    validation_cycles = np.asarray([0.0, 1.0, 2.0], np.float32)
    return (
        x,
        y,
        groups,
        cycles,
        validation_x,
        validation_y,
        validation_groups,
        validation_cycles,
    )


def _fit(seed=7):
    values = _small_problem()
    config = GCIEConfig(
        hidden_dims=(8,),
        noise_dim=2,
        learning_rate=5e-3,
        max_epochs=20,
        patience=5,
        sample_count=4,
        seed=seed,
    )
    return fit_gcie(*values, config=config), values


def test_support_grade_excludes_self_unit_centroid():
    x, _, groups, *_ = _small_problem()
    fitted, loo_grade = fit_support_grade(x, groups)
    ordinary_grade = support_grade(fitted, x)
    assert fitted.centroid_groups.tolist() == ["a", "b", "c"]
    assert np.all(loo_grade > ordinary_grade)
    assert np.isfinite(loo_grade).all()


def test_prediction_shape_finiteness_bounds_and_deterministic_seed():
    fitted, values = _fit()
    x = values[4]
    first = predict_samples(fitted, x, sample_count=6, seed=123)
    second = predict_samples(fitted, x, sample_count=6, seed=123)
    assert first.shape == (len(x), 6)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert np.all(first >= 0.0)
    assert np.all(first <= fitted.target_cap)
    np.testing.assert_array_equal(
        predict_mean(fitted, x, sample_count=6, seed=123), first.mean(axis=1)
    )


def test_same_fit_seed_replays_training_and_best_state():
    first, values = _fit(seed=11)
    second, _ = _fit(seed=11)
    p1 = predict_mean(first, values[4], seed=4)
    p2 = predict_mean(second, values[4], seed=4)
    assert np.array_equal(p1, p2)
    assert first.selection["selected_epoch"] <= first.selection["epochs_executed"]


def test_rho_zero_is_exact_fallback_and_zero_stochastic_mass():
    fitted, values = _fit()
    fallback = np.asarray([0.25, 1.25, 2.25], dtype=np.float64)
    samples = predict_samples(
        fitted, values[4], sample_count=4, seed=99, rho=0.0, fallback=fallback
    )
    assert np.array_equal(samples, np.repeat(fallback[:, None], 4, axis=1))
    deterministic = predict_samples(
        fitted, values[4], sample_count=4, seed=99, rho=0.0
    )
    assert np.array_equal(deterministic, deterministic[:, :1].repeat(4, axis=1))


def test_cvar_targets_worst_group_and_monotonic_penalty_detects_increase():
    target = torch.zeros(4)
    samples = torch.tensor([[0.0, 0.0], [0.0, 0.0], [2.0, 2.0], [2.0, 2.0]])
    groups = torch.tensor([0, 0, 1, 1])
    total, balanced, cvar = grade_cvar_energy_loss(
        target, samples, groups, cvar_fraction=0.5, cvar_weight=1.0
    )
    assert torch.isclose(balanced, torch.tensor(1.0))
    assert torch.isclose(cvar, torch.tensor(2.0))
    assert torch.isclose(total, torch.tensor(3.0))

    labels = np.asarray(["u", "u", "u"])
    cycles = np.asarray([0.0, 1.0, 2.0])
    decreasing = monotonic_violation_penalty(
        torch.tensor([3.0, 2.0, 1.0]), labels, cycles
    )
    increasing = monotonic_violation_penalty(
        torch.tensor([1.0, 2.0, 3.0]), labels, cycles
    )
    assert decreasing.item() == 0.0
    assert increasing.item() > 0.0
