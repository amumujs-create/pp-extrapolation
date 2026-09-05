import numpy as np
import torch

from pp_extrapolation.model import PPNet, equal_group_weights, fit_pp, predict, solve_weighted_affine


def test_nonlinear_path_starts_at_zero():
    torch.manual_seed(3)
    model = PPNet(input_dim=3, width=7)
    value = torch.randn(9, 3)
    with torch.no_grad():
        assert torch.count_nonzero(model.nonlinear(value)) == 0
        assert torch.equal(model(value), model.affine(value).squeeze(1))


def test_each_group_has_equal_total_weight():
    groups = np.asarray(["a", "a", "a", "b"])
    weights = equal_group_weights(groups)
    np.testing.assert_allclose(weights[groups == "a"].sum(), weights[groups == "b"].sum())


def test_affine_solver_recovers_line_without_penalty():
    x = np.arange(20, dtype=float)[:, None]
    y = 2.0 * x[:, 0] + 3.0
    solution = solve_weighted_affine(x, y, np.ones(len(x)), alpha=0.0)
    np.testing.assert_allclose(x @ solution.weight + solution.bias, y, atol=1e-6)


def test_soft_anchor_fit_is_finite_and_records_penalty():
    x = np.arange(12, dtype=float)[:, None]
    split = lambda indices: {
        "x": x[indices], "y": (2 * x[indices, 0] + 1),
        "groups": np.asarray([f"g{i // 3}" for i in indices]),
    }
    train, validation, test = split(np.arange(8)), split(np.arange(8, 10)), split(np.arange(10, 12))
    fit = fit_pp(train, validation, seed=7, max_epochs=5, patience=2,
                 affine_anchor_weight=0.1)
    assert np.isfinite(predict(fit, test["x"])).all()
    assert fit.selection["affine_anchor_weight"] == 0.1
    assert all("affine_anchor_loss" in row for row in fit.selection["loss_history"])
