import numpy as np
import torch

from pp_extrapolation.model import PPNet, equal_group_weights, solve_weighted_affine


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

