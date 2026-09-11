import numpy as np
import torch

from pp_extrapolation.ctbf import CTBFNet, predict_rate_quotient


def make_model(*, weak_rate_prior=True):
    return CTBFNet(
        dimension=3,
        width=4,
        boundary=0.8,
        quadrature_points=64,
        weak_rate_prior=weak_rate_prior,
        residual_bound=1.0,
        rate_floor=1e-3,
        center=np.zeros(3, dtype=np.float32),
        scale=np.ones(3, dtype=np.float32),
    )


def test_velocity_is_strictly_positive():
    model = make_model()
    x = torch.tensor([[0.9, -0.002, 1.0], [0.85, 0.001, 2.0]])
    assert torch.all(model.local_velocity(x) > 0)


def test_boundary_time_is_exactly_zero():
    model = make_model()
    x = torch.tensor([[0.8, -0.002, 1.0]])
    assert model(x).item() == 0.0


def test_constant_velocity_integral_matches_analytic_solution():
    model = make_model(weak_rate_prior=True)
    x = torch.tensor([[0.9, -0.002, 1.0]])
    expected = (0.9 - 0.8) / 0.002
    assert np.isclose(model(x).item(), expected, rtol=1e-5)


def test_time_decreases_toward_boundary():
    model = make_model()
    x = torch.tensor([[0.95, -0.002, 1.0], [0.85, -0.002, 1.0]])
    prediction = model(x).detach().numpy()
    assert prediction[0] > prediction[1] > 0


def test_rate_quotient_respects_boundary_and_rate_floor():
    x = np.asarray([[0.8, -0.01], [0.9, 0.01]], dtype=float)
    result = predict_rate_quotient(x, boundary=0.8, rate_floor=0.001)
    assert np.allclose(result, [0.0, 100.0])
