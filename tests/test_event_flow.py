import numpy as np
import torch

from pp_extrapolation.event_flow import (
    EventCoordinateFlow,
    adjacent_group_pairs,
)


def model(observed):
    return EventCoordinateFlow(
        3,
        4,
        observed_coordinate=observed,
        quadrature_points=32,
        center=np.zeros(3, dtype=np.float32),
        scale=np.ones(3, dtype=np.float32),
        margin_scale=1.0,
        initial_time_scale=10.0,
    )


def test_observed_boundary_prediction_is_zero():
    net = model(True)
    x = torch.tensor([[0.0, 1.0, 2.0]])
    assert net(x).item() == 0.0


def test_event_flow_prediction_is_nonnegative():
    net = model(False)
    x = torch.randn(8, 3)
    assert torch.all(net(x) >= 0)


def test_latent_margin_is_positive():
    net = model(False)
    x = torch.randn(8, 3)
    assert torch.all(net.margin(x) > 0)


def test_adjacent_pairs_never_cross_group_boundary():
    groups = np.asarray(["a", "a", "b", "b", "b", "c"])
    assert np.array_equal(
        adjacent_group_pairs(groups),
        np.asarray([[0, 1], [2, 3], [3, 4]]),
    )
