"""Contract tests for screening, including post-selection memory expansion."""
import importlib.util
from pathlib import Path

import numpy as np

from pp_extrapolation.relation_local_transport import fit_relation_transport

path = Path(__file__).resolve().parents[1]/'experiments/relation_local_transport_screen.py'
spec = importlib.util.spec_from_file_location('relation_screen', path)
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)


def test_metrics_count_unit_and_seed_coverage_separately():
    y = np.array([1., 2., 3., 4.])
    groups = np.array(['a', 'a', 'b', 'b'])
    result = screen.metrics(y, groups, np.array([y, y+10]))
    assert result['positive_seeds'] == 1
    assert result['ensemble']['positive_units'] == 0
    assert result['finite_fraction'] == 1.
    assert result['units'] == 2


def test_constant_target_has_no_fabricated_r2():
    result = screen.metrics(np.ones(3), np.repeat('a', 3), np.ones((1, 3)))
    assert result['ensemble']['r2'] is None
    assert result['ensemble']['positive_units'] == 0


def test_memory_expansion_preserves_geometry_and_original_fit():
    x = np.tile(np.linspace(.1, 1., 20), 4)[:, None]
    train = dict(x=x, y=5*x[:, 0], groups=np.repeat(['a', 'b', 'c', 'd'], 20))
    val = dict(x=x[:10], y=5*x[:10, 0], groups=np.repeat('e', 10))
    fit = fit_relation_transport(train, val, progress_index=0, increasing=False,
        prior_approved=True, prior_train=4*x[:, 0], prior_validation=4*x[:10, 0],
        boundary_index=0, learn=False)
    full = {k: np.concatenate((train[k], val[k])) for k in train}
    final = screen.memory_refit(fit, full, 4*full['x'][:, 0])
    np.testing.assert_array_equal(final.center, fit.center)
    np.testing.assert_array_equal(final.scale, fit.scale)
    for name, value in fit.model.state_dict().items():
        np.testing.assert_array_equal(value.numpy(), final.model.state_dict()[name].numpy())
    assert len(fit.bank_indices) == 80
    assert len(final.bank_indices) == 90
    np.testing.assert_allclose(final.bank_values*final.value_scale, 1.)
