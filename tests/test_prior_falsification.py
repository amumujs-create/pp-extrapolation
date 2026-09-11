import numpy as np

from pp_extrapolation.prior_falsification import (
    certify_prior_falsification,
    unit_log_regret,
)


def sample(candidate_error=0.2):
    groups = np.repeat(np.arange(8), 10)
    y = np.tile(np.linspace(0, 1, 10), 8)
    fallback = y + 1.0
    candidate = y + candidate_error
    return y, groups, fallback, candidate


def test_unit_log_regret_uses_physical_units():
    regret = unit_log_regret(*sample())
    assert regret.shape == (8,)
    np.testing.assert_allclose(regret, np.log(0.2))


def test_consistently_better_prior_is_not_falsified():
    certificate = certify_prior_falsification(*sample(), seed=7)
    assert not certificate.falsified
    assert certificate.upper_confidence_bound < 0


def test_consistently_worse_prior_is_falsified():
    certificate = certify_prior_falsification(
        *sample(candidate_error=2.0), seed=7
    )
    assert certificate.falsified
    assert certificate.upper_confidence_bound > 0


def test_single_unit_is_insufficient_and_falsified():
    y = np.arange(5.0)
    groups = np.zeros(5)
    certificate = certify_prior_falsification(
        y, groups, y + 1, y + 0.1
    )
    assert certificate.falsified
    assert np.isinf(certificate.upper_confidence_bound)


def test_misaligned_inputs_raise():
    y, groups, fallback, candidate = sample()
    try:
        unit_log_regret(y[:-1], groups, fallback, candidate)
    except ValueError:
        pass
    else:
        raise AssertionError("misaligned inputs must raise")
