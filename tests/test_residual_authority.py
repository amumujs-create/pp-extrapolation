import numpy as np

from pp_extrapolation.residual_authority import (
    apply_residual_authority,
    crossfit_residual_authority,
    distance_shell_edges,
    fit_residual_authority,
)


def grouped_problem():
    groups = np.repeat(np.arange(6), 12)
    distance = np.tile(np.linspace(0.05, 2.95, 12), 6)
    y = 10.0 - distance
    fallback = y + 1.0
    prior = y + 0.6
    candidate = y + np.where(distance < 1.5, 0.05, 0.45)
    return y, groups, fallback, prior, candidate, distance


def test_edges_are_label_free_and_end_at_infinity():
    edges = distance_shell_edges(np.linspace(0, 4, 100))
    assert edges[0] == 0
    assert np.isinf(edges[-1])
    assert all(right > left for left, right in zip(edges, edges[1:]))


def test_authority_is_monotone_and_bounded():
    y, groups, fallback, prior, candidate, distance = grouped_problem()
    policy = fit_residual_authority(
        y, groups, fallback, prior, candidate, distance,
        edges=(0.0, 1.0, 2.0, np.inf),
        epsilon=0.20,
    )
    active = [value for value in policy.authorities if value is not None]
    assert all(0 <= value <= 1 for value in active)
    assert all(left >= right for left, right in zip(active, active[1:]))
    prediction = apply_residual_authority(
        prior, candidate, fallback, distance, policy
    )
    for index, authority in enumerate(policy.authorities):
        mask = np.searchsorted(policy.edges[1:-1], distance, side="right") == index
        if authority is None:
            assert np.array_equal(prediction[mask], fallback[mask])
        else:
            assert np.allclose(
                prediction[mask] - prior[mask],
                authority * (candidate[mask] - prior[mask]),
            )


def test_unsupported_far_shell_uses_exact_fallback():
    y, groups, fallback, prior, candidate, distance = grouped_problem()
    # Only one group is represented in the far shell.
    distance = distance.copy()
    distance[groups != 0] = np.minimum(distance[groups != 0], 1.9)
    policy = fit_residual_authority(
        y, groups, fallback, prior, candidate, distance,
        edges=(0.0, 1.0, 2.0, np.inf),
        epsilon=0.20,
        minimum_groups_per_shell=2,
    )
    assert policy.authorities[-1] is None
    prediction = apply_residual_authority(
        prior, candidate, fallback, distance, policy
    )
    assert np.array_equal(prediction[distance >= 2], fallback[distance >= 2])


def test_crossfit_rejects_harmful_prior_family():
    y, groups, fallback, _, _, distance = grouped_problem()
    prior = y + 3.0
    candidate = y + 2.5
    result = crossfit_residual_authority(
        y, groups, fallback, prior, candidate, distance,
        edges=(0.0, 1.0, 2.0, np.inf),
        minimum_relative_gain=0.01,
    )
    assert not result.approved
    assert all(value is None for value in result.policy.authorities)
    prediction = apply_residual_authority(
        prior, candidate, fallback, distance, result.policy
    )
    assert np.array_equal(prediction, fallback)


def test_crossfit_does_not_call_exact_fallback_an_approval():
    y, groups, fallback, _, _, distance = grouped_problem()
    result = crossfit_residual_authority(
        y, groups, fallback, fallback, fallback, distance,
        edges=(0.0, 1.0, 2.0, np.inf),
    )
    assert not result.approved
    assert result.oof_relative_gain_vs_fallback == 0
    assert all(value is None for value in result.policy.authorities)


def test_crossfit_approves_replicated_gain():
    y, groups, fallback, prior, candidate, distance = grouped_problem()
    result = crossfit_residual_authority(
        y, groups, fallback, prior, candidate, distance,
        edges=(0.0, 1.0, 2.0, np.inf),
        epsilon=0.20,
        minimum_relative_gain=0.01,
    )
    assert result.approved
    assert result.oof_relative_gain_vs_fallback > 0
    assert result.policy.active_shells > 0
