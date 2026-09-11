import numpy as np

from pp_extrapolation.prior_set_consensus import (
    PriorSetPolicy,
    apply_prior_set_consensus,
    crossfit_prior_set_consensus,
    fit_prior_set_policy,
)


def sample():
    groups = np.repeat(np.arange(6), 10)
    x = np.tile(np.linspace(0, 1, 10), 6)
    y = x + np.repeat(np.linspace(-0.1, 0.1, 6), 10)
    fallback = y + 0.4
    experts = np.vstack((y + 0.1, y + 0.15, y + 1.0))
    return y, groups, fallback, experts, x


def test_policy_approves_only_supported_experts():
    y, groups, fallback, experts, distance = sample()
    policy = fit_prior_set_policy(
        y,
        groups,
        fallback,
        experts,
        distance,
        edges=(0.0, 0.5, np.inf),
        nested=False,
    )
    assert policy.approved_experts == ((0, 1), (0, 1))
    assert policy.relative_gain_vs_fallback > 0


def test_empty_set_is_exact_fallback():
    fallback = np.arange(5.0)
    experts = np.vstack((fallback + 1, fallback - 1))
    policy = PriorSetPolicy(
        edges=(0.0, np.inf),
        approved_experts=((),),
        disagreement_limits=(0.0,),
        nested=True,
        validation_loss=0.0,
        relative_gain_vs_fallback=0.0,
        active_fraction=0.0,
    )
    prediction, active, _ = apply_prior_set_consensus(
        experts, fallback, np.ones(5), policy
    )
    np.testing.assert_array_equal(prediction, fallback)
    assert not np.any(active)


def test_disagreement_wider_than_envelope_is_exact_fallback():
    fallback = np.zeros(3)
    experts = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.3, 0.1]])
    policy = PriorSetPolicy(
        edges=(0.0, np.inf),
        approved_experts=((0, 1),),
        disagreement_limits=(0.2,),
        nested=True,
        validation_loss=0.0,
        relative_gain_vs_fallback=0.0,
        active_fraction=0.0,
    )
    prediction, active, width = apply_prior_set_consensus(
        experts, fallback, np.ones(3), policy
    )
    assert active.tolist() == [True, False, True]
    assert prediction[1] == fallback[1]
    np.testing.assert_allclose(width, [0.1, 0.3, 0.1])


def test_nested_sets_cannot_expand_with_distance():
    y, groups, fallback, experts, distance = sample()
    policy = fit_prior_set_policy(
        y,
        groups,
        fallback,
        experts,
        distance,
        edges=(0.0, 0.5, np.inf),
        nested=True,
    )
    assert set(policy.approved_experts[1]).issubset(
        policy.approved_experts[0]
    )


def test_crossfit_rejects_harmful_prior_set_to_exact_fallback():
    y, groups, fallback, experts, distance = sample()
    harmful = np.vstack((y + 2.0, y - 2.0))
    result = crossfit_prior_set_consensus(
        y, groups, fallback, harmful, distance
    )
    assert not result.approved
    prediction, active, _ = apply_prior_set_consensus(
        harmful, fallback, distance, result.policy
    )
    np.testing.assert_array_equal(prediction, fallback)
    assert not np.any(active)
