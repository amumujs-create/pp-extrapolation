import numpy as np

from pp_extrapolation.structural_prior_set import (
    StructuralPriorSetPolicy,
    apply_structural_prior_set,
    crossfit_structural_prior_set,
    fit_affine_prior,
    fit_history_rate_prior,
    fit_monotone_health_prior,
    fit_structural_prior_set_policy,
    health_distance,
    predict_affine_prior,
    predict_history_rate_prior,
    predict_monotone_health_prior,
)


def sample_health():
    health = np.tile(np.linspace(0.6, 1.0, 12), 6)
    groups = np.repeat(np.arange(6), 12)
    y = 80.0 * health + np.repeat(np.linspace(-2, 2, 6), 12)
    rate = np.full_like(health, -0.01)
    x = np.column_stack([health, rate, health ** 2])
    return x, health, rate, y, groups


def test_affine_and_history_priors_recover_a_linear_rule():
    x, health, rate, y, groups = sample_health()
    affine = fit_affine_prior(x, y, groups)
    history = fit_history_rate_prior(health, rate, y, groups)
    monotone = fit_monotone_health_prior(health, y)
    affine_pred = predict_affine_prior(affine, x)
    history_pred = predict_history_rate_prior(history, health, rate)
    monotone_pred = predict_monotone_health_prior(monotone, health)
    assert np.corrcoef(affine_pred, y)[0, 1] > 0.95
    assert np.corrcoef(history_pred, y)[0, 1] > 0.90
    assert np.corrcoef(monotone_pred, y)[0, 1] > 0.95


def test_monotone_tail_does_not_increase_with_lower_health():
    _, health, _, y, _ = sample_health()
    model = fit_monotone_health_prior(health, y)
    below = np.asarray([model.health_min - 0.2, model.health_min - 0.1])
    prediction = predict_monotone_health_prior(model, below)
    assert prediction[0] <= prediction[1] + 1e-12


def test_health_distance_is_zero_inside_train_support():
    health = np.asarray([0.9, 0.4])
    distance = health_distance(health, train_min=0.6, scale=0.2)
    np.testing.assert_allclose(distance, [0.0, 1.0])


def test_empty_set_is_exact_fallback():
    fallback = np.arange(4.0)
    experts = np.vstack((fallback + 1, fallback - 1))
    distances = np.zeros_like(experts)
    policy = StructuralPriorSetPolicy(
        names=("a", "b"),
        edges=((0.0, np.inf), (0.0, np.inf)),
        approved_shells=((), ()),
        disagreement_limits=(0.0, 0.0),
        ceilings=(1.0, 1.0),
        minimum_set_size=1,
        nested=True,
        validation_loss=0.0,
        relative_gain_vs_fallback=0.0,
        active_fraction=0.0,
    )
    prediction, active, _ = apply_structural_prior_set(
        experts, fallback, distances, policy
    )
    np.testing.assert_array_equal(prediction, fallback)
    assert not np.any(active)


def test_support_ceiling_rejects_far_rows():
    fallback = np.zeros(3)
    experts = np.asarray([[1.0, 1.0, 1.0]])
    distances = np.asarray([[0.1, 5.0, 0.2]])
    policy = StructuralPriorSetPolicy(
        names=("affine",),
        edges=((0.0, np.inf),),
        approved_shells=((0,),),
        disagreement_limits=(1.0, 1.0),
        ceilings=(1.0,),
        minimum_set_size=1,
        nested=True,
        validation_loss=0.0,
        relative_gain_vs_fallback=0.0,
        active_fraction=0.0,
    )
    prediction, active, _ = apply_structural_prior_set(
        experts, fallback, distances, policy
    )
    assert active.tolist() == [True, False, True]
    assert prediction[1] == 0.0


def test_consensus_requires_two_priors_when_requested():
    x, health, rate, y, groups = sample_health()
    fallback = y + 8.0
    experts = np.vstack((y + 0.2, y + 0.3, y + 12.0))
    distances = np.vstack((
        np.zeros(len(y)),
        np.zeros(len(y)),
        np.zeros(len(y)),
    ))
    singleton = fit_structural_prior_set_policy(
        y, groups, fallback, experts, distances,
        edges=((0.0, np.inf),) * 3,
        ceilings=(1.0, 1.0, 1.0),
        minimum_set_size=1,
        nested=False,
    )
    consensus = fit_structural_prior_set_policy(
        y, groups, fallback, experts, distances,
        edges=((0.0, np.inf),) * 3,
        ceilings=(1.0, 1.0, 1.0),
        minimum_set_size=2,
        nested=False,
    )
    assert singleton.approved_shells[0] == (0,)
    assert singleton.approved_shells[2] == ()
    assert consensus.minimum_set_size == 2
    prediction, active, _ = apply_structural_prior_set(
        experts, fallback, distances, consensus
    )
    assert np.mean(active) > 0
    np.testing.assert_allclose(prediction[active], np.median(experts[:2], axis=0)[active])


def test_crossfit_rejects_harmful_bank_to_exact_fallback():
    x, health, rate, y, groups = sample_health()
    fallback = y + 0.1
    experts = np.vstack((y + 8.0, y - 8.0, y + 12.0))
    distances = np.zeros_like(experts)
    result = crossfit_structural_prior_set(
        y, groups, fallback, experts, distances,
        edges=((0.0, np.inf),) * 3,
        ceilings=(10.0, 10.0, 10.0),
    )
    assert not result.approved
    prediction, active, _ = apply_structural_prior_set(
        experts, fallback, distances, result.policy
    )
    np.testing.assert_array_equal(prediction, fallback)
    assert not np.any(active)
