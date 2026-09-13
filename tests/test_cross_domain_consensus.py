import numpy as np

from pp_extrapolation.cross_domain_consensus import (
    consensus_representation,
    fit_cross_domain_consensus_head,
    normalized_residual_target,
    predict_cross_domain_consensus,
)


def training_sample():
    features, targets, domains = [], [], []
    for index, name in enumerate(("a", "b", "c", "d")):
        base = np.linspace(1.0, 5.0, 12) + index
        seeds = np.stack([base - 0.2, base, base + 0.2])
        rep = consensus_representation(seeds)
        y = base + 0.1 * rep.features[:, 0]
        features.append(rep.features)
        targets.append(normalized_residual_target(y, rep))
        domains.extend([name] * len(y))
    return np.concatenate(features), np.concatenate(targets), np.asarray(domains)


def test_representation_is_affine_scale_free():
    base = np.linspace(2.0, 10.0, 20)
    predictions = np.stack([base - 0.3, base, base + 0.3])
    first = consensus_representation(predictions)
    second = consensus_representation(7.0 * predictions + 11.0)
    np.testing.assert_allclose(first.features, second.features, atol=1e-10)
    assert abs(first.q90_seed_disagreement - second.q90_seed_disagreement) < 1e-12


def test_consensus_shrinkage_preserves_mean_and_halves_spread():
    x, target, domains = training_sample()
    model = fit_cross_domain_consensus_head(x, target, domains)
    base = np.linspace(1.0, 5.0, 12)
    seeds = np.stack([base - 1.0, base, base + 1.0])
    result = predict_cross_domain_consensus(
        model, seeds, disagreement_cutoff=10.0, seed_shrinkage=0.5
    )
    np.testing.assert_allclose(result.ensemble, np.mean(seeds, axis=0))
    np.testing.assert_allclose(
        np.std(result.seeds, axis=0), 0.5 * np.std(seeds, axis=0)
    )


def test_active_head_changes_ensemble_without_target_input():
    x, target, domains = training_sample()
    model = fit_cross_domain_consensus_head(x, target, domains)
    base = np.linspace(1.0, 5.0, 12)
    seeds = np.stack([base - 1.0, base, base + 1.0])
    result = predict_cross_domain_consensus(
        model, seeds, disagreement_cutoff=0.0
    )
    assert result.active_residual
    assert np.any(np.abs(result.correction) > 0)
    np.testing.assert_allclose(result.ensemble, np.mean(result.seeds, axis=0))


def test_fewer_than_three_training_domains_is_rejected():
    x, target, domains = training_sample()
    keep = np.isin(domains, ["a", "b"])
    try:
        fit_cross_domain_consensus_head(x[keep], target[keep], domains[keep])
    except ValueError:
        pass
    else:
        raise AssertionError("two domains must be insufficient")
