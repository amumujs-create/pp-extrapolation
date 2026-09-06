import pytest
from pp_extrapolation import certify_categorical_regime, certify_extrapolation

def test_unseen_regime_abstains_without_labels():
    c=certify_categorical_regime([1,1,1],[2,2])
    assert not c.accepted
    assert c.covered_fraction==0
    assert c.unseen_levels==("2",)

def test_covered_regimes_pass():
    c=certify_categorical_regime(["a","b"],["b","a"])
    assert c.accepted and c.covered_fraction==1

def test_empty_context_rejected():
    with pytest.raises(ValueError): certify_categorical_regime([], [1])

def test_extrapolation_certificate_requires_all_checks():
    accepted=certify_extrapolation(validation_r2=.4,baseline_relative_mse_gain=.03,
        normalized_seed_disagreement=.1,regime_covered=True)
    assert accepted.accepted
    rejected=certify_extrapolation(validation_r2=.4,baseline_relative_mse_gain=.01,
        normalized_seed_disagreement=.1,regime_covered=True)
    assert not rejected.accepted
    assert rejected.reasons == ("beats_baseline_margin",)

def test_extrapolation_certificate_rejects_unstable_or_uncovered():
    cert=certify_extrapolation(validation_r2=.4,baseline_relative_mse_gain=.03,
        normalized_seed_disagreement=.3,regime_covered=False)
    assert not cert.accepted
    assert set(cert.reasons)=={"regime_covered","seed_stable"}
