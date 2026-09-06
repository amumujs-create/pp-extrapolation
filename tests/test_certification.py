import pytest
from pp_extrapolation import certify_categorical_regime

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
