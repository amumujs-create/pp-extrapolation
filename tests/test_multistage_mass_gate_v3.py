import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from multistage_rpt_ppx_mass_gate_v3 import apply_policy, freeze_policy


def _validation():
    groups = np.repeat(np.arange(4), 4)
    return {
        "y": np.zeros(16),
        "groups": groups,
        "progress": np.tile(np.arange(4), 4),
    }


def test_safe_alpha_mix_is_selected_when_it_improves_discrete_anchor():
    validation = _validation()
    predictions = {
        "persistence": np.full(16, 2.0),
        "PP_latest_successful": np.full(16, 1.5),
        "CCMR": np.full(16, -1.0),
        "CRT": np.full(16, 1.0),
    }
    policy = freeze_policy(validation, predictions)
    assert all(item["selected_route"] == "CCMR_CRT_mix" for item in policy)
    assert all(item["selected_alpha"] == 0.5 for item in policy)

    combined, route, alpha = apply_policy(
        policy,
        np.tile((0.125, 0.375, 0.625, 0.875), 4),
        predictions,
    )
    assert np.array_equal(combined, np.zeros(16))
    assert np.all(route == "CCMR_CRT_mix")
    assert np.all(alpha == 0.5)


def test_alpha_without_improvement_uses_exact_safe_anchor():
    validation = _validation()
    predictions = {
        "persistence": np.full(16, 2.0),
        "PP_latest_successful": np.full(16, 0.1),
        "CCMR": np.full(16, 0.2),
        "CRT": np.full(16, 0.4),
    }
    policy = freeze_policy(validation, predictions)
    assert all(
        item["selected_route"] == "PP_latest_successful"
        and item["selected_alpha"] is None
        and item["selection_reason"] == "anchor_exact_fallback"
        for item in policy
    )
