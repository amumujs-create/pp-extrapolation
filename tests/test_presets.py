from pp_extrapolation import (battery_dual_scale_pp_config,
                              robust_generalization_policy_config,
                              safety_continuation_pp_config,
                              safety_continuation_trust_grid)


def test_presets_return_fresh_dictionaries():
    first = battery_dual_scale_pp_config()
    second = battery_dual_scale_pp_config()
    first["width"] = -1
    assert second["width"] == 64


def test_safety_continuation_contains_plain_nn_path():
    config = safety_continuation_pp_config()
    assert config["direct_residual_mixture"] is True
    assert config["residual_seed_replay"] is True
    assert config["residual_zero_init"] is False
    assert 0 < config["affine_gate_initial_trust"] < 0.05


def test_safety_trust_grid_contains_exact_nn_and_only_nested_routes():
    grid = safety_continuation_trust_grid()
    assert grid[0]["fixed_affine_trust"] == 0.0
    assert all(row["direct_residual_mixture"] for row in grid)
    assert all(row["residual_seed_replay"] for row in grid)
    assert [row["fixed_affine_trust"] for row in grid] == sorted(
        row["fixed_affine_trust"] for row in grid
    )


def test_robust_generalization_policy_is_conservative():
    policy = robust_generalization_policy_config()
    assert policy["min_relative_gain"] == 0.02
    assert policy["confidence"] == 0.95
    assert policy["bootstrap_replicates"] >= 5000
