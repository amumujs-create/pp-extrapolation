from pp_extrapolation import battery_dual_scale_pp_config, safety_continuation_pp_config


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
