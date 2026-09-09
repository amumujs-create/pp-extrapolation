"""Frozen development presets used by the reported PP experiments."""


def battery_dual_scale_pp_config() -> dict:
    """Return the frozen boundary-prior executor used for battery cohorts.

    A new dictionary is returned so callers cannot mutate package state.
    The feature at index 2 is the standardized causal-window heterogeneity
    coordinate produced by the battery boundary-quotient adapter.
    """
    return {
        "width": 64,
        "alpha": 1000.0,
        "learning_rate": 1e-3,
        "weight_decay": 1e-2,
        "residual_bound": 2.0,
        "broad_residual_bound": 6.0,
        "local_saturation_weight": 0.4,
        "support_gate_feature": 2,
        "support_gate_threshold": 0.5,
        "support_gate_temperature": 0.25,
        "support_adaptive_saturation": True,
    }


def safety_continuation_pp_config() -> dict:
    """Return a PP executor initialized near its matched plain-NN submodel.

    The direct residual path is initialized with the same random draw as the
    standalone MLP.  A low-trust learned gate can introduce the affine prior
    when validation supports it, while retaining a continuous route back to
    the NN solution when that prior is unhelpful.
    """
    return {
        "width": 32,
        "learning_rate": 1e-3,
        "weight_decay": 2.0,
        "learned_affine_gate": True,
        "direct_residual_mixture": True,
        "affine_gate_initial_trust": 0.02,
        "residual_seed_replay": True,
        "residual_zero_init": False,
    }


def safety_continuation_trust_grid() -> tuple[dict, ...]:
    """Return nested PP routes containing the exact direct-NN submodel.

    ``fixed_affine_trust=0`` is intentionally included.  With seed replay this
    route has exactly the same initialization, optimizer and predictions as the
    matched standalone MLP.  Validation can therefore decline an unsupported
    affine prior without averaging predictions from separate model families.
    """
    common = {
        "direct_residual_mixture": True,
        "residual_seed_replay": True,
        "residual_zero_init": False,
    }
    return tuple(
        {**common, "fixed_affine_trust": trust}
        for trust in (0.0, 0.02, 0.05, 0.1, 0.2, 0.4)
    )
