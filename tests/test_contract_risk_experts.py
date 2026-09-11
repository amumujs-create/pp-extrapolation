import numpy as np

from pp_extrapolation.contract_risk_experts import (
    fit_contract_risk_experts,
    predict_contract_risk_experts,
)
from pp_extrapolation.contracts import trajectory_contract


def fixture():
    rng = np.random.default_rng(17)

    def rows(labels, starts):
        correction, context, y, groups = [], [], [], []
        for label, start in zip(labels, starts):
            health = start + np.arange(16) * 0.025
            health += rng.normal(0, 0.001, len(health))
            for index in range(2, 15):
                rate = health[index] - health[index - 1]
                long_rate = (health[index] - health[index - 2]) / 2
                curve = rate - long_rate
                correction.append([rate, long_rate, curve])
                context.append([
                    health[index], rate, long_rate, curve,
                    np.mean(health[index - 2:index + 1]),
                    np.std(health[index - 2:index + 1]), 1.0,
                ])
                y.append(health[index + 1])
                groups.append(label)
        return tuple(map(np.asarray, (correction, context, y, groups)))

    return (
        rows(["a", "b", "c", "d"], [0.8, 1.0, 1.2, 1.4]),
        rows(["e", "f", "g"], [0.9, 1.1, 1.3]),
    )


def test_contract_limits_expert_family_and_predicts():
    train, validation = fixture()
    contract = trajectory_contract(condition_shift=True)
    model = fit_contract_risk_experts(
        contract, *train, *validation,
        include_engression=False, portfolio_seeds=(42,),
    )
    assert tuple(expert.name for expert in model.prior_bank.experts) == (
        "linear_rate",
        "damped_acceleration",
    )
    prediction, evidence = predict_contract_risk_experts(
        model, validation[0], validation[1]
    )
    assert prediction.shape == validation[2].shape
    assert np.isfinite(evidence["strong_anchor"]).all()


def test_contract_fallback_exactly_restores_strong_anchor():
    train, validation = fixture()
    model = fit_contract_risk_experts(
        trajectory_contract(), *train, *validation,
        include_engression=False, portfolio_seeds=(42,),
    )
    shifted = validation[1].copy()
    shifted[:, 0] += 1000.0
    prediction, evidence = predict_contract_risk_experts(
        model, validation[0], shifted
    )
    assert np.array_equal(prediction, evidence["strong_anchor"])
    assert np.all(evidence["support_rejected"])
