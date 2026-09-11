"""Fold-consensus, raw-regret safety policy for PP prior residuals."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .risk_budgeted_prior import apply_prior_residual, group_mse


@dataclass(frozen=True)
class StabilityFirstDecision:
    accepted: bool
    selected_trust: float
    alpha: float
    reason: str
    fold_prior_rate: float
    modal_trust_agreement: float
    fold_alpha_iqr: float
    oof_mean_regret: float
    oof_cvar_regret: float
    oof_max_regret: float
    validation_mean_regret: float
    validation_cvar_regret: float
    validation_max_regret: float
    fold_selections: tuple[tuple[float, float], ...]


def unit_regret(
    y: np.ndarray,
    groups: np.ndarray,
    baseline: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    """Regularized relative excess MSE for each physical unit."""
    baseline_loss = group_mse(y, baseline, groups)
    candidate_loss = group_mse(y, prediction, groups)
    stabilizer = max(float(np.median(baseline_loss)), 1e-12)
    return (candidate_loss - baseline_loss) / (baseline_loss + stabilizer)


def raw_unit_regret(
    y: np.ndarray,
    groups: np.ndarray,
    baseline: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    """Raw relative excess MSE for each physical unit."""
    baseline_loss = group_mse(y, baseline, groups)
    candidate_loss = group_mse(y, prediction, groups)
    denominator = np.maximum(baseline_loss, 1e-12)
    return (candidate_loss - baseline_loss) / denominator


def regret_summary(regret: np.ndarray, cvar_fraction=0.20):
    regret = np.asarray(regret, dtype=np.float64)
    if regret.ndim != 1 or len(regret) == 0:
        raise ValueError("regret must be a nonempty vector")
    count = max(1, int(np.ceil(cvar_fraction * len(regret))))
    return (
        float(np.mean(regret)),
        float(np.mean(np.sort(regret)[-count:])),
        float(np.max(regret)),
    )


def _feasible(summary, mean_cap, cvar_cap, max_cap):
    return (
        summary[0] <= mean_cap + 1e-12
        and summary[1] <= cvar_cap + 1e-12
        and summary[2] <= max_cap + 1e-12
    )


def _select_fold(
    y, groups, baseline, priors, trusts, alpha_grid, mean_cap, cvar_cap,
    max_cap, cvar_fraction, instability_penalty,
):
    baseline_scale = max(float(np.mean(group_mse(y, baseline, groups))), 1e-12)
    candidates = [(1.0, 0.0, 0.0)]
    for index, trust in enumerate(trusts):
        for alpha in alpha_grid[1:]:
            prediction = apply_prior_residual(
                baseline, priors[index], float(alpha)
            )
            regret = unit_regret(y, groups, baseline, prediction)
            summary = regret_summary(regret, cvar_fraction)
            if not _feasible(summary, mean_cap, cvar_cap, max_cap):
                continue
            score = (
                np.mean(group_mse(y, prediction, groups)) / baseline_scale
                + instability_penalty * np.std(regret)
            )
            candidates.append((float(score), float(alpha), float(trust)))
    _, alpha, trust = min(candidates, key=lambda row: (
        row[0], row[1], row[2]
    ))
    return trust, alpha


def select_stability_first_prior(
    y: np.ndarray,
    groups: np.ndarray,
    baseline: np.ndarray,
    prior_predictions: np.ndarray,
    trusts: np.ndarray,
    *,
    alpha_grid=np.linspace(0.0, 1.0, 21),
    mean_cap=0.02,
    cvar_cap=0.05,
    max_cap=0.10,
    cvar_fraction=0.20,
    instability_penalty=0.10,
    min_prior_rate=0.80,
    min_trust_agreement=0.60,
    max_alpha_iqr=0.25,
    min_groups=5,
) -> StabilityFirstDecision:
    """Select a conservative prior only under physical-unit fold consensus."""
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)
    baseline = np.asarray(baseline, dtype=np.float64)
    priors = np.asarray(prior_predictions, dtype=np.float64)
    trusts = np.asarray(trusts, dtype=np.float64)
    alpha_grid = np.asarray(alpha_grid, dtype=np.float64)
    if (
        y.ndim != 1
        or groups.shape != y.shape
        or baseline.shape != y.shape
        or priors.ndim != 2
        or priors.shape[1] != len(y)
        or trusts.shape != (len(priors),)
    ):
        raise ValueError("outcomes, groups, baseline, priors, and trusts must align")
    labels = np.unique(groups)
    if min_groups < 3 or len(labels) < min_groups:
        raise ValueError(
            f"at least {max(3, min_groups)} physical units are required"
        )
    if (
        np.any(trusts <= 0)
        or alpha_grid[0] != 0
        or np.any(np.diff(alpha_grid) <= 0)
        or alpha_grid[-1] > 1
    ):
        raise ValueError("positive trusts and ordered alpha grid are required")

    selections = []
    oof = np.empty_like(y)
    for held_out in labels:
        keep = groups != held_out
        trust, alpha = _select_fold(
            y[keep], groups[keep], baseline[keep], priors[:, keep], trusts,
            alpha_grid, mean_cap, cvar_cap, max_cap, cvar_fraction,
            instability_penalty,
        )
        selections.append((trust, alpha))
        take = ~keep
        if alpha == 0:
            oof[take] = baseline[take]
        else:
            index = int(np.flatnonzero(np.isclose(trusts, trust))[0])
            oof[take] = apply_prior_residual(
                baseline[take], priors[index, take], alpha
            )

    selected = np.asarray(selections)
    positive = selected[:, 1] > 0
    prior_rate = float(np.mean(positive))
    if np.any(positive):
        values, counts = np.unique(selected[positive, 0], return_counts=True)
        modal_trust = float(values[np.argmax(counts)])
        modal = positive & np.isclose(selected[:, 0], modal_trust)
        agreement = float(np.mean(modal))
        modal_alpha = selected[modal, 1]
        alpha_iqr = float(np.subtract(*np.quantile(modal_alpha, (0.75, 0.25))))
    else:
        modal_trust = agreement = alpha_iqr = 0.0
        modal_alpha = np.asarray([0.0])

    oof_summary = regret_summary(
        unit_regret(y, groups, baseline, oof), cvar_fraction
    )
    conditions = (
        (prior_rate >= min_prior_rate, "insufficient fold prior consensus"),
        (agreement >= min_trust_agreement, "insufficient trust consensus"),
        (alpha_iqr <= max_alpha_iqr, "unstable fold alpha"),
        (
            _feasible(oof_summary, mean_cap, cvar_cap, max_cap),
            "OOF raw-regret certificate failed",
        ),
    )
    failure = next((reason for passed, reason in conditions if not passed), None)
    if failure is not None:
        return StabilityFirstDecision(
            accepted=False,
            selected_trust=0.0,
            alpha=0.0,
            reason=failure,
            fold_prior_rate=prior_rate,
            modal_trust_agreement=agreement,
            fold_alpha_iqr=alpha_iqr,
            oof_mean_regret=oof_summary[0],
            oof_cvar_regret=oof_summary[1],
            oof_max_regret=oof_summary[2],
            validation_mean_regret=0.0,
            validation_cvar_regret=0.0,
            validation_max_regret=0.0,
            fold_selections=tuple((float(t), float(a)) for t, a in selections),
        )

    index = int(np.flatnonzero(np.isclose(trusts, modal_trust))[0])
    alpha_cap = float(np.quantile(modal_alpha, 0.25))
    final_alpha = 0.0
    final_summary = (0.0, 0.0, 0.0)
    for alpha in np.linspace(0.0, alpha_cap, 101):
        prediction = apply_prior_residual(
            baseline, priors[index], float(alpha)
        )
        summary = regret_summary(
            unit_regret(y, groups, baseline, prediction), cvar_fraction
        )
        if _feasible(summary, mean_cap, cvar_cap, max_cap):
            final_alpha = float(alpha)
            final_summary = summary
    accepted = final_alpha > 0
    return StabilityFirstDecision(
        accepted=accepted,
        selected_trust=modal_trust if accepted else 0.0,
        alpha=final_alpha,
        reason="accepted" if accepted else "full-validation certificate failed",
        fold_prior_rate=prior_rate,
        modal_trust_agreement=agreement,
        fold_alpha_iqr=alpha_iqr,
        oof_mean_regret=oof_summary[0],
        oof_cvar_regret=oof_summary[1],
        oof_max_regret=oof_summary[2],
        validation_mean_regret=final_summary[0],
        validation_cvar_regret=final_summary[1],
        validation_max_regret=final_summary[2],
        fold_selections=tuple((float(t), float(a)) for t, a in selections),
    )
