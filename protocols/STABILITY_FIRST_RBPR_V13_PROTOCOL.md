# Stability-First RBPR v1.3 development protocol

**Frozen:** 2026-09-10, 박진서. Stanford and ISU are already opened; this is
retrospective robustness development, not confirmation.

## Priority

Stability and non-inferiority take precedence over the best single-cohort R2.
All experts are fixed architecture-by-seed portfolios. Test labels never select
trust, alpha, thresholds, or acceptance.

## Unit regret

For physical unit `g`,

`regret_g = (MSE_g(candidate)-MSE_g(baseline)) /
            (MSE_g(baseline)+median_g MSE_g(baseline))`.

The regularized denominator prevents a nearly perfect baseline unit from
making ratios numerically unbounded without hiding its absolute excess loss.

Validation feasibility requires:

- mean raw unit regret <= 2%;
- worst-20% raw unit-regret CVaR <= 5%;
- maximum raw unit regret <= 10%.

No hierarchical shrinkage is allowed in a hard certificate.

## Group-OOF consensus

For every held-out validation physical unit, select trust
`.02,.05,.10,.20,.40` and alpha `0,.05,...,1` on the remaining units. Candidate
score is unit-mean MSE plus `0.10 * baseline_mean_MSE * SD(unit regret)`.

Accept a prior only if all hold:

- at least 80% of folds select alpha > 0;
- one positive trust has at least 60% fold agreement;
- selected-alpha IQR <= .25;
- assembled OOF predictions satisfy all three raw-regret limits.

The deployment alpha is the 25th percentile among folds agreeing on the modal
trust, then reduced if needed on full validation. If any condition fails,
return the exact no-prior baseline portfolio.

## Ablation

Report baseline, previous fixed RBPR, continuous portfolio, and stability-first
RBPR. Include fold prior rate, trust agreement, alpha IQR, validation/test raw
mean/CVaR/max regret, pooled metrics, and exact fallback audit.
