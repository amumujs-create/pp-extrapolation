# Risk-Budgeted Prior Residual development protocol

**Frozen:** 2026-09-10, 박진서. Retrospective method development on already
opened Stanford and ISU 250 mAh cohorts; not confirmation.

## Model

`prediction = baseline + alpha * modulation(x) * (prior - baseline)`.

- Baseline: equal portfolio of all four trust-zero architectures and five seeds.
- Prior candidates: equal architecture/seed portfolio at each fixed positive
  trust `.02,.05,.10,.20,.40`.
- `alpha`: largest value on grid `0,.01,...,1` satisfying both validation
  group-risk constraints below.
- Global arm uses `modulation=1`.
- Local arm uses
  `modulation=1/(1+support_distance+seed_disagreement)`, where both terms are
  divided by their validation median. Train-only feature center/range defines
  support distance. Prediction-only seed dispersion defines disagreement.

## Empirical risk budget

For physical-unit MSE loss relative to the baseline:

1. mean group MSE may not exceed baseline mean group MSE by more than 2%;
2. worst-20% CVaR excess MSE may not exceed 2% of baseline mean group MSE.

Select the prior trust with lowest feasible validation mean group MSE; ties use
smaller alpha then smaller trust. Test labels are used only after trust, alpha,
and modulation scales are frozen.

## Ablation

Compare baseline, unconstrained validation-best prior, global risk-budget arm,
and local support/seed risk-budget arm. Remove support distance and seed
disagreement one at a time, compare mean-only versus CVaR constraints, and run
an epsilon sensitivity audit at `0,.01,.02,.05,.10`. Report pooled and unit
metrics, selected trust/alpha, validation constraints, and test outcome.
