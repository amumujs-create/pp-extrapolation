# CCMR v2.2 small-cohort evidence route firewall

Frozen 2026-09-11 by 박진서 before v2.2 implementation.

Alloy A and Multi-Stage RPT test outcomes, errors, and competitor results are
excluded from route development and threshold selection. The stochastic v2.1
candidate is rejected and is not part of v2.2.

CCMR v2.2 keeps the frozen v2.0 causal dynamics predictor unchanged. It adds
one route for validation cohorts too small for the replicated 10-unit stable
certificate.

The `small_crossfit_bank` route is eligible only when:

- validation physical units are at least 3 and fewer than 10;
- train-group cross-fit expert sign/dispersion consensus is already active on
  at least 50% of validation rows;
- validation unit-macro RMSE improvement is at least 10%;
- validation raw mean/CVaR20/maximum regret satisfy 0%/1%/2%;
- the selected deployment mass is positive.

The route deploys the existing v2.0 bank with its row-level consensus and
context-support fallback. It does not relax support, risk, residual bounds, or
the exact persistence fallback.

For 10 or more validation units, the frozen v2.0 stable/cautious rules remain
unchanged. For fewer than 3 units, only the cautious causal route or exact
fallback is permitted.

Promotion requires at least one strict non-holdout improvement over v2.0,
zero false accepts, maximum raw test regret no greater than 2%, and exact
fallback replay error zero. Only after promotion and SHA-256 freezing may the
two excluded holdouts be replayed once.
