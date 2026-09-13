# Falsification-Calibrated Credal Conformal PP-X protocol

**Frozen before execution:** 2026-09-13, 박진서

## Status

This is a retrospective structural-development audit on the 12 opened
common-backbone domains. It is not prospective confirmation.

## Model

Let `f0` be the direct fallback and `fPP` the PP candidate. Validation
physical units produce a one-sided 95% upper confidence bound for
`log(RMSE_PP / RMSE_0)`.

- If the upper bound is negative, deploy `fPP` and set authority to `{1}`.
- Otherwise deploy the exact fallback `f0` and set credal authority to `[0,1]`.

The prediction set is the Minkowski sum of the authority segment and the
legacy support-scaled block-conformal radius:

```text
{f0 + a(fPP - f0): a in A} + [-q s(x), q s(x)].
```

Because `1` belongs to every authority set, this interval contains the legacy
PP conformal interval for every row. Coverage therefore cannot decrease for
any fixed evaluation sample. This is a set-inclusion guarantee, not a claim
of distribution-free validity under arbitrary domain shift.

## Frozen settings

- nominal row coverage: 90%
- support scale: `median(abs(y_validation - fPP_validation)) * (1 + distance)`
- physical-unit block score: within-unit 90th percentile
- finite-sample higher quantile across validation units
- falsification confidence: 95%
- physical-unit bootstrap replicates: 20,000
- nonnegative output lower bound: zero
- no threshold grid or test-outcome model selection

## Comparisons

- point prediction: always direct, always PP, certificate-routed FCC-PPX
- interval: current PP block-conformal versus FCC-PPX credal conformal
- report equal-domain and pooled coverage, per-domain non-inferiority, width
  inflation, point log-RMSE gain, and false-harm domains

## Promotion rule

- point mean unit log-RMSE improvement must be positive
- no point false-harm domain among activated PP routes
- updated interval coverage must be at least current coverage in every domain
- interval nesting violations must be exactly zero
- width inflation must be reported, not optimized on test outcomes
- an unopened cohort remains required before a confirmatory novelty claim
