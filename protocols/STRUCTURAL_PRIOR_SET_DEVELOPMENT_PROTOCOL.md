# Heterogeneous structural prior-set development protocol

**Frozen before execution:** 2026-09-12, 박진서

## Status

Stanford and ISU 250 mAh are opened retrospective development cohorts. No
result from this run is prospective confirmation.

## Question

When the prior bank contains **structurally distinct frozen continuations**,
not trust-strength copies of one PP-X residual, can a distance-indexed
approval set improve on the current prior-off fallback without increasing
physical-unit tail harm?

## Frozen prior bank

Every prior is fitted on train rows only. Validation labels never enter prior
parameters. Test labels never enter prior parameters or the approval policy.

1. affine ridge: group-weighted linear continuation of all source features
2. monotone health: isotonic map from normalized health to remaining life,
   with a nonnegative linear tail below the train health minimum
3. history-rate: remaining-life continuation
   \(y \approx a + b \cdot h / \max(-\dot h, \varepsilon)\)

The prior-off fallback is the already stored PP-X v1.1 trust-0 portfolio.
It is a comparator, not a fourth structural prior.

## Prior-specific distance and support ceiling

Each prior has its own distance:

- affine: axis-hull distance on all features
- monotone: health distance below the train health minimum
- history-rate: axis-hull distance on \((h, \dot h)\)

Validation distances define 0–50%, 50–80%, and 80–100% shells and a 99%
support ceiling. A prior is ineligible on a row whose own distance exceeds
its ceiling, even if that row falls in the last finite shell.

## Approval rule

Within each prior-specific shell a continuation is retained only when:

- at least two physical validation units are represented
- mean physical-unit MSE is strictly lower than the fallback
- worst-20% excess physical-unit MSE is no greater than 2% of fallback loss
- a farther shell cannot reintroduce a prior rejected in a nearer shell

The prediction is the row-wise median of priors that remain eligible on that
row. If fewer than two priors remain, the row uses the single approved prior
when one exists, otherwise exact fallback. A separate consensus-only arm
requires two approved priors and otherwise uses exact fallback.

The 90th-percentile validation width of the retained envelope is frozen as a
disagreement limit. Wider test envelopes revert to exact fallback.

Every physical validation unit is predicted by a policy fitted without that
unit. The primary nested set is approved only with strictly positive
cross-fitted group-MSE gain and mean/worst-20% harm ratios within 2%.

## Arms

1. fallback
2. affine only
3. monotone only
4. history-rate only
5. mean ensemble of all three priors
6. best single prior by validation group-MSE
7. consensus-only prior set
8. nested approved set, allowing a singleton (primary)

## Decision

Promotion requires improvement over both the fallback and the existing
continuous portfolio, not merely exact fallback behavior. A new untouched
cohort is required for confirmation.
