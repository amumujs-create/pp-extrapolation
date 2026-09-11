# Distance-indexed prior-set consensus development protocol

**Frozen before execution:** 2026-09-12, 박진서

## Status

Stanford and ISU 250 mAh are opened retrospective development cohorts. No
result from this run is prospective confirmation.

## Question

Can PP-X improve on a single validation-selected prior by retaining a
distance-indexed **set** of validation-supported prior continuations, using
their consensus only when their prediction envelope remains supported?

## Candidate prior continuations

The already fitted PP-X v1.1 trust portfolios are reused without retraining:

- prior-off fallback: trust 0
- candidate continuations: trust .02, .05, .10, .20, .40
- each portfolio averages the same four architecture configurations and five
  seeds already stored in the frozen v1.1 artifact

These are continuation-strength candidates, not claims of five independent
physical laws.

## Distance-indexed prior set

Axis-support distance is fitted on train rows only and normalized by the
median positive validation distance. Validation distances define 0–50%,
50–80%, and 80–100% shells.

Within each shell, a candidate continuation is retained only when:

- at least two physical validation units are represented;
- mean physical-unit MSE is lower than the fallback;
- worst-20% excess physical-unit MSE is no greater than 2% of fallback loss.

The prediction is the row-wise median of the retained continuations. The
90th-percentile validation width of the retained prediction envelope is frozen
as a disagreement limit. A test row uses exact fallback if:

- no continuation remains in its shell;
- its prior-set envelope is wider than the frozen limit; or
- its distance lies outside all supported shells.

## Primary and ablation arms

1. fallback
2. best single continuation selected by validation group-MSE
3. existing continuous convex portfolio
4. global prior-set consensus
5. free shell-specific prior sets
6. nested shell prior sets: farther shells cannot introduce a continuation
   rejected in a nearer shell (primary)

Every physical validation unit is predicted by a policy fitted without that
unit. The final primary policy is approved only with strictly positive
cross-fitted group-MSE gain and mean/worst-20% harm ratios within 2%.

## Decision

Promotion requires improvement over the existing continuous portfolio and
current PP-X, not merely exact fallback behavior. A new untouched cohort is
required for confirmation.
