# Prior falsification certificate retrospective protocol

**Frozen before execution:** 2026-09-12, 박진서

## Status

The 12 common-backbone datasets are opened retrospective development data.
This experiment evaluates a fixed certificate and is not prospective
confirmation.

## Question

Does a physical-unit bootstrap upper confidence bound on fallback-relative
prior regret remove false prior approvals more reliably than the current
PP-X validation policy?

## Evidence

For each validation physical unit \(u\),

\[
r_u=\log
\frac{\operatorname{RMSE}_{u,\mathrm{PP}}}
     {\operatorname{RMSE}_{u,\mathrm{fallback}}}.
\]

Positive values falsify the prior-residual route. The certificate is the
95% one-sided bootstrap upper bound \(U_{.95}\) of the mean unit log regret.
Resampling is performed over physical units, never rows or random seeds.

## Frozen decision

The primary policy first applies the existing paper thresholds:

- validation MSE improvement at least 2%
- unit win fraction at least 60%
- worst-unit RMSE ratio no greater than 1.10

It approves the PP route only if the additional falsification certificate
satisfies \(U_{.95}<0\). Otherwise it uses exact direct fallback.

No certificate threshold or confidence level is selected using test outcomes.

## Comparators

1. always direct
2. always PP
3. simple validation-MSE selection
4. frozen paper PP-X policy
5. certificate only
6. paper PP-X plus falsification certificate (primary)
7. test oracle, explicitly nondeployable

## Evaluation

- decision accuracy across 12 datasets
- false accepts and false rejects
- equal-dataset mean selected test log-RMSE improvement
- dataset bootstrap 95% CI
- accepted-dataset wins and losses

Test outcomes are scored only after all validation decisions have been
materialized in memory.

## Promotion

The certificate is a development candidate only if it reduces false accepts
without producing a negative selected test effect. A new untouched domain is
required before it can replace the frozen paper policy.
