# Falsification-aware residual authority head protocol

**Frozen before execution:** 2026-09-12, 박진서

## Status

This is a leave-one-domain-out retrospective development experiment on the
12 opened common-backbone datasets. It is not prospective confirmation.

## Model

For a PP candidate \(f_{\mathrm{PP}}\) and exact direct fallback \(f_0\), a
learned head estimates the upper bound of physical-unit log regret:

\[
\widehat U(x)
=
\widehat{\mathbb E}
\left[
\log\frac{\mathrm{RMSE}_{\mathrm{PP}}}{\mathrm{RMSE}_0}
\mid z(x)
\right]
+q_{.95}.
\]

The residual authority is

\[
a(x)=
\operatorname{clip}
\left(
\frac{-\widehat U(x)}{s_R},0,1
\right),
\qquad
\hat y=f_0+a(x)(f_{\mathrm{PP}}-f_0).
\]

Thus the PP correction has zero authority when its predicted regret upper
bound is nonnegative. A binary ablation uses \(a=1[\widehat U<0]\).

## Outcome-free head inputs

Features are computed separately for each physical unit without labels:

- log row count
- mean and 90th percentile support distance
- fraction outside train support
- direct and PP seed disagreement, normalized by direct prediction IQR
- mean and 90th percentile PP-versus-direct correction magnitude
- signed correction mean
- PP/direct prediction-range ratio

No target, residual, test score, dataset identity, or test-time fitted
parameter enters these features.

## Training target

The head target is validation-unit

\[
r_u=\log
\frac{\operatorname{RMSE}_{u,\mathrm{PP}}}
     {\operatorname{RMSE}_{u,\mathrm{direct}}}.
\]

Ridge alpha is selected by inner leave-one-domain-out squared error among
the 11 training domains. Domain-balanced unit weights prevent large cohorts
from dominating.

The upper-bound offset \(q_{.95}\) is the 95th percentile of inner
leave-one-domain-out residuals \(r_u-\hat r_u\). The held-out domain never
contributes labels to its model, feature scaling, alpha selection, authority
scale, or residual calibration.

## Outer evaluation

Each of the 12 domains is held out once. The head is trained on validation
units from the other 11 domains and applied to outcome-free test-unit
features of the held-out domain. Test outcomes are read only for scoring.

## Arms

1. direct fallback
2. always PP
3. existing within-domain 95% bootstrap certificate
4. learned binary authority
5. learned continuous authority (primary)
6. test oracle, nondeployable

## Promotion

Promotion requires:

- positive equal-domain mean unit log-RMSE improvement
- no more false-harm domains than the existing certificate
- nonzero authority in more than one held-out domain
- better performance than binary authority, showing value from continuous
  residual authority

Independent prospective confirmation remains mandatory.
