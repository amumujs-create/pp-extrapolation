# PP-X structural trio: Virkler frozen protocol

**Frozen before execution:** 2026-09-12  
**Researcher:** 박진서

## Objective

Screen three prediction-function changes that exactly contain the final
support-gated PP-X as an off-state:

1. projected residual-state;
2. causal temporal self-consistency projection;
3. prior-geometry-conditioned residual.

This is a development experiment on the existing Virkler train,
validation, and sealed-test unit split. It is not a prospective result.

## Common base

- Data and unit split: `prepare_virkler()`
- PP-X: `fit_pp`, seeds 42--46, 300 epochs
- Support decay beta:
  `{0, .05, .1, .25, .5, 1, 2, 4, 8}`, selected per seed using validation
- Base prediction:
  `clip(affine + exp(-beta*d)*residual, 0, target_scale)`
- All sequence operations are causal and ordered by observed crack length
  within specimen. Test labels are never used in fitting or selection.

## Arm A: projected residual-state

For the support-gated residual `r_t`, define

\[
i_t=r_t-r_{t-1},\qquad
\Delta_t=\operatorname{clip}(\lambda\Delta_{t-1}+i_t,-\tau,\tau).
\]

The prediction is `clip(affine + Delta, 0, cap)`. At
`lambda=1, tau=infinity`, the recurrence telescopes and exactly reproduces
PP-X.

- lambda: `{0.50, 0.75, 0.90, 1.00}`
- tube fraction of target scale:
  `{0.10, 0.25, 0.50, 1.00, infinity}`

## Arm B: causal temporal self-consistency

Virkler remaining life obeys the specimen-level identity that elapsed life
plus remaining life is constant. From the PP-X prediction define

\[
z_t=\hat y_t+e_t,\quad
\bar z_t=\eta\bar z_{t-1}+(1-\eta)z_t,
\]

and blend the causal projected prediction
`bar_z_t - e_t` with PP-X using strength `gamma`.

- eta: `{0.50, 0.75, 0.90, 0.95}`
- gamma: `{0, 0.25, 0.50, 0.75, 1.00}`
- `gamma=0` exactly reproduces PP-X.

## Arm C: prior-geometry-conditioned residual

Let `p` be the frozen affine prior and `r` the support-gated PP-X residual.
Construct causal/local prior-geometry features from observed inputs and the
prior trajectory:

- standardized support distance;
- prior slope with respect to crack length;
- prior curvature;
- Paris geometry `sqrt(a / cos(pi*a/152.4))`.

A no-intercept Ridge head fits

\[
y-p
=
r\{\beta_0+\beta_d z_d+\beta_s z_s+\beta_c z_c+\beta_Pz_P\}.
\]

Validation performance is estimated with validation-unit leave-one-out;
the selected head is then fitted on all validation units and applied once
to test.

- Ridge alpha: `{0.1, 1, 10, 100, 1000}`
- the off candidate is exact PP-X.

## Selection and reporting

Each arm selects the lowest pooled validation RMSE, ties favoring the exact
PP-X/off state and then the simpler configuration. A non-off route is
approved only when validation RMSE is at least 0.5% lower than PP-X.
Otherwise that arm executes exact PP-X.

Report for every executed arm:

- pooled RMSE, MAE, and R-squared;
- unit-macro RMSE;
- units won;
- worst-unit RMSE ratio;
- unit log-RMSE-ratio bootstrap 95% CI;
- exact-off maximum absolute replay error;
- finite/range violations;
- full-row coverage.

## Promotion and rejection

An arm is a promotion candidate only if:

1. full-row coverage equals PP-X;
2. exact-off replay error is at most `1e-6`;
3. test RMSE is no more than 2% worse than PP-X;
4. worst-unit RMSE ratio is at most 1.10;
5. it beats PP-X on test RMSE, or supplies a statistically supported
   stability gain without lowering accuracy materially.

It is rejected immediately as a PP-X replacement if test RMSE is more than
5% worse than PP-X. No public-SOTA claim is made without a separately
audited equal-budget literature benchmark.
