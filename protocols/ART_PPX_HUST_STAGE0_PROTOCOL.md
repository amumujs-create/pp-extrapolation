# ART-PPX HUST Stage-0 protocol

**Frozen before execution:** 2026-09-12, 박진서

## Question

Can a support-boundary anchored residual-derivative transport improve the
final HUST PP-X route without sacrificing its exact coverage or stability?

## Base route

Replay the final HUST PP-X:

- `fit_pp`, width 64, affine anchor weight 0.1
- seeds 42–46
- validation-group-LOO Ridge regime transport over the frozen
  `(kind, alpha)` grid

The ART correction is fitted only after this route is fixed.

## Anchored residual transport

Let \(d(x)=\max(h_{\min}^{train}-h(x),0)\). The correction is the integral of
a residual derivative from the train-support boundary:

\[
\Delta(x)
=
\int_0^{d(x)}
\left[
\beta_0+\beta_1u+\beta_c^\top c+
u\beta_{uc}^\top c
\right]du.
\]

The implemented integrated design is

\[
[d,\ d^2/2,\ dc,\ d^2c/2].
\]

It has no intercept and is exactly zero at \(d=0\). Ridge coefficients are
fitted to PP-X residuals on validation groups. A bounded correction is

\[
\Delta_B=B\tanh(\Delta/B).
\]

Final prediction:

\[
\widehat y_{ART}
=
\operatorname{clip}(\widehat y_{PPX}+a\Delta_B,0,y_{\max}).
\]

At \(a=0\), ART-PPX exactly reproduces PP-X.

## Validation-only selection

Use group leave-one-out predictions over:

- context: state `(0,1)`, rate `(2,3)`, all `(0,1,2,3)`
- Ridge alpha: `{0.1, 1, 10, 100, 1000}`
- correction bound fraction: `{0.02, 0.05, 0.10, 0.20}`
- authority: `{0, 0.25, 0.5, 0.75, 1}`

Select the lowest validation group-LOO RMSE, with ties preferring smaller
authority, smaller bound, fewer context variables, and larger Ridge penalty.

## Approval

A nonzero authority is approved only if all hold against exact PP-X fallback:

1. validation RMSE improves by at least 2%;
2. at least 60% of validation units improve;
3. worst validation-unit RMSE ratio is at most 1.05;
4. 20,000-draw unit bootstrap RMSE-difference CI lower bound is positive.

Otherwise set authority to zero.

## Test and early stop

Fit the selected derivative coefficients on all validation rows and apply once
to test. Stop this model family before MICH if:

- exact fallback replay error exceeds \(10^{-6}\);
- approved ART test RMSE exceeds PP-X by more than 5%;
- worst test-unit RMSE ratio exceeds 1.10;
- no nonzero route is approved.

The last condition treats safe fallback as insufficient model-novelty evidence
for continuing this challenger.
