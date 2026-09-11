# Contracted time-to-boundary flow development protocol

**Frozen before execution:** 2026-09-12, 박진서

## Status

Stanford and ISU 250 mAh are opened retrospective development cohorts.
This is a model-concept challenger, not prospective confirmation.

## Boundary contract

- ordered coordinate: normalized health, feature column 0
- local rate: one-cycle health difference, feature column 1
- failure boundary: normalized health 0.80
- output: nonnegative remaining cycles to first boundary crossing

Rows, physical-unit splits, and tail filters are identical to the existing
PP-X v1.1 development artifacts.

## Model

CTBF learns a positive degradation velocity and computes remaining time by
quadrature:

\[
\widehat T(x)
=
\int_{0.8}^{h(x)}
\frac{du}{v_\theta(u,c(x))}.
\]

The primary weak-rate model uses the observed local degradation magnitude as
an incomplete prior:

\[
v_\theta(u,c)
=
v_{\mathrm{weak}}(c)
\exp\left[
B\tanh r_\theta(u,c)
\right],
\qquad
v_{\mathrm{weak}}=\max(-\Delta h,\epsilon).
\]

Operationally the expression is implemented in log velocity so positivity is
exact. Context is held fixed along the quadrature ray while the health
coordinate moves from the boundary to the current state.

## Arms

1. observed-rate quotient \((h-0.8)/\max(-\Delta h,\epsilon)\)
2. direct positive velocity integral without a rate prior
3. weak-rate log-velocity residual integral (primary)
4. matched direct MLP
5. stored PP-X v1.1 continuous portfolio

## Fitting

- seeds: 42, 43, 44
- widths: 16, 32
- log-velocity residual bounds: 0.5, 1.0, 2.0
- AdamW learning rate: 0.001
- weight decay: 0.1
- maximum epochs: 350
- patience: 60
- quadrature points: 24
- physical-unit-balanced training loss

Each arm receives the same train features and validation checkpoint labels.
The selected configuration minimizes validation physical-unit mean MSE;
ties choose the smaller width and smaller residual bound.

## Evaluation

- pooled R²/RMSE
- unit-macro R²
- per-unit RMSE win fraction against direct MLP and stored PP-X
- physical-unit bootstrap RMSE difference
- boundary consistency
- direct velocity versus weak-rate residual ablation

## Promotion

CTBF is promoted only if the primary weak-rate model:

- improves both datasets over matched direct MLP
- improves at least one dataset over stored PP-X without material loss on the
  other
- beats direct velocity, demonstrating value from the weak rate prior
- has positive physical-unit bootstrap evidence

Failure is retained as a negative model-development result.
