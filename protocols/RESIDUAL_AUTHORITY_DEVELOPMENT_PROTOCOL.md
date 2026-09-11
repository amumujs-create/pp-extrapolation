# Residual-authority retrospective development protocol

**Frozen before execution:** 2026-09-12, 박진서

## Status

Stanford and ISU 250 mAh have already been opened. This run is retrospective
method development and cannot provide prospective confirmation.

## Question

Does a validation-calibrated, distance-indexed authority over a frozen-prior
residual improve over:

1. the matched prior-off MLP fallback;
2. the frozen affine prior alone;
3. the full prior-residual candidate;
4. one global residual authority;
5. unconstrained shell-specific authority?

## Data separation

The existing outer train, validation, and test unit splits are unchanged.
Outer-train physical units are deterministically split again:

- four of every five sorted units: model fitting;
- every fifth sorted unit: checkpoint selection.

The original outer validation labels are used only for cross-fitted authority
calibration and model-configuration selection. Test labels are read only after
the complete authority policy is frozen.

## Candidate models

- widths: 16, 32
- learning rate: 0.001
- weight decay: 2.0
- seeds: 42, 43, 44
- maximum epochs: 300
- patience: 50
- prior-residual candidate: frozen affine initialization plus nonlinear
  residual, with residual decay 0.3
- fallback: matched two-hidden-layer tanh MLP with the same width, optimization
  budget, seed set, and inner split

## Distance and authority

Axis-support distance is fitted on model-fitting rows only and normalized by
the median positive outer-validation distance. Shell edges are the 0th, 50th,
80th, and 100th percentiles of normalized outer-validation distance; repeated
edges are collapsed and the final edge is infinity.

Authority candidates are `0, .25, .50, .75, 1`. For the primary arm:

- authority is non-increasing with distance;
- after the first fallback shell, all farther shells also use fallback;
- every active shell requires at least two physical validation units;
- mean and worst-20% physical-unit excess MSE versus fallback must each be no
  greater than 2%.

Each physical validation unit is predicted once by a policy fitted without
that unit. The final policy is approved only if cross-fitted relative group-MSE
gain is strictly positive and both cross-fitted harm ratios are within 2%.

## Arms

1. fallback
2. prior only
3. full residual
4. global authority: one shell
5. unconstrained shell authority
6. monotone shell authority (primary)

Configuration selection uses cross-fitted group-mean validation MSE, with
failed approval ranked behind approved configurations.

## Reporting

- pooled R², RMSE, and MAE
- unit-macro R²
- selected authority per shell
- cross-fitted validation gain
- mean and worst-20% test excess group-MSE ratios
- exact fallback equality for rejected/unsupported shells

No arm may be promoted from this run without a new untouched cohort.
