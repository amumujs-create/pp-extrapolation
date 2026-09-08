# SJTU fatigue-crack small external cohort protocol

Frozen before downloading or inspecting `4. Cycles vs Crack Length.xlsx`.

## Source and scope

- Dataset: *Image-Driven Prediction of Fatigue Crack Growth in Metal Materials via
  Spatiotemporal Neural Network*, Mendeley Data V1.
- DOI: https://doi.org/10.17632/dywwnjv22h.1
- Licence displayed by the publisher: CC BY 4.0.
- Only the publisher's 42.1 KB workbook `4. Cycles vs Crack Length.xlsx` is used.
  The 8.01 GB image/DIC/model bundle is not downloaded.
- The publisher states that the workbook contains cycle--crack-length relations
  for four measured specimens.

This cohort is new to PP and is intended as a small non-battery feasibility
replication.  Four specimens are insufficient for a strong population-level
external-validation claim.

## Deterministic parsing and split

1. Read every nonempty numeric cycle--crack-length trajectory in workbook order.
2. Preserve the workbook order; do not reorder specimens using outcomes.
3. Specimens 1--2 are training, specimen 3 is validation, specimen 4 is the
   locked test specimen.
4. Duplicate trajectories are checked by SHA-256 of the cleaned numeric arrays.
5. If fewer than four usable trajectories or a duplicate is found, report
   `inconclusive` without changing the split.

## Failure boundary and target

- Crack length is the ordered degradation coordinate.
- A common failure boundary is the minimum terminal crack length among specimens
  1--3.  It is therefore determined entirely from development specimens.
- Each trajectory is truncated at its first crossing of that boundary.
- RUL is the remaining fatigue-cycle count until that crossing.
- If the locked test specimen never crosses the development boundary or has
  fewer than 10 pre-boundary measurements, report `inconclusive`.

## Features and causal contract

At each point the model receives current crack length, boundary margin, elapsed
cycles, finite-difference crack-growth rates over available lags, and causal
expanding/rolling summaries.  No future test measurement, specimen ID, final
lifetime, or test label is an input.

The primary PP is the log-boundary-quotient architecture frozen after the
Ferrara development:

`RUL = margin * expm1(affine_log_quotient + bounded_NN_residual)`.

The Ferrara vibration-onset gate is inapplicable because this workbook contains
crack length rather than vibration.  It is disabled by the prior contract.

## Model selection and evaluation

- Alpha and stopping epoch are selected using specimen 3 only.
- Final PP refit uses specimens 1--3 with the selected settings.
- Seeds: 42--46; the five predictions are averaged.
- Matched comparators: boundary-rate, Ridge, plain MLP, and affine-only
  log-quotient PP, using the same causal rows.
- Test evaluation uses the last 30% of specimen 4's pre-boundary trajectory;
  all earlier measurements are available only through causal features.
- Primary metric: pooled R² on the locked specimen tail.  Secondary metrics:
  RMSE, MAE, seed dispersion, and ordered-coordinate hull outside fraction.

## Predeclared success

Success requires all of:

1. test-tail pooled R² greater than zero;
2. PP RMSE lower than matched plain MLP RMSE;
3. at least 10 test-tail observations;
4. no data-quality gate failure.

All outcomes are retained, including failure or inconclusive status.
