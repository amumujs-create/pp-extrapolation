# UConn–ISU–ILCC 1.2 Ah LFP cohort — frozen PP-X v1.2 protocol

**Frozen:** 2026-09-10, 박진서. Written before cloning the processed-data
repository or reading any capacity trajectory, cell lifetime, split outcome,
validation loss, or test score.

## Cohort and untouched status

- Source: UConn–ISU–ILCC LFP/Gr Battery Aging Dataset, Nowacki et al. (2025).
- Public metadata: 64 LithiumWerks APR18650M1B cells, nominal capacity 1.2 Ah,
  11 aging conditions, cycled to 80% remaining capacity.
- Data path: processed LFP capacity data from the authors' public repository
  `REIL-UConn/rapid-soh-estimation-from-short-pulses`.
- Raw 1 Hz archives are not required because the task uses only causal RPT
  capacity trajectories and fixed cycling-condition metadata.
- Workspace search found no prior PP/PAE use of UConn, REIL, Nowacki, or this
  archive. This is a model-level external cohort, not a privately sequestered
  clinical-style confirmation.

## Eligibility and split

1. Use every processed LFP cell with at least 20 finite, ordered RPT capacity
   observations and an observed first crossing of 80% of its median first-five
   RPT capacities.
2. Do not replace or remove a cell for lifetime, condition, curve shape, or
   model score. Parser exclusions are schema/readability-only and reported.
3. Sort cell IDs numerically where an integer ID exists, otherwise
   lexicographically. First 60% are train, next 20% validation, final 20% test.
   Stop as inconclusive if fewer than 30 eligible cells, six validation cells,
   or six test cells remain.
4. Unit identity is never an input. Cycling-condition metadata may be used only
   if it is constant per cell and available for all splits.

## Target, inputs, and strict tail

- Event: first observed 80% crossing. Target: cycles remaining to that event.
- Health: current capacity divided by the median first-five capacity.
- Causal inputs: current health, 1/3/5-RPT slopes, causal window mean/std,
  observed cycle index, and complete constant cycling metadata.
- On all pre-event train rows, freeze the health cutoff at their 25th
  percentile. Retain train rows strictly above it. The actual train boundary is
  the minimum retained health.
- Validation and test score only rows strictly below that boundary. Earlier
  held-out rows may build causal history but are not labels for fitting.
- Stop as infeasible if validation or test has fewer than 50 scored rows or is
  not 100% below the actual train boundary.

## Models and validation-only selection

Matched architecture grid for the exact neural fallback and PP continuation:

- widths `{16, 32}`
- learning rates `{5e-4, 1e-3}`
- weight decay `2.0`
- trust `{0, .02, .05, .1, .2, .4}`
- selection seeds `42,43,44`; final seeds `42–46`
- maximum 300 epochs, patience 50

Trust zero must replay the matched MLP to maximum absolute error `<=1e-5`.
A positive-trust candidate is approved only if all hold on validation:

1. pooled RMSE improves by at least 2%;
2. at least 60% of validation cells have lower RMSE;
3. no validation cell's RMSE exceeds 1.10 times fallback RMSE;
4. physical-cell bootstrap 95% CI lower bound for
   `fallback RMSE − candidate RMSE` is positive.

Otherwise select trust zero. If selected/fallback validation pooled R² is
non-positive, label the numerical test prediction `uncertified`; still report
it, but do not count it as a successful deployable prediction.

## Outcomes

Report eligibility, hashes, split IDs, hull audit, validation search, selected
trust, exact replay error, five seeds, pooled/unit-macro/per-cell R²/RMSE/MAE,
and paired cell bootstrap.

Confirmatory model success requires selected ensemble pooled R² > 0 and RMSE no
higher than the matched MLP. Prior-expansion success additionally requires
positive trust. Trust zero with positive R² is a **safe-fallback success**, not
evidence that the PP prior generalized.

No threshold, split, feature, trust, or model grid may change after the first
test score is materialized.
