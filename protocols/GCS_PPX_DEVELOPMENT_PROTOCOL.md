# Grouped Cross-Fitted Safe Consensus PP-X protocol

## Objective

Develop a PP-X extension that preserves dataset coverage and seed stability
while improving equal-setting pooled R2 after the five opened external settings
are included. This is retrospective model development, not prospective
confirmation.

## Model

GCS-PPX combines three mechanisms:

1. A domain-balanced normalized residual head learns only from PP-X five-seed
   prediction geometry and observed development residuals.
2. A cross-domain jackknife removes each training cohort in turn. A correction
   is admitted row by row only when the jackknife heads agree on its sign and
   their normalized interquartile range is below a fixed threshold. Otherwise
   the model abstains and returns the original PP-X ensemble.
3. Seed deviations are contracted toward the accepted ensemble prediction.
   This changes seed stability without changing an abstained ensemble.

The resulting predictor is one structural PP-X extension, not a test-time
ensemble with MLP or Engression.

## Leakage control

- The nine main settings are separate cohort groups.
- Axial-fan `1P_8F`, `4P_1F`, and `4P_8F` share one cohort group and are always
  held out together.
- MATWI and Misata are separate groups.
- Every reported prediction is produced by an outer leave-one-cohort-out fit.
- Hyperparameters for an outer fold are selected only by inner
  leave-one-cohort-out predictions among the remaining groups.
- No row from the held-out cohort fits the head or selects its policy.

## Selection objective

Candidate policies are ordered lexicographically by:

1. number of positive-R2 settings;
2. worst setting R2;
3. 10th-percentile setting R2;
4. mean setting R2;
5. mean single-seed R2;
6. lower mean within-setting seed R2 standard deviation;
7. lower residual authority.

This ordering encodes the previously stated definition of coverage: stable
positive performance across seeds and datasets before mean performance.

## Search space

- ridge alpha: `{10, 100, 1000}`;
- residual authority: `{0, 0.15, 0.30, 0.50}`;
- jackknife sign agreement: `{0.60, 0.75, 0.90}`;
- maximum normalized jackknife IQR: `{0.05, 0.15, 0.30}`;
- maximum normalized correction magnitude: `{0.10, 0.25}`;
- seed contraction: `{0.25, 0.50}`.

## Reporting

Report the outer cross-fitted result separately for the nine main settings and
the five external settings. Axial configurations are settings, not independent
external datasets. A future untouched cohort is required before making an
external superiority claim.
