# Axial-fan untouched PP confirmation protocol

Frozen before downloading or inspecting any numeric train, test, or RUL file.
Only the public landing-page description, filenames, publisher hashes and byte
sizes were inspected before this commit.

## Source and untouched status

- Dataset: *Condition monitoring of axial fans for road tunnels*.
- Publisher: Mendeley Data V1, DOI https://doi.org/10.17632/mzjvw6kbt7.1
- Licence shown by the publisher: CC BY 4.0.
- Four official configurations: `1P_1F`, `1P_8F`, `4P_1F`, `4P_8F`.
- Each configuration has an official train file, truncated test file, and test
  endpoint RUL file.  The publisher describes two operating-point columns and
  eight health variables (three winding temperatures, three bearing
  temperatures and two bearing-vibration variables).
- Total publisher-reported size of all 12 text files is below 9 MB.
- Repository-wide text search found no prior use of this dataset or DOI in PP or
  the adjacent PAE study.  This is therefore dataset-level untouched at freeze.

## Integrity and feasibility gates

1. Download all 12 files only after this protocol commit.
2. Verify every file against the publisher SHA-256 and retain a manifest.
3. Require finite whitespace-separated tables, at least 13 columns in train,
   consistent feature columns in train/test, at least 10 train units and at
   least 5 official test units per configuration.
4. Require one endpoint truth per official test unit and no exact duplicate
   unit trajectory across train and test within a configuration.
5. The first column is unit, second is time, next two are operating point, next
   eight are health.  The remaining train-only column must agree with
   time-to-end RUL: nonnegative, nonincreasing within unit and zero at its last
   row.  If this audit fails, stop as `inconclusive`; do not search columns using
   test performance.

## Frozen split and causal representation

For each configuration independently, sort train unit identifiers and use a
seed-42 random 80/20 unit split for model fitting and validation.  Official test
units remain sealed until every prediction is produced.

The representation is frozen from the successful C-MAPSS mechanism repair:

- assign operating regimes from the two operating-point columns using train
  data only (`1P`: one regime; `4P`: four KMeans regimes, seed 42);
- normalize each health channel within its train operating regime;
- from a causal 30-record window compute health current value, mean, standard
  deviation and slope;
- append current-minus-first-20-record train-normalized health, current
  operating point, window regime occupancy and `log1p(time)`;
- pad short histories with their first available observation;
- use every fifth train record plus the final record; evaluate one final record
  per official test fan.

No unit ID, configuration ID, final lifetime, future measurement or test label
is a model input.  Every scaler and operating-regime model is fit on model-fit
units for selection and all official train units for the final refit.

## Frozen models

- PP: one default width-32 network with a weighted-Ridge initialized frozen
  affine path plus a two-layer tanh NN residual; default optimizer and clipping.
- Matched plain MLP: the same width-32 tanh path, input, loss weighting and
  validation checkpoint rule, without the affine path.
- Affine-only prediction is retained as a diagnostic.
- Seeds: 42, 43, 44, 45, 46.  Primary prediction is the five-seed mean.
- Hyperparameters are not changed after any official test RUL is read.

## Endpoints and success rule

Primary endpoint: configuration-macro official-test R² (equal weight for the
four configurations).  Secondary endpoints: pooled R² over all official test
fans, per-configuration R²/RMSE/MAE, five-seed dispersion, paired fan bootstrap,
and causal-feature hull-out fraction.

A confirmatory success requires all of:

1. PP pooled R² is positive;
2. PP configuration-macro R² exceeds matched MLP configuration-macro R²;
3. PP RMSE is lower than MLP in at least three of four configurations;
4. paired fan bootstrap probability of positive pooled RMSE gain is at least
   0.95;
5. all integrity gates pass.

Failure and inconclusive outcomes are retained without changing the protocol.
If fewer than 100% of official endpoints are outside the train feature hull,
the result is described as unseen-unit truncated-future RUL transfer and the
measured hull-out subset is reported separately; it is not renamed strict
convex-hull extrapolation.
