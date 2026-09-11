# Concrete material-shift CCMR v1.7 protocol

**Frozen:** 2026-09-11, 박진서, before downloading any resource.

## Source

- SPP 2020 Experimental Fatigue Data on High-Strength and
  Ultra-High-Strength Concrete.
- DOI `10.25835/l9n63a7u`, ODbL 1.0, published 2025.
- Seven ZIP resources, total 25.1 MB.
- 53 RH-1 high-strength concrete specimens and 38 RU-1 ultra-high-strength
  concrete specimens.
- Constant-amplitude fatigue at stress levels 0.75/0.05 and 1 Hz until
  failure.

This dataset has not previously been downloaded, screened, or scored in this
project.

## Frozen material and laboratory split

- train: RH-1 HPC Lab 1, 2, and 3;
- validation: RH-1 HPC Lab 4;
- one-shot test: RU-1 UHPC Lab 5, 6, and 7.

The test therefore changes concrete material and laboratory batch. Specimen
identity is parsed from the resource metadata or source filename and must not
cross splits.

## Target and strong extrapolation

- Primary target fixed before opening: stiffness on the decreasing loading
  branch in MPa.
- Sort each specimen by the absolute load-cycle count in column A.
- Never use the related load-cycle count in column B because it may encode
  terminal-life normalization.
- Keep finite rows with strictly increasing cycle count.
- Normalize stiffness by the median of the first 10 observations.
- Require at least 100 observations per specimen.
- Causal history: 10 observations.
- Forecast horizon: `max(5, floor(0.05 * specimen_length))` observations.
- train origins: progress <= 0.30.
- validation origins: progress 0.40--0.60.
- test origins: progress 0.75--0.90.
- Anchor: current normalized stiffness persistence.
- Correction: stiffness slopes over lags 1, 3, and 9 plus slope-1 minus
  slope-9.
- Context: current health, correction features, trailing mean/std, and causal
  history fraction.
- Exclude absolute cycle, progress, concrete type, laboratory, specimen ID,
  temperatures, and future statistics from model inputs.

If the declared stiffness or cycle columns cannot be identified uniquely,
fewer than 20 train or test specimens are eligible, validation has fewer than
5 specimens, or test has fewer than 500 origins, stop as inconclusive. Do not
switch target after opening.

## Frozen v1.7 and decision

Use `protocols/CCMR_V17_CAUSAL_BACKTEST_PROTOCOL.md` exactly, including the
five consecutive causal shadow wins and gated-validation global veto.

Serialize all candidate, causal-gated, and deployed predictions before test
metrics. Strong confirmation requires:

1. 100% of test origins beyond maximum train progress;
2. positive pooled R-squared;
3. pooled and unit-macro RMSE improvements of at least 0.5%;
4. raw test unit mean/CVaR20/max regret <= 0%/1%/2%;
5. deployed correction coverage >= 10%;
6. exact fallback replay error zero.

Safe fallback is not a second successful sealed cohort. Only a strong
confirmation may be appended as the second and final confirmation cohort.

## Pre-score schema clarification

After downloading and reading the textual column definitions, but before
constructing rows or fitting/scoring a model, column A was confirmed as
absolute cycle count and column B as related cycle count. The protocol was
clarified to use A and prohibit B to avoid possible terminal-life leakage.
The target, split, thresholds, and outcomes were not changed.
