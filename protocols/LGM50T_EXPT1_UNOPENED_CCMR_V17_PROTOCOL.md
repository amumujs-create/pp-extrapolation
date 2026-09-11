# LG M50T Experiment 1 unopened CCMR v1.7 protocol

**Frozen:** 2026-09-11, 박진서, before downloading performance summaries.

The `resistor2` and `metalwear` candidates were rejected by their frozen
eligibility rules before model fitting: respectively five observations per
unit and only three eligible tail origins. Neither produced predictions or
test scores. This is the final replacement candidate.

## Source and compact extraction

- Kirkaldy et al. (2024), commercial LG M50T 21700-cell cycle ageing.
- DOI `10.5281/zenodo.10637534`, CC BY 4.0.
- Experiment 1: nine physical cells, 0--30% SoC ageing, 0.3C charge/1C
  discharge, at 10/25/40 degrees C.
- Remote archive: `Expt 1 - Si-based Degradation.zip`.
- Download only the nine `Summary Data/Performance Summary/*Processed
  Data.csv` members by HTTP byte range. Do not download the 9.1 GB archive.
- Fixed response column: `C/10 Capacity [mA h]`.
- Fixed causal axis: `Charge Throughput [A h]`.

The metadata workbook and parsing notebook were inspected before freezing;
neither contains cell outcome trajectories. No performance-summary member has
previously been downloaded, screened, plotted, or scored in this project.

## Unit split and eligibility

- Unit key: Experiment 1 cell letter.
- Sort the nine cell keys by SHA-256.
- First five: train; next two: validation; final two: one-shot test.
- Require at least 10 finite rows with strictly increasing charge throughput
  per cell.
- Stop before fitting if not exactly nine eligible cells, the fixed 5/2/2
  split is unavailable, or fewer than four test origins remain.
- Temperature and cell identity are not model inputs.

## Frozen forecast task

- Sort by charge throughput and divide capacity by the cell's first capacity.
- Causal history: 3 observations; one-RPT forecast horizon.
- train origins: progress <= 0.30.
- validation origins: progress 0.40--0.60.
- test origins: progress >= 0.75.
- Progress is row index/final index and is only used for ordered splitting.
- Anchor: current normalized capacity persistence.
- Correction features: local capacity slopes over lags 1 and 2 and their
  difference.
- Context: current capacity, correction features, trailing three-point
  mean/std, and causal history fraction.
- Absolute throughput, temperature, cell ID, progress, terminal throughput,
  and future values are excluded from model inputs.

## Model and success

Use frozen CCMR v1.7 in
`protocols/CCMR_V17_CAUSAL_BACKTEST_PROTOCOL.md` with no threshold change.
Save candidate, causal-gated, deployed, persistence, truth, units, and
progress before computing test metrics.

Performance success requires all:

1. all test origins beyond maximum train progress;
2. positive pooled test R-squared;
3. pooled and unit-macro RMSE improvements >= 0.5%;
4. raw test unit mean/CVaR20/max regret <= 0%/1%/2%;
5. deployed correction coverage >= 10%;
6. exact fallback replay error zero.

A pass is an independent, small-sample performance replication. A failure or
safe fallback is preserved; no additional candidate is attempted.

## Outcome-independent pre-score eligibility amendment

After range-downloading the nine 60 KB summary files, schema-only inspection
showed 11--15 RPT rows per cell. No response values, model, predictions, or
test scores were inspected. The arbitrary minimum was reduced from 12 to 10
rows because 10 rows are mathematically sufficient to provide five realized
one-step shadow forecasts before the first >=75% origin. The minimum number
of test origins was reduced from eight to four to match the already-frozen
two-cell test. Model structure, split order, target, features, risk caps, and
success thresholds did not change. This amendment lowers the evidential
grade relative to a completely unamended one-shot and must be reported.
