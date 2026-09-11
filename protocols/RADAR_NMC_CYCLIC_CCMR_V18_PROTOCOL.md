# RADAR NMC cyclic-aging CCMR v1.8 protocol

**Frozen:** 2026-09-11, 박진서, before downloading result data.

This is the final sequentially selected replication after earlier outcomes
were observed. It is not pristine first-choice confirmation. No additional
dataset follows this test.

## Source

- Luh and Blank (2024), 228 commercial NMC/C+SiO cells.
- Result-data v2 DOI `10.35097/1969`, CC BY 4.0.
- Nature Scientific Data article DOI `10.1038/s41597-024-03831-x`.
- Public archive size: 333,402,624 bytes.
- Use post-processed `cell_eocv2` CSV records only.
- Fixed columns follow the authors' public code:
  `age_type`, `cyc_condition`, `cyc_charged`, `num_cycles_op`,
  `timestamp_s`, and `cap_aged_est_Ah`.

The repository metadata and public parser code were inspected. No EOC
outcome file from this dataset has previously been downloaded, screened,
fitted, plotted, or scored in this project.

## Population, rows, and physical-unit split

- Retain cyclic-aging cells: `age_type == 2`.
- Retain regular-operation discharge completions:
  `cyc_condition == 1` and `cyc_charged == 0`.
- Unit key is the full cell filename identity `Pxxx_n_Sxx_Cxx`.
- Sort by `num_cycles_op`, break ties by `timestamp_s`, and retain the last
  row at each cycle.
- Require at least 500 finite, strictly increasing cycles per cell.
- Sort eligible unit keys by SHA-256.
- First 60%: train; next 20%: validation; remainder: one-shot test.
- Stop before fitting if fewer than 100 eligible cells, 60 train, 20
  validation, 20 test, or 5,000 test origins remain.
- Parameter set, replicate, channel, temperature, SoC, rates, and profile are
  not model inputs.

## Frozen forecast task

- Initial capacity: maximum of first five retained capacity estimates.
- Health: `cap_aged_est_Ah / initial_capacity`.
- Apply a causal trailing five-cycle median.
- Causal history: 20 cycles; horizon: 10 cycles.
- train origins: progress <= 0.30.
- validation origins: progress 0.40--0.60.
- test origins: progress 0.75--0.90.
- Progress is row index/final index and is only used for splitting.
- Anchor: current smoothed health persistence.
- Correction features: slopes over lags 1, 5, and 20 plus
  slope-1 minus slope-20.
- Context: current health, correction features, trailing 20-cycle mean/std,
  and causal history fraction.
- Absolute cycle, timestamp, progress, unit/condition identity, terminal
  cycle, and future values are excluded from model inputs.

## Frozen model and success

Use CCMR v1.8 exactly as frozen in
`protocols/CCMR_V18_REGIME_ROUTER_PROTOCOL.md`. Route selection is
validation-only. Seal candidate, routed, deployed, persistence, truth, unit,
origin, target, and progress arrays before test scoring.

Performance success requires all:

1. 100% strict ordered-axis OOD;
2. positive pooled test R-squared;
3. pooled and unit-macro RMSE improvements >= 0.5%;
4. raw test unit mean/CVaR20/max regret <= 0%/1%/2%;
5. deployed correction coverage >= 10%;
6. exact fallback replay error zero.

A pass is registered as a selection-conditioned performance-success cohort,
not as pristine prospective confirmation. A failure or fallback is preserved
without replacement.
