# Multi-Stage Stage-2 TP_k CCMR v2.0 one-shot protocol

**Frozen:** 2026-09-11, 박진서, before downloading any Stage-2 RPT member
contents or capacity outcomes.

This cohort is chosen because CCMR already beat Engression on the same-DOI
Stage-1 MultiStage RPT regime under CCMR v2.0. It is therefore a
**selection-conditioned replication**, not a pristine first-choice
confirmation. Stage-1 `TP_z*` physical units are excluded.

## Source and population

- Stroebl et al., Multi-Stage Lithium Ion Battery Aging Study.
- DOI `10.6084/m9.figshare.25975315.v1`, CC BY 4.0.
- Folder: `Multi-Stage_Aging_Study/Stage_2`.
- Use only archives matching `TP_k*.zip` (66 cells; 11 conditions × 6
  replicates), exactly as listed with IDs, sizes, and MD5 values in
  `protocols/MULTISTAGE_STAGE2_TPK_PUBLIC_MANIFEST.json`.
- Exclude every Stage-2 `TP_z*.zip` archive to avoid reusing Stage-1 cells.
- Range-extract only RPT members matching `_ET_T23.csv`, `_CU.csv`,
  `_exCU.csv`, or `_AT_T23.csv`; cycling files and other temperatures are
  excluded.

## Capacity trajectory

- Sort each cell's RPT members by the two-digit sequence number after the
  complete `TP_kNN_RR_` cell prefix.
- Read `run_time`, `c_cur`, and `step_type`.
- Convert `run_time` with `pandas.to_timedelta` to seconds.
- Charge capacity: integrate current over `step_type == 21` and
  `c_cur > 0`.
- Discharge capacity: integrate current over `step_type == 22` and
  `c_cur < 0`.
- RPT capacity is their mean in Ah, matching the publisher's public
  `feature_extraction.capacity` rule and the Stage-1 extractor.
- Require at least 10 finite RPT capacities per cell.
- Normalize each cell by its first RPT capacity. No smoothing.

## Split and forecast

- Sort cell IDs by SHA-256 of the exact archive stem (e.g. `TP_k01_01`).
- First floor(60%): train; next floor(20%): validation; remainder: one-shot
  test.
- Causal history: 3 RPTs; horizon: 1 RPT.
- train origins: progress <= 0.30.
- validation origins: progress 0.40--0.60.
- test origins: progress 0.75--0.90.
- Require at least 50 eligible cells, 20 validation origins, and 15 test
  origins; otherwise stop before fitting and mark inconclusive.
- Anchor: current normalized RPT capacity.
- Correction features: slopes over lags 1 and 2 and their difference.
- Context: current capacity, correction features, trailing-three mean/std,
  and causal history fraction.
- Cell, condition, cycle count, calendar time, terminal RPT count, progress,
  and future values are excluded from model inputs.

## Model and success

Use frozen CCMR v2.0 exactly as hashed in
`protocols/CCMR_V20_FROZEN_PREDICTOR_MANIFEST.json`:

- `fit_causal_dynamics_bank` / `predict_causal_dynamics_bank`
- adaptive causal backtest gate from `causal_backtest.py`
- stable-bank route when validation units >= 10, active coverage and regret
  caps match the v2.0 holdout replay contract; otherwise exact persistence
  fallback

Seal candidate, gated/deployed, persistence, truth, unit, origin, target,
and progress before test scoring.

Success requires all:

1. test progress strictly exceeds maximum train progress for every chosen
   origin;
2. pooled test R-squared is positive;
3. pooled and unit-macro RMSE improvements are at least 0.5%;
4. raw test unit mean/CVaR20/max regret are at most 0%/1%/2%;
5. deployed correction coverage is at least 10%;
6. exact fallback replay error is zero.

Pass or failure is preserved without relabeling or swapping in Stage-1
`TP_z` cells. A pass is evidence for the win-favorable set; it does not
automatically become a counted sealed cohort.

## Competitor claim gate

Only after the sealed CCMR decision may competitors be fit on the same
features with validation-only selection. Engression and other figure-matched
baselines are retrospective once test is opened; they are not used to choose
the cohort.
