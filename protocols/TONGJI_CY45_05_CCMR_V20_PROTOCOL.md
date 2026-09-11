# Tongji CY45-05 CCMR v2.0 one-shot protocol

**Frozen:** 2026-09-11, 박진서, before extracting CY45-05 discharge
capacities for CCMR scoring.

Zip member names were used only to fix the population to the 56
`*CY45-05*.pkl` files. This archive was previously opened for PP-X v1
(infeasible hull), so the run is **CCMR-contract new** but not
dataset-level untouched. Do not claim pristine external confirmation.

## Source and population

- BatteryLife processed `Tongji.zip` (Zenodo `14934405`).
- Local path: `data/tongji_ppx/Tongji.zip`.
- Use every zip member whose filename contains `CY45-05` and ends with
  `.pkl` (56 cells; single temperature/C-rate tag).
- Exclude all other Tongji conditions (`CY25-*`, etc.).

## Capacity trajectory

- For each cycle row, take the maximum finite `discharge_capacity_in_Ah`.
- Collapse duplicate `cycle_number` values with the median capacity.
- Require at least 50 finite cycle capacities and a strictly positive
  first capacity.
- Normalize each cell by its first cycle capacity. No smoothing.
- Unit ID: zip stem (e.g. `Tongji1_CY45-05_1-#1`).

## Split and forecast

- Sort unit IDs by SHA-256 of
  `tongji-cy45-05-ccmr-v20:{unit}`.
- First floor(60%): train; next floor(20%): validation; remainder: test.
- Causal history: 3 cycles; horizon: 1 cycle.
- train origins: progress <= 0.30.
- validation origins: progress 0.40--0.60.
- test origins: progress 0.75--0.90.
- Require at least 40 eligible cells, 20 validation origins, and 15 test
  origins; otherwise stop before fitting.
- Anchor: current normalized capacity.
- Correction: slopes over lags 1 and 2 and their difference.
- Context: current capacity, correction features, trailing-three
  mean/std, and causal history fraction.
- Cell ID, condition tag, absolute cycle, ambient, terminal length,
  progress, and future values are excluded from model inputs.

## Model and success

Frozen CCMR v2.0 from
`protocols/CCMR_V20_FROZEN_PREDICTOR_MANIFEST.json`, including stable /
cautious / exact-fallback routing identical to the MultiStage holdout
replay contract.

Success requires all:

1. every chosen test origin exceeds maximum train progress;
2. pooled test R-squared > 0;
3. pooled and unit-macro RMSE improvements >= 0.5%;
4. raw test unit mean/CVaR20/max regret <= 0%/1%/2%;
5. deployed correction coverage >= 10%;
6. exact fallback replay error is zero.

Pass or failure is preserved. A pass is win-set evidence and does not
auto-promote to counted sealed success.
