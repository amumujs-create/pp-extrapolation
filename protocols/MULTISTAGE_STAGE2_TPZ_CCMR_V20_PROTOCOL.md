# Multi-Stage Stage-2 TP_z CCMR v2.0 one-shot protocol

**Frozen:** 2026-09-11, 박진서, after Stage-2 `TP_k` inconclusive and after
zip **member-name** RPT-count screening only (no capacity outcomes opened).

This is the remainder of Stage-2 after `TP_k`. It is **not** an independent
physical-unit cohort: 72/72 Stage-2 `TP_z` archive names already appear in
the sealed Stage-1 MultiStage cohort. Treat results as same-cell continued
aging / selection-conditioned replication only.

## Source and population

- DOI `10.6084/m9.figshare.25975315.v1`, folder
  `Multi-Stage_Aging_Study/Stage_2`.
- Use all `TP_z*.zip` archives listed in
  `protocols/MULTISTAGE_STAGE2_TPZ_PUBLIC_MANIFEST.json` (72 cells).
- Exclude `TP_k*`.
- Range-extract RPT members matching `_ET_T23.csv`, `_CU.csv`,
  `_exCU.csv`, or `_AT_T23.csv`.
- Sequence token: after complete `TP_zNN_RR_` prefix.
- `run_time` via `pandas.to_timedelta` to seconds; capacity = mean of
  charge (`step_type==21`) and discharge (`step_type==22`) Ah integrals.
- Require ≥10 finite RPT capacities per cell.
- Normalize by first RPT capacity.

## Split and forecast

- Sort cell IDs by SHA-256 of archive stem; salt string
  `multistage-stage2-tpz-ccmr-v20:` prefixed before the cell ID.
- floor(60%) train / floor(20%) validation / remainder test.
- history 3, horizon 1.
- train progress ≤0.30; validation 0.40–0.60; test 0.75–0.90.
- Stop before fitting unless ≥50 eligible cells, ≥20 validation origins,
  ≥15 test origins.
- Features identical to Stage-1 MultiStage CCMR contract.

## Model and success

Frozen CCMR v2.0 from
`protocols/CCMR_V20_FROZEN_PREDICTOR_MANIFEST.json` with the same
stable / cautious / exact-fallback route caps as Stage-1 holdout replay.

Success criteria identical to
`protocols/MULTISTAGE_STAGE2_TPK_CCMR_V20_PROTOCOL.md`.

A pass does **not** auto-promote to counted sealed success and must not be
described as a new independent cohort.
