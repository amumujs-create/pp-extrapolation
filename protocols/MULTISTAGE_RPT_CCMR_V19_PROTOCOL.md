# Multi-Stage battery RPT CCMR v1.9 one-shot protocol

**Frozen:** 2026-09-11, 박진서, before downloading RPT outcome members.

This is a sequentially selected replication and not pristine first-choice
confirmation. It is the one additional dataset requested after prior
outcomes; no replacement follows this test.

## Source and population

- Stroebl et al., Multi-Stage Lithium Ion Battery Aging Study.
- DOI `10.6084/m9.figshare.25975315.v1`, CC BY 4.0.
- 279 Samsung INR21700-50E cells under 71 conditions.
- Use all 75 Stage-1 cyclic-aging (`TP_z*`) cells, three physical replicates
  at each of 25 conditions.
- Figshare metadata fixes 75 archive IDs, names, byte sizes, and MD5 values
  before outcome access.
- Range-extract only RPT members matching `_ET_T23.csv`, `_CU.csv`,
  `_exCU.csv`, or `_AT_T23.csv`; cycling files and other temperatures are
  excluded.

## Capacity trajectory

- Sort each cell's RPT members by the two-digit sequence number.
- Read `run_time`, `c_cur`, and `step_type`.
- Charge capacity: integrate current over `step_type == 21` and
  `c_cur > 0`.
- Discharge capacity: integrate current over `step_type == 22` and
  `c_cur < 0`.
- RPT capacity is their mean in Ah, exactly following the publisher's public
  `feature_extraction.capacity` implementation.
- Require at least 10 finite RPT capacities per cell.
- Normalize each cell by its first RPT capacity. No smoothing.

## Split and forecast

- Sort cell IDs by SHA-256.
- First floor(60%): train; next floor(20%): validation; remainder: one-shot
  test.
- Causal history: 3 RPTs; horizon: 1 RPT.
- train origins: progress <= 0.30.
- validation origins: progress 0.40--0.60.
- test origins: progress 0.75--0.90.
- Require at least 72 eligible cells, 30 validation origins, and 20 test
  origins; otherwise stop before fitting.
- Anchor: current normalized RPT capacity.
- Correction features: slopes over lags 1 and 2 and their difference.
- Context: current capacity, correction features, trailing-three mean/std,
  and causal history fraction.
- Cell, condition, cycle count, calendar time, terminal RPT count, progress,
  and future values are excluded from model inputs.

## Model and success

Use frozen CCMR v1.9 from
`protocols/CCMR_V19_REPLICATED_ROUTER_PROTOCOL.md`. Seal candidate, routed,
deployed, persistence, truth, unit, origin, target, and progress before test
scoring.

Success requires all:

1. test progress strictly exceeds maximum train progress for every origin;
2. pooled test R-squared is positive;
3. pooled and unit-macro RMSE improvements are at least 0.5%;
4. raw test unit mean/CVaR20/max regret are at most 0%/1%/2%;
5. deployed correction coverage is at least 10%;
6. exact fallback replay error is zero.

Pass or failure is preserved without relabeling or dataset replacement.

## Pre-fit parser amendment

The first extraction preserved `run_time` as its source `HH:MM:SS.sss` text,
so numeric coercion produced zero capacities; its RPT sequence regex also
captured the replicate token. Eligibility therefore stopped with zero cells
before fitting, prediction, or scoring, and that artifact is retained.

The amended parser converts `run_time` with `pandas.to_timedelta` to seconds
and matches the sequence token after the complete `TP_zNN_RR_` cell prefix.
It writes a distinct summary and result directory. Population, split,
features, route, and every success threshold remain unchanged.

The corrected structural pass found that all three `TP_z04` cells had fewer
than 10 finite RPT capacities. They were excluded by the already-frozen QC
rule, leaving 72 eligible cells; no model was fitted or scored. The fixed
45/15/15 count is therefore replaced by its originally intended deterministic
60%/20%/remainder SHA-256 split, yielding 43/14/15 units. This second pre-fit
amendment writes another distinct result directory and changes no model or
success threshold.
