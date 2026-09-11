# Perovskite phenotype-shift CCMR-L v1.6.1 protocol

**Frozen:** 2026-09-11, 박진서, before downloading or opening the outcome
pickle.

## Source and scope

- Hartono et al., Perovskite Solar Cells Ageing Dataset v1.0.
- Zenodo DOI `10.5281/zenodo.8185883`, CC BY 4.0.
- File `20230303_mySeriesDrop.pkl`, 33.0 MB.
- Published MD5 `0984467a38e7181c51e77a355bdc0a1f`.
- Published metadata: 2,245 cleaned MPPT-PCE traces, 150 hours resampled every
  10 minutes, nominally 900 observations per cell.

The public pickle contains the traces only and omits fabrication-batch IDs.
Therefore this experiment must not be described as a fabrication-batch shift.
It is an early-trajectory phenotype condition shift.

## Causal preprocessing

- Verify MD5 before loading the trusted academic pickle.
- Use the provided cleaned/resampled values without additional smoothing.
- Do not use whole-trajectory maximum, minimum, mean, or scaler.
- Normalize each cell by the median of its first 12 PCE values.
- Require at least 856 finite positive observations. Do not impute after
  opening the file; an ineligible cell is dropped before clustering.

## Outcome-blind condition split

Only the first 30% of every eligible trajectory may define its phenotype.
For each cell compute:

1. normalized PCE at 10%, 20%, and 30%;
2. median first-difference in 0--10%, 10--20%, and 20--30%;
3. MAD of first differences in 0--30%;
4. fraction of positive first differences in 0--30%.

Robust-standardize these eight descriptors and run K-means with K=5,
`random_state=161`, `n_init=20`. Rank clusters by centroid Euclidean distance
from the global descriptor center. The farthest cluster is the one-shot test,
the second farthest is validation, and the remaining three are train. Ties
use ascending cluster ID. No late value participates in split construction.

The run is inconclusive if validation or test has fewer than 100 cells, train
has fewer than 500 cells, or any cluster is empty.

## Strong ordered-axis extrapolation

- Causal history: 19 observations.
- Forecast horizon: 90 observations (15 hours).
- train origins: progress <= 0.30.
- validation origins: 0.40--0.60.
- test origins: 0.75--0.85.
- Anchor: current normalized PCE persistence.
- Correction features: slopes over lags 1, 6, and 18 plus slope-1 minus
  slope-18.
- Context: current health, correction features, trailing mean/std, and causal
  history fraction.
- Absolute time, progress, cell index, cluster ID, and any future statistic
  are excluded from model inputs.

## Frozen CCMR-L v1.6.1

Use the raw-regret-corrected CCMR v1.6 structure. For scalability, replace
leave-one-cell residual fits with deterministic 20-fold physical-cell fits;
LED development showed identical decisions to the full ensemble.

- ridge grid `.1,1,10,100,1000,10000`;
- fold sign agreement >= 90%, relative MAD <= 0.5;
- full-context nearest-prototype validation quantile 99%;
- deployment mass grid 0--0.25;
- validation raw cell mean/CVaR20/max regret caps 0%/1%/2%;
- exact persistence on every rejected row.

## One-shot decision

Serialize predictions and gates before test metrics. Strong confirmation
requires:

1. 100% test origins beyond maximum train progress;
2. positive pooled R-squared;
3. pooled and cell-macro RMSE each improve by at least 0.5%;
4. test raw cell mean/CVaR20/max regret <= 0%/2%/5%;
5. correction coverage >= 10%;
6. fallback maximum replay error exactly zero.

If correction mass or coverage is zero with exact replay, report safe fallback
rather than confirmation. Do not change K, split, horizon, thresholds, or
normalization after outcomes are opened.
