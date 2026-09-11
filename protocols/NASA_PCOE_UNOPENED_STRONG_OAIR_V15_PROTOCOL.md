# NASA PCoE unopened strong-extrapolation OAIR v1.5 protocol

**Frozen:** 2026-09-10, 박진서, before reading capacity outcomes or loading
any selected MAT file.

## Source and untouched set

- Official NASA PCoE `5. Battery Data Set.zip`, 209,708,670 bytes.
- SHA-256:
  `82302a7db4fc1b34e0b6676326610438d43b816bdf11a69d1d012a464ef2f92e`.
- Exclude every cell previously used or screened in this project:
  B0005--B0018, B0029--B0032, B0046--B0048, B0053--B0056.
- Selected score-unopened cells:
  B0025--B0028, B0033, B0034, B0036, B0038--B0044, B0049--B0052.

The outer archive is used only to extract cycle-level discharge capacity. Save
a compact CSV and do not expand raw telemetry.

## Frozen condition split

- train: B0025--B0028, B0033, B0034, B0036, B0038--B0040 (10 cells)
- validation: B0041--B0044 (4 cells)
- one-shot test: B0049--B0052 (4 cells)

Train covers room temperature square-wave/constant loads and mixed
24/44-degree loads. Validation is 4-degree mixed fixed loads. Test is a
separate 4-degree fixed-2A experiment that ended after a control-software
crash. Cell identity and condition labels are not features.

## Strong extrapolation task

- For every discharge operation, use the recorded `Capacity`.
- Keep finite positive values; do not delete the documented anomalously low
  cycles.
- Sort in acquisition order and normalize each cell by the median of its first
  five capacities.
- Five-cycle causal history, two-cycle-ahead normalized-capacity target.
- Define forecast-origin progress from zero at the first history-valid origin
  to one at the last horizon-valid origin.
- train origins: progress <= 0.30.
- validation origins: 0.40 <= progress <= 0.60.
- test origins: progress >= 0.75.
- Features: current health, lag-1/3/4 slopes, trailing mean/std, and history
  fraction. Absolute acquisition progress, cell ID, temperature, current, and
  cutoff voltage are excluded.
- The run is inconclusive if any split has fewer than its declared cells or
  test has fewer than 40 rows.

## Frozen OAIR v1.5

Use the exact post-LED structure without retuning:

`future health = current health + gamma * bounded ridge(slopes_1,3,4)`.

- exact baseline: persistence;
- equal-cell weighted ridge alpha grid:
  `.1,1,10,100,1000,10000`;
- correction bound: twice train 95th percentile absolute two-cycle change;
- validation mass grid `0,.01,...,1` under raw cell mean/CVaR/max regret caps
  2%/5%/10%;
- deployment mass: 25% of selected validation mass;
- slope-support distance beyond the maximum unlabeled validation distance:
  exact persistence.

## One-shot success

Predictions must be serialized before test metrics. Success requires:

1. 100% test origins beyond maximum train progress;
2. positive pooled test R-squared;
3. OAIR test RMSE below persistence;
4. raw cell mean/CVaR/max regret within 2%/5%/10%;
5. no change after the test score is read.

This cohort is the independent external check for OAIR v1.5. Inconclusive or
failure results must remain unchanged.
