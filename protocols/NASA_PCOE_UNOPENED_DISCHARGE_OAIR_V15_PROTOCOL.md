# NASA PCoE unopened discharge-trajectory OAIR v1.5 protocol

**Frozen:** 2026-09-10, 박진서, after capacity-count screening but before
loading voltage/current/temperature outcomes or computing any model score.

The lifecycle-capacity protocol remains inconclusive. This is a different,
pre-score task on the same score-unopened cells.

## Fixed condition split

- train cells: B0025--B0028, B0033, B0034, B0036, B0038--B0040
- validation cells: B0041--B0044
- one-shot test cells: B0049--B0052

Every valid discharge operation is a trajectory. Episodes are nested inside
cells; all metrics and regret certification use the physical cell as the
independence unit.

## Strong within-discharge extrapolation

- Keep aligned finite `Time`, `Voltage_measured`, `Current_measured`, and
  `Temperature_measured` samples.
- Require at least 100 samples and strictly increasing usable times.
- Normalize voltage by the median of the first five samples.
- Causal history: ten samples.
- Forecast horizon: `max(5, floor(0.10 * episode_length))` observations.
- Define origin progress from zero at episode start to one at its last sample.
- train origins: progress <= 0.30.
- validation origins: 0.40 <= progress <= 0.60.
- test origins: 0.75 <= progress <= 0.85.
- Target: normalized voltage at the future horizon.
- Features: current normalized voltage, voltage slopes over lags 1/5/9,
  trailing voltage mean/std, current, temperature, and history fraction.
- Cell, episode, ambient condition, cutoff voltage, and absolute progress are
  excluded from model features.

The test is inconclusive if fewer than three test cells, 30 valid test
episodes, or 1,000 pooled test origins remain.

## Frozen OAIR v1.5

Use `fit_invariant_residual` unchanged:

- persistence anchor: current normalized voltage;
- residual inputs: voltage slopes 1/5/9 only;
- ridge grid `.1,1,10,100,1000,10000`;
- correction bound: twice train 95th percentile absolute target change;
- validation raw-cell regret caps 2%/5%/10%;
- deployment mass: 25% of validation-safe mass;
- slope-support beyond maximum unlabeled validation distance: exact
  persistence.

## Success

Predictions are serialized before test scoring. Success requires 100% of test
origins beyond train progress, positive pooled R-squared, RMSE below
persistence, and raw physical-cell mean/CVaR/max regret within 2%/5%/10%.
No test-result-dependent amendment is permitted.
