# Carbon-film resistor unopened CCMR v1.7 protocol

**Frozen:** 2026-09-11, 박진서, before downloading or reading outcomes.

## Source and independence

- `Auburngrads/SMRD.data` GPL-2 package, `resistor2` dataset.
- Shiomi and Yanagisawa (1979); Suzuki, Maki, and Yokogawa (1993).
- Public metadata: 145 rows and four columns (`resistor`, `celsius`,
  `hours`, `resistance`), measured at three accelerated temperatures.
- Frozen source:
  `https://raw.githubusercontent.com/Auburngrads/SMRD.data/master/data/resistor2.rda`

No resistor outcome has previously been downloaded, screened, plotted, or
scored in this project.

## Unit split and eligibility

- Physical-unit key: exact string `celsius|resistor`; temperature is included
  because a repeated resistor label across temperatures must not merge units.
- Sort unique keys by SHA-256 of the key.
- First 60% of keys: train; next 20%: validation; remainder: one-shot test.
- Require at least 8 finite, strictly increasing-time observations per unit.
- Stop as inconclusive if fewer than 10 eligible units, 6 train units,
  2 validation units, 2 test units, or 4 test origins remain.
- Temperature, resistor label, and split membership are not model inputs.

## Frozen forecast task

- Sort each unit by accumulated hours.
- Response: resistance divided by that unit's first observed resistance.
- Causal history: 3 observations.
- Forecast horizon: one observation.
- train origins: progress <= 0.30.
- validation origins: progress 0.40--0.60.
- test origins: progress >= 0.75.
- Progress is row index divided by final index and is used only to construct
  the ordered-axis split, never as a model feature.
- Anchor: current normalized resistance persistence.
- Correction features: resistance slopes over lags 1 and 2 and their
  difference.
- Context: current normalized resistance, correction features, trailing
  three-point mean/std, and causal history fraction.
- Absolute time, temperature, unit label, progress, terminal time, and future
  statistics are excluded from model inputs.

## Model and immutable decision

Use CCMR v1.7 exactly as frozen in
`protocols/CCMR_V17_CAUSAL_BACKTEST_PROTOCOL.md`, including:

- physical-unit cross-fit consensus and context support veto;
- validation raw unit-regret minimax;
- five consecutive non-overlapping causal shadow wins;
- gated-validation global veto and exact persistence fallback.

Save candidate, gated, deployed, persistence, and truth arrays before
computing test metrics. No split, feature, threshold, target, or model change
is allowed after opening.

## Performance-success criteria

All must hold:

1. 100% of test origins are beyond maximum train progress;
2. test pooled R-squared is positive;
3. pooled and unit-macro RMSE improve by at least 0.5%;
4. raw test unit mean/CVaR20/max regret are <= 0%/1%/2%;
5. deployed correction coverage is at least 10%;
6. exact fallback replay error is zero.

Only a pass is added as an additional performance-success sealed cohort.
A safe fallback or failure remains an audit result and is not replaced by
another attempted dataset.
