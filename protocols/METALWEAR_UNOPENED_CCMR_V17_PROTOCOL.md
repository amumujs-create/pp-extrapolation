# MetalWear unopened CCMR v1.7 protocol

**Frozen:** 2026-09-11, 박진서, before downloading or reading outcomes.

The earlier `resistor2` candidate was rejected at the eligibility stage because
all 29 units had only five observations, below its frozen minimum of eight.
No resistor model, prediction, or test score was produced. This protocol is
the single replacement after that pre-model screening failure.

## Source

- `Auburngrads/SMRD.data` GPL-2 package, `metalwear` dataset.
- Meeker and Escobar (1998), *Statistical Methods for Reliability Data*.
- Public metadata: 96 rows, with `microns`, `unit`, `cycles`, and applied
  `grams`.
- The experiment measured sliding-wear degradation of a real metal alloy.
- Frozen source:
  `https://raw.githubusercontent.com/Auburngrads/SMRD.data/master/data/metalwear.rda`

No MetalWear outcome has previously been downloaded, screened, plotted, or
scored in this project.

## Unit split and eligibility

- Physical-unit key: exact `unit` label. A unit may not cross splits.
- Sort unique unit strings by SHA-256.
- First 50%: train; next 25%: validation; remainder: one-shot test.
- Require at least 8 finite observations with strictly increasing cycles.
- Stop before fitting if fewer than 8 eligible units, 4 train units,
  2 validation units, 2 test units, or 4 test origins remain.
- Applied grams, unit identity, and split membership are not model inputs.

## Frozen forecast task

- Sort each unit by accumulated cycles.
- Response: microns divided by the absolute first observation when that
  observation is nonzero; otherwise subtract the first observation and divide
  by the train-only median absolute first increment. This branch is determined
  mechanically before splitting rows.
- Causal history: 3 observations; one-observation forecast horizon.
- train origins: progress <= 0.30.
- validation origins: progress 0.40--0.60.
- test origins: progress >= 0.75.
- Progress is row index/final index and is used only for splitting.
- Anchor: current normalized wear persistence.
- Correction features: local slopes over lags 1 and 2 and their difference.
- Context: current response, correction features, trailing three-point
  mean/std, and causal history fraction.
- Absolute cycle, grams, unit ID, progress, terminal cycle, and future
  statistics are excluded from model inputs.

## Model and success

Apply frozen CCMR v1.7 from
`protocols/CCMR_V17_CAUSAL_BACKTEST_PROTOCOL.md` without retuning. Save
candidate, gated, deployed, persistence, truth, units, and progress before
test scoring.

Performance success requires all:

1. 100% strict ordered-axis OOD;
2. positive pooled test R-squared;
3. pooled and unit-macro RMSE improve by at least 0.5%;
4. raw unit mean/CVaR20/max regret <= 0%/1%/2%;
5. deployed correction coverage >= 10%;
6. exact fallback replay error zero.

Because the public dataset is very small, a pass is an independent but
low-power replication. A failure or safe fallback is preserved and no further
replacement cohort is attempted.
