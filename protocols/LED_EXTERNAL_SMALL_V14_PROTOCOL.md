# LED external small-cohort PP-X v1.4 protocol

**Frozen:** 2026-09-10, 박진서, before reading Data_ID_1/2/3/5/6 numeric
outcomes. Data_ID_4 was inspected for schema and is restricted to training.

## Source and split

- Source: `Roberock/Degradation-data-LED-packages-and-lamps`, MIT license.
- Pinned source commit: `07edea554ab0e157719b24ffc10dddfa43945db0`.
- Six CSV files total 1,971,430 bytes.
- Training test sets: Data_ID_1 and Data_ID_4.
- Validation test sets: Data_ID_3 and Data_ID_5.
- One-shot test sets: Data_ID_2 and Data_ID_6.
- `Test_ID` and file identity are never model features.
- Each `Sample_ID` within a file is an independent physical unit and is
  namespaced by file identity.

## Target and admissibility

- Health is normalized luminous flux `Flux_t`; nominal start is one.
- LED failure is L70: the first of two consecutive finite `Flux_t <= 0.70`.
- A unit requires at least ten observations and a crossing index of at least
  eight. Units not crossing L70 are right-censored and excluded without
  replacement.
- A causal five-observation history supplies current health, margin to L70,
  slopes over lags 1/3/4, trailing mean/std, normalized elapsed time, and
  available-history fraction.
- Target is time remaining to the observed L70 crossing.
- Fit on all eligible training rows at or above the 25th percentile of train
  health. Score validation/test rows strictly below the realized minimum train
  health.
- Stop as inconclusive if fewer than eight test units or 40 pooled test-tail
  rows remain.

## Frozen model

- Auto-Regime Weak-Prior PP-X v1.4, with train-only K=3.
- Architecture bank: width 16/32, learning rate `1e-3`, weight decay 2.0,
  seeds 42--46.
- Fixed trust bank: `0,.02,.05,.10,.20,.40`.
- Regime-local alpha: `0,.05,...,.25`; a regime needs at least five validation
  units or returns to baseline.
- Final raw unit mean/CVaR/maximum regret caps: 2%/5%/10%.
- Test labels cannot influence fitting, regime assignment, trust, alpha, or
  fallback.

## Success rule

Success requires pooled test R-squared above zero, RMSE no worse than the
trust-zero baseline, and raw test mean/CVaR/maximum regret within 2%/5%/10%.
Report each held-out test set separately, physical-unit metrics, regime
occupancy, route decisions, and exact fallback error.

NASA ALT small-summary screening remains separately recorded as inconclusive;
its observed 80%-EOL shortage must not be used to alter this LED protocol.
