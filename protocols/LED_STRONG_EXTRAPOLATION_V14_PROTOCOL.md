# LED strong-extrapolation PP-X v1.4 protocol

**Frozen:** 2026-09-10, 박진서, before computing any score for this
fixed-horizon task. Numeric endpoint ranges were screened previously, so this
is development evidence rather than untouched confirmation.

## Cohort set

Use Data_ID_2, Data_ID_5, and Data_ID_6 as three separate cohorts. Combined
size is 246,685 bytes. The physical-unit key is `(Test_ID, Sample_ID)` because
sample numbers are reused across test conditions. Data_ID_1/4, which informed
the earlier L92.5 experiment, are excluded.

Within each cohort, sort physical-unit keys and permute with
`default_rng(20260910 + data_id)`. Allocate 60%/20%/20% physical units to
train/validation/test.

## Strong extrapolation contract

- Normalize each unit's luminous flux by its first-three-observation median.
- Use a causal five-observation history and predict normalized health two
  observations ahead. The horizon was reduced from five during a pre-score
  feasibility check because Data_ID_6 has only eleven unique times per unit.
- Let `T` be the 90th percentile of the maximum usable time among train units.
- Define forecast-origin progress from zero at the first origin allowed by the
  five-step history to one at the last origin allowed by the two-step horizon.
- Training rows: progress <= 0.30.
- Validation rows: 0.40 <= progress <= 0.60.
- Test rows: progress >= 0.75.
- Consequently every validation/test origin is ordinally later than the
  allowed training origins; unit identity is also unseen.
- Features: current health, lag-1/3/4 slopes, trailing mean/std, time/T, and
  history fraction. Test-set/file identity is not a feature.
- Stop a cohort if fewer than four test units or 20 test rows remain. This
  threshold was reduced from 30 during the same pre-score feasibility check
  because Data_ID_6 deterministically supplies 20 valid deep-tail rows.

## Model and contract-aware OOD guard

- Widths 16/32, learning rate `1e-3`, weight decay 2.0, seeds 42--46.
- Trust bank `0,.02,.05,.10,.20,.40`.
- Auto-regime router: train-only K=2, at least three validation units per
  active regime, alpha `0,.05,...,.25`.
- Ordered coordinates (time and health level) are deliberately extrapolated
  and must not trigger OOD rejection.
- OOD support is computed only from regime coordinates: lag slopes and
  trailing standard deviation. Prior mass is zero when that distance exceeds
  the maximum unlabeled validation distance.
- Raw unit mean/CVaR/maximum regret caps remain 2%/5%/10%.

## Success

For each cohort report baseline and guarded v1.4 pooled/unit metrics, raw
regret, OOD fallback rate, and ordered-time hull violation. Overall success
requires:

1. 100% test rows outside the train forecast-origin progress interval;
2. guarded RMSE no worse than baseline on all three cohorts;
3. raw mean/CVaR/maximum regret caps on all three;
4. positive R-squared on at least two cohorts.

No threshold, horizon, feature partition, or route setting may change after
the three scores are materialized.
