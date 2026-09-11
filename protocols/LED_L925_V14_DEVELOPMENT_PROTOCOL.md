# LED L92.5 PP-X v1.4 post-screen development protocol

**Frozen:** 2026-09-10, 박진서, after the preregistered L70 experiment was
declared infeasible and before any model score was computed.

This is explicitly exploratory. L92.5 was chosen after inspecting endpoint
ranges and must not be reported as prospective confirmation.

## Cohorts

- Source and pinned commit are those in `LED_EXTERNAL_SMALL_V14_PROTOCOL.md`.
- Development source: Data_ID_1. Its physical-unit key is
  `(LED_type, Sample_ID)` because `Sample_ID` is reused across LED types.
- Eligible units are those with at least ten observations and two consecutive
  unit-normalized flux values at or below 0.925, with crossing index >= 8.
- Sort eligible sample IDs and apply `default_rng(20260910).permutation`;
  allocate 60%/20%/20% to train/validation/internal test.
- Separate regime-shift stress test: every eligible Data_ID_4 unit under the
  same L92.5 rule. Data_ID_4 was previously inspected and is never used as
  confirmation.

## Examples and target

- Normalize each unit by the median of its first three finite flux values.
- Use a causal five-observation history.
- Features: current health, margin to L92.5, lags 1/3/4 slopes, trailing
  mean/std, normalized elapsed time, and history fraction.
- Target: observed time remaining to the first L92.5 crossing.
- Fit train rows at or above the train-health 25th percentile; score
  validation/test rows strictly below the realized minimum fit health.

## Model and safety

- Widths 16/32, learning rate `1e-3`, weight decay 2.0, seeds 42--46.
- Trust bank `0,.02,.05,.10,.20,.40`.
- Low-volume router: train-only K=2, minimum three validation units per regime.
- Alpha `0,.05,...,.25`; baseline weight at least 75%.
- Raw unit mean/CVaR/maximum validation regret caps 2%/5%/10%, exact fallback.

Report internal and regime-shift sets separately. A useful development result
requires baseline noninferiority and cap compliance on both; it does not
rescue the failed L70 confirmation.

## Post-test failure analysis

The original v1.4 run is immutable. After it exposed a scale-shift failure, a
separate diagnostic guard sets prior mass to zero when train-support distance
exceeds the maximum unlabeled validation distance. This guard was designed
after seeing the failed stress result and is reported only as a post-test
repair ablation, never as the frozen v1.4 score.
