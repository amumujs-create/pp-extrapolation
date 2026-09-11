# CCMR v1.7 causal-backtest safety protocol

**Frozen:** 2026-09-11, 박진서, before opening the next cohort.

Luminosity is post-test development only. Its test is never reused as
independent confirmation.

## Structure

1. Fit raw-regret CCMR-L v1.6.1 with a deterministic 20-fold physical-unit
   ensemble.
2. For each current unit, issue shadow forecasts throughout its observed
   history.
3. At origin `t`, only use shadow outcomes whose target index is no later than
   `t`; future outcomes are inaccessible.
4. Activate the current correction only if the latest five non-overlapping
   same-horizon shadow forecasts all have positive squared-error gain over
   persistence.
5. Apply the same causal gate on validation. Approve the model globally only
   when gated validation:
   - correction coverage is at least 10%;
   - pooled and unit-macro RMSE improve by at least 0.5%;
   - raw unit mean/CVaR20/max regret are at most 0%/1%/2%.
6. If validation approval fails, deploy exact persistence to every test row.

The row-level gate and global veto are both required. Rejected predictions
must be copied from the anchor, not obtained by multiplying a correction by
zero.

## Development observation

On the opened luminosity shift, the five-win gate alone activated 0.57% of
test rows and reduced the previous maximum raw regret from 29.34% to 1.44%.
However, gated validation violated all risk criteria, so the frozen global
veto selects exact persistence. This is a safe rejection, not performance
evidence.
