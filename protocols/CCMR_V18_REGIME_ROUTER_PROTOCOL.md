# CCMR v1.8 validation-regime router

**Frozen:** 2026-09-11, 박진서, before downloading the final 228-cell
replication cohort.

All v1.8 design decisions below are post-test development based on previously
opened Luminosity, Concrete, LG M50T, and SIT LFP datasets. None of those
datasets is independent v1.8 evidence.

## Motivation

The five-win causal gate is useful when validation safety is mixed, but on
smooth battery trajectories it can select rare transient streaks and discard
a broadly reliable base correction. CCMR v1.8 first diagnoses the validation
regime, then chooses one of two already-defined routes.

## Frozen route selection

Fit CCMR-L v1.6.1 as before. Select `stable_base` only if all validation-only
conditions hold:

1. base CCMR active fraction >= 50%;
2. base unit-macro improvement >= 10%;
3. base raw unit mean, CVaR20, and maximum regret are each <= -5%.

If all hold, deploy the base CCMR candidate, retaining its row-level
cross-fit consensus and context-support exact fallback.

Otherwise select `cautious_causal`: apply the frozen v1.7 five-shadow-win
causal gate and its gated-validation global veto. If that validation
certificate fails, deploy exact persistence everywhere.

The route is selected before test prediction and cannot use test labels,
coverage, scores, or condition identities.

## Opened-data development checks

- SIT LFP: stable route selected. Post-test candidate audit would improve
  pooled/macro RMSE by 0.77%/3.15%, with raw mean/CVaR/max regret
  -16.03%/0%/0% and 47.00% coverage.
- LG M50T: stable route selected, but test context support rejects every row,
  yielding exact persistence.
- Luminosity and Concrete: stable certificate fails because maximum
  validation regret is positive; they remain on the cautious fallback route.

These observations define behavior but are not confirmatory results.
