# Luminosity temperature-shift CCMR-L v1.6.1 protocol

**Frozen:** 2026-09-11, 박진서, before downloading or opening
`luminosity.rda`.

## Source and limitation

- `Auburngrads/SMRD.data` v0.13.1, GPL-2.
- Git blob `468c44a8658ce6cb0add25cd9d453f84f2863a7b`.
- Published object metadata: 6,664 bytes, 2,175 rows, columns `hours`,
  `celsius`, `unit`, `luminosity`.
- Source citation: Meeker and Escobar (1998), *Statistical Methods for
  Reliability Data*.

The package documentation does not identify the exact luminous device or
state whether values are measured or simulated. Report this provenance
limitation regardless of score.

## Frozen physical condition split

After loading labels only, sort distinct temperatures ascending.

- one-shot test: all physical units at the lowest temperature;
- validation: all units at the second-lowest temperature;
- train: all units at every remaining temperature.

The physical-unit key is `(celsius, unit)`. No unit may cross splits. The run
is inconclusive if there are fewer than three temperatures, fewer than 20
train units, or fewer than 5 validation/test units.

## Strong extrapolation

- Sort each unit by `hours`.
- Normalize luminosity by the median of its first three observations.
- History 3, next-observation horizon.
- train origins: within-unit progress <= 0.30.
- validation origins: progress 0.40--0.60.
- test origins: progress >= 0.75.
- Anchor: current normalized luminosity persistence.
- Correction features: one-step slope, two-step slope, and their difference.
- Context: current health, correction features, trailing mean/std, and causal
  history fraction.
- Exclude time, progress, temperature, unit identity, and future statistics
  from model features.

Require at least 20 test origins and test progress wholly beyond train
progress.

## Frozen model and success

Use raw-regret-corrected CCMR-L v1.6.1 with deterministic 20-fold physical-unit
ensemble and all thresholds from
`protocols/PEROVSKITE_PHENOTYPE_SHIFT_CCMR_V161_PROTOCOL.md`.

Serialize predictions before test scoring. Strong confirmation requires
positive pooled R-squared, pooled and unit-macro RMSE improvements of at least
0.5%, raw test unit mean/CVaR20/max regret <= 0%/2%/5%, correction coverage
>=10%, and exact fallback replay error zero. Otherwise report safe fallback
or failure without changing the split.
