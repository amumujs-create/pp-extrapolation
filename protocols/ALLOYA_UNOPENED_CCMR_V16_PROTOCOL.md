# Alloy A unopened strong-extrapolation CCMR v1.6 protocol

**Frozen:** 2026-09-11, 박진서, before downloading or reading `alloya.rda`.

## Source

- `Auburngrads/SMRD.data`, GPL-2, package version 0.13.1.
- Source experiment: Hudak et al. (1978), AFML-TR-78-40.
- 21 real metallic-alloy specimens under the same cyclic load.
- Crack length measured every 10,000 cycles until 1.6 inches or 1.2 million
  cycles.
- Published object metadata: 262 rows, 3 columns, 773 bytes.
- Pinned Git blob: `d51e3a7dbe4891636173b8af3242c9fa1c19c096`.

This object and its crack outcomes have not previously been used or screened
in this project.

## Frozen split

Compute SHA-256 of the string `alloya-ccmr-v16:<specimen-id>` and sort by the
hex digest. Assign the first 13 specimens to train, next 4 to validation, and
last 4 to the one-shot test. No outcome-dependent replacement is allowed.

## Task

- Sort each specimen by `megacycles`.
- Normalize crack length by its first observed value.
- Causal history: 3 observations; horizon: next observation.
- Origin progress: row index divided by the final row index for that specimen.
- train origins: progress <= 0.30.
- validation origins: 0.40 <= progress <= 0.60.
- test origins: progress >= 0.75.
- Anchor: persistence, the current normalized crack length.
- Correction features: one-step slope, two-step slope, and slope difference.
- Context features: current health, the three correction features, trailing
  mean/std, and history fraction.
- Absolute time, progress, specimen ID, and failure status are excluded.

The result is inconclusive if any declared specimen is absent, any split is
empty, fewer than 8 test origins remain, or test origins are not 100% beyond
maximum train progress.

## Frozen model

Use CCMR v1.6 exactly as specified in
`protocols/CCMR_V16_POSTTEST_DEVELOPMENT_PROTOCOL.md`:

- ridge alpha grid `.1,1,10,100,1000,10000`;
- train-unit leave-one-out sign agreement >= 90%;
- relative fold dispersion <= 0.5;
- validation context-support quantile 99%;
- deployment mass grid 0--0.25;
- validation raw-unit mean/CVaR20/maximum regret caps 0%/1%/2%;
- exact persistence for rejected rows or rejected model.

## One-shot success

Serialize predictions before reading test metrics. Success requires positive
pooled R-squared, pooled RMSE below persistence, test raw-unit mean/CVaR20/max
regret no greater than 0%/2%/5%, and exact fallback replay error zero.
Thresholds and split remain unchanged after scoring.
