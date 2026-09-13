# Cross-Domain Consensus Residual PP-X protocol

**Frozen implementation audit:** 2026-09-13, 박진서

## Objective

Coverage means predictive performance coverage, not prediction-interval
coverage. The model must preserve positive R2 across seeds and datasets while
improving both the weakest tail and global pooled performance.

## Architecture

For five final PP-X seed predictions, construct an affine-scale-free feature
vector from their mean, median, trimmed mean, spread, range, and signed seed
deviations. A domain-balanced ridge head predicts the normalized residual of
the PP-X ensemble.

Each held-out dataset is processed by a head trained only on the other eight
datasets. At target time the head receives PP-X predictions only, never target
labels.

The frozen update is:

```text
meta correction active when q90 normalized seed disagreement >= 0.12
corrected ensemble = PP-X mean + 0.50 * normalized meta residual
corrected seed k = corrected ensemble + 0.50 * (seed k - PP-X mean)
ridge alpha = 100
```

The final term contracts seed-specific deviations while preserving the
corrected ensemble exactly.

## Development disclosure

Ridge alpha, authority, disagreement cutoff, and seed shrinkage were selected
during retrospective Stage-0 exploration of the nine opened final PP-X
datasets. The LODO audit estimates transfer behavior but is not independent
confirmation. These values must remain frozen on the next unopened cohort.

## Promotion metrics

- 9/9 dataset ensembles retain positive R2
- 45/45 seed-dataset predictions retain positive R2
- no dataset ensemble R2 decreases
- global normalized pooled R2 exceeds 0.918338
- mean dataset R2 exceeds 0.807123
- minimum dataset R2 exceeds 0.465698
- minimum seed R2 exceeds 0.298610
- seed R2 standard deviation decreases in all nine datasets

## Comparators

- frozen final PP-X mean ensemble
- frozen final PP-X individual seeds
- equal-budget Engression where available

No test row from the held-out dataset may enter its fitted meta-head.
