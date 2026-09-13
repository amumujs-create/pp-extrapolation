# CDCR-PPX external cohort replay protocol

**Frozen replay implementation:** 2026-09-13, 박진서

## Status

Axial-fan, MATWI, and Misata outcomes were opened before this replay. This is
a retrospective external-cohort stress test, not a new prospective result.

## Frozen model

- Train one CDCR normalized residual head on the nine final PP-X development
  datasets.
- Ridge alpha: 100.
- Activate the residual at q90 normalized seed disagreement >= 0.12.
- Residual authority: 0.50.
- Seed-deviation shrinkage: 0.50.
- Do not use any external-cohort label during fitting or prediction.

## Cohorts

- Axial-fan `1P_8F`, `4P_1F`, and `4P_8F`: original five PP seeds frozen
  before official RUL labels were opened.
- MATWI tool-life: original five PP seeds frozen before scoring.
- Misata machine degradation: original five PP seeds frozen before scoring.

## Metrics

- ensemble R2, RMSE, and MAE
- individual-seed R2 mean, minimum, and standard deviation
- comparison with the original PP ensemble and matched MLP ensemble
- absolute-success coverage across external settings

The replay must report all settings whether improved or harmed.
