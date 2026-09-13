# SRCIST-PPX external-development protocol

## Objective

Improve CIST-PPX external-cohort coverage without using external labels or target
covariates. The strict success criterion remains higher R2 than Engression on all
five external settings while preserving positive R2 wherever currently positive.

## Frozen mechanisms

1. Mild inverse-frequency weighting across five source RUL quantile regions.
2. Robust affine calibration estimated only from the source validation split.
3. Three-fold cross-fitted gating: calibration is enabled only when it improves
   worst-quartile validation MSE without increasing pooled validation MSE by more
   than one percent.
4. Calibration slope, center shift, and strength are constrained.
5. No external labels, external covariate weighting, or cohort-specific rule.

## Reporting

Report every seed, ensemble R2/RMSE/MAE, mean and minimum R2, seed dispersion,
positive-setting count, and head-to-head wins against the frozen Engression table.
Failed criteria are retained as negative experimental evidence.
