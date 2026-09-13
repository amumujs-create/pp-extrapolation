# EDIST-PPX external-development protocol

## Objective

Improve cross-cohort coverage by optimizing CIST against source-discovered operating
environments rather than pooled source error. External labels, covariates, and
Engression predictions remain sealed from training and model selection.

## Frozen method

1. Remove the progress coordinate and robustly standardize remaining source features.
2. Project to at most five source principal coordinates.
3. Discover two to four deterministic operating environments using source-only
   quantile-initialized k-means.
4. Fit a pilot CIST and measure validation MSE within each source environment.
5. Construct bounded entropic group-DRO weights with range control and retain fifty
   percent empirical-risk contribution.
6. Fit one DRO-CIST using those training weights.
7. Select a pilot/DRO consensus from weights 0, 0.25, 0.5, 0.75, and 1 by minimizing
   worst-environment validation risk plus pooled-risk and dispersion penalties.
8. Require pooled validation MSE within one percent of the better component.

## Acceptance criteria

Primary: exceed frozen Engression R2 on all five external settings. Secondary:
improve CIST mean R2, minimum R2, and seed dispersion while retaining at least four
positive-R2 settings. All seed-level and ensemble results are reported even when
the criteria fail.
