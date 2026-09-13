# MCST-PPX external-development protocol

## Hypothesis

CIST and SRCIST fail on different source regimes. A source-only minimax consensus
gate can retain CIST's structural extrapolation while using SRCIST only where its
lifecycle balancing is stable.

## Frozen design

1. Expert A is the unchanged CIST slope-integral estimator.
2. Expert B is SRCIST with mild lifecycle balancing and guarded source calibration.
3. Candidate Expert-B weights are fixed at 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, and 1.2.
4. Gate selection uses source validation only.
5. The objective combines pooled MSE, worst RUL-quartile MSE, and three-pseudo-fold
   MSE instability. A candidate must remain within one percent of the better
   expert's pooled validation MSE.
6. No external labels, external covariates, or dataset-specific gate rules are used.

## Acceptance criteria

Primary: beat frozen Engression R2 on all five external settings. Secondary:
improve over CIST in mean R2, minimum R2, and seed dispersion without reducing the
count of positive-R2 settings. Every failed criterion remains reported.
