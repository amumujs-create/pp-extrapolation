# RCIST-PPX external development protocol

RCIST-PPX retains CIST's stochastic damage-rate integral and adds four frozen
mechanisms: target-covariate source weighting without target labels,
regime-by-progress interaction features, deterministic normal quadrature for
the predictive mean, and validation-only output-scale calibration.

Target similarity uses only observed test covariates. Source and validation
weights are exponential functions of nearest target-covariate distance after
source standardization and PCA. Axial validation remains one final
pseudo-endpoint per unit. Output calibration searches authority
`{0,.25,.5,.75,1}` between identity and a weighted affine validation map.

The CIST grid and seeds remain unchanged. Strict success remains R2 above
official Engression in all five opened settings. This transductive use of
unlabeled target covariates is disclosed and does not constitute untouched
external confirmation.
