# MATR 2019 latent-transition PP confirmatory protocol

Status: frozen before downloading or inspecting the 2019-01-24 MATR file. This cohort has not been used in PP/PAE development in this workspace.

## Eligibility and split

Read every valid cell in stored order. Abort without scoring if fewer than 12 cells have at least 24 valid cycle-level discharge-capacity observations. With `n` eligible cells, assign the first floor(0.60n) to train, the next floor(0.20n) to validation, and all remaining cells to untouched test. Cell removal after viewing capacity trajectory or lifetime is prohibited; only non-finite or structurally unreadable cells may be excluded and must be reported.

## Deep-future construction

For every cell create causal eight-cycle windows from `QDischarge`. Features are current and eight-cycle mean capacity, current and mean nonnegative capacity-loss rate, capacity endpoint change, and loss-rate endpoint change. Cell identity, charge policy, final lifetime, future capacity, and future cycle index are excluded.

Set the labelled train boundary to the 25th percentile of train capacity endpoints. Retain train windows strictly above it. Validation and test candidates must be strictly below the minimum retained train capacity, hence outside the one-dimensional train capacity convex hull. Abort if validation or test has fewer than 20 candidates or less than 100% outside-hull coverage. Target is remaining recorded cycles to the cell endpoint.

## Frozen models

Compare Ridge/affine, plain NN, original PP, counterfactual-Jacobian regime-spline PP, and latent-transition PP with seeds 42--46. Architectures, optimizer, epoch limit, clipping, and seed list are the public implementation at protocol commit. Jacobian weight grid is `(0, .01, .1, 1)` with ray multiplier 1. Latent expert-separation grid is `(0, .001, .01)` and gate regularization grid is `(0, .001, .01)`. Select only by validation MSE independently per seed. No test ensemble member, epoch, coefficient, cutoff, or feature may be selected after scoring.

## Endpoints and decision

Primary endpoint is pooled R² on every eligible test-tail observation. Also report pooled RMSE/MAE, unit-macro R², five-seed mean and SD, prediction ensemble, gate distribution, and Jacobian violation rate. Latent-transition PP is confirmatorily supported only if its ensemble pooled R² exceeds both plain NN and original PP and its mean single-seed R² exceeds both with no greater seed SD than plain NN. Report all results even if this criterion fails.

Data source: MATR file `2019-01-24_batchdata_updated_struct_errorcorrect.mat`, distributed at data.matr.io and used in Attia et al., Nature 2020.
