# MEMSS Fatigue untouched PP confirmation protocol

Frozen before downloading or inspecting the numeric `Fatigue` table.

## Source and rationale

- Source: `MEMSS::Fatigue`, documented from Lu and Meeker (1993).
- The public metadata state 262 observations from 21 independent metal test
  paths (`A`--`U`), measured every 10,000 cycles.
- Every specimen starts from a 0.90-inch notch. Failure is defined at a crack
  length above 1.60 inches; testing otherwise stops at 120,000 cycles.
- This is a small, real, non-battery degradation cohort with a directly
  observed scalar health coordinate and a known physical failure boundary.
- Repository search found no prior use of this dataset. It is therefore an
  untouched confirmation candidate for PP at the time of this freeze.

## Frozen split

After reading only the unit identifiers, sort them lexicographically and apply
NumPy `default_rng(42).permutation`. Assign the first 13 paths to training, the
next 4 to validation, and the final 4 to test. No path may appear in more than
one split. The split and all preprocessing remain fixed across seeds.

## Target and admissibility

- Convert documented relative crack length to an absolute boundary coordinate
  with `failure_rel_length = 1.60 / 0.90`.
- For each path, define failure time as the first observed cycle at or above
  this boundary. Linear interpolation between the two bracketing measurements
  is allowed and must then be used for every split.
- A path that never brackets the boundary is right-censored and excluded from
  supervised RUL fitting and scoring; its count must be reported.
- RUL is failure cycle minus current cycle and is clipped at zero.
- A usable path must contain at least 8 pre-failure observations. Confirmation
  is inconclusive if fewer than 3 test paths or fewer than 12 pooled test rows
  remain eligible.

## Causal examples and evaluation region

At time `t`, features may use observations at or before `t` only:

- current relative crack length and remaining boundary margin;
- normalized cycle time;
- first-observation delta;
- recent finite-difference slope and robust median slope;
- recent curvature and local residual noise;
- prefix length / history availability.

For each eligible path, the first 70% of its strictly pre-failure observations
form the prefix region and the final 30% form the tail region. Training uses all
available rows from training paths. Validation and test scoring use tail rows
only, so the task is future-boundary extrapolation on unseen specimens.

## Frozen models and selection

- Primary model: one log boundary-quotient PP,
  `RUL = max(margin, 0) * exp(affine + bounded neural residual)`, with the
  repository implementation and a residual bound of 0.5.
- Comparator: a matched-width plain MLP trained on the identical features and
  rows.
- Diagnostic baselines: frozen affine quotient and recent-rate projection.
- Seeds: 42, 43, 44, 45, 46. Width 32. Validation chooses the epoch separately
  for PP and MLP; no test label can influence selection.
- The primary reported prediction is the mean across the five frozen seeds.
  Per-seed results and dispersion must also be reported.

## Outcome handling and success rule

Predictions, unit order, and eligibility must be serialized before test targets
are passed to the scoring routine. The run is a confirmatory success only if:

1. the admissibility threshold is met;
2. PP pooled test R-squared is positive;
3. PP pooled test RMSE is lower than matched MLP RMSE; and
4. the paired unit bootstrap 95% interval for `MLP RMSE - PP RMSE` has a
   positive lower bound.

Otherwise record failure or inconclusive status without changing this protocol.
