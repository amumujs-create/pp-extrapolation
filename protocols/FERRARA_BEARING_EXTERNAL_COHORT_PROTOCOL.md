# Ferrara bearing external-cohort protocol — frozen before signal download

Status: candidate and protocol fixed on 2026-09-08 after inspecting only the
publisher metadata, dataset article, README-level operating conditions, archive
names, and archive sizes. No vibration trajectory, lifetime, model prediction,
or target metric has been inspected. This is a prospective external-trajectory
evaluation relative to the PP workspace. Public archive metadata are already
known, so it is not described as a privately sequestered or newly recruited
cohort.

## Why this cohort

The University of Ferrara dataset contains six accelerated run-to-failure tests
of the same 1205 ETN 9 self-aligning double-row bearing. Radial acceleration was
sampled for 5 s every 5 min at 25.6 kHz. Speed is fixed at 40 Hz and every test
ends when the acceleration peak reaches the known 20 g failure boundary. All six
bearings subsequently show an outer-race defect. This is a non-battery domain in
which PP's boundary-margin contract is observable before the outcome.

Sources:

- Part 1, E1--E3: https://doi.org/10.17632/htk59pp5wx.1
- Part 2, E4--E6: https://doi.org/10.17632/zz8hpyx939.1
- Dataset article: https://doi.org/10.1016/j.dib.2024.110620

## Frozen physical-unit split

The split follows experiment IDs and published loads; archive byte size and
trajectory length are not used.

| role | experiments | published load | shaft speed |
|---|---|---:|---:|
| train | E1, E2, E3 | 4.0 kN | 40 Hz |
| validation | E4 | 3.0 kN | 40 Hz |
| test | E5, E6 | 4.7, 5.0 kN | 40 Hz |

E5 and E6 remain unopened until extraction code, integrity checks, features,
model settings, and success criteria below have been committed. No experiment
may be moved or removed because of lifetime or prediction quality.

## Target and prediction setting

- One row corresponds to one 5 s acquisition.
- Event time is the first acquisition whose raw absolute acceleration peak is
  at least 20 g. If no such acquisition exists, use the recorded terminal time
  only when the source README explicitly confirms that acquisition triggered the
  20 g stop; otherwise mark the unit right-censored and do not score it.
- `RUL_minutes = 5 * (event_index - current_index)`; no interpolation-generated
  feature or evaluation rows.
- At every prediction time, the model receives all real acquisitions up to that
  time. It receives no future signal, future summary, event index, or lifetime-
  normalized age.
- Fit rows are the first 70% of each training trajectory. Validation and test
  rows are the final 30% of their trajectories, with the preceding history
  available as input. The 70/30 boundary is fixed before trajectories are read.

This evaluates unseen physical units and late-life temporal extrapolation. It is
called strict state-range extrapolation only if at least 50% of scored test rows
lie outside the training interval of the scalar degradation coordinate defined
below. Multi-dimensional convex-hull results are reported separately and are not
substituted for that condition.

## Frozen causal signal representation

For each 5 s waveform, compute raw absolute peak, RMS, standard deviation,
kurtosis, crest factor, peak-to-peak amplitude, and fixed log band powers over
0--1, 1--3, 3--6, and 6--12.8 kHz. Do not select bands after viewing E5/E6.
The scalar boundary coordinate is

`z_t = max_{s <= t} raw_absolute_peak_s`,

and the known remaining margin is `m_t = max(20 - z_t, 0)` g. Additional causal
history inputs are one-, three-, and six-acquisition slopes; expanding mean and
standard deviation; history length; elapsed minutes; load; and shaft speed.
Every numeric transform is fitted on train rows only. The same feature matrix is
provided to PP and all learned baselines.

## Frozen PP and comparators

Primary PP is the current full-development-refit boundary-quotient executor:

`RUL_hat = m_t * softplus(q_affine + bounded_residual_NN)`.

The quotient affine path may use load, elapsed time, current boundary coordinate,
and causal slopes. The residual receives the full frozen representation. Epoch
and residual activation are chosen using E4 only; the selected epoch is then used
for a full E1--E4 development refit, exactly as in
`FULL_DEVELOPMENT_REFIT_AUDIT_KO.md`. Seeds are 42--46 and seed averaging is
reported separately from single-seed robustness.

Primary matched comparator is a direct-RUL MLP with the identical features,
development rows, seeds, optimization budget, validation rule, and full
development refit. Ridge and the current FT-Transformer implementation are
secondary comparators. A simple boundary-rate estimator and elapsed-time-only
baseline are mandatory diagnostics. No baseline is deliberately under-tuned;
their validation grids and compute are reported.

No physics lifetime equation, test-load exponent, test-label calibration, or
post-test routing is introduced in this PP paper experiment.

## Data and outcome gates

Before scoring, require all of the following:

1. SHA-256 duplicate audit finds no equality across E1--E6 archives or extracted
   trajectory fingerprints.
2. E5 and E6 are both readable, have a verified failure endpoint, and each has at
   least 30 final-tail acquisitions.
3. At least three training units and one validation unit remain eligible.
4. Extraction failures are fixed only by source-format rules applied identically
   to every unit; they cannot change the split, target, boundary, or model.

If a data gate fails, record `inconclusive` and do not weaken it.

Primary model success requires PP pooled R² > 0 and PP pooled RMSE lower than the
matched MLP. Also report pooled RMSE/MAE/R², unit-macro metrics, every seed,
per-unit metrics, training-coordinate outside fraction and distance shells.
Because there are only two test bearings, bootstrap intervals and the exact
paired sign test are descriptive; no population-level superiority claim is made.

After the first E5/E6 outcome is materialized, any architecture or threshold
change makes subsequent Ferrara results development evidence. The one-shot
result remains unchanged in the audit trail.

