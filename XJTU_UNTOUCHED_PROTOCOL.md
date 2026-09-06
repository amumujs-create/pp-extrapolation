# Locked untouched XJTU-SY protocol

Protocol status: frozen after the distance-shell implementation commit `f6eb4c3` and
before downloading or parsing XJTU-SY signals.

## Cohorts

- Train: all five bearings at condition 2 (2250 rpm, 11 kN).
- Validation: all five bearings at condition 1 (2100 rpm, 12 kN).
- Test: all five bearings at condition 3 (2400 rpm, 10 kN).
- Units are disjoint. Test labels and lifetimes are opened only after fitting and the
  shell-gate decision have been serialized.

Conditions 1 and 3 have equal Euclidean distance from condition 2 after scaling speed
by 150 rpm and load by 1 kN. This makes validation a direction-opposed, equal-distance
calibration shell rather than a nearer interpolation split.

## Causal features and target

For every vibration recording, compute train-independent time-domain statistics for
each axis: mean, standard deviation, RMS, peak absolute amplitude, skewness, kurtosis,
and crest factor. Add observed speed and load. Use the most recent eight recordings to
form last, mean, and endpoint-change summaries for each signal statistic. Do not use
cycle index, total lifetime, future measurements, bearing identity, or filename count
as model inputs. Target is remaining recording intervals to the final observation.

## Models and decision

- Ridge/affine head, plain NN, PP, and support-adaptive PP use the existing frozen
  training code and seeds 42--46.
- The shell gate uses `DISTANCE_SHELL_PROTOCOL.md` without changing edges or thresholds.
- The gate decision is recorded before test labels are scored.
- Primary metrics: raw pooled R² and RMSE; secondary: bearing-macro R² and prediction
  coverage. Report every result, including abstention or negative R².

The dataset source revision and hashes will be recorded after acquisition. Any parser
fix needed after inspecting file format must not change cohorts, features, target,
models, or gate constants.
