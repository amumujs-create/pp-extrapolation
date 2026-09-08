# C-MAPSS FD001/FD003 PP cohort protocol

Frozen before any PP score is computed on the FD001 or FD003 official test RUL
files.  These subsets were used in the preceding PAE project, so the evidence is
**PP-model-level prospective**, not dataset-level untouched evidence.  The PP
repository previously evaluated FD002+FD004, not this protocol.

## Data and size

- NASA C-MAPSS FD001 and FD003 train/test/RUL text files already present in the
  sibling research data directory.
- Relevant files occupy about 13 MB; no new large download is required.
- FD001: one operating condition and one fault mode.
- FD003: one operating condition and two fault modes.
- Each subset is evaluated separately, then combined only as a two-cohort
  replication summary.

## Split

For each subset:

1. Sort official run-to-failure training engines by numeric unit ID.
2. First 80% of training engines are model-training units; final 20% are
   validation units.
3. Refit on every official training engine using validation-selected settings.
4. Evaluate one endpoint for every official test engine against the official
   RUL file.  Test RUL values are loaded only after model settings and predictions
   have been fixed in memory.

## Target and causal representation

- Use uncapped physical cycle RUL for both training and test evaluation.
- At each training row, RUL is the unit's terminal cycle minus current cycle.
- Inputs contain the 24 published operating/sensor channels, current cycle, and
  causal window summaries (last value, mean, standard deviation and endpoint
  slope over up to 30 observations).
- Constant channels are removed using model-training data only.
- No unit ID, terminal lifetime, future sensor value or test RUL is an input.

## PP architecture

There is no observed scalar failure boundary in C-MAPSS, so boundary-quotient PP
is ineligible.  Use the generic PP contract: frozen affine RUL path plus bounded
NN residual, with validation-selected affine alpha and stopping epoch.  This is
the same prior role as the existing C-MAPSS support PP, applied to official
unseen-engine endpoints.

Matched comparators are affine-only Ridge and a same-width plain MLP using the
identical features, group weighting, seeds and validation split.  Seeds are
42--46; ensemble means are primary.

## Metrics and success

- Primary: pooled R² across official test-engine endpoints.
- Secondary: RMSE, MAE, five-seed dispersion and per-subset error.
- A subset succeeds if PP pooled R² is positive and PP RMSE is lower than the
  matched plain MLP RMSE.
- The combined replication succeeds only if both FD001 and FD003 satisfy the
  subset rule.  All failures remain reported.
