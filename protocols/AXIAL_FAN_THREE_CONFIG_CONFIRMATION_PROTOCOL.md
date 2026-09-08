# Axial-fan three-configuration confirmation protocol

Frozen after a schema-only audit and before opening the three confirmation RUL
files.  The earlier protocol at commit `39cb586` stopped because the publisher's
column description did not match the numeric schema.

## Quarantine and correction

The actual 13-column train/test schema is unit, time, **three** operating
variables and eight health variables.  There is no train-only RUL column.  The
publisher describes train files as complete time-to-end data, so train RUL is
`last observed unit time - current time`.

During the audit, `RUL_FAN_1P_1F.txt` was printed and is now outcome-seen.  The
entire `1P_1F` configuration is quarantined as schema-development data and will
not contribute to confirmation metrics.  The numeric contents of
`RUL_FAN_1P_8F.txt`, `RUL_FAN_4P_1F.txt`, and `RUL_FAN_4P_8F.txt` have not been
opened.  Their publisher hashes were verified without parsing and the files
were moved to `data/axial_fan/sealed/` until predictions are frozen.

## Confirmation cohorts

- `1P_8F`, `4P_1F`, `4P_8F` only.
- Each has 100 complete train units and 100 official truncated test units.
- Unit and time are excluded from model features except `log1p(current time)`.
- Within each train unit, RUL is derived from its own final time.  This terminal
  information creates labels only and is never an input feature.

## Frozen representation and models

Use the representation and model rules in the first protocol with the corrected
three operating columns.  For the `1P` cohort use one train-only regime.  For
`4P` cohorts use four train-only KMeans regimes.  Every normalization and KMeans
fit uses fit units during validation selection and all train units during final
refit.  The causal window is 30 records and train stride is five.

Use the same default width-32 frozen-affine PP and matched width-32 plain MLP,
seeds 42--46.  Select affine alpha and epoch on a seed-42 random 80/20 split of
complete train units.  Refit on all 100 train units for the selected epoch.
Generate and save all predictions before parsing any sealed RUL file.

## Frozen outcome rule

Primary endpoint: equal-weight macro R² across the three confirmation
configurations.  Secondary endpoints: pooled R², per-configuration RMSE/MAE,
five-seed stability, paired unit bootstrap and convex-hull audit.

Confirmation succeeds only if PP pooled R² is positive, PP macro R² exceeds
matched MLP macro R², PP RMSE is lower in at least two of three configurations,
and the paired-unit bootstrap probability of positive pooled RMSE gain is at
least 0.95.  All three cohorts are reported.  No model or feature is changed
once a sealed outcome is opened.
