# GaAs laser leave-one-device-out PP confirmation protocol

Frozen before downloading or inspecting the numeric `gaaslaser` table.

## Source and task

- Source: `Auburngrads/SMRD.data::gaaslaser`, based on Meeker and Escobar
  (1998), *Statistical Methods for Reliability Data*.
- Public metadata state 255 rows from 15 GaAs laser devices aged at 80 C.
  Operating current increase was recorded every 250 hours.
- Failure is explicitly defined as a 10% current increase.
- The scalar degradation coordinate, common known boundary, repeated history,
  and small non-battery footprint make this a high-priority PP cohort.
- Repository search found no prior use of this dataset.

## Frozen cohort evaluation

- Use leave-one-device-out (LODO) evaluation across every eligible device.
- In each outer fold, all rows of one device are test-only and every row from
  that device is absent from fitting, normalization, model choice, and epoch
  selection.
- A device is eligible if it brackets the 10% boundary and has at least 8
  strictly pre-failure observations. Failure time is linearly interpolated
  between the two bracketing observations.
- The confirmation is inconclusive if fewer than 8 devices are eligible or
  fewer than 24 pooled outer-test tail rows are available.
- RUL is interpolated failure hour minus current hour, clipped at zero.

## Causal inputs and extrapolation region

At a prediction time only the observed prefix may be used. Features are current
increase, boundary margin, normalized hour, delta from initial value, recent
and robust prefix slopes, recent curvature, local residual noise, observed
history fraction, and prefix length. The final 30% of strictly pre-failure rows
in the held-out device form the scored tail; earlier rows provide history only.

## Development-only model selection

Within every outer fold, use deterministic grouped inner folds over the
remaining devices. No outer-device value may select a model or epoch.

- PP candidates: ordinary boundary-quotient, log boundary-quotient,
  frozen-affine residual PP, and learned affine-gate residual PP.
- Width `{16, 32, 64}`, learning rate `{5e-4, 1e-3}`, weight decay
  `{0.5, 2.0, 5.0}`, quotient residual bound `{0.25, 0.5, 1.0}`, and
  log-quotient margin floor `{0.02, 0.05, 0.10}`.
- Use a staged search: screen one seed (`42`) on inner folds, retain the best
  three configurations, then rank those with seeds `42,43,44` by mean inner
  device RMSE. A configuration within 1% of the best uses the simpler/narrower
  model.
- Refit the selected configuration on outer-development devices with seeds
  `42,43,44,45,46`. The primary fold prediction is their mean; per-seed
  dispersion is retained.
- The matched plain MLP receives the same staged width/learning-rate/
  weight-decay budget and identical inputs and inner folds.

## Frozen success rule

Serialize all outer-fold predictions before computing aggregate metrics.
Confirmation succeeds only if eligibility is met, PP pooled LODO tail R² is
positive, PP pooled RMSE is lower than the tuned matched MLP, and a paired
device bootstrap 95% interval for `MLP RMSE - PP RMSE` has a positive lower
bound. Otherwise report failure or inconclusive without changing this protocol.
