# Device-B degradation boundary PP confirmation protocol

Frozen before downloading or inspecting numeric `deviceb` values.

## Source and scope

- `Auburngrads/SMRD.data::deviceb`, from Meeker and Escobar (1998).
- Metadata describe 570 repeated measurements of integrated-circuit power-output
  degradation across individual devices and three junction temperatures.
- Failure is explicitly defined as power output dropping more than 0.5 dB below
  its initial value. The dataset was simulated from physical theory and limited
  proprietary observations, so it is controlled physics-based evidence rather
  than a new real-device cohort.
- Repository search found no prior use.

## Frozen evaluation

- Use leave-one-device-out evaluation across all eligible devices.
- A device is eligible if it has at least 8 pre-failure observations and its
  observed powerdrop brackets 0.5 dB. Interpolate the failure hour between the
  two bracketing measurements.
- Score the final 30% of strictly pre-failure rows for every held-out device.
  Confirmation is inconclusive with fewer than 8 eligible devices or 32 pooled
  score rows.
- Inputs are temperature, current powerdrop, exact boundary margin, normalized
  hour, prefix-only slope/robust slope/curvature/noise, initial-relative change,
  and prefix length. Device ID is excluded.

## Frozen model selection

- Primary PP is log boundary-quotient PP; ordinary boundary-quotient and
  safety-continuation gated PP are prespecified alternatives.
- Inner grouped folds over outer-development devices select width `{16,32,64}`,
  learning rate `{5e-4,1e-3}`, weight decay `{0.5,2,5}`, residual bound
  `{0.25,0.5,1.0}`, and margin floor `{0.02,0.05,0.10}` as applicable.
- Stage 1 seed 42 retains five configurations; stage 2 seeds 42--44 select by
  held-out-device RMSE. Five seeds 42--46 form each outer prediction.
- Matched MLP receives the same inputs and tuning budget.
- Serialize every LODO prediction before aggregate scoring.

## Success

Require at least the stated eligibility, positive pooled PP R², lower PP RMSE
than matched tuned MLP, and positive lower 95% bound for paired-device
`MLP RMSE - PP RMSE`. Otherwise report failure or inconclusive unchanged.
