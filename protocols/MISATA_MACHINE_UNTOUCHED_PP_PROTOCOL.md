# Misata machine-degradation untouched PP confirmation protocol

Frozen before downloading or inspecting the numeric archive.

## Evidence tier and source

- Public dataset: Misata Studio *Machine degradation*.
- Public metadata specify 100 simulated machines, 23,118 longitudinal readings,
  four failure modes, exact run-to-failure RUL, and a machine-disjoint 80/20
  train/test split. The download is 526 KB compressed and CC0.
- This is an untouched **controlled synthetic external cohort**. It tests
  mechanism coverage and implementation generalization but is not presented as
  independent real-world confirmation.

## Frozen task

- Respect the publisher's train/test machine split exactly.
- Never use `rul_cycles`, `machine_failure`, `life_cycles`, or the separate
  `ground_truth.damage` as model inputs.
- Causal inputs are cycle, observed failure mode, control type, tool wear,
  vibration, torque, process temperature, air temperature, rotation speed,
  missingness masks, and prefix-only recent mean/std/slope/curvature features.
- All normalization is fitted on publisher-train machines only.
- Test score uses the final 30% of nonterminal readings from every publisher-test
  machine. Earlier readings provide causal history only.
- The dataset is inconclusive if fewer than 15 test machines or 500 scored rows
  remain after integrity checks.

## Development-only tuning

Split publisher-train machines deterministically into 64 fitting and 16
validation machines with `default_rng(42)` after sorting IDs.

- Training endpoint region: final `{30%, 50%, 70%}` of each fitting machine.
- PP: frozen affine residual or learned affine-gate/direct-residual executor.
- Width `{16,32,64}`, learning rate `{5e-4,1e-3}`, weight decay `{0.5,2,5}`.
- Stage 1 uses seed 42 and retains five configurations by validation-machine
  RMSE; stage 2 uses seeds 42,43,44. Within 1% select the smaller/stronger
  regularized model.
- Matched MLP receives the same search budget and identical inputs.
- Refit on all 80 publisher-train machines with seeds 42--46. Serialize all
  publisher-test predictions before scoring.

## Frozen success rule

Success requires positive PP pooled test R², lower PP RMSE than tuned matched
MLP, and a machine-paired bootstrap 95% interval for `MLP RMSE - PP RMSE` with
a positive lower bound. Report the result explicitly as synthetic evidence.
