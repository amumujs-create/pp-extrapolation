# Tongji CY45-05 CCMR v2.2 one-shot protocol

**Frozen:** 2026-09-11, 박진서.

Population, capacity, split, features, and success caps are identical to
`protocols/TONGJI_CY45_05_CCMR_V20_PROTOCOL.md`. Only the deployment route
is upgraded to frozen CCMR v2.2.

## Model

Use frozen CCMR v2.2 from `protocols/CCMR_V22_FROZEN_MANIFEST.json`:

- predictor: unchanged `fit_causal_dynamics_bank` /
  `predict_causal_dynamics_bank`
- route: `select_ccmr_v22_route`
  (`stable_bank` | `small_crossfit_bank` | `cautious_causal` |
  `exact_fallback`)
- adaptive causal gate unchanged from v2.0

Write to a distinct result directory. Do not overwrite the v2.0 Tongji
artifacts.
