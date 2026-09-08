# Locked external evaluation: HNEI fixed-protocol cells

Status: frozen before downloading `HNEI.zip` or inspecting outcomes.  
Distribution: BatteryLife processed Zenodo record 17958489; original HNEI/Battery Archive data.

Use every readable HNEI cell with at least 20 finite cycle-capacity observations. Sort identifiers lexicographically and assign first 60% train, next 20% validation, remainder test. Train rows end at 70% of each training cell lifetime; validation and test use the final 30% of held-out cells with causal history. Target is cycles remaining to the final valid capacity observation. Inputs, model budgets, seeds, paired inference, and conformal construction are identical to `CALCE_EXTERNAL_LOCKED_PROTOCOL.md`.

The applicability certificate fixed before this download approves PP only if median support distance > 0.5, train-unit degradation-law heterogeneity < 0.05, descriptor train units >= 4, and calibration units >= 2. Primary model success requires PP ensemble R² > 0 and PP R² >= matched plain MLP R². Certificate success means its approve/reject decision agrees with whether PP achieves that primary success. No thresholds, split, features, or model settings may change after outcome materialization.
