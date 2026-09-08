# Locked external evaluation: NASA/UCF randomized and recommissioned batteries

Status: frozen before downloading or inspecting the archive contents or outcomes.  
Source: NASA PCoE, `battery_alt_dataset.zip`, https://data.nasa.gov/docs/legacy/battery_alt_dataset.zip  
Dataset metadata: 26 two-cell 18650 packs under constant, random, and recommissioned loading conditions.

## Eligibility and preprocessing

1. Use first-life packs having at least 10 finite capacity/health observations and an unambiguous monotone time or cycle coordinate.
2. Exclude files only for unreadable structure, duplicate records, or missing capacity/time; record every exclusion before scoring.
3. Define each pack endpoint as the last valid life-cycle observation supplied by the dataset. If an explicit EOL flag is supplied, use that endpoint instead. This is dataset-end RUL unless the archive documents an engineering EOL threshold.
4. Sort eligible pack identifiers lexicographically. Assign the first 60% to train, next 20% to validation, and remainder to test. Never reorder by lifetime, capacity trajectory, condition, or model result.
5. Train observations are restricted to the first 70% of each training pack lifetime. Validation and test scores use observations in the final 30% of their respective held-out pack lifetimes. Earlier held-out observations may only construct causal history features.
6. Inputs use capacity/health, normalized elapsed cycle/time, causal slopes, and causal multiscale history available at the prediction instant. Final lifetime, future capacity, pack identity, and test condition labels are not inputs.

## Frozen models and selection

- Matched plain MLP and affine-tail PP use width 32, two tanh layers, AdamW, equal-unit weighting, seeds 42–46, and validation-only checkpoint selection as in `cross_domain_mechanism_study.py`.
- The four applicability descriptors are normalized horizon, train-unit slope heterogeneity, train degradation SNR, and train curvature gain, exactly as implemented in commit-time `cross_domain_mechanism_study.py`.
- The applicability route is the leave-one-domain-out rule finalized in `meta_applicability_analysis.py` before archive outcomes are read. No threshold may be changed after extraction.
- A 90% interval uses validation-unit block conformal scores scaled by `1 + hull distance`. Test labels are used once for final metrics and interval coverage.

## Predeclared outcomes

- Primary: prediction-ensemble pooled R² on held-out late-tail test rows.
- Secondary: unit-macro R², RMSE, PP-minus-MLP R², physical-unit paired RMSE, hull-distance shell performance, 90% interval coverage and width.
- Confirmatory success requires selected-route pooled R² > 0 and selected-route R² not lower than matched plain MLP.
- Applicability-gate success additionally requires that the frozen route selects the empirically better of PP and plain MLP.
- Interval success requires empirical coverage in [0.85, 0.98]. Coverage above 0.98 is reported as conservative rather than counted as calibrated success.

Any parser repair made after seeing only headers/file types is allowed and logged. Any model, feature, split, threshold, or eligibility change after outcome values are materialized converts the result to development evidence.
