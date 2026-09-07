# Frozen Oxford outcome-held-out PP protocol v2

Status: frozen after covariate-only feasibility counts and before Cell7--Cell8 RUL targets or model scores are materialized.

## Cohort

- External dataset: Oxford Battery Degradation Dataset 1.
- Train: Cell1--Cell4.
- Validation: Cell5--Cell6.
- Outcome-held-out test: Cell7--Cell8.
- This is an external-dataset cell holdout. The source file and test covariate counts were inspected during feasibility, so it is not called a globally untouched dataset.

## Boundary and rows

Normalize each cell's measured discharge capacity by its first characterization capacity. Using Cell1--Cell4 only, set the boundary to the median of causal-window endpoint health. Retain train endpoints strictly above the boundary and validation/test endpoints strictly below it. Use an eight-observation causal history. Abort without scoring if validation or test contains fewer than eight eligible rows or if either test cell has fewer than eight rows.

The frozen feasibility values are boundary `0.8396755456924438`, train `123` rows, validation `17` rows, and test-covariate count `56` rows. Test RUL values have not been computed for this protocol at freeze time.

## Inputs and target

Inputs use only observations available through the prediction time. Candidate causal presets match the final NASA PP:

- `short`: last relative health plus 3-step health mean and slope;
- `multiscale`: last health plus 3/5/8-step mean, slope, and standard deviation;
- `moments`: multiscale plus recent mean increment and increment acceleration.

No cell identity, final lifetime, future capacity, or test target is an input. The target is remaining Oxford characterization observations to the final recorded observation; this is dataset-end RUL and is not claimed to equal an engineering EOL threshold.

## Frozen models and selection

Primary model is causal multiscale latent PP with a frozen affine path, learned early/late residual experts, and latent regime gate. Validation-only candidates are the three presets above crossed with separation weights `{0, 0.001, 0.01}`. Candidate selection uses pooled validation MSE; seeds are 42--46. Report both validation-selected-per-seed and one configuration selected by mean validation MSE across seeds.

Comparators are Ridge/affine and official Engression 0.1.9. Engression searches hidden width `{32,64}`, learning rate `{0.001,0.005}`, beta `{0.5,1}`, and `(layers, epochs)` `{(2,250),(3,500)}` by validation MSE, then refits seeds 42--46.

## Metrics and decision

Primary: pooled R². Secondary: pooled RMSE/MAE, cell-macro R², per-cell RMSE, seed mean/SD, hull-out fraction and standardized hull distance. Success requires PP pooled R² > 0 and PP pooled RMSE lower than both Ridge and Engression. All outcomes are reported. No model, feature, boundary, clipping rule, or metric changes are permitted after scoring.
