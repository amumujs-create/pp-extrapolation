# Adaptive affine controlled experiment

Post-hoc MATR 2019 development. No new untouched confirmation is claimed.

1. PP adaptive: same nine width/optimizer candidates as extended PP, same training rule and seeds; only affine weights become trainable. Compare against archived unregularized PP under the identical rule.
2. FT affine frozen and FT affine adaptive: retain exactly the validation-selected baseline FT configuration and five refit seeds. Prediction is affine(x)+FT(x)-FT_initial(x). The copied initial FT is frozen and ensures the initial predictor equals affine. This is a new FT-based research variant, not the previous PP. Fixed versus trainable affine isolates freezing under the same residual formulation.

Same 150 epoch cap, patience 25, group weighting, clipping and train/validation split as the extended benchmark. Hyperparameters are not picked using test performance. All arms are reported regardless of gain. Frozen-initial FT doubles the FT forward computation; this compute cost must be disclosed. Success on this development cohort does not establish generalizable novelty.
