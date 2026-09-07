# FT comparison with original PP budget

Post-hoc extension; previously inspected test datasets, not new confirmation. Original PP is preserved unchanged. FT remains the official rtdl-revisiting-models 0.0.2 FT architecture with no PP gate, prior activation, affine residual or Jacobian loss.

Match original PP: 300 epoch maximum, stop when epoch-best_epoch >70, seed-specific validation selection among nine candidates for each of seeds42–46. Same within-dataset inputs, archived splits, equal-group training MSE, scaling from training only, clipped validation/test predictions. Select each candidate checkpoint and each seed's configuration with validation only. Use float64 validation scoring to match PP's target precision. FT grid is the previously declared extended grid (width/depth and optimizer triples); PP's original grid consists of its own regularizer strengths. Candidate count and stopping rule match, not search axes or compute. Two independent CPU seed jobs run concurrently with two torch threads each.

Data: HUST/Virkler archived seen-unit later-coordinate evaluation, MATR2019 archived unseen-cell capacity-tail. These are not interchangeable extrapolation types.

Primary output: all single-seed scores and mean/SD, five-seed ensemble pooled R²/RMSE, macro and per-unit metrics. Preserve prior scores separately. Save selected checkpoints, all candidate validation scores and raw/clipped test predictions. No test-based model reselection. This is finite-budget comparison, not a claim of globally optimal FT or PP.

Validation distinction: PP's learned prior/gate and its architecture-specific activation are part of PP. Generic checkpoint/hyperparameter validation selection is also offered to FT. A claim about the advantage of the selection procedure itself requires a separate ablation, not withholding ordinary validation tuning from baselines.
