# Extended neural comparison — post-hoc MATR 2019

Same archived split and row membership as the original confirmation, including documented boundary deviation. This is additional development benchmarking, not a new confirmation.

Twelve neural arms: PP (latent architecture, regularization losses disabled for this controlled optimizer comparison), MLP, author RTDL ResNet and FT-Transformer, generic two-expert MoE, single-expert PP, fixed-gate PP, unconstrained-gate PP, affine-removed PP, GRU, causal dilated CNN (TCN-style), temporal Transformer.

Tabular arms receive the same six causal summary features. Temporal arms receive the same eight observed capacities and loss rates used to construct those summaries; no future sequence or additional sensor. Test is current-window RUL prediction at unseen-cell capacity tails, not open-loop forecasting from an earlier fixed prefix.

Each neural arm has nine prespecified width/depth/optimizer configurations, searched at seed 42 with validation MSE. Select once, then refit seeds 42–46. Maximum 150 epochs, patience 25, batch 512, group-weighted MSE, identical clipping and target scaling. This is a bounded equal-trial comparison, not exhaustive tuning or equal compute. PP depth is fixed by its current architecture; varying depth applies to the generic models. PP regularizer search from historical results remains a separate arm, never selected by current test scores.

Save every validation candidate, refit score, raw/clipped row prediction, checkpoint, parameter count and timing. Standard baselines Ridge, spline and boosting will use the same summary input and validation-only selection. TabPFN requires a functioning local runtime/checkpoint and any training subsampling must be explicit.

Official tabular implementation: rtdl-revisiting-models 0.0.2, https://github.com/yandex-research/rtdl-revisiting-models . Temporal models are compact task-specific implementations, not claims of reproducing a named paper's complete tuned pipeline.

## Post-hoc extension after initial benchmark results

`pp_joint` repeats nine width/optimizer configurations with expert and gate coefficients tied to the optimizer row: 0 for lr=0.0002, 0.001 for lr=0.0005, 0.01 for lr=0.001. This additional arm was introduced after the unregularized PP result was inspected, and is exploratory, not a prospectively fixed comparison. It does not constitute a full factorial search or proof of optimal PP tuning.

Local TabPFN 8.0.7 with v3 checkpoint: CPU, one internal estimator, random 1,000 training rows for each seed 42–46. This supplementary result uses less training data; it must not be described as full-data parity.
