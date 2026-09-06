# FEMTO/PRONOSTIA PP model-level prospective protocol

Locked before parsing signal values or scoring PP on the official Test_set.

- Source: NASA PCoE `10. FEMTO Bearing.zip`, SHA-256 `e21bb22bd8d54fd18ebe98b4b4e094c0c40469bda19811a2a642d5cc84ebd81f`.
- Evidence label: PP model-level prospective; PAE previously used this dataset and that fact must be disclosed.
- Learning bearings, ordered: Bearing1_1, Bearing1_2, Bearing2_1, Bearing2_2, Bearing3_1 for train; Bearing3_2 for validation.
- Final test: the last observed record of all 11 official truncated Test_set bearings. The target is the published PHM2012 remaining time at that cutoff.
- Inputs: operating condition and causal vibration summaries only: horizontal/vertical RMS, kurtosis and three FFT bands per axis. No total lifetime, future signal, official answer, normalized life fraction or file-count-to-failure feature.
- Signal sampling: every fifth acceleration file, matching the pre-existing loader. One endpoint per official test bearing prevents long trajectories from dominating pooled metrics.
- PP settings: repository defaults, seeds 42–46; affine initialization selected on Bearing3_2 only.
- Comparators: affine/Ridge head, matched plain NN, PP. OOF uncertainty is declared ineligible because only six full run-to-failure Learning bearings exist.
- Primary metric: raw pooled R² across 11 test bearings. Secondary: RMSE, MAE and five-seed dispersion.
- Convex-hull audit: report outside-support status over declared causal vibration inputs and condition. Do not relabel the experiment strict hull extrapolation if the endpoint is inside train support.
- No model, feature, clipping, seed, checkpoint or inclusion rule may be changed after official test scores are read.
