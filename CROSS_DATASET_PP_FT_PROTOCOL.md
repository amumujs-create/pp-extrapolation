# Cross-dataset PP / FT comparison

Retrospective development benchmark, not an untouched confirmation. Transfer the existing extended benchmark protocol without test-based changes to HUST and Virkler; reuse completed MATR 2019 under that protocol.

For each dataset compare pp (latent architecture without auxiliary losses), pp_joint (existing nine joint candidates), and official FT-Transformer. Nine prespecified candidates at seed 42, validation MSE selection once, then five refits 42–46; 150 epochs max, patience 25. pp_joint uses the previously declared coupling of optimizer and regularizer weights. This is bounded tuning, not full factorial optimization. PP and FT have the same candidate count and stopping rule but distinct architectures and compute costs.

HUST/Virkler use the archived regime_spline_deep_future splits and four features. MATR uses archived unit-disjoint capacity-tail rows and six features. Inputs and splits are identical between competing models within each dataset, not between datasets. HUST/Virkler involve seen units at later degradation coordinates; MATR involves unseen units. Current-state RUL prediction is not open-loop forecasting. Previously reported PP 0.811 / 0.257 scores have different training protocols and are not substituted into the controlled table.

Report all arms, pooled single-seed mean/SD, ensemble R², per-unit scores, raw/clipped predictions and configuration selections. No model is chosen from test performance. Source implementation: rtdl-revisiting-models==0.0.2.
