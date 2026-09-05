# Support-gated PP: strict late-tail extrapolation

Train, validation, and final-test physical units are disjoint. Validation and test rows are 100% beyond the one-dimensional train hull along the declared degradation coordinate. Beta and neural checkpoints use validation only.

The model is `affine(x) + exp(-beta * hull_distance) * correction_NN(x)`. Beta zero exactly recovers PP; far outside support the correction vanishes.

| dataset | affine/Ridge head pooled R² | PP pooled R² | support-gated PP pooled R² | 5-seed PP ensemble |
|---|---:|---:|---:|---:|
| HUST | 0.601±0.000 | 0.724±0.039 | 0.736±0.040 (+0.013) | **0.773** |
| Virkler | -0.823±0.000 | 0.857±0.019 | 0.857±0.019 (+0.000) | **0.888** |
| NASA battery | 0.424±0.000 | 0.495±0.004 | 0.495±0.004 (-0.001) | **0.495** |

The primary metric is raw pooled R². HUST uses a censored proxy endpoint; Virkler and NASA use observed endpoints. The support gate is an ablation, not the default: it helped HUST seed-average performance, was neutral on Virkler, and was slightly worse on NASA. These are development results because the same source datasets were used in earlier architecture work.
