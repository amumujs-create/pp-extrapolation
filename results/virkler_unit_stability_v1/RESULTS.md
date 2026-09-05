# Virkler source-unit stability audit

The 48 source specimens are divided into three folds. Each held-out fold is split again into checkpoint and route-audit specimens. Training uses crack length <=33 mm and both held-out subsets use the >33 mm tail.

| fold | plain NN R² | PP R² | PP MSE gain | seed wins | stable |
|---:|---:|---:|---:|---:|:---:|
| 0 | -0.773±0.399 | 0.764±0.092 | +0.867 | 5/5 | True |
| 1 | -0.703±0.492 | 0.786±0.044 | +0.875 | 5/5 | True |
| 2 | -0.803±0.204 | 0.517±0.118 | +0.732 | 5/5 | True |

Unanimous PP route: **True** (3/3 folds).
