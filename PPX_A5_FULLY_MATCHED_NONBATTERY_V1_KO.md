# PP-X A5 — Fully Matched Non-Battery Prior Conditioning

Prior-OFF와 Prior-ON은 동일 split, feature, seed, nonlinear architecture, initialization replay, optimizer, budget, checkpoint rule을 사용했다. 차이는 frozen affine prior contribution뿐이다.

| Setting | No-prior NN R² | Prior+same residual R² | ΔR² | Unit RMSE reduction 95% CI |
|---|---:|---:|---:|---:|
| HUST | 0.862 | 0.744 | -0.118 | [-30.907, -12.090] |
| MATR-b2 | 0.767 | 0.686 | -0.081 | [-4.527, -1.454] |
| N-CMAPSS | 0.488 | 0.922 | +0.434 | [-6.514, +12.923] |

Positive/negative settings: **1/2**

This is a retrospective matched ablation, not prospective confirmation.
