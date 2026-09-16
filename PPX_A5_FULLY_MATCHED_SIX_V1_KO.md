# PP-X A5 — Fully Matched Six-Setting Prior Contribution

No-prior NN과 prior+residual에 동일 nonlinear capacity, split, seed, optimizer, budget, checkpoint rule을 적용했다.

| Setting | No-prior R² | Prior+Residual R² | ΔR² | Unit RMSE reduction CI | q_BH |
|---|---:|---:|---:|---:|---:|
| Sunwoda | -1.352 | 0.718 | +2.070 | [+0.539, +0.587] | 0.0117 |
| RWTH | 0.633 | 0.788 | +0.155 | [+0.102, +0.282] | 0.0234 |
| MICH | 0.684 | 0.759 | +0.075 | [+0.004, +0.028] | 0.075 |
| HUST | 0.862 | 0.744 | -0.118 | [-30.978, -12.103] | 0.00421 |
| MATR-b2 | 0.767 | 0.686 | -0.081 | [-4.520, -1.454] | 0.0156 |
| N-CMAPSS | 0.488 | 0.922 | +0.434 | [-6.514, +12.923] | 0.75 |

## Summary

- positive/negative settings: **4/2**
- normalized mean effect: **+0.129**
- setting-bootstrap 95% CI: **[-0.139, +0.395]**
- individually significant prior benefit/harm after BH: **2/2**

**Conclusion:** prior conditioning is heterogeneous and is not a universal performance improvement. Residual correction remains necessary, while the structural prior acts as a setting-dependent extrapolation reference.

This is retrospective matched evidence, not prospective confirmation.
