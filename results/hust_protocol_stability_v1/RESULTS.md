# HUST source-protocol stability audit

Each protocol 1–6 is held out in turn. Half of its cells select checkpoints; the remaining cells audit the route. All audit rows lie below the inner-train capacity support. PP passes a protocol only with at least 2% audit-MSE gain and at least four of five seed wins over plain NN.

| protocol | plain NN R² | PP R² | PP MSE gain | seed wins | stable |
|---:|---:|---:|---:|---:|:---:|
| 1 | 0.641±0.073 | 0.847±0.033 | +0.574 | 5/5 | True |
| 2 | 0.248±0.322 | 0.754±0.086 | +0.672 | 4/5 | True |
| 3 | 0.492±0.057 | 0.553±0.155 | +0.120 | 4/5 | True |
| 4 | 0.593±0.052 | 0.837±0.148 | +0.598 | 4/5 | True |
| 5 | 0.641±0.087 | 0.874±0.052 | +0.649 | 5/5 | True |
| 6 | 0.598±0.251 | 0.475±0.351 | -0.307 | 2/5 | False |

PP route stability: **5/6 protocols**. Five-of-six gate: **True**.

This audit was designed after the outer HUST result was observed and is development evidence only.
