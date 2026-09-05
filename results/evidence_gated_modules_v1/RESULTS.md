# Evidence-gated modular extrapolation heads

Candidate heads were frozen as plain NN, affine tail, and PP. A non-plain head is eligible only when its mean validation MSE improves by at least 2% and it beats plain NN in at least four of five seeds. Final-test labels are not used for routing.

| dataset | validation-selected route | plain NN R² | affine R² | PP R² | selected R² |
|---|---|---:|---:|---:|---:|
| HUST | **pp** | 0.758±0.032 | 0.601±0.000 | 0.724±0.039 | **0.724±0.039** |
| Virkler | **pp** | -0.604±0.472 | -0.823±0.000 | 0.857±0.019 | **0.857±0.019** |
| NASA | **B0005:pp, B0006:pp, B0007:pp, B0018:pp** | 0.233±0.094 | 0.424±0.000 | 0.495±0.004 | **0.495±0.004** |

This is a retrospective development test of the routing algorithm. The selected route must be frozen before evaluation on a new cohort to support a confirmatory coverage claim.
