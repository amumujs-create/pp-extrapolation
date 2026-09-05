# PP affine-tail path ablation

Plain MLP and PP use identical causal inputs, unit-disjoint strict-tail splits, two width-32 tanh layers, optimizer, seed set, group weighting, clipping, and validation checkpoint rule. Plain MLP uses standard PyTorch initialization; PP adds its frozen validation-selected affine path and zero-start correction.

| dataset | plain MLP pooled R² | PP pooled R² | PP gain |
|---|---:|---:|---:|
| HUST | 0.758±0.032 | **0.724±0.039** | -0.035 |
| Virkler | -0.604±0.472 | **0.857±0.019** | +1.460 |
| NASA | 0.233±0.094 | **0.495±0.004** | +0.263 |

This is a retrospective architectural ablation on previously inspected datasets. It tests the affine-tail path against a matched plain NN; it is not independent confirmation of a fallback routing rule.
