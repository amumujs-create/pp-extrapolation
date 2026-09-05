# NASA health-v2: distance-decayed counterfactual transport

This is post-hoc method development on the same four previously inspected NASA cells. It is not independent confirmation. One hypothesis was fixed for this run: transport confidence decays exponentially with distance from the train boundary, with decay scale equal to 10% of the train health-coordinate range. Weight 10 and tolerance 2 cycles were retained from the uniform transport experiment. No test-based sweep was performed.

| Model | Pooled R² mean ± seed SD | Delta vs PP |
|---|---:|---:|
| PP | 0.495104 ± 0.003786 | — |
| Uniform transport | 0.469558 ± 0.010039 | -0.025545 |
| Distance-decayed transport | 0.494120 ± 0.004428 | -0.000983 |

The paired seed deltas versus PP were -0.000254, -0.001315, -0.001361, -0.002208, and +0.000221. The decay reduced the damage caused by uniform transport, but it did not improve mean accuracy.

Per-cell mean R² changes versus PP were:

| Test cell | PP | Distance-decayed | Delta |
|---|---:|---:|---:|
| B0005 | 0.761385 | 0.752683 | -0.008702 |
| B0006 | 0.217885 | 0.217885 | +0.000000 |
| B0007 | 0.306204 | 0.306217 | +0.000013 |
| B0018 | 0.769689 | 0.769709 | +0.000020 |

The average confidence assigned to the 64 transport relations was about 0.074. Most of the remaining loss came from B0005; other cells were effectively unchanged. This supports a limited conclusion: distance decay can make a misspecified prior less harmful. It does not establish an accuracy gain, a causal effect, or a reliable applicability estimator.

The train-only pseudo-extrapolation gate also failed on these reused data; see `NASA_PRIOR_GATE_RESULTS.md`. Together, the experiments show that shallower internal extrapolation is not sufficient evidence that a prior remains valid farther outside support. Continuing to tune decay or weights against these four test cells would convert the test set into training data. The next valid accuracy test requires a newly frozen rule and untouched cells or another dataset.

Reproduce with `experiments/nasa_distance_decayed_prior.py`. The output protocol stores input hashes and all five seed predictions.

