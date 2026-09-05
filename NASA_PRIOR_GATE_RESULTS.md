# NASA health-v2: train-only prior reliability gate

This is post-hoc method development on the previously inspected NASA cells, not independent confirmation. The gate used only the two outer-training cells in each fold. It alternated them as pseudo-training and pseudo-validation units, retained pseudo-validation rows strictly outside the pseudo-training hull, and used one fixed selector seed. A prior was enabled only when its inner pooled R² exceeded PP by at least 0.01.

| Outer test cell | Inner PP R² | Best inner prior R² | Gate choice |
|---|---:|---:|---|
| B0005 | 0.094081 | 0.174995 | both |
| B0006 | 0.168917 | 0.168917 | PP |
| B0007 | 0.381039 | 0.393906 | both |
| B0018 | 0.371436 | 0.409374 | both |

The selected fold policy was `both / PP / both / both`. On the unchanged 255 outer hull-out rows, its five-seed pooled R² was **0.469328 ± 0.010168**, versus **0.495104 ± 0.003786** for PP. The gate therefore lost **0.025776 R²**.

A fixed-seed control randomly permuted the four inner reliability scores within each fold 10,000 times. Its mean outer R² was 0.486624, with a 2.5–97.5% interval of 0.469105–0.495326. Only 5.13% of shuffled gates scored at or below the proposed gate. Across all 256 possible fold-wise arm assignments, the gate was at the 12.5th percentile from the bottom. These are descriptive reuse analyses, not population-level significance tests.

The experiment falsifies the current gate hypothesis on this reused split: improvement on shallower train-only pseudo-extrapolation did not predict improvement on the farther outer extrapolation. The result does not justify selecting one of the 256 policies by its outer score. The slightly best outer policy was found only after reading test outcomes and is not a valid model choice.

Reproduce with `experiments/nasa_prior_reliability_gate.py`. The output protocol stores SHA256 hashes of the fold and candidate-prediction archives. Source data and result archives are not redistributed.

