# NASA health-v2 prior ablation

This is an exploratory comparison on the previously inspected four NASA cells,
not a new independent validation. It uses 4 arms x 5 seeds x 4 folds = 80 fits.
All arrays and targets match the existing NASA health-v2 protocol (255 test rows).
No original dataset is redistributed here.

To build the input archive in the original research workspace with NASA data:

```python
import numpy as np
from run_affine_tail_external_nasa_health_v2 import prepare_folds
folds, _ = prepare_folds()
values = {
    f'f{i}_{part}_{key}': (
        fold[part][key].astype(str) if key == 'groups' else fold[part][key]
    )
    for i, fold in enumerate(folds)
    for part in ('train', 'validation', 'test')
    for key in ('x', 'y', 'groups')
}
np.savez_compressed('/tmp/pp_nasa_health_folds.npz', **values)
```

With this repository installed:

```bash
python experiments/nasa_prior_ablation.py \
  --folds /tmp/pp_nasa_health_folds.npz \
  --output results/nasa_prior_ablation_v1
```

The output directory must not already exist. A protocol and input SHA256 are saved
before fitting. Priors use a design grid derived from train support and health zero;
no test inputs or labels are used to build that grid. Interpolated health states
are structural assumptions, not observed interventions or identified counterfactuals.
All weights are fixed at 10 (same exploratory setting as the synthetic ablation).
Health steps are paired symmetrically about the lower train boundary; allowed
transport error is 2 cycles. Every arm is reported without test-based selection.

Metrics use clipped predictions, whereas prior diagnostics use raw network outputs.
Every epoch records normalized data MSE, relation loss, transport loss, and validation
MSE. A zero relation loss indicates no sampled violation at that stage, not proof
of global monotonicity. The fit selected by validation may differ from the final
epoch. Confidence intervals over only four cells are highly limited; seeds are not
independent datasets.
