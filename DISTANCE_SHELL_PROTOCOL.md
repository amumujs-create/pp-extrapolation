# Frozen distance-shell gate protocol

The gate partitions normalized train-hull distance into fixed shells
`[0,.5), [.5,1), [1,2), [2,4), [4,inf)`. It uses held-out-unit validation labels
only. A shell is eligible when it contains at least 20 observations, the five-seed PP
ensemble has nonnegative MSE gain over the frozen affine/Ridge head, and normalized
seed disagreement is at most 0.25. Source labels are never inputs to the gate.

At inference, an observation is predicted only when its source distance falls in an
eligible shell and its typed mechanism coverage certificate passes. Otherwise PP
returns `ABSTAIN`. Empty and farther-than-calibrated shells remain rejected.

These constants are frozen before acquiring an untouched bearing dataset. Existing
datasets may be used for diagnostic replay but cannot alter the constants. The next
valid evaluation must report prediction coverage, selective pooled R²/RMSE, failure
recall, and a risk-coverage curve on a dataset not inspected during PP or PAE work.
