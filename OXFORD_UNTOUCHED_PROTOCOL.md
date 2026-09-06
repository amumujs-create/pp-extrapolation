# Locked untouched Oxford battery protocol

This protocol is frozen before downloading or inspecting Oxford Battery Degradation
Dataset 1. The dataset has not been used in the PP or PAE experiments in this workspace.

## Split

- Train units: Cell1--Cell5.
- Validation unit: Cell6.
- Untouched test units: Cell7--Cell8.
- Each characterization cycle is one chronological observation.
- Compute an internal cutoff as the 25th percentile of train endpoint capacity. Retain
  train observations above the cutoff and validation/test observations strictly below
  the retained train capacity support. Abort and report infeasibility if validation or
  test has fewer than eight usable causal windows; do not alter cells or cutoff.

## Features and target

- Extract discharge capacity from the final charge coordinate of each 1C discharge.
- From the latest eight characterization observations use capacity and causal capacity
  loss rate, summarized by last, mean, and endpoint change. Add no cycle index, cell
  identity, total lifetime, or future value.
- Target is the number of remaining characterization observations to the cell's final
  available observation.

## Frozen models and gate

Use Ridge/affine, plain NN, PP, and support-PP with seeds 42--46. Calibrate the distance
shell gate from Cell6 only, using the constants in `DISTANCE_SHELL_PROTOCOL.md`. Write
the gate decision before materializing Cell7--Cell8 targets. Then open test labels once
and report pooled R²/RMSE, cell-macro R², coverage, and selective regret. No result may
change the frozen shell thresholds.

Data citation: Howey, D. and Birkl, C. (2017), Oxford Battery Degradation Dataset 1,
University of Oxford, DOI `10.5287/bodleian:KO2kdmYGg`, ODC-ODbL.
