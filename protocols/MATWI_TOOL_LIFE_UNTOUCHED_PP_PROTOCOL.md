# MATWI tool-life untouched PP confirmation protocol

Frozen before downloading or inspecting the numeric `labels.csv` file.

## Source and rationale

- Dataset: MATWI, KU Leuven RDR, DOI `10.48804/GK6LHH`, version 1.0.
- Public metadata state that 17 physically separate carbide milling tools were
  run from new condition to failure with about 100 measurements over each life.
- Only the public 534.8-KB `labels.csv` is used; images and raw sensor archives
  are not downloaded.
- Expert wear, wear type, tool identifier, and causal measurement order are the
  available inputs. The target is cuts remaining to the documented terminal
  failure record of each tool.
- Repository search found no prior MATWI use. This is a real, small,
  non-battery untouched cohort.

## Frozen split

Read only unique tool/set identifiers, sort them lexicographically, then apply
NumPy `default_rng(42).permutation`. The first 12 tools are development and the
last 5 are test. No numeric wear value, terminal length, or test target may
alter this allocation.

## Admissibility and causal rows

- Order each tool by its provided sequential measurement identifier, using
  timestamp only to break a documented tie.
- Exclude a duplicate measurement row only by exact identifier duplication and
  report every exclusion.
- A tool is eligible with at least 30 ordered records and a finite expert wear
  value at at least 80% of its records. Causal forward fill within a tool is
  allowed after its first finite wear value; backward fill is forbidden.
- RUL is the number of subsequent recorded cuts until that tool's final failure
  record. It is never supplied as an input.
- At an endpoint, features use its prefix only: current wear, initial-relative
  wear, recent mean/std/slope, robust prefix slope, recent curvature/noise,
  prefix length, and one-hot wear type. Tool ID and file names are excluded.
- Test scoring uses the final 30% of nonterminal rows of each held-out tool.
  Earlier test rows may provide causal history but are not scored.
- Confirmation is inconclusive if fewer than 4 test tools or 80 pooled scored
  rows remain eligible.

## Development-only nested tuning

Use deterministic four-fold grouped CV across the 12 development tools. Every
validation fold holds out whole tools. Candidate selection uses mean
validation-tool RMSE and cannot access the five test targets.

- Endpoint training region: final `{30%, 50%, 70%}` of each development tool.
- PP executor: frozen affine plus bounded residual, or learned affine-gate with
  direct residual sharing (the axial-fan repair architecture).
- Width `{16, 32, 64}`, learning rate `{5e-4, 1e-3}`, weight decay
  `{0.5, 2.0, 5.0}`.
- Stage 1 evaluates all configurations at seed 42 for at most 300 epochs and
  keeps the best five. Stage 2 evaluates those at seeds 42, 43, and 44 for at
  most 450 epochs. Within 1% of best mean RMSE, choose the smaller width and
  stronger weight decay.
- A matched plain MLP receives the identical feature, endpoint-region, width,
  learning-rate, weight-decay, fold, epoch, and seed search budget.
- Refit the selected PP and MLP separately on all 12 development tools with
  seeds `42,43,44,45,46`; primary prediction is the five-seed mean and seed
  dispersion is reported.

## Frozen outcome rule

Serialize five-seed predictions and test row order before aggregate scoring.
Confirmation succeeds only if admissibility is met, PP pooled test R² is
positive, PP pooled RMSE is lower than tuned matched MLP, and the paired-tool
bootstrap 95% interval for `MLP RMSE - PP RMSE` has a positive lower bound.
Otherwise report failure or inconclusive without changing this protocol.
