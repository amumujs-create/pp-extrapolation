# PP Extrapolation

PP is a compact neural regressor for strict out-of-support RUL experiments. It contains a frozen affine path and a learned two-layer tanh correction path. The model forward pass contains no domain degradation equation.

## Install

```bash
git clone REPOSITORY_URL
cd pp-extrapolation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
pytest -q
```

Python 3.9 or newer is required. Installing PyTorch may take several minutes on a new machine.

## Check the full pipeline with generated data

```bash
python examples/make_synthetic.py

pp-run \
  --csv examples/synthetic_rul.csv \
  --features health recent_rate step \
  --target RUL \
  --unit unit \
  --coordinate health \
  --train-cutoff 0.60 \
  --direction low \
  --max-epochs 30 \
  --output results/synthetic
```

The command saves `results.json` and per-seed predictions in `predictions.npz`.

## Run on another dataset

Prepare one CSV row per observation time. The CSV needs:

- a physical unit identifier such as machine, cell, or specimen;
- a target RUL at the current observation;
- causal features available at that observation;
- one predeclared ordered degradation coordinate for the included CLI splitter.

Example:

```bash
pp-run \
  --csv /path/to/my_data.csv \
  --features health elapsed_time prefix_rate recent_rate \
  --target RUL \
  --unit machine_id \
  --coordinate health \
  --train-cutoff 0.70 \
  --direction low \
  --output results/my_data
```

Use `--direction low` when the evaluation region lies below the train coordinate range, such as decreasing health. Use `--direction high` for increasing crack length or wear. The command first splits physical units with seed 42, keeps train rows on the selected side of `--train-cutoff`, and retains validation/test rows strictly beyond the actual train support. It aborts if validation or test is not 100% outside the one-dimensional train convex hull.

For a multi-coordinate convex hull or a domain-specific cohort policy, construct three dictionaries directly:

```python
train = {"x": x_train, "y": y_train, "groups": train_unit_ids}
validation = {"x": x_validation, "y": y_validation, "groups": validation_unit_ids}
test = {"x": x_test, "y": y_test, "groups": test_unit_ids}
```

Then call `select_affine_initialization`, `fit_pp`, and `predict` from `pp_extrapolation`. Use `audit_convex_hull_support` on the predeclared hull coordinates before fitting.

## Frozen model protocol

- Width 32, two tanh hidden layers
- AdamW, learning rate `5e-4`, nonlinear weight decay `2.0`
- Batch size 512, maximum 300 epochs, patience 70
- Gradient clipping at 2.0
- Seeds 42–46
- Train-only feature standardization
- Target scale and output cap equal to maximum train RUL
- Equal total loss weight for every physical unit
- Weighted affine initialization selected on hull-out validation MSE from `0.1, 1, 10, 100, 1000, 10000`
- Affine path frozen while the nonlinear correction is trained

The primary score is raw pooled R² over all test rows. The result file also reports RMSE, MAE, unit-macro R², and the mean-ensemble score.

## Experimental rule

Split physical units before filtering observation rows. Select the hull coordinates and failure boundary before inspecting test labels. Use validation only for alpha and checkpoint selection. Any feature or split change made after test inspection is development evidence and needs another untouched cohort for confirmation.

See [MODEL_AND_SPLIT_KO.md](MODEL_AND_SPLIT_KO.md) for the detailed Korean protocol.

