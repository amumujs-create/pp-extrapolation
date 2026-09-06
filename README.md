# PP Extrapolation

PP is a compact neural regressor for strict out-of-support RUL experiments. It contains an affine tail path and a learned two-layer tanh correction path. The model forward pass contains no domain degradation equation.

## Install

```bash
git clone https://github.com/amumujs-create/pp-extrapolation.git
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

The optional soft-anchor variant lets the affine path move under a validation-selected
quadratic prior penalty. Its retrospective cross-domain results are in
[SOFT_ANCHOR_RESULTS_KO.md](SOFT_ANCHOR_RESULTS_KO.md).

The support-adaptive extension attenuates the NN correction as an observation moves
outside train support. See
[SUPPORT_ADAPTIVE_PP_RESULTS_KO.md](SUPPORT_ADAPTIVE_PP_RESULTS_KO.md).

## Optional prior learning (experimental)

The Python API accepts `PriorPairs` (confidence-weighted output-difference intervals)
and `TransportTriples` (detached inward secant transfer to a declared outward path).
Both loss weights default to zero. These are soft assumptions, not causal
identification or global monotonicity/accuracy guarantees. The CSV CLI remains
baseline PP. See [prior learning documentation](PRIOR_LEARNING_KO.md).

```bash
python examples/prior_ablation.py
```

This runs a synthetic development ablation including incorrect priors; it does not
reproduce or replace the previous HUST, Virkler, or NASA results.

The [NASA real-data ablation](NASA_PRIOR_ABLATION_RESULTS.md) reports 80 fits:
direction-only matched PP (inactive penalty), while the fixed transport penalty
reduced pooled R² from 0.495 to 0.470. It is exploratory negative evidence;
these prior options are not recommended as an established accuracy improvement.

The [train-only reliability gate](NASA_PRIOR_GATE_RESULTS.md) also failed: it
selected priors on three of four folds and scored 0.469. A fixed
[distance-decayed transport](NASA_DISTANCE_DECAY_RESULTS.md) limited the damage
and scored 0.494, but still did not improve on PP's 0.495. These reused NASA
results support failure analysis, not selection of a new winning variant.

## Temporal prior assembly (research prototype)

`DegradationContract` configures an increasing/decreasing indicator and failure
boundary. `causal_history` assembles short/long trend priors and their past
one-step forecast errors without future observations. A GRU can adjust the
prior weights and supply a bounded residual. See the
[Korean design](TEMPORAL_PRIOR_DESIGN_KO.md) and
[full negative NASA result and synthetic regime audit](TEMPORAL_PRIOR_RESULTS.md).
This prototype does not improve on baseline PP; it is not the default model.

## Tail and unseen-unit benchmark

The [Korean benchmark report](TAIL_UNSEEN_BENCHMARK_KO.md) narrows evaluation to
late-tail extrapolation and previously unseen physical units. It covers strict
C-MAPSS FD002/FD004 engine extrapolation plus HUST, Virkler, and NASA late tails.
The optional support gate decays only PP's nonlinear correction as train-hull
distance grows. Its cross-domain experiment is reproducible with:

```bash
PYTHONPATH=src python experiments/support_gated_cross_domain.py \
  --legacy-root /path/to/ca-css-ncmapss \
  --output results/support_gated_cross_domain_v1
```

The gate improved HUST but did not improve Virkler or NASA, so it remains an
ablation rather than the default PP predictor.

## PP versus a matched plain neural network

The [fallback evidence report](PP_FALLBACK_EVIDENCE_KO.md) compares PP with a
same-information width-32 plain MLP. PP strongly improves Virkler and NASA but
loses on HUST. Consequently, failure of a domain prior is not sufficient to
select PP: its affine-tail path also needs a source-only extrapolation validation
gate before it can serve as a fallback.

The subsequent [modular gate experiment](MODULAR_GATE_RESULTS_KO.md) shows a
further limitation: validation correctly selected PP for Virkler and NASA but
mis-selected it for HUST, even though PP passed five of six nested source
protocol audits. Source evidence can reject weak modules, but it is not itself a
guarantee that head rankings transfer to a new protocol.

A stricter post-hoc [unanimous-stability summary](results/conservative_gate_summary_v1/RESULTS.md)
routes HUST to plain NN and Virkler/NASA to PP, increasing the three-dataset
macro score by 0.012 over always-PP. Because the unanimity rule was introduced
after observing HUST final rankings, this is a repair hypothesis requiring a new
locked cohort.
