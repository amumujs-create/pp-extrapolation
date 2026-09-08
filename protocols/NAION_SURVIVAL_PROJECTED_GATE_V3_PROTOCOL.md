# Na-ion prospective survival-projected PP gate (v3)

Version 3 is frozen after observing group-2 and before downloading or inspecting
any group-3 cell. Group 2 revealed a specific failure: a few late predictions
rose to the global training-target ceiling even though remaining life must shrink
with current age. Group 2 is model-development evidence; group 3 is the untouched
prospective confirmation cohort.

## Locked cohort and base protocol

- Public BatteryLife raw files `270040-3-[1-8]-*.csv`, in lexicographic order.
- First four cells train, next two validation, final two test.
- Eligibility, causal features, train-only observation boundary, matched PP/plain
  MLP, and seeds 42--46 are unchanged from the v1/v2 protocols.

## Survival-support projection

For both PP and the matched MLP, every raw RUL prediction is projected as

`RUL_projected(t) = clip(RUL_raw(t), 0, L_known_max - age(t))`.

`age(t)` is the observed cycle span at prediction time. During validation,
`L_known_max` is the maximum train-cell lifetime. During test, it is the maximum
lifetime among train and validation cells, whose outcomes are available before
test prediction. No test endpoint or test RUL enters the bound. This is a causal
feasibility constraint rather than a fitted output calibration.

## Frozen pre-outcome gate

The v2 four-condition gate is retained, but validation relative MSE and seed
disagreement are computed after survival projection. Test prefix compatibility
still uses only observations through the train-defined boundary. The evaluator
must write `gate_decision_preoutcome.json` before constructing test targets.

Prospective success requires approval followed by projected PP pooled R2 > 0 and
projected PP pooled R2 at least projected plain-MLP pooled R2, or a correct
rejection otherwise. Raw and projected metrics, all seeds, and per-cell scores
must be retained. No rule may change after outcome materialization.

