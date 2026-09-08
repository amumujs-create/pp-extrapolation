# Na-ion prospective relative-skill prefix gate (v2)

Version 2 is frozen after the v1 group-1 result and before downloading or
inspecting any group-2 file. The v1 result showed that requiring absolute
positive validation R2 can reject a stable PP that strongly improves its matched
baseline and succeeds on test. Group 1 is therefore development evidence only;
group 2 is the untouched prospective cohort for this rule.

## Locked cohort and split

- Public BatteryLife raw Na-ion files `270040-2-[1-8]-12.csv`.
- Lexicographic order; first four train, next two validation, last two test.
- Eligibility, feature construction, 60%-of-median-train-lifetime boundary,
  causal rolling-tail evaluation, PP/plain MLP implementations, and seeds 42--46
  are identical to `NAION_PREFIX_GATE_PROTOCOL.md`.

## Frozen gate

Approve PP only when all four conditions pass before test RUL is materialized:

1. PP validation ensemble MSE improves over matched plain MLP by at least 2%;
2. normalized PP seed disagreement on validation is at most 0.25;
3. at least four train and two validation cells are available;
4. every test cell's boundary-prefix robust nearest-train distance is at most 3.0.

Absolute validation R2 is recorded as a diagnostic but is not a gate condition.
The decision concerns whether PP is preferable to its matched baseline; it does
not certify a minimum absolute accuracy. The evaluator writes a committed
`gate_decision_preoutcome.json` before constructing test RUL targets.

## Prospective success

Gate success means approval followed by PP pooled R2 > 0 and PP pooled R2 at
least plain MLP pooled R2, or rejection when either outcome condition fails.
Coverage, decision correctness, per-cell scores, and all five seed scores are
reported. No threshold or split may be changed after group-2 outcome scoring.

