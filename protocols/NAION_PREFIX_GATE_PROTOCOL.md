# Na-ion prospective prefix-gate protocol (v1)

This protocol is fixed before downloading or inspecting the Na-ion cell files and
before materializing any test RUL outcome.  It is a prospective external check of
the PP applicability gate, not a development rerun of XJTU or Oxford.

## Data and deterministic cohort

- Public source: BatteryLife raw Na-ion data, files matching
  `270040-1-[1-8]-*.csv`.
- Cell order: lexicographic filename order.
- Train: first four cells; validation: next two; test: final two.
- A cell is eligible when it contains at least 40 distinct cycles with finite,
  positive discharge capacity.
- No file or cell may be exchanged after outcome scoring.

## Prediction protocol

- Capacity for a cycle is the maximum finite discharge-capacity value in that
  cycle. Health is capacity divided by the cell's first-cycle capacity.
- The common observation boundary is 60% of the median train-cell lifetime,
  rounded down to an integer cycle span. It is computed from train cells only.
- Train examples end at that boundary. Validation and test RUL are evaluated at
  every causal time point after the boundary. Features at time `t` use only
  observations at or before `t`; the target is final observed cycle minus `t`.
- The matched plain MLP and PP use the repository's fixed implementations and
  seeds 42--46. Predictions are averaged across seeds for primary pooled R2.

## Gate fixed before test outcomes

The route is approved only if every condition passes:

1. validation PP pooled R2 is positive;
2. validation PP ensemble MSE improves on plain MLP by at least 2%;
3. validation PP seed disagreement, mean predictive standard deviation divided
   by validation-target standard deviation, is at most 0.25;
4. at least four train cells and two validation cells are available;
5. every test cell's boundary-prefix descriptor has robust nearest-train distance
   at most 3.0.

The prefix descriptor contains boundary health, full-prefix slope, early slope,
late slope, quadratic curvature, and detrended-noise scale. It uses no future
test point. Each coordinate is centered by the train median and scaled by train
IQR (with a numerical floor); distance is Euclidean distance divided by the
square root of six.

The evaluator must write `gate_decision_preoutcome.json` before it computes or
accesses test RUL targets. The file includes the decision inputs and a SHA-256
commitment over the test filenames and prefix features. Test labels are scored
after that artifact exists and are never used to change the rule.

## Success criteria

The prospective gate succeeds if it approves a route for which PP pooled R2 is
positive and PP pooled R2 is at least the matched plain MLP pooled R2, or rejects
a route for which either condition fails. Coverage and correctness are reported
separately. With only two test cells, uncertainty and per-cell results must be
reported and superiority significance is not claimed.

