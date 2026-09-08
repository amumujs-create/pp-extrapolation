# New non-battery PP cohort protocol — 2026-09-08

Registered before capacitor MAT extraction/model outcomes. This is a new-domain evaluation relative to the local PP reports/scripts searched, not a claim of an independently recruited cohort. HNEI and other successful cohorts remain unchanged.

## Candidate eligibility (outcome-independent)
- NASA capacitor electrical stress: first crossing of C(t)/C(0) <= 0.8, following Renwick et al. (2015), https://papers.phmsociety.org/index.php/phmconf/article/view/2713 . Event time is interval-censored; linear interpolation provides an explicitly approximate threshold-hitting target, not a directly observed catastrophic failure time.
- Use pre-first-crossing observations only. No arbitrary last-record RUL. Noncrossing devices are censored, not assigned an event at record end.
- PHM2019 lap-joint data is acquired for audit. Only 3–9 labeled inspections per training specimen and unverified fracture endpoints: do not count as a successful RUL cohort or multiply ultrasonic repeats into independent specimens.

## Split and data adequacy
- Capacitor IDs in natural ascending order. First floor(0.6 N) train, next floor(0.2 N) validation, remaining test, assigned before event eligibility filtering. No swapping IDs after outcomes.
- Require >= 2 training devices, >=1 validation device and >=2 test devices, and at least 3 eligible observations per evaluated device.
- Strict state-tail task: train uses health >0.9; validation/test uses 0.8 < health <=0.9. Test devices never enter fitting. A separate new-device evaluation can use all pre-event rows but must not be labeled state extrapolation.
- Inputs: current health, elapsed time scaled by training maximum, causal lag slopes (1/2/3 records), available-history mean/std, health change, running minimum, history length/missingness. Only observations up through prediction time. No endpoint-normalized time or future smoothing.
- No interpolation-generated training/evaluation rows. Missing early history is supported with shorter windows and a history-length feature. Threshold event interpolation is for labels only.

## Frozen comparison
- Seeds 42–46, matched information and 300 epochs/70 patience.
- PP default frozen affine plus NN residual (existing code); plain MLP; ridge selected on validation.
- Secondary safe-continuation PP: lambda in {0,.02,.05,.1,.2}, selected by validation ensemble RMSE. Lambda=0 must be described as the NN route, never a PP victory. No new architecture/search after test evaluation.
- Clip each seed's predictions using training target range before ensemble averaging.
- Report pooled R²/RMSE, all seed values, per-device metrics, unit-level paired bootstrap/exact sign-flip and standardized health-hull distance/outside proportion. With few devices these are pilot estimates, not evidence of statistical significance.
- If data adequacy fails, save the audit and mark the cohort not evaluable. Do not lower the eligibility bar after seeing performance.

## Stress-2 auxiliary new-device audit (before model fitting)
The official legacy URL unexpectedly serves a small MATLAB summary (11 ages x 6 columns), although the main NASA page says unavailable. Retain all six provided columns; a numerically similar column is not proof it is a derived average. Independent physical-unit interpretation remains contingent on source documentation. No claim of strict state extrapolation if the tail has fewer than three observations. Separately run the pre-declared all-pre-event/new-device pilot: train IDs 1–3 (right-censored ID1 has no point-RUL labels), validation4, test5–6. All observed pre-event rows are eligible; future values only determine the threshold label. Do not claim a new state-tail success. Model configuration remains unchanged, and no model outcomes have been computed when this addendum is written. Use C as percent capacitance loss provisionally, pending source-reference confirmation; label results provisional until confirmed.
