# PP-X Paper Method v1 — Frozen Protocol

Owner: 박진서  
Status: frozen before any new prospective endpoint is opened

## Paper identity

The paper method is **PP-X**, described as:

> A validation-approved prior-residual framework for contract-conditioned
> extrapolation.

CCMR is not the paper-level model. It is a trajectory-domain risk-aware
executor used as supporting mechanism evidence.

## Scope

PP-X addresses inductive extrapolation to unseen physical units or regimes.
Test labels, test-batch statistics, test-time adaptation, and test-informed
route changes are forbidden.

## Inputs available before test prediction

1. A typed source contract:
   - whether a failure boundary is declared independently of test outcomes;
   - whether ordered progression and causal history are available;
   - whether a regime identifier is observed;
   - whether support heterogeneity can be computed from train data.
2. Group-disjoint source/validation predictions for:
   - matched direct fallback;
   - prior-only path;
   - unbounded prior-residual core;
   - any contract-admissible optional executor.
3. Physical-unit identifiers for all validation rows.

## Candidate executors

- `unbounded`: common prior-residual core;
- `bounded`: fixed residual bound;
- `dual_scale`: support-adaptive local/broad residual bound;
- `regime_transport`: source-regime output-error transport;
- `history`: validation-selected causal history representation;
- `direct_fallback`: matched direct model;
- `persistence_fallback`: trajectory persistence when required by a declared
  safety contract.

Candidates that are not admissible under the typed contract are never scored.

## Frozen decision order

1. **Prior admissibility (PP-X Final, `PRIOR_GATE_VERSION=final`)**
   - a declared boundary admits a BQ / boundary prior;
   - otherwise the affine prior is used.
   - Group count, OOF prior regret, and mode stability are **not** computed
     or executed on this path.

   Archived `v1_declared` still contains the unused OOF/group/mode ladder for
   historical audits (DS03, FEMTO). Call
   `select_ppx_route(..., version="v1_declared")` to replay it. Do not describe
   that ladder as the Final 9-setting gate.
2. **Executor approval** (uses **frozen** thresholds; not retuned per dataset)
   - compare admissible candidates on identical group-disjoint validation
     folds;
   - an optional executor must reduce validation loss by at least **2%** relative
     to the simpler admissible reference;
   - unit-win fraction at least **60%** and worst-unit RMSE ratio at most
     **1.10** (physical-unit risk);
   - `dual_scale` additionally requires train-only support heterogeneity
     at least 0.50;
   - ties within numerical tolerance select the simpler executor.
3. **Safety**
   - if prior admissibility fails, use the prespecified fallback;
   - no test input except the individual sample's causal features may alter
     the selected route.

### Threshold origin (tuning once, then freeze)

The triple `(2%, 60%, 1.10)` is **not** re-tuned on each new dataset and is
**not** chosen by maximizing test accuracy.

It is an operational freeze point selected under
`protocols/PPX_THRESHOLD_TUNING_OOF_PROTOCOL.md`:

1. declare the candidate grid
   `{1,2,5%} × {50,60,70%} × {1.05,1.10,1.20}`;
2. score each cell by validation-unit OOF route stability and worst-case
   policy regret (development units only);
3. freeze `τ=(2%, 60%, 1.10)` inside the stable region;
4. on every subsequent dataset, keep `τ` fixed and use that dataset's
   validation only to approve/reject executors.

Evidence: `PPX_OOF_THRESHOLD_ROBUSTNESS_RESULTS_KO.md`,
`PPX_THRESHOLD_SENSITIVITY_RESULTS_KO.md`.


## Primary evaluation

- pooled R² and RMSE;
- physical-unit RMSE;
- equal-dataset log RMSE ratio;
- hierarchical dataset–unit bootstrap;
- paired physical-unit sign-flip tests with Benjamini–Hochberg correction;
- route coverage, false accept, false reject, and fallback rate.

Rows are never treated as independent inferential samples.

## Required comparisons

1. prior-only;
2. matched direct model;
3. common prior-residual core;
4. always-on optional executor;
5. validation-approved PP-X;
6. test oracle, clearly labelled as a nondeployable upper bound;
7. strong same-split competitors under the same information and tuning budget.

## Evidence tiers

- **Retrospective mechanism:** existing development datasets and module
  ablations.
- **Retrospective policy audit:** frozen replay on already-opened datasets;
  cannot establish prospective validity.
- **Prospective confirmation:** a protocol commit precedes endpoint opening,
  with method, candidate set, thresholds, budget, and success criteria frozen.

No retrospective result may be relabelled as prospective.

## Prospective success criteria

Report regardless of outcome:

- no test-informed route change;
- route selected exactly by this protocol;
- nonnegative pooled R²;
- PP-X RMSE no worse than the prespecified fallback;
- no physical-unit raw regret above the prespecified domain safety cap;
- route coverage and all abstentions.

Failure of any performance criterion is a failed confirmation, not a reason to
retune the frozen method.
