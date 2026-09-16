# PP-X Paper Method v4 — Hierarchical Validation Router

Owner: 박진서  
Status: retrospective operational freeze; prospective confirmation pending

Canonical selector:
`ppx_forward_selector.select_min_validation_loss` and
`ppx_forward_selector.select_replicate_validation_min`.

## Scope

PP-X predicts unseen physical units or regimes without test-time adaptation.
Test labels, test-batch statistics, registry outcomes, and dataset identity are
forbidden selector inputs.

## Inputs available before test

1. Typed source contract and train-derived structure.
2. Group-disjoint validation targets and physical-unit identifiers.
3. Validation predictions from every contract-computable PP-X candidate.
4. Seed/fold-local validation losses for candidate internal configurations.

## Candidate opening

The contract determines computability, not the winner.

- BQ may be opened when boundary and progression quantities are computable.
- Affine prior may be opened when state/rate representation is computable.
- User-supplied ``group_key``, ``time_key``, and ``regime_key`` take
  precedence. Blank group keys use standard names, then an unambiguous
  repeated-ID/contiguous-block top-level column audit; ambiguity requires user
  input. Blank time/regime keys
  trigger train-only detection. Time is accepted from a semantic name
  (``cycles/time/progress/coordinate``), or from exactly one monotonic numeric
  top-level column; zero or multiple candidates disable history. Regime is
  inferred from discrete, unit-stable or within-unit-varying train signals.
  Feature position, including ``x[:, 0]``, is never interpreted as time.
- Transport candidates require the corresponding regime/context variables.
- Dual-scale candidates require train-computable support information.

BQ is optional. If both BQ and affine are computable, both may enter the same
validation tournament.

## Frozen decision order

1. Fit all contract-computable prior–executor candidates without test access.
   A known boundary opens BQ but does not close affine or force BQ.
2. Within every admissible prior family, identify its minimum-validation-loss
   executor. The resulting loss is that prior family's score.
3. Select the prior family with minimum family score:

   \[
   \hat p=\arg\min_p\min_{e\in\mathcal E_p}\mathrm{ValMSE}(p,e).
   \]

4. Within the chosen prior/executor family, select each seed/fold's internal configuration:

   \[
   \hat h_r=\arg\min_h \mathrm{ValMSE}_{r,h}.
   \]

5. Freeze all choices.
6. Produce one test prediction per selected replicate and ensemble them.
7. Attach dataset names and test metrics for reporting only after selection.

Finite-loss ties use candidate-label lexical order. Candidate labels identify
model arms, not datasets.

## Direct comparison

Matched direct is a required control but is not a PP-X route candidate for the
paper figure. Consequently, this protocol does not abstain to direct. Any
direct-safe router result must be reported as a distinct policy and must not
replace the PP-X Final figure values.

## Reproduction acceptance

The reproduction passes only if:

1. every recorded selection equals validation argmin;
2. no selector API receives dataset identity or test arrays;
3. all nine test scores round to the frozen figure values;
4. the output explicitly remains retrospective.

Expected rounded pooled R²:

```text
HUST       0.958
Sunwoda    0.939
N-CMAPSS   0.937
Virkler    0.888
RWTH       0.878
MATR-b2    0.862
MICH       0.751
NASA       0.584
MATR2019   0.466
```

Reproduce:

```bash
PYTHONPATH=src:experiments:../ca-css-ncmapss \
  python experiments/ppx_final_result_reproduction_v1.py
```

Artifacts:

- `results/ppx_final_result_reproduction_v1/results.json`
- `PPX_FINAL_RESULT_REPRODUCTION_KO.md`

## Evidence boundary

This protocol reconstructs an already-developed nine-setting result. It is not
prospective evidence. A new confirmation must freeze candidate menus, split,
budget, selection scope, and endpoint before opening the untouched test labels.
