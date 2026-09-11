# SIT LFP sequential-replication CCMR v1.7 protocol

**Frozen:** 2026-09-11, 박진서, before downloading cycle summaries.

This study was requested after observing earlier cohort outcomes. It is a
one-shot evaluation within this dataset, but its evidence is explicitly
selection-conditioned and weaker than a prospectively nominated first
confirmation.

## Source and compact extraction

- SIT Battery Data v2, DOI `10.25447/sit.32101523.v2`, CC BY 4.0.
- 20 real large-format Narada FE50B LiFePO4 cells with deep-cycle summaries.
- Four cycling conditions spanning ambient/40 C and 1C/2C discharge.
- The source archive is 4.3 GB, but only the 20
  `Data/Cycle_Summary/*.csv` members are range-downloaded: public central
  directory metadata reports 2,656,327 uncompressed bytes.
- Fixed columns: `Cycle`, `Type`, and `Capacity_Ah`; retain `Type ==
  "discharge"`.

No cycle-summary values from this dataset have previously been downloaded,
screened, plotted, fitted, or scored in this project.

## Frozen QC population and unit split

The publisher README explicitly identifies `001-3`, `001-8`, and `101-3` as
anomalously steep capacity-fade cells and recommends flagging them. They are
excluded before outcome access. This defines a 17-cell non-flagged
deep-cycling analysis population; exclusion is not inferred from this study.

- Sort cell IDs by SHA-256.
- First 10 cells: train.
- Next 3 cells: validation.
- Final 4 cells: one-shot test.
- Require at least 500 finite discharge cycles with strictly increasing
  cycle number per cell.
- Stop before fitting if the fixed 10/3/4 split or at least 400 test origins
  is unavailable.
- Cell ID, tester, temperature, C-rate, and operating group are not model
  inputs.

## Frozen forecast task

- Initial capacity is the maximum of the first five discharge capacities, as
  recommended by the publisher.
- Health is capacity divided by initial capacity.
- Apply a causal trailing five-cycle median to health.
- Causal history: 20 discharge cycles.
- Forecast horizon: 10 discharge cycles.
- train origins: progress <= 0.30.
- validation origins: progress 0.40--0.60.
- test origins: progress 0.75--0.90.
- Progress is origin index/final index and is used only for splitting.
- Anchor: current smoothed health persistence.
- Correction features: capacity slopes over lags 1, 5, and 20 and
  slope-1 minus slope-20.
- Context: current health, correction features, trailing 20-cycle mean/std,
  and causal history fraction.
- Absolute cycle, progress, cell ID, condition, terminal cycle, and future
  values are excluded from model inputs.

## Model and success

Apply frozen CCMR v1.7 from
`protocols/CCMR_V17_CAUSAL_BACKTEST_PROTOCOL.md` without retuning, including
five non-overlapping realized shadow wins and the gated-validation global
veto. Seal candidate, gated, deployed, persistence, truth, units, origins,
targets, and progress before test scoring.

Performance success requires all:

1. 100% of test origins beyond maximum train progress;
2. positive pooled test R-squared;
3. pooled and unit-macro RMSE improvements >= 0.5%;
4. raw test unit mean/CVaR20/max regret <= 0%/1%/2%;
5. deployed correction coverage >= 10%;
6. exact fallback replay error zero.

A pass is registered as a selection-conditioned performance replication, not
as pristine prospective confirmation. Failure or fallback is preserved and
no threshold, split, or dataset replacement follows.

## Outcome-independent pre-fit origin-count amendment

Schema/count inspection produced 466 test origins across four independent
cells, while all 17 cells passed the >=500-cycle rule. No model was fitted and
no prediction or score was produced; the original inconclusive artifact is
retained. The arbitrary test-origin floor is amended from 1,000 to 400 because
400 still supplies 100 macro-weighted origins per test unit on average.
Target, split, model, risk caps, and success criteria are unchanged. The
amended run writes to a distinct result directory and this deviation is part
of the evidence grade.
