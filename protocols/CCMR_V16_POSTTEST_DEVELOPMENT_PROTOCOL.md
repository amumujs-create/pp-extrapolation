# CCMR v1.6 post-test development protocol

**Frozen:** 2026-09-11, 박진서.

NASA B0049--B0052의 independent failure는 slope-only support가 위험한
correction을 식별하지 못한다는 개발 동기로만 사용한다. 해당 test의
수치에 v1.6을 다시 맞추거나 v1.6 확증 결과로 재사용하지 않는다.

## Structure

CCMR is a Persistence-Anchored Cross-fit Consensus--Minimax Residual.

1. Fit an equal-unit ridge residual on ordered-axis-invariant local slopes.
2. Refit it while leaving out each train physical unit.
3. Permit a row only when at least 90% of the fits agree on correction sign
   and fold MAD / correction magnitude is at most 0.5.
4. Fit a robust context representation excluding absolute age/progress.
   Reject rows beyond the 99th percentile of unlabeled validation
   nearest-prototype distance.
5. Select deployment mass on `[0, 0.25]`. Validation raw-unit mean, CVaR20,
   and maximum regret must be at most 0%, 1%, and 2%, respectively.
6. If no positive mass improves validation macro-MSE, use exact persistence.

## Development and ablation

Use the already opened LED Data_ID_2/5/6 strong-extrapolation splits only.
Compare full CCMR, no cross-fit consensus, and minimax-only. This is post-test
architecture development, not independent evidence.

Freeze all settings above before opening the next external cohort.
