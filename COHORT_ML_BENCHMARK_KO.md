# 주요 고호트 ML 사후 비교

동일 split과 동일 causal context를 사용했다. Ridge, random forest,
ExtraTrees, histogram gradient boosting, MLP의 설정은 validation에서
선택했다. Test가 이미 개봉된 뒤 수행한 분석이므로 확증 증거가 아니라
모형 구조 비교다.

## 핵심 결과

- 미래 health를 직접 예측한 일반 ML은 5개 고호트에서 모두
  persistence보다 크게 나빴다.
- `현재 health + 예측 residual` 구조로 바꾸면 평균 성능은 대폭
  개선됐다.
- Alloy A에서는 residual linear가 pooled RMSE 53.37%, macro RMSE
  57.89%를 개선하고 최대 regret도 -72.31%로 CCMR보다 좋았다.
- Concrete에서는 residual MLP가 pooled RMSE를 12.20% 개선했지만
  최대 unit regret가 732.94%였다. CCMR의 전부 거부가 훨씬 안전했다.
- SIT LFP에서는 residual RF가 29.00% 개선했지만 최대 regret가
  35.75%였다. Residual ExtraTrees는 23.18% 개선과 -7.35% 최대
  regret로 이 열린 test에서는 가장 균형이 좋았다.
- RADAR NMC에서는 residual histogram boosting이 40.97% 개선하고
  최대 regret -2.69%를 기록했다. 열린 test 기준으로는 CCMR raw
  candidate의 1.43%보다 우수했다.
- MultiStage RPT에서는 residual MLP가 57.35% 개선했지만 최대
  regret가 53.80%였다. CCMR은 개선 13.01%로 작지만 최대 regret
  0%를 유지했다.

## 모형적 해석

결과는 backbone 자체보다 **anchor-residual 구조**가 외삽 성능을
결정한다는 점을 보여준다. 일반 residual ML은 평균 정확도에서 CCMR을
앞설 수 있지만 고호트에 따라 일부 unit을 크게 손상한다. CCMR의
차별점은 최고 평균점수가 아니라 cross-fit consensus, context support,
exact fallback, validation risk routing을 결합해 손상을 제한하는 데
있다.

원자료:

- `experiments/cohort_ml_benchmark.py`
- `results/cohort_ml_benchmark/results_v2.json`
