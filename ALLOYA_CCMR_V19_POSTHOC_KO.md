# Alloy A CCMR v1.9 사후 replay

## 결과

CCMR v1.9를 Alloy A의 기존 frozen split에 적용했다. Test는 v1.6
평가에서 이미 개봉됐으므로 이 결과는 확증이 아니라 사후 비교다.

- v1.9 route: `cautious_causal`
- stable certificate: 실패
- 이유: validation physical units가 4개로 최소 기준 10개 미충족
- validation causal coverage: 25%
- validation pooled RMSE 개선: 5.77%
- validation global approval: 통과
- test coverage: 30%
- test R²: 0.764333
- pooled/macro RMSE 개선: 1.33%/2.23%
- raw mean/CVaR/max regret: -7.87%/0%/0%

## v1.6과의 관계

v1.9의 최종 test 예측, active mask, truth, persistence를 v1.6 봉인
artifact와 직접 비교한 결과 최대 절대차가 모두 0이었다. 즉 Alloy A에서
v1.9는 cautious route를 통해 **v1.6과 정확히 같은 배포 결과**를 냈다.

따라서 순위도 바뀌지 않는다.

- Engression: R² 0.991385, 최대 regret -93.63%
- CCMR v1.9: R² 0.764333, 최대 regret 0%

Alloy A에서는 Engression이 평균 정확도와 unit 안전성 모두 우수하다.
CCMR v1.9로 버전을 올려도 이 고호트에서 최고 모델이 되지는 않는다.

재현 산출물:

- `experiments/alloya_ccmr_v19_posthoc.py`
- `results/alloya_ccmr_v19_posthoc/results.json`
- `results/alloya_ccmr_v19_posthoc/predictions.npz`
