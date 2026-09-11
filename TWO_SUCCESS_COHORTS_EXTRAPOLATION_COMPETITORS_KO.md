# 두 성공 고호트 외삽 경쟁모델 비교

작성자: 박진서  
대상: Alloy A, MultiStage RPT

## 비교 계약

그림의 비교군인 V-REx, GroupDRO, Monotone NN, Linear-tail RBF,
Engression, linear-mean GP, TabPFN v3를 모두 실행했다. PP는 새로
재학습하지 않고 각 고호트에서 실제 성공한 최신 봉인 배포 예측을 사용했다.

- Alloy A: CCMR v1.6
- MultiStage RPT: CCMR v1.9
- stochastic comparator: seeds 42--46 및 prediction ensemble
- hyperparameter 선택: validation only
- 출력 계약: 물리적으로 0 이상만 제한하고 train target 상한은 강제하지 않음
- test는 이미 열린 뒤의 비교이므로 신규 확증이 아닌 retrospective benchmark

기존 그림용 공통 코드는 감소형 health/RUL에 맞춰 train target 상한을
강제했다. 증가형 Alloy A에서는 이것이 모든 외삽 예측을 같은 상수로
잘랐으므로 해당 최초 실행은 폐기하고, 두 고호트에 공통으로 타당한
nonnegative-only 계약으로 전체를 다시 실행했다.

## Alloy A

Train/validation/test는 13/4/4 specimens, 25/12/10 origins다.

- Persistence: R² 0.757918, RMSE 0.102620, MAE 0.090000.
- CCMR v1.6: R² 0.764333, RMSE 0.101251, MAE 0.086365,
  pooled/macro 개선 1.33%/2.23%, raw regret mean/CVaR/max
  -7.87%/0%/0%.
- V-REx: R² -10.852997, RMSE 0.718067, pooled 개선 -599.73%,
  최대 regret 13,594.80%.
- GroupDRO: R² -10.830009, RMSE 0.717370, pooled 개선 -599.05%,
  최대 regret 13,424.40%.
- Monotone NN: R² -10.851528, RMSE 0.718022, pooled 개선 -599.69%,
  최대 regret 13,596.04%.
- Linear-tail RBF: R² 0.920013, RMSE 0.058988, pooled/macro 개선
  42.52%/42.12%, 최대 regret -51.85%.
- Engression: R² 0.991385, RMSE 0.019359, pooled/macro 개선
  81.13%/81.67%, 최대 regret -93.63%.
- Linear-mean GP: R² 0.952065, RMSE 0.045664, pooled/macro 개선
  55.50%/54.73%, 최대 regret -51.85%.
- TabPFN v3: R² -1.958540, RMSE 0.358748, pooled 개선 -249.59%,
  최대 regret 1,205.53%.

Alloy A에서는 Engression이 정확도와 모든 unit의 regret에서 CCMR보다
명확히 좋다. 따라서 이 고호트에서 최신 PP가 비교군 전체의 성능
승자라고 주장할 수 없다.

## MultiStage RPT

Train/validation/test는 43/14/15 cells, 110/50/33 origins다.

- Persistence: R² 0.972776, RMSE 0.006129, MAE 0.005594.
- CCMR v1.9: R² 0.979397, RMSE 0.005332, MAE 0.004615,
  pooled/macro 개선 13.01%/16.39%, raw regret mean/CVaR/max
  -35.30%/-5.05%/0%.
- V-REx: R² -8.801685, RMSE 0.116299, pooled 개선 -1,797.46%,
  최대 regret 88,535.67%.
- GroupDRO: R² -16.959516, RMSE 0.157424, pooled 개선 -2,468.44%,
  최대 regret 177,198.01%.
- Monotone NN: R² -7.512289, RMSE 0.108380, pooled 개선 -1,668.26%,
  최대 regret 81,534.35%.
- Linear-tail RBF: R² -5.069552, RMSE 0.091517, pooled 개선
  -1,393.14%, 최대 regret 64,207.44%.
- Engression: R² 0.979404, RMSE 0.005331, pooled/macro 개선
  13.02%/15.13%, 최대 regret 205.49%.
- Linear-mean GP: R² 0.962759, RMSE 0.007169, pooled 개선 -16.96%,
  최대 regret 183.74%.
- TabPFN v3: R² 0.630492, RMSE 0.022581, pooled 개선 -268.41%,
  최대 regret 5,041.57%.

Engression은 pooled R²와 RMSE에서 CCMR보다 수치상 극미하게 좋지만
차이는 사실상 동률이다. 반면 macro 개선은 CCMR 16.39% 대 Engression
15.13%이고, 최대 regret는 CCMR 0% 대 Engression 205.49%다.
따라서 MultiStage에서는 평균 정확도 동률권, unit 안전성은 CCMR의
명확한 우세다.

## 모형별 결론

- 평균 정확도 승자: Alloy A는 Engression, MultiStage는
  Engression과 CCMR의 동률권.
- 최악 unit 안전 승자: Alloy A는 Engression, MultiStage는 CCMR.
- V-REx, GroupDRO, Monotone NN은 이 두 소표본 강한 시간 외삽에서
  direct neural prediction이 크게 붕괴했다.
- TabPFN v3도 두 고호트 모두 persistence를 넘지 못했다.
- CCMR의 근거는 두 고호트 모두의 절대 최고 평균점수가 아니라,
  MultiStage와 같은 레짐에서 평균 개선과 max-regret 0%를 동시에
  달성하는 선택적 안전성이다.

## 재현 산출물

- 실행기:
  `experiments/two_success_cohorts_extrapolation_competitors.py`
- 전체 점수:
  `results/two_success_cohorts_extrapolation_competitors_v2_nonnegative/results.json`
- ensemble 예측:
  `results/two_success_cohorts_extrapolation_competitors_v2_nonnegative/*_ensemble_predictions.npz`
- 사용 패키지: Engression 0.1.15, TabPFN 8.5.0의 v3 model
