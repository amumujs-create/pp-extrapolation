# Auto-Regime Weak-Prior PP-X v1.4 개발 결과

작성자: 박진서  
상태: 이미 본 Stanford·ISU retrospective 구조 개발이며 신규 확증이 아니다.

## 구조

v1.4는 데이터 유형과 레짐을 두 단계로 다룬다.

1. 상위 contract layer는 target·boundary·causal-history 정의로 prior family를
   제한한다. 현재 prototype은 direct battery continuation family다.
2. 하위 regime layer는 train covariate만 robust scaling한 뒤 deterministic
   k-means로 3개 레짐을 찾는다.
3. validation/test row는 train-frozen 거리로 soft assignment한다.
4. 각 레짐은 고정 prior bank `trust=.02,.05,.10,.20,.40` 중 하나를
   physical-unit OOF stability 기준으로 고른다.
5. prior residual alpha는 최대 .25이므로 baseline weight는 최소 75%다.
6. 레짐별 route를 soft blend하고 전체 raw mean/CVaR/max unit regret를 다시
   검사한다. 실패하면 모든 alpha를 함께 줄이거나 exact baseline으로 복귀한다.

자동 레짐 탐지는 통계적 운전 상태를 찾는 것이며 물리식을 자동으로 새로
발명하는 기능은 아니다. 물리 prior family는 사전 등록된 bank 안에서만 고른다.

## Stanford

- baseline/v1.3 R²: 0.0714, RMSE 81.444
- fixed RBPR R²: 0.0730
- auto-regime v1.4 R²: 0.0717, RMSE 81.433
- test raw regret:
  - mean -0.040%
  - CVaR -0.001%
  - maximum -0.0004%
- 9/9 physical unit에서 baseline보다 수치적으로 개선

train에서 찾은 세 레짐의 test row 점유는 2373/21/33이었다. 큰 레짐 두 개는
prior를 거부했고, 작은 레짐 하나만 trust .40과 alpha .25를 채택했다.
따라서 fixed RBPR보다 성능은 낮지만 손상 없이 필요한 구간에만 약한 residual을
허용했다.

## ISU 250 mAh

- baseline/v1.3 R²: 0.4798, RMSE 1.671
- fixed RBPR R²: 0.4866, RMSE 1.660
- continuous portfolio R²: 0.4917, RMSE 1.652
- auto-regime v1.4 R²: **0.4897**, RMSE **1.655**
- 29/46 physical unit 개선
- test raw regret:
  - mean -1.61%
  - CVaR +1.96%
  - maximum +3.65%

두 active 레짐이 서로 다른 prior를 자동 선택했다.

- 레짐 A: trust .40, alpha .25
- 레짐 B: trust .05, alpha .25
- 데이터가 없는 레짐 C: baseline

명목 trust가 큰 레짐도 최종 residual의 25%만 사용하므로 실제 실행은
baseline-dominant weak prior다. validation 전체 raw mean/CVaR/max regret도
-1.75%/+1.67%/+4.10%로 사전 cap 2%/5%/10% 안에 들었다.

## 구조 안정성

레짐 수는 결과 전에 K=3으로 고정했다. 사후 민감도는 다음과 같다.

- ISU K=2/3/4 R²: 0.4903 / 0.4897 / 0.4896
- ISU CVaR: 1.95% / 1.96% / 1.99%
- Stanford K=2: exact baseline
- Stanford K=3/4: R² 0.0717 / 0.0717

ISU 결과는 K 변화에 거의 민감하지 않았다. hard assignment의 ISU R²는
0.4894로 soft assignment 0.4897보다 약간 낮았고 위험은 유사했다. soft
전환이 레짐 경계의 불연속을 줄이면서 성능도 보존했다.

## 판정

v1.4는 두 retrospective 고호트에서 다음을 동시에 달성한 첫 prototype이다.

1. Stanford 실패 구간에서 baseline 비악화
2. ISU에서 fixed RBPR보다 성능 개선
3. ISU raw CVaR와 maximum regret를 사전 cap 안에 유지
4. 레짐별 서로 다른 prior 자동 선택
5. 빈 레짐과 근거 부족 레짐의 baseline fallback

따라서 **v1.4를 다음 미개봉 direct-continuation cohort용 개발 후보로 승격**한다.
다만 현재 성능은 기존 test를 본 retrospective 결과이며 확증이 아니다.

전체 데이터 유형까지 자동화하려면 v1.4 위에 CRPE contract layer를 붙여
boundary quotient, causal latent, regime transport prior bank를 구분해야 한다.
그 family 분류는 outcome score가 아니라 입력 schema와 target 정의로
결정해야 한다.

## 재현

```bash
python experiments/auto_regime_weak_prior_v14_development.py
PYTHONPATH=src pytest -q tests/test_regime_prior_bank.py
```

결과:
`results/auto_regime_weak_prior_v14_development/results.json`
