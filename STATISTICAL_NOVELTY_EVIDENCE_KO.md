# PP 통계 근거와 강화된 노벨티

## 정정

PP에는 이미 상당한 비교·통계 근거가 있다. 동일 예산 또는 동일 입력 비교에는 Ridge,
plain MLP, ResNet, FT-Transformer, GRU, TCN, temporal Transformer, boosting, spline,
MoE와 제한된 TabPFN 비교가 포함된다. 대부분 5회 재학습의 평균·표준편차와 ensemble을
분리했고, MATR batch2에서는 test label 공개 전 protocol 및 unit-paired bootstrap도
수행했다. 따라서 비교실험이 부족하다는 표현보다 **흩어진 증거를 주 논지에 맞춰 통합하고
consensus의 임계값을 정당화해야 한다**는 표현이 정확하다.

## 임의의 5/5 규칙을 통계 규칙으로 변경

각 seed가 group-LOO validation에서 bounded affine을 선택했는지를 Bernoulli vote로 둔다.
귀무가설은 임의 seed가 affine을 선택할 확률이 0.5 이하라는 것이다. 단측 exact binomial
tail probability를 계산하고 `alpha=0.05`일 때만 보정을 승인한다.

5 seeds에서 가능한 값은 다음과 같다.

| affine votes | 단측 p-value | 승인 |
|---:|---:|---|
| 5/5 | 0.03125 | 승인 |
| 4/5 | 0.18750 | 거절 |

따라서 기존 unanimity는 임의의 경험적 숫자가 아니라, 현재 seed 수에서 exact test가
요구하는 최소 합의다. 이 검정은 최적화 초기화에 대한 **algorithmic stability test**이며,
seed를 독립적인 물리 실험으로 간주하는 과학적 유의성 검정은 아니다.

## 물리 unit 기준 paired bootstrap

승인된 데이터에서 calibration 전후 ensemble의 unit별 RMSE 차이를 20,000회 bootstrap했다.

| 데이터 | 개선 unit | 평균 unit RMSE 변화 | 95% bootstrap CI |
|---|---:|---:|---:|
| HUST | 13/16 | **-25.19** | **[-34.38, -14.54]** |
| RWTH | 8/8 | **-23.27** | **[-41.79, -8.94]** |
| MATR2019 | 10/10 | **-10.63** | **[-15.67, -6.37]** |

세 데이터 모두 CI가 0을 포함하지 않는다. 보정이 일부 긴 sequence의 행 수 때문에 pooled
점수만 좋아진 것이 아니라, 독립 평가 단위 수준에서도 일관된 개선이라는 근거다.

승인된 데이터셋은 3/3 모두 개선됐지만 데이터셋 단위 exact sign test는 `p=0.125`이다.
즉 unit 내부 근거는 강하지만 서로 다른 도메인 수준의 표본 수는 아직 작다. 이 결과를
과장하지 않고 추가 도메인을 요구하는 근거로 사용한다.

## 강화된 방법 정의

최종 방법을 단순 `PP + affine calibration`으로 부르지 않는다. 다음 세 단계의
**evidence-transported prior-residual network**로 정의한다.

1. **Shape learning:** frozen affine tail 위에서 nonlinear 또는 latent-regime residual이
   열화 곡선의 순서와 형태를 학습한다.
2. **Group transport:** 서로 다른 validation unit을 하나씩 제외해도 재현되는 scale/offset
   이동만 추정한다.
3. **Exact consensus test:** 초기화별 선택이 exact binomial 기준을 통과할 때만 보정을
   test로 운반하고, 실패하면 identity를 적용한다.

이 정의의 핵심은 calibration 계수 자체가 아니라 **보정 가능한 외삽 오류인지 검증한 뒤
source evidence만으로 correction의 운반을 승인하는 것**이다. Hybrid NN, calibration,
selective regression 각각은 기존 기술이지만, strict out-of-support RUL에서 이 책임 분리와
통계적 승인 절차를 하나의 모델 실행 규칙으로 만든 것이 현재의 가장 강한 노벨티다.

## 현재 증거 수준

- 알고리즘 안정성: 승인 데이터마다 seed vote 5/5, exact p=0.03125.
- unit 수준 효과: 세 승인 데이터 모두 paired-bootstrap CI가 0 아래.
- 안전성 감사: 7개 중 3개 개선, 4개 identity 유지, 악화 0개.
- 공정 대조: MATR에서 동일 calibration을 적용한 FT 0.377, PP 0.466.
- 독립 확증: MATR batch2에서 PP가 9/9 cells에서 Ridge보다 낮은 RMSE를 보였고 기존
  bootstrap CI도 0 아래였음.
- 한계 표본: 승인된 독립 데이터 유형이 세 개여서 domain-level 유의성은 아직 부족함.

기계 판독 결과는 `results/consensus_statistical_audit_v1/results.json`에 저장했다.
