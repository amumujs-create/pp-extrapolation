# 추가 실데이터 외삽 결과

고정 구현·프로토콜 `c9a7eda`로 Sunwoda, RWTH, MICH를 한 번 평가했다. 모두
처음 보는 unit의 late-health 구간이며, validation과 test 표본의 train health
convex-hull 밖 비율은 100%다. 수치는 5 seeds 평균±표준편차의 raw pooled R²다.

| 데이터셋 | train/val/test windows | test units | Ridge | 일반 NN | PP | support-PP | 판정 |
|---|---:|---:|---:|---:|---:|---:|---|
| Sunwoda | 1,312/141/1,468 | 9 | 0.844±0.000 | -1.035±1.478 | **0.862±0.009** | 0.857±0.007 | PP 성공 |
| RWTH | 2,295/231/1,859 | 8 | 0.419±0.000 | -0.113±1.663 | **0.506±0.020** | **0.506±0.020** | PP가 안정적으로 Ridge 상회 |
| MICH | 1,439/259/202 | 8 | -1.522±0.000 | -1.116±1.053 | -1.522±0.000 | -1.522±0.000 | 실패, abstain 대상 |

## Unit-macro R²

| 데이터셋 | Ridge | 일반 NN | PP | support-PP |
|---|---:|---:|---:|---:|
| Sunwoda | 0.806±0.000 | -1.206±1.520 | **0.836±0.015** | 0.822±0.009 |
| RWTH | 0.439±0.000 | -0.197±1.786 | **0.518±0.018** | **0.518±0.018** |
| MICH | -1.981±0.000 | -1.497±1.052 | -1.981±0.000 | -1.981±0.000 |

## Hull 거리와 해석

| 데이터셋 | validation outside | test outside | test median distance | test max distance |
|---|---:|---:|---:|---:|
| Sunwoda | 100% | 100% | 4.421 | 6.788 |
| RWTH | 100% | 100% | 2.870 | 5.626 |
| MICH | 100% | 100% | 3.426 | 5.060 |

PP는 Sunwoda와 RWTH에서 Ridge를 각각 0.018, 0.086 R² 상회했고 일반 NN보다
seed 분산이 작았다. support 감쇠는 Sunwoda에서 기본 PP보다 약간 낮았고 RWTH와
MICH에서는 validation이 beta=0을 골랐다. 따라서 support 감쇠를 보편적 성능 향상으로
주장할 수 없다.

MICH의 모든 PP seed가 Ridge와 사실상 같다는 것은 validation checkpoint가 nonlinear
correction을 채택하지 않았다는 뜻이다. 이 실패를 숨기지 않고 mechanism coverage 또는
validation skill gate의 abstain 사례로 사용해야 한다. 세 데이터는 과거 PAE 개발에서
이미 관찰되었으므로 이 결과는 교차모델 재현이며, 최종 논문의 독립 검증은 별도 untouched
cohort가 필요하다.

PHM2010은 로컬에 실데이터가 아니라 합성 sample만 있어 이 표에 포함하지 않았다.
