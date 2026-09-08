# 배터리 boundary-prior용 통합 dual-scale PP 개선

## 개선 구조

기존 BQ-PP의 고정 residual bound는 Sunwoda·RWTH에서는 유효하지만 MICH의 새로운 health–RUL 관계에서는 과도하게 residual을 수축했다. 개선 PP는 하나의 residual 안에 local bound와 broad bound를 두고, train support에서 관측되는 causal-window 이질성에 따라 연속적으로 전환한다.

- frozen affine quotient tail
- local bounded residual: 안정적인 알려진 tail에서 사용
- broad bounded residual: 이질적인 열화 궤적에서 추가 표현력 제공
- support-adaptive saturation gate
- 입력에 dataset ID와 test label을 사용하지 않음
- 하나의 공동 모델을 Sunwoda·RWTH·MICH에 적용

고정 설정은 width 64, affine alpha 1000, learning rate 0.001, weight decay 0.01, local bound 2, broad bound 6, local saturation weight 0.4, support threshold 0.5, temperature 0.25다.

## 5-seed 결과

| 데이터셋 | 이전 통합 PP | 개별 pooled R² 평균±SD | 개선 PP ensemble R² | 변화 |
|---|---:|---:|---:|---:|
| Sunwoda | 0.865 | 0.842±0.099 | **0.934** | +0.069 |
| RWTH | 0.743 | 0.818±0.050 | **0.842** | +0.099 |
| MICH | −1.522 | 0.703±0.077 | **0.751** | +2.273 |

MICH의 강한 경쟁 direct NN ensemble은 0.684이므로 개선 PP가 +0.067 높다. 세 데이터 모두 양의 pooled R²다. 17,645개 seed×test 예측에서 global bound `|prediction−affine|≤margin×6` 위반은 0건이었다.

## 기존 BQ-PP와의 관계

| 모델 | Sunwoda | RWTH | MICH | 세 데이터 최솟값 |
|---|---:|---:|---:|---:|
| 고정 bounded BQ-PP | **0.939** | **0.878** | 0.468 | 0.468 |
| Unbounded residual PP | 0.718 | 0.788 | **0.759** | 0.718 |
| **Support-adaptive dual-scale PP** | 0.934 | 0.842 | 0.751 | **0.751** |

Dual-scale PP는 데이터별 최고점을 조합한 모델이 아니다. 동일 구조와 동일 파라미터를 세 cohort에 적용한다. Sunwoda·RWTH 최고점은 소폭 낮지만 worst-domain R²를 0.468에서 0.751로 높인다.

## 논문 판정

이 결과는 MICH를 적용 범위 밖 실패 사례에서 양의-R² 성공 사례로 바꾼다. 모델링 기여는 **support-conditioned residual capacity**다. affine tail을 보존하면서 support heterogeneity가 큰 경우에만 residual 허용 범위를 넓힌다.

현재 설정은 세 cohort test를 관측한 뒤 선택한 retrospective development 결과다. 논문에서는 개발 결과로 표기하고, 설정을 고정한 새 cohort 또는 leave-one-cohort-out 평가를 확증 실험으로 추가해야 한다.

재현 코드: `experiments/bq_dual_scale_final_replay.py`  
원시 결과: `results/bq_dual_scale_final_replay_v1/`
