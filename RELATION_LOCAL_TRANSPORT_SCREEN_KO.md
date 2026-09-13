# 프라이어 보정 / prior-off fallback 공통 연산자: 구현 및 1차 판정

## 결론

2026-09-13. 구현과 두 데이터셋의 고정 비교를 완료했다. **후속 모델로 채택하지 않는다. 기존 PP-X 기본 경로는 변경하지 않았다.**

- MICH: 후보 원시 예측은 기존 full PP-X보다 좋지만, 검증 단계에서 후보 채택이 거절된다. 테스트 점수를 보고 이 결정을 뒤집지 않는다.
- DS03: 새 fallback은 검증과 테스트에서 전체 오차가 악화된다.
- 관계 손실의 추가 기여도 확인되지 않는다. 이 구현을 모델링 노벨티 확보로 표현하지 않는다.
- 이번 검증은 2개 설정에 한정되며 기존 전체 벤치마크 커버리지 우위를 의미하지 않는다.

## 구현한 하나의 알고리즘

가칭 Relation-trained Local Transport. TRAIN 유닛들로 만든 국소 선형 연산자 Sθ(x; v)를 두 경로에서 공통 사용한다.

1. 프라이어 승인, 알려진 수명 경계 m(x)가 있는 경우:

   y_hat(x) = p(x) + m(x) Sθ(x; (y_i - p(x_i)) / m(x_i)).

2. 프라이어 거절:

   y_hat(x) = Sθ(x; y_i).

프라이어 거절 경로는 프라이어 값을 읽지도 않는다. 승인 경로는 경계에서 p(x)=0이면 보정도 정확히 0이다. 출력 비음수 제약과 DS03의 기존 cap을 유지한다.

Sθ는 학습된 대각 거리와 bandwidth로 이웃 가중치를 계산하고, 가중 국소 ridge 회귀의 수준과 기울기를 함께 사용한다. θ는 거리 계수 d개, bandwidth 1개, ridge 1개다. 고정 기하에서는 bank 값 v에 선형인 연산자이며 최종 clipping 이후까지 선형이라는 뜻은 아니다.

TRAIN 내부에서 서로 다른 유닛을 anchor/query로 나누고, 진행 좌표의 40/60/80% 절단점 너머 query를 예측한다. MICH는 낮은 health margin 방향, DS03는 높은 cycle 방향이다. 각 episode의 anchor와 query는 유닛 및 진행 구간이 겹치지 않는다. 전체 TRAIN으로 normalization을 계산하므로 이 episode 성적은 독립적인 nested 평가가 아니다.

학습 목적은 정규화 점 예측 MSE + 0.1 × 인접 query 간 변화량 오차 + 거리 정규화다. 기하는 raw TRAIN y로 학습하고, 실제 연산자 bank만 승인 여부에 따라 y 또는 residual quotient로 바꾼다. 따라서 이 구현은 **프라이어의 성분별 전달 오류를 식별하는 완전한 모델이 아니다.** 잠재 health 동역학·레짐별 프라이어 분포·인과적 반사실 식별도 구현한 것으로 주장하지 않는다.

## 평가 규칙

- 고정 4개 arm: kernel mean / 고정 local linear / 학습 local linear(점 손실) / 학습 local linear(점+관계 손실).
- primary는 마지막 arm으로 사전 고정. 테스트로 arm을 선택하지 않는다.
- seed 42/43/44. 고정 모델은 세 번 동일하므로 독립 반복의 안정성 증거로 해석하지 않는다.
- validation에서 epoch 선택. 최종 ensemble 후보 채택 조건: 유닛 균등 MSE 2% 초과 개선, 60% 이상 유닛 승리, 각 유닛 RMSE 증가 5% 이내. 탈락 시 기존 PP-X 예측을 그대로 반환한다.
- MICH: full PP-X dual-scale을 원래 공동 3-battery TRAIN/validation과 설정으로 재학습하고, 선택된 epoch만큼 TRAIN+validation으로 재초기화 refit. 보정 기하는 MICH TRAIN에서 학습, 최종 bank는 MICH TRAIN+validation으로 확장하되 기하와 normalization은 고정한다. 기존 full PP-X archive 예측과 재현 일치를 확인했다.
- DS03: 고정된 prior-off PP-X direct fallback 및 equal-budget archive의 Engression 중 첫 3 seed를 비교. 양쪽 학습 데이터와 동일한 TRAIN 유닛 1–6만 새 bank로 사용한다. validation 7–9, test 10–15.
- selection 파일을 저장한 뒤 이 실행에서 test를 로드한다. 하지만 이 test들은 과거 이미 평가됐으므로 **새로운 prospective 증거는 아니다.**
- baseline 탐색 예산 및 목적 함수가 달라 동일 계산량 비교를 주장하지 않는다.
- 커버리지는 positive-R² 유닛/seed 수와 유한 예측 비율이다. 예측구간 coverage가 아니다.

## 결과: 정책 적용 전 원시 후보

RMSE와 최악 유닛 RMSE는 원래 수명 단위다. 아래 수치는 3-seed ensemble이다.

| 데이터 | 모델 | R² | RMSE | 최악 유닛 RMSE | R²>0 유닛 |
|---|---|---:|---:|---:|---:|
| MICH | full PP-X | 0.710033 | 8.63012 | 10.58407 | 8/8 |
| MICH | Engression | -3.492444 | 33.96908 | 81.09638 | 3/8 |
| MICH | 고정 local linear | 0.709741 | 8.63447 | 10.32594 | 8/8 |
| MICH | 학습, 점 손실 | 0.745423 | 8.08634 | 9.46543 | 8/8 |
| MICH | 학습, 점+관계 손실(primary) | 0.745207 | 8.08977 | 9.48710 | 8/8 |
| DS03 | 기존 PP-X fallback | 0.879373 | 7.81132 | 11.13140 | 6/6 |
| DS03 | Engression | 0.900449 | 7.09618 | 10.50889 | 6/6 |
| DS03 | 고정 local linear | 0.863711 | 8.30293 | 12.69170 | 6/6 |
| DS03 | 학습, 점 손실 | 0.863711 | 8.30293 | 12.69170 | 6/6 |
| DS03 | 학습, 점+관계 손실(primary) | 0.863711 | 8.30293 | 12.69170 | 6/6 |

MICH primary RMSE는 원시 결과에서 약 6.3% 개선된다. 그러나 아래 검증 기준을 만족하지 않으므로 채택된 모델의 개선으로 보고하지 않는다. 점 손실보다 관계 손실 결과가 미세하게 나빠 추가 학습 규칙의 효용도 입증되지 않는다.

DS03의 학습 arm은 3 seed 모두 validation에서 epoch 0이 선택됐다. 즉 업데이트는 실행됐지만 학습된 변경이 채택되지 않았고, 최종 모델은 고정 local linear와 같다. 학습 효과가 있었다고 표현하지 않는다.

## 검증 게이트와 외삽 범위

| 데이터 | 후보/기존 validation 유닛 MSE | 승리 유닛 비율 | 최대 유닛 RMSE 비율 | 채택 |
|---|---:|---:|---:|---|
| MICH | 1.06275 | 50% | 1.25031 | 거절 |
| DS03 | 1.69483 | 0% | 1.33896 | 거절 |

두 경우 모두 guarded primary는 기존 PP-X와 **완전히 동일한 예측**이다. 손해를 피했을 뿐 개선한 것은 아니다.

- MICH test 202개 행은 100% TRAIN의 health-margin 범위보다 아래다. TRAIN margin [0.37444, 1.06054], test [0.00224, 0.36996]. 유닛도 분리되어 있다.
- DS03 test 438개 행은 유닛이 분리되지만 TRAIN과 test의 cycle 범위가 모두 [1,93]이다. cycle 범위 밖 비율 0%. 따라서 여기서 DS03를 엄밀한 cycle-support 외삽 성능으로 부르지 않는다. 이 결과는 기존 prior-off fallback의 새 유닛 일반화 검사다. 다른 feature 방향의 OOD 여부는 이 단일 좌표 감사로 판정하지 않는다.

## 다음 설계에 남는 가설

관측된 것은 MICH에서 학습된 기하의 원시 test 성적이 개선됐다는 것뿐이다. 프라이어 성분별 공유 규칙의 발견이나 반사실 강건성은 아니다. 후속 연구를 계속한다면 다음을 별도 가설로 정하고, 이미 확인한 test 점수로 게이트를 완화하지 않아야 한다.

1. 승인 경로에서는 raw y의 유사성 대신 **source-unit cross-fit 프라이어의 잔차 전달 가능성**을 직접 학습한다. query 유닛으로 학습한 프라이어가 그 query의 supervision 생성에 참여하지 않게 한다.
2. prior-off에서는 level과 slope의 유닛 간 전이를 분리한다. DS03 epoch 0 선택은 현재 source episode 학습이 validation 개선으로 이어지지 않았다는 증거이며, 그 원인 자체를 확정한 것은 아니다.
3. 위 성분별 전이 구조가 평범한 metric learning/local linear baseline보다 나은지를 새로운 잠금 분할에서 ablation한다. 새로운 기여의 정의와 문헌 검증은 별도로 필요하다.

이번 후보를 단순히 더 튜닝해 테스트 점수를 높이는 방식으로 채택하지 않는다.

## 구현 및 재현

- 모듈: `src/pp_extrapolation/relation_local_transport.py`
- 실행: `experiments/relation_local_transport_screen.py --case mich` 또는 `--case ds03`
- 검증: `experiments/verify_relation_local_transport_screen.py`
- 결과: `results/relation_local_transport_{mich,ds03}_v1/`
- 기존 결과 폴더가 있으면 실행기는 덮어쓰지 않고 중단한다.
- 24개 후보 checkpoint 재실행, 20개 metric matrix 재계산, baseline replay, validation gate 재계산, source hash 검증 통과.
- 관련 테스트 26개 통과: 국소 연산자·평가 runner·이전 integral 모듈 회귀 검사.
- 초기 실행에서 MICH scale key의 대소문자 불일치를 수정했다. 첫 실행 출력은 각각 `_initial` 폴더에 보존했고, 수정 후 양쪽을 재실행했다. 모델 구조나 학습 hyperparameter는 결과를 보고 변경하지 않았다.

실행 중 PP-X 기본 설정이나 기존 결과 파일을 덮어쓰지 않았다.
