# 재분석 및 두 번째 개발 라운드

2026-09-13. 이전 실패 이후 모델 구조와 학습 목표를 변경해 재실험했다. **DS03에서 검증으로 채택된 보정의 개선은 확인했지만, Engression 초과·전체 외삽 커버리지·방법론적 노벨티 조건은 충족하지 못했다. PP-X 기본 경로는 그대로 유지한다.**

## 1. 먼저 무엇을 다시 분석했나

이전 `relation_local_transport`에는 다음 설계 불일치가 있었다.

1. 승인 경로는 `(y - prior) / margin`을 전달하면서, 거리와 기하의 학습 목표는 raw y였다. 수명값이 비슷한 유닛이 프라이어 오류까지 유사하다는 보장은 없다.
2. prior-off에서는 기존 NN을 국소회귀로 대체했다. 그 결과 기존 모델이 잡은 비선형 관계를 잃을 수 있다. 이는 코드상 구조 차이이며, 성능 저하의 유일한 원인으로 확정한 것은 아니다.
3. DS03의 학습 변경은 3 seed 모두 validation에서 epoch 0에 밀렸다. 관계 손실을 넣어도 실제 선택된 모델은 바뀌지 않았다.
4. DS03 TRAIN 유닛 1–6을 확인하면 RUL 대 cycle 기울기는 모두 수치 오차 범위에서 -1이었다. 각각의 종료 cycle은 72, 73, 67, 60, 93, 63이다. 따라서 기존 인접 변화량 손실은 거의 알려진 countdown 관계를 반복한다. 중요한 유닛 간 차이는 종료 시점이다. 이 관측은 TRAIN만으로 계산했다.

## 2. 실제로 바꾼 모델: Cross-fitted Component Transfer

기존 전체 PP-X 예측 b(x)를 유지한다. prior-on이면 승인된 전체 PP-X, prior-off이면 기존 direct fallback이 b다. prior-off 경로에 거절된 물리 프라이어를 다시 넣지 않는다.

각 donor의 오차를 국소 ridge 회귀로 다음 세 성분으로 나눈다.

- 수준 성분 c₀: 가까운 donor들의 평균 오차.
- 진행 방향 기울기 c₁: margin 또는 cycle 방향에서 예측되는 오차 변화.
- 나머지 방향 기울기 c₂: 나머지 입력 특성 방향의 오차 변화.

최종 예측은 `b(x) + a₀(z)c₀(x) + a₁(z)c₁(x) + a₂(z)c₂(x)`다. 각 a는 0~1이며, 선택이 거절되면 세 값 모두 정확히 0이다. 승인된 경계가 있으면 residual quotient로 세 성분을 만들고 query margin을 곱해 경계에서 보정이 정확히 0이 되게 한다.

z는 세 성분의 값, donor까지 거리, 유효 이웃 비율, donor 잔차 표준편차, 진행 좌표 차이, 기존 예측 크기의 8개 관측 가능 값이다. query 정답은 입력에 들어가지 않는다. 조건부 gate는 8→12→3 Tanh/Sigmoid 네트워크이며 147개 파라미터다. 국소회귀의 bandwidth와 ridge는 고정해, raw-y 거리 학습과 residual 보정의 불일치를 제거했다.

### 학습 데이터 구성의 변화

TRAIN 유닛을 고정 3-fold로 나누고 60%, 80% 진행 절단점을 만든다. **query 유닛 전체와 query보다 뒤쪽 진행 구간을 제외한 데이터로 base 모델부터 다시 적합한다.** 해당 base가 donor에서 만든 잔차로 세 성분을 구성한 후, 보지 않은 query 유닛의 뒤쪽 구간에서 gate를 학습한다.

- MICH는 낮은 margin 방향 외삽. query MICH 유닛을 base 학습에서 제외하며, 공동 battery 학습 행도 같은 정규화 margin 절단점으로 제한한다. base는 고정 200 epoch, query/외부 validation으로 checkpoint를 고르지 않는다.
- DS03는 높은 cycle 방향의 source 가상 외삽. base의 checkpoint는 donor TRAIN 자체에서만 선택하고 query 유닛을 사용하지 않는다.
- case당 3 seed × 3 folds × 2 cutoffs = 18개 source base를 적합한다.
- donor 잔차는 donor를 학습한 base의 in-sample 잔차이고 query 잔차는 out-of-fold이다. 이는 배포 시 TRAIN donor memory / 새 query 구성을 모사하지만, donor 잔차와 query 오차의 크기 차이까지 제거하는 것은 아니다.

목표는 source 유닛/절단점별 기존 오류 대비 상대 MSE와 보정 크기 규제다. primary에는 유닛별 상대 위험이 1보다 커지는 경우의 제곱 벌점 0.25도 추가한다. 보정 0인 모델을 checkpoint 후보에 포함한다.

고정 ablation은 (i) 보정 전체 적용, (ii) 전역 성분 gate, (iii) 입력별 성분 gate, (iv) 입력별 성분 gate+위험 벌점(primary)이다. 결과를 보고 primary를 바꾸지 않았다.

## 3. 검증으로 실제 채택된 DS03 개선

모두 seed 42/43/44 ensemble, 같은 기존 test 438행이다.

| 모델 | R² | RMSE | 최악 유닛 RMSE | positive-R² 유닛 |
|---|---:|---:|---:|---:|
| 기존 PP-X fallback | 0.879373 | 7.81132 | 11.13140 | 6/6 |
| Engression | 0.900449 | 7.09618 | 10.50889 | 6/6 |
| 잔차 전체 적용 | 0.860336 | 8.40510 | 12.95020 | 6/6 |
| 전역 성분 gate | 0.879373 | 7.81132 | 11.13140 | 6/6 |
| 입력별 성분 gate | 0.882925 | 7.69545 | 10.86424 | 6/6 |
| 입력별 성분 gate+위험 벌점(primary) | **0.882914** | **7.69579** | **10.86556** | **6/6** |

primary는 validation에서 기존 대비 유닛 균등 MSE 비율 0.965229, 유닛 승리 3/3, 최대 유닛 RMSE 비율 0.997714였다. 기존의 `2% 초과 개선 / 60% 이상 승리 / 개별 RMSE 증가 5% 이내` 기준을 바꾸지 않고 통과했다.

테스트 RMSE는 기존 fallback 대비 약 1.48%, 최악 유닛 RMSE는 약 2.39% 감소했다. 단순 잔차 전체 적용보다 안정적이며 전체 행에 예측을 제공한다.

그러나 중요한 제한이 있다.

- seed 42/44는 보정이 꺼졌고 seed 43만 활성화됐다. 세 seed 모두 독립적으로 좋아졌다는 뜻이 아니다.
- 위험 벌점을 빼도 비슷하며 오히려 아주 조금 좋다. 추가 위험 벌점의 효용은 입증하지 못했다.
- Engression보다 RMSE가 여전히 크다. 새 후속 연구의 전체 성공 조건은 불충족이다.
- DS03 test cycle 범위는 TRAIN과 동일한 [1,93]이다. 이 결과는 **prior-off의 새 유닛 일반화 개선**이지 엄밀한 test cycle-support 외삽 개선은 아니다.

## 4. 엄밀한 margin 외삽인 MICH에서는 아직 미채택

MICH test 202행은 모두 TRAIN의 margin 범위보다 아래이며 test 유닛도 다르다. 기존 full PP-X dual-scale과 동일한 canonical train+validation refit을 비교했다.

이번 MICH의 최종 donor bank는 TRAIN+validation으로 다시 구성하면서 bank normalization도 다시 계산한다. 선택된 gate 가중치와 gate context normalization은 고정한다. 이전 후보의 bank normalization 고정 방식과 차이가 있으므로 두 후보의 test 성적 차이를 교차적합 하나의 효과로 해석하지 않는다.

| 모델 | 정책 적용 전 R² | RMSE | 최악 유닛 RMSE |
|---|---:|---:|---:|
| 기존 full PP-X | 0.710033 | 8.63012 | 10.58407 |
| Engression | -3.492444 | 33.96908 | 81.09638 |
| 전역 성분 gate | 0.721728 | 8.45428 | 10.18904 |
| 입력별 성분 gate | 0.715881 | 8.54264 | 10.52478 |
| primary | 0.715892 | 8.54249 | 10.52518 |

primary의 validation 유닛 MSE 비율은 **이전 후보 1.06275에서 이번 0.95637로 개선**됐다. 하지만 유닛 승리가 3/6이고 최대 개별 유닛 RMSE 비율이 1.07448이어서 안전 조건을 통과하지 못한다. 최종 guarded 경로는 기존 full PP-X로 돌아간다. 테스트 원시 개선을 보고 gate를 완화하지 않았다.

MICH와 DS03 모두 개선을 보장하는 후속 모델을 확보했다고 표현할 수 없다. 더구나 MICH는 전역 gate가 조건부 gate보다 test에서 좋으므로 조건부 구조가 모든 외삽 상황에서 우월하다는 증거도 없다.

## 5. 종료 시점 / countdown 분리 대조 실험도 수행

DS03 TRAIN에서 확인한 공통 기울기 -1을 분리하고 신경망에는 종료 시점 `RUL + cycle`을 학습시켰다. clock feature는 NN 입력에서 제외하고 출력에만 선형으로 넣었다. unit consistency 0 / 0.1 두 고정 arm, 3 seed를 비교했다.

이 실험은 **알려진 관계를 분리하면 개선되는가**에 대한 대조군이지 노벨티 주장 모델이 아니다. 공통 감소 기울기와 affine 관계가 TRAIN에서 성립하지 않으면 사용을 거절한다.

| 대조 모델 | validation MSE 비율 | test R² | test RMSE | 최악 유닛 RMSE |
|---|---:|---:|---:|---:|
| clock 분리 | 0.918286 | 0.875836 | 7.92499 | 9.86619 |
| clock 분리+unit consistency | 0.891892 | 0.875758 | 7.92750 | 9.82817 |

둘 다 validation gate는 통과했지만 전체 test RMSE는 기존 fallback보다 나빴다. 최악 유닛 개선만으로 전체 성능 조건을 대체하지 않는다. **이 후보도 배포하지 않는다.** 이 결과는 validation gate가 분포 이동 뒤의 무손실을 보장하지 않는다는 실제 사례이기도 하다.

## 6. 노벨티에 대한 정직한 판정

교차적합 잔차 효용으로 기본 예측의 변경 여부를 학습하는 아이디어 자체는 이미 [Cross-Fitted Residual Utility for Primary-Preserving Cognitive Decision Correction](https://arxiv.org/abs/2608.02063)에서 다룬다. 또한 국소선형 회귀와 bias correction을 통한 전이도 [Transfer Learning and Locally Linear Regression for Locally Stationary Time Series](https://arxiv.org/abs/2511.12948)와 관련된다. 이 두 문헌의 초록을 확인했으며 전체 문헌에 대한 우선권 심사를 완료한 것은 아니다.

이번 후보의 구체적 검토 대상은 **유닛과 진행 구간을 동시에 제외한 오차로, 경계를 보존하는 세 보정 성분의 공유량을 따로 학습한다**는 규칙이다. 다만 DS03의 제한적 개선만으로 이 규칙의 독창성·필요성을 입증할 수 없다. 아직 component별 분리와 하나의 scalar gate를 직접 맞춘 ablation, source cross-fitting 자체를 제거한 matched control, 새 잠금 cohort 확인이 부족하다. 전역 성분 gate는 세 개의 독립 상수이므로 하나의 scalar gate와 같은 대조군이 아니다.

## 7. 이제 남은 병목과 다음 작업의 기준

이번 라운드로 “기존 NN을 버리고 raw-y 국소회귀로 바꾸기”보다 “기존 예측을 유지하고 보정의 입력별 유효성을 학습하기”가 DS03에서 더 유망하다는 제한적 증거를 얻었다. 하지만 바로 전체 벤치마크로 확대하거나 gate를 완화할 단계는 아니다.

다음 가설은 **보정값이 큰가가 아니라, source에서 해당 성분의 오류 감소 방향이 유닛/절단점에 걸쳐 일관적인가를 학습하는 것**이다. 이를 진행하려면 기존 conditional gate 대비 component-sharing만 다른 matched ablation과 TRAIN-only 정보로 만든 유효성 target이 필요하다. 잔차, 교차적합, MoE라는 이름만 추가하는 것으로는 충분하지 않다. 새 가설로 별도 기록하고, 현재 열린 test는 개발 자료로만 취급해야 한다.

## 8. 구현과 재현

- `src/pp_extrapolation/component_transfer.py`
- `experiments/component_transfer_screen.py`
- `src/pp_extrapolation/clock_factor_fallback.py`
- `experiments/clock_factor_fallback_screen.py`
- `experiments/verify_component_transfer.py`
- `tests/test_component_transfer.py`, `tests/test_clock_factor_fallback.py`
- 결과: `results/component_transfer_{mich,ds03}_v1/`, `results/clock_factor_fallback_ds03_v1/`

각 실행은 기존 결과를 덮어쓰지 않는다. MICH의 개별 batch와 원래 joint batch 사이 float32 반올림 차이(최대 1.49e-7) 때문에 초기 bitwise 비교가 중단됐다. 원래 joint batch로 예측하도록 수정하고 저장된 모델·validation 선택을 그대로 둔 채 reveal만 재개했다. 수정 전후 source hash와 이유는 각 결과 폴더의 `execution_fix.json`에 보존했다. 이 수정으로 모델이나 학습 hyperparameter를 바꾸지 않았다.

재검증은 source episode 36개, gate 모델 18개, clock 모델 6개와 각 validation/test 예측·metric·gate를 포함하며 모두 통과했다. 관련 단위/회귀 테스트 37개도 통과했다. 기존 PP-X 기본 모듈과 설정은 이번 라운드에서 수정하지 않았다.
