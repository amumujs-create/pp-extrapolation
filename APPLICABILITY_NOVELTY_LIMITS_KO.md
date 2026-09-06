# PP가 되는 경우와 안 되는 경우: 적용성, 노벨티, 한계

## 먼저 답할 수 있는 것과 없는 것

새 test의 정답 없이 미래 R²를 정확히 알 수는 없다. PP가 해야 하는 일은 성공을
보증하는 것이 아니라, 예측 전에 관측 가능한 증거로 적용 가능성을 검사하고 증거가
부족하면 `ABSTAIN`하는 것이다. 따라서 최종 출력은 숫자 하나가 아니라

`prediction + applicability certificate + extrapolation distance + uncertainty/disagreement`

이어야 한다.

## Test label 없이 확인할 네 단계

1. **역할·메커니즘 coverage**
   train에 존재하지 않는 material, 운전조건, 고장 mode 또는 센서 의미가 source에
   나타나면 거절한다. 현재 categorical certificate가 이 검사를 구현한다. NASA
   Milling의 material 1→2 실패는 이 단계에서 정답 없이 탐지된다.
2. **내부 pseudo-tail skill**
   train unit의 더 이른 구간으로 학습하고, 별도 validation unit의 hull 밖 tail에서
   Ridge·일반 NN·PP를 비교한다. PP의 validation R²가 양수이고 기준 모델보다 정해진
   margin 이상 좋아야 승인한다. 이 threshold는 새 test를 보기 전에 고정해야 한다.
3. **거리와 경로 안정성**
   train convex hull 밖 거리, seed 간 분산, affine tail과 NN correction의 차이를 함께
   낸다. 거리가 calibration 범위를 넘거나 seed/disagreement가 임계값보다 크면
   숫자를 내지 않는다. hull 밖이라는 사실만으로 실패가 결정되지는 않는다.
4. **regime-change 감시**
   최근 causal window에서 rate/잔차의 level 또는 slope가 validation에서 관찰하지 못한
   방식으로 바뀌면 기존 prior route를 중단한다. 이는 정답을 사용하지 않는 입력 기반
   감시이며, 구현 후 별도 calibration이 필요하다.

## 현재 사례가 보여 주는 것

| 사례 | 결과 | 사전에 알 수 있는 신호 | 해석 |
|---|---|---|---|
| Sunwoda | PP 0.862, Ridge 0.844 | 큰 hull 거리; 온도 조건 이동 | PP는 성공했지만 엄격한 unseen-condition gate는 보수적으로 abstain할 수 있음 |
| RWTH | PP 0.506, Ridge 0.419 | 동일한 측정 역할과 unit-disjoint pseudo-tail 검증 | 현재 가장 깨끗한 적용 성공 사례 |
| MICH | PP -1.522 | hull 밖이며 late-tail 관계가 학습 affine 경로로 보존되지 않음 | coverage만으로는 부족하며 skill/stability gate가 필요 |
| NASA Milling | PP -4.826 | train material 1, source material 2 | categorical mechanism gate가 실패를 사전 탐지 |

Sunwoda는 중요한 반례다. 강한 coverage gate는 위험을 줄이지만 성공 가능한 조건 이동도
거절한다. MICH는 반대 반례다. 같은 데이터셋과 feature 의미라는 이유만으로 승인해도
성공하지 않는다. 그러므로 coverage는 **필요 조건 후보**이지 충분 조건이나 정확도
보증이 아니다.

## 논문의 중심 노벨티 후보

강한 주장은 단순한 `affine + residual NN` 구조가 아니다. 그 조합만으로는 기존 hybrid
model 또는 residual learning과 구별하기 어렵다. 논문의 중심은 다음과 같은
**선택적 외삽 시스템**으로 잡는다.

1. 도메인마다 사용 가능한 단서를 typed prior contract로 표현한다.
2. contract가 있는 모듈만 조립하고, 없는 모듈은 식을 임의로 만들지 않는다.
3. affine tail과 NN correction의 신뢰도를 support와 validation evidence로 정한다.
4. label-free mechanism coverage와 calibrated gate를 통과한 경우에만 예측한다.
5. 거절까지 포함해 coverage-risk와 selective R²/regret로 평가한다.

이를 논문 용어로는 **contract-compiled selective neural extrapolation** 또는
**verified modular prior routing for RUL extrapolation**으로 정리할 수 있다. 이 명칭의
새로움 자체를 주장하는 것이 아니라, compiler·executor·certificate·abstention을 하나의
고정 protocol로 만들고 실험으로 각 부분의 기여를 증명하는 것이 노벨티 후보다.

## 반드시 구분할 주장

- 현재 증거: PP-unseen 데이터에서 성공과 실패가 모두 관측됐고, 두 데이터에서는
  Ridge보다 높은 pooled R²와 낮은 seed 변동을 보였다.
- 아직 없는 증거: 임의의 새 도메인에서 성공한다는 보편 일반화, 실패를 항상 사전에
  검출한다는 보증, 물리적 인과관계의 식별.
- 최종적으로 필요한 증거: untouched dataset에서 gate를 완전히 고정한 prospective
  평가, 기존 OOD/uncertainty 방법과 동일 coverage 비교, module ablation, 실패 비용을
  포함한 risk-coverage curve.

## 현재 구현의 한계

- hull 검사는 현재 주로 health 축을 사용하므로 다변량 support 전체를 대표하지 못한다.
- R²가 작은 unit에서는 unit-macro R²가 매우 불안정할 수 있어 pooled R²·RMSE와 함께
  해석해야 한다.
- validation pseudo-tail이 실제 먼 test tail보다 가깝다. 가까운 외삽 성능이 먼 외삽
  성능을 보장하지 않는다.
- 데이터셋 수가 아직 작고 battery 비중이 높아 domain diversity가 제한된다.
- RWTH·MICH·Sunwoda는 PP에는 처음이지만 PAE 연구에서 관찰됐으므로 완전한 연구 수준의
  untouched cohort는 아니다.
- 현재 certificate는 범주 coverage 중심이다. 연속 조건 shift와 새로운 degradation
  regime을 검출하는 calibrated monitor는 아직 구현·검증되지 않았다.

따라서 현재 가장 정직하고 강한 결론은 “PP가 항상 외삽한다”가 아니라,
**PP는 일부 PP-unseen tail에서 안정적인 이득을 보이며, 성공 조건을 명시적으로 검사하고
검사 실패 시 거절하도록 확장할 수 있는 모듈형 외삽 프레임워크**라는 것이다.
