# 논문 제출 전 코드 기반 재점검

판정: 유망한 결과가 있으나 현재는 제출 준비 완료 상태가 아니다. 기존 점수와 프로토콜을 보존하고 아래 주장 및 통제를 보완해야 한다. 이 문서는 이전의 저널 등급·모델 노벨티에 관한 과도한 단정을 정정한다.

## 실제 구현과 주장 정합성

- latent PP와 Jacobian spline PP는 서로 다른 실험군이다. `fit_latent_regime_pp`에는 Jacobian loss나 applicability certificate가 없다. 0.257은 이들을 모두 결합한 시스템의 결과가 아니다.
- gate는 context 고정 시 q에 대해 증가한다. 실제 궤적에서는 context도 변하므로 gate가 감소할 수 있다. irreversible transition, 추정된 물리 change-point, 보정된 전환 확률을 주장할 수 없다.
- affine 동결은 기준 경로를 제공하지만 안전성 보장은 아니다. latent residual에 hinge가 곱해져 외삽 거리와 함께 커질 수 있으며 최종 출력은 clipping된다. raw 결과와 clipping 비율이 필요하다.
- gate regularizer는 (mean(g)-0.5)^2 + mean(g(1-g)) = 0.25 - Var(g)이다. 모든 상수 gate가 같은 penalty를 가진다. 분산을 장려하지만 balanced usage 또는 collapse 방지를 보장하지 않는다.
- q만 바꾼 future ray는 용량 평균·변화율 등을 고정하므로 실제 가능한 궤적이라는 보장이 없다. 인과적 반사실이 아닌 합성 입력 방향 정규화로 기술한다.

## 평가 범위와 확증 한계

- MATR 2019는 unit-disjoint capacity-tail 평가다. 현재 시점의 관측 capacity를 입력으로 받으므로, 고정된 과거 prefix만으로 미관측 미래 궤적 전체를 예측한 실험과 다르다. HUST의 seen-unit late/deep-future 실험과도 분할·입력이 다르다.
- 1차 capacity 좌표가 train 범위 밖이면 이를 포함한 전체 입력 hull 밖임도 성립한다. 그러나 보고된 distance 자체는 1차 capacity 거리이며 전체 feature hull 거리로 부르면 안 된다.
- target은 마지막 기록 cycle까지 남은 cycle이다. 실제 80% EOL/종료 기준과 일치하는지 cell별 확인 전에는 모두 물리적 RUL이라고 단정하지 않는다.
- loader가 test capacity와 cycle 전체를 처음부터 메모리에 읽는다. 학습 루프가 test label을 사용한 증거는 없지만 물리적으로 봉인된 test label이라는 표현은 부정확하다.
- 코드에서 train boundary 최소값 행을 다시 제거하고, validation/test는 <= boundary를 사용한다. 프로토콜의 train 유지 및 strict <와 정확히 일치하지 않는다. 영향 행 수 감사가 필요하다. 원결과를 덮어쓰지 않고 deviation으로 보고한다.
- pooled 0.257과 사전 비교 기준 통과는 보존한다. plain NN 대비 paired unit RMSE CI [-27.61,15.25]는 모집단 우월성을 확정하지 못한다. unit-macro -0.511도 함께 보고한다.

## 제출 전 우선순위

1. 위 정의·구현·프로토콜 차이와 종료 기준을 감사하고 Methods 및 기존 서술을 수정한다.
2. 공정한 탐색 예산 비교: 현재 latent는 seed당 9개, Jacobian은 4개, plain은 1개 설정이다. validation에서 같은 탐색 횟수/계산 예산으로 선택한 NN 대조가 필요하다. 파라미터 수는 6입력 latent 614, plain 1313으로 latent가 더 작지만 탐색 예산 차이는 남는다.
3. 핵심 구성요소 통제: 동일 expert 하나만 사용, fixed gate, unconstrained gate, affine 제거 대조를 개발 데이터에서 수행한다. 현재 PP→Jacobian→latent는 여러 요소가 함께 변하므로 gate의 독립 효과를 식별하는 ablation이 아니다.
4. 재현성: row-level prediction, raw/clipped prediction, unit/row ID, 모델 checkpoint, 모든 validation 후보 점수, 실행 환경·시간을 저장한다. 현재 확증 JSON만으로는 원예측·오차 궤적을 재구성할 수 없다. 고정 재실행이 필요하면 원결과와 수치 일치를 검사한 재현 실험으로 명시한다.
5. 이후 외부 cohort 한 개 추가 검증을 권장한다. cell-level gate/certificate는 개발/validation에서만 선택하며 MATR test로 설계한 규칙은 확증으로 부르지 않는다.

## 논문 가치

모델링 신규성은 아직 가설이다. 기존 MoE/잔차/단조 gate 대비 어떤 특성이 외삽 이득을 만드는지 통제로 입증해야 한다. 특정 상위 저널에 충분하다거나 확률적 gate만 넣으면 가능하다는 이전 평가는 근거가 부족했다. 적용 범위를 명확히 제한한 논문 후보이며 공정한 baseline과 핵심 ablation 이후 투고 수준을 판단하는 것이 타당하다. 확률적 change-point 추가는 제출 필수조건이 아니다.
