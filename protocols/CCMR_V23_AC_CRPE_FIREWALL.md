# CCMR v2.3 AC-CRPE 개발 방화벽

작성자: 박진서  
동결 시각: 2026-09-11  
모형: CCMR v2.3 Anchor-Competitive Contract-Conditioned
Risk-Budgeted Prior Experts (AC-CRPE)

## 개발 자료

구조 선택과 승격 판정에는 다음 다섯 trajectory development domain만 쓴다.

- Concrete material shift
- LG M50T
- SIT LFP
- RADAR NMC
- Luminosity temperature shift

각 domain의 train으로 모형을 적합하고 validation으로 domain 내부 anchor와
배포 route를 선택한다. 구조 수준 설정은 leave-one-domain-out 방식으로
나머지 네 domain의 동일 가중 점수만 사용한다.

## 금지 자료

다음 두 고호트는 source와 development 결과가 동결될 때까지 로드하지 않는다.

- Alloy A
- MultiStage RPT

두 고호트의 기존 점수, label, prediction, validation ranking은 구조,
하이퍼파라미터, risk cap, expert 허용 규칙 선택에 사용하지 않는다. 과거에
관측된 고호트이므로 동결 후 replay도 신규 독립 확증으로 표현하지 않는다.

기존 12-domain final PP test artifact는 row alignment와 사후 호환성 감사에만
사용하며 v2.3의 설정 선택으로 역류시키지 않는다.

## 사전 고정 목표

Primary objective는 다섯 domain을 동일 가중한 test RMSE ratio의 기하평균이다.
승격은 다음 조건을 모두 만족해야 한다.

1. CCMR v2.2 대비 domain-equal geometric-mean RMSE가 엄격히 감소한다.
2. 다섯 domain 중 최소 세 domain에서 v2.2보다 비악화한다.
3. 승인한 correction의 false accept가 0이다.
4. persistence 대비 raw physical-unit maximum regret가 모든 domain에서 2% 이하이다.
5. correction이 거부된 행은 selected strong anchor와 정확히 동일하다.

하나라도 실패하면 v2.3은 rejected development model로 기록하고 보류 replay를
실행하지 않는다.

## 고정 anchor 및 prior 후보

No-prior anchor bank는 persistence, weighted ridge, linear-tail RBF,
MLP ensemble, Engression ensemble로 제한한다. 설치 또는 수치 실패가 난
후보는 명시적으로 unavailable 처리하며 다른 후보의 평가를 막지 않는다.

Prior residual은 data contract가 허용한 causal dynamics expert만 사용한다.
dataset 이름이나 test 결과를 router 입력으로 쓰지 않는다. raw mean,
worst-20% CVaR, maximum unit regret 중 하나라도 validation cap을 넘으면
strong anchor를 정확히 복원한다.
