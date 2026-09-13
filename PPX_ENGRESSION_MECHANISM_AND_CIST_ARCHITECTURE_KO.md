# Engression 메커니즘 벤치마킹과 CIST-PPX 독창 구조

## 1. 현재 외부 결과가 말하는 것

Engression은 Axial-fan 세 설정에서 모두 가장 높았지만 MATWI에서는 음의 R2가
가장 낮았고, 전체 외부 seed 변동도 PP-X 계열보다 컸다. 따라서 장점은 보편적인
외삽 안전성이 아니라 **endpoint와 학습 support가 비교적 가까운 곳에서 유연하고
매끄러운 조건부 평균을 만드는 능력**이다. PP-X는 먼 tail과 반복 seed 안정성에
강하지만, 고정 affine prior와 bounded tanh residual이 endpoint censoring 이동에서
평균함수를 과도하게 제한했다.

## 2. 로컬 구현에서 확인한 Engression의 실제 장점

1. 입력과 출력을 각각 평균과 표준편차로 중심화한다. PP-X의 max-target scaling보다
   출력 0과 상단 clipping 경계에 덜 끌린다.
2. noise가 출력 scale에만 붙지 않고 각 stochastic hidden layer에 주입된다.
   비선형성 뒤에서 평균을 취하므로 `E[f(x, epsilon)]`이 deterministic network와
   다른 유연한 평균함수가 된다.
3. beta-energy loss는 beta 1에서 절대오차형 data-fit과 sample 간 diversity를
   동시에 최적화한다. 일부 unit의 큰 제곱오차가 gradient를 독점하기 어렵다.
4. ReLU와 BatchNorm 기반의 piecewise-linear representation은 tanh residual보다
   경계 밖에서 덜 빨리 포화된다.
5. 한 모델에서 100개 response sample을 평균하고 다시 5개 학습 seed를 평균한다.
   stochastic training의 흔들림이 point prediction에서 상당 부분 상쇄된다.

## 3. 그대로 가져오면 안 되는 부분

- generic hidden noise와 energy loss를 그대로 붙이면 Engression 재구현에 가깝고
  PP-X의 구조적 기여가 약해진다.
- support 밖 residual authority, 물리 unit 균형, 단조 RUL 진행, affine prior 거절,
  worst-unit 위험 제약이 없다.
- 실제로 MICH, MATR2019, MATWI에서 큰 음의 R2가 발생했고 외부 seed SD도 컸다.
- 그러므로 energy loss를 복제하는 대신 stochasticity가 작용하는 위치를
  degradation mechanism에 맞춰 제한해야 한다.

## 4. CIST-PPX

**Censoring-aligned Innovation Slope Transport PP-X**는 response를 직접 생성하지
않고, 진행좌표를 따라 누적되는 손상률의 분포를 생성한다.

`RUL(x, p, epsilon) = gate(x) * [anchor(context) - integral_0^p rate(context, t, epsilon) dt] + (1-gate(x)) * direct(x)`

- `anchor`: 관측 tail 시작 경계의 RUL 기준점;
- `rate`: 양의 softplus stochastic damage rate;
- `integral`: 고정 Gauss midpoint로 계산하는 확률 slope transport;
- `direct`: 구조 가정이 맞지 않을 때 사용하는 deterministic residual route;
- `gate`: flow prior를 관측별로 거절할 수 있는 하나의 내부 gate;
- `support decay`: 진행좌표가 train support 밖으로 나갈수록 direct route authority를
  줄여 무제한 neural 외삽을 막는다.

Engression처럼 stochastic samples와 proper energy objective의 장점은 유지하지만,
noise가 임의 hidden representation이 아니라 해석 가능한 damage-rate field에만
들어간다. 이 차이가 핵심 구조 노벨티다.

## 5. Censoring alignment

Axial test는 fan당 한 개의 truncated endpoint만 갖는다. 기존 validation은 한 fan의
여러 tail row를 평균해 선택했으므로 test task와 달랐다. CIST 선택에서는 validation
unit당 마지막 pseudo-endpoint 하나만 사용한다. test label이나 test RUL 분포는 쓰지
않고 관측 형식만 일치시킨다. MATWI와 Misata는 원래 test가 여러 tail row이므로 기존
validation 형식을 유지한다.

## 6. 승격 기준

- 외부 다섯 설정 각각에서 official Engression pooled R2 초과;
- 양수 coverage 감소 금지;
- seed 평균, 최저 seed, seed SD 공개;
- 실패 설정 삭제 및 외부 설정별 사후 route 변경 금지.

모든 현재 outcome은 이미 열려 있으므로 이 기준을 통과하더라도 개발 성공일 뿐이다.
일반적 외부 우월성에는 새 untouched cohort가 필요하다.
