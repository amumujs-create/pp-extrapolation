# PP 노벨티 강화 로드맵

## 핵심 방향

PP의 노벨티를 `affine + NN`, hard boundary, PINN, 또는 다중 데이터셋의
조합으로 설명하지 않는다. 핵심은 **외삽 경로와 학습 보정의 예측 권한을
구조적으로 분리하는 것**이다.

Boundary-Quotient PP를 예로 들면

\[
\hat y=m\,\operatorname{softplus}(\ell(z)+c_\theta(z)),\qquad
\ell(z)=a^\top z+b,\qquad |c_\theta(z)|\le B.
\]

`ℓ`은 train에서 먼저 적합한 뒤 동결하고, NN은 bounded correction `cθ`만
학습한다. `m`은 EOL health margin이다.

## 이론적으로 바로 성립하는 성질

affine-only 예측을

\[
\hat y_A=m\,\operatorname{softplus}(\ell(z))
\]

라 하면 softplus가 1-Lipschitz이므로 모든 입력에서

\[
\boxed{|\hat y-\hat y_A|\le mB}
\]

가 성립한다. 따라서

1. `m=0`이면 어떤 OOD 입력에서도 `ŷ=0`이다.
2. EOL에 가까워질수록 NN이 affine 예측을 바꿀 수 있는 절대 폭이
   `mB`로 선형 수축한다.
3. 깊은 late tail에서 무제한 neural correction이 폭발하는 것을 구조적으로
   막는다.
4. affine을 동결하므로 이 envelope의 중심은 supervised residual 학습 중에
   움직이지 않는다.

이를 **prior-centered boundary contraction**이라 부른다. `distance × NN`이라는
일반 hard-boundary 식이 아니라, 사전 적합된 외삽 경로 주변에서 neural
자유도가 장애 경계로 갈수록 수축된다는 점을 주장한다.

코드의 `BoundaryQuotientPPNet.components`가 affine score, bounded correction,
positive quotient를 분리해 반환하고, unit test가 `|ŷ-ŷA|≤mB`와 margin 반감 시
correction 반감을 검사한다.

## 실험으로 더 증명해야 할 것

### 1순위: matched architecture control

같은 history representation, width, parameter budget, optimizer, validation rule로 다음을
비교한다.

1. direct NN
2. boundary penalty만 loss에 넣은 soft-boundary NN
3. `margin × fully trainable NN`
4. frozen affine만 사용한 boundary quotient
5. frozen affine + unbounded residual
6. **frozen affine + bounded residual BQ-PP**

이 실험이 없으면 심사자는 성능 이득을 encoder, 다른 전처리, 또는
boundary gate 효과로 해석할 수 있다.

### 2순위: contraction 기전 검증

- health-margin shell별 `|ŷ-ŷA|`와 theoretical envelope `mB`를 보고한다.
- late shell로 갈수록 direct/unbounded NN의 error는 커지고 BQ-PP의 correction은
  수축하는지 검사한다.
- 동일 평균 R²뿐 아니라 worst-shell RMSE, calibration, seed SD를 함께 보고한다.

### 3순위: relationship-shift 평가축

데이터셋 이름별 평균을 넘어, validation→test에서 health–RUL quotient가
얼마나 변했는지 정량화한다.

- conditional quotient shift
- local slope/curvature shift
- validation-to-test residual transport error
- hull distance와 quotient shift의 분리

이를 통해 `멀리 나가서 실패`와 `health–RUL law가 바뀌어서 실패`를
구분한다.

### 4순위: 고정 후 cohort 검증

현재 Sunwoda·RWTH·MICH는 개발 결과다. matched control과 selection rule을 고정한
뒤 새 battery cohort 또는 아직 사용하지 않은 unit/condition에 한 번 적용한다.
이는 노벨티를 만드는 실험이 아니라, 제안한 노벨티가 사후 설명이
아님을 입증하는 실험이다.

## 투고 전 판정 기준

PP를 상위 저널용 방법론으로 강하게 주장하려면 최소한 다음이
필요하다.

- matched 6-arm에서 BQ-PP가 dataset-macro 정확도와 worst-shell 안정성 중
  적어도 하나에서 유의한 이득을 보일 것
- frozen/unbounded 제거군이 외삽 shell에서 악화될 것
- 새 cohort에서 현재 선택 규칙이 유지될 것
- 실패 unit과 거절 조건을 함께 보고할 것

이 중 matched control이 가장 급하다. 새 Transformer나 데이터셋을 먼저
추가하는 것보다 논문 노벨티를 더 직접적으로 강화한다.
