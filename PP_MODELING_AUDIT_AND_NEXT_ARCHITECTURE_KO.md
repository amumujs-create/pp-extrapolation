# PP 모델링 점검과 다음 구조

## 현재 구조에서 유지할 부분

현재 Zn-ion 개발 모델은 다음 두 경로를 사용한다.

1. **Boundary quotient(BQ)**: `RUL = health margin × positive quotient`로 두어 알려진 고장 경계에서 0이 되는 귀납 편향을 준다.
2. **RBF lifetime memory**: 초기 prefix가 비슷한 development cell의 실제 EOL을 거리 가중 평균하고, `estimated EOL - current age`로 장수명 regime을 보완한다.

동일 EOL 정보를 사용한 대조에서도 RBF-only pooled R² 0.780, macro R² -2.611에 비해 결합 PP가 0.822, 0.375였다. 따라서 BQ 경로는 단순 memory 복사 이상의 역할을 한다. alpha=1000 개선 모델을 현재 결과표의 개발 모델로 유지한다.

## 구조적으로 남은 문제

### 1. lifetime 경로가 경계 조건을 위반한다

BQ는 margin=0에서 정확히 0이지만 RBF countdown은 실제 EOL에서 0일 필요가 없다. 두 경로를 섞은 최종 출력도 hard boundary를 보장하지 않는다. 단순히 `margin/(margin+tau)`를 곱인 후보는 development LOO에서 채택되지 않았다. 경계 조건은 사후 감쇠가 아니라 lifetime 경로 자체의 parameterization에 들어가야 한다.

### 2. 장수명 prototype 하나에 과도하게 의존한다

현재 memory에는 1010-cycle 장수명 prototype이 사실상 하나다. 해당 unit을 제외하면 장수명 fold가 무너진다. RBF bandwidth를 더 튜닝해도 없는 regime 정보를 만들 수 없다. prototype 수를 늘리거나, 개별 EOL lookup 대신 여러 unit에서 공유되는 lifetime-scale 함수를 학습해야 한다.

### 3. gate가 학습된 regime detector가 아니다

현재 gate는 초기 prefix slope에 대한 고정 sigmoid다. tail에서 새로 나타나는 knee, 곡률 변화, rate acceleration을 보고 갱신하지 않는다. 따라서 동일 prefix 뒤 서로 다른 전이를 구분할 수 없다.

### 4. BQ residual과 lifetime head가 따로 학습된다

현재 경로들은 독립적으로 맞춘 뒤 고정 규칙으로 혼합된다. 어느 경로의 오차가 큰지 학습 loss가 gate에 전달되지 않는다. 동시에, gate를 단순 end-to-end MSE로 학습하면 validation unit이 적을 때 한 expert로 붕괴할 가능성이 크다.

## 다음 모델: Dynamic Boundary-Scale PP

다음 구조는 세 항을 공동 학습하되 최종 경계 조건을 항상 보존한다.

\[
\hat R(t)=m(t)\,\operatorname{softplus}\{q_a(x_t)+
g_t q_s(h_t)+(1-g_t)q_r(h_t)\},
\]

- `m(t)`: 현재 health와 알려진 failure boundary 사이의 margin
- `q_a`: 안정적인 affine quotient
- `q_s`: 여러 development unit에서 공유해 학습하는 장기 lifetime-scale correction
- `q_r`: 현재 rate·curvature·knee evidence에 반응하는 local regime correction
- `g_t`: 현재 시점까지의 history만 사용하는 동적 gate

이 구조에서는 모든 경로가 margin 밖이 아니라 **quotient 안에서 결합**되므로 margin=0일 때 전체 예측이 정확히 0이다. 개별 EOL을 그대로 꺼내는 RBF lookup도 제거할 수 있다.

### gate 제약

gate에는 다음 제약을 함께 둔다.

- 시간 인접 consistency: 변화 증거가 없으면 `g_t`가 급변하지 않게 한다.
- 변화점 supervision: train unit에서만 계산한 slope/curvature change score를 보조 target으로 사용한다.
- entropy floor: 초기에 한 expert로 붕괴하지 않게 한다.
- support shrinkage: 학습 support에서 멀어질수록 neural correction 전체를 affine quotient 쪽으로 줄인다.

불확실성 head는 별도 ensemble 대신 unit-level out-of-fold residual을 target으로 학습한다. 이 값은 correction shrinkage와 prediction interval 폭에만 사용하고 test label로 보정하지 않는다.

## 채택 여부를 정할 실험

1. Development unit 전체를 하나씩 제외하는 nested LOO에서 hyperparameter와 epoch을 선택한다.
2. 모든 PP와 NN 대조군에 동일한 prefix, EOL supervision, inner validation을 제공한다.
3. `affine BQ`, `+ scale quotient`, `+ dynamic gate`, `+ support shrinkage` 순서로 ablation한다.
4. pooled R²와 unit-macro R²를 공동 기준으로 사용한다. 한 장수명 unit의 행 수가 pooled 점수를 지배하는지 항상 보고한다.
5. 이미 본 v2/v3는 개발 재생으로만 쓰고, 모델 동결 후 사전 표본 수를 만족하는 다음 cohort에서 한 번 평가한다.

## 현재 결론

즉시 유지할 모델은 alpha=1000 RBF-regime PP다. 단순 boundary attenuation은 검증에서 실패했으므로 추가하지 않는다. 다음 유효한 구조 개선은 **RBF EOL lookup을 boundary-consistent latent quotient로 바꾸고, 고정 slope gate를 causal dynamic regime gate로 바꾸는 것**이다. 다만 독립 장수명 development unit이 추가되지 않으면 이 구조의 자유도만 늘어나 과적합 위험이 커지므로, 새 구조의 확증 주장에는 더 많은 unit이 필요하다.
