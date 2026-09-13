# RAVEN-X: 레짐 유효성·관계 개입 기반 확률적 외삽 모델 설계

## 1. 목표와 판정 범위

RAVEN-X(Regime-Aware Validity and Edge-intervention Network for Extrapolation)는 레짐 변화가 발생할 때 PP-X 계열의 외삽 성능이 약해지는 문제를 대상으로 한다. 목적은 과거의 어느 구간에서 추정한 관계가 미래의 어느 범위까지 유효한지 학습하고, 레짐 전환 시 물리적으로 연결된 관계만 변화시키며, 구분할 수 없는 전환은 하나의 예측으로 붕괴시키지 않는 것이다.

이 문서는 구현 명세다. 모델 성능, 인과 식별성 또는 학술적 신규성을 입증한 결과가 아니다. 관측자료만 있는 경우 생성 경로는 엄밀한 인과 반사실이 아니라 **물리 제약을 둔 반사실적 레짐 스트레스 시나리오**로 부른다. 실제 운전 조건 개입과 교란 통제가 있을 때만 `do`-개입 또는 인과 반사실이라는 표현을 사용한다.

## 2. 기존 후보와 다른 모델링 가설

기존 함수 프라이어 공유기는 현재 행 또는 prefix 요약으로 하나의 함수계수 분포를 만들었다. 그 관계가 어느 과거 구간에서 나왔고, 미래 어디까지 유지되는지, 유지되지 않으면 어떤 관계가 변하는지는 직접 모델링하지 않았다.

RAVEN-X의 검증 대상 가설은 다음 하나다.

> 숨겨진 뒤 구간의 외삽 오차를 최소화하도록 `관계의 근거 구간`, `관계의 미래 유효기간`, `전환 시 변화할 관계 모서리`를 공동 학습하면, 일반 시간 어텐션이나 전체 변수 노이즈보다 레짐 전환 외삽을 개선할 수 있다.

시간 어텐션, switching dynamical system, 관계 그래프, 반사실 생성 각각을 신규 기여로 주장하지 않는다. Temporal Fusion Transformer는 시계열의 장기 의존성과 변수 선택을 위한 시간 어텐션을 사용한다. Recurrent SLDS는 연속 상태에 의존하는 레짐 전환을 학습하며, switching nonlinear dynamical system은 비선형 동역학으로 시계열을 레짐별 분할한다. Graph Switching Dynamical Systems는 동적으로 변하는 관계 그래프와 모드 전환을 결합한다. 시계열 반사실 예측도 Causal Transformer 등의 기존 영역이다.

관련 1차 문헌:

- Temporal Fusion Transformer: https://arxiv.org/abs/1912.09363
- Recurrent Switching Linear Dynamical Systems: https://proceedings.mlr.press/v54/linderman17a.html
- Switching Nonlinear Dynamical Systems: https://proceedings.mlr.press/v119/dong20e.html
- Graph Switching Dynamical Systems: https://proceedings.mlr.press/v202/liu23z.html
- Causal Transformer: https://proceedings.mlr.press/v162/melnychuk22a.html

잠재적 기여는 이 구성요소들의 존재가 아니라 **외삽 전달 손실로 학습되는 구간-유효기간 어텐션과, 선택된 관계만 변화시키는 물리 함수 전환 생성 규칙**이다. 최종 신규성 판단에는 별도의 체계적 문헌조사가 필요하다.

## 3. 입력, 출력 및 적용 조건

### 입력

유닛 `u`의 현재 시점 `t`까지 사용 가능한 인과적 prefix:

`X_u,1:t = {margin, rate proxy, causal rolling features, operating covariates}`

MICH 현재 개발 자료에서는 기존 11개 인과 특징과 cycle 순서를 사용한다. 각 행의 rolling feature가 원자료의 미래를 포함하지 않는다는 기존 생성 검증을 그대로 요구한다. RUL 정답은 소스 TRAIN의 학습 손실에만 사용하고 신규 유닛의 입력에는 넣지 않는다.

### 출력

- 현재 레짐 사후확률 `q(z_t | X_1:t)`
- 전환 없음과 전환 위치의 분포 `q(h_tau, z_next | X_1:t)`
- 과거 근거 구간 어텐션
- 전환 시 활성화될 관계 모서리 확률
- 전환 전·후 함수계수의 결합분포
- 일관된 건강 경로 표본과 RUL 분포
- 레짐이 구분되지 않을 때의 혼합분포 및 경고 지표

### 적용 조건

최소 구현은 알려진 고장 건강 경계와 양의 열화 방향을 필요로 한다. 이 조건이 없는 DS03에는 cycle을 건강 변수로 가장하여 적용하지 않는다. DS03 대응은 별도 상태 좌표 또는 prior-off fallback을 개발해야 한다.

## 4. 전체 구조

```text
관측 prefix
   │
   ├─ 다중 길이 인과 구간 생성 ──→ 구간별 관계 토큰과 불확실성
   │                                      │
   │                                      ↓
   ├─ 현재 레짐 posterior ───────→ 구간 근거 어텐션
   │                                      │
   ├─ 전환 hazard / 다음 레짐 ──→ 미래 유효기간 어텐션
   │                                      │
   └─ 희소 관계 그래프 ─────────→ 관계 제한 전환 생성기
                                          │
                                          ↓
                         전환 없음 / 전환 경로 혼합
                                          │
                                          ↓
                         물리 일관 건강 경로 적분 → RUL 분포
```

## 5. 다중 구간 관계 토큰

### 5.1 후보 구간

유닛별 마지막 관측에서 끝나는 네 개의 상대 길이 구간을 사용한다.

- 최근 구간: 사용 가능한 prefix의 마지막 1/8
- 단기 구간: 마지막 1/4
- 중기 구간: 마지막 1/2
- 전체 prefix

각 구간은 최소 4개 유효 행이 있어야 하며 부족하면 mask한다. 절대 길이 8/16/32로 고정하면 유닛별 측정 밀도 차이가 길이 의미를 바꾸므로 1차 구현에서는 상대 길이를 사용한다. 이후 실제 시간 간격을 사용할 수 있을 때 duration을 별도 입력한다.

### 5.2 관계 토큰

구간 `w=[s,e]`에서 다음 토큰을 만든다.

`r_w = [현재값, robust 평균, IQR, 시간 기울기, 기울기 변화, log 길이, 함수계수 평균, 함수계수 불확실성]`

함수계수는 기존 양의 역속도 함수의 `theta=(a,b,c)`다. 신규 유닛에서는 RUL로 직접 적합하지 않고, 소스에서 학습한 prefix-to-coefficient encoder로 추정한다. 짧은 구간에서 계수를 확정하지 않고 평균과 공분산을 함께 출력한다.

### 5.3 구간 근거 어텐션

일반 softmax 대신 sparsemax 또는 entmax를 사용해 실제로 선택되지 않는 구간을 허용한다.

`alpha_w,k = entmax(score(r_w, regime k))`

레짐마다 참고 구간이 달라질 수 있다. 모든 구간이 불충분하면 어텐션을 강제로 균일화하지 않고 population prior로 후퇴한다.

## 6. 레짐과 유효기간

### 6.1 현재 레짐

최소 모델은 세 상태를 둔다.

- `z=0`: 관계 지속 또는 전환 증거 없음
- `z=1`: 역속도 수준 변화 중심
- `z=2`: 기울기·곡률 변화가 포함된 가속 전환

이는 실제 물리 레짐의 정답 라벨이 아니라 함수 변화 원형이다. 데이터가 지지하지 않으면 상태를 추가하지 않는다.

### 6.2 전환 위치와 지속시간

현재 건강 거리 `m_t`에서 고장 경계 방향의 미래 건강 위치 `h`를 사용한다. 전환 위치 `h_tau`는 `0 <= h_tau <= m_t`다. 전환까지의 이동은 `delta=m_t-h_tau`다.

명시적 duration hazard:

`lambda_k(delta | X_1:t) = softplus(g_k(context, delta))`

`S_k(delta) = exp(-integral_0^delta lambda_k(v) dv)`

`S_k`는 현재 레짐이 미래 거리 `delta`까지 유지될 확률이다. 전환 없음은 `h_tau=0`에 해당하는 별도 질량으로 둔다. 이 구조로 최근 관계가 가까운 미래에는 중요하지만 장기 외삽에서는 약해질 수 있다.

### 6.3 구간-미래 유효성 어텐션

최종 어텐션은 다음과 같다.

`W_w,k(delta) proportional alpha_w,k * S_w,k(delta)`

즉, 과거 구간 선택과 미래 유효기간을 분리한다. `alpha`가 높더라도 `S`가 빨리 감소하면 그 관계를 먼 미래까지 운반하지 않는다. 논문에서 시각화할 대상은 raw attention만이 아니라 `W(delta)`와 구간 제거 효과다.

## 7. 희소 관계 개입 어텐션

### 7.1 관계 그래프

노드는 두 종류다.

- 원인 후보 노드: 운전 조건, 관측 열화율, 변동성, 최근 변화량, 장기 추세
- 함수 노드: `a`, `b`, `c` 및 전환 hazard

모서리 `E_i->j`는 hard-concrete 또는 entmax gate로 추정한다.

`M_i,j ~ BernoulliRelaxed(A_i,j)`

금지할 모서리는 사전에 0으로 mask하고, 물리적으로 가능한 모서리만 학습한다. 단, 물리 지식이 방향을 확정하지 못하면 ‘가능 모서리’일 뿐 인과 모서리라고 부르지 않는다.

### 7.2 관계 제한 전환 생성기

전환 원형 `k`에 대해:

`delta_theta_j = sum_i M_i,j * T_k,i,j(delta_r_i) + sigma_k,j * epsilon_j`

`theta_after = theta_before + delta_theta`

선택되지 않은 관계 성분은 동일하게 유지한다. 독립적으로 모든 변수에 노이즈를 주지 않는다. 잔여 노이즈도 레짐별 공분산으로 묶어 관련 성분이 함께 변하도록 한다.

최소 버전의 구조적 mask:

| 전환 경로 | 변경 가능 성분 | 기본적으로 고정할 성분 |
|---|---|---|
| 지속 | 없음 | a, b, c |
| 수준 전환 | a, 제한된 b | c |
| 가속 전환 | b, c, 제한된 a | 없음 |

`제한된` 성분에는 작은 shrinkage prior를 적용한다. 위 표의 hard mask, 전체 성분 허용, 완전 학습 mask를 ablation으로 비교한다.

### 7.3 ‘반사실’의 정확한 정의

관측자료만 있을 때 생성하는 것은 다음 질문에 대한 스트레스 시나리오다.

> 현재까지 관측된 관계는 유지하면서, 학습된 전환 원형에 연결된 메커니즘만 달라졌다면 미래 수명 경로가 어떻게 달라지는가?

운전조건 변경과 같은 명시적 개입 자료가 있으면 해당 변수의 값을 고정하고 연결된 후손만 구조식을 통해 변경한다. 명시적 개입이 없으면 `do(X=x')` 표기를 사용하지 않는다.

## 8. 전환 후에도 일관된 함수 경로

현재 레짐 함수 `theta_0`, 전환 후 함수 `theta_1`, 전환 건강 위치 `h_tau`가 주어지면:

`RUL = integral_h_tau^m exp(phi(h)^T theta_0) dh + integral_0^h_tau exp(phi(h)^T theta_1) dh`

따라서 전환 전후의 누적 시간은 연결되고 RUL은 음수가 되지 않으며 고장 경계에서 정확히 0이다. 건강 상태 자체는 전환점에서 연속이고 역속도의 수준·기울기는 전환 원형에 따라 불연속일 수 있다. 실제 물리가 속도 연속성을 요구한다면 `theta_1`의 intercept를 경계 일치 조건으로 결정하는 ablation을 둔다.

경로 표본 하나는 구간 선택, 현재 레짐, 전환 여부·위치, 관계 mask, 전환 후 계수를 공동으로 가진다. 외삽 거리마다 독립적으로 RUL을 생성하지 않는다.

## 9. 학습 방법

### 단계 A: 소스 함수·prefix encoder 선학습

소스 유닛의 RUL을 이용해 유닛 함수 분포를 적합하고, 다양한 prefix에서 전체 유닛 함수 분포를 예측하도록 encoder를 학습한다. 한 유닛을 제외한 손실을 반드시 포함한다. 짧은 prefix의 계수 오차가 큰 경우 covariance가 증가하도록 proper score를 사용한다.

### 단계 B: 잠재 전환 원형 학습

TRAIN 유닛 내부에 여러 후보 경계를 두고 경계 전후 함수 변화량을 계산한다. 하나의 hard pseudo-label을 확정하지 않고, 전환 없음과 세 원형의 likelihood를 합산한다. 상태가 너무 자주 바뀌지 않도록 sticky prior와 최소 duration prior를 둔다.

RUL 라벨이 함수 적합에 사용되므로 이 단계에서 얻은 전환을 ‘센서만으로 검출한 실제 레짐’이라고 보고하지 않는다.

### 단계 C: 중첩 외삽 에피소드 학습

외부 TRAIN 분할의 donor 내부에서 다시 유닛과 건강 구간을 제외한다. 내부 소스의 앞부분만 보여주고 내부 query의 뒤 구간을 예측한다. 여기서 구간 어텐션, duration hazard, 관계 mask, 전환 생성기를 학습한다.

주 손실:

`L_pred`: 전체 RUL 혼합분포의 energy score 또는 CRPS

`L_path`: 여러 외삽 거리에서 같은 경로 표본이 내는 예측의 일관성

`L_phys`: 음의 열화·경계 위반·허용하지 않은 불연속에 대한 벌점

`L_sparse`: 관계 모서리와 구간 선택의 복잡도 벌점

`L_invariance`: 선택되지 않은 관계가 스트레스 생성 전후 바뀌는 것에 대한 벌점

`L_duration`: 전환 빈도와 지속시간 prior

`L_diversity`: 전환 원형이 하나로 붕괴하거나 단순 분산 확대만 하는 것을 막는 벌점

최종 목적함수:

`L = L_pred + lambda_path L_path + lambda_phys L_phys + lambda_sparse L_sparse + lambda_inv L_invariance + lambda_duration L_duration + lambda_div L_diversity`

가중치는 테스트 결과로 조정하지 않는다. 최소 탐색 집합을 TRAIN 내부에서 사전 고정하고 동일 예산 대조군과 비교한다.

### 단계 D: 반사실적 구간 제거 감독

구간 어텐션이 실제 외삽 효과를 나타내도록, 구간을 제거했을 때의 손실 변화를 측정한다.

`I_w = L_suffix(X without w) - L_suffix(X)`

어텐션 순위와 `I_w` 순위를 맞추는 손실을 보조적으로 사용한다. 전체 prefix의 일부 값을 0으로 만드는 대신 해당 구간 토큰을 missing token으로 대체한다. 구간 제거가 원자료상 불가능한 관측을 만들 수 있으므로 이를 인과 개입이라고 부르지 않는다.

## 10. 최소 구현 사양

MICH TRAIN은 18유닛이고 유닛별 행 수는 17~141이므로 큰 Transformer는 사용하지 않는다.

- 구간 수: 최대 4
- 전환 경로: 지속, 수준 전환, 가속 전환의 3개
- 전환 횟수: 예측 경로당 최대 1회
- 구간 encoder: 공유 MLP, hidden 16
- 구간 어텐션: 2-head가 아니라 단일 sparse head; 레짐별 query만 분리
- 관계 그래프: 원인 후보 7~10개 × 출력 4개 이하
- 함수 성분: a,b,c
- 전환 위치: 건강 거리 8-bin + 전환 없음 질량
- 분포 표본: 학습 16+16, 평가 Sobol 1,024 이상
- 파라미터 상한: 첫 구현 5,000개 이하
- seed: 최소 5개. 작은 유닛 수 때문에 seed 평균만 보고하지 않고 개별 결과를 보존

첫 구현에서 다중 전환, Transformer stack, 자유로운 관계 그래프, 임의 K 탐색을 동시에 넣지 않는다.

## 11. 필수 대조군과 ablation

모든 대조군은 동일 입력, 함수 decoder, 데이터 분할, 학습 step, 표본 수를 사용한다.

1. 기존 회귀형 함수 분포
2. PP-X 전체 모델
3. 공식 Engression 설정
4. 일반 시간 softmax attention, 레짐 없음
5. 레짐 혼합, 구간 유효기간 없음
6. 레짐·유효기간, 관계 mask 없음: 모든 성분 변화
7. 관계 mask, 반사실 구간 제거 감독 없음
8. 관계 제한 생성기 대신 독립 Gaussian noise
9. 전환 위치를 점추정으로 붕괴
10. 물리 경로 적분 대신 horizon별 독립 RUL 출력
11. hard 물리 mask와 완전 학습 mask 비교

핵심 구조의 기여는 5→6→7과 8의 비교로 본다. 단순 파라미터 증가 효과를 통제하기 위해 일반 어텐션 대조군의 파라미터 수를 맞춘다.

## 12. 평가

### 외삽 정확도

- 전체 및 레짐 전환 인접 구간 RMSE/R2
- 유닛 평균 RMSE와 최악 유닛 RMSE
- 전환 전, 전환 근처, 전환 후 거리별 오차
- R2>0 유닛과 코호트 수

### 분포 품질

- energy score 또는 CRPS
- 50/80/90/95% coverage와 평균 폭
- 외삽 거리별 calibration
- 전환 경로 posterior entropy
- 틀린 레짐을 높은 확률로 선택한 비율

### 레짐 및 관계 선택

- 라벨이 있는 경우 전환 탐지 지연과 precision/recall
- 라벨이 없으면 suffix 예측 향상만 주 평가로 사용
- 구간 제거 효과와 attention 순위의 일치
- 선택되지 않은 관계의 생성 전후 변화량
- 원형별 사용률과 붕괴 여부

실제 레짐 라벨이 없는 상황에서 attention heatmap이 그럴듯하다는 사실은 성능 증거로 사용하지 않는다.

## 13. 개발 및 승격 절차

1. MICH TRAIN 내부에서 중첩된 유닛 제외 + 엄격한 건강 구간 외삽으로 구조를 개발한다.
2. 기본 회귀형, 일반 attention, 전체 성분 noise를 모두 이길 때만 설정을 고정한다.
3. 고정 후 VAL에서 PP-X 전체와 비교한다. VAL 실패 시 테스트를 열지 않고 후보를 종료한다.
4. VAL을 통과한 경우에만 이미 사용된 테스트는 개발 결과로 평가한다.
5. 우월성 주장은 새 잠금 코호트와 전체 PP-X/Engression 동등 예산 비교가 있어야 한다.
6. MICH에서 성공해도 DS03 및 기존 전체 설정의 커버리지를 자동으로 주장하지 않는다.

현재 프로젝트의 상위 승격 조건은 `protocols/SUCCESSOR_EXTRAPOLATION_ACCEPTANCE_PROTOCOL.md`를 그대로 따른다. 모든 설정에서 PP-X와 Engression을 이기고 커버리지를 보존하기 전에는 후속 기본 모델로 승격하지 않는다.

## 14. 실패 및 fallback 규칙

- 전환 posterior가 구분되지 않음: 경로 혼합을 유지하고 불확실성을 보고
- 모든 구간 근거가 약함: population function prior로 후퇴
- hazard가 학습 범위 밖에서 폭주: 전환 없음/학습된 duration prior 혼합으로 제한
- 물리 위반 표본: 단순 clip이 아니라 재매개변수화로 방지; 불가피한 표본은 실패율 보고
- 예측분포가 비정상적으로 넓음: PP-X 점예측으로 숨기지 않고 해당 후보를 미채택
- 알려진 건강 경계 없음: RAVEN-X 비적용 및 기존 검증된 fallback 사용

fallback은 성능 결과를 보고 사후 선택하지 않는다. TRAIN/VAL에서 규칙과 임계값을 고정한다.

## 15. 구현 순서

첫 번째 milestone은 다음 네 파일로 제한한다.

- `src/pp_extrapolation/regime_validity_attention.py`: 구간 토큰, sparse 구간 어텐션, duration hazard
- `src/pp_extrapolation/relation_intervention_generator.py`: 관계 mask와 3개 전환 원형
- `src/pp_extrapolation/piecewise_health_path.py`: 한 번의 전환을 포함한 양의 경로 적분
- `experiments/ravenx_train_screen.py`: 중첩 TRAIN-only 고정 screen과 ablation

1차 screen에서는 실자료 점수와 함께 synthetic falsification을 실행한다. synthetic 자료는 어떤 관계만 바뀌었는지 정답을 알고 있으므로 mask 회복, 잘못된 관계 보존, 전환 위치 회복을 검사한다. synthetic 성공은 실자료 성능이나 인과성 증거가 아니며, 구현이 의도한 구조를 학습할 수 있는지 확인하는 단위 시험이다.

## 16. 이 설계가 성공이라고 말할 수 있는 조건

다음이 모두 성립해야 이 설계의 핵심 가설이 지지된다.

- TRAIN 내부 엄격 외삽에서 일반 attention, 레짐 혼합만 있는 모델, 전체 관계 noise보다 개선
- 성분/관계 mask 제거 시 일관되게 악화
- 전환 구간에서만 좋아지고 비전환 구간을 훼손하는 모델이 아님
- 점예측과 proper distribution score가 함께 개선
- 단순히 구간 폭을 키워 coverage만 높인 것이 아님
- synthetic에서 직접 바뀐 관계와 보존해야 할 관계를 구분
- 여러 seed와 제외 유닛에서 같은 방향
- 고정 후 PP-X 전체 및 Engression과 동등 조건 평가 통과

조건 일부만 통과하면 유용한 진단 또는 보조 모듈일 수는 있지만 모델링 노벨티와 후속 모델 승격을 주장하지 않는다.
