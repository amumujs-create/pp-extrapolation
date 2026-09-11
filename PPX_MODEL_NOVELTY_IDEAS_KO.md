# PP-X 모델적 노벨티 개선 아이디어

작성자: 박진서  
기준 모델: **PP-X — validation-approved prior-residual framework for
contract-conditioned extrapolation**

## 1. 문서의 목적

이 문서는 PP-X의 모델적 노벨티를 강화할 수 있는 후보를 빠짐없이 정리한
연구 아이디어 목록이다. 이미 실패한 아이디어를 새로운 양의 결과처럼
재사용하지 않으며, 현재 최종 PP-X보다 동일 조건에서 좋아진 경우에만
승격한다.

현재 PP-X의 방어 가능한 핵심은 다음과 같다.

1. test outcome과 독립적으로 허용 가능한 구조 가정을 contract로 선언한다.
2. frozen prior가 외삽 방향을 담당하고 NN residual은 source-supported
   deviation만 학습한다.
3. optional executor는 physical-unit validation evidence가 있을 때만
   실행하며, 근거가 부족하면 fallback 또는 abstention한다.

새 아이디어는 이 세 원리를 강화해야 하며 단순한 gate, ensemble 또는
후처리 추가에 그치면 안 된다.

---

## 2. 최우선 추천: Contract-indexed credal prior

### 핵심 개념

하나의 prior를 고르는 대신 서로 다른 구조 가정이 만드는 **허용 가능한
prior 집합**을 예측식의 일부로 둔다.

\[
\mathcal P(x)
=
\left\{
p_j(x):
C_j\ \text{is admissible and validation-approved}
\right\}.
\]

여기서 \(C_j\)는 boundary, monotonicity, causal history, regime invariance
등 prior별 contract다. prior 집합의 중심과 폭은 다음처럼 정의한다.

\[
c(x)=\operatorname{median}_{p\in\mathcal P(x)}p(x),
\qquad
\rho(x)=\max_{p\in\mathcal P(x)}p(x)-\min_{p\in\mathcal P(x)}p(x).
\]

최종 예측은

\[
\hat y(x)
=
c(x)
+a\{d(x),\rho(x),R(x)\}\,
b(x)\tanh\left(\frac{r_\theta(x)}{b(x)}\right)
\]

로 구성한다. \(R(x)\)는 validation-calibrated harm risk다.

### 노벨티가 강해지는 조건

- 같은 모델의 trust 변형이 아니라 서로 다른 구조 가정의 prior를 사용
- prior마다 별도의 contract와 거리 \(d_j(x)\)를 가짐
- prior disagreement를 uncertainty 장식이 아니라 residual 권한의 상한으로 사용
- prior 집합이 비면 prior-off general extrapolator 또는 abstention으로 분기
- set coverage와 residual harm을 physical-unit 단위로 보정

### 필요한 prior

- affine 또는 boundary quotient
- monotone/shape-constrained continuation
- causal-history/state-space continuation
- 관측 regime이 있으면 regime transport
- 실제 물리식이 있는 도메인에서는 mechanistic law

### 현재 상태

trust .02–.40을 사용한 prior-set은 서로 독립된 prior가 아니어서 기각됐다.
affine·monotone·history를 사용한 첫 이질적 prior-set도 Stanford와 ISU의
validation에서 승인되지 않았다. 따라서 개념은 미검증이며 현재 PP-X에
포함하지 않는다.

### 주요 공격

- prior 후보를 데이터셋별로 사후 제작했다.
- agreement가 correctness를 의미하지 않는다.
- 후보 수가 늘면서 선택 자유도가 증가한다.
- validation unit이 적어 prior별 승인을 안정적으로 추정할 수 없다.

### 승격 조건

- 최소 3개 구조 prior를 사전에 고정
- 완전 nested unit-cross-fit
- disagreement가 실제 unit error 또는 regret와 유의하게 연관
- single best prior와 단순 평균을 모두 능가
- 현재 PP-X보다 성능 또는 calibrated coverage가 개선
- 미개봉 cohort에서 정책을 한 번 확인

---

## 3. Risk-calibrated residual authority

### 핵심 개념

승인된 prior라도 NN residual이 prior를 수정할 수 있는 크기를 외삽 거리와
위험에 따라 제한한다.

\[
\left|\hat y(x)-p(x)\right|
\le
a\{d(x),u(x)\}\,b(x),
\qquad
1\ge a_1\ge a_2\ge\cdots\ge a_K\ge0.
\]

여기서 \(u(x)\)는 OOF disagreement 또는 예상 regret다. \(a\)는 일반적인
attention weight가 아니라 residual의 수정 권한이다.

### 강한 버전

- 거리 shell별 authority
- support에서 멀어질수록 비증가하는 단조 제약
- physical-unit mean/CVaR harm budget
- prior 미승인 시 prior 자체가 아니라 prior-off fallback
- one-sided confidence bound로 위해 상한 제어

### 현재 상태

5개 cohort에서 4개는 exact fallback, RWTH는 full residual과 실제 예측이
동일했다. 안전한 거절은 확인했으나 거리별 authority의 추가 성능과 구조적
필요성은 입증되지 않아 기각됐다.

### 재도전 조건

- validation/test가 여러 거리 shell에 실제로 분포
- shell당 충분한 validation unit
- global authority와 distance authority의 예측이 실제로 달라지는 cohort
- 사전 고정한 authority가 새 cohort에서 이득

---

## 4. Prior-specific geometry

### 핵심 개념

모든 prior에 하나의 Euclidean 또는 axis-hull 거리를 사용하는 대신 prior가
표현하는 가정에 맞는 거리를 정의한다.

\[
d_j(x)=
\begin{cases}
d_{\text{boundary}}(x), & j=\text{boundary}\\
d_{\text{order}}(x), & j=\text{monotone}\\
d_{\text{state}}(x), & j=\text{history}\\
d_{\text{regime}}(x), & j=\text{transport}.
\end{cases}
\]

### 예시

- boundary prior: known failure boundary까지의 normalized margin
- monotone prior: partial-order violation distance
- history prior: train trajectory manifold까지의 causal state distance
- regime prior: train regime posterior와 target regime posterior의 divergence

### 노벨티

중간 이상. 단순 support distance보다 contract-conditioned framework와
직접 연결된다. 단, 거리만 새로 정의하면 metric engineering으로 보일 수 있다.
거리와 prior validity/harm의 관계를 증명하거나 검증해야 한다.

### 필수 실험

- 공통 거리 대 prior-specific 거리 ablation
- distance-error 및 distance-regret calibration
- leave-one-domain-out threshold 일반화
- 거리별 coverage와 fallback rate

---

## 5. Contract compiler

### 핵심 개념

사람이 데이터셋마다 모듈을 조립하는 대신 typed contract를 실행 가능한
모형 제약으로 자동 변환한다.

\[
C
\longrightarrow
\left(
p_C,\,
b_C,\,
d_C,\,
\mathcal E_C,\,
\mathcal F_C
\right).
\]

- \(p_C\): frozen prior
- \(b_C\): residual bound
- \(d_C\): contract-specific support distance
- \(\mathcal E_C\): 허용 executor
- \(\mathcal F_C\): fallback/abstention

### 예시

- `known_boundary=True` → boundary quotient와 nonnegative decoder 생성
- `ordered_progression=True` → monotone pair constraints 생성
- `causal_history=True` → short/long causal adapter 허용
- `observed_regime=True` → regime transport 후보 허용

### 노벨티

강함. PP-X를 데이터셋별 모델 모음이 아니라 하나의 **가정 컴파일
architecture**로 보이게 할 수 있다.

### 공격과 대응

- 공격: 단순한 rule engine이다.
- 대응: contract가 loss, architecture, distance, executor set을 동시에
  생성하고 잘못된 contract의 거절까지 포함함을 보인다.
- 공격: 사람이 contract를 사후 선택한다.
- 대응: contract declaration sheet와 outcome-free admissibility test를 공개한다.

### 필수 실험

- 동일 contract에서 결정론적으로 같은 graph가 생성되는지
- contract field별 on/off ablation
- 잘못된 contract를 넣었을 때 validation이 거절하는지
- unseen domain에서 contract만 선언하고 frozen compiler 실행

---

## 6. Counterfactual pseudo-extrapolation training

### 핵심 개념

실제 test tail을 보지 않고 train 내부에서 여러 외삽 grade를 인위적으로
만들어 residual과 gate를 학습한다.

\[
\mathcal D_{\text{train}}
\rightarrow
\left\{
\mathcal D^{(1)}_{\text{pseudo-tail}},
\ldots,
\mathcal D^{(K)}_{\text{pseudo-tail}}
\right\}.
\]

각 unit의 후반부를 숨기고 grade별 pseudo-tail에서 prior, residual, fallback의
regret를 계산한다.

\[
g_\phi(z_k)
\approx
\arg\min_e
\operatorname{Risk}\left(e;\mathcal D^{(k)}_{\text{pseudo-tail}}\right).
\]

### 강한 버전

- row random split이 아니라 unit-disjoint causal backtest
- 여러 horizon/grade에서 학습
- gate가 모델 identity보다 예상 regret를 출력
- grade가 증가할수록 prior/residual authority가 비증가
- outer validation은 gate calibration에만 사용

### 현재 상태

유사 pseudo-extrapolation과 source-only gate를 이미 시험했으나 Engression과
worst-unit 기준을 동시에 넘지 못한 경로가 있다. 재도전하려면 단순 global
gate가 아니라 contract별 regret surface 학습으로 바뀌어야 한다.

### 노벨티

중간~강. “validation에서 하나 고르기”를 “source에서 외삽 실패면을 학습하기”로
바꿀 수 있다.

---

## 7. Regret-predicting executor gate

### 핵심 개념

executor 분류를 직접 학습하지 않고 각 executor의 fallback 대비 conditional
regret를 예측한다.

\[
\widehat{\Delta R}_e(x)
=
\widehat R_e(x)-\widehat R_{\text{fallback}}(x).
\]

\[
e^*(x)=
\arg\min_e \widehat{\Delta R}_e(x),
\quad
\text{execute only if }
U_e(x)<-\delta.
\]

\(U_e(x)\)는 예상 regret의 one-sided upper confidence bound다.

### 노벨티

일반 mixture-of-experts보다 강하다. 정확도 softmax gate가 아니라
fallback-relative risk certificate를 출력하기 때문이다.

### 필수 제약

- target은 nested OOF physical-unit regret
- test-time label이나 batch adaptation 금지
- upper confidence bound가 0보다 작을 때만 실행
- 나머지는 exact fallback

### 필수 실험

- gate AUC보다 calibration curve와 selective risk 보고
- false-accept/false-reject 분리
- 기존 Algorithm 1 임계 규칙과 비교
- leave-one-domain-out 일반화

### 2026-09-12 1차 실험 판정

거리, support 이탈률, 5-seed disagreement, prior correction 크기를 입력으로
사용한 ridge regret-bound head를 12-domain leave-one-domain-out으로 평가했다.
95% inner-LODO residual upper bound가 모든 held-out domain에서 양수였고
binary/continuous authority가 모두 0이 됐다. 즉 cross-domain feature로
unit regret를 예측하는 현재 구조는 기각한다. 상세 결과는
`FALSIFICATION_AUTHORITY_HEAD_RESULTS_KO.md`에 기록한다.

---

## 8. Invariant residual decomposition

### 핵심 개념

prior 주변 residual을 invariant 부분과 environment-specific nuisance로
분해한다.

\[
y=p(x)+r_{\text{inv}}(z)+r_{\text{env}}(z,e).
\]

test에서는 \(r_{\text{inv}}\)만 사용하고 \(r_{\text{env}}\)는 train에서
shift mechanism을 식별하거나 불확실성을 계산하는 데만 사용한다.

### 가능한 구현

- environment adversarial residual
- group-wise residual variance penalty
- invariant causal representation
- leave-one-regime-out residual consistency

### 노벨티

중간. IRM, V-REx, domain generalization과 겹치므로 prior-preserving
decomposition과 연결해야 한다.

### 필수 실험

- direct invariant model과 비교
- prior 없이 invariant residual만 쓴 arm
- residual invariance와 test gain의 연관
- regime label 오류 민감도

---

## 9. Causal multiscale residual bank

### 핵심 개념

short/medium/long history residual을 하나로 합치지 않고 각각 별도 expert로
학습하고, train-only causal backtest가 horizon별 권한을 정한다.

\[
r(x)
=
\sum_{h\in\mathcal H}
a_h(d,\rho)\,r_h(x_{t-h:t}).
\]

### 강한 버전

- 각 scale은 미래 정보가 없는 causal adapter
- scale별 OOF regret와 stability 측정
- 외삽 horizon이 길어질수록 short-history 권한 감소
- regime shift가 크면 long-history도 거절 가능

### 노벨티

중간. 단순 multiscale temporal network는 흔하다. PP-X에서는 history scale을
prior 수정 권한과 연결하고, 승인되지 않은 scale을 exact 제거해야 차별화된다.

---

## 10. Mechanism graph executor

### 핵심 개념

executor를 평면 후보 목록이 아니라 선후관계가 있는 mechanism graph로 둔다.

\[
\text{boundary}
\rightarrow
\text{residual}
\rightarrow
\text{support adaptation}
\rightarrow
\text{regime transport}.
\]

상위 노드는 하위 mechanism의 증거가 있을 때만 활성화된다. 예를 들어
boundary consistency가 없으면 boundary residual과 그 위의 dual-scale은
후보에서 제거한다.

### 장점

- optional module 조합 폭발 감소
- 사후 dataset-specific 조립이라는 공격 완화
- executor 선택의 인과적 순서를 설명 가능

### 노벨티

중간~강. typed contract를 graph grammar로 구현하면 contract compiler와
결합해 강해진다.

### 필수 실험

- flat search와 graph-constrained search 비교
- 후보 수와 계산비용
- false accept 감소
- graph edge별 mechanism ablation

---

## 11. Set-valued prediction과 selective abstention

### 핵심 개념

prior disagreement가 클 때 하나의 점 예측을 강제로 출력하지 않고 예측 집합
또는 abstention을 출력한다.

\[
\Gamma_\alpha(x)
=
\left[
c(x)-q_\alpha s(x),
c(x)+q_\alpha s(x)
\right].
\]

\[
s(x)=
\epsilon
+\widehat u_{\text{OOF}}(x)
+\lambda_d d(x)
+\lambda_\rho\rho(x).
\]

### 노벨티

점 예측 정확도보다 안전한 외삽을 목표로 할 경우 강하다. 다만 conformal
prediction 자체는 기존 방법이므로 contract-specific nonconformity와
physical-unit block calibration이 핵심이다.

### 필수 실험

- 80/90/95% empirical coverage
- 거리 shell별 coverage와 width
- risk-coverage curve
- abstention rate
- calibration/test regime exchangeability 붕괴 사례

---

## 12. Distributional prior-residual model

### 핵심 개념

residual의 평균만 학습하지 않고 조건부 분포를 학습하되, prior로부터 벗어날
수 있는 distributional mass를 제한한다.

\[
Y
=
p(x)
+a(x)R_\theta(x,\epsilon).
\]

\[
\Pr\left(
|R_\theta|>b(x)
\mid x
\right)
\le \eta(x).
\]

### 후보 구현

- quantile residual network
- energy-score residual generator
- diffusion/flow residual
- stochastic dynamics ensemble

### 노벨티

중간. Engression과 직접 경쟁하려면 필요할 수 있지만, 단순 확률모델 추가는
PP-X 고유 기여가 아니다. prior-conditioned support와 tail-risk 제약이 있어야
한다.

### 현재 교훈

CRT와 GCIE는 DS03에서 Engression을 넘지 못했다. 확률적 residual 자체보다
언제 prior-conditioned distribution을 허용할지에 대한 risk certificate가
먼저 필요하다.

---

## 13. Minimax prior-residual optimization

### 핵심 개념

평균 validation loss를 최소화하지 않고 plausible shift 또는 prior 집합에서
최악의 regret를 줄인다.

\[
\min_{\theta,a}
\max_{q\in\mathcal Q(C)}
\mathbb E_q
\left[
\ell\{p+a r_\theta,y\}
-\ell\{f_0,y\}
\right].
\]

\(\mathcal Q(C)\)는 contract가 허용하는 shift 집합이다.

### 노벨티

중간~강. 일반 DRO와 달리 ambiguity set을 typed contract와 extrapolation
grade로 구성해야 한다.

### 필수 실험

- GroupDRO/V-REx와 직접 비교
- 평균 성능과 worst-unit 성능의 trade-off
- ambiguity radius 민감도
- contract가 틀렸을 때의 failure mode

---

## 14. Bilevel validation-approved learning

### 핵심 개념

모델 학습과 executor 승인을 분리하되 승인 규칙 자체를 외부 unit 기준으로
최적화한다.

\[
\theta_e^*
=
\arg\min_\theta L_{\text{inner-train}}(\theta,e),
\]

\[
\pi^*
=
\arg\min_\pi
R_{\text{outer-unit}}
\left(
\pi\{\theta_e^*\}
\right).
\]

### 주의

같은 validation dataset에서 반복 최적화하면 meta-overfitting이다.
leave-one-domain-out 또는 여러 development domain이 필요하다.

### 노벨티

수학적으로는 강해 보이나 데이터 요구량이 크다. 현재 데이터 규모에서는
복잡한 neural bilevel gate보다 단순하고 해석 가능한 regret certificate가
안전하다.

---

## 15. Partial-order monotone residual

### 핵심 개념

output 전체에 단조성을 강제하지 않고, contract가 확인한 비교쌍에서 residual의
방향만 제한한다.

\[
x_i\preceq_C x_j
\Rightarrow
r_\theta(x_i)-r_\theta(x_j)
\in[L_{ij},U_{ij}].
\]

### 장점

- 전역 monotone model보다 유연
- prior가 틀린 영역에서 과도한 shape constraint를 피함
- residual이 prior의 순서를 뒤집는 조건을 직접 통제

### 노벨티

중간. monotonic neural networks와 겹치므로 typed partial order와
validation rejection이 핵심이다.

### 필수 실험

- global monotone NN
- unconstrained residual
- partial-order residual
- 잘못된 pair 주입 민감도

---

## 16. Prior falsification head

### 핵심 개념

prior를 잘 적용하는 방법뿐 아니라 prior가 틀렸다는 증거를 출력하는 별도
head를 학습한다.

\[
F_j(x)
=
\Pr\left(
\Delta R_j(x)>0
\mid z_{\text{source}}
\right).
\]

승인 조건은 단순 confidence가 아니라 falsification probability의 상한이다.

\[
\operatorname{UCB}\{F_j(x)\}<\tau.
\]

### 노벨티

강한 편. “assumption execution”뿐 아니라 “assumption falsification”을
architecture에 넣는다는 점에서 PP-X 철학과 잘 맞는다.

### 위험

- 오류 label은 실제 test가 아니라 nested pseudo-tail에서 만들어야 함
- calibration이 나쁘면 confidence head 장식에 그침
- 실패 예가 충분하지 않으면 학습 불가

### 필수 실험

- prior failure detection calibration
- false-accept 감소
- prior-off cohort에서 rejection sensitivity
- synthetic contract violation과 실제 negative cohort 모두 평가

---

## 17. Contract uncertainty propagation

### 핵심 개념

contract를 참/거짓 boolean으로만 두지 않고 신뢰 구간이나 ambiguity로 표현한다.

\[
C_j\sim\operatorname{Bernoulli}(\pi_j),
\qquad
\hat y=
\mathbb E_{C}
\left[f(x\mid C)\right].
\]

또는 안전하게

\[
\hat y
=
\arg\min_z
\max_{C\in\mathcal C_{\text{plausible}}}
\ell\{z,f(x\mid C)\}.
\]

### 노벨티

강하지만 난도가 높다. contract confidence가 임의 hyperparameter가 되지
않도록 source-only falsification 또는 expert elicitation protocol이 필요하다.

---

## 18. Executor interaction attribution

### 핵심 개념

bound, dual scale, history, transport의 단일 on/off가 아니라 상호작용을
명시적으로 모델링한다.

\[
\Delta R(S)
=
R(\varnothing)-R(S),
\qquad
\phi_e=\text{Shapley interaction over admissible executors}.
\]

### 용도

이것은 새 예측 모델보다는 구조 필요성을 설명하는 분석 기여다. 예를 들어
MICH에서 fixed bound가 악화되고 dual scale이 복구하는 현상을 interaction으로
정량화할 수 있다.

### 노벨티

단독 모델 노벨티는 낮지만 리뷰어가 요구하는 “각 모듈이 왜 필요한가”에 대한
설명력은 높다.

---

## 19. Meta-learned applicability law

### 핵심 개념

데이터셋 수준의 train-only meta-feature로 어떤 contract/executor가 유효한지
예측한다.

후보 meta-feature:

- normalized extrapolation horizon
- trajectory heterogeneity
- degradation SNR
- source curvature
- boundary consistency
- regime coverage
- support-density ratio

\[
\Pr(e\text{ improves}\mid m_1,\ldots,m_K).
\]

### 한계

현재 domain 수로 복잡한 meta-model을 학습하면 식별되지 않는다. 우선
10–12개 이상의 동일 조건 dataset outcome이 필요하며, leave-one-domain-out
평가가 필수다.

### 노벨티

충분한 domain이 있으면 강함. 현재는 연구 로드맵으로만 유지한다.

---

## 20. 구조적으로 피해야 할 약한 확장

다음은 복잡도는 늘지만 노벨티와 증거가 약해질 가능성이 높다.

- 같은 prior의 trust 값만 늘린 softmax ensemble
- validation row를 독립 표본으로 학습한 neural gate
- test distance를 본 뒤 shell threshold를 조정
- 모든 데이터셋에 monotonicity 또는 boundary를 강제
- 평균 R²만 높이는 데이터셋별 사후 executor
- fallback 사용률 100%를 새로운 안정성 성능으로 주장
- random seed를 독립 physical unit으로 취급
- Engression 블록을 그대로 붙이고 PP-X 노벨티라고 주장
- 여러 기존 모듈을 병렬 연결한 뒤 새 architecture 이름만 부여

---

## 21. 우선순위

### A급: 실제로 논문 노벨티를 크게 높일 후보

1. **Contract compiler**
2. **Prior falsification head**
3. **Contract-indexed credal prior + residual authority**
4. **Regret-predicting executor certificate**

이 네 개는 하나의 통합 구조로 합칠 수 있다.

\[
\text{Declare contract}
\rightarrow
\text{Compile priors}
\rightarrow
\text{Falsify unsupported priors}
\rightarrow
\text{Construct credal set}
\rightarrow
\text{Allocate residual authority}
\rightarrow
\text{Fallback/abstain}.
\]

### B급: 성능 또는 검증을 보강할 후보

5. prior-specific geometry
6. counterfactual pseudo-extrapolation
7. causal multiscale residual bank
8. set-valued prediction과 selective abstention
9. partial-order monotone residual

### C급: 데이터가 더 필요하거나 후속 논문에 적합

10. meta-learned applicability law
11. bilevel validation-approved learning
12. contract uncertainty propagation
13. minimax contract-DRO
14. distributional residual model

---

## 22. 가장 현실적인 다음 실험

현재 자료에서 가장 먼저 할 수 있는 것은 **prior falsification certificate**다.

### 이유

- 기존 PP-X의 핵심인 승인/거절을 모델 구조로 강화한다.
- 새로운 prior를 많이 만들지 않아도 된다.
- 이미 확보한 성공·실패 executor와 negative cohort를 학습 표본으로 활용할 수 있다.
- 단순 accuracy gate보다 false accept를 직접 줄이는 목적이 명확하다.
- prospective cohort에서 평가할 판정 기준을 사전에 고정할 수 있다.

### 최소 구현

1. 각 physical unit의 nested OOF prior regret를 target으로 생성
2. train-only distance, curvature, disagreement, boundary violation을 입력
3. one-sided quantile head로 regret upper bound 예측
4. upper bound가 0보다 작을 때만 prior executor 승인
5. 기존 Algorithm 1, CI-low rule, 항상 prior, 항상 fallback과 비교

### 성공 기준

- false accept 감소
- 승인된 route의 mean unit log-RMSE 개선
- fallback 대비 worst-unit ratio 제한
- 기존 PP-X보다 route decision accuracy 개선
- leave-one-domain-out 결과와 미개봉 cohort 결과가 같은 방향

---

## 23. 최종 권고

첫 PP-X 논문을 지금 제출한다면 검증되지 않은 새 모듈을 추가하지 않는다.
현재 논문은 contract-conditioned prior-residual framework와 검증된 executor
conditionality를 중심으로 유지한다.

모델 노벨티를 한 단계 더 높이는 별도 개발은 다음 통합 구조를 목표로 한다.

> **A contract compiler that constructs admissible structural priors, a
> falsification certificate that removes unsupported assumptions, and a
> credal residual authority mechanism that limits data-driven correction
> according to prior disagreement and extrapolation risk.**

이 구조가 현재 PP-X보다 동일 조건에서 실제로 개선되고 prospective 검증까지
통과할 때만 PP-X v2 또는 후속 논문의 메인 모델로 승격한다.
