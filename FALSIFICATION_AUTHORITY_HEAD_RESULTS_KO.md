# PP-X 학습형 falsification-aware authority head 결과

작성자: 박진서  
상태: 12개 공개 domain의 retrospective leave-one-domain-out 개발  
프로토콜: `protocols/FALSIFICATION_AWARE_AUTHORITY_HEAD_PROTOCOL.md`

## 결론

학습 가능한 regret upper-bound head와 연속 residual authority를 구현했지만
**현재 PP-X 또는 bootstrap falsification certificate로 승격하지 않는다.**

12개 outer held-out domain 모두에서 예측된 95% regret upper bound가 양수였다.
따라서 binary authority와 continuous authority가 모두 정확히 0이 되었고,
모든 예측이 direct fallback과 같았다.

이 결과는 안전한 거절 동작은 확인하지만 학습형 head의 성능이나 모델적
필요성을 입증하지 않는다.

## 구조

각 physical unit에서 outcome을 사용하지 않고 다음 feature를 계산했다.

- row 수
- support distance 평균·90% 분위수·support 밖 비율
- direct/PP 5-seed disagreement
- PP와 direct의 평균·90% correction magnitude
- signed correction
- PP/direct prediction-range ratio

다른 11개 domain의 validation unit에서

\[
r_u=\log
\frac{\operatorname{RMSE}_{u,\mathrm{PP}}}
     {\operatorname{RMSE}_{u,\mathrm{direct}}}
\]

를 학습하고, inner leave-one-domain-out residual의 95% 분위수를 더해
held-out domain의 regret upper bound를 만들었다.

\[
a_u=
\operatorname{clip}
\left(
\frac{-\widehat U_u}{s_R},0,1
\right),
\qquad
\hat y=f_0+a_u(f_{\mathrm{PP}}-f_0).
\]

held-out test label, dataset identity, test residual은 head 입력이나 calibration에
사용하지 않았다.

## 전체 결과

| arm | active domain | false-harm domain | 개선 domain | 평균 log-RMSE 개선 |
|---|---:|---:|---:|---:|
| direct fallback | 0 | 0 | 0 | 0 |
| always PP | 12 | 6 | 6 | 0.0819 |
| within-domain certificate | 4 | 0 | 4 | **0.1543** |
| learned binary authority | 0 | 0 | 0 | 0 |
| learned continuous authority | 0 | 0 | 0 | 0 |
| unit oracle | 10 | 0 | 10 | 0.3104 |

기존 within-domain certificate의 dataset bootstrap 95% CI는
[0.00085, 0.42942]였고, learned authority는 fallback과 같아 [0, 0]이었다.

## 실패 원인

### 1. Cross-domain feature의 예측력이 부족했다

12개 outer fold 중 11개에서 ridge alpha가 최대 후보인 1000으로 선택됐다.
이는 correction magnitude, seed disagreement, support distance와 validation
unit regret 사이의 안정적인 선형 관계를 inner LODO가 찾지 못했다는 뜻이다.
XJTU fold만 alpha 0.01이 선택됐지만 test upper bound는 여전히 양수였다.

### 2. Domain 간 residual upper tail이 너무 컸다

inner-LODO upper residual quantile은 약 1.30–1.50 log-RMSE였다. 이 보정값이
mean head prediction을 압도해 모든 test unit의 upper bound를 양수로 만들었다.

예측 upper bound의 최소값도 다음과 같이 양수였다.

- HUST +1.322
- Virkler +1.309
- MATR2019 +1.324
- Sunwoda +1.325
- RWTH +1.326
- MICH +1.144
- MATR-b2 +1.320
- XJTU +0.478
- FEMTO +1.198
- Milling +1.316
- NASA battery +1.318
- N-CMAPSS +1.322

### 3. 12개 domain으로 universal head를 학습하기에는 shift가 너무 달랐다

Virkler와 Sunwoda의 PP 이득은 크지만 Milling의 손실도 매우 크다. 동일한
feature-to-regret 관계를 모든 fatigue, battery, milling, turbofan domain에
강제하면 one-sided bound가 넓어질 수밖에 없다.

### 4. Continuous authority의 추가 가치는 평가되지 못했다

모든 upper bound가 양수여서 \(a=0\)이었다. binary와 continuous arm이
동일하므로 연속 residual authority가 binary rejection보다 낫다는 승격 조건을
통과하지 못했다.

## 노벨티 판정

구조 자체는 기존 bootstrap certificate보다 모델적이다.

- outcome-free unit feature encoder
- cross-domain learned regret head
- one-sided residual calibration
- predicted harm bound와 residual authority의 직접 연결

그러나 새로운 구조는 성능과 active coverage를 함께 보여야 한다. 현재처럼
모든 domain을 거절하면 안전한 모델이 아니라 비활성 모델이다. 따라서 논문
메인 기여로 사용하지 않는다.

## 다음 설계에 대한 교훈

universal cross-domain head를 더 복잡한 neural network로 바꾸는 것은 권장하지
않는다. domain 표본이 12개뿐이라 과적합만 증가할 가능성이 높다.

재도전한다면 다음 조건이 필요하다.

1. typed contract별로 domain을 나눈 contract-conditioned head
2. source 내부 여러 pseudo-tail grade에서 훨씬 많은 regret target 생성
3. domain identity가 아니라 contract violation feature 사용
4. global certificate가 prior를 승인한 뒤 local authority를 조절하는 계층 구조
5. 새 domain에서 authority coverage와 false harm을 동시 검증

## 최종 판정

- 모델 구조 구현: 통과
- label leakage 방지: 통과
- leave-one-domain-out 평가: 통과
- 비영(非零) authority: 실패
- 기존 certificate 대비 성능: 실패
- continuous authority의 필요성: 실패
- prospective 확인: 미실시

**현재 채택 모델은 기존 PP-X + physical-unit bootstrap falsification
certificate이며, 학습형 authority head는 기각된 challenger로 보존한다.**
