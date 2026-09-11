# 전체 데이터셋 기반 PP-X 해결 구조 감사

작성자: 박진서  
감사 범위: 로컬 `data/` 36개 cohort, legacy benchmark 8개, 실험 스크립트
234개, 결과 JSON 242개, protocol 38개

## 결론

현재 문제는 PP prior 자체가 약해서가 아니라, 서로 다른 데이터 생성 구조에
동일한 PP-X core와 하나의 validation gate를 적용한 데 있다. 전체 증거가
지지하는 해법은 더 복잡한 단일 gate가 아니라 다음의
**Contract-Conditioned Risk-Budgeted Prior Experts(CRPE)** 구조다.

1. 데이터 계약으로 expert family를 먼저 고른다.
2. 각 family 안에서 강한 no-prior baseline을 anchor로 둔다.
3. prior residual만 연속적으로 허용한다.
4. raw physical-unit CVaR hard cap을 통과하지 못하면 baseline을 정확히 복원한다.

이 구조는 아직 신규 미개봉 cohort에서 확증되지 않은 설계안이다.

## 전체 데이터가 보여 준 사실

### 1. 단일 범용 PP/gate는 일반화되지 않았다

12-domain matched audit에서 plain MLP 대비 PP가 이긴 dataset은 6/12였다.
validation ensemble RMSE만 사용한 gate는 7/12만 올바르게 선택했고, false PP
accept가 4개였다. train label과 test covariate descriptor만 사용한
leave-one-domain-out stump gate는 1/12 정확도였다.

support distance도 해결 변수가 아니었다. ordered hull, PCA hull, kNN density와
PP unit gain의 dataset-level 연관은 모두 유의하지 않았다
(permutation p ≥ .15). 따라서 거리만으로 prior를 줄이는 현재 RBPR의 한계는
구조적이다.

### 2. task별 modular route는 작동했다

route를 데이터 계약에 맞게 분리한 final modular evidence에서는 9개 dataset,
77 physical unit에서 ensemble 기준 9/9 승리했고 dataset sign test
p=.0039, geometric-mean RMSE reduction은 32.5%였다.

단, 이 route들은 서로 다른 retrospective 개발을 거쳐 선택됐으므로 그 숫자를
새 범용모형의 확증 성능으로 부르면 안 된다. 여기서 얻을 수 있는 정당한 결론은
“하나의 PP 식”보다 “계약별 prior expert”가 맞다는 것이다.

### 3. 현재 저용량 battery 체인만으로는 범용 gate를 학습할 수 없다

- Stanford: 41 eligible, split 24/8/9, tail rows 11407/1572/2427
- ISU 250 mAh: 229 eligible, split 137/46/46, tail rows 1375/167/148
- UConn ILCC/NMC: eligible 수가 protocol 최소치 미달
- Tongji/SNL: 엄격한 outer tail split이 infeasible
- CALB/UL-PUR/NASA PCoE2: eligible unit 부족

즉 v1.2/RBPR가 실제로 학습·재생된 strict 저용량 cohort는 Stanford와 ISU 두
개뿐이다. 두 dataset을 반복해서 보며 gate를 복잡하게 만드는 것은
cross-dataset 일반화가 아니라 test 재튜닝이다.

### 4. 소표본 unit 위험은 수축하면 안 된다

ISU hierarchical CVaR 실험에서 `n0=10` 부분수축은 R²를 .5205까지 높였지만
raw CVaR는 11.41%였다. 수축된 CVaR는 -3.71%로 위험을 숨겼다. 따라서
hierarchical estimate는 보조 추정량으로만 쓸 수 있고 raw unit risk는 반드시
hard constraint로 남아야 한다.

### 5. 단위·scale contract가 구조보다 먼저다

기존 scale audit에서는 bounded correction이 필요한 affine error의 0.23%만
도달 가능했다. target, boundary age, residual bound의 단위가 맞지 않으면
어떤 gate도 개선할 수 없다.

## 제안 모형: CRPE

\[
\hat y(x)=B_c(x)+\alpha_c(x)\{P_c(x)-B_c(x)\}
\]

여기서 \(c\)는 validation 성능으로 고르는 latent class가 아니라 입력·목표
정의에서 사전에 결정되는 **data contract**다.

### Expert family

1. **Observed monotone boundary trajectory**
   - 예: battery capacity/SOH, wear, crack length
   - expert: dimensionless boundary quotient 또는 log-boundary quotient
   - 필요 조건: 명시적 ordered health coordinate와 boundary

2. **Latent RUL from causal history**
   - 예: N-CMAPSS, waveform/inspection RUL
   - expert: causal multiscale latent prior
   - 미래·test-batch 통계 사용 금지

3. **Operating-regime transfer**
   - 예: HUST, MATR batch transfer, chemistry/condition shift
   - expert: regime transport/mixture
   - regime identifier 또는 train-only latent regime 필요

4. **Unresolved or evidence-poor**
   - validation physical unit가 적거나 unit당 tail 행이 부족
   - expert: no-prior baseline portfolio
   - 결과 상태를 실패가 아니라 `inconclusive`로 반환

### 공통 안전층

- baseline: 단일 MLP가 아니라 architecture × seed no-prior portfolio
- alpha: family 내부 group-OOF에서만 학습
- hard constraints:
  - raw unit-mean excess MSE ≤ 사전 예산
  - raw worst-20% unit CVaR ≤ 사전 예산
  - pooled RMSE non-inferiority
- soft auxiliary:
  - hierarchical risk, disagreement, support distance
- hard constraint가 하나라도 실패하면 alpha=0
- shrinkage로 raw CVaR를 대체하지 않음

이 공통 안전층의 첫 구현은 `Stability-First RBPR v1.3`이다. Stanford에서는
fold 합의 부족, ISU에서는 maximum raw unit regret 초과로 prior를 거부해 두
고호트 모두 exact baseline을 선택했다. 성능 최고점보다 안정성을 우선하는
실행 정책으로 채택했다.

후속 `Auto-Regime Weak-Prior v1.4`는 이 안전층을 레짐별로 적용했다. train-only
K=3 레짐에서 prior residual을 최대 25%만 허용한 결과 ISU R² 0.4897,
raw CVaR 1.96%, maximum regret 3.65%를 얻었고 Stanford도 baseline을
비악화했다. 이는 CRPE의 direct-continuation expert prototype으로 유지한다.

그러나 신규 1.97 MB LED stress set에서는 시간척도 이동을 기존 레짐으로
오인해 baseline 대비 mean/CVaR/max regret가 11.91%/18.86%/22.05%로
악화됐다. validation support distance 최대가 2.91인데 stress 최소가
41.65였으므로, 다음 버전에는 cluster posterior와 별도로 absolute OOD
rejection이 필요하다. 사후 support guard는 exact baseline을 복원했지만
확증 결과로 사용하지 않는다.

이어진 246 KB LED 3-cohort 강한 외삽에서는 기존 absolute-output v1.4가
세 set 모두 음의 R²로 실패했다. post-test OAIR v1.5는 persistence anchor,
ordered-axis invariant slope residual, correction bound와 25% deployment
shrinkage를 결합해 세 set 모두 양의 R²(0.545/0.515/0.276), persistence
RMSE 비악화, raw unit-risk cap을 달성했다. 구조 후보로는 승격하지만 같은
점수를 본 사후 개발이므로 미개봉 확증이 필요하다.

이후 공식 NASA PCoE의 미사용 18-cell 자료를 조건별로 고정하고, 방전
초반 0--30%에서 학습해 별도 4-cell의 75--85%를 예측하는 one-shot
확증을 수행했다. Persistence는 pooled R² 0.9711을 유지했지만 OAIR
v1.5는 RMSE를 0.10846에서 0.10927로 0.75% 악화시켜 확증에 실패했다.
Slope-only support guard가 test 행을 거부하지 못하면서 B0051·B0052
보정이 악화된 것이 핵심이다. 따라서 v1.5를 확증 모델로 승격하지 않고,
selective correction gate를 별도 development 자료에서 재설계해야 한다.

후속 CCMR v1.6은 train-unit cross-fit 방향 합의, full-context support,
validation minimax raw risk를 동시에 요구한다. LED post-test
development에서 Data_ID_5만 작은 RMSE 개선을 허용하고 Data_ID_2/6은
exact persistence로 거부했다.
구조와 기준을 고정한 뒤 처음 개봉한 773-byte Alloy A 실제 피로균열
21-specimen 고호트에서는 train 0--30% / test 75% 이후 강한 시간 외삽
RMSE를 0.10262에서 0.10125로 1.33% 개선했고, test raw-unit maximum
regret 0과 fallback replay error 0을 얻었다. 사후 metric-definition
감사에서 selector의 stabilizer를 제거한 진짜 raw regret 구현도 봉인
예측과 최대 절대 차이 0임을 확인했다. 다만 test가 4 specimens, 10
origins인 저검정력 결과이므로 더 큰 condition-shift 확증이 필요하다.

추가 105°C train / 65°C validation / 25°C test luminosity condition
shift에서는 pooled RMSE 2.45%, unit-macro RMSE 3.11%를 개선했지만,
test raw unit CVaR20과 maximum regret가 13.13%와 29.34%로 폭증했다.
행의 97.71%가 활성화돼 정적 context support가 in-support concept shift를
거의 거부하지 못했다. 따라서 v1.6.1은 평균 성능 모델로는 개선됐지만
condition-shift 강건성 확증에는 실패했으며, 다음 버전에는 과거에 관측
완료된 동일-horizon shadow forecast 이득 기반 causal backtest gate가
필요하다.

## 왜 이 구조가 현재 RBPR보다 낫나

현재 RBPR는 Stanford·ISU의 같은 direct residual family 안에서 prior 양만
조절한다. CRPE는 prior의 종류 자체가 데이터 계약과 맞지 않을 때 alpha를
조절하는 대신 다른 expert family를 사용한다.

거리·seed disagreement는 “얼마나 멀고 불안정한가”는 알려 주지만
“어떤 물리적 continuation law가 맞는가”는 알려 주지 않는다. 전체 dataset의
성공 패턴은 후자가 더 중요하다는 증거다.

## 구현·검증 순서

1. 모든 target과 correction을 dimensionless scale로 통일
2. 네 contract를 코드 enum과 필수 입력 schema로 고정
3. 기존 canonical 12-domain을 contract별 development set으로만 사용
4. route family 단위 leave-one-dataset-out 검증
5. raw-CVaR dual certificate와 exact baseline fallback 시험
6. contract별 완전히 미개봉 physical-unit cohort를 한 번만 평가

성공 기준:

- 각 contract family에서 baseline 대비 dataset-macro RMSE 비악화
- raw unit CVaR 예산 준수
- 최소 두 family에서 독립 cohort 개선
- fallback max absolute replay error 0

## 당장 유지할 것

- Stanford·ISU direct continuation: Auto-Regime Weak-Prior v1.4를 차기
  개발 후보, Stability-First v1.3을 공통 fallback 안전층으로 유지
- hierarchical RBPR와 crossfit monotone RBPR: 실패 ablation으로 유지
- final modular route 결과: CRPE expert 초기값으로 사용
- 신규 확증 전에는 기존 test를 보고 threshold, shrinkage, family rule을 바꾸지 않음

## 핵심 근거

- `results/final_modular_pp_evidence_v1/results.json`
- `results/journal_validation_route_v1/results.json`
- `results/meta_applicability_v1/results.json`
- `results/journal_support_geometry_v1/results.json`
- `results/pp_scale_contract_audit_v1/results.json`
- `results/cohort_evidence_audit_v2/results.json`
- `results/ppx_unified_final_v1/registry.json`
- `RBPR_HIERARCHICAL_RISK_DEVELOPMENT_KO.md`
