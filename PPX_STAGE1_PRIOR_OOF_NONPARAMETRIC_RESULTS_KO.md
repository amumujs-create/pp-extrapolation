# Stage-1 Prior OOF Regret 비모수 감사 결과

작성자: 박진서  
프로토콜: `protocols/PPX_STAGE1_PRIOR_OOF_NONPARAMETRIC_PROTOCOL.md`  
실험: `experiments/ppx_stage1_prior_oof_nonparametric.py`  
결과: `results/ppx_stage1_prior_oof_nonparametric_v1/results.json`

## 한 줄

**prior-only(ridge affine)는 validation에서 matched direct MLP보다 대체로 진다
(9/12 regret>0).** 다만 prior가 져도 stage2가 PP를 승인하고 test에서도 PP가
이긴 도메인이 3개 있어, “잔차가 prior를 구제”하는 경우는 **실제로 관측**된다.
이 결과로 hard stage-1을 풀지는 않는다(이전 Policy A 거절 유지).

## 비교

| | 정의 |
|---|---|
| prior | train-unit LOO로 α 선택 → 전체 train ridge 재적합 → val 예측 |
| direct | `cross_domain_mechanism_v2`의 `validation_plain` ensemble |

α 선택에 validation label 미사용.

## 요약

| 지표 | 값 |
|---|---:|
| stage1 pass (regret≤0) | 3 |
| stage1 fail | 9 |
| prior mean log-RMSE 우세 도메인 | 4/12 (binomial p=.39) |
| S1 pass ∧ S2 accept | 3 |
| S1 fail ∧ S2 accept | **3** |
| S1 fail ∧ test PP 이득 | **5** |

### Pass / Fail

- pass: femto, milling, ncmapss  
  - femto·milling은 validation unit **1개(L0)** → 확증 해석 금지  
  - **ncmapss만 L1 확증 후보**: 6/6 unit 우세, sign-flip p=.031, Wilcoxon p=.031
- fail: hust, virkler, matr, sunwoda, rwth, mich, matr_batch2, xjtu, nasa_battery

### Residual-rescue (S1 fail ∩ S2 accept ∩ test helped)

| 데이터 | prior regret | unit 방향 | S2 | test |
|---|---:|---|---|---|
| virkler | +1118 | 0/10 prior 패배 (p=.002) | accept | helped |
| matr_batch2 | +2773 | 0/9 prior 패배 (p=.004) | accept | helped |
| nasa_battery | +191 | 2/4 | accept | helped |

→ “prior만 보고 막으면 결합 이득을 놓친다”는 우려의 **실증 사례**.  
그러나 Policy A 감사에서 false rescue(femto, milling) 때문에 **strict 승격은
이미 거절**했고, 이번 결과도 hard-S1 해제를 정당화하지 않는다.

## 비모수 관점

- hust/rwth/matr_batch2/virkler: prior 열세가 sign-flip·Wilcoxon에서 유의
- ncmapss: prior 우세가 L1에서 유의 (유일하게 깨끗한 S1 pass)
- L0(femto/milling/sunwoda n≤2): p를 확증으로 쓰지 않음

## 해석 (방법론)

1. Affine prior 단독은 강한 비교기(direct MLP)에 자주 진다 → stage-1 OOF 가지가
   **보수적으로 자주 탈락**하는 것이 데이터와 맞다.
2. 최종 PP-X 이득의 상당 부분은 **prior+residual**에서 나오며, prior-only 승리가
   전제는 아니다. 다만 운영에서는 known-boundary 등으로 S1이 열리는 경로와
   OOF-regret 경로를 구분해서 써야 한다.
3. Algorithm 1의 τ·hard-S1은 **변경하지 않음**. 근거 문서만 추가.

## 한계

- prior = weighted ridge affine (PPNet affine 초기화와 동일 계열). NN 내부
  learned gate 후의 “유효 prior”와는 다를 수 있다.
- retrospective; prospective DS03 OOF 증거 부재 문제는 별개.
- femto 등 스케일·unit 수 이상치는 L0로 격리.
