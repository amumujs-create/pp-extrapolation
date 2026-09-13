# PP-X Prior-Gate Softening (Policy A) — Promotion Audit

작성자: 박진서  
상태: retrospective counterfactual audit (not a method promotion)

## 질문

1단계 prior admissibility가 탈락했을 때도 2단계 unit-risk approval로
`prior+residual`을 후보에 올리면(승격 후보 **Policy A**) frozen hard-reject보다
나은가?

이 감사는 **승격 여부를 결정하기 위한 근거**를 만든다. 결과가 부정적이면
Policy A를 거절하고, 거절 사유를 기록한다.

## 배경

- Frozen: `select_ppx_route`에서 prior가 탈락하면 `prior_weight=0` →
  즉시 matched direct fallback. prior+residual은 평가하지 않는다.
- Policy A: 1단계 탈락과 무관하게, 동일 validation fold에서
  prior+residual vs matched direct를 frozen τ=(2%, 60%, 1.10)로만 승인한다.

아카이브 `results/cross_domain_mechanism_v2`에는 prior-only 예측이 없어
실제 OOF prior regret을 재계산하지 않는다. 대신 **1단계가 탈락한
세계**를 강제하고, 그 세계에서 A와 frozen을 비교한다.

## 비교 정책

| 정책 | 1단계 | 2단계 |
|---|---|---|
| `frozen_s1_fail` | prior 강제 탈락 | 평가 안 함 → 항상 direct |
| `policy_A_s1_fail` | prior 탈락을 무시 | τ=(2%,60%,1.10)로 PP 승인 |
| `operational_s1_pass` | prior 통과로 고정 (참고) | 동일 τ | 현재 operational 재생 |

## 데이터

- 12-setting common-backbone archive
- 결정: validation only
- test labels: FA/FR 및 deployed log-RMSE **사후 채점만**

## 사전 등록 승격 기준

Forced S1-fail 세계에서 A가 새로 여는 승인(= stage2 accept)을
rescue라고 부른다. test에서 PP가 이득이면 true rescue, 아니면 false rescue.

### Strict (승격에 필요)

1. `n_true_rescues ≥ 1`
2. `n_false_rescues = 0`  
   (hard S1 reject 대비 **새 FA를 하나도 열지 않음**)
3. equal-dataset mean deployed log-RMSE ratio가 `frozen_s1_fail`보다 큼

### Weak (참고만; 승격 불가)

- FR↓, deployed effect↑, FA ≤ operational  
  → “잘못된 S1 탈락 시 stage2가 always-direct보다 낫다”는 **완화 근거**일 뿐,
  stage2가 이미 가진 FA를 다시 여는 것이므로 frozen hard-S1을 바꾸지 않는다.

**Strict 실패 시 Policy A 승격 거절.** Weak만 통과하면 limitation 문장으로만 남긴다.

## 비범위

- test로 threshold 재튜닝 금지
- DS03 Engression fallback 교체 금지
- Algorithm 1 운영 τ 변경 금지 (이 실험은 A 승격 여부만)
