# PP-X Prior-Gate Softening (Policy A) 승격 감사 결과

작성자: 박진서  
프로토콜: `protocols/PPX_PRIOR_GATE_SOFTENING_A_PROTOCOL.md`  
결과: `results/ppx_prior_gate_softening_a_v1/results.json`  
상태: **Policy A 승격 거절** (strict 실패 / weak만 성립)

## 한 줄 결론

1단계 prior 탈락을 풀어 prior+residual을 2단계에 올리는 Policy A는,
잘못된 S1 차단 대비 always-direct보다 낫지만(**weak**), stage2가 이미 가진
false accept(femto, milling)를 다시 열기 때문에 **strict 승격 기준을 통과하지
못한다.** frozen hard stage-1을 유지한다.

## 실험 설계

- 아카이브: 12-setting common-backbone (`cross_domain_mechanism_v2`)
- prior-only 예측이 없어 OOF prior regret은 재계산하지 않음
- 대신 **S1 강제 탈락** 세계에서 비교
  - `frozen_s1_fail`: 항상 matched direct
  - `policy_A_s1_fail`: τ=(2%, 60%, 1.10) stage2만으로 PP 승인
- test는 FA/FR·deployed effect 사후 채점만

## 요약 지표

| 정책 | accept | correct | FA | FR | deployed mean log-RMSE ratio |
|---|---:|---:|---:|---:|---:|
| frozen_s1_fail | 0 | 6 | 0 | 6 | 0.000 |
| policy_A_s1_fail | 6 | 8 | 2 | 2 | +0.041 |
| operational_s1_pass (참고) | 6 | 8 | 2 | 2 | +0.041 |

## Rescue 분해 (A가 hard-S1 대비 새로 연 승인)

- true rescue (4): virkler, matr_batch2, nasa_battery, ncmapss
- false rescue (2): femto, milling

## 승격 판정

| 기준 | 결과 |
|---|---|
| Strict: true rescue ≥ 1 | 통과 |
| Strict: false rescue = 0 | **실패** (2) |
| Strict: deployed effect ↑ | 통과 |
| Weak: FR↓ / FA≤operational / effect↑ | 통과 |

**결정:** `reject_policy_A_keep_frozen_hard_s1`

## 해석

- “prior만 보고 막으면 residual 결합 이득을 놓친다”는 우려는 **일부 맞다**:
  S1이 잘못 탈락한 세계에서는 A가 FR 6→2, deployed effect 0→+0.041로
  always-direct를 이긴다.
- 그러나 A는 새로운 안전장치라기보다 **기존 stage2를 다시 켜는 것**이다.
  stage2 false accept(femto, milling)가 그대로 false rescue가 된다.
- 따라서 “1단계를 풀면 구조가 개선된다”고 승격할 근거는 없고,
  hard S1은 **unsupported prior 강제 금지**용으로 유지한다.
- S1이 실제로 틀렸을 때의 완화는 별도(약한 prior trust 상한 등)로
  설계해야 하며, 단순 stage1 skip은 채택하지 않는다.

## 한계

- Retrospective counterfactual이며 prospective 보장이 아니다.
- prior-only OOF regret을 아카이브에서 직접 재현하지 못했다.
- operational 경로(S1 pass)와 A(S1 fail+stage2)의 수치 동일은
  설계상 예상된 결과다.
