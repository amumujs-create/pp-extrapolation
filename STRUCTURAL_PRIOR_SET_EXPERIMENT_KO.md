# 이질적 구조 prior-set 개발 실험

## 결론

현재 구현은 **PP-X 최종 모델로 승격하지 않는다**.

이번 후보는 같은 PP-X의 trust 변형이 아니라 train-only affine ridge,
monotone health, history-rate 세 개의 서로 다른 구조 가정이다. 후보 간
성능 차이가 크고 disagreement도 실질적이었다. 그러나 physical-unit
cross-fit 승인 규칙은 Stanford 8/8, ISU 45/45 fold에서 집합을 비웠고,
최종 예측은 100% prior-off fallback이었다.

ISU test에서 affine 단독과 세 prior 평균은 fallback보다 높았다. 이
수치는 validation에서 재현되지 않았으므로 승인 근거가 아니다.

## 무엇을 시험했나

- fallback: 이미 저장된 PP-X v1.1 trust-0 portfolio
- affine: train-only group-weighted ridge
- monotone: health에 대한 isotonic map + 낮은 health 쪽 비음 선형 tail
- history-rate: \(y \approx a + b \cdot h / \max(-\dot h, \varepsilon)\)
- prior별 거리와 validation 99% support ceiling
- 멀어질수록 집합이 줄어드는 nested 승인
- singleton 허용 arm과 두 개 이상 consensus arm
- 모든 validation unit은 자신을 제외한 정책으로 평가

프로토콜은 실행 전에
`protocols/STRUCTURAL_PRIOR_SET_DEVELOPMENT_PROTOCOL.md`로 동결했다.

## 결과

### Stanford

| arm | pooled R² | RMSE | unit-macro R² |
|---|---:|---:|---:|
| fallback | 0.0714 | 81.4444 | 0.2391 |
| affine | -0.1579 | 90.9485 | -0.1732 |
| monotone | -0.1683 | 91.3537 | -0.1555 |
| history | -6.7706 | 235.6039 | -9.3380 |
| mean ensemble | -0.2318 | 93.8059 | -0.5551 |
| nested / consensus | 0.0714 | 81.4444 | 0.2391 |

validation group-MSE는 fallback 1620.4에 대해 affine 4339, monotone 4147,
history 71805였다. 세 구조 prior 모두 source 밖 구간에서 fallback보다
나빴고, 8/8 fold가 집합을 비웠다. test에서도 affine은 9 unit 중 3개만
이겼다.

### ISU 250 mAh

| arm | pooled R² | RMSE | unit-macro R² |
|---|---:|---:|---:|
| fallback | 0.4798 | 1.6712 | -1.8509 |
| affine | 0.6635 | 1.3442 | -0.5486 |
| monotone | -0.0973 | 2.4272 | -0.6584 |
| history | -0.4796 | 2.8185 | -8.0765 |
| mean ensemble | 0.7063 | 1.2557 | -0.1958 |
| continuous portfolio | 0.4917 | 1.6520 | -1.6828 |
| nested / consensus | 0.4798 | 1.6712 | -1.8509 |

affine은 test에서 46 unit 중 41개를 이겼고 pooled RMSE를 약 19.6%
낮췄다. 그러나 validation group-MSE는 fallback 2.934에 대해 affine 3.234로
오히려 0.300 높았다. 따라서 leave-one-unit-out 규칙은 45/45 fold에서
affine을 거절했다.

mean ensemble의 test R² 0.706은 monotone·history가 각각 실패한 뒤에 생긴
사후 평균이다. validation에서 세 prior 모두 fallback보다 나빠 승인되지
않았다.

## 해석

이전 trust-set 실험과 달리, 이번 후보는 실제로 다른 가정을 구현한다.
history-rate는 두 데이터 모두에서 붕괴했고, monotone은 Stanford에서
거의 이득이 없었으며, affine만 ISU test에서 강했다. 즉 “여러 prior를
거리별로 섞으면 일반 외삽이 자동으로 버틴다”는 가설은 이 설계에서도
지지되지 않았다.

ISU affine의 test 우위는 무시할 수 없는 신호다. 다만 같은 후보가
validation에서 이미 평균 unit loss를 키웠다. 현재 PP-X 논문의 승인
규칙을 유지하면 이 이득은 쓸 수 없다. 규칙을 풀어 test 수치를 취하면
바로 post-hoc 공격 대상이 된다.

support ceiling과 prior별 거리는 구현상 동작했다. 승인 집합이 비어
실제 ceiling 차단이 test 예측을 바꾼 경우는 없었다.

## 논문 판단

- 현재 PP-X 정의와 주장은 변경하지 않는다.
- 이 실험을 main positive result나 novelty 근거로 사용하지 않는다.
- “이질적 prior consensus가 검증됐다”고 쓰지 않는다.
- ISU affine test 우위를 선택정책의 성공으로 쓰지 않는다.
- 필요하면 supplement의 negative development result로 공개할 수 있다.

후속에서 affine만 다시 보려면, 지금처럼 전체 validation을 한 번에
보지 말고 train-only prior + 완전 nested unit-cross-fit + 미개봉
cohort를 따로 고정해야 한다. 그때도 현재 게이트를 사후에 완화하지
않는다.
