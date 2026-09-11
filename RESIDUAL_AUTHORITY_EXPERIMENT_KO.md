# Distance-indexed Residual Authority 개발 실험

## 결론

**현재 PP-X의 최종 구조로 승격하지 않는다.**

거리별 residual authority는 validation evidence가 부족하거나 위해 위험이 큰 경우
정확히 prior-off fallback으로 돌아갔다. 이 때문에 평가한 5개 retrospective cohort
모두에서 선택된 경로가 fallback보다 나빠지지는 않았다. 그러나 4개 cohort에서
authority가 전부 거절되었고, residual을 승인한 RWTH에서도 full residual과 test
예측이 같았다. 현재 결과는 안전한 거절 동작은 확인하지만, 거리별 authority 자체의
추가 성능이나 구조적 필요성을 입증하지 못한다.

## 실험 구조

예측은 다음 세 경로를 명시적으로 분리했다.

\[
\hat y(x)=p(x)+a_k\{f_{\mathrm{PR}}(x)-p(x)\},
\qquad d(x)\in S_k .
\]

- \(p(x)\): frozen prior
- \(f_{\mathrm{PR}}(x)\): 학습된 full prior-residual candidate
- \(a_k\): 거리 shell \(S_k\)에서 residual이 prior를 수정할 수 있는 권한
- 미승인 shell: prior가 아니라 exact prior-off fallback

Primary arm에는 다음 조건을 적용했다.

- authority grid: 0, .25, .50, .75, 1
- support에서 멀어질수록 authority가 증가하지 않는 단조 제약
- mean 및 worst-20% physical-unit excess MSE가 fallback 대비 2% 이내
- leave-one-physical-unit-out authority calibration
- cross-fitted gain이 양수일 때만 승인
- 한 shell이 fallback이면 더 먼 shell도 fallback

모델 학습용 checkpoint unit과 authority calibration용 outer-validation unit을
분리했다. 모든 결과는 이미 열린 데이터에 대한 retrospective development이며
prospective confirmation이 아니다.

## 결과

| cohort | fallback R² | full residual R² | authority R² | 판정 |
|---|---:|---:|---:|---|
| Stanford | -0.2115 | -0.7415 | -0.2115 | reject → exact fallback |
| ISU 250 mAh | 0.1275 | 0.2255 | 0.1275 | unit-tail harm으로 reject |
| Sunwoda | -1.9277 | 0.9286 | -1.9277 | validation unit 2개로 calibration 불가 |
| RWTH | 0.0842 | 0.8714 | 0.8714 | approve |
| MICH | 0.5397 | 0.4699 | 0.5397 | reject → exact fallback |

RWTH의 최종 authority는 validation shell 순서대로 `(1.0, .75, .50)`이었다.
그러나 source/test rows가 첫 shell에만 들어가 실제 test 예측은 full residual과
동일했다. 따라서 이 데이터에서 test 개선을 distance modulation의 효과로 해석할
수 없다.

Sunwoda에서는 full residual이 크게 우세했지만 validation physical unit이 2개뿐이라
사전 규칙상 authority를 보정할 수 없었다. 기준을 사후에 낮춰 이득을 채택하지
않는다.

ISU에서는 pooled R²만 보면 full residual이 높았지만, physical-unit 평균 및
worst-tail harm 기준을 통과하지 못했다. authority가 성능 이득을 포기하고 fallback을
선택한 사례다.

## Ablation 해석

- global authority, unconstrained shell authority, monotone shell authority는
  Stanford·ISU·MICH에서 모두 fallback을 선택했다.
- RWTH에서는 세 authority arm이 모두 residual을 승인했으며 test 성능도 같았다.
- free-shell과 monotone-shell 결과가 달라지는 데이터가 없어 단조 제약의 필요성은
  입증되지 않았다.
- 선택된 authority는 5/5 cohort에서 fallback 대비 비열등한 방향이었지만,
  4/5가 exact fallback이므로 이를 새로운 예측 성능으로 주장할 수 없다.

## 현재 PP-X와의 비교

논문용 PP-X의 기존 battery executor 결과는 Sunwoda 0.939, RWTH 0.878,
MICH 0.751이다. 이번 authority challenger는 각각 -1.928, 0.871, 0.540으로
현재 최종 PP-X보다 우수하지 않다. 학습 분리와 seed 수가 달라 완전한 일대일
성능 비교는 아니지만, 승격을 지지할 신호가 없다는 판단에는 충분하다.

## 판정

1. **확인됨:** residual authority의 미승인 shell이 exact fallback으로 작동한다.
2. **확인됨:** prior와 residual이 함께 실패하는 Stanford·MICH에서 위해를 차단한다.
3. **미확인:** 거리별 authority가 global authority보다 낫다.
4. **미확인:** 단조 authority가 unconstrained authority보다 낫다.
5. **실패:** 현재 최종 PP-X보다 높은 예측 성능.
6. **실패:** 새로운 prospective cohort에서의 일반화 증거.

따라서 residual authority는 현 논문의 메인 구조에 추가하지 않는다. 후속 연구에서
사용하려면 validation unit이 충분하고 validation/test 거리 shell이 실제로 겹치는
새 cohort를 먼저 확보한 뒤, frozen policy로 다시 검증해야 한다.

## 재현 경로

- 구현: `src/pp_extrapolation/residual_authority.py`
- 일반 cohort 실험: `experiments/residual_authority_development.py`
- battery 실험: `experiments/residual_authority_battery_development.py`
- 단위 테스트: `tests/test_residual_authority.py`
- 일반 프로토콜: `protocols/RESIDUAL_AUTHORITY_DEVELOPMENT_PROTOCOL.md`
- battery 프로토콜: `protocols/RESIDUAL_AUTHORITY_BATTERY_DEVELOPMENT_PROTOCOL.md`
