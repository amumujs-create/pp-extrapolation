# PP-X v1.1 강건 일반화 개발

기록: 박진서. 2026-09-10.

## 결론

PP-X v1.0의 prior를 더 복잡하게 만들지 않고, **prior가 검증에서 명확히
이길 때만 사용하고 아니면 동일 네트워크의 matched MLP 경로로 정확히
후퇴**하도록 일반화 정책을 강화했다.

이 변경의 목표는 모든 데이터에서 PP 점수를 높이는 것이 아니다. prior가 맞는
외삽에서는 PP의 이득을 허용하고, prior가 틀린 포메이션·수명척도·중기 구간에서는
PP가 MLP보다 더 크게 망가지는 것을 방지하는 것이다.

## 단일 모형

`direct_residual_mixture=True`, `residual_seed_replay=True`인 PP 네트워크에서
affine trust를 `(0, .02, .05, .1, .2, .4)` 중 선택한다.

- trust `0`: 같은 seed, 초기화, optimizer의 matched MLP와 정확히 같은 하위모형
- trust `>0`: affine prior를 일부 사용
- 별도 모델의 test 예측을 보고 혼합하지 않음

## 보수적 승인 규칙

train/validation만 사용한다. trust가 0보다 큰 후보는 다음을 모두 만족해야 한다.

1. matched MLP 대비 validation pooled RMSE를 상대 **2% 이상** 개선
2. validation physical-unit bootstrap에서 `MLP RMSE − candidate RMSE`의
   **95% CI 하한이 0보다 큼**
3. validation 독립 유닛이 최소 3개

하나라도 실패하면 trust=0이다. test label, test batch 통계, test unit ID는
선택에 사용하지 않는다.

구현:

- `src/pp_extrapolation/generalization_policy.py`
- `src/pp_extrapolation/presets.py::robust_generalization_policy_config`
- `tests/test_generalization_policy.py`

## Stanford 개발 재평가

Stanford v1.0 결과를 본 뒤 만든 규칙이므로 **새 확증이 아니라 post-test
development**다. split, 25백분위수 경계, 입력, seed는 바꾸지 않았다.

| 항목 | PP-X v1.0 | PP-X v1.1 safety |
|---|---:|---:|
| 선택 prior trust | 기존 affine core | **0 (prior 거절)** |
| validation 최선 prior 상대 RMSE 이득 | — | **0.13%** |
| test pooled R² | −0.228 | **0.087** |
| test RMSE | 93.65 | **80.77** |
| test MAE | 72.50 | **59.98** |
| matched MLP와 최대 예측 오차 | 해당 없음 | **0.0** |

2% 승인 문턱을 통과하지 못해 prior를 거절했다. 선택된 출력은 matched MLP와
정확히 같았다. 따라서 Stanford에서 PP의 추가 손실을 제거했지만, 이는
Stanford를 본 뒤의 복구이며 일반화 확증이 아니다.

결과:

- `results/stanford_ppx_v11_safety_development/results.json`
- `results/stanford_ppx_v11_safety_development/predictions.npz`
- `experiments/stanford_ppx_v11_safety.py`

## 기존 v1.0과의 관계

- v1.0 주표 9셋과 점수는 변경하지 않는다.
- v1.1은 v1.0 결과를 소급해 대체하지 않는다.
- 기존 성공 고호트에서도 같은 validation 승인 규칙을 replay해 prior 유지율과
  성능 보존을 확인해야 한다.
- 최종 일반화 주장은 아직 불가하다. 다음 미개봉 고호트에서 규칙을 바꾸지 않고
  한 번 평가해야 한다.

## 남은 위험

validation과 test의 regime이 다르면 보수적 validation도 잘못 승인할 수 있다.
반대로 유닛 수가 작으면 유효한 prior도 거절한다. 이 정책은 PP 승률을 최대화하는
게이트가 아니라, **지원되지 않은 prior의 위해를 줄이는 안전 정책**이다.
