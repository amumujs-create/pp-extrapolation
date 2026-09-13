# DS03 prior-off fallback — unit-risk validation 재설계

작성자: 박진서  
상태: development audit (Algorithm 1 prior-on core 미변경)

## 목표

prior가 거절된 DS03에서 fallback을 고를 때, pooled validation MSE 대신
**unit-risk 계열 규칙**을 쓰면 Engression/FT가 validation에서 승인되는지 본다.

## 후보

`{PP-X direct_fallback, plain_mlp, Engression, FT-Transformer, GroupDRO, V-REx, monotone, RBF, SVGP}`

각 후보는 frozen equal-budget selected config를 seeds 42–46으로 재적합하고
validation 예측만으로 통계를 낸다. test R²는 선택 후에만 읽는다.

## 규칙

1. `pooled_mse` — 이전 baseline
2. `unit_macro_rmse` — unit 평균 RMSE 최소
3. `worst_unit_rmse` — 최악 unit RMSE 최소 (maximin)
4. `paper_unit_gate_vs_direct` — PP-X 논문 게이트(≥2% MSE, win≥0.6, worst ratio≤1.1)
5. `lexicographic_unit_risk` — worst unit → win fraction → unit-macro → MSE
6. `unit_gain_ci_then_mse` — baseline 대비 unit-gain CI_low ≥ 0인 것만 MSE 최소

## 성공 기준

- 어떤 unit-risk 규칙이 Engression/FT를 고르고 test에서도 PP-X direct보다
  개선되면, prior-off fallback 재설계 가설 지지.
- 모든 규칙이 Engression을 탈락시키거나 test 개선이 없으면, unit-risk만으로는
  DS03 약점이 자동 해결되지 않는다고 기록한다.
