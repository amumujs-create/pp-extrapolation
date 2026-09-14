# 최종 PP-X 구성요소 ablation 및 통계검정

PP-X Final prior는 `known_boundary`만 본다. 경계 있으면 BQ, 없으면 affine. 아래 표는 그 prior family 안의 executor/core on/off다.
동일 test row와 seeds 42–46의 저장 예측을 재집계했다. 주 지표는 prediction-ensemble pooled R²이며, 통계 표본은 독립 물리 unit이다. 양의 RMSE 감소는 구성요소가 유리하다는 뜻이다.

| 데이터셋 | 구성요소(on) | off R² | on R² | ΔR² | unit 승 | 평균 unit RMSE 감소 [95% CI] | sign-flip p / BH q | seed p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Sunwoda | nonlinear residual | 0.281 | **0.939** | +0.659 | 9/9 | +0.338 [+0.266, +0.402] | 0.003906 / 0.01875 | 0.0625 |
| Sunwoda | frozen affine path | 0.900 | **0.939** | +0.039 | 7/9 | +0.038 [-0.020, +0.091] | 0.2422 / 0.2857 | 0.4375 |
| Sunwoda | fixed residual bound | 0.718 | **0.939** | +0.221 | 7/9 | +0.153 [+0.074, +0.225] | 0.01562 / 0.03125 | 0.0625 |
| Sunwoda | support-adaptive dual scale | 0.939 | **0.934** | -0.005 | 5/9 | -0.005 [-0.035, +0.021] | 0.7578 / 0.7578 | 0.25 |
| Sunwoda | complete selected PP-X vs direct NN | -1.352 | **0.939** | +2.291 | 9/9 | +0.718 [+0.655, +0.772] | 0.003906 / 0.01875 | 0.0625 |
| RWTH | nonlinear residual | 0.659 | **0.878** | +0.220 | 8/8 | +0.312 [+0.220, +0.394] | 0.007812 / 0.02083 | 0.0625 |
| RWTH | frozen affine path | 0.855 | **0.878** | +0.024 | 7/8 | +0.078 [-0.071, +0.184] | 0.3047 / 0.3324 | 0.125 |
| RWTH | fixed residual bound | 0.788 | **0.878** | +0.090 | 6/8 | +0.177 [+0.013, +0.332] | 0.09375 / 0.15 | 0.0625 |
| RWTH | support-adaptive dual scale | 0.878 | **0.842** | -0.037 | 0/8 | -0.062 [-0.084, -0.042] | 0.007812 / 0.02083 | 0.0625 |
| RWTH | complete selected PP-X vs direct NN | 0.633 | **0.878** | +0.245 | 7/8 | +0.381 [+0.148, +0.558] | 0.02344 / 0.04018 | 0.0625 |
| MICH | nonlinear residual | -3.343 | **0.468** | +3.811 | 8/8 | +0.305 [+0.248, +0.344] | 0.007812 / 0.02083 | 0.0625 |
| MICH | frozen affine path | 0.319 | **0.468** | +0.149 | 6/8 | +0.023 [-0.008, +0.054] | 0.2109 / 0.2664 | 0.0625 |
| MICH | fixed residual bound | 0.759 | **0.468** | -0.291 | 1/8 | -0.054 [-0.074, -0.031] | 0.01562 / 0.03125 | 0.5625 |
| MICH | support-adaptive dual scale | 0.468 | **0.751** | +0.283 | 7/8 | +0.058 [+0.029, +0.087] | 0.01562 / 0.03125 | 0.0625 |
| MICH | complete selected PP-X vs direct NN | 0.684 | **0.751** | +0.067 | 6/8 | +0.020 [-0.001, +0.040] | 0.1172 / 0.1758 | 0.125 |
| Sunwoda | full causal rate history vs current margin | 0.590 | **0.939** | +0.350 | 9/9 | +0.216 [+0.154, +0.263] | 0.003906 / 0.01875 | 0.0625 |
| RWTH | full causal rate history vs current margin | -0.378 | **0.878** | +1.256 | 8/8 | +1.143 [+1.036, +1.243] | 0.007812 / 0.02083 | 0.0625 |
| MICH | full causal rate history vs current margin | 0.672 | **0.468** | -0.204 | 1/8 | -0.032 [-0.051, -0.015] | 0.02344 / 0.04018 | 0.0625 |
| HUST | regime transport | 0.829 | **0.958** | +0.128 | 15/16 | +31.507 [+21.725, +40.574] | 0.0001221 / 0.00293 | 0.0625 |
| MATR batch 2 | support-distance decay (transport off) | 0.674 | **0.675** | +0.002 | 2/9 | -1.023 [-2.361, +0.222] | 0.1992 / 0.2656 | 0.5 |
| MATR batch 2 | decay x transport interaction | 0.820 | **0.862** | +0.042 | 7/9 | +0.568 [-0.055, +1.479] | 0.1992 / 0.2656 | 0.3125 |
| MATR batch 2 | regime transport after support decay | 0.675 | **0.862** | +0.187 | 9/9 | +7.314 [+5.544, +9.338] | 0.003906 / 0.01875 | 0.0625 |
| NASA battery | validation-selected causal history | 0.572 | **0.584** | +0.012 | 2/4 | +0.057 [-0.139, +0.311] | 0.75 / 0.7578 | 0.0625 |
| N-CMAPSS | validation-selected multiscale history | 0.928 | **0.937** | +0.009 | 3/3 | +0.825 [+0.311, +1.814] | 0.25 / 0.2857 | 0.25 |

## 판정 기준

- ΔR²와 unit 평균 효과가 모두 양수이고 unit-bootstrap CI의 하한이 0보다 크면 강한 표본 내 근거로 본다.
- CI가 0을 포함하면 개선 방향은 보여도 모집단 수준의 유의성은 확정하지 않는다.
- 5-seed exact sign-flip 검정은 완전한 5/5 동일 방향이어도 양측 p의 최솟값이 0.0625이므로 seed p<0.05를 요구하지 않는다.

## 보관 `v1_declared` (Final 아님)

FEMTO prior abstention은 OOF/group 사다리 재생이다. Final 9-setting과 Final BH 표에 넣지 않는다.

| 데이터셋 | 구성요소(on) | off R² | on R² | ΔR² | unit 승 | 평균 unit RMSE 감소 [95% CI] | sign-flip p / BH q | seed p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| FEMTO | v1_declared prior abstention | -1.347 | **0.075** | +1.422 | 7/11 | +1361.247 [-676.630, +3276.055] | 0.2324 / — | 0.3125 |

## 해석상 제한

NASA와 N-CMAPSS의 제거 arm도 같은 seeds로 재학습해 포함했다. 두 효과는 각각 약 +0.012, +0.009로 작고 unit 수도 4개와 3개뿐이므로, 방향성 ablation으로 보고 모집단 유의성을 주장하지 않는다.
