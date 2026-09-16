# PP-X Main Model Ablation — A0 / A1 / A2 / A3

> **Matching audit:** A1→A2 correction 비교와 A2→A3 executor 비교는 동일 setting·seed·저장 예측 내의 nested comparison이다. A0→A2의 A0는 Battery 3개에서만 architecture/예산을 맞춘 direct NN이며, HUST·MATR-b2·N-CMAPSS는 저장된 GroupDRO direct control이다. 따라서 6-setting A0→A2를 전체적으로 `matched-capacity prior effect`라고 주장하지 않는다.

Selector-policy, hard-gate, tail-risk, mixture experiments are excluded. All comparisons use the same six stored row-aligned five-seed settings.

| Setting | A0 Direct NN | A1 Prior-only | A2 Prior+Residual | A3 Full PP-X |
|---|---:|---:|---:|---:|
| Sunwoda | -1.352 | 0.281 | 0.939 | 0.939 |
| RWTH | 0.633 | 0.659 | 0.878 | 0.878 |
| MICH | 0.684 | -3.343 | 0.468 | 0.751 |
| HUST | 0.934 | 0.793 | 0.829 | 0.958 |
| MATR batch 2 | 0.691 | -1.279 | 0.675 | 0.862 |
| N-CMAPSS | 0.838 | 0.807 | 0.922 | 0.926 |

## A0→A2 Prior-conditioning effect

- normalized mean unit RMSE reduction: +0.1795
- setting-bootstrap 95% CI: [-0.2322, +0.5834]
- positive/negative/neutral settings: 4/2/0
- individually significant improvements after within-contrast BH: 2/6

- Sunwoda: ΔR²=+2.291, unit CI=[+0.656,+0.771], q=0.0117
- RWTH: ΔR²=+0.245, unit CI=[+0.147,+0.562], q=0.0352
- MICH: ΔR²=-0.216, unit CI=[-0.053,-0.021], q=0.0312
- HUST: ΔR²=-0.104, unit CI=[-38.677,-11.212], q=0.0117
- MATR batch 2: ΔR²=-0.015, unit CI=[-1.295,+3.821], q=0.371
- N-CMAPSS: ΔR²=+0.084, unit CI=[+1.646,+9.655], q=0.3

## A1→A2 Correction effect

- normalized mean unit RMSE reduction: +0.5117
- setting-bootstrap 95% CI: [+0.3162, +0.6757]
- positive/negative/neutral settings: 6/0/0
- individually significant improvements after within-contrast BH: 4/6

- Sunwoda: ΔR²=+0.659, unit CI=[+0.265,+0.402], q=0.0117
- RWTH: ΔR²=+0.220, unit CI=[+0.222,+0.394], q=0.0117
- MICH: ΔR²=+3.811, unit CI=[+0.248,+0.344], q=0.0117
- HUST: ΔR²=+0.036, unit CI=[-2.238,+17.015], q=0.207
- MATR batch 2: ΔR²=+1.954, unit CI=[+28.014,+42.404], q=0.0117
- N-CMAPSS: ΔR²=+0.115, unit CI=[+2.711,+5.167], q=0.25

## A2→A3 Executor/routing effect

- normalized mean unit RMSE reduction: +0.2148
- setting-bootstrap 95% CI: [+0.0651, +0.3774]
- positive/negative/neutral settings: 4/0/2
- individually significant improvements after within-contrast BH: 3/6

- Sunwoda: ΔR²=+0.000, unit CI=[+0.000,+0.000], q=1
- RWTH: ΔR²=+0.000, unit CI=[+0.000,+0.000], q=1
- MICH: ΔR²=+0.283, unit CI=[+0.029,+0.087], q=0.0312
- HUST: ΔR²=+0.128, unit CI=[+21.606,+40.534], q=0.000732
- MATR batch 2: ΔR²=+0.187, unit CI=[+5.543,+9.338], q=0.0117
- N-CMAPSS: ΔR²=+0.004, unit CI=[+0.052,+0.310], q=0.375

## Main interpretation

- A0→A2: prior conditioning is heterogeneous; it is not a universal-improvement claim.
- A1→A2: learned nonlinear correction is required and is the strongest component result.
- A2→A3: typed executors add value selectively; unchanged settings remain at the core route.
- Direct NN remains an ablation control and is not a PP-X route candidate.

Figure: `figures/paper/fig_ppx_main_model_ablation_a0_a3.png`
JSON: `results/ppx_main_model_ablation_a0_a3_v1/results.json`
