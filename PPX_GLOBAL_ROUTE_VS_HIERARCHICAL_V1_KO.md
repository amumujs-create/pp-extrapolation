# Global Fixed Route vs PP-X v4 Hierarchical Router

동일한 6개 setting과 저장된 5-seed 예측을 사용했다. 고정 executor가 contract상
계산 불가능한 setting에서는 matched prior+residual core를 사용했다.

| Global policy | Mean R² | Worst R² | Router W/T/L | Router normalized unit-RMSE gain |
|---|---:|---:|---:|---:|
| core | 0.785 | 0.468 | 4/2/0 | +0.215 |
| bounded | 0.785 | 0.468 | 4/2/0 | +0.215 |
| dual_scale | 0.826 | 0.675 | 5/1/0 | +0.181 |
| transport | 0.838 | 0.468 | 2/4/0 | +0.065 |
| history | 0.786 | 0.468 | 3/3/0 | +0.209 |

Hierarchical router mean R²: **0.886**

Hierarchical router worst-setting R²: **0.751**

## Setting-level ΔR² (router - fixed)

- **core**: Sunwoda=+0.000, RWTH=+0.000, MICH=+0.283, HUST=+0.128, MATR batch 2=+0.187, N-CMAPSS=+0.004
- **bounded**: Sunwoda=+0.000, RWTH=+0.000, MICH=+0.283, HUST=+0.128, MATR batch 2=+0.187, N-CMAPSS=+0.004
- **dual_scale**: Sunwoda=+0.005, RWTH=+0.037, MICH=+0.000, HUST=+0.128, MATR batch 2=+0.187, N-CMAPSS=+0.004
- **transport**: Sunwoda=+0.000, RWTH=+0.000, MICH=+0.283, HUST=+0.000, MATR batch 2=+0.000, N-CMAPSS=+0.004
- **history**: Sunwoda=+0.000, RWTH=+0.000, MICH=+0.283, HUST=+0.128, MATR batch 2=+0.187, N-CMAPSS=+0.000

## Interpretation

- 한 executor를 모든 setting에 고정하면 해당 executor가 필요한 setting만 개선하고 나머지는 core에 머문다.
- 계층 router는 admissible family 안에서 setting별 validation evidence로 서로 다른 executor를 선택한다.
- 이 결과는 retrospective stored-prediction comparison이며 새 prospective cohort 검증은 아니다.
