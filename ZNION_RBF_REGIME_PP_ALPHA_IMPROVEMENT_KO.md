# RBF-regime PP affine shrinkage 개선

## 문제

순차 v3의 pooled R²는 0.558로 성공했지만 unit-macro R²는 -0.309였고 `412-1`은 -1.694였다. 원인은 latent regime path보다 음의 slope에서 사용하는 BQ affine quotient의 변동이었다.

## Test-independent 선택

BQ ridge alpha만 `{0.1, 1, 10, 100, 1000, 3000, 10000, 30000}`에서 비교하고 기존 development validation pooled R²가 가장 큰 값을 선택했다. v3 test 점수는 선택 목적함수에 사용하지 않았다.

| alpha | validation pooled R² |
|---:|---:|
| 0.1 | -0.604 |
| 1 | -0.599 |
| 10 | -0.559 |
| 100 | -0.312 |
| **1000** | **0.181** |
| 3000 | 0.036 |
| 10000 | -0.193 |
| 30000 | -0.297 |

즉 validation이 독립적으로 강한 affine shrinkage를 선택했다. 탐색 상한을 30000까지 늘려도 1000 뒤에서 점수가 다시 낮아져 경계 최적이 아니었다. residual bound, RBF memory, temperature, gate, seed는 바꾸지 않았다.

## 재생 결과

| cohort | pooled R² | unit-macro R² |
|---|---:|---:|
| 이미 본 development Zn-ion | 0.790 | 0.835 |
| v2 unique eligible `205-3` | 0.803 | 0.803 |
| v3 unique eligible 3 cells | **0.822** | **0.375** |

v3 셀별 R²:

- `209-1`: **0.772**
- `412-1`: **-0.405**
- `446-1`: **0.757**

기존 v3과 비교하면 pooled R²는 `0.558 → 0.822`, macro R²는 `-0.309 → 0.375`, `412-1`은 `-1.694 → -0.405`로 회복했다. v2는 `0.959 → 0.803`으로 낮아졌지만 여전히 양수이고 plain MLP 0.057보다 크다.

## 해석

강한 ridge는 boundary quotient의 데이터 적합 계수를 작게 만들어 외삽 시 affine prior에 더 가깝게 유지한다. 즉 성능 개선은 test 셀별 gate를 조정한 것이 아니라, validation이 지지한 **prior-dominant extrapolation shrinkage**에서 나왔다.

다만 이 설정은 v3 결과를 확인한 뒤 만든 개선 버전이다. selection rule은 test-independent이지만 0.822를 새로운 untouched 확증 결과로 재분류하지 않는다. 차기 데이터셋에서 alpha-selection 규칙 전체를 동결해야 한다.
