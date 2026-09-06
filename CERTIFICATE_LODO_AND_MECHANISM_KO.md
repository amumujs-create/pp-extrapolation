# PP certificate leave-one-domain-out와 성공/실패 메커니즘

## Leave-one-domain-out

한 도메인을 완전히 제외하고 나머지 5개 도메인에서 false accept가 없도록 threshold를 선택한 뒤 제외 도메인을 판정했다. Certificate가 의미를 가지려면 양의 증거를 요구해야 하므로 validation gain과 residual activity 후보에서 0은 허용하지 않았다.

| 제외 도메인 | 예측 | 실제 test | 판정 |
|---|---|---|---|
| HUST | 사용 가능 | 성공 | 정확 |
| Virkler | 사용 가능 | 성공 | 정확 |
| Sunwoda | 사용 가능 | 성공 | 정확 |
| RWTH | 사용 가능 | 성공 | 정확 |
| MICH | 사용 불가 | 실패 | 정확 |
| FEMTO | 사용 불가 | 실패 | 정확 |

LODO 결과는 6/6, false accept 0, false reject 0이었다. 각 fold에서 선택된 최소 증거는 validation MSE gain 0.1%, residual activity 0.01%였다. Correlation threshold는 0으로 선택되어 feature–RUL marginal correlation은 판정에 추가 정보를 주지 않았다.

표본이 6개 도메인뿐이고 모두 이미 결과가 알려진 retrospective audit이므로 100% 정확도를 일반화 성능으로 해석하면 안 된다. 이 결과는 certificate의 변수 선택을 단순화하는 근거다.

## 왜 성공했는가

| 데이터 | validation gain | residual activity | seed disagreement | 선택 epoch |
|---|---:|---:|---:|---|
| HUST | 49.8% | 50.2% | 21.7% | 17–88 |
| Virkler | 84.1% | 89.7% | 8.9% | 56–106 |
| Sunwoda | 0.6% | 1.4% | 7.4% | 0–1 |
| RWTH | 28.9% | 24.7% | 5.8% | 65–198 |

HUST·Virkler·RWTH에서는 affine path가 장거리 추세를 제공하고 NN residual이 validation에서 반복 가능한 국소 편차를 학습했다. Seed disagreement도 residual activity보다 작아 correction의 공통 방향이 존재했다. Sunwoda는 경계 사례로, correction은 작지만 test에서 PP가 양의 R²를 유지했다.

## 왜 실패했는가

| 데이터 | validation gain | residual activity | 선택 epoch | test R² |
|---|---:|---:|---|---:|
| MICH | ≈0 | ≈0 | 모든 seed 0 | -1.522 |
| FEMTO | ≈0 | 0 | 모든 seed 0 | -1.378 |

두 데이터 모두 marginal feature–RUL Spearman은 높았지만 이것은 한 trajectory 내부의 시간 순서를 반영할 수 있다. 새로운 unit에서 같은 mapping이 유지된다는 증거는 아니다. 실제로 nonlinear correction은 validation 오차를 줄이지 못했고 early stopping이 모든 seed에서 초기 affine 상태를 선택했다.

- MICH 결과는 cohort 사이 health–RUL mapping 변화와 일치한다. 학습 residual이 source cohort로 전이되지 않았다.
- FEMTO는 완전수명 Learning bearing이 6개뿐이고 vibration summary가 bearing별 수명 차이를 설명하지 못했다. 공식 endpoint의 90.9%가 train feature hull 내부여서 strict-hull 문제도 아니었다.

가장 정확한 메커니즘 해석은 `강한 상관이 있으면 성공`이 아니라 다음과 같다.

> PP succeeds when the affine path captures the transferable tail trend and a nonzero nonlinear residual repeatedly improves held-out-unit validation; it abstains when that residual evidence collapses.

따라서 최종 certificate는 correlation보다 validation gain, residual activity, seed stability를 중심으로 두는 편이 타당하다.
