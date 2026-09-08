# Deterministic RBF-regime PP 순차 확증 v3 결과

사전 동결 commit `44e7baf`이 난 뒤 BatteryLife 공식 Zn-ion test 목록의 나머지 8개 후보를 모두 받아 한 번 평가했다. 모델, eligibility, 성공 기준은 실행 전에 고정했다.

## Eligibility

- train, validation, 기존 test와 정확히 중복된 원본: 0개
- 오른쪽 중단: 0개
- boundary-to-EOL tail 50 step 미만: 5개
- unique eligible cell: **3개**

따라서 사전 최소 표본 조건 `n >= 2`를 통과했다.

## 주 결과

| 모델 | pooled R² | RMSE | MAE | unit-macro R² |
|---|---:|---:|---:|---:|
| single-seed plain MLP | -0.123 | 244.920 | 189.889 | -1.818 |
| deterministic RBF-regime PP | **0.558** | **153.577** | **130.468** | **-0.309** |

사전 기준인 `PP pooled R² > 0` 및 `PP pooled R² > MLP pooled R²`를 모두 만족했다. 따라서 **이 실험은 사전 정의상 순차 untouched confirmation 성공**이다.

## 셀별 결과

| cell | tail n | PP R² | MLP R² | 상대 승자 |
|---|---:|---:|---:|---|
| `209-1` | 67 | **0.367** | -0.912 | PP |
| `412-1` | 58 | **-1.694** | -4.014 | PP |
| `446-1` | 741 | **0.399** | -0.528 | PP |

PP는 3개 셀 모두에서 MLP보다 높았지만 `412-1`에서는 PP도 음의 R²다. 또한 pooled 지표는 tail 741행인 `446-1`의 비중이 크고 unit-macro R²는 아직 -0.309다. 따라서 `모든 셀에서 절대 성공`이나 `일반적으로 양의 macro R²`는 주장하지 않는다.

## 의미와 한계

이 결과는 이전에 비어 있던 `사전에 고정한 개선 PP가 unique untouched cell에서 성공한 사례`를 처음 제공한다. 특히 ensemble 없이 결정론적 단일 모델로 pooled R²를 양수로 만들고 MLP를 3/3 cell에서 넘었다.

그러나 독립 unit이 3개뿐이고 모두 같은 Zn-ion 데이터셋에서 나온 순차 held-out cell이다. 이 결과만으로 화학계 간 일반화나 통계적 유의성을 주장하지 않는다. 논문에서는 pooled 확증 성공과 negative macro 한계를 같이 보고해야 한다.
