# Zn-ion BQ-PP untouched 확증 결과

## 결론

Na-ion 실패 후 개발한 normalized Boundary-Quotient PP를 공식 BatteryLife Zn-ion 분할에서 동결하고 한 번 평가했다. **외부 화학계 확증은 실패했다.** 따라서 현재 BQ-PP가 화학계를 넘어 일반화된다고 주장할 수 없다.

| 모델 | Untouched pooled R² | RMSE | MAE | unit-macro R² |
|---|---:|---:|---:|---:|
| plain MLP (5-seed ensemble) | **-0.204** | **239.02** | **179.03** | -0.236 |
| boundary affine | -0.379 | 255.85 | 202.10 | -0.005 |
| normalized BQ-PP (5-seed ensemble) | -0.379 | 255.83 | 202.07 | **-0.004** |

Test 3개를 고정했지만 `422-1`은 EOL이 train에서 고정한 152-cycle 경계보다 빨라 late-tail 평가 행이 없었다. 이를 사후에 다른 셀로 교체하지 않았다. 수치는 나머지 2개 셀의 965개 late-tail 관측치에서 계산했다.

## 셀별 결과가 밝힌 실패 구조

| test cell | EOL 후 late-tail n | BQ-PP R² | plain MLP R² | 우세 |
|---|---:|---:|---:|---|
| `435-1` | 227 | **0.873** | 0.150 | BQ-PP |
| `438-3` | 738 | -0.882 | **-0.621** | MLP |

BQ-PP의 5개 seed는 거의 같은 결과를 냈다. 따라서 이 실패는 초기화 분산이 아니라 구조적 bias다. `435-1`에서는 80%-EOL boundary factorization이 정확했지만, 장수명 `438-3`의 시간척도를 train prefix에서 식별하지 못했다. Validation에서도 EOL이 약 182와 1019 cycle로 나뉘어 같은 지정 SOH margin이 같은 RUL 시간척도를 뜻하지 않았다.

## 정적 gate 개선에 대한 결론

이 결과는 기존 static prefix descriptor만 정교하게 튜닝해서는 해결되지 않는다. 현재 health, slope, curvature, noise는 관측된 prefix의 모양만 알려주고, 그 뒤의 속도 전환과 cell별 latent lifetime scale을 구분하지 못한다.

다음 모델은 BQ를 버리기보다 다음 두 변수를 분리해야 한다.

1. `RUL = boundary margin × latent lifetime scale × local quotient`
2. 여러 prefix window의 기울기 변화로 regime transition hazard를 추정

latent scale은 학습 셀의 수명 regime을 혼합한 posterior로 추정하고, regime hazard가 높으면 NN residual과 시간척도의 불확실성을 함께 키워야 한다. 이것이 정적 승인/거절 gate보다 현재 실패에 맞는 동적 확장이다.

## 증거 상태

- 프로토콜·코드·validation 결과 동결 commit: `36d95be4d4057524ee72332f7980cafa733c21ee`
- 그 후 고정한 test xlsx 3개를 다운로드하고 `--phase test`를 한 번 실행했다.
- 원본 결과: `results/znion_bq_confirmatory_v1/test.json`
- 예측 배열: `results/znion_bq_confirmatory_v1/test_predictions.npz`

셀이 2개뿐이므로 이 결과로 통계적 유의성을 주장하지 않는다. 현재 증거는 BQ-PP의 성공 범위와 장수명 regime 실패를 동시에 특정한 반증 자료다.
