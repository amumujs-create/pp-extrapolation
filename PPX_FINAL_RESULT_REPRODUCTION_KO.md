# PP-X Final 그림 결과 — 데이터 기반 재현

작성: 박진서

PP-X 계산 가능 후보만 연 뒤 validation MSE 최소 후보를 선택하고, 선택을 고정한 다음 test prediction을 ensemble했다. 데이터셋 이름·test label·그림 점수는 selector 입력이 아니다.

BQ는 강제 prior가 아니다. Boundary battery에서는 `BQ/affine × executor` 후보를 함께 평가했고 validation이 BQ arm을 선택했다.

그림 반올림 점수 재현 **9/9** · 양의 R² **9/9** · macro mean R² **0.807**

| Setting | Validation-selected arm/config | 재계산 pooled R² | 그림 R² |
|---------|--------------------------------|------------------:|--------:|
| HUST | `rate|10, rate|1, rate|0.1, rate|0.1, rate|10` | 0.957959 | 0.958 |
| Sunwoda | `bq_bounded` | 0.939451 | 0.939 |
| N-CMAPSS | `basic|0, moments|0, multiscale|0, multiscale|0.01, multiscale|0` | 0.937271 | 0.937 |
| Virkler | `beta=0, beta=0, beta=0, beta=0, beta=0` | 0.887977 | 0.888 |
| RWTH | `bq_bounded` | 0.878382 | 0.878 |
| MATR-b2 | `transport` | 0.862391 | 0.862 |
| MICH | `bq_dual_scale` | 0.751225 | 0.751 |
| NASA | `short|0, multiscale|0, multiscale|0, short|0.01` | 0.583751 | 0.584 |
| MATR2019 | `affine, affine, affine, affine, affine` | 0.465698 | 0.466 |

## 판정

- 9개 모두 저장된 validation argmin 기록과 일치한다.
- 9개 모두 그림의 3-decimal PP-X Final 값으로 반올림된다.
- direct fallback은 이 재현 selector의 후보가 아니다. 이것이 보수적 v3 direct-safe replay와 결과가 달랐던 핵심 이유다.
- 이 결과는 기존 개발 cohort의 retrospective reproduction이다. 새 untouched cohort의 prospective 성능을 증명하지 않는다.

재현:

```bash
PYTHONPATH=src:experiments:../ca-css-ncmapss \
  python experiments/ppx_final_result_reproduction_v1.py
```

JSON: `results/ppx_final_result_reproduction_v1/results.json`
