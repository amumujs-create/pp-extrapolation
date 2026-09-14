# PP-X 게이트 분기 정리 (발표 7–10장)

작성자: 박진서  
대응 PPT: `ppt/PP-X_Research_Detailed_v2.pptx` 발표 순서 7–10장  
코드: `transferability_gate.select_ppx_route`, `paper_ppx.select_paper_ppx`

## 7장 — 전체도

```
C1 Contract → Prior gate → C2 Fit → Executor gate → C4 Freeze
                 │                         │
            Prior OFF                 Val FAIL
                 └──────────┬──────────────┘
                            ▼
                     Direct fallback
```

- **Prior gate (Final)**: 경계 → BQ, 없음 → affine. 둘 다 Prior ON. 거절 없음.  
- **Executor gate 거절**: 비교는 했으나 Val 미달 → fallback  
- **승인**: Val PASS 후보 중 최소 loss (동점이면 단순) → freeze  
- **Prior OFF / neural_safety**: `v1_declared`만. Final 9-setting에 없음.

## 8장 — Prior gate (`select_ppx_route`)

| 분기 | 조건 | 결과 |
|---|---|---|
| `boundary_pp` | `known_boundary=True` | BQ prior ON |
| `transferable_prior` | `known_boundary=False` | affine prior ON |

Final은 group 수 / OOF regret / mode stability를 계산하지 않고 검사하지도 않는다.
그 사다리는 `v1_declared`에만 보관한다.

**known_boundary** = 고장 **조건(경계)** 을 안다 ≠ 고장 **시점** 을 안다.

## 9장 — Executor gate (`select_paper_ppx`)

Prior ON일 때만. Contract가 연 후보만:

`prior-only · unbounded · bounded · dual* · transport* · history* · direct fallback`

승인 AND:

- Gain ≥ 2%  
- Unit wins ≥ 60%  
- Worst ratio ≤ 1.10  

PASS → 최소 Val loss executor freeze  
FAIL → Direct/persistence fallback  

## 10장 — 이름 사전

| 이름 | Prior | 다음 |
|---|---|---|
| boundary_pp | ON | executor gate |
| transferable_prior | ON | executor gate |
| neural_safety | OFF (`v1_declared`만) | 바로 fallback |
| 승인 executor | ON | C4 freeze |
| Direct fallback | — | 예측만 |

두 함수:

- `select_ppx_route` = Prior 쓸지  
- `select_paper_ppx` = 어떤 executor인지  

## 관련 ablation MD (유지)

- `BQ_AFFINE_MIXTURE_ABLATION_RESULTS_KO.md` — BQ+Affine 혼합은 승격 거절, 메인 불변
