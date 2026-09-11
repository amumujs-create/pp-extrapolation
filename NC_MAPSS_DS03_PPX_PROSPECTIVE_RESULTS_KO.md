# N-CMAPSS DS03 PP-X prospective 결과

작성자: 박진서  
프로토콜 최초 동결 커밋: `76fda59`  
원본 해시 동결 커밋: `c4b82b4`  
selection artifact 동결 커밋: `ffb7387`

## 봉인 절차

1. DS03를 내려받기 전에 unit split, 입력 계약, candidate route, gate 임계값,
   평가 기준을 GitHub에 동결했다.
2. 다운로드 후 HDF5를 열기 전에 원본 크기와 SHA-256을 별도 커밋했다.
3. `Y_test`를 읽지 않는 `prepare-select` 단계만 실행했다.
4. selection JSON과 모델 artifact를 커밋한 뒤 test endpoint를 한 번 공개했다.

Selection SHA-256:
`f965976428058f33c3655a18e2baaf3aadd74f25420b21649a25140db2af8014`

## Validation 선택

| route | validation MSE | unit wins | worst unit RMSE ratio |
|---|---:|---:|---:|
| direct fallback | **37.639** | 기준 | 1.000 |
| basic prior-residual | 85.774 | 0% | 1.999 |
| multiscale prior-residual | 95.858 | 0% | 1.652 |

DS03에는 알려진 failure boundary나 관측된 failure-mode regime이 없고,
train-only prior admissibility evidence도 충분하지 않았다. 따라서 동결 PP-X
gate는 **direct fallback**을 선택했다.

## Prospective test

6개 완전 미개봉 test engine, 438개 cycle row:

- 선택 PP-X route: direct fallback
- pooled R²: **0.882**
- pooled RMSE: **7.732 cycles**
- physical-unit macro RMSE: **7.161 cycles**

사후 route audit:

| PP-X 후보 | test pooled R² |
|---|---:|
| **선택된 direct fallback** | **0.882** |
| basic prior-residual | 0.832 |
| multiscale prior-residual | 0.869 |

따라서 validation 이전에 동결된 gate는 test에서 실제로 가장 좋은 PP-X route를
선택했다. prior를 거절하지 않았다면 성능이 악화됐을 것이다.

## 30-candidate 비교

모든 비교군은 같은 train/validation/test rows에서 validation 후보 30개,
search seed 42, refit seeds 42–46을 사용했다.

| 모델 | pooled R² |
|---|---:|
| Engression | **0.901** |
| FT-Transformer | 0.899 |
| linear-tail RBF | 0.882 |
| **PP-X selected fallback** | **0.882** |
| GroupDRO | 0.881 |
| SVGP | 0.878 |
| V-REx | 0.866 |
| plain MLP | 0.866 |
| monotone NN | 0.866 |

PP-X는 Engression보다 pooled R²가 `0.0195` 낮았다. 6개 unit 중 PP-X가
Engression보다 RMSE가 낮은 unit은 1개였고, mean unit log-RMSE ratio는
`−0.250`이다. worst-unit RMSE ratio는 `1.989`로 사전 안전 한계 2 이하였다.

## 사전 성공기준 판정

| 기준 | 판정 |
|---|---|
| 정보계약 위반 없음 | PASS |
| pooled R² ≥ 0 | PASS |
| strongest equal-budget baseline 대비 mean unit effect > 0 | **FAIL** |
| catastrophic unit ratio > 2 없음 | PASS |

전체 predictive superiority 기준은 통과하지 못했다.

## 결론

이 prospective 실험은 두 주장을 분리한다.

1. **선택정책 미래 일반화:** 지지됨. PP-X gate가 미개봉 cohort에서 부적절한
   prior 두 개를 거절했고, test에서 가장 좋은 PP-X route를 선택했다.
2. **최강 비교모델 대비 정확도 우월성:** 지지되지 않음. Engression이 더 높았다.

따라서 논문에서는 “prospective route-selection success”라고 쓸 수 있지만
“prospective predictive superiority”라고 쓰면 안 된다. 또한 DS03는 DS02와
같은 N-CMAPSS 계열이므로 독립 실제 도메인 확증은 별도로 필요하다.

## CCMR v2.2 후속 실행

사용자 요청에 따라 prospective 공개가 끝난 뒤 PP-X direct prediction을 anchor로
두고 CCMR v2.2 dynamics bank를 train units 1–6과 validation units 7–9에서
적합했다. validation risk certificate가 correction을 승인하지 않아
`exact_fallback`이 선택됐다.

- CCMR v2.2 deployed R²: **0.882**
- anchor PP-X direct R²: **0.882**
- pooled/macro improvement: **0**
- maximum raw regret: **0**

즉 이 데이터에서 CCMR는 보정을 적용하지 않아 PP-X fallback과 정확히 같은
결과를 냈다. 이 후속 실행은 test 공개 뒤 요청된 exploratory analysis이므로
prospective 성능 근거에는 포함하지 않는다.
