# PP-X GCIE 승격 기각 실험 기록

## 최종 결정

**Grade-CVaR Implicit Expert(GCIE)를 PP-X의 prior-rejected generic executor로
승격하지 않는다.** 기존 PP-X 논문 구조와 CCMR executor는 변경하지 않는다.

GCIE는 Engression을 내부 모델로 사용하지 않고 다음 요소를 결합한 개발 후보였다.

- source-unit centroid 거리로 계산한 pointwise support grade
- bounded implicit distributional generator
- group-balanced energy score와 unit-tail CVaR
- unit 내 ordered-coordinate monotonicity penalty
- unit-disjoint nested pseudo-extrapolation
- source-only global transport-mass gate와 exact fallback

DS03 test는 이전 연구에서 이미 공개됐으므로 모든 결과는 retrospective
development evidence다. 구성과 gate는 test label 없이 고정했지만 새로운
prospective 확증으로 해석하지 않는다.

## 승격 기준

다음 세 조건을 모두 만족해야 했다.

1. equal-budget Engression보다 pooled R²가 높을 것
2. equal-budget Engression보다 macro RMSE가 낮을 것
3. direct fallback 대비 최대 physical-unit RMSE regret가 2% 이하일 것

## DS03 결과

| 방법 | pooled R² | macro RMSE | 최대 unit regret |
|---|---:|---:|---:|
| Direct fallback | 0.881805 | 7.160983 | 0.00% |
| GCIE v4 validation gate | 0.885020 | 6.939566 | 2.57% |
| Nested GCIE v5 | 0.872670 | 6.289642 | 30.59% |
| Equal-budget Engression | **0.901323** | **6.097458** | 1.20% |

### GCIE v4

12개 구성을 validation unit 7--9에서 탐색하고 5개 seed로 재적합했다. 안전
혼합률은 `rho=0.1`로 고정됐다. Direct fallback보다 aggregate 성능은 개선됐지만
Engression보다 pooled R²가 0.016302 낮고 macro RMSE가 0.842109 컸다. 최대
unit regret도 2% 기준을 초과했다.

### Nested GCIE v5

Train unit 1--6에서 outer unit-disjoint OOF와 내부 pseudo-tail cutoff
`(0.50, 0.65, 0.80)`를 사용했다. 각 outer unit은 model fit, early stopping,
configuration selection, support-grade fit에서 제외했다. 반복 cutoff는 독립
unit으로 세지 않았다.

Source OOF gate는 전역 `alpha=0.75`를 승인했지만 test에서 transfer되지 않았다.
Macro RMSE는 v4보다 낮아졌으나 pooled R²는 direct fallback보다도 낮아졌고,
한 unit의 상대 RMSE regret가 30.59%에 달했다. Source grade 범위 밖 0.46%의
row에는 exact fallback이 적용됐다.

## 실패 해석

1. 6개 source unit의 outer OOF risk는 새로운 unit의 tail risk를 안정적으로
   추정하기에 부족했다.
2. Source OOF에서 승인된 큰 혼합률 `alpha=0.75`가 test의 unit heterogeneity에
   전이되지 않았다.
3. Group-CVaR와 energy-score 최적화만으로 pooled point-RMSE와 worst-unit
   regret를 동시에 제어하지 못했다.
4. Support distance는 test-batch 통계 없이 계산됐지만, 거리 자체가 residual
   correction의 방향과 크기를 충분히 식별하지 못했다.
5. Nested episode를 추가해도 Engression의 일반 외삽 성능을 넘지 못했다.

## 정보 계약과 한계

- Engression prediction이나 package는 GCIE 학습·선택에 사용하지 않았다.
- Application grade는 frozen source state만 사용하며 full-unit rank와
  test-batch statistics를 사용하지 않았다.
- Configuration과 gate는 test scoring 전에 고정했다.
- v5 산출물 생성 중 수치 검증 오류로 test loader 호출 시도는 세 차례 있었으나
  정책 변경이나 test-label tuning 없이 동일한 동결 정책으로 최종 1회 scoring했다.
- DS03 test가 역사적으로 공개된 상태이므로 성공했더라도 새 cohort 확증이
  추가로 필요했다.

## 논문 반영

- GCIE와 nested distribution gate는 현재 논문 Algorithm 1에 포함하지 않는다.
- Prior가 거절된 경우 기존 fallback/abstention을 유지한다.
- PP-X가 prior-rejected generic extrapolation에서 Engression보다 우수하다고
  주장하지 않는다.
- Prior가 승인되는 trajectory setting은 기존 CCMR executor를 유지한다.

## 재현 산출물

- `src/pp_extrapolation/grade_cvar_implicit_expert.py`
- `src/pp_extrapolation/pseudo_extrapolation.py`
- `src/pp_extrapolation/source_only_distribution_gate.py`
- `experiments/ncmapss_ds03_gcie_v4.py`
- `experiments/ncmapss_ds03_nested_gcie_v5.py`
- `results/ncmapss_ds03_gcie_v4_dev/results.json`
- `results/ncmapss_ds03_nested_gcie_v5_dev/results.json`

## 결론

새 구조는 direct fallback을 일부 개선하는 신호를 보였지만 Engression과
사전 안전 기준을 동시에 넘지 못했다. 특히 nested source-only gate의 승인과
test unit risk가 크게 불일치했다. 따라서 현재 가장 방어 가능한 결정은
**기존 PP-X를 유지하고 GCIE를 기각된 후속 개발 실험으로 보존하는 것**이다.
