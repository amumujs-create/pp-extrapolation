# PP-X CRT 승격 기각 실험 기록

## 최종 결정

**Contract-Certified Residual Transport(CRT)를 PP-X의 논문용 최종 구조로 승격하지 않는다.**
논문 메인은 기존의 validation-approved, contract-conditioned PP-X로 유지한다.

CRT는 prior가 거절된 경우에도 일반 외삽 성능을 강화하기 위해 개발했다. 그러나
독립 validation unit 또는 unit-disjoint leave-one-out evidence로 정책을 고정한 뒤
평가했을 때, 정확도와 unit-level 안정성을 동시에 만족하지 못했다. 이 결과에 따라
CRT, 이진 grade gate, 연속 transport-mass gate는 논문 성능 주장에 포함하지 않는다.

## 평가 원칙

- test label을 route, shell, transport mass 선택에 사용하지 않았다.
- DS03은 train unit 1--6, validation unit 7--9, test unit 10--15로 분리했다.
- MultiStage는 validation 14개 unit의 leave-one-unit-out evidence로 정책을 정했다.
- 안전성은 physical-unit RMSE와 worst-unit relative regret로 평가했다.
- 실패한 설정과 결과는 덮어쓰지 않고 별도 버전으로 보존했다.

## DS03 결과

| 방법 | pooled R² | macro RMSE | 최대 unit regret |
|---|---:|---:|---:|
| Direct fallback | 0.881805 | 7.160983 | 0.00% |
| Ungated CRT | 0.879899 | 7.738462 | 69.62% |
| Continuous mass gate v3 | 0.885072 | 7.111223 | 4.30% |
| Equal-budget Engression | **0.901323** | **6.097458** | 1.20% |

연속 gate는 validation에서 depth shell별 transport mass를
`[0, 0, 0.75, 0.2]`로 선택했다. Direct fallback보다 pooled R²는 0.003267
높아졌지만 Engression보다 0.016251 낮았고 macro RMSE도 1.013765 컸다.
따라서 prior-rejected generic extrapolator의 승격 기준을 통과하지 못했다.

## MultiStage 결과

| 방법 | pooled R² | macro RMSE | 최대 unit regret |
|---|---:|---:|---:|
| PP latest | 0.979397 | 0.004775 | 0.00% |
| Engression | 0.979404 | 0.004846 | 205.49% |
| Ungated CRT | 0.988071 | 0.003625 | 163.31% |
| Continuous safe gate v3 | 0.985019 | 0.003938 | 0.00% |
| CCMR | **0.988847** | **0.002866** | **0.00%** |

연속 safe gate는 shallow shell에서 PP, middle shell에서 CCMR--CRT
혼합(`alpha=0.1`), deep shell에서 CCMR를 선택했다. 안전성은 확보했지만
CCMR 단독보다 pooled R²가 0.003828 낮고 macro RMSE가 0.001072 컸다.
따라서 CRT 혼합은 가장 강한 기존 안전 executor를 개선하지 못했다.

## 구조적 해석

1. Residual transport의 distributional energy score 개선이 point-RMSE의
   unit-tail risk 제어를 보장하지 않았다.
2. Validation 평균 개선만으로 transport를 승인하면 일부 physical unit에서
   큰 regret가 발생했다.
3. Shell별 연속 shrinkage는 binary gate보다 나았지만 validation-to-test
   transfer 오차를 제거하지 못했다.
4. Prior가 승인되지 않는 DS03에서는 Engression의 일반 외삽 성능을 넘지 못했다.
5. Prior와 trajectory evidence가 있는 MultiStage에서는 기존 CCMR가 정확도와
   안정성 모두 가장 우수했다.

## 논문에 반영할 경계

- PP-X의 핵심 주장은 **승인된 구조 가정이 있는 영역에서 prior-residual core와
  validation-approved executor가 강하다**는 것으로 제한한다.
- Prior가 거절된 모든 문제에서 PP-X가 최강이라고 주장하지 않는다.
- Prior 거절 시에는 exact fallback/abstention을 유지하며, DS03에서는 강한
  distributional baseline이 더 우수했다는 반례를 공개한다.
- CRT는 후속 연구 후보로만 남기고 현재 논문 Algorithm 1에는 포함하지 않는다.

## 재현 산출물

- Core: `src/pp_extrapolation/residual_transport.py`
- Grade gate: `src/pp_extrapolation/extrapolation_grade_gate.py`
- DS03 실험: `experiments/ncmapss_ds03_crt_experiment.py`,
  `experiments/ncmapss_ds03_crt_grade_gate_v2.py`,
  `experiments/ncmapss_ds03_crt_mass_gate_v3.py`
- MultiStage 실험: `experiments/multistage_rpt_ppx_residual_transport.py`,
  `experiments/multistage_rpt_ppx_grade_gate_v2.py`,
  `experiments/multistage_rpt_ppx_mass_gate_v3.py`
- 결과: `results/ncmapss_ds03_crt_*`, `results/multistage_rpt_ppx_*`
- 검증: 전체 `182 passed`, 변경 파일 lint 오류 없음

## 결론

CRT는 흥미로운 정확도 신호를 보였지만 가장 강한 comparator와 안전성 기준을
동시에 넘지 못했다. 따라서 현재 가장 방어 가능한 선택은 **기존 PP-X를 논문
메인으로 유지하고, CCMR를 trajectory-domain executor로 유지하며, CRT를
기각된 개발 실험으로 보존하는 것**이다.
