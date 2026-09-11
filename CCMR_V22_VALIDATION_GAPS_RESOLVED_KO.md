# CCMR v2.2 모델 검증 공백 보강 결과

작성: 박진서  
범위: 개발 5도메인만  
금지: Alloy A / MultiStage를 이용한 선택·재튜닝

## 추가한 검증

1. 동일 train/validation/test-tail split과 동일 `context` feature 계약에서
   CCMR v2.2와 Ridge, linear-tail RFF, MLP ensemble, Engression ensemble,
   validation-selected portfolio 비교 (seeds 42–46)
2. v2.2 대 v2.0 physical-unit paired 통계
3. false accept / false reject proxy 감사
4. frozen route threshold 27조합 민감도
5. 기존 구조 ablation과 결합한 성능–안전 해석

## 1. 동일 split 경쟁모형 결과

| 모델 | 평균 pooled R² | GM RMSE / persistence | 양의 개선 도메인 | 최대 raw regret |
|---|---:|---:|---:|---:|
| **CCMR v2.2** | 0.565 | **0.987** | 2/5 | **0%** |
| Persistence | 0.565 | 1.000 | 0/5 | 0% |
| Ridge | -0.757 | 1.103 | 2/5 | 1,712% |
| Linear-tail RFF | -0.125 | 1.330 | 1/5 | 2,038% |
| MLP ensemble | 0.407 | 1.594 | 1/5 | 6,388% |
| Engression ensemble | 0.252 | 1.083 | 3/5 | 2,355% |
| Validated portfolio | **0.568** | **0.981** | 3/5 | **64.6%** |

해석:
- unconstrained 경쟁모형은 일부 도메인 R²가 높지만 worst-unit regret가 매우 크다.
- validated portfolio는 평균 RMSE가 CCMR보다 조금 낮지만 max regret 64.6%로
  CCMR의 2% 안전 계약을 충족하지 않는다.
- CCMR의 방어 가능한 주장은 **최고 평균 정확도**가 아니라
  **5도메인 max regret 0을 유지한 선택적 개선**이다.

도메인별 최고 R²는 Concrete=MLP, SIT/RADAR=Ridge, Luminosity=portfolio,
LG=CCMR이었다. 따라서 “모든 도메인 정확도 1위” 주장은 기각한다.

## 2. Physical-unit paired 결과

v2.2와 v2.0의 test physical unit 93개를 ID로 맞췄다.

| 도메인 | unit | v2.2 승 / 동률 | 평균 상대 RMSE 이득 | sign-flip p | bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| Concrete | 38 | 0 / 38 | 0 | 1.0 | [0, 0] |
| LG M50T | 2 | 0 / 2 | 0 | 1.0 | [0, 0] |
| SIT LFP | 4 | 2 / 2 | +9.07% | 0.5 | [0, 23.61%] |
| RADAR NMC | 24 | 0 / 24 | 0 | 1.0 | [0, 0] |
| Luminosity | 25 | 0 / 25 | 0 | 1.0 | [0, 0] |

집계:
- positive domain 1/5
- domain sign-flip p = **1.0**
- equal-domain hierarchical mean relative gain = **1.81%**
- hierarchical 95% CI = **[0, 6.78%]**

따라서 v2.2의 성능 우월은 유의하다고 주장하지 않는다. 이득은 SIT의
small-cohort route에 국소화되어 있다.

## 3. Gate 오탐·미탐 감사

Frozen v2.2 설정:
- 승인: SIT LFP, RADAR NMC
- false accept: **0**
- unsafe accept: **0**
- safe-gain false reject: **0**

단순 pooled false-reject proxy는 2개지만, 두 건 모두 raw bank max regret가
2%를 초과한 **안전한 거절**이다. 즉 평균 이득만 보면 놓친 것처럼 보이지만
안전 계약 기준으로는 false reject가 아니다.

## 4. Route threshold 민감도

진단 grid:
- active fraction: 0.40 / 0.50 / 0.60
- stable macro gain: 2.5% / 5% / 10%
- small macro gain: 5% / 10% / 15%
- 총 27조합

결과:
- 27개 중 **18개**가 frozen v2.2와 동일하게 SIT/RADAR만 승인
- 이 18개는 false accept 0, unsafe accept 0
- stable gain을 2.5%로 낮춘 9개 조합은 Concrete를 추가 승인하고
  unsafe accept 1개를 발생

따라서 frozen stable gain 5%는 단일 날카로운 점이 아니다. 5–10% 구간에서
승인 집합과 안전성이 유지된다. 반대로 2.5%로 완화하면 위험해진다.
이 sweep은 **사후 진단**이며 임계값 재선택에는 사용하지 않는다.

## 5. Seed / fold 검증 해석

CCMR v2.2 bank는 ridge 기반 deterministic group cross-fit이며 난수 seed를
사용하지 않는다. 따라서 일반적인 NN seed stability는 해당하지 않는다.
검증 단위는 seed가 아니라 physical-unit ID와 group cross-fit이다.

남은 기술적 보강은 `max_ensemble_folds={5,10,20}`에 대한 fold-cap
민감도다. 현 구현은 group별 leave-out 예측을 결정론적으로 생성하므로,
이는 독립 확증보다 구현 민감도 분석에 해당한다.

## 논문 결론

> CCMR v2.2는 개발 5도메인에서 최고 평균 정확도의 범용 모델은 아니다.
> 동일 split·동일 context feature의 강한 경쟁모형은 일부 도메인에서 더
> 높은 R²를 보이지만 최대 raw unit regret가 65%에서 6,388%까지 증가한다.
> CCMR은 max regret
> 0을 유지하면서 SIT와 RADAR에서만 보정을 승인한다. Route threshold
> 27조합 중 18조합에서 같은 승인 집합이 유지되며, stable gain 임계값을
> 2.5%로 낮출 때만 unsafe accept가 발생한다. 따라서 검증된 핵심은
> universal accuracy가 아니라 risk-aware selective correction이다.

## 산출물

- `experiments/ccmr_v22_matched_development_benchmark.py`
- `results/ccmr_v22_matched_development_benchmark/results.json`
- `experiments/ccmr_v22_validation_gap_audit.py`
- `results/ccmr_v22_validation_gap_audit/results.json`
- `experiments/ccmr_v22_paper_structure_ablation.py`
- `results/ccmr_v22_paper_structure_ablation/results.json`

## 남은 근본 한계

신규 미개봉 prospective cohort가 없다. 위 결과는 개발 도메인의
retrospective 검증을 강화하지만 외부 확증을 대신하지 않는다.
