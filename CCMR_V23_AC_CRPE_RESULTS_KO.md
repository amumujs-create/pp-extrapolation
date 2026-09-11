# CCMR v2.3 AC-CRPE 범용 예측기 강화 결과

작성자: 박진서  
모형: CCMR v2.3 Anchor-Competitive Contract-Conditioned
Risk-Budgeted Prior Experts

## 판정

**승격하지 않는다.**

최종 nested leave-one-domain-out 결과는 CCMR v2.2 대비 domain-equal
geometric RMSE ratio `1.0000021469`였다. 즉 RMSE가 `0.0002147%`
미세하게 커져 사전 고정한 “엄격 개선” 기준을 통과하지 못했다.

다만 최종 구조는 다섯 domain 중 4개에서 v2.2 비악화, false accept 0,
test raw maximum unit regret 0, exact fallback 전부 통과했다. 따라서
강건성은 회복했지만 성능 상승 증거는 아니다.

## 최종 LOCO 결과

| 제외한 domain | 다른 domain으로 선택된 정책 | test RMSE | raw max regret |
|---|---|---:|---:|
| Concrete | strict | 0.0080409430 | 0 |
| LG M50T | strict | 0.0820519176 | 0 |
| SIT LFP | strict | 0.0061859803 | 0 |
| RADAR NMC | anchor only | 0.0060265368 | 0 |
| Luminosity | strict | 0.0247092644 | 0 |

SIT LFP의 극미세 악화 때문에 기하평균 ratio가 1보다
`2.15e-6` 높았다. 반올림으로 동률처럼 보이더라도 strict 승리로
해석하지 않았다.

## 구현한 모형 구조

\[
\hat y = B_{v2.2}
+ \alpha_a(\hat y_{\text{ML portfolio}}-B_{v2.2})
+ \alpha_p(P_c-\hat y_{\text{anchor}})
\]

1. `B_v2.2`: 이전 동결 모델을 exact safety anchor로 유지한다.
2. ML portfolio: persistence, weighted ridge, linear-tail RFF,
   MLP ensemble, Engression ensemble을 validation unit-risk로 선택한다.
3. Contract expert: target 계약에 맞는 causal dynamics expert만 허용한다.
4. Safety: absolute support, group cross-fit consensus, weak deployment mass,
   causal shadow wins, raw mean/CVaR/max regret를 모두 적용한다.
5. 증거가 부족하면 v2.2 예측을 정확히 복원한다.

dataset 이름은 router 입력에 쓰지 않았다. observed trajectory,
latent RUL, condition transfer, known boundary의 target 계약만 expert
허용 범위를 정한다.

## 개발 ablation에서 확인된 점

- 강한 anchor를 그대로 쓰면 geometric RMSE는 3.44% 악화되고
  max regret는 386.9%까지 증가했다.
- absolute support guard만 추가하면 평균 RMSE는 0.88% 개선됐지만
  max regret 32.8%로 안전성에 실패했다.
- 약한 anchor mass는 max regret를 0.67%까지 줄였지만 평균 성능이
  1.38% 악화됐다.
- v2.2 nested safety anchor와 5회 연속 causal shadow wins를 결합하면서
  catastrophic regret와 false accept를 제거했다.
- contract별 evidence threshold를 추가하면 v2.2 exact fallback에
  수렴했다. 안전성과 성능 상승 사이의 남은 차이는 `0.0002147%`다.

전체 개발 경로는
`results/ccmr_v23_development_ablation/results.json`에 보존했다.

## 12-domain 사후 호환성 감사

기존 최종 PP-X 12개 route artifact는 모두 존재했다. 이 중 9개는
row-aligned paired evidence가 있고, 9/9 승리 및 domain-equal geometric
RMSE reduction 32.55%를 유지한다. XJTU, FEMTO, NASA milling은
retrospective extension tier로 분리했다.

이 12-domain 수치는 v2.3 구조 선택에 사용하지 않았고, 기존 heterogeneous
final route의 호환성 감사일 뿐 v2.3의 새 matched benchmark가 아니다.

## 보류 고호트 처리

Alloy A와 MultiStage는 개발 과정에서 로드하지 않았다. 최종 promotion
gate가 실패했으므로 frozen success manifest를 만들지 않았고, 두 고호트의
replay 및 holdout ablation도 실행하지 않았다.

대신 `protocols/CCMR_V23_REJECTED_MANIFEST.json`에 source와 최종 개발
결과의 SHA-256을 기록하고 `holdout_replay_authorized=false`로 봉인했다.
따라서 Alloy에서 Engression을 이겼다는 주장은 이번 버전에는 없다.

## 결론

AC-CRPE의 핵심 기여는 “강한 데이터 모델을 PP로 교체”하는 방식이 아니라
“동결된 안전 모델 위에 contract별 약한 예측 보정을 올리고 실패 시 정확히
복원”하는 구조다. 그러나 현재 다섯 development domain에서는 안전성만
확보했고 엄격한 평균 성능 상승은 입증하지 못했다. v2.3은 실패 ablation과
다음 개발 기반으로 유지하며 배포 모델은 CCMR v2.2로 남긴다.
