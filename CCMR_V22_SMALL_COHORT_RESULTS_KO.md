# CCMR v2.2 small-cohort route 결과

작성자: 박진서

## 판정

CCMR v2.2는 v2.0 causal dynamics predictor를 바꾸지 않고, validation
physical unit이 3--9개인 경우 train-group cross-fit 합의와 validation
raw-risk로 인증하는 `small_crossfit_bank` route를 추가했다.

비보류 개발 승격에는 성공했지만 Alloy A에서 Engression을 이기지는 못했다.

## 비보류 개발

- 개발 고호트: Concrete, LG M50T, SIT LFP, RADAR NMC, Luminosity;
- false accept: 0/5;
- 최대 raw test regret: 0%;
- exact fallback error: 전부 0;
- v2.0 strict 승리: 1개.

SIT LFP에서 v2.0은 전부 거부했지만 v2.2는 46.78%를 배포했다.

- pooled RMSE 개선: 0.70%;
- macro RMSE 개선: 3.01%;
- 최대 regret: 0%.

RADAR NMC의 기존 pooled/macro 개선 5.91%/7.52%와 최대 regret 0%도
그대로 유지했다.

## 동결 후 보류 replay

### Alloy A

- route: `small_crossfit_bank`;
- CCMR v2.2 R²: 0.771452;
- pooled/macro RMSE 개선: 2.84%/5.12%;
- raw mean/CVaR/max regret: -16.20%/0%/0%;
- Engression R²: 0.991385;
- Engression pooled 개선: 81.13%.

v2.2의 최종 예측은 v2.0과 동일했다. 작은 validation cohort를 정식
cross-fit route로 승인했지만, v2.0 cautious route가 이미 같은 30% 행을
배포했기 때문이다. 따라서 Alloy의 Engression 격차는 줄지 않았다.

### MultiStage RPT

- route: `stable_bank`;
- CCMR v2.2 R²: 0.988847;
- pooled/macro RMSE 개선: 35.99%/49.82%;
- raw mean/CVaR/max regret: -70.94%/-9.79%/0%;
- Engression R²: 0.979404;
- Engression pooled 개선: 13.02%, 최대 regret 205.49%.

MultiStage 결과도 v2.0과 동일하며 Engression 우위와 최대 regret 0%를
유지했다.

## 결론

v2.2는 SIT LFP에서 적용 범위를 안전하게 넓힌 일반화 개선이다. 그러나
Alloy의 predictor 값 자체를 바꾸지 않았으므로 Engression을 이기기 위한
해결책은 아니었다.

Alloy 격차를 줄이려면 route가 아니라 smooth global continuation을 담당하는
새 mean-function expert가 필요하다. Alloy 결과를 다시 사용해 그 expert를
튜닝하면 독립성은 사라지므로 이번 결과 이후 추가 변경은 하지 않았다.

## 산출물

- route: `src/pp_extrapolation/small_cohort_route.py`
- 개발: `results/ccmr_v22_trajectory_development/results.json`
- manifest: `protocols/CCMR_V22_FROZEN_MANIFEST.json`
- 보류 replay: `results/ccmr_v22_frozen_holdout_replay/results.json`
- 예측: `results/ccmr_v22_frozen_holdout_replay/predictions.npz`
