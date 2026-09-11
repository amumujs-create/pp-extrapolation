# CCMR v2.1 확률 예측기 개발 결과

작성자: 박진서

## 판정

CCMR v2.1 stochastic dynamics predictor는 비보류 개발 게이트에서
**승격 실패**했다. 따라서 Alloy A와 MultiStage RPT replay는 실행하지
않았으며 최신 배포 모델은 CCMR v2.0으로 유지한다.

## 구현

v2.0 causal dynamics bank를 확률분포의 중심으로 두고, noise-conditioned
width-16 network를 추가했다. 8개 stochastic sample의 group-balanced energy
score와 mean-square stabilization으로 학습했다.

- five-bucket train-unit cross-fit;
- 90% validation conformal interval;
- cross-fit sign agreement 80%;
- relative dispersion 최대 75%;
- deterministic v2.0 context support;
- raw unit mean/CVaR/max regret cap 0%/1%/2%;
- 모든 거절과 비유한 입력의 bit-exact persistence fallback.

## 비보류 개발 결과

Concrete, LG M50T, SIT LFP, RADAR NMC, Luminosity만 사용했다.

- validation 승인: 2/5;
- false accept: 0/5;
- 최대 test raw regret: 0.59%;
- fallback replay 오차: 전부 0;
- frozen v2.0보다 strict하게 개선한 고호트: 0/5;
- promotion: false.

RADAR NMC에서는 v2.1이 pooled RMSE를 5.18% 개선했지만 v2.0의 5.91%보다
낮았다. 나머지 고호트의 최종 배포는 persistence와 같았다. 즉 확률 head는
안전했지만 기존 deterministic dynamics bank 이상의 예측 이득을 만들지
못했다.

## Engression이 더 잘된 이유에 대한 결과

Engression의 분포 학습 아이디어만 추가한다고 자동으로 성능이 오르지는
않았다. CCMR에서는 conformal·cross-fit·support·unit-risk 검사를 연속해서
통과해야 하므로 stochastic correction의 coverage가 줄었고, 승인된 RADAR
에서도 deterministic v2.0보다 오차가 컸다.

이는 Alloy에서 Engression을 이기기 위해 같은 모델을 흉내 내는 것보다,
CCMR의 물리적 mean function 자체를 더 잘 식별하거나 작은 validation에서도
일반화 가능한 별도 함수형 expert가 필요하다는 뜻이다.

## 보류 고호트를 열지 않은 이유

사전 protocol은 비보류 고호트 중 최소 하나에서 v2.0을 strict하게 이겨야
v2.1을 동결·평가하도록 규정했다. 이 조건이 0/5로 실패했다. 이 상태에서
Alloy 점수를 확인하고 구조를 다시 바꾸면 사용자가 요구한 독립 개발 조건을
위반한다.

## 산출물

- predictor: `src/pp_extrapolation/stochastic_dynamics.py`
- 개발·ablation:
  `results/ccmr_v21_trajectory_development/results.json`
- 실패 manifest:
  `protocols/CCMR_V21_REJECTED_PREDICTOR_MANIFEST.json`
- firewall:
  `protocols/CCMR_V21_STOCHASTIC_PREDICTOR_FIREWALL.md`
