# CCMR v2.0 예측기 독립 강화 결과

작성자: 박진서

## 결론

CCMR v1.9의 라우터만 바꾼 것이 아니라 predictor 자체를 causal dynamics
expert bank로 교체했다. Alloy A와 MultiStage RPT는 설계·튜닝에서 제외하고,
다른 trajectory 고호트 개발을 끝낸 뒤 manifest를 해시로 동결하고 한 번만
replay했다.

새 이름은 **CCMR v2.0 Causal Dynamics Bank**다.

## 예측기 구조

기존 단일 ridge residual 대신 네 expert를 사용한다.

- multi-scale linear-rate;
- damped acceleration;
- monotone hinge continuation;
- small nonlinear residual.

Train physical-unit cross-fit 방향 합의, context support, validation raw
mean/CVaR/max regret 제약으로 expert와 deployment mass를 선택한다. Replicated
stable route에는 causal shadow coverage 인증을 추가하고, 그렇지 않으면
adaptive causal gate를 통과한 행만 보정한다. 모든 거절 행은 persistence를
bit-exact하게 복원한다.

## 보류 고호트 이전 개발 결과

Concrete, LG M50T, SIT LFP, RADAR NMC, Luminosity 5개만 사용했다.

- false accept: 0/5;
- 개발 test 최대 raw regret: 0%;
- 모든 fallback replay 오차: 0;
- RADAR NMC: pooled/macro RMSE 5.91%/7.52% 개선;
- Concrete, LG, SIT, Luminosity: 위험 또는 증거 부족으로 exact fallback.

Luminosity에서는 과거 모델의 평균 2.45% 개선을 포기했지만 최대 regret
29.34% 문제도 함께 제거했다. 즉 개발 단계에서 정확도와 위험의 교환 규칙이
의도대로 작동했다.

동결 manifest:
`protocols/CCMR_V20_FROZEN_PREDICTOR_MANIFEST.json`

Holdout replay 전에 동결된 manifest SHA-256:
`a45466f0f6fefc4af9d764f5d594a01cc036d70e8317985ed61b2ee99347c3aa`

## Alloy A 보류 replay

선택된 predictor는 damped-acceleration expert이고 cautious causal route가
배포됐다.

- Persistence: R² 0.757918;
- 기존 CCMR v1.6/v1.9: R² 0.764333, pooled/macro 개선 1.33%/2.23%;
- CCMR v2.0: R² 0.771452, pooled/macro 개선 2.84%/5.12%;
- v2.0 raw mean/CVaR/max regret: -16.20%/0%/0%;
- v2.0 coverage: 30%.

따라서 v2.0은 기존 CCMR보다 정확도와 평균 unit 위험을 개선했고 최대
손상 0%를 유지했다. 다만 Engression의 R² 0.991385와 RMSE 개선 81.13%에는
여전히 크게 못 미친다.

## MultiStage RPT 보류 replay

선택된 predictor는 nonlinear-residual expert이고 stable-bank route가
배포됐다.

- Persistence: R² 0.972776;
- 기존 CCMR v1.9: R² 0.979397, pooled/macro 개선 13.01%/16.39%;
- Engression: R² 0.979404, pooled/macro 개선 13.02%/15.13%;
- CCMR v2.0: R² 0.988847, pooled/macro 개선 35.99%/49.82%;
- v2.0 raw mean/CVaR/max regret: -70.94%/-9.79%/0%;
- v2.0 coverage: 81.82%.

여기서는 v2.0이 기존 CCMR과 Engression을 평균 정확도에서 모두 넘었고,
최대 regret도 0%로 유지했다. Engression의 최대 regret는 205.49%였다.

## 주장 가능한 범위

이 결과는 보류 고호트 정보를 모델 개발에 재사용하지 않은 frozen replay지만,
두 데이터셋의 과거 결과가 이미 알려져 있었으므로 완전히 새로운 독립 확증은
아니다.

정당한 결론은 다음과 같다.

> 다른 trajectory 고호트로만 개발한 CCMR v2.0 predictor는 두 보류
> 고호트에서 기존 CCMR을 모두 개선했다. MultiStage에서는 비교군 최고
> 평균성능도 넘으면서 최대 unit regret 0%를 유지했다.

“모든 외삽에서 최고” 또는 “신규 미개봉 확증 완료”라고 표현하면 안 된다.

## 산출물

- predictor: `src/pp_extrapolation/causal_dynamics_bank.py`
- adaptive causal gate: `src/pp_extrapolation/causal_backtest.py`
- 개발 결과: `results/ccmr_v20_trajectory_development/results.json`
- expert ablation:
  `results/ccmr_v20_trajectory_development/ablation.json`
- 보류 replay:
  `results/ccmr_v20_frozen_holdout_replay/results.json`
- 예측: `results/ccmr_v20_frozen_holdout_replay/predictions.npz`
