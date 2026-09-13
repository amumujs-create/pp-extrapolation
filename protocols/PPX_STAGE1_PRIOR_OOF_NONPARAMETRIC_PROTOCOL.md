# PP-X Stage-1 Prior OOF Regret — Nonparametric Audit

작성자: 박진서  
상태: retrospective confirmatory-style audit

## 질문

Stage-1 prior admissibility의 핵심 비교

\[
\mathrm{oof\_prior\_regret} = \mathrm{MSE}(\mathrm{prior}) - \mathrm{MSE}(\mathrm{matched\ direct})
\]

를 **validation unit**에서 재구성하고, 소표본 비모수 검정으로
“prior가 direct보다 나쁘지 않다”는 근거가 있는 도메인을 센다.

## 비교 모델

| 이름 | 정의 |
|---|---|
| prior_only | train에서 group-LOO로 α를 고른 weighted ridge affine → 전체 train 재적합 후 validation 예측 |
| matched direct | common-backbone archive의 `validation_plain` ensemble (동일 split) |

α 선택에 validation label을 쓰지 않는다 (train-unit LOO만).

## 추론 규칙

`protocols/PPX_STATISTICAL_INFERENCE_PROTOCOL.md` 따름:

- unit effect: \(\delta_u=\log\mathrm{RMSE}_{\mathrm{direct},u}-\log\mathrm{RMSE}_{\mathrm{prior},u}\)  
  (양수 = prior가 unit RMSE에서 이김)
- exact sign-flip (주), Wilcoxon (보조)
- domain: regret≤0 이면 stage-1 pass 후보
- BH on confirmatory domains (L1+)

## 성공 기준 (감사)

- 12-setting(가능 범위)에 prior_only vs direct 표가 생긴다.
- stage-1 pass/fail과 stage-2 / test 정합을 보고한다.
- 이 실험으로 Algorithm 1의 τ나 hard-S1을 바꾸지 않는다. 근거만 남긴다.

## 비범위

- prior+residual 재학습 없음 (그건 stage-2)
- test로 α/threshold 재튜닝 금지
