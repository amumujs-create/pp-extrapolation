# Distance–uncertainty PP 개발 실험

PP의 affine tail은 유지하고 seed disagreement가 큰 NN residual만 관측별로 감쇠했다.

\[
\hat y_s(x)=\hat y_{a,s}(x)+\exp[-\beta d(x)-\gamma u(x)]\hat r_s(x)
\]

`d`는 train support까지의 정규화 거리, `u`는 5개 PP residual의 표준편차를 target scale로 나눈 값이다. `beta`와 `gamma`는 hull-out validation에서 **seed별 MSE의 평균**을 최소화하도록 선택했다. 이는 ensemble 최고점 대신 한 번 재학습할 때의 위험을 줄이는 선택 기준이다. XJTU와 MATR는 구조 및 계수 선택에서 제외했다.

## Pooled R²

표의 `single`은 5회 재학습 평균±표준편차, `ens`는 5-seed 평균 예측이다.

| 데이터 | PP single | DU-PP single | PP ens | DU-PP ens | 선택 결과 |
|---|---:|---:|---:|---:|---|
| HUST | 0.724±0.039 | **0.732±0.035** | 0.773 | **0.774** | β=0, γ=4 |
| Virkler | 0.857±0.019 | 0.857±0.019 | 0.888 | 0.888 | β=0, γ=0 |
| RWTH | 0.506±0.020 | 0.506±0.020 | 0.507 | 0.507 | β=0, γ=0 |
| MICH | -1.522±0.000 | -1.522±0.000 | -1.522 | -1.522 | β=0, γ=0 |
| NASA | 0.495±0.004 | **0.502±0.004** | 0.495 | **0.502** | fold별 선택 |

NASA 네 fold 중 하나가 γ=64를 선택했고 하나가 거리 감쇠 β=0.5를 선택했다. 나머지는 PP를 그대로 유지했다.

## 해석

불확실성 감쇠는 HUST와 NASA에서 single-seed 평균과 ensemble pooled R²를 함께 높였고 HUST의 seed 표준편차도 줄였다. Virkler와 RWTH에서는 validation이 γ=0을 선택해 성능을 보존했다. 따라서 이 모듈은 적어도 개발 데이터에서는 손상을 자동 회피하는 선택적 보정기로 작동했다.

MICH 실패는 해결하지 못했다. 이 데이터에서는 PP residual 자체가 모든 seed에서 사실상 0이어서 disagreement도 0이다. 즉 이 gate는 이미 붕괴한 residual을 복구하는 모듈이 아니라 불안정한 residual을 줄이는 모듈이다.

개선 폭은 HUST +0.008 single-seed R², NASA +0.007로 작다. 이것만으로 강한 성능 novelty를 주장할 수는 없다. 논문에서는 `retraining-risk calibrated residual executor`로 정의하고, 새 untouched cohort에서 PP 대비 평균 성능, seed 분산, calibration 및 coverage를 함께 확인해야 한다. 일반 NN과의 mixture gate는 핵심 모델에서 제외한다.

재현 명령:

```bash
PYTHONPATH=src python experiments/distance_uncertainty_pp.py --max-epochs 300
```

원시 결과는 `results/distance_uncertainty_pp_v1/results.json`에 저장된다.
