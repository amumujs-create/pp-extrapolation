# Latent transition PP: deep-future 개발 결과

## 모델

기존 PP의 frozen affine 안전 경로 위에 두 개의 학습형 neural tail expert와 단조 증가 transition gate를 추가했다.

`y_hat = y_affine + (1-g) r_early + g r_late`

`g = sigmoid(softplus(a) q + b + h(context))`

`q`는 열화 진행 좌표이고 `h(context)`는 시간, 누적 및 최근 열화율에서 학습된다. gate는 진행 방향으로 되돌아가지 않지만, 전환 위치와 두 expert의 식은 데이터에서 학습한다. 두 expert가 동일해지거나 gate 하나만 학습되는 것을 막는 regularizer의 사용 여부도 validation MSE로만 선택했다.

## pooled R²

| 데이터/방법 | single-seed 평균 ± SD | 5-seed prediction ensemble |
|---|---:|---:|
| HUST plain NN | 기록값 별도 비교 | 0.340 |
| HUST Jacobian-ray PP | 0.117 ± 0.353 | 0.255 |
| HUST latent-transition PP | **0.709 ± 0.071** | **0.811** |
| Virkler plain NN | 기록값 별도 비교 | -0.688 |
| Virkler Jacobian/free spline PP | -4.979 ± 7.189 | -2.398 |
| Virkler latent-transition PP | -5.496 ± 1.248 | -5.374 |

HUST에서는 일반 NN ensemble과 Jacobian PP를 모두 크게 넘어섰고 seed 분산도 작았다. 따라서 연속 열화에서 late-regime expert를 학습하는 경로는 유망하다.

Virkler에서는 gate가 0.49~0.68 부근에 머물고 pooled R²가 음수였다. sparse crack-length 단계 하나만 validation에 존재하여 전환 시점과 전환 후 곡률을 동시에 식별하기 어렵고, frozen affine 경로 및 bounded correction도 급가속 균열 꼬리를 충분히 수정하지 못한다. 이 결과는 더 복잡한 NN을 넣는 것만으로 해결되지 않음을 보여준다.

## 연구 판단

현재 latent-transition PP는 HUST 유형을 위한 강한 박사논문 확장 후보이나 범용 해법으로 주장할 수 없다. Virkler 유형에는 change-point 이후의 성장 법칙을 식별할 더 조밀한 validation support, 또는 crack-growth increment를 직접 예측하고 누적하는 state-space/hazard formulation이 필요하다. 다음 단계는 test 재튜닝이 아니라 이 구조를 고정해 untouched 연속열화 cohort에서 확증하고, sparse fracture는 별도의 모델 계열로 명시하는 것이다.
