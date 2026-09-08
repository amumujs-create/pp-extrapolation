# 외부 실패 코호트 PP 구조 개선 결과

## 범위와 증거 등급

성공한 HNEI locked 결과는 읽거나 재학습하지 않았다. NASA/UCF와 CALCE만 개발 대상으로 사용했다. 두 데이터의 test outcome은 이미 앞선 실험에서 확인했으므로 아래 결과는 **development evidence**이며 새로운 독립 확증 결과가 아니다.

## 구조 변경: safe-continuation PP

기존 PP는 affine prior에 neural residual을 더해 prior가 틀린 도메인에서도 affine 편향이 항상 남았다. 개선 구조는 하나의 네트워크 안에서 다음을 계산한다.

\[
\hat y_\lambda(x)=\lambda\,\hat y_{affine}(x)+(1-\lambda)\,\hat y_{NN}(x).
\]

두 경로는 같은 입력과 같은 loss로 공동 학습한다. `λ=0`은 별도 모델 조합이 아니라 동일 seed·동일 초기화·동일 학습량의 plain NN과 정확히 같은 safety continuation이다. PP 후보는 `λ∈{0.02,0.05,0.10,0.20}`이며 test label 없이 validation pooled RMSE가 가장 낮은 양의 λ를 선택했다.

## 5-seed 결과

| 데이터 | 동일 조건 NN ensemble R² | 개선 PP ensemble R² | ΔR² | 선택 λ | PP unit wins | Exact sign-flip p |
|---|---:|---:|---:|---:|---:|---:|
| NASA/UCF | −0.785 | **−0.391** | **+0.394** | 0.02 | 3/3 | 0.250 |
| CALCE | 0.188 | **0.218** | **+0.030** | 0.20 | 3/4 | 0.375 |

NASA/UCF는 세 unit 모두 RMSE가 감소했고 unit-mean 감소량은 `0.394 days`, bootstrap 95% CI `[0.162, 0.526]`였다. 표본이 3개라 exact two-sided sign-flip p의 최솟값이 0.25이므로 유의성을 주장하지 않는다. 절대 pooled R²는 여전히 음수이므로 이 코호트가 해결됐다고 표현하지 않고, **NN 대비 손실을 구조적으로 줄였다**고 표현한다.

CALCE는 unit-mean RMSE가 `3.037 cycles` 감소했지만 95% CI `[−2.552, 8.625]`로 0을 포함했다. 대신 seed 평균 R²가 NN `0.054±0.131`에서 PP `0.147±0.093`으로 상승해 재학습 분산도 감소했다.

## 해석

- 공유 가능한 prior가 약한 NASA/UCF에서는 validation이 2%만 승인했다. 작은 prior도 ensemble R²를 0.394 높였지만 절대 예측력은 부족하다.
- CALCE에서는 validation이 20% prior를 승인했고 pooled·seed 평균 성능이 모두 개선됐다.
- 이 구조는 prior를 항상 강제하던 PP의 failure mode를 완화하며, prior가 유효하지 않으면 `λ→0`으로 일반 NN에 연속적으로 돌아갈 수 있다.
- 현재 λ는 데이터셋 단위 상수다. 다음 모델 단계는 unit의 causal history로 λ(x)를 예측하되, `λ=0` safety arm보다 validation 성능이 나빠지면 승인을 거부하는 것이다.

## 재현 파일

- 실험: `experiments/external_failure_pp_recovery.py`
- PP 구현: `src/pp_extrapolation/model.py`
- 최종 결과: `results/external_failure_pp_recovery_v1/safe_final_results.json`
- 최종 예측: `results/external_failure_pp_recovery_v1/nasa_alt_safe_final.npz`, `results/external_failure_pp_recovery_v1/calce_safe_final.npz`

