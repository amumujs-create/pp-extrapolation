# MATR2019 strict state-space 외삽의 전체-history 보강 결과

## 목적

기존 MATR2019 strict capacity-tail 결과 `latent PP 0.257 < FT-Transformer 0.331`을
직접 개선하기 위해 원래 train/validation/test cell과 capacity boundary를 그대로 유지했다.
각 예측 시점 이전의 최근 64-cycle 원자료와 8·16·32·64·128-cycle 및 전체 prefix의
SOH·loss·temperature 통계, 현재 cycle과 관측된 충전 정책을 사용했다. 미래 측정값,
test lifetime 및 test 전체 trajectory 통계는 사용하지 않았다.

Train은 cell당 최대 400개 시점을 균등 추출했고 validation/test strict-tail 행은 모두
유지했다. Direct attention과 hazard-PP는 동일 입력과 동일 설정 탐색 예산을 사용했다.
이미 확인한 MATR2019 test에 대한 사후 개발 실험이다.

## 결과

| 모델 | ensemble pooled R² | 판정 |
|---|---:|---|
| 기존 latent PP | **0.257** | 유지 |
| 기존 동일예산 FT-Transformer | **0.331** | 현재 최고 |
| 전체-history direct attention | -1.119 | 폐기 |
| 전체-history hazard-PP | -0.004 | 폐기 |
| 정책 입력 제거 stable hazard-PP | -0.052 | 폐기 |
| Stable hazard-PP + policy Group-DRO | -0.134 | 폐기 |

전체-history direct attention의 개별 seed R²는 `-0.872, -6.404, -2.040, -1.950,
-3.716`이었다. Hazard-PP는 `-0.078, 0.042, -0.122, 0.004, -0.053`으로 더
안정적이었지만 양의 ensemble R²를 얻지 못했다.

정책 번호·현재 나이·온도를 직접 입력에서 제거하고 SOH와 열화율의 multiscale 변화만
남긴 결과도 -0.052였다. 정책을 입력으로 주지 않고 loss 계산에서 domain으로만 사용해
최악 정책 오차를 강조한 Group-DRO는 -0.134로 더 낮았다. 조건 정보를 제거하면 정책별
열화속도 차이를 식별할 수 없고, train 정책의 최악 손실을 줄이는 것만으로 파일순서
validation/test cohort 차이는 해결되지 않았다. 두 경로 모두 채택하지 않는다.

## 해석

History를 제공하는 원칙이 잘못된 것은 아니다. 이 파일순서 split에서는 train 평균 수명
769, validation 683, test 885 cycle이고 validation/test에 train에 없는 충전 정책이 섞여
있다. 전체 history와 policy context를 사용한 모델이 validation MSE를 크게 낮췄지만 그
관계가 test cohort로 전이되지 않았다. 짧은 여섯 summary feature를 사용한 기존 FT와
latent PP가 오히려 이 cohort shift에 덜 민감했다.

따라서 이 결과를 보고 test에 맞춰 history 길이나 policy encoding을 다시 고르지 않는다.
사진의 `0.257 < 0.331` 표는 아직 해결되지 않았다. 전체-history 모델은 정책 균형 split의
별도 개발 결과와 혼합하지 않고, 원 strict-tail 주 결과에서는 폐기한다.

기계 판독 결과: `results/matr_strict_history_pp_v1/results.json`

실행 코드: `experiments/matr_strict_history_pp.py`
