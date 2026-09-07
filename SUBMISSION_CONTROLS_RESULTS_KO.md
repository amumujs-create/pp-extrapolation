# 제출 전 추가 대조 결과

MATR 2019 동일 분할, 사후 추가 실험. 기존 확증 결과는 보존했다. NN 대조는 각 seed 9개 학습률/weight decay 설정을 validation으로 선택했다. Latent는 기존 9개 regularizer 설정 선택 결과다. 후보 수는 같지만 탐색 축 및 실행 시간은 동일하지 않다.

| 모델 | ensemble pooled R² | single-seed 평균 ± SD |
|---|---:|---:|
| plain | 0.214 | 0.010 ± 0.124 |
| single | -0.130 | -0.444 ± 0.329 |
| fixed | -0.138 | -0.388 ± 0.424 |
| histgb | -0.425 | deterministic |
| 기존 latent PP | 0.257 | 0.171 ± 0.109 |

Latent vs tuned plain unit RMSE delta: 0.577, 95% CI [-9.867, 12.830], wins 7/10.

Plain NN을 튜닝하면 ensemble R²가 -0.184에서 0.214로 개선된다. Latent의 우위는 0.043으로 축소되므로 큰 성능 우위라는 기존 표현은 수정해야 한다. 단일 expert 및 고정 0.5 gate보다 latent가 높아 학습 gate의 유용성을 지지하지만, 다른 초기화/정규화 효과를 완전히 분리한 증명은 아니다. Gradient boosting은 -0.425였다.

남은 비교는 동일 optimizer 조건의 gate 제거 실험과 강한 외삽 모델(예: 선형 tail spline), 적용 가능한 로컬 TabPFN의 동일 입력 비교다. 현재 비교만으로 최신 SOTA 우월성을 주장하지 않는다.

실행 중 histogram boosting의 병렬 라이브러리 오류가 발생해 OMP/BLAS 스레드 1로 해당 군만 재실행했다. NN 요약/후보 점수는 보존됐지만 오류 전 메모리에 있던 row prediction은 저장되지 않아 이번 산출물은 JSON 평가 요약 중심이다.
