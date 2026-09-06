# MATR 2019 추가 분석과 논문 수준 판단

## 확증 결과와 통계적 강도

Latent-transition PP의 pooled R² 0.257은 사전고정 primary endpoint에서 plain NN(-0.184), original PP(-0.666), Ridge(-1.796)를 모두 넘었다. single-seed 평균 0.171±0.109도 plain NN -0.477±0.304보다 좋았다.

독립 test cell 10개를 동일 가중한 paired bootstrap 결과는 다음과 같다. 값은 `latent RMSE - comparator RMSE`이므로 음수가 latent 우세다.

| 비교 | 평균 unit RMSE 차이 | latent 승리 cell | 95% bootstrap CI |
|---|---:|---:|---:|
| Latent vs Ridge | -49.76 | 9/10 | [-73.97, -23.79] |
| Latent vs original PP | -25.32 | 9/10 | [-41.62, -8.57] |
| Latent vs Jacobian PP | -10.49 | 6/10 | [-23.71, 3.55] |
| Latent vs plain NN | -6.61 | 6/10 | [-27.61, 15.25] |

따라서 latent PP가 Ridge와 original PP를 unit 수준에서도 개선했다는 근거는 강하다. plain NN 대비 pooled 성능은 명확히 좋지만, 새 cell 모집단 전체에서 평균적으로 우월하다는 paired CI는 아직 확정적이지 않다.

## Cell별 이질성

Latent PP는 c37, c38, c41, c42, c43, c44에서 plain NN보다 낮은 RMSE를 보였고 c35, c36, c39, c40에서는 나빴다. 특히 c36과 c39의 latent R²는 각각 -2.20, -4.85였다. 반대로 c37, c38, c42, c44에서는 0.55~0.70의 양의 R²를 얻었다. 이 차이가 pooled 양수와 unit-macro 음수가 동시에 나온 이유다.

라벨 없이 계산 가능한 세 지표와 `latent RMSE gain vs plain NN`의 Spearman 상관은 모두 약하고 유의하지 않았다.

| 후보 진단값 | Spearman rho | 탐색적 p-value |
|---|---:|---:|
| capacity hull 거리 | 0.345 | 0.328 |
| 전체 feature 최근접 train 거리 | 0.127 | 0.726 |
| recent-rate shift | 0.006 | 0.987 |

따라서 단순 support 거리나 열화율 shift로 성공 cell을 가르는 certificate를 지금 만들면 test에 과적합된다. 현재 test에서 임계값을 만들지 않는다.

## Ablation이 말하는 것

성능 순서는 original PP(-0.666) < Jacobian PP(-0.139) < latent PP(0.257)였다. Jacobian 방향 제약은 평균 성능을 개선했지만 test 기울기 위반률이 seed별 45.5~100%였고 양의 R²에는 도달하지 못했다. Latent late-tail expert가 추가돼야 양의 pooled R²가 나왔다. 이는 단순한 monotonic loss보다 전환 후 함수 형태를 학습하는 구조가 핵심이라는 ablation 근거다.

다만 test gate가 모든 seed에서 평균 0.997~1.000이므로 test-tail 내부의 혼합 효과를 입증한 것은 아니다. gate는 train/validation에서 전환을 학습한 뒤 test를 late expert로 routing했다. 논문에서는 `dynamic test mixture`가 아니라 `learned transition and late-regime routing`으로 표현해야 한다.

## 논문 수준 판단

현재 상태는 방법론과 사전고정 외부 cohort가 있어 일반적인 응용 논문 수준은 넘는다. 그러나 최상위 저널의 강한 범용 외삽 주장에는 부족하다.

- 강점: convex-hull 밖 test를 명시했고, test 전 프로토콜을 공개 고정했으며, 일반 NN·Ridge·기존 PP·Jacobian PP를 함께 비교했다. single-seed 재학습 안정성도 제시했다.
- 약점: confirmatory test cell이 10개뿐이고 plain NN 대비 paired CI가 0을 포함한다. unit-macro R²가 음수이며 Virkler 유형에서는 실패했다. mixture-of-experts와 physics/data hybrid 자체는 이미 활발한 연구 주제라 구조 이름만으로는 강한 novelty가 아니다.

현재 증거로는 Reliability Engineering & System Safety, Mechanical Systems and Signal Processing, Engineering Applications of Artificial Intelligence 계열을 현실적인 목표로 볼 수 있다. Applied Energy를 노리려면 최소한 독립 battery cohort 하나를 더 추가하고, cell-level failure를 test-label 없이 판별하는 calibration/certificate를 별도 validation cohort에서 개발한 뒤 새 test에서 검증해야 한다. Nature Energy급 주장을 하려면 더 다양한 chemistry와 protocol에 걸친 대규모 외부검증 및 실제 uncertainty calibration이 필요하다.

## 제출 전 남은 필수 작업

1. 저장된 split과 모델을 이용해 논문용 Methods와 data-flow diagram을 작성한다.
2. HUST 개발과 MATR 2019 확증을 명확히 분리하고, Virkler 실패를 scope boundary로 보고한다.
3. Cell bootstrap CI와 모든 per-cell 결과를 본문 또는 supplement에 싣는다.
4. 파라미터 수, 학습 시간, seed별 결과를 표로 추가한다.
5. 가능하면 다른 untouched battery cohort에서 완전히 고정된 모델을 한 번 더 평가한다. 이 실험 전에는 certificate threshold를 새로 만들지 않는다.
