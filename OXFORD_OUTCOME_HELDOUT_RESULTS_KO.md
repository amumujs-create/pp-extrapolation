# Oxford outcome-held-out PP v2 결과

프로토콜을 commit `864c2b4`로 고정한 뒤 Cell7--Cell8 RUL을 한 번 계산했다. 이 데이터셋은 외부 Oxford 데이터이며 두 cell의 outcome은 이번 protocol에서 처음 평가했지만, 파일과 test covariate 수는 feasibility 단계에서 확인했으므로 globally untouched라고 부르지 않는다.

| 모델 | Pooled R² | RMSE | Cell-macro R² |
|---|---:|---:|---:|
| Ridge | -3.785 | 18.711 | -5.683 |
| Engression 0.1.9 | -0.946 | 11.932 | -1.911 |
| **Causal PP** | **-0.119** | **9.050** | **0.074** |

사전 성공 기준 `PP R² > 0`을 통과하지 못했다. PP는 두 경쟁모델보다 오차가 작았지만 외부 cohort 성공으로 계산하지 않는다.

| Cell | n | PP R² | PP RMSE | Engression R² | Engression RMSE |
|---|---:|---:|---:|---:|---:|
| Cell7 | 24 | **0.705** | **3.762** | -4.011 | 15.496 |
| Cell8 | 32 | -0.557 | 11.520 | **0.190** | **8.309** |

PP와 Engression의 실패 cell이 반대다. PP의 validation-selected short-history tail은 Cell7에는 전달됐지만 Cell8의 lifetime scale에는 맞지 않았다. Engression은 반대 패턴이다. test는 100% train health hull 밖이고 median 거리는 0.833 train SD, 최대 1.979 SD였다. 거리 자체보다 cell별 health--remaining-life mapping 차이가 주요 실패 원인이다.

이 결과는 test label 없이 어느 tail law가 새 cell에 맞는지를 판별하는 applicability mechanism이 아직 부족하다는 직접 증거다. Cell7/Cell8 결과를 보고 mixture gate를 새로 만들면 외부 검증의 의미가 사라지므로 추가 튜닝하지 않는다. 논문에는 성공 데이터와 함께 negative external validation으로 포함하고, 보편적 우월성 주장을 제한한다.

재현: `experiments/oxford_outcome_heldout_v2.py`, `results/oxford_outcome_heldout_v2/results.json`.
