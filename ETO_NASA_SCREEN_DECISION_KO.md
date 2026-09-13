# ETO 실측 외삽 평가와 채택 판단

**판정: 현재 ETO 채택하지 않음. 추가 구조 개발 중단.**
사용자가 요구한 모델 노벨티, 기존 적용 범위 유지, PP-X와 Engression 모두에
대한 우월성을 함께 충족하지 못했다. 기존 PP-X를 교체하지 않았다.

## 실제 실행한 평가

NASA B0005/B0006/B0007/B0018 원본 방전 용량을 사용했다. cell별 4 folds이며
각 fold는 train 2개, validation 1개, test 1개 cell이다. 재사용 데이터에 대한
새로운 회고적 평가이며 미개봉 cohort 확인은 아니다.

16개 관측 window를 동일하게 주고 train/validation에서는 1·2·4주기 후의
실측 용량을 학습한다. test에서는 8·16·32주기 후를 예측한다. query horizon이
train 최대를 넘으므로 horizon을 포함한 입력 공간의 convex hull 밖이다.
이는 원래 PP-X의 건강상태/운전조건 외삽·RUL 평가와는 다른 목표다.

마지막 관측 인덱스는 15·23·31·39로 고정했다. 정답은 정확히
`capacity[end + horizon]`이며 실제 cycle 차이도 검사했다. 총 48개 고정
test query, 3 seeds(42·43·44), 150 full-batch epochs, validation 선택만
사용한다. 동일 seed 수의 앙상블을 비교하며 모든 예측에 동일한 train-target
clip을 적용한다. 모델별 native 정규화·optimizer·parameter count는 다르므로
동일 FLOPs 또는 완전 튜닝 비교라고 주장하지 않는다.

## 배터리별 RMSE (Ah, 3-seed ensemble)

| Test cell | ETO | PP 코어 | 공식 Engression |
|---|---:|---:|---:|
| B0005 | 0.05072 | 0.05052 | 0.06961 |
| B0006 | 0.11888 | 0.10657 | 0.08192 |
| B0007 | 0.05583 | 0.09105 | 0.04450 |
| B0018 | 0.08440 | 0.10582 | 0.09568 |
| cell 균등 평균 | 0.07746 | 0.08849 | 0.07292 |
| 최악 cell | 0.11888 | 0.10657 | 0.09568 |
| 최장 32주기 horizon, cell 평균 | 0.11409 | 0.12993 | 0.10380 |

개별 seed-fit RMSE의 평균도 ETO 0.07846, PP 코어 0.08928,
Engression 0.07603으로 Engression 우위다. ETO는 두 비교 모델 각각에 대해
2/4 cell에서 이겼으며, 둘 모두에 이긴 cell은 B0018 하나다. B0018에서는
세 모델 모두 R2가 음수이고 persistence RMSE 0.07321이 ETO보다 낮다.
실행 전 정한 '각 cell·각 horizon에서 두 비교 모델 모두 초과' 기준은 실패했다.

PP 코어는 현재 태스크에 맞춰 동일 16개 관측·시간·horizon을 입력한
`fit_pp`, alpha=10 모델이다. **full frozen PP-X policy가 아니다.**
따라서 이 표를 기존 PP-X보다 우수하다는 결과로 사용할 수 없다.
후보를 탈락시키는 screen으로만 사용한다.

## 커버리지

| 지표 | ETO | PP 코어 | Engression |
|---|---:|---:|---:|
| 유한 예측 query 수 | 48/48 | 48/48 | 48/48 |
| cell별 전체 R2 > 0 | 3/4 | 2/4 | 2/4 |
| cell × horizon별 R2 > 0 | 4/12 | 4/12 | 4/12 |

양수 cell 수가 많아도 평균·최악 오차 개선을 뜻하지 않는다. 네 cell은 단일
cohort이며 겹치는 prefix/horizon은 독립 표본이 아니다. 이 결과로 confidence
interval coverage, 기존 9+5 설정의 범용성, 통계적 우월성을 주장하지 않는다.
기존 운전조건 외삽 등에 필요한 입력/target adapter가 현재 ETO에 없으므로
원래 적용 범위 유지 조건 역시 통과하지 못했다.

## 모델 노벨티 판정

현 구현은 causal GRU encoder, autonomous latent ODE, decoder다.
[Latent ODE (NeurIPS 2019)](https://proceedings.neurips.cc/paper/2019/file/42a6845a557bef704ad8ac9cb4461d43-Paper.pdf)는
잠재 ODE를 이용한 시간 외삽을 이미 직접 다룬다. 현재 구현에 그것과 구별되는
입증된 모델링 기여는 없다. 같은 ODE의 semigroup 오차는 수치 적분 및 flow
일관성을 진단하지만 단독으로 신규 메커니즘이나 외삽 정확도를 입증하지 않는다.
초기 zero vector field도 이 오차가 0이다. 또한 초기 decoder는 임의 출력이므로
'시간에 따른 상수 출력'이지 '마지막 관측값 persistence'와 같지는 않다.

## 기록과 재현

- 실행: `python experiments/eto_nasa_horizon_screen.py`
- 설정·실행 전 코드/data SHA256: `results/eto_nasa_horizon_screen_v1/protocol.json`
- 모든 seed/epoch/parameter count/시간/평가: `results/eto_nasa_horizon_screen_v1/results.json`
- cell/end/horizon ID 및 정답·seed별 예측: `results/eto_nasa_horizon_screen_v1/predictions.npz`
- 채택 조건: `protocols/SUCCESSOR_EXTRAPOLATION_ACCEPTANCE_PROTOCOL.md`

출력 경로는 이미 존재하면 실행을 거부하여 기존 결과를 덮어쓰지 않는다.
저장된 예측에서 4개 unit split, 48개 query ID, 3개 학습 모델의 finite output,
보고된 RMSE 일치를 별도로 재계산해 확인했다.

이전 합성 RMSE는 비교 근거에서 철회한다. 당시 train/test horizon이 같았고
prefix 끝 인덱스 대비 target horizon이 한 칸 어긋나 있었다.
