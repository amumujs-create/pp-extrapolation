# 비교 알고리즘 공정성 보강 프로토콜

현재 보관된 결과는 방법의 범위를 보여 주는 **개발 결과**다. 모든 비교군이 모든
데이터셋에서 같은 탐색·학습 예산으로 실행된 결과표로 인용하지 않는다. 이 문서는
투고용 최종 표를 다시 만드는 고정 프로토콜이다.

## 발견된 결손

| 항목 | 현재 상태 | 최종 표에서의 처리 |
|---|---|---|
| V-REx, GroupDRO, monotone NN, linear-tail RBF | `extrapolation_competitors_all_v1`에서 다수 split 실행 | 동일 adapter와 seed로 전체 재실행 |
| Engression | HUST·Virkler·Sunwoda·RWTH·MATR·MATR-b2·N-CMAPSS·NASA만 실행 | MICH·Milling 포함 전체 재실행 |
| linear-mean GP | exact GP, train 750행 무작위 제한 | full-train inducing-point variational GP로 교체 |
| TabPFN / FT-Transformer | 일부 배터리 split 및 별도 예산 | 가능한 split에서 같은 train/val/test artifact로 재실행 |
| PP 후보 탐색 | route별 역사적 9–33 후보 | 최종 공통 candidate budget으로 별도 재학습 |

MATR batch 2의 문헌 CPGRU 수치와 재현 수치는 절대 섞지 않는다. 논문 표는
`this-work matched protocol` 열만 주 지표로 사용한다. 문헌 수치는 별도의
`reported protocol` 열에 dataset split, tail 정의, label horizon, test row 수를 함께
쓴다. 수치가 다르면 어느 구현이 우월한지를 주장하지 않고 protocol mismatch로
보고한다.

## 고정된 최종 비교 계약

1. 각 데이터셋은 `train / validation / test` 행, group id, feature adapter, RUL
   target을 하나의 immutable `.npz` manifest로 동결한다. 모든 baseline은 test
   prediction을 내기 전 truth와 group hash를 검증한다.
2. PP-X, plain MLP, FT-Transformer, V-REx, GroupDRO, monotone NN, Engression,
   linear-tail RBF, SVGP는 **동일한 train features, 동일 validation rows, 동일한
   five seeds (42–46)** 를 쓴다. 시계열 모델은 같은 causal history window를 쓴다.
3. 모델별로 30개의 validation 후보를 사용한다. 후보는 architecture/learning-rate/
   regularization을 포함하며, seed 42에서 validation MSE가 가장 작은 하나를 고른다.
   선택된 설정만 seed 42–46으로 재학습한다. PP-X도 30개를 넘는 historical route
   search를 최종 비교 점수에 사용하지 않는다.
4. GP는 750행 exact GP가 아니라, 전체 train 행을 쓰는 inducing-point SVGP를
   사용한다. inducing point 수는 `min(512, floor(sqrt(n_train)*16))`, 30개
   hyperparameter 후보, validation early stopping으로 고정한다. GPU가 없어 SVGP를
   실행할 수 없을 때 해당 행을 미실행으로 두며, 750행 GP로 대체해 우열을 말하지
   않는다.
5. data가 너무 작아 train/validation에서 30 후보를 안정적으로 고를 수 없는 경우
   (Milling처럼 validation group 하나)에는 **all-model comparable tuning 불가**로
   분리한다. 이 경우 같은 고정 설정의 sensitivity 결과만 보조자료에 둔다.

## 실행 산출물

`results/final_equal_budget_v1/<dataset>/`마다 아래를 저장한다.

- `split_manifest.npz`: train/validation/test truth와 group hash
- `selection.json`: 30 candidate의 validation MSE·학습시간·선택 설정
- `predictions.npz`: 각 seed의 test prediction과 ensemble prediction
- `metrics.json`: pooled R², unit-macro R², RMSE, MAE
- `budget.json`: candidate 수, epoch/patience, train row 수, parameter 수, runtime

최종 논문 표에는 각 값과 함께 `all-data`, `same-budget`, `five-seed`, `not-run`
상태를 표시한다. 이 표가 만들어지기 전에는 PP의 성능 우위를 ‘공통 예산에서
검증됨’이라고 쓰지 않는다.

## 우선 실행 순서

1. **MICH**: Engression, V-REx, GroupDRO, monotone NN, linear-tail RBF,
   FT-Transformer, SVGP를 final PP-X raw-cycle adapter와 동일한 202행에 실행한다.
2. **NASA Milling**: train/validation group 수가 작아진 이유와 target boundary를
   감사한 뒤, 비교 가능 여부를 결정한다. 가능하면 위 세트를 실행하고, 불가능하면
   small-sample sensitivity로 격하한다.
3. HUST, Virkler, Sunwoda, RWTH, MATR2019, MATR-b2, N-CMAPSS, XJTU, FEMTO도
   같은 manifest·30-candidate·5-seed 체계로 재생성한다.
4. 마지막으로 cluster bootstrap과 paired group permutation을 **seed 평균 예측**에
   적용한다. 행 단위 IID p-value는 보고하지 않는다.
