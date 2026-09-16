# PP-X Final Paper v4 — Data-Driven Hierarchical Router

작성: 박진서
Document ID: `paper-final-v4-hierarchical-router`
상태: **논문 Method·그림 재현 정본** (retrospective freeze; prospective confirmation 전)
동결 manifest: [`protocols/PPX_FINAL_PAPER_V4_MANIFEST.json`](protocols/PPX_FINAL_PAPER_V4_MANIFEST.json)

## 1. 핵심 정의

PP-X v4는 contract에서 계산 가능한 prior–residual 후보를 만들고, **validation
손실만으로 후보와 seed/fold별 내부 설정을 선택**한 뒤 선택을 고정하여 test를 한 번
평가한다.

- 데이터셋 이름, registry R², test label/test batch는 selector 입력이 아니다.
- BQ는 강제 prior가 아니라 계산 가능한 경우 열리는 옵션이다.
- Boundary battery에서는 `BQ/affine × executor`를 같은 validation 기준으로 비교한다.
- Direct는 비교용 대조군이며 PP-X 후보 메뉴에는 넣지 않는다.
- 후보 선택 후에만 test prediction을 ensemble하고 성능을 보고한다.

## 2. Algorithm 1

```text
typed contract (explicit time semantics)
  → computable PP-X family candidates
  → family-level aggregate validation MSE argmin
  → seed/fold별 내부 configuration validation MSE argmin
  → 선택 고정
  → test prediction ensemble
```

### 2.1 후보 생성

Contract는 이름이 아니라 필요한 변수가 실제로 계산 가능한지를 판정한다.

- 사용자가 `group_key`를 입력하면 그 unit-ID 컬럼을 우선한다. 빈칸이면
  `groups/units/unit/engine/cell` 이름을 찾고, 없으면 top-level 1차원 컬럼에서
  반복 ID와 label별 연속 행 블록을 검사한다. 후보가 없거나 동률로 모호하면
  추측하지 않고 `group_key` 입력을 요구한다.
- boundary와 progression이 있으면 BQ 후보를 열 수 있다.
- affine state/rate가 계산되면 affine prior 후보를 연다.
- 사용자가 `time_key`를 입력하면 그 컬럼을 우선한다. 빈칸이면
  `cycles/time/progress/coordinate` 이름을 먼저 찾고, 없으면 train의 top-level
  1차원 수치 컬럼 중 unit 내부 단조 후보가 정확히 하나일 때만 자동 선택한다.
- 시간 후보가 없거나 여러 개면 ordered progression과 causal history를 OFF한다.
  `x[:, 0]` 또는 다른 feature 위치를 시간으로 추정하지 않는다.
- 사용자가 `regime_key`를 입력하면 그 컬럼을 우선한다. 빈칸이면 train의 이름 있는
  regime 컬럼과 이산 feature를 검사해 unit-stable regime 또는 time-varying condition을
  찾는다.
- regime/context가 있으면 transport 후보를 연다.
- support heterogeneity가 계산되면 dual-scale 후보를 연다.

### 2.2 Family 선택

Family \(f\)의 validation 예측과 정답으로

\[
\hat f=\arg\min_{f\in\mathcal F_{\mathrm{computable}}}
\frac{1}{|\mathcal V|}\sum_{i\in\mathcal V}(y_i-\hat y_{f,i})^2
\]

를 계산한다. 동률은 candidate label 사전순으로만 해소한다. 데이터셋 이름은 이
함수에 전달되지 않는다.

### 2.3 Replicate 내부 선택

Seed 또는 held-out fold마다 내부 hyperparameter 후보 \(h\)를 validation MSE로
선택한다.

\[
\hat h_r=\arg\min_h \mathrm{ValMSE}_{r,h}
\]

각 replicate의 \(\hat h_r\)로 만든 test prediction만 평균한다. N-CMAPSS에서
validation 선택을 seed 평균 뒤 한 번만 수행하면 0.922가 되지만, 원래 동결된
replicate-local 선택을 재생하면 그림의 0.937이 재현된다.

## 3. Blindness와 평가 순서

| 단계 | 사용 가능 | 금지 |
|------|-----------|------|
| Candidate fit | train, causal feature, typed contract | test label·registry |
| Selection | group-disjoint validation y/prediction | dataset name·test |
| Frozen forward | 선택된 candidate의 test X | 재선택·TTA |
| Report | dataset name, test score | report 결과의 selector 주입 |

구현:

- `infer_ppx_contract_from_train_rows` (train-only candidate admissibility)
- `select_min_validation_loss`
- `select_replicate_validation_min`
- `experiments/ppx_final_result_reproduction_v1.py`

## 4. BQ는 옵션

Battery 3개 setting에서는 다음 다섯 family를 동일 validation MSE로 비교했다.

```text
bq_unbounded
bq_bounded
bq_dual_scale
affine_unbounded
affine_bounded
```

그 결과 이름 조건 없이 Sunwoda=`bq_bounded`, RWTH=`bq_bounded`,
MICH=`bq_dual_scale`이 선택됐다. 따라서 “boundary dataset이면 BQ 강제”가 아니라,
현재 후보와 validation에서 BQ arm이 데이터로 선택된 것이다.

## 5. 그림 결과 재현

정본: [`PPX_FINAL_RESULT_REPRODUCTION_KO.md`](PPX_FINAL_RESULT_REPRODUCTION_KO.md)

| Setting | Validation-selected arm/config | 재계산 pooled R² | 그림 R² |
|---------|--------------------------------|------------------:|--------:|
| HUST | replicate-local rate transport | 0.957959 | 0.958 |
| Sunwoda | bq_bounded | 0.939451 | 0.939 |
| N-CMAPSS | replicate-local history preset | 0.937271 | 0.937 |
| Virkler | support gate beta | 0.887977 | 0.888 |
| RWTH | bq_bounded | 0.878382 | 0.878 |
| MATR-b2 | transport | 0.862391 | 0.862 |
| MICH | bq_dual_scale | 0.751225 | 0.751 |
| NASA | fold-local history preset | 0.583751 | 0.584 |
| MATR2019 | affine calibration | 0.465698 | 0.466 |

요약:

- 그림 반올림 점수 재현: **9/9**
- 양의 pooled R²: **9/9**
- Macro mean pooled R²: **0.807**
- 저장된 선택과 validation argmin 일치: **9/9**

## 6. 동일예산 외부 비교

정본: [`FULL_EQUAL_CANDIDATE_BUDGET_RESULTS_KO.md`](FULL_EQUAL_CANDIDATE_BUDGET_RESULTS_KO.md)

- 각 setting의 최강 30-candidate baseline 대비 PP-X 우세: **8/9**
- Two-sided exact dataset sign test: **p=0.0391**
- 유일한 열세: Virkler, FT-Transformer 0.890 대 PP-X 0.888

이 비교와 위 route 선택은 모두 기존 개발 cohort의 retrospective 결과다.

## 7. v3와 결과가 달랐던 이유

보수적 v3 replay는 direct를 후보로 넣고 eligibility safety를 통과하지 못하면 direct로
돌아갔다. 또한 일부 setting에서 seed prediction을 먼저 평균한 뒤 단일 arm을 골랐다.
그 정책은 4/9 non-direct였고 그림의 PP-X Final 정의와 달랐다.

v4는 그림을 생성한 실제 PP-X 정의에 맞춰:

1. direct를 PP-X route 후보가 아닌 외부 대조군으로 분리하고,
2. PP-X family 안에서 validation argmin을 사용하며,
3. 동결 실험이 seed/fold-local로 고른 내부 설정을 같은 순서로 재생한다.

## 8. 재현

```bash
cd pp-extrapolation

PYTHONPATH=src:experiments:../ca-css-ncmapss \
  python experiments/ppx_final_result_reproduction_v1.py

PYTHONPATH=src:experiments:../ca-css-ncmapss \
  pytest -q tests/test_ppx_forward_selector.py
```

생성물:

- `results/ppx_final_result_reproduction_v1/results.json`
- `PPX_FINAL_RESULT_REPRODUCTION_KO.md`

## 9. 논문 Method 문단

> PP-X uses a hierarchical validation router over contract-computable
> prior–residual candidates. BQ is an optional candidate rather than a
> dataset-name rule. A family is selected by aggregate group-disjoint
> validation MSE, while replicate-local nuisance configurations are selected
> independently within each seed or held-out fold. All choices are frozen
> before test prediction, after which the selected replicate predictions are
> ensembled. Dataset identity, registry scores, test labels, and test-batch
> statistics are not selector inputs. Temporal executors require an explicit
> or semantically named time coordinate; feature position is never interpreted
> as time. Physical-unit IDs likewise use a user key first and otherwise an
> unambiguous repeated-ID train-column audit. Direct models are reported as external
> controls and are not route candidates in the PP-X figure reproduction.

## 10. 주장 가드레일

- 가능: validation-driven, dataset-name-blind retrospective reproduction,
  그림 9/9 점수 재현, BQ optional-candidate selection, equal-budget 8/9 비교.
- 불가: prospective validity, 보편적 우월성, test를 이용한 선택, retrospective
  결과를 미래 성공률로 표현.
