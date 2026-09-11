# PP-X 연구 전체 데이터셋 역사적 비교

> **Canonical paper pointer:** 논문 메인은 PP-X다. 발표·논문용 route 표는
> `FINAL_PP_BENCHMARK_TABLE_KO.md`, 최신 equal-budget 수치는
> `FULL_EQUAL_CANDIDATE_BUDGET_RESULTS_KO.md`, 주장 경계는
> `PPX_TOP_JOURNAL_VALIDATION_PACKAGE_KO.md`를 사용한다. 아래 표에는 초기
> legacy PP, 봉인 confirmatory PP, 중간 support gate 결과가 함께 남아 있어
> PP-X 최종 성능표로 인용하면 안 된다.

> 외삽/OOD 특화 경쟁군을 12개 설정 전부에 추가한 최신 표는 `ALL_DATASET_EXTRAPOLATION_COMPETITORS_KO.md`에 있다.

## 역사적 latent PP와 FT-Transformer의 직접 비교

동일 데이터·분할에서 각 seed마다 9개 validation 후보를 선택하고, 최대 300 epoch 및 patience 70으로 학습했다. PP와 FT의 후보 수와 선택 기준은 같지만 탐색 축과 연산량은 동일하지 않다. 아래 주 지표는 5개 seed 예측을 평균한 pooled R²다.

| 데이터/외삽 설정 | 원래 PP ensemble R² | FT ensemble R² | PP 단일 seed 평균±SD | FT 단일 seed 평균±SD | 해석 |
|---|---:|---:|---:|---:|---|
| HUST deep-future | **0.811** | 0.208 | **0.709±0.071** | -0.125±0.617 | PP 우세, FT seed 불안정 |
| Virkler deep-future | -5.374 | **-3.658** | **-5.496±1.248** | -15.658±30.473 | 둘 다 실패; FT ensemble 우위는 불안정 seed 오차 상쇄 영향 |
| MATR2019 unseen-cell tail | 0.257 | **0.331** | 0.171±0.109 | **0.260±0.074** | FT 우세 |

Unit-macro R²는 HUST에서 PP 0.342, FT -0.812이고, Virkler에서 PP -6.107, FT -4.101이며, MATR2019에서 PP -0.511, FT -0.076이다. Pooled가 주 지표지만 unit-macro 결과도 함께 공개해야 한다.

MATR에서 PP의 width·learning rate·weight decay를 추가로 9개 탐색한 사후 결과는 ensemble R² 0.183이었다. 원래 PP 0.257보다 낮았으므로 test를 보고 두 결과 중 좋은 값을 고르는 방식은 쓰지 않는다.

## 현재 저장된 다른 데이터셋의 PP 증거

아래 행은 데이터 분할과 PP 버전이 서로 다르다. 직접 PP–FT 비교표가 아니며, FT 빈칸은 미실행을 뜻한다. `평균`은 5개 seed의 R² 평균, `ensemble`은 seed 예측 평균의 R²다.

| 데이터/설정 | 집계 | Ridge/affine | 일반 NN | 기본 PP | support·개선 PP | FT / TabPFN† | 판정 |
|---|---|---:|---:|---:|---:|---:|---|
| HUST unseen-protocol tail | seed 평균 | 0.601 | 0.758±0.032 | 0.724±0.039 | **0.910±0.013** | 미실행 | 개선 PP 성공 |
| Virkler unseen-specimen tail | seed 평균 | -0.823 | -0.604±0.472 | 0.857±0.019 | **0.886±0.025** | 미실행 | PP 성공 |
| NASA battery health tail | seed 평균 | 0.424 | 0.233±0.094 | 0.495±0.004 | **0.513±0.002** | 미실행 | PP 성공 |
| Sunwoda unseen-cell tail | 5-seed 평균±SD / ensemble | 0.844 | -1.035±1.478 | 0.862±0.009 | **0.842±0.099 / 0.934 dual-scale PP** | -0.886±0.033† | 개선 PP 성공 |
| RWTH unseen-cell tail | 5-seed 평균±SD / ensemble | 0.419 | -0.113±1.663 | 0.506±0.020 | **0.818±0.050 / 0.842 dual-scale PP** | -2.175±0.106† | 개선 PP 성공 |
| MICH unseen-cell tail | 5-seed 평균±SD / ensemble | -1.522 | 0.339±0.359 / 0.684 direct NN | -1.522±0.000 | **0.703±0.077 / 0.751 dual-scale PP** | 미실행 | 개선 PP로 복구 |
| MATR batch 2 | 5-seed 평균±SD / ensemble | -1.276 | 0.744 ensemble | 0.471 | **0.852±0.061 / 0.862‡** | TabPFN 0.616±0.041 / 0.618† | 최종 개발 PP 우세 |
| XJTU untouched | ensemble | -1.477 | -1.565 | **-1.308** | -1.317 | 미실행 | 모두 실패 |
| FEMTO prospective | seed 평균 | -1.378 | **-0.902** | -1.378 | — | 미실행 | 모두 실패 |
| C-MAPSS FD002+FD004 strict OP-hull | seed 평균 | — | — | — | **0.747±0.011** | 미실행 | support PP 양의 R², 직접 대조 부족 |
| N-CMAPSS hard (unseen engine × high TRA × late life) | seed 평균 | 0.485 cycle-isotonic | 0.767±0.006 sequence Transformer+iso | 0.931±0.012 raw PP / 0.886±0.004 PP+iso | **0.934±0.007 adaptive multiscale PP** | 미실행 | adaptive PP ensemble 0.937; TabPFN 단일 seed 0.934와 동률권 |
| NASA milling material transfer | seed 평균 | — | -15.462±19.427 | **-4.826±0.000** | — | 미실행 | 모두 실패 |

† TabPFN v3 CPU는 5 seeds(42–46), estimator 1개, train에서 label을 보지 않는 deterministic equal-unit subsampling(최대 1,000행)으로 실행했다. Sunwoda/RWTH와 MATR batch 2 모두 seed 평균±표준편차와 prediction ensemble을 분리해 보고한다. 모두 PP와 같은 고정 test 행이지만 train cap이 달라 보조 비교다. 상세 수치·예측은 `results/tabpfn_external_batteries_v1/`에 보관한다.

‡ MATR batch 2의 0.471/0.523은 원본 봉인 confirmatory PP/support-PP 결과다. 0.862는 그 뒤 같은 고정 split에서 architecture·optimizer·state/rate transport를 validation으로 재선택한 최종 개발 PP다. 따라서 최종 모델 비교에는 0.862를 쓰되, 독립 confirmatory 증거로는 0.523을 대체하지 않는다.

MATRb2의 완전한 동일 seed 비교는 다음과 같다. `평균±SD`는 개별 pooled R², `ensemble`은 다섯 예측 평균의 pooled R²다.

| 모델 | seeds | 개별 pooled R² 평균±SD | prediction ensemble R² |
|---|---|---:|---:|
| **최종 PP** | 42–46 | **0.852±0.061** | **0.862** |
| V-REx | 42–46 | 0.046±0.468 | 0.850 |
| TabPFN v3 | 42–46 | 0.616±0.041 | 0.618 |
| BatteryLife CPGRU | 42–46 | 0.386±0.334 | 0.537 |
| BatteryLife CPTransformer | 42–46 | −0.081±1.044 | 0.380 |

PP는 같은 seed의 네 경쟁모델 비교에서 모두 5승 0패였다. seed별 값과 paired 검정은 `MATR_BATCH2_FIVE_SEED_COMPARISON_KO.md`에 있다.

## 현재 PP-X 문서에서의 해석

FT는 MATR2019의 초기 비교에서는 PP보다 높았지만 HUST에서는 크게 낮고, Virkler에서는 seed에 따라 붕괴한다. 이후 validation-only calibration을 포함한 최종 PP는 MATR2019에서도 0.466으로 개선됐다. 반대로 PP도 MICH·XJTU·FEMTO·Milling에서 실패하므로 보편적 우월성을 주장할 수 없다.

MATR2019의 Transformer 패배를 해결하기 위해 temporal latent PP, GRU residual PP, attention-Jacobian PP를 추가 실험했으나 각각 0.203, -0.236, 0.259로 FT 0.331을 넘지 못했다. 최종 attention-tail PP는 validation에서 beta=0을 선택해 FT 0.331을 정확히 복원한다. 따라서 성능 하락은 막았지만 이 split에서 prior 고유 이득은 없다.

후속 행·cell 진단에서 latent PP의 prediction-target correlation은 FT보다 높았지만 출력이
평균 25.2 cycles 낮게 calibration된 것을 확인했다. Leave-one-validation-cell-out으로
선택한 bounded affine output calibration은 PP를 0.257에서 **0.466**으로 개선했다.
동일 calibration을 적용한 FT는 0.331에서 0.377로 개선돼, 공정 보정 후에도 calibrated
PP가 +0.088 높았다. 이 수치는 MATR test 확인 후의 개발 결과이며 기존 confirmatory
0.257을 대체하지 않는다.

현재 방어 가능한 주장은 PP-X가 outcome-free contract로 후보를 제한하고
validation evidence가 승인한 prior-residual executor만 사용하며, 근거가
부족하면 fallback/abstention한다는 것이다. 동일 후보예산 비교와 DS03 실패를
함께 보고하며 universal SOTA를 주장하지 않는다.

공통 residual-gain 개선을 13개 설정에서 감사한 결과, validation MSE가 2% 이상 좋아지고 validation unit의 80% 이상에서 이길 때만 모듈을 승인하는 group-robust 규칙은 RWTH를 0.507→0.551, XJTU를 -1.308→-1.229로 개선하고 나머지 base PP를 보존했다. N-CMAPSS causal multiscale 모듈은 ensemble 0.934→0.937로 개선했다. 관측한 설정에서는 3개 개선·10개 동일·0개 악화였지만, 이 규칙은 사후 개발 결과이므로 새로운 untouched cohort 검증이 필요하다.

N-CMAPSS의 0.767은 표형 FT-Transformer가 아니라 과거 window를 입력으로 받는 sequence Transformer 결과다. PP는 같은 causal window의 마지막 값·평균·끝점 기울기를 입력으로 사용했다. 동일 unit-isotonic 조건에서 PP는 0.886±0.004였고, raw PP는 0.931±0.012였다. PAE 결과는 이 PP 비교표에서 제외했다. 표형 FT-Transformer의 동일 프로토콜 실험은 아직 수행하지 않았다.

모든 결과는 이미 관측한 cohort에 대한 개발·사후 비교다. 새로운 untouched cohort에서 모델과 선택 규칙을 고정해 검증하기 전에는 확증적 일반화 결과로 표현하지 않는다.
