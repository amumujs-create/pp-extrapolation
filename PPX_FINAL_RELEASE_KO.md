# PP-X 최종 개발 버전 v1.0

## 모델 동결 정의

PP-X v1.0은 `prior-residual core + validation-approved executor`다. 모든 optional module을 동시에 켜지 않는다.

- **Core:** causal domain adapter, frozen extrapolative prior/affine path, nonlinear residual
- **선택 executor:** fixed residual bound, support-adaptive dual scale, regime transport, multiscale history
- **Safety:** source evidence가 prior를 승인하지 않으면 neural safety로 fallback

선택과 hyperparameter는 train/validation 또는 group-LOO validation만 사용한다. test에는 frozen forward prediction만 실행한다. TTA, test-batch statistics, test-prefix fitting, test unit 간 정보 공유는 포함하지 않는다.

## 최종 development portfolio

| 데이터셋 | 최종 PP-X executor | pooled R² | 주요 비교군 | 상태 |
|---|---|---:|---|---|
| HUST | regime transport | **.958** | GroupDRO .934 | 개발 우세 |
| Virkler | support-gated prior residual | **.888** | linear-tail RBF .805 | 개발 우세 |
| NASA battery | causal multiscale latent | **.584** | linear-tail RBF .550 | 개발 우세 |
| Sunwoda | fixed bounded boundary quotient | **.939** | linear-tail RBF .838 | 개발 우세 |
| RWTH | fixed bounded boundary quotient | **.878** | V-REx .645 | 개발 우세 |
| MATR2019 | validation-calibrated latent | **.466** | calibrated FT .377 | retrospective development |
| MATR batch 2 | support decay + regime transport | **.862** | V-REx .850 | development; exact 5-seed comparison |
| N-CMAPSS | causal multiscale latent | **.937** | Engression .932 | near tie |
| MICH | support-adaptive dual-scale boundary | **.751** | direct NN .684 | development recovery |
| XJTU | reflected-scale temporal | **.257** | linear-tail RBF −1.418 | retrospective development |
| FEMTO | prior abstention → waveform neural safety | **.075*** | corrected causal GRU −.248 | unstable ensemble-only recovery |
| NASA milling | inspection boundary quotient | **.341** | tuned NN −.476 | retrospective development |

`*` FEMTO는 five-prediction ensemble pooled R²만 양수이며, 각 individual seed R²는 모두 음수다. 논문의 robust success count나 strong-comparator win count에는 넣지 않는다.

## ablation이 정한 최종 선택

| 구성요소 | 최종 지위 | 근거 |
|---|---|---|
| nonlinear residual | core 유지 | Sunwoda·RWTH·MICH에서 유의한 unit-level 개선 |
| fixed residual bound | boundary cohort의 conditional executor | Sunwoda 유의 개선, MICH 유의 악화 |
| support-adaptive dual scale | MICH형 heterogeneous support에서만 승인 | MICH 유의 개선, RWTH 유의 악화 |
| regime transport | group-LOO 승인 시만 사용 | HUST·MATR-b2에서 유의 개선 |
| multiscale history | validation-selected adapter setting | NASA·N-CMAPSS 효과는 작고 통계력 부족 |
| prior abstention | safety fallback | FEMTO는 prospective success가 아님 |

20개 matched ablation에서 BH 보정 뒤 유의한 개선 9개, 유의한 악화 2개, inconclusive 9개다. 따라서 PP-X의 기여는 optional module을 누적하는 것이 아니라 evidence에 따라 **필요한 executor만 채택하는 정책**이다.

## 논문에서 쓸 수 있는 주장

> PP-X preserves an extrapolative prior direction with a nonlinear residual, while source-validation evidence selects only those residual constraints and transport operators supported by physical-unit replication.

이 표는 heterogeneous retrospective development portfolio다. 새 untouched cohort에서 선택 규칙까지 고정한 확증 성능으로 표현하지 않는다.

## 연결 자료

- 최종 benchmark: `FINAL_PP_BENCHMARK_TABLE_KO.md`
- 최종 model contract: `PPX_FINAL_PAPER_MODEL_KO.md`
- full registry: `results/ppx_unified_final_v1/registry.json`
- ablation/statistics: `results/ppx_final_ablation_statistics_v1/REPORT_KO.md`
- competition table: `ALL_DATASET_EXTRAPOLATION_COMPETITORS_KO.md`
- figures: `figures/paper/ppx_final_ablation/`
