# 결과 저장·검수·다른 모델에 넘길 프롬프트

## 신규 디렉터리별 산출물
```
results/<dataset>_<experiment>_v2/
  protocol.json
  data_manifest.json
  schema_audit.json
  trials.jsonl
  selection_manifest.json
  checkpoints/<model>_<seed>.pt
  loss_history/<model>_<seed>.json
  predictions.npz
  metrics.json
  bootstrap.json
  REPORT_KO.md
```
큰 raw 데이터와 캐시는 커밋하지 않는다. report/JSON/필요한 소형 prediction만 기존 저장소 규칙에 따라 관리.

## JSON 필수 항목
protocol: status, task_definition, split_ids, label_definition, allowed_input, history_policy, selection_metric, budgets, seeds.
data_manifest: feature_names, unit/time/row_id count, source file hashes, adapter version, y hash, endpoint rule.
trials: model_id, config, fold, seed, train loss, validation metrics, selected_epoch, runtime, exception.
selection_manifest: selected config/model per candidate family, selection_rule, input/schema hash, code revision 또는 dirty diff hash, created_at.
metrics: per_seed pooled R²/RMSE, seed mean/sample-SD(ddof=1), ensemble pooled R²/RMSE, per_unit, model_type, actual_refit_count.
predictions: y, row_id, groups, time, each model prediction. NN shape=[seed,row], deterministic=[row].

## 통계
R²를 서로 다른 도메인의 seconds/cycles를 섞어 전체 pooled 계산하지 않는다. dataset마다 pooled R²를 계산.
paired unit bootstrap: 동일한 unit resample index로 두 모델의 모든 시계열 행을 함께 재표집하고 pooled RMSE 차이 계산. 20,000회, seed=20260909.
개별 unit이 한 endpoint이면 unit R²는 undefined(null). AE와 aggregate pooled R² 사용.
bootstrap 재표본의 y variance=0이면 R²는 null; RMSE CI는 계산 가능.
5개 seed는 5개 독립 데이터셋이 아니다. seed 변동과 unit bootstrap을 분리.

## 논문용 그림(실제 결과 확보 후)
- train/validation loss curve, 선택 epoch 표시.
- unit별 예측 대 실제 RUL, observed time 축, 원래 단위.
- PP-minus-baseline unit RMSE와 bootstrap CI.
- ablation: 같은 feature/동일 budget에서 component on/off.
- XJTU capped vs uncapped와 progress 변환.
- FEMTO 잘못된 시간열 캐시 vs 올바른 진동 cache 설명 그림. 잘못된 버전을 성능 baseline 승리로 세지 않음.
- milling Q0/Q1/NN residual과 offset sensitivity. inspection 인과효과라고 제목 짓지 않음.
PNG와 PDF/SVG로 저장. 원시 표도 함께 보존.

## 완료 판정
### 데이터 수정 완료
schema/causality/endpoint/row alignment 테스트 통과.
### 비교 완료
같은 input/rows/scaling policy/budget에서 후보와 비교군 재실행, 선택 manifest 존재.
### 모델링 개선 확인
원래 baseline보다 좋아진 것 외에 같은 정보의 강한 NN 대비 PP component 기여가 재현됨. CI가0을 포함하면 유의성 불명으로 기록.
### 전체 강건성 확인
세 설정만 개선해서 전체 데이터 강건성 완료라고 하지 않음. 향후 공통 executor 정책을 먼저 고정하고 기존9개 설정의 regression replay를 별도 승인 범위에서 수행.

## 최종 표 형식
| 데이터셋 | adapter/protocol | 모델 실제 정체 | 비교 모델 | seed R² mean±SD | ensemble R² | RMSE | unit CI | 판정 |
항상 정보변경 여부와 test exposure 상태를 적는다. 이전 잘못된 결과와 새 결과를 같은 protocol로 표시하지 않는다.

## 실행 순서용 체크박스
- [ ] 00 및 해당 데이터 문서를 읽음
- [ ] git status 및 적용 AGENTS 확인
- [ ] 기존 cache/result 보존
- [ ] adapter 테스트 작성·통과
- [ ] 작은 train-only smoke run
- [ ] comparison grid와 선택 metric 고정
- [ ] train/validation selection 완료
- [ ] selection_manifest 기록
- [ ] 최종 test 평가
- [ ] 실제 독립 seed/단위 통계 저장
- [ ] 모든 실패/비개선 포함 report
- [ ] FINAL_PP_BENCHMARK_TABLE_KO.md의 해당 행을 검증된 protocol로만 갱신

## 모델에 복사할 실행 프롬프트
아래를 그대로 전달하고 DATASET 문서명만 바꾼다.

> 저장소 `/Users/baghyeongbae/Desktop/연구/pp-extrapolation`에서 작업하세요. 먼저 `docs/execution_handoff/00_START_HERE_KO.md`, 해당 데이터 문서, `04_REPORT_AND_ACCEPTANCE_KO.md`를 읽으세요. 해당 문서의 단계 순서대로 adapter 오류 수정, 테스트, train/validation 튜닝, 최종 test 평가와 보고서 저장까지 수행하세요. 신규 CLI는 구현할 인터페이스이며 존재한다고 가정하지 마세요. 이번 대상은 DATASET 하나이며 새 cohort나 다른 성공 데이터의 학습/선택/평가를 하지 마세요. 기존 결과는 보존하고 v2 경로에 기록하세요. 성능 향상을 보장하거나 Ridge/수식/NN fallback을 PP 신경망 개선으로 바꿔 말하지 마세요. 막히면 실패한 검증, 재현 명령, 필요한 결정만 구체적으로 보고하고 독립적으로 가능한 작업은 계속하세요. test 결과로 후보/metric을 다시 선택하면 별도 posthoc 버전을 남기세요. 끝에는 실제 점수, 동일 정보 비교 여부, 미완료 항목, 결과 경로를 보고하세요.

FEMTO 문서: `01_FEMTO_KO.md`
XJTU 문서: `02_XJTU_KO.md`
Milling 문서: `03_MILLING_KO.md`

## 공유 파일 충돌 예방
한 모델씩 실행하면 가장 단순하다. 병렬 실행은 사용자가 별도 지시한 경우에만 수행.
서로 다른 작업이 model.py/__init__.py/최종표를 동시에 수정하지 않도록 담당자를 정한다. dataset별 신규 파일에 우선 구현하고 통합 담당자가 공통 API를 반영한다.
