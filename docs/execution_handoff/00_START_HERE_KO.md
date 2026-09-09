# PP 개선 실행 인수인계: 이 파일부터 읽기

> **2026-09-09 범위 확대:** 사용자 최신 요청은 기존 나머지 데이터셋도 모두 개선 대상으로 포함하는 것이다. 최신 전체 설계는 `07_ALL_DATASETS_STRUCTURAL_IMPROVEMENT_KO.md`, 베어링 상세는 `06_DEEP_PP_RECOVERY_PLAN_KO.md`를 우선 읽는다. 아래의 3개 설정 한정 및 기존 성공 설정 제외는 이전 작업 범위다. 기존 최종 route 재현 시에는 원래 refit 규칙을 보존하고, 새로운 공통 프로토콜과 구분한다. 보류 cohort 제외는 계속 유효하다.

## 사용자 목표와 이번 범위
- PP의 실제 모델 성능과 강건성을 개선한다. FEMTO, XJTU, NASA milling 세 설정이 대상이다.
- 기존 성공 설정과 보류한 새 cohort는 이번 작업에서 학습·선택·평가에 사용하지 않는다.
- 이번 문서는 구현할 작업 명세다. 문서에 제안된 모델은 아직 구현/검증 완료가 아니다.
- 승리나 R²>0.3을 보장하지 않는다. 새로운 test 결과로 선택한 모델을 train/validation-only 선택이라고 보고하지 않는다.
- 순서: FEMTO 데이터 수정 → FEMTO 공정 비교 → XJTU → milling → 공통 인터페이스 통합.
- 사용자 요청은 실행 인수인계 작성이다. 이 문서를 작성한 작업에서는 대규모 재학습을 실행하지 않았다.

## 위치
작업 저장소: `/Users/baghyeongbae/Desktop/연구/pp-extrapolation`
기존 로더 저장소: `/Users/baghyeongbae/Desktop/연구/ca-css-ncmapss`
상대 경로는 모두 첫 저장소 기준이다. 실제 파일시스템은 한글 유니코드 정규화가 다를 수 있으므로 cwd에서 Path.resolve()를 사용한다.

## 문서 읽기 순서
1. 이 파일
2. `01_FEMTO_KO.md`
3. `02_XJTU_KO.md`
4. `03_MILLING_KO.md`
5. `04_REPORT_AND_ACCEPTANCE_KO.md`
추가 근거: 루트 `MODELING_IMPROVEMENT_REAUDIT_KO.md`.

## 반드시 바로잡아야 하는 기존 주장
| 기존 숫자 | 실제 의미 | 현재 취급 |
|---|---|---|
| FEMTO -0.571 | 진동 대신 시간 열을 사용한 캐시 + prefix NN | 잘못된 입력의 과거 결과. 새 비교 기준 아님 |
| XJTU -0.843 | Ridge로 logit progress를 회귀한 모델 | NN PP의 개선 증거 아님. 비교 후보로 유지 |
| milling 0.341 | NN 비활성화, `(0.53-health)/rate` | 경험적 보정 수식. NN 기여 0 |
| 모든 설정 PP 우세 | 입력/구조가 달라진 결과를 이전 비교군과 비교 | 동일 정보 재실행 전 보류 |
| 추가 데이터 없으면 양수 불가능 | 제한된 모델과 통계로 일반화한 판단 | 입증되지 않음 |

## 공통 실험 계약
### 데이터
각 adapter는 다음을 반환한다.
```
TrainSplit: x, y, groups, row_id, observed_time, feature_names
ValSplit:   x, y, groups, row_id, observed_time, feature_names
TestInputs: x, groups, row_id, observed_time, feature_names  # y 없음
TestTargets: row_id, y  # 평가 코드만 읽기
```
row_id는 dataset/unit/raw recording index로 만든 고유 문자열이다. x/y/groups를 따로 정렬하지 않는다. merge(validate='one_to_one')와 row_id 일치를 assert한다.
전처리/표준화/PCA/결측치 대체는 fit 대상 train에서만 학습한다. inner fold마다 다시 fit한다.
미래 삭제 불변성: 시점 t 이후 파일을 없애거나 값을 바꿔도 시점 <=t의 특징과 예측은 동일해야 한다.
전체 test 길이, 최종 파일번호, test max cycle, test failure age를 개별 시점의 입력으로 사용하지 않는다.
알려진 운전조건은 허용하지만 feature 입력인지 gate context인지 기록한다.

### 학습/선택
- 후보 탐색 단계의 함수는 train과 validation만 인자로 받는다. test 예측 호출 금지.
- 설정 선택: selection seeds 42,43,44. 최종 독립 학습: 42..46.
- validation 목표: **seed별 pooled MSE의 평균**을 기본으로 한다. inference ensemble만 좋은 후보를 고르는 일을 방지한다.
- validation ensemble MSE도 보조 저장. unit별 normalized RMSE·worst-unit RMSE도 저장하되 test를 본 후 주 선택 목적을 바꾸지 않는다.
- 선택에 inner grouped CV를 썼다면 공식 validation은 checkpoint/후보 확정 용도로 한정한다. 두 수준의 역할을 명세에 먼저 기록한다.
- 최종 5 seeds에서는 선택한 구조/전처리/hyperparameter를 고정한다. checkpoint는 validation으로 고른다.
- outer train+validation 합쳐 재학습하지 않는다. 필요하면 별도 프로토콜로 분리한다.
- deterministic 수식/Ridge는 1회 모델이다. 같은 예측을 5번 복사해 재학습 안정성으로 보고하지 않는다.

### 탐색 예산
1차 NN grid: width [16,32,64] × learning_rate [0.0003,0.001] × weight_decay [0.01,0.1] = 12 후보/모델.
max_epochs=400, patience=60, batch_size=min(512,n_train), group-balanced loss, gradient norm clip=2.
seed=42로 12 후보를 평가하고 상위 3개만 42..44로 확정한다. 기존 seed42 fit은 재사용한다.
각 비교 모델도 동일 12+상위3 정책. PP-only prior penalty 등 추가 후보 비용은 숨기지 말고 비교군에 같은 추가 학습 예산의 확장 grid를 제공하거나 별도 비용 비교로 표기한다.
최고 후보가 epoch 상한에 도달하고 validation이 여전히 개선 중이면 train/validation만으로 800 epoch 연장. 동등한 현상이 있는 비교군에도 같은 규칙 적용.
width=8은 group 수 공식으로 강제하지 않는다. 소형 후보 추가가 필요하면 PP/MLP 양쪽에 같은 예산으로 추가한다.

### 운영
- 큰 cache는 versioned 이름으로 새로 생성. 기존 결과/캐시를 덮어쓰지 않는다.
- 이미 관측한 데이터이므로 status는 `retrospective_development`.
- `selection_manifest.json`을 test 평가 전에 기록하고 SHA-256을 결과에 넣는다.
- test를 본 뒤 설계를 바꾸면 v2 폴더에 새 개발 실험으로 기록한다. v1 실패도 보존한다.
- 중단 후 재개는 config hash/seed/fold가 완전 일치하는 완료 trial만 재사용한다.
- 패키지의 다른 프로젝트 파일을 수정하지 말고 새 PP-local adapter로 격리하는 것을 우선한다.
- 공개 push는 해당 실행 작업에서 사용자 승인을 확인한 뒤 진행한다.

## 실제 실행 가능 환경 점검
아래는 기존 환경 점검 명령이다.
```sh
cd '/Users/baghyeongbae/Desktop/연구/pp-extrapolation'
git status --short
python -c 'import numpy, scipy, pandas, torch, sklearn; print(torch.__version__)'
PYTHONPATH=src pytest -q
```
아래 개별 문서에 나온 신규 CLI는 **구현해야 할 인터페이스**다. 파일이 없는데 완료된 실행 명령으로 보고하지 않는다.

## 공통 수정할 모델 결함
`src/pp_extrapolation/model.py`의 fit_pp checkpoint와 predict는 train 최대 y로 clip한다. output policy를 명시적 옵션으로 만들고 validation/predict에 동일 적용한다.
기존 기본값은 유지해 성공 설정 회귀를 막는다. 신규 실험에서는 lower_only와 train_max를 대조한다. `plain_mlp_ablation.py`와 비교 trainer에도 동일 정책을 구현한다.
`extrapolation_competitors_matr.py`의 monotone은 x[:,0]을 자동 선택한다. FEMTO x[:,0]은 condition이므로 이 페널티는 부적절하다. `monotone_feature`/방향을 명시하게 하고 category에는 적용하지 않는다.
