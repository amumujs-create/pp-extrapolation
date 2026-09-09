# NASA milling 실행 명세

## 목표
공식 label boundary0.50을 유지하면서 prior-only, 경험적 보정, NN residual의 기여를 분리하고 새 material에 강건한 PP를 만든다.
현재0.341은 test에서 NN residual이 꺼진 결정론적 수식 결과다. 5-seed 안정성 증거가 아니다.
+0.03은 실증된 inspection delay가 아니라 경험적 effective-margin offset이다. NN 개선 수치로 표기하지 않는다.

## 읽을 파일
- `../ca-css-ncmapss/nasa_milling_causal.py`: BOUNDARY, _one_split, prepare_causal_milling, CORE_HASHES
- `experiments/milling_locked_transfer.py`: strict wear-tail mask/FEATURES
- `experiments/milling_boundary_quotient_route.py`: 현재 MAE offset 선택과 residual training
- `src/pp_extrapolation/adaptive_routes.py`: inspection_boundary_quotient
- raw: `../ca-css-ncmapss/data/nasa_milling/mill.mat`

## M0. label/split 고정
train cases [1,2,3,9,10,11], validation [12], test [5,8,13,14,15,16].
row eligibility와 train health q60 cutoff를 기존대로 유지. 실제 test는 eligible5 units/10 rows이며 case16이 빠질 수 있다. 이를 임의 추가/제거하지 않는다.
train33 rows, validation4 rows, test10 rows를 manifest에 기록하고 기존 row_id/time/y 대조.
RUL은 VB>=0.50이 처음 관측된 time까지의 시간이다. continuous crossing time으로 label을 변경하지 않는다.
material/DOC/feed는 실제 raw record에서 읽는다. test material=2를 상수로 hardcode하는 현재 gate는 교체한다.

## M1. 반드시 함께 실행할 최소 비교
| ID | 정의 | 학습 대상 |
|---|---|---|
| Q0 | max(0,0.50-health)/causal_rate | 없음 |
| Q1 | max(0,0.50+delta-health)/causal_rate | delta |
| Q2 | max(0,a*Q0+b) | a,b |
| N0 | 같은 정보의 MLP | NN |
| P0 | Q0+signed NN residual | NN |
| P1 | Q1+signed NN residual | delta와 NN |
| P2 | P1, residual=0 | Q1과 같아야 함 |
N0 입력에는 health/rate/elapsed/previous_interval와 Q0를 준다. 같은 정보 제공 없이 물리 prior를 PP만 주고 범용 NN과 공정 우월성을 주장하지 않는다.
가능하면 동일 feature의 GroupDRO도 공통 budget으로 재실행. 기존 -0.691은 과거 reference로 분리.

## M2. 보정 선택의 안정성
현재 validation4점/1unit으로 MAE를 골랐고 이전 대화에서 test offset sweep도 했다. 이를 사전 고정 성공으로 설명하지 않는다.
신규 outer test는 같은 것이므로 status는 계속 retrospective_development.
train 내부 held-unit folds로 보정법(RMSE/MAE/Huber), offset 제약, regularization을 선택한다.
원칙: inner training mask는 원래 outer train33행 안에서만 만든다. 원래 제외된 train unit의 late labels를 추가 학습하면 별도 expanded-training 실험으로 분리.
권장 bounded offset candidates=[0,.01,.02,.03,.04,.05]; negative offsets는 필요 시 별도 사전 정의 후보로 포함. 마지막 test 최고값을 보고 범위 변경 금지.
inner에서 정한 보정법을 공식 validation에 적용해 delta 최종 fit. 목적함수와 fitting data를 따로 기록.
leave-one-validation-row-out delta 분산은 sensitivity만 보고한다. 같은 unit4행을 독립 cohort4개라고 보지 않는다.
추가로 delta=0을 필수 대조군으로 유지. 큰 offset이 요구되면 time discretization 설명보다 model misspecification 가능성을 기록.

## M3. signed residual trainer 수정
현재 코드의 residual+offset을 기존 비음수 clipped MLP로 학습하면 residual이 임의 범위에 묶인다.
신규 `src/pp_extrapolation/known_boundary_pp.py`:
- r_target=y-Q. signed 값 그대로 허용.
- train residual scale로 표준화하되 y를 정규화할 때와 단위를 구분.
- `y_pred=max(0,Q+g*r)`만 최종 nonnegative 처리. train_max cap 기본 사용 안 함.
- checkpoint score는 원래 RUL MSE.
- g 후보 {0,.1,.3,1}; validation/inner folds로 선택. 데이터셋 이름 분기 금지.
- 경계에서0이라는 구조 보장을 주장하려면 offset>0, additive residual과 양립 여부 검사. offset>0인 현재 수식은 health=0.50에서0을 보장하지 않는다.

## M4. 고장 속도에 따른 구조 후보
급격한 material shift에 직접 시간을 회귀하기보다 단위 없는 시간을 학습:
`q=max(boundary-health,0)/max(causal_rate,rate_floor)`
`y_pred=q*softplus(k_theta(prefix_context))+interval*softplus(d_theta(prefix_context))`.
여기서 interval은 현재까지 관측된 previous_interval만. 미래 검사 간격/실제 crossing gap 사용 금지.
이 구조는 제안이며 검증되지 않았다. 먼저 constant k/d로 저차원 baseline, 다음 NN k/d 순서로 ablation.
rate_floor는 train rate의 하위 quantile 또는 고정 numerical floor를 train/validation에서 선택. test 분포로 재설정 금지.
k/d를 동시에 학습하면 서로 보상하는 비식별성이 생기므로 regularization과 constant head 대조를 둔다.
새 material에서 residual을 끌지는 실제 context coverage로 판단하되, 끈 결과는 prior-only이다. gate 성공과 NN 기여를 혼동하지 않는다.

## 구현할 CLI
```
python experiments/milling_boundary_audit_v2.py --stage audit
python experiments/milling_boundary_audit_v2.py --stage select --output results/milling_boundary_v2
python experiments/milling_boundary_audit_v2.py --stage evaluate --manifest results/milling_boundary_v2/selection_manifest.json
```

## 통과/실패 기준
- Q1과 residual=0 P2 예측 allclose. 같으면 PP NN 개선이라고 부르지 않는다.
- deterministic formula는 한번 계산하고 model_type=deterministic.
- NN은 실제5개 state_dict/seed/epoch/loss history 저장.
- 5unit cluster bootstrap으로 RMSE 차이 CI. n=5이므로 불확실성 크게 보고.
- label boundary0.50과 prediction effective boundary0.50+delta 모두 저장.
- positive R²나0.3 초과라도 NN 기여, 선택 안정성, 동일 정보 비교가 각각 통과해야 강건성 개선으로 채택.
