# XJTU 실행 명세

## 목표와 한계 해석
현재 -0.843은 logit(progress)를 Ridge로 회귀한 결과이며 NN PP 결과가 아니다.
train y 최대525, test y의70.7%가 이를 넘는다. train-max로 출력 제한 시 oracle R²=-0.516은 해당 제한만의 상한이다. 임의 모델의 성능 상한이 아니다.
고정 총수명 모델의 oracle0.231도 모든 모델의 한계로 확대하지 않는다.

## 입력 파일
- `experiments/xjtu_untouched.py`: build_cache, windows
- `data/xjtu_sy/features_locked.npz`: stats,units,conditions,positions,lives
- `experiments/remaining_failures_structural_pp.py`: 기존 progress route
- `src/pp_extrapolation/model.py`, `adaptive_routes.py`
- `experiments/plain_mlp_ablation.py`

## X0. 순서와 정보 경계
train=37.5Hz11kN, validation=35Hz12kN, test=40Hz10kN 유지.
새 adapter `experiments/xjtu_causal_adapter_v2.py`는 actual raw positions로 정렬하고 observed_time을 반환한다.
현재 positions(rows)는 y 길이를 보고 arange(8,...). 새 구현은 입력 positions를 직접 전달하여 누락 recording에도 정확하게 한다.
windows(...include_targets=False)에도 현재 life를 먼저 읽는 구현이 있으므로 입력 생성과 target 생성을 분리한다.
Test input builder에 lives를 삭제한 dict를 넣어도 동일 입력이 나와야 한다.
각 t의 입력: 최근 8개 stats, 현재까지 baseline 대비 변화, 실제 elapsed/log elapsed, valid length, 알려진 speed/load.
full-unit lifetime, 최대 position, unit 데이터 전체 길이를 feature로 쓰지 않는다.

## X1. 우선 출력 상한·입력·target 변환 분리
신규 `experiments/xjtu_scale_benchmark_v2.py`.
동일 feature block별로 다음 모델을 비교한다.
| ID | 모델 | target | 출력 |
|---|---|---|---|
| D0 | plain MLP | raw RUL | [0,train_max] |
| D1 | plain MLP | raw RUL | lower_only |
| P0 | 기존 affine+NN PP | raw RUL | [0,train_max] |
| P1 | 기존 affine+NN PP | raw RUL | lower_only |
| R0 | Ridge | logit progress | quotient 복원 |
| N0 | MLP | logit progress | quotient 복원 |
| P2 | affine+NN PP | logit progress | quotient 복원 |
feature blocks: 기존44개 특징 / 동일44개+elapsed+log elapsed. 모든 모델에 동일 block 제공.
R0는 비교군 이름을 `progress_Ridge`로 쓴다. P2와 구분한다.

## X2. 변환 target 전용 trainer
`p=t/(t+y)`, z=logit(clip(p,eps,1-eps)), eps=1e-4를 train에서 계산.
`y_hat=t*exp(-z_hat)`; t=0이면 이 역변환을 쓰지 않고 첫 valid prefix부터 평가. 기존 window8은 t>0.
중요: 기존 fit_pp는 y 비음수 clip과 max scaling이 전제다. 음수 z를 그대로 넣지 않는다.
신규 `src/pp_extrapolation/progress_pp.py`에 signed real target trainer 구현:
- train만으로 x scale과 z center/std fit.
- z 표준화 공간에서 affine init+NN residual.
- validation checkpoint 점수는 역변환된 원래 RUL의 MSE.
- z를 [0,1]로 clamp하지 않는다.
- exp overflow 방지용 numerical clamp는 사전 고정 예: [-12,12]. clamp 활성 비율 저장. 이는 수명 prior가 아닌 numerical safeguard임을 명시.
- supervised loss 후보는 z-space Huber 또는 original-RUL scaled MSE. 목적함수 변경은 별도 후보로 manifest에 기록.

## X3. 강건성 후보
단일 train 조건에서 speed/load가 상수이므로 그 조건에 대한 기울기는 train에서 식별되지 않는다. zero-variance condition 열을 scale=1 처리했다는 이유만으로 조건별 scale을 학습했다고 주장하지 않는다.
우선 prefix 변화와 elapsed를 쓰는 개인별 상태 표현을 학습한다.
추가 loss 후보:
1. 시간 일관성: `Huber(((r_i-r_j)-(t_j-t_i))/s_y)`.
2. unit-balanced supervised loss.
3. prior trust λ=0을 포함한 affine/residual 강도 선택.
기본 supervised 모델을 먼저 완료하고 loss는 한 개씩 추가한다. 모두 합쳐 좋아졌다고만 보고하지 않는다.
Temporal consistency가 단지 elapsed를 외워 constant failure age를 만드는지 unit별 predicted_failure_age=t+r 곡선을 저장해 확인한다.

## X4. 튜닝/비교
공통 grid/seed 정책 적용. capped/uncapped 정책은 각 모델 validation checkpoint에도 동일 적용.
내부 robustness 점검은 train5 bearing을 held-unit로 순환하며 관측 prefix를 끝에서20%,40% 잘라 가상 외삽 평가한다. 이 cutoff는 train-only 검증 설정이며 실제 test row를 바꾸지 않는다.
최종 선택에 쓰는 목적과 공식 validation의 역할을 미리 결정한다. 추천: inner folds로 loss/feature 후보 축소, 공식 validation으로 최종 HP/checkpoint. fold별 전처리 재학습.
같은 데이터라 결과를 본 뒤 이를 untouched로 재분류하지 않는다.

## 구현할 CLI
```
python experiments/xjtu_causal_adapter_v2.py --audit-only
python experiments/xjtu_scale_benchmark_v2.py --stage select --output results/xjtu_scale_v2
python experiments/xjtu_scale_benchmark_v2.py --stage evaluate --manifest results/xjtu_scale_v2/selection_manifest.json
```

## 판정
1. D1>D0이면 상한 제거 효과. PP만의 novelty 아님.
2. N0>D1이면 target 변환 효과. PP affine/residual 기여는 P2 vs N0로 판단.
3. P2가 N0/R0보다 낮으면 변환 Ridge를 PP 승리로 바꾸지 않는다.
4. positive pooled R², per-unit error, seed variance 각각 기록. 양수 아니어도 재현 가능한 감소는 개선으로 기록하되 해결이라고 쓰지 않는다.
5. 모든 후보가 부진하면 unit별 early/mid/late error와 scale bias를 보고. 추가 데이터 필요성을 가설로 제시하되 불가능 증명으로 쓰지 않는다.
