# ETO 개선 1차: 구조·학습 제거 실험과 미채택 판단

사용자 요청으로 탈락 후보의 구조 개선을 다시 검토했다. 채택 조건은 그대로
유지한다: 독립 모델링 기여, 기존 적용 범위 유지, full PP-X와 Engression 모두에
대한 우월성이 필요하다. 이번 개선안은 조건을 충족하지 못해 **미채택**이다.

## 변경과 가설

1. `anchored_eto`: 기존 GRU/ODE/decoder를 유지하되
   `y_last + decoder(flow(z,h)) - decoder(z)`로 출력한다.
   h=0에서 마지막 실제 관측값을 잇도록 강제한다. 독립 노벨티가 아닌 제거 실험이다.
2. `profiled_fixed`: 관측 가능한 진행좌표의 기저
   `phi(s)=[s, tau_k(1-exp(-s/tau_k))]`와 regularized least squares로
   prefix별 계수 c를 계산한다. 출력은
   `y_last + [phi(s_last+h)-phi(s_last)]c`다. 임의 latent decoder 및 수치 ODE
   solver를 없애고 관측 이력의 정보로 국소 변화율을 식별하는 baseline이다.
3. `profiled_learned`: 위 선형 solve를 미분해 공통 tau, ridge 강도, 계수의
   중심값을 source 미래 관측 손실로 학습한다. 총 11개 학습 파라미터다.
   test prefix에서는 optimizer를 돌리지 않고 고정된 사전 정의 solve만 수행한다.

두 profile 모델의 signed coefficients는 회복/증가/감소를 허용한다. 양의 tau는
relaxation mode의 안정성만 주며 전체 출력의 단조성이나 정확도를 보장하지 않는다.
시간 외의 **순서가 있는** 물리 좌표를 받을 수 있지만, 실제로 시험한 것은 NASA
시간축뿐이다. unordered operating covariate나 scalar RUL 적용 범위는 미입증이다.

## 고정 비교 조건

이전에 열어본 NASA 4 cells·48 queries에 대한 post-test 개발 결과다. 추가
코호트나 prospective evidence로 부르지 않는다. train/validation horizon은
1/2/4, test는 8/16/32이며 prefix 16개 관측, fold별 2 train/1 val/1 test cell을
그대로 유지했다. 모델/결과 SHA256과 정확한 query ID·truth 일치를 확인한 뒤
기존 ETO·PP 코어·Engression 예측을 재사용했다.

새 변형은 최대 150 epochs, validation-only selection, 동일 clipping,
전부 보고하는 3개 ablation이다. profile의 고정 초기화와 full-batch 계산은
결정론적이므로 3 seed 복제 결과는 완전히 동일하다. seed 안정성 또는 독립적인
앙상블 증거로 해석할 수 없다. 모델 크기도 달라 동일 계산량 비교가 아니다.

## 결과 (Ah, cell 균등 평균)

| 모델 | 평균 RMSE | 32주기 RMSE | 최악 cell RMSE | R2>0 cell | R2>0 cell×horizon |
|---|---:|---:|---:|---:|---:|
| 기존 ETO | .07746 | .11409 | .11888 | 3/4 | 4/12 |
| 관측값 연결 ETO | .08455 | .11970 | .11810 | 3/4 | 3/12 |
| 고정 profile | .07918 | .11509 | .11397 | 2/4 | 2/12 |
| 학습 profile | .07886 | .11469 | .11397 | 2/4 | 2/12 |
| PP 코어 (full PP-X 아님) | .08849 | .12993 | .10657 | 2/4 | 4/12 |
| 공식 Engression | .07292 | .10380 | .09568 | 2/4 | 4/12 |

모든 모델이 48/48 query에 finite prediction을 냈다. 분포 예측구간의 coverage는
측정하지 않았다. 양수 R2 수와 예측구간 coverage를 혼용하지 않는다.

학습 profile의 B0018 RMSE는 .05005로 기존 ETO .08440보다 낮아졌으나,
B0005/B0007은 악화됐다. B0005/B0006/B0007에서 validation은 epoch 0을
선택했다. B0018만 epoch 150을 선택했다. 따라서 differentiable dictionary
학습의 이득이 여러 cell에서 일관되게 확인됐다고 말할 수 없다.

세 변형 모두 '각 cell·horizon에서 두 비교 모델보다 낮은 RMSE' 조건은 실패다.
배터리별 사후 model switching이나 test-best ensemble로 이를 감추지 않았다.

## 노벨티와 적용 범위

[Optimized DMD / variable projection](https://arxiv.org/abs/1704.02343)는
exponential basis와 계수 추정을 이미 다룬다. 이번 differentiable prefix solve와
공통 사전값 학습을 그것과 구별되는 논문 기여라고 확정할 근거는 없다.
신규 명칭이나 parameter 수 감소 자체를 노벨티로 인정하지 않는다.

full PP-X 비교 및 기존 9+5 설정의 커버리지는 이번 실험에 포함되지 않았다.
소규모 선별에서 이미 실패한 후보를 full-suite 성공으로 제시하지 않는다.

## 결론과 코드 상태

구조 변경과 학습 경로는 구현했고 진단은 검증했다. **성공적인 성능 개선 및
후속 논문 모델 채택에는 이르지 못했다.** 새 module은 실험용으로 보존하며
PP-X 기본 경로나 package 최상위 export에 연결하지 않았다.

- `src/pp_extrapolation/profiled_coordinate_flow.py`
- `experiments/eto_profiled_improvement_screen.py`
- `tests/test_profiled_coordinate_flow.py`
- `results/eto_profiled_improvement_screen_v1/protocol.json`
- `results/eto_profiled_improvement_screen_v1/results.json`
- `results/eto_profiled_improvement_screen_v1/predictions.npz`

테스트: 경계에서 정확한 관측값 연결, 좌표 평행이동, NaN padding 격리,
batch 독립성, 선형 solve를 통한 gradient, 잘못된 입력 거부를 검사했다.
관련 7 tests 통과. 저장된 모든 모델의 RMSE를 별도 재계산하여 일치 확인했다.

재실행은 `python experiments/eto_profiled_improvement_screen.py`이며 이미 존재하는
output directory를 덮어쓰지 않도록 거부한다.
