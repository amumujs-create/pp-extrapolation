# FEMTO latent failure-mode survival PP 설계 및 실행

## 결론

설계·구현·5-seed 실행을 완료했지만 FEMTO를 해결하지 못했다. validation은 3-mode 구조를 선택하지 않았고 single-mode PP를 선택했다. 최종 prediction ensemble pooled R²는 **−1.864**, matched direct total-life GRU는 **−2.544**다. PP가 같은 survival decoder의 direct 모델보다는 높지만 절대 성능이 음수라 성공으로 채택하지 않는다.

이 결과는 이전 5+1 corrected GRU-PP −2.898보다 개선됐지만, 기존 raw-waveform direct ensemble 0.075보다 낮다. 따라서 최종 FEMTO 행을 대체하지 않는다.

## 모델 구조

목표를 매 시점의 RUL 직접 회귀에서 total lifetime 추정으로 바꿨다.

    h_t = GRU(H_{<=t})
    pi_t = softmax(g(h_t))
    delta_k = B tanh(r_k(h_t) / B)
    log T_hat = log T_prior(condition) + sum_k pi_tk delta_k
    RUL_hat = 100 softplus((exp(log T_hat) - elapsed_t) / 100)

`T_prior`는 train bearing만 사용한 condition별 log-total-life Ridge다. unit-balanced weight를 사용해 긴 bearing의 행 수가 prior를 지배하지 않게 했다. NN은 total lifetime prior를 log scale에서 제한적으로 보정한다. decoder는 예측 수명이 현재 elapsed보다 작을 때도 미분 가능한 양의 RUL을 낸다.

causal 입력은 condition, elapsed, 현재 진동 상태, 4/16 관측 전 변화, 유효 history 길이다. 각 bearing의 현재 시점까지 이용 가능한 초기 관측으로 정규화한다. summary 14개와 summary+spectrum 110개 후보를 비교했다.

학습 loss는 normalized raw-RUL MSE와 log-total-life MSE를 결합했다. 3-mode 후보에는 mode 사용 균형과 entropy penalty를 추가했다. direct 비교군도 같은 GRU, causal 입력, total-life decoder와 학습 예산을 사용하며 analytic prior만 제거했다.

## 선택과 결과

train은 Bearing1_1, 1_2, 2_1, 2_2, 3_1, validation은 Bearing3_2, test는 공식 truncated Test_set 11개 bearing endpoint다. 구조 선택은 validation all-prefix MSE로 수행했다. 데이터셋은 이전 연구에서 이미 test를 본 retrospective development 상태다.

선택된 PP는 summary 입력, modes=1, residual bound=.5, lr=.001이었다. 3-mode summary 후보 validation MSE는 약 64.6–64.9M, 선택된 1-mode는 62.9M이었다. spectrum 후보도 더 좋지 않았다. 따라서 latent mode가 구현됐지만 관측 데이터가 mode 분리를 지지했다는 주장은 할 수 없다.

PP seed별 test R²는 −3.432, −1.732, −1.558, −3.464, −3.165다. ensemble −1.864는 우연한 단일 seed 성공이 아니다. direct seed는 모두 약 −2.53∼−2.56으로 안정적으로 실패했다.

## 왜 mode 구조가 해결하지 못했나

mode gate는 bearing identity label을 받지 않으며 causal vibration history에서만 mode를 찾아야 한다. 그러나 train은 조건별 1–2개 complete bearing뿐이고 validation condition3에는 실질적으로 train source가 Bearing3_1 하나뿐이다. validation으로 3개의 미래 고장 유형을 식별할 정보가 없다.

NN의 mode 이름만 여러 개로 만드는 것은 새로운 failure physics를 만들어내지 않는다. mode가 식별되려면 최소한 다음 중 하나가 필요하다.

- 충분한 complete bearing과 failure-mode annotation
- 고장 위치/유형을 구분하는 sensor 또는 inspection label
- 다음 vibration state를 예하는 auxiliary task가 실제 failure progression과 연결된다는 unit-held-out 근거
- 더 많은 source complete bearing 또는 source failure-mode 정보

현재 데이터로는 첫 두 조건이 없다. 세 번째는 아직 검증하지 않았고, 네 번째를 inductive 결과로 섞으면 안 된다.

## 다음 설계 판단

FEMTO를 PP의 공용 성능 표에서 무조건 이겨야 하는 데이터로 다루기보다, **prior admissibility failure 사례**로 쓰는 것이 현재 증거에 맞다. 프레임워크는 validation에서 prior가 도움이 되는가만 보지 말고, source validation이 target failure modes를 덮는지도 평가해야 한다. condition당 complete unit 수가 2 미만이거나 OOF mode stability가 낮으면 PP prior를 승인하지 않는 coverage gate가 필요하다.

후속 연구도 inductive extrapolation으로 제한한다. Test_set prefix를 label 없이 representation 학습에 넣는 transductive self-supervised 학습과 TTA는 사용하지 않는다. Source train/validation만으로 waveform representation, gate, survival head 및 모든 normalization 통계를 확정하고, test에서는 개별 unit의 causal history를 frozen model에 입력한다. 다른 test unit의 정보는 공유하지 않는다.

## 산출물

- 구현: `experiments/femto_latent_survival_pp_v13.py`
- 선택 기록: `results/femto_latent_survival_pp_v13/selection_manifest.json`
- 5-seed loss 곡선: 같은 결과 폴더의 `direct_curve_*.json`, `pp_curve_*.json`
- 예측: `direct.npz`, `pp.npz`
- 최종 지표: `results.json`

모든 파일은 기존 결과를 덮어쓰지 않는 v13 경로에 저장했다.
