# Health–RUL relationship shift 개선: Boundary-Quotient PP

## 문제와 구조

기존 PP는 health와 RUL의 관계가 train cohort에서 test cohort로 전이된다고
암묵적으로 가정한다. MICH에서는 이 관계가 바뀌어 residual이 epoch 0에서
꺼졌고, 기존 PP는 Ridge와 같은 pooled R² `-1.522`를 냈다.

새 **Boundary-Quotient PP (BQ-PP)**는 다음과 같이 예측을 분해한다.

\[
\widehat{RUL}=m\,\mathrm{softplus}\left(a^\top z+b+B\tanh f_\theta(z)\right)
\]

- `m`: 현재 health와 알려진 EOL 경계 사이의 차원 없는 margin
- `aᵀz+b`: train에서 먼저 맞춘 뒤 동결하는 RUL/margin affine tail
- `fθ`: 인과적 8-step health·rate history로 quotient의 편차만 학습하는 NN
- `B`: NN이 affine tail을 무제한으로 뒤집지 못하게 하는 residual bound

EOL에서는 `m=0`이므로 RUL이 정확히 0이다. 나머지 health–RUL 관계는
고정 식으로 주지 않고 quotient network과 동결 affine path가 학습한다.

## 실험 설계

- Sunwoda, RWTH, MICH의 세 배터리 cohort를 하나의 BQ-PP로 공동 학습했다.
- dataset ID와 unit ID는 입력하지 않았다.
- 모든 feature/target scale은 train으로 적합했다.
- 18개 architecture/regularization 조합은 validation dataset-macro normalized MSE로만
  선택했다. 선택 결과는 width 64, affine alpha 1000, residual bound 2이다.
- seed 42–46에서 validation으로 epoch를 선택한 뒤 train+validation을 그 epoch 수만큼
  재학습했다.
- source test는 unseen unit이며 train/validation보다 늦은 health 범위이다.
- 이 실험은 기존 test 결과를 본 후의 후향적 개발 실험이다.

## 결과

| dataset | 기존 PP pooled R² | affine quotient only | BQ-PP single seed R² | BQ-PP 5-seed pooled R² | BQ-PP macro R² | 별도 PAE pooled R² |
|---|---:|---:|---:|---:|---:|---:|
| Sunwoda | 0.862 | 0.340 | 0.909±0.020 | **0.939** | **0.937** | 0.677 |
| RWTH | 0.506 | 0.575 | 0.871±0.016 | **0.878** | **0.885** | 0.737 |
| MICH | -1.522 | -3.036 | 0.468±0.0004 | **0.468** | 0.230 | **0.798** |

세 dataset의 pooled R² 단순 평균은 기존 PP `-0.051`에서 BQ-PP `0.762`로
증가했다. BQ-PP의 dataset-macro unit R²는 `0.684`로, 별도 PAE 배터리
공유 모델의 `0.659`보다 높다. 다만 MICH 단일 dataset에서는 PAE가 더 높다.

MICH 8 units 중 7개가 양의 R²를 얻었고, unit 31은 `-1.447`로 남았다.
그러므로 pooled 성능 복구는 확인됐지만 모든 새 cohort의 관계 변화를 해결했다고
주장할 수는 없다.

## 기전 ablation

| BQ-PP feature | Sunwoda pooled R² | RWTH pooled R² | MICH pooled R² |
|---|---:|---:|---:|
| current margin only | 0.590 | -0.378 | 0.672 |
| margin history | 0.550 | -0.197 | **0.715** |
| margin history + cycle | 0.777 | -0.251 | 0.702 |
| full margin + rate history | **0.939** | **0.878** | 0.468 |

MICH는 단순 head가 test에서 더 높았지만, validation은 full-history head를 명확히
선택했다. 따라서 test 결과를 보고 MICH만 단순 head로 바꾸는 route는 채택하지
않았다. 이는 현재 validation support에서 보이지 않던 새 late-life mechanism은
완전히 식별할 수 없다는 한계를 보여준다.

## 논문에서의 판정

BQ-PP는 **관측 가능한 EOL health 경계가 있는 continuous-degradation domain**에
적용하는 PP의 선택적 module로 유효하다. 세 데이터셋 모두에서 기존 PP를
개선했고 MICH shape failure를 양의 pooled R²로 복구했다.

다음 제한은 유지한다.

1. EOL health 경계가 없거나 장비별로 바뀌면 BQ-PP를 적용할 수 없다.
2. 관측 이력에 전조가 전혀 없는 새 failure mechanism은 학습만으로 알 수 없다.
3. MICH unit 31처럼 개별 unit 관계가 큰 폭으로 바뀌는 경우는 calibration unit,
   운전 조건, 또는 추가 state 센서가 필요하다.
4. 현재 결과는 후향 개발 결과이므로, 모듈과 선택 규칙을 고정한 뒤
   새 cohort에서 확증해야 한다.

재현 코드는 `experiments/boundary_quotient_pp_batteries.py`, feature ablation은
`experiments/boundary_quotient_feature_ablation.py`, raw 결과는
`results/boundary_quotient_pp_batteries_v1/results.json` 및
`results/boundary_quotient_feature_ablation_v1/results.json`에 있다.

선행연구 중복과 matched affine ablation을 포함한 노벨티 판정은
`BQ_PP_NOVELTY_AUDIT_KO.md`에 별도로 정리했다.
