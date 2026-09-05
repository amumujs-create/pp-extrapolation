# PP soft-anchor 개선 실험

기존 PP는 validation에서 선택한 affine 외삽 경로를 완전히 고정하고 NN 잔차만
학습했다. 개선형은 affine 계수도 학습하되 다음 prior anchor loss를 적용한다.

`L = L_data + lambda_anchor * ||theta_affine - theta_prior||²`

`lambda_anchor`는 `{0, .01, .1, 1, 10, 100}` 중 validation MSE로만 고른다.
prior가 맞으면 보존하고 어긋나면 계수를 수정한다. 별도 기성 예측기를 조합한
구조가 아니다.

## Strict tail / unseen-unit 결과

모든 수치는 pooled R², seeds 42–46 평균±표준편차다.

| 데이터 | 고정 PP | soft-anchor PP | 개선 | matched plain NN |
|---|---:|---:|---:|---:|
| HUST | 0.724±0.039 | **0.813±0.062** | +0.089 | 0.758±0.032 |
| Virkler | 0.857±0.019 | **0.869±0.030** | +0.012 | -0.604±0.472 |
| NASA health | 0.495±0.004 | **0.513±0.002** | +0.017 | 0.233±0.094 |

HUST에서 PP가 plain NN에 지던 문제가 뒤집혔다(+0.055). 세 데이터셋 모두
soft-anchor가 고정 PP를 평균 기준으로 이겼고 NASA는 5/5 seed에서 개선됐다.
이미 살펴본 데이터에 대한 retrospective 개발 실험이므로 독립 확증 결과는 아니다.

## PFN 비교 해석

현재 PP 프로토콜의 같은 test 행 비교에서는 기존 PP도 PFN보다 높았다(HUST
0.724 대 0.251, Virkler 0.857 대 0.612, NASA health 0.495 대 -0.678).
개선형은 기존 PP보다 다시 높다. PFN 수치는 단일 seed 또는 학습행 제한 결과라
최종 우위 주장에는 PFN 다중 seed 재실행이 필요하다.

PAE NASA corrected-hull의 `0.794 < 0.828`은 정보량이 달랐다. prior 경로는
capacity 한 열, PFN은 전압·온도를 포함한 9열을 사용했다. capacity 한 열로
맞춘 감사에서는 PFN `0.679±0.001`, PAE `0.794`로 prior 경로가 +0.115 높았다.

원시 결과는 `results/soft_anchor_ablation/*/results.json`에 있다.
