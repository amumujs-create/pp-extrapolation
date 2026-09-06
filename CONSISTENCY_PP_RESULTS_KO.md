# Single-deployment consistency PP 실험

5개 PP teacher의 train-set 평균 예측을 하나의 PP에 증류하는 consistency loss를 구현했다. 최종 추론에는 PP 하나만 필요하다. Validation의 feature와 label은 distillation에서 제외하고 checkpoint 및 loss weight 선택에만 사용했다.

| 데이터 | plain NN single | PP single | consistency PP single | PP ensemble | consistency PP ensemble |
|---|---:|---:|---:|---:|---:|
| HUST | 0.759±0.032 | 0.724±0.039 | 0.720±0.046 | 0.773 | 0.778 |
| Virkler | -0.604±0.472 | 0.857±0.019 | **0.877±0.018** | 0.888 | **0.893** |
| RWTH | -0.113±1.663 | 0.506±0.020 | 0.500±0.014 | 0.507 | 0.501 |
| MICH | -1.116±1.053 | -1.522±0.000 | -1.522±0.000 | -1.522 | -1.522 |
| NASA | 0.233±0.094 | 0.495±0.004 | 0.496±0.004 | 0.495 | 0.496 |

Consistency distillation은 Virkler에서 명확히 개선되고 RWTH에서는 seed 분산을 줄였지만 평균 정확도를 약간 낮췄다. HUST도 single-seed 평균이 낮아졌다. 따라서 현재 형태를 논문의 기본 PP로 교체할 근거는 없다.

Validation feature를 teacher target 생성에 포함한 초기 진단에서는 HUST가 0.815까지 상승했지만, 이는 validation을 학습 입력으로 재사용한 transductive 결과이므로 폐기했다. 위 표는 train 좌표만 사용한 최종 결과다.

결론적으로 uncertainty residual gate는 HUST/NASA에서 작은 안전한 개선을 보였고, consistency loss는 도메인 의존적이다. 논문 주력은 기존 PP와 uncertainty-aware executor로 유지하고 consistency는 ablation으로 보고해야 한다.
