# 시간축 끝과 처음 보는 unit에 집중한 PP 벤치마크

이 벤치마크는 두 문제만 다룬다.

1. 한 시계열의 관측 범위보다 뒤쪽인 late tail 예측
2. 학습에 없던 엔진, 셀, 시편, 운전 프로토콜에 대한 일반화

모든 데이터에서 물리 unit을 먼저 train/validation/final test로 분리한다. 모델과
하이퍼파라미터는 train과 validation만 사용한다. 표의 R²는 각 데이터셋 final-test
행 전체를 합친 raw pooled R²다. 서로 단위와 행 수가 다른 데이터셋들을 다시 한
숫자로 합친 cross-dataset pooled R²는 보고하지 않는다.

| 데이터셋 | 외삽 정의 | final unit | affine/Ridge head | PP 5 seeds | PP ensemble |
|---|---|---:|---:|---:|---:|
| C-MAPSS FD002+FD004 | 처음 보는 엔진, train OP1–OP3 convex hull 밖 | 엔진 분리 | 0.710 | 0.749±0.015 | 별도 미산출 |
| HUST | 처음 보는 protocol의 train capacity hull 아래 late tail | 셀/프로토콜 분리 | 0.601 | 0.724±0.039 | **0.773** |
| Virkler | 처음 보는 시편의 33 mm 이후 균열 tail | 시편 분리 | -0.823 | 0.857±0.019 | **0.888** |
| NASA battery | 처음 보는 셀의 정규화 health 0.5 아래 tail | leave-one-cell-out | 0.424 | 0.495±0.004 | **0.495** |

C-MAPSS 수치는 기존의 고정된 strict OP-hull v5 실험에서 가져왔다. 이 행은 시간축
끝 자체보다는 처음 보는 엔진과 운전조건 지지영역 외삽을 검증한다. HUST, Virkler,
NASA는 unit 분리와 late-tail hull-out을 동시에 만족한다. HUST의 0.880 Ah 도달시간은
실제 관측 완료값이 아니라 검열된 시계열의 terminal-slope proxy이므로 논문의 핵심
확증 결과로 단독 사용하면 안 된다.

## 새 구조 ablation: support-gated PP

새 선택형 구조는 다음 출력 형태를 사용한다.

```text
prediction = affine(x) + exp(-beta * train-hull-distance) * correction_NN(x)
```

`beta=0`이면 기존 PP와 정확히 같다. train 지지영역에서 멀어지면 NN 보정만 줄고
affine tail이 남는다. beta는 unit이 분리된 hull-out validation에서만 선택했다.

| 데이터셋 | PP 5 seeds | support-gated PP | 변화 |
|---|---:|---:|---:|
| HUST | 0.724±0.039 | 0.736±0.040 | +0.013 |
| Virkler | 0.857±0.019 | 0.857±0.019 | +0.000 |
| NASA battery | 0.495±0.004 | 0.495±0.004 | -0.001 |

따라서 support gate를 PP의 기본값으로 바꾸지는 않는다. 이 실험은 멀리서 NN
보정을 줄이는 발상이 HUST에는 도움이 되지만 데이터셋 전체의 정확도 개선으로
일반화되지는 않았다는 ablation 근거다. 현재 재현 가능한 권장값은 기본 PP의
5-seed 평균 ensemble이며, support gate는 새 독립 cohort에서 잠근 상태로 다시
검증할 후보 구조다.
