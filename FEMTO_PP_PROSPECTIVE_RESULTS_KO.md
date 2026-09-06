# FEMTO/PRONOSTIA PP prospective 결과

프로토콜은 test feature와 label을 파싱하기 전 공개 커밋 `a44b528`로 고정했다. PAE에서 이 데이터가 사용된 적은 있지만 PP/OFF-UQ 구조 선택에는 사용하지 않았으므로 PP model-level prospective 결과로 분류한다.

## 데이터

- Train: 완전 run-to-failure Learning bearing 5개, 1,182 rows
- Validation: Learning bearing 1개, 328 rows
- Test: 공식 truncated Test bearing 11개의 마지막 관측 endpoint
- 입력: condition, RMS, kurtosis, FFT band 요약
- 금지: total lifetime, future signal, official RUL answer, life fraction

## 결과

| 모델 | pooled R² | RMSE (s) | seed SD |
|---|---:|---:|---:|
| affine/Ridge head | -1.378 | 4057 | — |
| PP single | -1.378 | 4057 | 0.000 |
| plain NN single | -1.225 | seed별 변동 | 0.908 |
| plain NN ensemble | -0.902 | 3628 | — |

PP residual은 validation 선택에서 활성화되지 않아 affine head와 동일했다. Plain NN은 일부 seed가 PP보다 나았지만 모든 aggregate R²가 음수였고 seed 변동이 매우 컸다.

Test endpoint 11개 중 full causal-feature train convex hull 밖은 1개(`9.1%`)뿐이다. 따라서 이 공식 challenge endpoint 실험을 strict convex-hull extrapolation evidence로 세면 안 된다. 이는 unseen-bearing RUL transfer이며 PP의 현재 affine-tail 구조가 약한 vibration health signal에서 실패한 negative boundary다.

결과 확인 후 feature, split, clipping 또는 checkpoint를 변경하지 않았다. FEMTO는 논문 limitation/negative-control 표에는 넣을 수 있지만 PP의 주 extrapolation 성능표에서는 제외해야 한다.

