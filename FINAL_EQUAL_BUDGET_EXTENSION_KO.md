# MICH·NASA Milling 공통 예산 비교군 확장

이 실험은 이전의 짧은 비교군 grid를 덮어쓰지 않는다. V-REx, GroupDRO, monotone
NN 각각에 architecture/learning-rate/weight-decay 24개와 penalty 5개, 총 **29개
validation 후보**를 주고 선택된 설정을 seed 42–46으로 재학습했다.

| 데이터셋 | 비교군 | 후보 수 | pooled ensemble R² | RMSE |
|---|---|---:|---:|---:|
| MICH raw-cycle RUL | V-REx | 29 | -0.697 | 20.875 |
| MICH raw-cycle RUL | GroupDRO | 29 | -0.696 | 20.875 |
| MICH raw-cycle RUL | monotone NN | 29 | -0.690 | 20.832 |
| MICH raw-cycle RUL | Engression | 32 | -1.580 | — |
| MICH raw-cycle RUL | full-train SVGP | 30 | -2.247 | — |
| NASA Milling material shift | V-REx | 29 | -1.086 | 2.874 |
| NASA Milling material shift | GroupDRO | 29 | -1.840 | 3.353 |
| NASA Milling material shift | monotone NN | 29 | -1.041 | 2.843 |

MICH에서 final PP-X의 동일 raw-cycle 기준 pooled R²는 0.826이다. 따라서 이번
세 robust baseline은 final PP-X를 넘지 못했다. 이 결과는 historical short-grid
결과와 별도로 보관한다.

Milling validation은 group 하나뿐이다. 따라서 위 수치는 test를 보지 않고 선택한
민감도 결과이지만, stable model-ranking 또는 통계적 우월성 근거로 쓰지 않는다.

Engression 0.1.15와 `gpytorch` 1.15.2를 설치한 뒤 MICH에서 실행했다.
Engression은 공식 package의 32 validation 후보와 5 seed refit을 사용했다. SVGP는
750행 표본 제한 없이 전체 train 행을 사용하고, 30 validation 후보와 최대 128개
inducing point를 사용했다. 두 방법 모두 PP-X를 넘지 못했다. 750행 exact GP는
최종 표에 쓰지 않는다.

재현 명령:

```bash
python experiments/equal_budget_mich_milling.py
```
