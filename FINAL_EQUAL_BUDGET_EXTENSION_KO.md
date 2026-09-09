# MICH·NASA Milling 공통 예산 비교군 확장

이 실험은 이전의 짧은 비교군 grid를 덮어쓰지 않는다. V-REx, GroupDRO, monotone
NN 각각에 architecture/learning-rate/weight-decay 24개와 penalty 5개, 총 **29개
validation 후보**를 주고 선택된 설정을 seed 42–46으로 재학습했다.

| 데이터셋 | 비교군 | 후보 수 | pooled ensemble R² | RMSE |
|---|---|---:|---:|---:|
| MICH raw-cycle RUL | V-REx | 29 | -0.697 | 20.875 |
| MICH raw-cycle RUL | GroupDRO | 29 | -0.696 | 20.875 |
| MICH raw-cycle RUL | monotone NN | 29 | -0.690 | 20.832 |
| NASA Milling material shift | V-REx | 29 | -1.086 | 2.874 |
| NASA Milling material shift | GroupDRO | 29 | -1.840 | 3.353 |
| NASA Milling material shift | monotone NN | 29 | -1.041 | 2.843 |

MICH에서 final PP-X의 동일 raw-cycle 기준 pooled R²는 0.826이다. 따라서 이번
세 robust baseline은 final PP-X를 넘지 못했다. 이 결과는 historical short-grid
결과와 별도로 보관한다.

Milling validation은 group 하나뿐이다. 따라서 위 수치는 test를 보지 않고 선택한
민감도 결과이지만, stable model-ranking 또는 통계적 우월성 근거로 쓰지 않는다.

Engression과 full-data SVGP는 이 로컬 환경에 각각 `engression`, `gpytorch`가
없어 실행하지 못했다. 750행 exact GP를 대체 수치로 추가하지 않았으며, 최종 표에는
`not run: dependency unavailable`로 남긴다.

재현 명령:

```bash
python experiments/equal_budget_mich_milling.py
```
