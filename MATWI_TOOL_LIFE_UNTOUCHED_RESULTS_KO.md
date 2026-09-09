# MATWI tool-life untouched PP 결과

## 판정

**확증 실패.** 프로토콜 커밋 `f5fc5e4` 뒤 KU Leuven RDR의 534.8-KB
`labels.csv`만 다운로드했다. 사전 분할의 test 공구 5개가 모두 적격했고,
168개 late-tail 행의 예측을 저장한 다음 처음으로 성능을 계산했다.

| 모델 | pooled R² | RMSE | MAE |
|---|---:|---:|---:|
| tuned PP | **-0.137** | **12.672** | **10.336** |
| matched tuned MLP | -0.151 | 12.753 | 10.474 |

PP가 점 추정치로는 MLP보다 조금 나았지만 절대 R²가 음수다. paired-tool
bootstrap의 `MLP RMSE - PP RMSE` 평균은 0.043이고 95% CI는
**[-0.604, 0.427]**이므로 우위도 확정되지 않았다.

## test를 보지 않고 선택된 설정

- PP: final 30% endpoint training, width 16, learning rate 1e-3,
  weight decay 5.0, frozen affine + bounded residual
- MLP: final 30%, width 64, learning rate 1e-3, weight decay 5.0
- 선택은 12개 development 공구의 4-fold grouped CV로만 수행했다.
- test 공구: 13, 7, 10, 6, 17

## 실패 해석

`labels.csv`만 쓰면 현재 wear와 wear type은 보이지만 공구마다 크게 다른 총수명
척도를 충분히 식별하지 못한다. test 공구의 관측 길이는 약 53--147회로 넓고,
동일 wear 값도 절삭속도·이송·재료 조건에 따라 서로 다른 생애 진행률을 뜻한다.
따라서 이 실패는 단순 epoch/width 부족보다 **미관측 절삭 context와 latent
lifetime-scale shift**에 가깝다.

이 test는 이미 관측됐으므로 이후 `sets.csv` 절삭조건이나 원시 센서를 추가해
개선하더라도 development evidence로만 취급한다.

재현 코드: `experiments/matwi_tool_life_untouched.py`

결과: `results/matwi_tool_life_untouched_v1/results.json`
