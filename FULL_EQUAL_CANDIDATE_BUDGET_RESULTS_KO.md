# PP-X 전체 동일 후보예산 비교 결과

작성자: 박진서  
실행일: 2026-09-11  
프로토콜: 모델별 validation 후보 30개, search seed 42, 선택 설정 refit
seeds 42–46

## 실험 범위

9개 main extrapolation setting에서 아래 8개 비교군을 모두 같은
train/validation/test adapter와 같은 후보 수로 다시 학습했다.

- plain MLP
- FT-Transformer
- V-REx
- GroupDRO
- monotone NN
- Engression
- linear-tail RBF
- full-train SVGP

NASA battery는 사전 정의된 4개 leave-one-cell-out fold에서 각각 선택·재학습한
뒤 row-aligned prediction을 합쳤다. 총 학습 job은 3,360개였고, 기록된 모델
실행시간 합계는 약 3.52시간이다.

PP-X는 데이터셋별 typed contract가 허용한 executor 후보 자체가 다르므로,
8개 범용 비교군과 같은 하나의 30개 hyperparameter grid로 재정의하지 않았다.
논문 방법 동결 전에 만들어진 최종 PP-X prediction을 그대로 사용했다. 따라서
아래 결과가 입증하는 것은 **모든 비교군 사이의 후보 수 공정성**이며,
“PP-X의 과거 구조개발 비용까지 정확히 30회였다”는 뜻은 아니다.

## 결과

| 데이터셋 | PP-X R² | 30-candidate 최강 비교군 | 비교군 R² | 차이 |
|---|---:|---|---:|---:|
| HUST | **0.958** | GroupDRO | 0.955 | +0.003 |
| Virkler | 0.888 | **FT-Transformer** | **0.890** | −0.002 |
| NASA battery | **0.584** | Engression | 0.583 | +0.000 |
| Sunwoda | **0.939** | linear-tail RBF | 0.838 | +0.102 |
| RWTH | **0.878** | linear-tail RBF | 0.732 | +0.146 |
| MICH | **0.751** | monotone NN | −0.686 | +1.437 |
| MATR2019 | **0.466** | FT-Transformer | 0.342 | +0.123 |
| MATR-b2 | **0.862** | plain MLP | 0.813 | +0.049 |
| N-CMAPSS | **0.937** | Engression | 0.932 | +0.005 |

PP-X는 9개 중 8개에서 최강 동일예산 비교군보다 pooled R²가 높았다.
Virkler에서는 FT-Transformer가 0.002 높았다. 8/9 dataset exact sign test는
양측 `p=0.0391`이다.

## 물리 unit 결과

각 데이터셋에서 test pooled R²가 가장 높은 동일예산 비교군을 선택한 뒤,
그 row-aligned prediction과 PP-X의 unit log-RMSE ratio를 계산했다.

| 데이터셋 | PP-X 승리 unit | mean log-RMSE ratio | unit bootstrap 95% CI | BH q |
|---|---:|---:|---:|---:|
| HUST | 8/16 | 0.089 | [−0.213, 0.438] | 0.951 |
| Virkler | 5/10 | −0.061 | [−0.687, 0.495] | 0.984 |
| NASA | 2/4 | −0.008 | [−0.228, 0.167] | 0.984 |
| Sunwoda | 9/9 | **0.498** | **[0.368, 0.657]** | **0.023** |
| RWTH | 8/8 | **0.406** | **[0.279, 0.529]** | **0.023** |
| MICH | 8/8 | **1.054** | **[0.902, 1.283]** | **0.023** |
| MATR2019 | 5/10 | 0.075 | [−0.118, 0.271] | 0.907 |
| MATR-b2 | 8/9 | **1.023** | **[0.558, 1.435]** | **0.026** |
| N-CMAPSS | 2/3 | −0.429 | [−1.368, 0.043] | 1.000 |

Sunwoda, RWTH, MICH, MATR-b2가 BH 보정 후 유의하다. N-CMAPSS는 pooled R²는
PP-X가 0.005 높지만 세 unit 중 한 unit의 RMSE 손실이 커서 unit 평균 효과는
음수다. 이 결과를 감추지 않는다.

최강 비교군은 test pooled R²로 고른 기술적 최강값이므로, 그 모델과의 paired
검정은 사전 지정된 단일 비교 가설이 아니다. PP-X에 불리한 보수적 secondary
audit로 보고하고, 주 통계는 전체 모델 표와 dataset sign test로 분리한다.

## 결론

비교모델의 후보 수가 작아서 PP-X가 유리했다는 공격은 해소됐다. 후보를 모두
30개로 맞춰도 PP-X는 8/9 setting에서 최강 비교군보다 높은 pooled R²를 보였다.
다만 Virkler의 미세 열세, NASA의 사실상 동률, N-CMAPSS의 unit 이질성을 고려하면
“모든 데이터셋에서 보편적으로 우월하다”고 쓰지 않는다.

재현:

```bash
python experiments/full_equal_candidate_budget.py
python experiments/summarize_full_equal_candidate_budget.py
```

원시 결과:

- `results/full_equal_candidate_budget_v1/`
- `results/full_equal_candidate_budget_summary_v1/results.json`
