# CTBF Stanford·ISU 개발 실험 결과

작성자: 박진서  
상태: retrospective model-concept challenger; PP-X 승격 실패

## 결론

RUL을 직접 회귀하지 않고 양의 열화속도를 failure boundary까지 적분하는
CTBF를 구현했다. 구조 자체는 경계에서 RUL이 정확히 0이고 예측시간이
음수가 될 수 없도록 보장한다.

그러나 두 개발 cohort에서 효과가 일관되지 않았다.

- **Stanford:** weak-rate CTBF가 direct MLP와 기존 PP-X보다 좋았다.
- **ISU 250 mAh:** direct MLP와 기존 PP-X보다 크게 나빴다.

사전 동결한 승격조건인 “두 데이터셋 모두에서 direct MLP 개선”을 충족하지
못했으므로 현재 논문의 메인 PP-X를 CTBF로 교체하지 않는다.

## 동일 split test 결과

| Cohort | 모델 | pooled RMSE | pooled R² | unit-macro R² |
|---|---:|---:|---:|---:|
| Stanford | direct MLP | 90.407 | -0.144 | 0.078 |
| Stanford | direct-velocity CTBF | 74.436 | 0.224 | 0.396 |
| Stanford | weak-rate CTBF | **70.929** | **0.296** | **0.478** |
| Stanford | stored PP-X v1.1 | 83.055 | 0.034 | 0.209 |
| ISU 250 mAh | direct MLP | 1.726 | 0.445 | -2.086 |
| ISU 250 mAh | direct-velocity CTBF | 2.745 | -0.403 | -2.234 |
| ISU 250 mAh | weak-rate CTBF | 2.956 | -0.628 | -2.431 |
| ISU 250 mAh | stored PP-X v1.1 | **1.717** | **0.451** | **-1.683** |

Stored PP-X 수치는 같은 cohort/split의 기존 동결 실험 결과이며, 이번 CTBF
탐색예산과 완전히 같은 재학습 비교는 아니다.

## 구조 근거와 통계

### Stanford

- weak-rate CTBF vs direct MLP RMSE 개선: 19.26 cycles
- physical-unit bootstrap 95% CI: [11.31, 25.13]
- 개선확률: 0.99995
- unit wins: 7/9
- weak-rate CTBF vs direct-velocity CTBF CI: [2.60, 5.85]
- unit wins: 8/9

즉 Stanford에서는 time-to-boundary 적분 구조와 weak local-rate anchor가
모두 독립적으로 도움을 줬다.

### ISU 250 mAh

- weak-rate CTBF vs direct MLP RMSE 개선: -1.19 cycles
- physical-unit bootstrap 95% CI: [-1.62, -0.70]
- 개선확률: 0
- unit wins: 17/46
- weak-rate CTBF vs direct-velocity CTBF CI: [-0.28, -0.11]
- unit wins: 11/46

ISU에서는 적분형 구조와 weak-rate anchor 모두 손해였다. 짧은 test tail,
많은 1–3 row unit, noisy one-step rate 때문에 순간속도를 미래 경계도달시간의
충분한 상태로 보는 가정이 맞지 않은 것으로 해석한다.

## 확인된 불변식

- 학습된 velocity는 항상 양수다.
- normalized health 0.80 경계에서 최대 절대 RUL 예측은 두 cohort 모두 0이다.
- boundary에 가까워질수록 constant-velocity 적분시간이 감소한다.
- 구현 단위테스트 5개가 통과했다.

## 논문 판단

CTBF를 범용 PP-X 후계모델로 주장하면 안 된다. 현재 결과가 지지하는 범위는
다음뿐이다.

> 장기 열화 궤적이 양의 scalar velocity field와 고정 failure boundary로
> 설명되는 cohort에서는, time-to-boundary integral이 direct RUL regression의
> 외삽을 개선할 수 있다.

다음 연구에서 사용하려면 validation으로 flow-contract 적합성을 먼저
검증하고, 부적합한 cohort에서는 기존 PP-X/direct fallback으로 보내는
executor 후보로 다뤄야 한다. 현재 PP-X 논문의 메인 결과와 모델 정의는
변경하지 않는다.

## 재현

```bash
PYTHONPATH=src /opt/anaconda3/bin/python \
  experiments/ctbf_stanford_isu_development.py
```

원시 결과:

- `results/ctbf_stanford_isu_development/results.json`
- `results/ctbf_stanford_isu_development/predictions.npz`
