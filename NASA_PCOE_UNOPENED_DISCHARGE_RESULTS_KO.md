# NASA PCoE 미개봉 고호트 강한 외삽 결과

## 결론

OAIR v1.5는 새 미개봉 NASA 배터리 고호트에서 독립 확증에
**실패했다**. Persistence는 강한 외삽에서도 높은 pooled R²를
유지했지만, OAIR의 약한 residual 보정은 pooled RMSE를 개선하지 못했다.
이 결과를 보고 모형이나 기준을 변경하지 않았다.

## 데이터와 계약

- 공식 NASA PCoE 18개 미사용 cell
- 소형 추출 자료: 9.19 MB
- train 10 cells / validation 4 cells / test 4 cells
- train 844 / validation 223 / test 96 discharge trajectories
- 72,759 / 15,627 / 2,376 forecast origins
- train은 방전 진행도 0--30%, test는 75--85%
- test origin 100%가 train 진행도 밖
- test physical cells: B0049--B0052

Lifecycle-capacity 계약은 B0049--B0052의 discharge cycle 수가
24/20/24/4에 불과해 사전 최소 deep-tail 행 수를 충족하지 못했다.
따라서 모델을 학습하지 않고 inconclusive로 남겼다.

## One-shot test

- Persistence: pooled R² 0.97109, RMSE 0.10846, MAE 0.06192
- OAIR v1.5: pooled R² 0.97066, RMSE 0.10927, MAE 0.06349

OAIR의 pooled RMSE는 persistence보다 0.75% 높았다. Cell-level raw
regret은 mean 0.38%, CVaR20 2.83%, maximum 2.83%로 손실 상한은
지켰지만, 사전 성공 조건인 RMSE 개선을 만족하지 못했다.

Cell별로는 B0050에서 개선됐지만 B0051과 B0052에서 각각 약 2.26%,
2.91% 악화됐다. B0051은 OAIR와 persistence 모두 cell별 R²가
음수였고 OAIR가 더 나빴다.

## 해석

실패 원인은 persistence anchor 자체가 아니라 **보정 허용 판단**이다.
현 slope-space OOD guard는 test 행을 하나도 거부하지 않았다. 그러나
검증에서 안전했던 slope 패턴만으로는 서로 다른 cutoff 및 후기 방전
형태를 구분하지 못했다. 따라서 현재 구조는 보정을 약하게 만들었지만,
보정을 적용해야 할 레짐과 exact persistence로 돌아가야 할 레짐을
충분히 판별하지 못한다.

다음 버전은 이 test를 다시 튜닝용으로 쓰면 안 된다. 별도 development
자료에서 다음을 설계한 뒤 또 다른 미개봉 고호트로 확인해야 한다.

1. slope뿐 아니라 anchor forecast error의 식별 가능한 proxy를 쓰는
   risk gate;
2. cell/condition 불변 표현에서 validation-supported improvement가
   없는 경우 mass를 정확히 0으로 만드는 selective correction;
3. pooled 개선뿐 아니라 모든 validation cell 비악화를 요구하는
   minimax mass selection.

원자료, 예측, 점수:

- `protocols/NASA_PCOE_UNOPENED_DISCHARGE_OAIR_V15_PROTOCOL.md`
- `results/nasa_pcoe_unopened_discharge_oair_v15/sealed_predictions.npz`
- `results/nasa_pcoe_unopened_discharge_oair_v15/results.json`
