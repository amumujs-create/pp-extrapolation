# Luminosity temperature-shift 강건성 결과

## 판정

CCMR-L v1.6.1은 온도 condition shift에서 평균 성능은 개선했지만
unit-tail 안정성 기준을 크게 위반해 **강건성 확증 실패**다.

- condition split: train 105°C / validation 65°C / test 25°C
- physical units: 각 split 25개
- forecast origins: train 175 / validation 125 / test 175
- ordered-axis split: train 최대 30%, test 최소 75%
- test origin 100%가 train 진행도 밖
- Persistence: pooled R² 0.86158, RMSE 0.02471
- CCMR-L: pooled R² 0.86829, RMSE 0.02410
- pooled RMSE 개선 2.45%, unit-macro RMSE 개선 3.11%
- raw unit regret: mean -6.66%, CVaR20 +13.13%, maximum +29.34%

평균 성능만 보면 성공이지만 사전 기준은 CVaR20 2%, maximum 5%였다.
따라서 성공이나 safe fallback으로 판정할 수 없다.

## 원인

Validation에서는 raw mean/CVaR/max regret가
-15.92%/+0.64%/+1.89%였으므로 mass 0.165가 허용됐다. 그러나 test에서는
행의 97.71%에 correction이 활성화됐고 context support가 거부한 행은
2.29%뿐이었다. 즉 validation에서 안전했던 보정이 저온 test의 일부
unit에서 부호 또는 크기가 달라졌지만, cross-fit 합의와 정적 context
거리만으로는 이를 감지하지 못했다.

이는 NASA B0051/B0052 실패와 같은 범주의 **in-support concept shift**다.
단순 support 확대나 전역 mass 축소만으로는 평균 이득과 unit-tail 손상을
동시에 해결하기 어렵다.

## 다음 구조

다음 개발 버전은 현재 unit에서 이미 관측 완료된 동일-horizon shadow
forecast의 실제 이득을 causal하게 계산해야 한다. 최근 비중첩 forecast가
persistence보다 이긴 경우에만 correction을 활성화하고, 나머지는 exact
persistence로 복원하는 causal backtest gate가 필요하다.

이 luminosity test는 이후 개발자료로만 사용할 수 있으며 독립 확증으로
재사용하지 않는다. 원 패키지는 정확한 luminous device 및 실측/합성
여부를 명시하지 않아 provenance도 제한적이다.

## 보존 상태와 다음 고호트

기존 Alloy A 성공 고호트는 append-only sealed registry에 등록해 변경하지
않는다. Perovskite 2,245-cell 고호트는 별도 프로토콜을 outcome 개봉 전에
고정했지만, 현재 Zenodo 공식 다운로드 endpoint가 반복적으로 HTTP 504를
반환해 아직 미개봉 상태다. 서버 복구 후 그대로 실행할 수 있다.

관련 파일:

- `protocols/LUMINOSITY_TEMPERATURE_SHIFT_CCMR_V161_PROTOCOL.md`
- `results/luminosity_temperature_shift_ccmr_v161/results.json`
- `results/luminosity_temperature_shift_ccmr_v161/sealed_predictions.npz`
- `protocols/PEROVSKITE_PHENOTYPE_SHIFT_CCMR_V161_PROTOCOL.md`
