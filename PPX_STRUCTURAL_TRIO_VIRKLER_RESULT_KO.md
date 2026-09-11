# PP-X 구조 후보 3종: Virkler 결과

실행일: 2026-09-12  
연구자: 박진서

## 결론

세 구조 모두 PP-X 승격에 실패했다.

- Projected Residual-State: validation이 exact PP-X를 선택
- Temporal Self-Consistency: validation이 exact PP-X를 선택
- Prior-Geometry Residual: validation에서는 승인됐지만 sealed test에서 크게 악화

따라서 어느 구조도 현재 PP-X에 포함하지 않는다.

## 공통 기준

- 동일 Virkler unit-disjoint train/validation/test split
- 최종 support-gated PP-X, seed 42--46
- test PP-X: RMSE 2.5769, R² 0.8880, MAE 1.6004
- test: 10 units, 20 rows
- 모든 후보는 off 상태에서 PP-X를 정확히 재현
- 모든 실행 결과의 finite prediction 및 full-row coverage 유지

## 1. Projected Residual-State

validation 최적점은 persistence 1, 무한 tube였다. 이는 recurrence가 PP-X
residual을 그대로 복원하는 exact-off 상태다. 비활성 후보보다 좋은
residual-state 구성이 없었으므로 구조는 승인되지 않았다.

- test RMSE: 2.5769
- test R²: 0.8880
- PP-X 대비 RMSE 변화: 0%
- exact replay 오차: 부동소수점 허용범위 이내
- 판정: 안전 fallback, 모델 개선 근거 없음

## 2. Causal Temporal Self-Consistency

validation 최적점은 projection strength 0이었다. Virkler late-tail split은
각 test specimen에 두 관측점만 있어 causal lifetime smoothing이 활용할
충분한 trajectory history가 없었다.

- test RMSE: 2.5769
- test R²: 0.8880
- PP-X 대비 RMSE 변화: 0%
- exact replay 오차: 0
- 판정: 안전 fallback, 현 contract에서 부적합

## 3. Prior-Geometry-Conditioned Residual

validation-unit LOO에서는 alpha 100이 선택됐고 0.5% 이상 개선 조건을
통과했다. 그러나 test에서는 일반화되지 않았다.

- validation RMSE: 3.9100 → 3.6793
- test RMSE: 2.5769 → 3.3830
- test R²: 0.8880 → 0.8069
- test RMSE 악화: 31.3%
- units won: 2/10
- worst-unit RMSE ratio: 5.820
- unit log-RMSE 개선 bootstrap 95% CI: [-0.765, 0.224]
- 판정: 5% harm 기준 초과로 즉시 기각

validation에서 얻은 geometry-residual 관계가 10개 validation specimen에서
10개 test specimen으로 이동하지 않았다. 이는 작은 validation unit 수에서
calibration head가 route-selection false accept를 일으킬 수 있음을 다시
보여준다.

## 해석과 다음 결정

1. PP-X를 exact nested off-state로 둔 설계는 성능과 coverage를 안전하게
   보존했다.
2. 그러나 exact nesting 자체는 새 구조의 효과 근거가 아니다.
3. Virkler late-tail에는 specimen당 두 점만 있어 trajectory-state나 causal
   projection을 검증하기에 정보가 부족하다.
4. prior-geometry calibration은 구조적으로 흥미롭지만 현재 표본에서는
   불안정하며 PP-X에 추가하면 안 된다.
5. 다음 trajectory 구조 실험은 더 긴 validation/test history가 있는
   데이터에서 해야 한다. Virkler에서는 이미 성능 신호가 확인된
   event-coordinate flow가 더 유력한 challenger다.

## 재현 자료

- protocol: `protocols/PPX_STRUCTURAL_TRIO_VIRKLER_PROTOCOL.md`
- experiment: `experiments/ppx_structural_trio_virkler.py`
- result: `results/ppx_structural_trio_virkler/results.json`
- predictions: `results/ppx_structural_trio_virkler/predictions.npz`
