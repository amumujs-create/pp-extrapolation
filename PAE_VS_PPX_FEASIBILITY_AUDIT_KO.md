# PAE 수식 prior 대 최종 PP-X feasibility 감사

## 답

**식 prior가 있는 두 setting에서 PAE의 직접 우세 증거가 생겼다.** Virkler에서는 Paris PAE가 동일 20개 test 행에서 최종 PP-X보다 R² `+0.081` 높고, MICH raw-cycle RUL에서는 centered boundary-quotient PAE가 formula-free generic PP보다 R² `+2.158` 높다. 두 결과 모두 retrospective development comparison이며, MICH의 generic PP와 later normalized-target PP-X executor는 target contract가 달라 별도 표로 유지한다.

## 비교 원칙

다음 네 가지가 모두 같을 때만 PP-X 대 PAE의 성능 우열을 주장한다.

1. train/validation/test unit과 test row
2. RUL target 및 censoring/endpoint 정의
3. prediction-time 입력 정보
4. clipping, isotonic projection 등 postprocessing

한 가지라도 다르면 같은 데이터셋 이름이라도 feasibility signal로만 분류한다.

## 현재 증거

| 도메인/설정 | PAE 식 또는 compiled prior | PAE R² | 최종 PP-X R² | 단순 차이 | 비교 상태 | 해석 |
|---|---|---:|---:|---:|---|---|
| Virkler crack tail | Paris equation-only | **.969** | .888 | **+.081 PAE** | exact matched 20 test rows, 10 specimens | unit RMSE p=.0215; 명시적 Paris 메커니즘 |
| MICH raw-cycle RUL | centered boundary quotient, 80% EOL | **.635** | −1.522 generic PP | **+2.158 PAE** | exact matched 202 rows, 8 cells | unit RMSE p=.0078; formula-free PP/MLP/DRO 모두 음수 |
| HUST battery proxy tail | rate-to-boundary equation | .834 | .958 | −.124 PAE | target proxy와 cohort가 유사하나 exact replay 없음 | 단순 현재 slope는 protocol/가속도 차이를 설명하지 못함 |
| N-CMAPSS hard | load/TRA counterfactual compiled pack | .949 | .937 | +.012 PAE | 같은 hard setting n=159이나 PAE CF·isotonic 경로가 달라 near-matched | 작은 이득. 수식 prior의 큰 승리 근거는 아님 |
| NASA battery | deterministic equation-only | .794 | .584 | +.210 표면상 PAE | split/health normalization/target이 달라 직접 비교 불가 | 수치만으로 PAE 승리로 쓰면 안 됨 |
| MATR battery | state-boundary sequence | .801 | .466 | +.335 표면상 PAE | strict-tail protocol과 동일하지 않음 | split을 맞춘 재실행 전에는 feasibility 후보일 뿐 |
| C-MAPSS FD001–004/PHM08 | condition-aware sequence pack | .784–.943 | 해당 exact PP-X 결과 없음 | — | PP-X와 다른 benchmark | PAE 단독 논문의 후보 evidence |

PAE 원자료: `../ca-css-ncmapss/results/pae_virkler_matched_ppx_v1/results.json`, `../ca-css-ncmapss/results/pae_mich_matched_pp_v1/results.json`.

## feasibility 결론

PAE의 강점은 “어떤 도메인에서나 식을 넣으면 PP보다 높다”가 아니다. **관측 가능한 상태변수와 미래 failure boundary를 잇는 식이 실제로 식별되는 도메인에서, 그 식을 executable prior로 쓰는 것**이다.

- **가장 적합:** Virkler와 MICH raw-cycle boundary setting. 각각 Paris law, boundary quotient가 입력·상태·경계·RUL 관계를 직접 제한한다.
- **조건부 적합:** N-CMAPSS. 운전조건/TRA의 effect는 있으나 현재 효과는 `+0.012`로 작다.
- **부적합 또는 약함:** HUST의 단순 rate-to-boundary. 현재 slope만으로 미래 가속/프로토콜 효과를 식별하지 못한다.
- **재검증 필요:** NASA·MATR. PAE 수치가 높아 보이지만 PP-X와 protocol이 달라 현재 숫자를 비교하면 안 된다.

## PAE 논문용 다음 한 실험

가장 깨끗한 feasibility test는 **Virkler matched equation experiment**다.

1. PP-X와 PAE에 동일한 train/validation/test crack specimen 및 same row manifest를 사용한다.
2. 입력은 crack length, causal growth-rate, load context만 허용한다.
3. PAE arms는 `Paris equation only`, `Paris + bounded residual`, `prior-off direct NN`으로 둔다.
4. 모든 hyperparameter와 Paris-residual strength는 validation specimen만으로 선택한다.
5. test에서는 unit-level RMSE bootstrap과 pooled R²를 함께 보고한다.

이 실험에서 PAE의 equation-only 또는 Paris+residual이 PP-X보다 유의하게 낮은 unit RMSE를 보이면, PAE 논문은 “정확한 기작식이 관측 가능한 경우 PP보다 더 강한 extrapolation bias를 제공한다”는 명확한 독립 기여를 얻는다.

반대로 Virkler에서도 이기지 못하면 PAE를 PP보다 성능이 높은 일반 후속 모델로 밀기보다, `typed prior admissibility and abstention` 연구로 좁히는 편이 정직하다.
