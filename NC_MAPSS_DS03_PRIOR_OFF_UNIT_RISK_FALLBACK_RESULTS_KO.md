# DS03 prior-off unit-risk fallback 재설계 결과

작성자: 박진서  
프로토콜: `protocols/DS03_PRIOR_OFF_UNIT_RISK_FALLBACK_PROTOCOL.md`  
실험: `experiments/ds03_prior_off_unit_risk_fallback.py`  
산출물: `results/ds03_prior_off_unit_risk_fallback_v1/`

## 한 줄 결론

**unit-risk validation 규칙으로도 DS03 약점은 줄지 않는다.**  
6개 규칙이 모두 `plain_mlp`를 고르고, test R²는 0.882 → **0.866으로 악화**한다.  
Engression/FT는 validation unit-risk에서 탈락한다.

## 설정

- prior-on core 미변경 (frozen PP-X는 계속 `direct_fallback`)
- 각 equal-budget selected config를 seeds 42–46으로 재적합 → **validation ensemble**으로 규칙 적용
- test R²는 선택 후 공개

## Validation unit-risk (요지)

| 모델 | val MSE | unit-macro RMSE | worst-unit RMSE | vs direct win | worst ratio | unit-gain CI_low | test R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| PP-X direct | 37.639 | 5.789 | 8.558 | — | 1.000 | — | **0.882** |
| **plain_mlp** | **22.722** | **4.551** | **6.408** | **1.00** | **0.948** | **+0.208** | 0.866 |
| vrex / monotone | ≈22.7–23.1 | ≈4.55 | ≈6.41 | 1.00 | ≈0.95 | +0.15~0.21 | ≈0.866 |
| Engression | 42.912 | 5.852 | 9.877 | 0.67 | **1.154** | −1.319 | **0.901** |
| FT-Transformer | 47.105 | 6.286 | 10.048 | 0.33 | 1.174 | −1.490 | 0.899 |
| GroupDRO | 40.175 | 5.937 | 8.944 | 0.33 | 1.098 | −0.393 | 0.881 |

## 규칙별 선택

| 규칙 | 선택 | test R² | vs PP-X |
|---|---|---:|---:|
| pooled MSE | plain_mlp | 0.866 | −0.016 |
| unit-macro RMSE | plain_mlp | 0.866 | −0.016 |
| worst-unit RMSE | plain_mlp | 0.866 | −0.016 |
| paper unit-gate (≥2% MSE, win≥0.6, ratio≤1.1) | plain_mlp (admitted: direct, mlp, vrex, monotone) | 0.866 | −0.016 |
| lexicographic unit-risk | plain_mlp | 0.866 | −0.016 |
| unit-gain CI ≥0 → MSE | plain_mlp | 0.866 | −0.016 |

- **Engression을 고른 규칙: 0개**
- paper gate에서 Engression 탈락 이유: val MSE가 direct보다 나쁘고, worst-unit ratio 1.154 > 1.10, unit-gain CI 음수
- FT는 win fraction·worst ratio·MSE 모두 더 불리

## 해석

1. validation unit-risk는 **validation unit에 잘 맞는 MLP**를 강하게 선호한다.
2. 그 선택은 test에서 Engression보다도, frozen PP-X direct보다도 못하다.
3. 따라서 “unit-risk로 fallback 포트폴리오를 고르면 DS03가 고쳐진다”는 가설은 **기각**이다.
4. Engression test 우세(0.901)는 여전히 **test-oracle 정보**에 해당한다.

## 논문 포지션에 대한 함의

- PP-X 노벨티(prior-on + validation gate)는 유지해도 된다.
- DS03 prior-off 손실은 **limitation**으로 두는 편이 정직하다.
- fallback을 Engression으로 고정하는 것은 validation 계약과 충돌한다.
- 추가 실험이 필요하면 NASA 동률권 확인 정도만 선택적으로,  
  “unit-risk로 Engression 승인” 재시도는 ROI가 낮다.
