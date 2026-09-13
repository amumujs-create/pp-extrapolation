# DS03 prior-off fallback portfolio 결과

작성자: 박진서  
프로토콜: `protocols/DS03_PRIOR_OFF_FALLBACK_PORTFOLIO_PROTOCOL.md`  
산출물: `results/ds03_prior_off_fallback_portfolio_v1/results.json`  
실험: `experiments/ds03_prior_off_fallback_portfolio.py`

## 한 줄 결론

**validation만으로 `{direct, Engression, …}`를 고르면 DS03 약점은 줄지 않는다.**  
Engression은 test에서 이기지만 validation에서는 frozen PP-X direct보다 나쁘다.

## 설정

- prior-on core는 그대로 두고, **prior가 이미 거절된 DS03 fallback만** 본다.
- frozen PP-X: `direct_fallback` (val MSE 37.639 → test R² **0.882**)
- equal-budget 모델들의 selected-config validation MSE로 fallback을 고른 뒤, test R²만 공개

## 숫자

| 후보 | val MSE ↓ | test R² |
|---|---:|---:|
| GroupDRO | **29.927** | 0.881 |
| plain_mlp | 30.346 | 0.866 |
| PP-X direct (frozen) | 37.639 | **0.882** |
| FT-Transformer | 42.484 | 0.899 |
| **Engression** | 45.967 | **0.901** (test oracle) |

### validation 선택 결과

1. **전체 portfolio** → GroupDRO  
   - test R² 0.881 ≈ PP-X 0.882 (**개선 없음**, Δ ≈ −0.0008)
2. **제한 set** `{PP-X direct, plain_mlp, Engression, FT}` → plain_mlp  
   - test R² 0.866 (**오히려 악화**, Δ ≈ −0.016)
3. **Engression은 validation에서 탈락**  
   - val MSE 45.97 > PP-X direct 37.64  
   - test에서만 1위 → validation 규칙으로 고를 수 없음

## 해석

- “fallback을 Engression으로 바꾸면 DS03가 고쳐진다”는 **test를 보고 고르는 선택**이다.
- 논문 claim을 지키는 validation-only 규칙에서는 Engression이 선택되지 않는다.
- 따라서 **PP-X 노벨티(prior-on / validation gate)는 유지**하되,  
  DS03 prior-off 약점은 “포트폴리오만 넓히면 자동으로 해결”되지 않는다.
- 다음 옵션(별도 설계 필요):
  1. prior-off를 **limitation**으로 두고 논문은 prior-on/stability에 집중
  2. fallback 선택 규칙을 val MSE 단독이 아닌 **unit-risk / coverage-aware**로 재설계 후 재실험
  3. Engression을 fallback 후보에 넣되, **언제 승인할지**를 사전 계약으로 고정

## 범위에 대한 답

진 구간(주로 DS03)만 먼저 추가 테스트하면 된다. 이번 1차가 그것이다.  
prior-on 9개 메인 설정을 다시 돌릴 필요는 없다.
