# 역사적 명칭 대응: SAAR와 legacy PP

> **현재 paper-level 이름은 PP-X**다. PP-X는 contract-conditioned
> prior-residual framework와 validation-approved executor/fallback 정책을
> 포함한다. SAAR는 과거 support-aware affine-residual backbone에 붙인
> **historical alias**이며 PP-X 전체와 동의어가 아니다. 기존 파일명과 결과
> 라벨은 재현성을 위해 바꾸지 않는다.

**역사적으로 채택했던 이름:** Support-Aware Affine–Residual Network
(**SAAR**)

**당시 artifact 라벨 대응:**
- adaptive dual-scale PP / dual-scale BQ-PP
- 코드·결과 md의 `PP`, `final adaptive dual-scale PP`

이 대응은 당시 backbone/실험 행을 찾기 위한 것이며, 해당 라벨을 현재
paper-selected PP-X route로 소급 개명하지 않는다.

**한 줄 정의:**  
관측 support를 고려해, 동결 affine 추세 위에 제한된 residual만 더하는 equation-free 외삽 네트워크.

**역사적 SAAR backbone 식:**

\[
\hat y = m\,\operatorname{softplus}\big[\ell(z)+c_\theta(z)\big]
\]

\[
c_\theta(z)=w(z)\,B_L\tanh r_\theta(z)
+[1-w(z)]\,B_H\tanh\!\left(\frac{r_\theta(z)}{B_H}\right)
\]

**역사적 경계 모듈:** EOL health 경계가 관측될 때의 quotient 확장은 당시
**BQ-SAAR** 또는 BQ-PP로 기록됐다. 현재 논문에서는 PP-X의
boundary-quotient executor로 설명한다.

**쓰지 말 것:**
- SAAR를 paper-level 메인 이름으로 쓰지 말 것
- SAAR와 PP-X 전체를 동의어로 쓰지 말 것
- SAAR를 PAE와 혼동하지 말 것 (PAE = 후보식 컴파일 쪽)
- “prior-free”라고 쓰지 말 것 (equation-free는 OK)

**원문 수치 표기:**  
`UNIFIED_SUPPORT_GATED_PP_RESULTS_KO.md`의 **최종 adaptive dual-scale PP**
행은 SAAR alias가 사용된 역사적 mechanism evidence다. PP-X paper-main 수치는
`PPX_TOP_JOURNAL_VALIDATION_PACKAGE_KO.md`를 우선한다.

기록일: 2026-09-07
