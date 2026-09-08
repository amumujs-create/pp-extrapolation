# 명칭 대응: SAAR ↔ 최종 PP

**채택 이름:** Support-Aware Affine–Residual Network (**SAAR**)

**동의어 (같은 최종 모형):**
- 최종 PP / adaptive dual-scale PP / dual-scale BQ-PP (통합 결과의 최종 행)
- 코드·결과 md에서 쓰던 `PP`, `final adaptive dual-scale PP`

**한 줄 정의:**  
관측 support를 고려해, 동결 affine 추세 위에 제한된 residual만 더하는 equation-free 외삽 네트워크.

**최종 식 (논문·발표 기준):**

\[
\hat y = m\,\operatorname{softplus}\big[\ell(z)+c_\theta(z)\big]
\]

\[
c_\theta(z)=w(z)\,B_L\tanh r_\theta(z)
+[1-w(z)]\,B_H\tanh\!\left(\frac{r_\theta(z)}{B_H}\right)
\]

**경계 모듈:** EOL health 경계가 관측될 때의 quotient 확장은 **BQ-SAAR** (구 BQ-PP)로 부를 수 있다.

**쓰지 말 것:**
- SAAR를 PAE와 혼동하지 말 것 (PAE = 후보식 컴파일 쪽)
- “prior-free”라고 쓰지 말 것 (equation-free는 OK)

**원문 수치 표기:**  
`UNIFIED_SUPPORT_GATED_PP_RESULTS_KO.md`의 **최종 adaptive dual-scale PP** 행 = SAAR 주 결과.

기록일: 2026-09-07
