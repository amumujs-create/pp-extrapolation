# PP-X Statistical Inference Protocol (small-n nonparametric tiers)

작성자: 박진서  
상태: frozen reporting rule for retrospective / prospective summaries

## 핵심 구분

| 질문 | 답 |
|---|---|
| 모수(표본)가 적을 때 **추론**은? | **비모수 / exact** (sign-flip, Wilcoxon, exact binomial, bootstrap CI) |
| 모수가 적다고 **예측기**를 비모수로 바꾸나? | **아니오.** Engression/TabPFN 등은 별도 모델 선택이며, 소표본 추론 규칙과 혼동하지 않는다. |

행(row)은 절대 독립 표본으로 쓰지 않는다. 추론 단위는 **물리 unit**, 그다음
**dataset/domain**이다.

## 표본 크기 티어

물리 unit 수 \(n_u\) 기준:

| 티어 | \(n_u\) | 허용 추론 | 금지 |
|---|---:|---|---|
| L0 descriptive | \(n_u \le 2\) | 점추정·방향만; CI는 폭 보고용 | p-value를 확증으로 해석 |
| L1 exact | \(3 \le n_u \le 20\) | exact paired sign-flip; Wilcoxon signed-rank(가능 시); unit bootstrap CI | 정규 t-test를 주 검정으로 사용 |
| L2 bootstrap | \(n_u \ge 21\) | L1 + 대용량 unit bootstrap; 필요 시 studentized bootstrap | row-level SE |

도메인 수 \(n_d\):

| 티어 | \(n_d\) | 허용 |
|---|---:|---|
| D0 | \(n_d \le 4\) | exact binomial / sign test만; “도메인 일반화 유의” 주장 금지 |
| D1 | \(n_d \ge 5\) | exact sign + hierarchical dataset–unit bootstrap CI |

## 주 검정 세트 (매칭된 PP vs matched direct)

1. **Unit paired effect:** \(\delta_u = \log\mathrm{RMSE}_{\mathrm{direct},u}-\log\mathrm{RMSE}_{\mathrm{PP},u}\)
   또는 relative RMSE gain. 양의 값이 PP 이득.
2. **Within-domain:** exact sign-flip p (주), Wilcoxon signed-rank p (보조, L1+).
3. **Across-domain multiplicity:** Benjamini–Hochberg on domain-level p.
4. **Equal-domain summary:** hierarchical bootstrap of equal-dataset mean effect.
5. **Domain direction:** exact binomial sign test on domain means.

## 정책 / prior-gate 감사

FA·FR 개수만 보고하지 않는다. dataset를 재표집하는 **policy bootstrap**으로
accuracy / FA / FR / deployed effect의 95% 구간을 함께 보고한다.
소표본(\(n_d=12\))이므로 구간이 넓을 수 있으며, 이를 숨기지 않는다.

## 이미 있는 것 / 이번 감사로 채우는 것

이미 있음:
- `experiments/journal_statistical_evidence.py` (sign-flip, BH, hierarchical bootstrap)
- frozen protocol의 primary evaluation 목록

부족했던 것 (이번 실험이 채움):
- Wilcoxon을 공식 보조 검정으로 고정
- \(n_u\) 티어 라벨·L0 가드레일
- 정책(operational unit-risk) 지표의 dataset bootstrap CI
- “소표본 → 비모수 예측기” 혼동 금지 문장

## 성공 기준 (이 프로토콜 자체)

- 모든 12-setting에 티어 라벨이 붙고, L0 도메인은 p를 확증으로 쓰지 않는다.
- Wilcoxon + sign-flip이 같은 방향으로 읽히는지 보고한다 (불일치 시 sign-flip 우선).
- 정책 bootstrap CI를 결과 문서에 남긴다.
