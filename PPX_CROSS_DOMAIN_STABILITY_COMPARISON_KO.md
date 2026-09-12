# PP-X Cross-Domain Stability Comparison

작성자: 박진서  
상태: canonical alias / entry point

이 파일은 cross-domain stability 증거의 **찾기 쉬운 고정 경로**다.
상세 수치·비모수 검정·재현 명령은 아래 canonical 문서를 본다.

## Canonical documents

- 상세 결과: [`PPX_DOMAIN_STABILITY_RESULTS_KO.md`](PPX_DOMAIN_STABILITY_RESULTS_KO.md)
- 원시 JSON: `results/ppx_domain_stability_v1/results.json`
- 재현: `python experiments/domain_stability_analysis.py`
- 논문 서사·전체 evidence registry:
  [`PPX_PAPER_NARRATIVE_AND_EVIDENCE_REGISTRY_KO.md`](PPX_PAPER_NARRATIVE_AND_EVIDENCE_REGISTRY_KO.md)

## Quick comparison (9 main settings, equal 30-candidate budget)

| Model | macro \(R^2\) | domain SD | domain MAD | \(R^2>0\) | paired exact MAD \(p\) vs PP-X |
|---|---:|---:|---:|---:|---:|
| **PP-X** | **0.807** | **0.174** | **0.061** | **9/9** | — |
| Engression | 0.257 | 0.949 | 0.280 | 7/9 | **0.0391** |
| GroupDRO | 0.011 | 0.998 | 0.619 | 5/9 | **0.0156** |
| FT-Transformer | 0.115 | 1.005 | 0.366 | 7/9 | 0.1250 |
| plain MLP | 0.093 | 1.007 | 0.554 | 6/9 | **0.0469** |
| V-REx | 0.071 | 1.008 | 0.535 | 6/9 | **0.0469** |
| monotone NN | 0.054 | 1.057 | 0.541 | 6/9 | 0.0625 |
| linear-tail RBF | −0.085 | 1.473 | 0.116 | 7/9 | 0.5625 |
| SVGP | −1.639 | 4.257 | 0.509 | 5/9 | 0.0703 |

## Claim boundary

- 가능: 현재 9개 retrospective setting에서 PP-X가 더 작은 도메인 간 편차와
  더 높은 성공률을 보였다.
- 개별 비교: Engression / GroupDRO / MLP / V-REx에서 exact MAD \(p<0.05\).
- 불가: 8개 모델 동시 Holm 보정 후 보편적 분산 우월이 확증됐다.
- 불가: 평균 성능 SOTA 또는 미래 cohort 보장.

안정성은 seed 분산만이 아니라 **도메인별 성공·실패와 \(R^2\) 편차**로 정의한다.
