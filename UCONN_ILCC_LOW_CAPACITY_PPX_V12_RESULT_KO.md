# 저용량 외부 고호트 PP-X v1.2 평가 결과

**결과:** 적격성 불충분으로 `inconclusive`. 모델 학습·검증·test 채점은 0회다.

## 무엇을 고정했나

- 고호트: UConn–ISU–ILCC LFP/Gr, 공칭 1.2 Ah, 공개 메타데이터 64셀
- 데이터: 저자 저장소의 processed RPT 자료 15.0 MB
- 프로토콜: `protocols/UCONN_ILCC_LOW_CAPACITY_PPX_V12_PROTOCOL.md`
- 파일 SHA-256:
  `9f1471702add0f20ea5be1ad2fd53337474953fed0164f9e993b4e6fdfd6837f`
- 다운로드·궤적 열람 전에 split, 적격 기준, tail cutoff, v1.2 승인식을
  기록했다.

v1.2는 prior 후보가 matched MLP보다 pooled validation RMSE 2% 이상
개선하는 것에 더해, validation 셀 60% 이상 승리, 최악 셀 RMSE 비율
1.10 이하, cell-bootstrap 95% CI 하한 양수를 모두 요구한다. 하나라도
실패하면 trust=0 matched MLP로 후퇴한다.

## 적격 감사

- 원자료 셀: **64**
- 유한 RPT 관측 20개 이상: **34**
- 셀별 초기 5개 RPT 중앙값의 80% 실제 교차: **26**
- 두 조건을 모두 만족: **11**
- 최소 필요 셀: **30**
- 적격 셀 ID: `7, 11, 12, 27, 34, 40, 43, 45, 48, 49, 64`
- 셀별 RPT 관측 수: 최소 16, 중앙 20, 최대 35

따라서 protocol의 최소 30셀 문에서 중단했다. train/validation/test를
만들지 않았고, PP-X와 matched MLP 모두 fit 0, test label 기반 수정 0이다.

## 해석

공개 설명의 “80% EOL까지 aging”과 processed RPT 자료에서 셀별 초기
용량을 기준으로 계산한 80% 교차는 동일하지 않았다. 일부 셀은 공칭
1.2 Ah의 80%인 0.96 Ah 부근까지 내려갔어도, 실제 초기용량의 80%에는
도달하지 않았다. 또한 RPT가 20개 미만인 셀이 30개였다.

결과를 본 뒤 경계를 0.96 Ah로 바꾸거나 최소 관측 수를 낮추면 적격 셀을
늘릴 수 있지만, 이는 사후 규칙 완화다. 이 실행에서는 하지 않았다.

이 결과는 PP-X가 나쁘다는 증거도, 좋다는 증거도 아니다. 정확한 결론은
**선택한 저용량 고호트가 동결된 엄격 평가 계약을 충족하지 못해 모델을
평가할 수 없었다**는 것이다.

## 산출물

- `results/uconn_ilcc_ppx_v12/results.json`: 64셀 전체 적격 감사
- `experiments/uconn_ilcc_ppx_v12.py`: 재현 스크립트
- `src/pp_extrapolation/generalization_policy.py`: v1.2 셀 승률·최악 셀 문
- 정책 단위시험: **7 passed**
